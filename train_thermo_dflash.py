"""
train_thermo_dflash.py
Distillation training for ThermoDFlashDraftModel.

Trains the thermodynamic draft model to match a frozen target Qwen3/LLaMA model
using token-level hard distillation (same objective as original DFlash training).

Usage:
    python train_thermo_dflash.py \\
        --target  Qwen/Qwen3-4B \\
        --dataset gsm8k \\
        --block-size 4 \\
        --n-gibbs-steps 4 \\
        --beta-start 0.8 --beta-end 1.2 \\
        --steps 2000 --lr 2e-4 \\
        --save-dir ./checkpoints/thermo_draft

The trained ThermoDFlashDraftModel can then be benchmarked with:
    python -m dflash.benchmark \\
        --backend transformers \\
        --model Qwen/Qwen3-4B \\
        --draft-model ./checkpoints/thermo_draft \\
        --dataset gsm8k

Note: The target model is always frozen. Only ThermoDFlashDraftModel trains.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from dflash.model import (
    build_target_layer_ids,
    extract_context_feature,
    sample as dflash_sample,
)
from dflash.model_thermo import ThermoDFlashDraftModel, thermo_distillation_loss
from dflash.benchmark import load_and_process_dataset


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

def make_thermo_config(target_config, args):
    """
    Build a Qwen3Config for ThermoDFlashDraftModel from the target config.

    ThermoDFlash uses fewer layers than the target (same as original DFlash
    draft models: typically 1 or 2 decoder layers).
    """
    from transformers import Qwen3Config
    import copy

    cfg = copy.deepcopy(target_config)

    # Draft model is shallow — 1-2 layers is typical in DFlash
    cfg.num_hidden_layers = args.draft_layers
    cfg.num_target_layers = target_config.num_hidden_layers

    # DFlash-specific config
    target_layer_ids = build_target_layer_ids(
        target_config.num_hidden_layers, args.draft_layers
    )
    cfg.dflash_config = {
        "target_layer_ids":  target_layer_ids,
        "block_size":        args.block_size,
        "mask_token_id":     target_config.vocab_size - 1,  # last token as mask
        # Thermodynamic hyperparameters
        "n_gibbs_steps":     args.n_gibbs_steps,
        "beta_start":        args.beta_start,
        "beta_end":          args.beta_end,
    }
    cfg.block_size = args.block_size
    return cfg


# ---------------------------------------------------------------------------
# Data collation: extract (input_ids, target_hidden, target_logits) batches
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_target_features(
    target: nn.Module,
    input_ids: torch.LongTensor,
    target_layer_ids: list[int],
    block_size: int,
    mask_token_id: int,
):
    """
    Run one forward pass through the frozen target model on a real sequence.
    Returns:
        target_hidden:   (1, seq, D*n_layers)  — concatenated mid-layer states
        target_logits:   (1, seq, vocab)        — full logit distribution
        noise_ids:       (1, seq, block_size)   — masked input blocks for draft
    """
    out = target(
        input_ids,
        output_hidden_states=True,
        use_cache=False,
    )
    target_hidden = extract_context_feature(out.hidden_states, target_layer_ids)
    target_logits = out.logits
    return target_hidden, target_logits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load frozen target ────────────────────────────────────────────────────
    print(f"Loading target model: {args.target}")
    try:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True) if args.load_8bit else None
    except ImportError:
        bnb_cfg = None

    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=bnb_cfg,
        device_map="auto" if device.type == "cuda" else None,
    ).eval()

    if device.type != "cuda":
        target = target.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target_config = target.config
    print(f"  Target: {target_config.num_hidden_layers} layers, "
          f"hidden={target_config.hidden_size}, vocab={target_config.vocab_size}")

    # ── Build draft model ─────────────────────────────────────────────────────
    print(f"Building ThermoDFlashDraftModel ({args.draft_layers} layers, "
          f"{args.n_gibbs_steps} Gibbs steps, β={args.beta_start}→{args.beta_end})")
    draft_config = make_thermo_config(target_config, args)
    draft = ThermoDFlashDraftModel(draft_config).to(device)

    n_params = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    print(f"  Draft trainable params: {n_params/1e6:.2f}M")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = AdamW(draft.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    dataset = load_and_process_dataset(args.dataset)
    random.shuffle(dataset)
    print(f"  {len(dataset)} samples")

    target_layer_ids = draft_config.dflash_config["target_layer_ids"]
    mask_token_id    = draft_config.dflash_config["mask_token_id"]
    block_size       = args.block_size

    # ── Training ──────────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    draft.train()
    step        = 0
    data_idx    = 0
    losses      = []
    ce_losses   = []
    t_start     = time.perf_counter()

    print(f"\nStarting distillation training — {args.steps} steps\n")
    pbar = tqdm(total=args.steps, desc="Training")

    while step < args.steps:
        # Cycle through dataset
        instance = dataset[data_idx % len(dataset)]
        data_idx += 1

        # Tokenise one prompt turn
        user_text = instance["turns"][0]
        messages  = [{"role": "user", "content": user_text}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            text = user_text

        token_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        if token_ids.shape[1] < block_size + 2:
            continue
        # Cap length to avoid OOM
        if token_ids.shape[1] > args.max_seq_len:
            token_ids = token_ids[:, :args.max_seq_len]

        seq_len = token_ids.shape[1]

        try:
            # ── Frozen target forward ─────────────────────────────────────────
            with torch.no_grad():
                target_hidden, target_logits = collect_target_features(
                    target, token_ids, target_layer_ids, block_size, mask_token_id
                )

            # ── Build draft input: noise = mask tokens at block positions ─────
            # We train on all positions simultaneously (teacher-forcing style)
            noise_ids = torch.full_like(token_ids, mask_token_id)
            noise_ids[:, :seq_len - block_size] = token_ids[:, block_size:]  # shifted

            noise_emb = target.model.embed_tokens(noise_ids)                 # (1, seq, D)

            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            # ── Draft forward ─────────────────────────────────────────────────
            draft_hidden = draft(
                position_ids=position_ids,
                noise_embedding=noise_emb,
                target_hidden=target_hidden,
                use_cache=False,
            )                                                                  # (1, seq, D)

            # ── Distillation loss ─────────────────────────────────────────────
            losses_dict = thermo_distillation_loss(
                draft_hidden=draft_hidden,
                target_lm_head=target.lm_head,
                target_logits=target_logits,
                temperature=args.temperature,
                gibbs_entropy_weight=args.entropy_weight,
                model=draft,
            )
            loss = losses_dict["loss"]

            # ── Backward ─────────────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            losses.append(float(losses_dict["loss"]))
            ce_losses.append(float(losses_dict["ce_loss"]))

        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue

        step += 1
        pbar.update(1)

        if step % args.log_every == 0:
            avg_loss    = np.mean(losses[-args.log_every:])
            avg_ce      = np.mean(ce_losses[-args.log_every:])
            elapsed     = time.perf_counter() - t_start
            steps_per_s = step / elapsed
            pbar.set_postfix({
                "loss":    f"{avg_loss:.4f}",
                "ce":      f"{avg_ce:.4f}",
                "lr":      f"{scheduler.get_last_lr()[0]:.2e}",
                "step/s":  f"{steps_per_s:.1f}",
            })

        if step % args.save_every == 0:
            ckpt_path = save_dir / f"step_{step:06d}"
            draft.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            print(f"\n  Saved checkpoint: {ckpt_path}")

    pbar.close()

    # ── Final save ────────────────────────────────────────────────────────────
    draft.save_pretrained(str(save_dir / "final"))
    tokenizer.save_pretrained(str(save_dir / "final"))

    total_time = time.perf_counter() - t_start
    print(f"\nTraining complete in {total_time/60:.1f} min")
    print(f"Final avg loss:    {np.mean(losses[-100:]):.4f}")
    print(f"Final avg CE loss: {np.mean(ce_losses[-100:]):.4f}")
    print(f"Checkpoint: {save_dir / 'final'}")

    print("\nTo benchmark:")
    print(f"  python -m dflash.benchmark \\")
    print(f"    --backend transformers \\")
    print(f"    --model {args.target} \\")
    print(f"    --draft-model {save_dir / 'final'} \\")
    print(f"    --dataset {args.dataset} \\")
    print(f"    --block-size {block_size}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train ThermoDFlashDraftModel via distillation from a frozen target LLM"
    )
    # Model
    parser.add_argument("--target",        type=str,   default="Qwen/Qwen3-4B",
                        help="Target HuggingFace model ID")
    parser.add_argument("--draft-layers",  type=int,   default=1,
                        help="Number of decoder layers in draft model (default 1)")
    parser.add_argument("--load-8bit",     action="store_true",
                        help="Load target in 8-bit (requires bitsandbytes)")

    # Thermodynamic hyperparameters
    parser.add_argument("--n-gibbs-steps", type=int,   default=4,
                        help="Gibbs refinement steps per attention call (default 4)")
    parser.add_argument("--beta-start",    type=float, default=0.8,
                        help="Initial inverse temperature (default 0.8)")
    parser.add_argument("--beta-end",      type=float, default=1.2,
                        help="Final inverse temperature (default 1.2)")
    parser.add_argument("--entropy-weight", type=float, default=0.01,
                        help="Weight for Gibbs entropy regularisation (default 0.01)")

    # DFlash
    parser.add_argument("--block-size",    type=int,   default=4,
                        help="Speculative decoding block size (default 4)")

    # Training
    parser.add_argument("--dataset",       type=str,   default="gsm8k",
                        choices=["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"])
    parser.add_argument("--steps",         type=int,   default=2000)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--temperature",   type=float, default=0.0,
                        help="Distillation temperature (0=hard targets)")
    parser.add_argument("--max-seq-len",   type=int,   default=512)
    parser.add_argument("--save-dir",      type=str,   default="./checkpoints/thermo_draft")
    parser.add_argument("--log-every",     type=int,   default=50)
    parser.add_argument("--save-every",    type=int,   default=500)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
