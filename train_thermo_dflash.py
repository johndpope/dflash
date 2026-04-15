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
    # layer_types must match num_hidden_layers exactly (Qwen3Config validates this)
    if hasattr(cfg, "layer_types") and cfg.layer_types:
        cfg.layer_types = list(cfg.layer_types)[:args.draft_layers]

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
    if args.resume_from:
        # Continue training an existing ThermoDFlash checkpoint
        print(f"Resuming ThermoDFlash from: {args.resume_from}")
        draft = ThermoDFlashDraftModel.from_pretrained(
            args.resume_from, torch_dtype=torch.bfloat16,
        )
        draft_config = draft.config
        # Override block_size if caller changed it (e.g. 4→8)
        draft_config.block_size = args.block_size
        if hasattr(draft_config, "dflash_config"):
            draft_config.dflash_config["block_size"] = args.block_size
        pretrained_layers = len(draft.layers)
        if pretrained_layers != args.draft_layers:
            print(f"  Note: checkpoint has {pretrained_layers} layers → using that")
            args.draft_layers = pretrained_layers
    elif args.pretrained_dflash:
        print(f"Loading pretrained DFlash model: {args.pretrained_dflash}")
        from dflash.model import DFlashDraftModel

        # Load the pretrained DFlash model
        pretrained_draft = DFlashDraftModel.from_pretrained(
            args.pretrained_dflash, torch_dtype=torch.bfloat16
        )

        # Check if layer count matches
        pretrained_layers = len(pretrained_draft.layers)
        if pretrained_layers != args.draft_layers:
            print(f"  Warning: pretrained model has {pretrained_layers} layers, "
                  f"training with {args.draft_layers} layers")
            args.draft_layers = pretrained_layers

        # Convert to ThermoDFlash by creating new config and copying weights
        draft_config = make_thermo_config(target_config, args)
        draft = ThermoDFlashDraftModel(draft_config)

        # Copy all compatible weights (Q/K/V/O projections, norms, MLPs, fc, rotary).
        # strict=False: Gibbs-only keys (J_proj, log_beta_offset) are absent in
        # the DFlash checkpoint and will stay at their near-zero init.
        missing_keys, unexpected_keys = draft.load_state_dict(
            pretrained_draft.state_dict(), strict=False
        )
        print(f"  Converted DFlash → ThermoDFlash")
        print(f"    Missing keys (new, near-zero init): {len(missing_keys)}")
        print(f"    Unexpected keys (ignored):          {len(unexpected_keys)}")
    else:
        print(f"Building ThermoDFlashDraftModel from scratch ({args.draft_layers} layers, "
              f"{args.n_gibbs_steps} Gibbs steps, β={args.beta_start}→{args.beta_end})")
        draft_config = make_thermo_config(target_config, args)
        draft = ThermoDFlashDraftModel(draft_config)

    # Match dtype to the target model so target_hidden flows through without casting
    target_dtype = next(target.parameters()).dtype
    draft = draft.to(device=device, dtype=target_dtype)

    n_params = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    print(f"  Draft trainable params: {n_params/1e6:.2f}M")

    # ── Optimizer — differential LR ──────────────────────────────────────────
    if args.pretrained_dflash or args.resume_from:
        # Gibbs params: train at full LR (they're new, need to learn fast)
        # Backbone params: train at LR/200 (barely move, just adapt to sigmoid attn)
        gibbs_param_names = {n for n, _ in draft.named_parameters() if "gibbs" in n}
        gibbs_params    = [p for n, p in draft.named_parameters() if n in gibbs_param_names]
        backbone_params = [p for n, p in draft.named_parameters() if n not in gibbs_param_names]
        optimizer = AdamW([
            {"params": gibbs_params,    "lr": args.lr,        "weight_decay": 1e-2},
            {"params": backbone_params, "lr": args.lr / 200,  "weight_decay": 1e-4},
        ])
        print(f"  Optimizer: Gibbs LR={args.lr:.1e}  Backbone LR={args.lr/200:.1e}")
    else:
        optimizer = AdamW(draft.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)

    # ── Dataset (supports comma-separated mix, e.g. "gsm8k,math500,humaneval") ─
    dataset_names = [d.strip() for d in args.dataset.split(",")]
    dataset = []
    for ds_name in dataset_names:
        print(f"Loading dataset: {ds_name}")
        ds = load_and_process_dataset(ds_name)
        dataset.extend(ds)
        print(f"  +{len(ds)} samples ({ds_name})")
    random.shuffle(dataset)
    print(f"  Total: {len(dataset)} samples")

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
    kl_losses   = []
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

            # pos_ids must span ctx_len + q_len so RoPE covers the full kv sequence
            # (target_hidden and noise are concatenated as keys → 2 × seq_len positions)
            position_ids = torch.arange(2 * seq_len, device=device).unsqueeze(0)

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
            nn.utils.clip_grad_norm_(
                [p for p in draft.parameters() if p.grad is not None], 1.0
            )
            optimizer.step()
            scheduler.step()

            losses.append(losses_dict["loss"].detach().item())
            ce_losses.append(losses_dict["ce_loss"].detach().item())
            kl_losses.append(losses_dict.get("kl_loss", torch.tensor(0.0)).detach().item())

        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue

        step += 1
        pbar.update(1)

        if step % args.log_every == 0:
            avg_loss    = np.mean(losses[-args.log_every:])
            avg_ce      = np.mean(ce_losses[-args.log_every:])
            avg_kl      = np.mean(kl_losses[-args.log_every:])
            elapsed     = time.perf_counter() - t_start
            steps_per_s = step / elapsed
            pbar.set_postfix({
                "loss":    f"{avg_loss:.4f}",
                "ce":      f"{avg_ce:.4f}",
                "kl":      f"{avg_kl:.4f}",
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
    parser.add_argument("--pretrained-dflash", type=str, default=None,
                        help="Path to pretrained DFlash model to distill from (optional)")
    parser.add_argument("--resume-from",   type=str,   default=None,
                        help="Resume training from a ThermoDFlash checkpoint")
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
                        help="Dataset(s) to train on, comma-separated "
                             "(choices: gsm8k, math500, humaneval, mbpp, mt-bench)")
    parser.add_argument("--steps",         type=int,   default=2000)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--temperature",   type=float, default=2.0,
                        help="Distillation temperature (>1=soft KL targets, 0=hard CE)")
    parser.add_argument("--max-seq-len",   type=int,   default=512)
    parser.add_argument("--save-dir",      type=str,   default="./checkpoints/thermo_draft")
    parser.add_argument("--log-every",     type=int,   default=50)
    parser.add_argument("--save-every",    type=int,   default=500)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
