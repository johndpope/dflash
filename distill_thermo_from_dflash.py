"""
distill_thermo_from_dflash.py
Warm-start ThermoDFlashDraftModel from a pre-trained z-lab DFlash checkpoint.

Strategy:
  1. Load pre-trained DFlashDraftModel (e.g. z-lab/Qwen3-4B-DFlash-b16)
  2. Port all compatible weights (Q/K/V/O/MLP/norms/fc/rotary) into
     ThermoDFlashDraftModel — acceptance rate immediately ≈ pre-trained baseline
  3. Freeze ported weights; train ONLY the new Gibbs coupling parameters:
       layers.*.self_attn.gibbs.*.J_proj.weight  (head_dim × head_dim per head)
       layers.*.self_attn.gibbs.*.log_beta_offset (scalar per head)
  4. Optionally unfreeze everything for full fine-tuning after warm-up

Since J_proj is initialised to identity, Gibbs step 0 ≈ standard softmax —
the model starts with the pre-trained acceptance rate and learns to exceed it.

Usage:
    python distill_thermo_from_dflash.py \\
        --draft  z-lab/Qwen3-4B-DFlash-b16 \\
        --target Qwen/Qwen3-4B \\
        --dataset gsm8k \\
        --steps 500 \\
        --save-dir ./checkpoints/thermo_qwen3_4b

    # Then benchmark against the original:
    python bench_thermo.py \\
        --target Qwen/Qwen3-4B \\
        --load-thermo ./checkpoints/thermo_qwen3_4b/final \\
        --train-steps 0 --eval-prompts 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

from dflash.model import DFlashDraftModel, extract_context_feature, build_target_layer_ids
from dflash.model_thermo import ThermoDFlashDraftModel, thermo_distillation_loss
from dflash.benchmark import load_and_process_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16


# ---------------------------------------------------------------------------
# Weight porting: DFlashDraftModel → ThermoDFlashDraftModel
# ---------------------------------------------------------------------------

def port_weights(src: DFlashDraftModel, dst: ThermoDFlashDraftModel) -> tuple[int, int]:
    """
    Copy all compatible weights from pre-trained DFlash into ThermoDFlash.

    Compatible: every key in src that also exists in dst (all Q/K/V/O/MLP/
    norm/fc/rotary). The only keys in dst that DON'T exist in src are:
        layers.*.self_attn.gibbs.*.J_proj.weight
        layers.*.self_attn.gibbs.*.log_beta_offset

    Returns (n_copied, n_skipped).
    """
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()

    n_copied  = 0
    n_skipped = 0
    mismatched = []

    for key, param in src_sd.items():
        if key not in dst_sd:
            n_skipped += 1
            continue
        if dst_sd[key].shape != param.shape:
            mismatched.append((key, param.shape, dst_sd[key].shape))
            n_skipped += 1
            continue
        dst_sd[key].copy_(param)
        n_copied += 1

    if mismatched:
        print(f"  [warn] {len(mismatched)} shape mismatches (skipped):")
        for k, s, d in mismatched[:5]:
            print(f"    {k}: src={s} dst={d}")

    dst.load_state_dict(dst_sd)
    return n_copied, n_skipped


def freeze_ported_weights(model: ThermoDFlashDraftModel) -> tuple[int, int]:
    """
    Freeze every parameter EXCEPT the new Gibbs coupling parameters.
    Returns (n_frozen, n_trainable).
    """
    gibbs_keys = {"J_proj", "log_beta_offset"}
    n_frozen, n_trainable = 0, 0

    for name, param in model.named_parameters():
        is_gibbs = any(k in name for k in gibbs_keys)
        param.requires_grad_(is_gibbs)
        if is_gibbs:
            n_trainable += param.numel()
        else:
            n_frozen += param.numel()

    return n_frozen, n_trainable


def unfreeze_all(model: ThermoDFlashDraftModel) -> None:
    for p in model.parameters():
        p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── Load pre-trained DFlash draft ─────────────────────────────────────────
    print(f"\nLoading pre-trained DFlash draft: {args.draft}")
    src_draft = DFlashDraftModel.from_pretrained(
        args.draft, trust_remote_code=True, torch_dtype=DTYPE,
    ).to(DEVICE).eval()

    src_cfg = src_draft.config
    n_src = sum(p.numel() for p in src_draft.parameters()) / 1e6
    print(f"  {src_cfg.num_hidden_layers}L  hidden={src_cfg.hidden_size}  "
          f"heads={src_cfg.num_attention_heads}  "
          f"block_size={src_cfg.block_size}  params={n_src:.1f}M")

    # ── Load frozen target ────────────────────────────────────────────────────
    print(f"\nLoading target model: {args.target}")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=DTYPE, attn_implementation="sdpa", device_map="auto"
    ).eval()
    tokenizer  = AutoTokenizer.from_pretrained(args.target)
    target_cfg = target.config
    print(f"  {target_cfg.num_hidden_layers}L  hidden={target_cfg.hidden_size}  "
          f"vocab={target_cfg.vocab_size}")

    # ── Build ThermoDFlashDraftModel with same architecture ───────────────────
    print(f"\nBuilding ThermoDFlashDraftModel ({args.n_gibbs_steps} Gibbs steps)...")
    dcfg = Qwen3Config(
        hidden_size          = src_cfg.hidden_size,
        num_hidden_layers    = src_cfg.num_hidden_layers,
        num_attention_heads  = src_cfg.num_attention_heads,
        num_key_value_heads  = src_cfg.num_key_value_heads,
        intermediate_size    = src_cfg.intermediate_size,
        rms_norm_eps         = src_cfg.rms_norm_eps,
        vocab_size           = src_cfg.vocab_size,
        attention_bias       = getattr(src_cfg, "attention_bias", False),
        attention_dropout    = getattr(src_cfg, "attention_dropout", 0.0),
        sliding_window       = None,
        layer_types          = ["full_attention"] * src_cfg.num_hidden_layers,
        rope_theta           = getattr(src_cfg, "rope_theta", 10000.0),
        head_dim             = getattr(src_cfg, "head_dim",
                                       src_cfg.hidden_size // src_cfg.num_attention_heads),
    )
    dcfg.num_target_layers = target_cfg.num_hidden_layers
    dcfg.dflash_config = {
        **src_cfg.dflash_config,
        "n_gibbs_steps": args.n_gibbs_steps,
        "beta_start":    args.beta_start,
        "beta_end":      args.beta_end,
    }
    dcfg.block_size = src_cfg.block_size

    dst_draft = ThermoDFlashDraftModel(dcfg).to(DEVICE).to(DTYPE)

    # ── Port weights ──────────────────────────────────────────────────────────
    print("\nPorting pre-trained weights...")
    n_copied, n_skipped = port_weights(src_draft, dst_draft)
    print(f"  Copied  : {n_copied} tensors")
    print(f"  Skipped : {n_skipped} tensors  (new Gibbs params → identity init)")
    del src_draft  # free VRAM

    # ── Phase 1: freeze ported weights, train only Gibbs params ──────────────
    n_frozen, n_trainable = freeze_ported_weights(dst_draft)
    print(f"\nPhase 1 — Gibbs-only fine-tune")
    print(f"  Frozen    : {n_frozen/1e6:.1f}M params (pre-trained DFlash weights)")
    print(f"  Trainable : {n_trainable/1e3:.1f}K params (J_proj + log_beta_offset)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, dst_draft.parameters()),
        lr=args.lr, weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )

    target_layer_ids = dcfg.dflash_config["target_layer_ids"]
    mask_token_id    = dcfg.dflash_config["mask_token_id"]

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.dataset}")
    import random
    dataset = load_and_process_dataset(args.dataset)
    random.shuffle(dataset)
    print(f"  {len(dataset)} samples")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────────
    dst_draft.train()
    losses, ce_losses = [], []
    data_idx = 0
    t_start  = time.perf_counter()

    print(f"\nStarting warm-start distillation — {args.steps} steps\n")
    pbar = tqdm(total=args.steps, desc="Distilling")

    # Phase 2 switch: unfreeze all weights after warm-up phase
    phase2_step = int(args.steps * args.phase2_frac)
    in_phase2   = False

    for step in range(args.steps):

        # Unfreeze all weights for full fine-tuning after warm-up
        if step == phase2_step and phase2_step > 0 and not in_phase2:
            unfreeze_all(dst_draft)
            optimizer = torch.optim.AdamW(
                dst_draft.parameters(), lr=args.lr * 0.1, weight_decay=1e-2
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.steps - step, eta_min=args.lr * 0.01
            )
            in_phase2 = True
            n_all = sum(p.numel() for p in dst_draft.parameters()) / 1e6
            tqdm.write(f"\n  → Phase 2: full fine-tune ({n_all:.1f}M params, lr={args.lr*0.1:.2e})")

        # Sample a prompt
        instance = dataset[data_idx % len(dataset)]
        data_idx += 1

        user_text = instance["turns"][0]
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except Exception:
            text = user_text

        ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        if ids.shape[1] < 4:
            continue
        if ids.shape[1] > args.max_seq_len:
            ids = ids[:, :args.max_seq_len]
        seq_len = ids.shape[1]

        try:
            with torch.no_grad():
                out = target(ids, output_hidden_states=True, use_cache=False)
                tgt_hidden = extract_context_feature(out.hidden_states, target_layer_ids)
                tgt_logits = out.logits

            noise_ids  = torch.full_like(ids, mask_token_id)
            noise_emb  = target.model.embed_tokens(noise_ids)
            # pos_ids spans ctx + noise for correct RoPE coverage
            pos_ids    = torch.arange(2 * seq_len, device=DEVICE).unsqueeze(0)

            draft_hidden = dst_draft(
                position_ids=pos_ids,
                noise_embedding=noise_emb,
                target_hidden=tgt_hidden,
            )
            ld = thermo_distillation_loss(
                draft_hidden, target.lm_head, tgt_logits,
                temperature=args.temperature,
                gibbs_entropy_weight=args.entropy_weight,
                model=dst_draft,
            )
            optimizer.zero_grad()
            ld["loss"].backward()
            nn.utils.clip_grad_norm_(dst_draft.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            losses.append(float(ld["loss"]))
            ce_losses.append(float(ld["ce_loss"]))

        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue

        pbar.update(1)

        if (step + 1) % args.log_every == 0:
            avg_ce  = np.mean(ce_losses[-args.log_every:])
            elapsed = time.perf_counter() - t_start
            phase   = "P2" if in_phase2 else "P1"
            pbar.set_postfix({
                "ce": f"{avg_ce:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "phase": phase,
            })

        if (step + 1) % args.save_every == 0:
            ckpt = save_dir / f"step_{step+1:06d}"
            dst_draft.save_pretrained(str(ckpt))
            tqdm.write(f"\n  Saved: {ckpt}")

    pbar.close()

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = save_dir / "final"
    dst_draft.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    total = time.perf_counter() - t_start
    print(f"\nDone in {total/60:.1f} min")
    print(f"Final CE: {np.mean(ce_losses[-50:]):.4f}")
    print(f"Checkpoint: {final_path}")
    print(f"\nBenchmark:")
    print(f"  python bench_thermo.py \\")
    print(f"    --target {args.target} \\")
    print(f"    --load-thermo {final_path} \\")
    print(f"    --train-steps 0 --eval-prompts 8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Warm-start ThermoDFlash from pre-trained z-lab DFlash checkpoint"
    )
    p.add_argument("--draft",          default="z-lab/Qwen3-4B-DFlash-b16",
                   help="Pre-trained DFlashDraftModel (HF hub or local path)")
    p.add_argument("--target",         default="Qwen/Qwen3-4B",
                   help="Frozen target LLM")
    p.add_argument("--dataset",        default="gsm8k",
                   choices=["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"])
    p.add_argument("--n-gibbs-steps",  type=int,   default=4)
    p.add_argument("--beta-start",     type=float, default=0.8)
    p.add_argument("--beta-end",       type=float, default=1.2)
    p.add_argument("--steps",          type=int,   default=500)
    p.add_argument("--lr",             type=float, default=1e-3,
                   help="LR for Gibbs phase (phase 2 uses lr*0.1)")
    p.add_argument("--phase2-frac",    type=float, default=0.5,
                   help="Fraction of steps before unfreezing all weights (0=skip)")
    p.add_argument("--temperature",    type=float, default=0.0)
    p.add_argument("--entropy-weight", type=float, default=0.01)
    p.add_argument("--max-seq-len",    type=int,   default=512)
    p.add_argument("--save-dir",       default="./checkpoints/thermo_qwen3_4b")
    p.add_argument("--log-every",      type=int,   default=25)
    p.add_argument("--save-every",     type=int,   default=250)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
