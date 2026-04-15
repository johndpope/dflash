"""
bench_thermo.py
Benchmark ThermoDFlashDraftModel vs standard DFlashDraftModel.

Runs three benchmarks:
  1. Kernel speed: ThermoGibbsSampler vs scaled-dot-product attention
     across varying seq lengths, head dims, and n_gibbs_steps.
  2. Draft model forward pass throughput (ThermoDFlash vs DFlash).
  3. Mini speculative decoding acceptance rate with Qwen3-1.7B
     after a short distillation training run (~100 steps).
"""

from __future__ import annotations

import argparse
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16

print(f"Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()


# ---------------------------------------------------------------------------
# Benchmark 1 — Kernel speed: ThermoGibbsSampler vs SDPA
# ---------------------------------------------------------------------------

def bench_kernel():
    from dflash.model_thermo import ThermoGibbsSampler

    print("=" * 60)
    print("Benchmark 1: ThermoGibbsSampler vs SDPA (kernel speed)")
    print("=" * 60)

    configs = [
        # (B, q_len, kv_len, head_dim, n_gibbs_steps)
        (4,   4,  32,  64,  1),
        (4,   4,  32,  64,  4),
        (4,   4,  32,  64,  8),
        (4,   4,  64, 128,  4),
        (4,   8,  64, 128,  4),
        (1,   4, 256, 128,  4),
    ]
    warmup, reps = 5, 50

    print(f"\n{'Config':<40} {'SDPA (ms)':>10} {'Thermo (ms)':>12} {'Overhead':>10}")
    print("-" * 75)

    for B, q_len, kv_len, D, n_steps in configs:
        q = torch.randn(B, q_len, D,  device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, kv_len, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, kv_len, D, device=DEVICE, dtype=DTYPE)

        sampler = ThermoGibbsSampler(D, n_steps=n_steps).to(DEVICE).to(DTYPE)

        # ── Standard SDPA ──────────────────────────────────────────────────
        # reshape to (B,1,seq,D) for torch sdpa
        q4 = q.unsqueeze(1)
        k4 = k.unsqueeze(1)
        v4 = v.unsqueeze(1)
        for _ in range(warmup):
            _ = F.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            F.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        sdpa_ms = (time.perf_counter() - t0) * 1000 / reps

        # ── ThermoGibbsSampler ─────────────────────────────────────────────
        for _ in range(warmup):
            _ = sampler(q, k, v)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            sampler(q, k, v)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        thermo_ms = (time.perf_counter() - t0) * 1000 / reps

        tag = f"B={B} q={q_len} kv={kv_len} D={D} steps={n_steps}"
        overhead = thermo_ms / sdpa_ms
        print(f"  {tag:<38} {sdpa_ms:>10.3f} {thermo_ms:>12.3f} {overhead:>9.2f}x")

    print()


# ---------------------------------------------------------------------------
# Benchmark 2 — Draft model forward pass throughput
# ---------------------------------------------------------------------------

def bench_draft_forward(target_hidden_size=2048, target_layers=28):
    from dflash.model_thermo import ThermoDFlashDraftModel
    from dflash.model import DFlashDraftModel, build_target_layer_ids
    from transformers import Qwen3Config

    print("=" * 60)
    print("Benchmark 2: ThermoDFlash vs DFlash draft forward pass")
    print("=" * 60)

    def make_config(num_draft_layers, thermo=False, n_gibbs=4):
        cfg = Qwen3Config(
            hidden_size=target_hidden_size,
            num_hidden_layers=num_draft_layers,
            num_attention_heads=16,
            num_key_value_heads=8,
            intermediate_size=target_hidden_size * 4,
            rms_norm_eps=1e-6,
            vocab_size=32000,
            attention_bias=False,
            attention_dropout=0.0,
            sliding_window=None,
            layer_types=["full_attention"] * num_draft_layers,
        )
        target_layer_ids = build_target_layer_ids(target_layers, num_draft_layers)
        cfg.num_target_layers = target_layers
        cfg.dflash_config = {
            "target_layer_ids": target_layer_ids,
            "block_size": 4,
            "mask_token_id": 0,
            "n_gibbs_steps": n_gibbs,
            "beta_start": 0.8,
            "beta_end": 1.2,
        }
        cfg.block_size = 4
        return cfg

    warmup, reps = 3, 20

    configs = [
        (1, 4, "DFlash baseline"),
        (1, 4, "Thermo n_gibbs=1"),
        (1, 4, "Thermo n_gibbs=4"),
        (1, 4, "Thermo n_gibbs=8"),
        (2, 4, "Thermo 2L n_gibbs=4"),
    ]

    B, block = 1, 4
    seq      = 64
    vocab    = 32000
    H        = target_hidden_size

    print(f"\n{'Model':<26} {'fwd (ms)':>10} {'tok/s':>10}")
    print("-" * 50)

    for n_layers, n_gibbs, label in configs:
        is_thermo = "Thermo" in label
        cfg = make_config(n_layers, thermo=is_thermo, n_gibbs=n_gibbs)
        n_target = len(cfg.dflash_config["target_layer_ids"])

        if is_thermo:
            model = ThermoDFlashDraftModel(cfg).to(DEVICE).to(DTYPE).eval()
        else:
            model = DFlashDraftModel(cfg).to(DEVICE).to(DTYPE).eval()

        noise_emb   = torch.randn(B, seq, H,           device=DEVICE, dtype=DTYPE)
        tgt_hidden  = torch.randn(B, seq, H * n_target, device=DEVICE, dtype=DTYPE)
        # position_ids must span ctx_len + q_len (both target_hidden and noise positions)
        # so RoPE cos/sin covers the full kv sequence length
        pos_ids     = torch.arange(seq * 2, device=DEVICE).unsqueeze(0)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6

        with torch.no_grad():
            for _ in range(warmup):
                model(position_ids=pos_ids, noise_embedding=noise_emb, target_hidden=tgt_hidden)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                model(position_ids=pos_ids, noise_embedding=noise_emb, target_hidden=tgt_hidden)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000 / reps

        toks_per_s = (B * seq * 1000) / ms
        print(f"  {label:<24} {ms:>10.2f} {toks_per_s:>10,.0f}  ({n_params:.1f}M params)")

        del model
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print()


# ---------------------------------------------------------------------------
# Benchmark 3 — Live acceptance rate with Qwen3-1.7B
# ---------------------------------------------------------------------------

def bench_acceptance(args):
    print("=" * 60)
    print("Benchmark 3: Speculative decoding acceptance rate")
    print(f"  Target : {args.target}")
    print(f"  Steps  : {args.train_steps} distillation steps then eval")
    print("=" * 60)

    import sys, os
    sys.path.insert(0, str(Path(__file__).parent))

    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, DynamicCache
    from dflash.model_thermo import ThermoDFlashDraftModel, thermo_distillation_loss
    from dflash.model import (
        build_target_layer_ids, extract_context_feature,
        sample as dflash_sample, dflash_generate,
    )

    # ── Load target ───────────────────────────────────────────────────────────
    print(f"\nLoading target model...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=DTYPE, attn_implementation="sdpa", device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    tcfg = target.config
    print(f"  {tcfg.num_hidden_layers}L hidden={tcfg.hidden_size} heads={tcfg.num_attention_heads}")

    # ── Build or load draft ───────────────────────────────────────────────────
    if args.load_thermo:
        print(f"\nLoading ThermoDFlash from checkpoint: {args.load_thermo}")
        draft = ThermoDFlashDraftModel.from_pretrained(
            args.load_thermo, torch_dtype=DTYPE,
        ).to(DEVICE).eval()
        dcfg = draft.config
    else:
        target_layer_ids = build_target_layer_ids(tcfg.num_hidden_layers, args.draft_layers)
        dcfg = Qwen3Config(
            hidden_size=tcfg.hidden_size,
            num_hidden_layers=args.draft_layers,
            num_attention_heads=tcfg.num_attention_heads,
            num_key_value_heads=tcfg.num_key_value_heads,
            intermediate_size=tcfg.intermediate_size,
            rms_norm_eps=tcfg.rms_norm_eps,
            vocab_size=tcfg.vocab_size,
            attention_bias=getattr(tcfg, "attention_bias", False),
            attention_dropout=getattr(tcfg, "attention_dropout", 0.0),
            sliding_window=None,
            layer_types=["full_attention"] * args.draft_layers,
            rope_theta=getattr(tcfg, "rope_theta", 10000.0),
        )
        dcfg.num_target_layers = tcfg.num_hidden_layers
        dcfg.dflash_config = {
            "target_layer_ids": target_layer_ids,
            "block_size": args.block_size,
            "mask_token_id": tcfg.vocab_size - 1,
            "n_gibbs_steps": args.n_gibbs_steps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
        }
    dcfg.block_size = args.block_size

    draft = ThermoDFlashDraftModel(dcfg).to(DEVICE).to(DTYPE)
    n_params = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    print(f"  Draft: {n_params/1e6:.2f}M params, {args.draft_layers}L, "
          f"{args.n_gibbs_steps} Gibbs steps")

    # ── Short distillation training ───────────────────────────────────────────
    if args.train_steps > 0:
        print(f"\nDistillation training ({args.train_steps} steps)...")
        optimizer = torch.optim.AdamW(draft.parameters(), lr=2e-4, weight_decay=1e-2)
        prompts = [
            "Solve: 2x + 5 = 13. Show your work.",
            "What is 17 * 23?",
            "A train travels at 60 mph for 2 hours. How far does it go?",
            "Write a Python function to reverse a string.",
            "Explain gradient descent in one paragraph.",
            "What is the capital of France?",
            "Factor x^2 + 5x + 6.",
            "If f(x) = x^2 + 1, what is f(3)?",
        ] * (args.train_steps // 8 + 1)

        losses = []
        draft.train()
        for step in range(args.train_steps):
            prompt = prompts[step]
            ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            if ids.shape[1] < 4:
                continue
            ids = ids[:, :128]  # cap length
            seq_len = ids.shape[1]

            with torch.no_grad():
                out = target(ids, output_hidden_states=True, use_cache=False)
                tgt_hidden = extract_context_feature(out.hidden_states, target_layer_ids)
                tgt_logits = out.logits

            noise_ids = torch.full_like(ids, dcfg.dflash_config["mask_token_id"])
            noise_emb = target.model.embed_tokens(noise_ids)
            # pos_ids must span ctx_len + q_len so RoPE covers the full kv sequence
            # (target_hidden and noise are both used as keys → 2 * seq_len positions)
            pos_ids   = torch.arange(2 * seq_len, device=DEVICE).unsqueeze(0)

            draft_hidden = draft(
                position_ids=pos_ids,
                noise_embedding=noise_emb,
                target_hidden=tgt_hidden,
            )
            ld = thermo_distillation_loss(
                draft_hidden, target.lm_head, tgt_logits,
                temperature=0.0, gibbs_entropy_weight=0.01, model=draft,
            )
            optimizer.zero_grad()
            ld["loss"].backward()
            nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
            optimizer.step()
            losses.append(float(ld["ce_loss"]))

            if (step + 1) % 20 == 0:
                print(f"  step {step+1:3d}/{args.train_steps}  "
                      f"ce={np.mean(losses[-20:]):.4f}")

        print(f"  Training done. Final CE: {np.mean(losses[-10:]):.4f}")
    else:
        print("\nSkipping training (--train-steps 0). Using random draft weights.")

    # ── Acceptance rate evaluation ────────────────────────────────────────────
    print(f"\nEvaluating acceptance rate ({args.eval_prompts} prompts)...")
    draft.eval()

    eval_prompts = [
        "What is 15 + 27?",
        "Explain the water cycle.",
        "Write hello world in Python.",
        "What is the speed of light?",
        "Solve x^2 - 4 = 0.",
        "Name three planets in our solar system.",
        "What is 8 * 7?",
        "Define photosynthesis.",
    ][:args.eval_prompts]

    block_size       = args.block_size
    mask_token_id    = dcfg.dflash_config["mask_token_id"]
    all_accept_lens  = []
    baseline_times   = []
    dflash_times     = []

    for prompt in eval_prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

        # Baseline: block_size=1 (no speculation)
        t0 = time.perf_counter()
        with torch.inference_mode():
            r1 = dflash_generate(
                draft, target=target,
                input_ids=ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                block_size=1,
                mask_token_id=mask_token_id,
                return_stats=True,
            )
        t1 = time.perf_counter()
        baseline_times.append((t1 - t0) / r1.num_output_tokens)

        # ThermoDFlash: block_size=N
        t0 = time.perf_counter()
        with torch.inference_mode():
            rN = dflash_generate(
                draft, target=target,
                input_ids=ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                block_size=block_size,
                mask_token_id=mask_token_id,
                return_stats=True,
            )
        t1 = time.perf_counter()
        dflash_times.append((t1 - t0) / max(rN.num_output_tokens, 1))
        all_accept_lens.extend(rN.acceptance_lengths)

        mean_acc = np.mean(rN.acceptance_lengths)
        print(f"  [{prompt[:30]:<30}]  accept={mean_acc:.2f}  out={rN.num_output_tokens}tok")

    mean_accept     = np.mean(all_accept_lens)
    baseline_tpot   = np.mean(baseline_times) * 1000   # ms/tok
    dflash_tpot     = np.mean(dflash_times) * 1000
    speedup         = baseline_tpot / max(dflash_tpot, 1e-9)

    histogram = [
        all_accept_lens.count(b) / len(all_accept_lens)
        for b in range(block_size + 1)
    ]

    print(f"\n{'=' * 60}")
    print(f"  Baseline throughput : {1000/baseline_tpot:>8.1f} tok/s")
    print(f"  ThermoDFlash tp     : {1000/dflash_tpot:>8.1f} tok/s")
    print(f"  Decoding speedup    : {speedup:>8.2f}x")
    print(f"  Avg acceptance len  : {mean_accept:>8.2f}  (block_size={block_size})")
    print(f"  Acceptance histogram: {[f'{x*100:.1f}%' for x in histogram]}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",        default="Qwen/Qwen3-1.7B")
    parser.add_argument("--draft-layers",  type=int,   default=1)
    parser.add_argument("--block-size",    type=int,   default=4)
    parser.add_argument("--n-gibbs-steps", type=int,   default=4)
    parser.add_argument("--beta-start",    type=float, default=0.8)
    parser.add_argument("--beta-end",      type=float, default=1.2)
    parser.add_argument("--train-steps",   type=int,   default=100)
    parser.add_argument("--eval-prompts",  type=int,   default=6)
    parser.add_argument("--max-new-tokens",type=int,   default=64)
    parser.add_argument("--temperature",   type=float, default=0.0)
    parser.add_argument("--skip-kernel",   action="store_true")
    parser.add_argument("--skip-draft-fwd",action="store_true")
    parser.add_argument("--skip-acceptance",action="store_true")
    args = parser.parse_args()

    if not args.skip_kernel:
        bench_kernel()

    if not args.skip_draft_fwd:
        try:
            bench_draft_forward()
        except Exception as e:
            print(f"Draft forward bench skipped: {e}\n")

    if not args.skip_acceptance:
        bench_acceptance(args)


if __name__ == "__main__":
    main()
