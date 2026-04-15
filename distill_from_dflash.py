#!/usr/bin/env python3
"""
distill_from_dflash.py
Distil a ThermoFlash draft model from any pretrained DFlash checkpoint.

Loads a pretrained DFlash draft model (e.g. z-lab/Qwen3.5-4B-DFlash), copies
the shared weights (embeddings, layer norms, MLPs, Q/K/V projections) into a
ThermoDFlashDraftModel, then fine-tunes only the thermodynamic parameters
(J_proj coupling matrix + log_beta_offset per head) via teacher-forced
distillation against the frozen target LLM.

Available pretrained DFlash checkpoints
----------------------------------------
  z-lab/Kimi-K2.5-DFlash
  z-lab/Qwen3.5-4B-DFlash
  z-lab/Qwen3.5-9B-DFlash
  z-lab/Qwen3.5-27B-DFlash
  z-lab/Qwen3.5-35B-A3B-DFlash
  z-lab/Qwen3-Coder-Next-DFlash
  z-lab/Qwen3-Coder-30B-A3B-DFlash
  z-lab/gpt-oss-20b-DFlash
  z-lab/gpt-oss-120b-DFlash
  z-lab/Qwen3-4B-DFlash-b16
  z-lab/Qwen3-8B-DFlash-b16
  z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat

Quick-start examples
--------------------
# Distil from Qwen3-4B DFlash (1 000 steps, default settings)
python distill_from_dflash.py \\
  --target Qwen/Qwen3-4B \\
  --pretrained-dflash z-lab/Qwen3-4B-DFlash-b16 \\
  --output-dir ./thermo_qwen3_4b

# More Gibbs steps + colder initial temperature
python distill_from_dflash.py \\
  --target Qwen/Qwen3-8B \\
  --pretrained-dflash z-lab/Qwen3-8B-DFlash-b16 \\
  --output-dir ./thermo_qwen3_8b \\
  --steps 2000 \\
  --n-gibbs-steps 8 \\
  --beta-start 1.0 --beta-end 2.0

# 8-bit target to save VRAM on single-GPU machines
python distill_from_dflash.py \\
  --target Qwen/Qwen3-4B \\
  --pretrained-dflash z-lab/Qwen3.5-4B-DFlash \\
  --output-dir ./thermo_qwen3.5_4b \\
  --load-8bit
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Distil ThermoFlash from a pretrained DFlash model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--target",           required=True,
                        help="Frozen target LLM (HuggingFace model ID or local path)")
    parser.add_argument("--pretrained-dflash", required=True,
                        help="Pretrained DFlash draft model (HuggingFace ID or local path)")
    parser.add_argument("--output-dir",       required=True,
                        help="Directory to save the distilled ThermoFlash model")

    # Thermodynamic hyperparameters
    parser.add_argument("--n-gibbs-steps",    type=int,   default=4,
                        help="Gibbs refinement steps per attention call")
    parser.add_argument("--beta-start",       type=float, default=0.8,
                        help="Initial inverse temperature")
    parser.add_argument("--beta-end",         type=float, default=1.2,
                        help="Final inverse temperature")
    parser.add_argument("--entropy-weight",   type=float, default=0.01,
                        help="Weight for Gibbs entropy regularisation")

    # DFlash
    parser.add_argument("--block-size",       type=int,   default=4,
                        help="Speculative decoding block size")

    # Training
    parser.add_argument("--steps",            type=int,   default=1000,
                        help="Number of distillation steps")
    parser.add_argument("--lr",               type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--dataset",          default="gsm8k",
                        choices=["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"],
                        help="Training dataset")
    parser.add_argument("--max-seq-len",      type=int,   default=512,
                        help="Maximum sequence length (cap to avoid OOM)")
    parser.add_argument("--temperature",      type=float, default=0.0,
                        help="Distillation temperature (0 = hard targets)")

    # Hardware
    parser.add_argument("--load-8bit",        action="store_true",
                        help="Load target in 8-bit via bitsandbytes (saves VRAM)")

    args = parser.parse_args()

    # Build the training command — delegates to train_thermo_dflash.py
    cmd = [
        sys.executable, "train_thermo_dflash.py",
        "--target",            args.target,
        "--pretrained-dflash", args.pretrained_dflash,
        "--save-dir",          args.output_dir,
        "--steps",             str(args.steps),
        "--n-gibbs-steps",     str(args.n_gibbs_steps),
        "--beta-start",        str(args.beta_start),
        "--beta-end",          str(args.beta_end),
        "--entropy-weight",    str(args.entropy_weight),
        "--block-size",        str(args.block_size),
        "--lr",                str(args.lr),
        "--dataset",           args.dataset,
        "--max-seq-len",       str(args.max_seq_len),
        "--temperature",       str(args.temperature),
        "--log-every",         "10",
        "--save-every",        str(args.steps),   # save once at end
    ]
    if args.load_8bit:
        cmd.append("--load-8bit")

    print("=" * 60)
    print("ThermoFlash distillation")
    print("=" * 60)
    print(f"  Target LLM      : {args.target}")
    print(f"  Pretrained DFlash: {args.pretrained_dflash}")
    print(f"  Output dir       : {args.output_dir}")
    print(f"  Gibbs steps      : {args.n_gibbs_steps}")
    print(f"  β schedule       : {args.beta_start} → {args.beta_end}")
    print(f"  Training steps   : {args.steps}  (lr={args.lr})")
    print(f"  Dataset          : {args.dataset}")
    print(f"  8-bit target     : {args.load_8bit}")
    print("=" * 60)
    print()

    result = subprocess.run(cmd)

    if result.returncode == 0:
        final_dir = f"{args.output_dir}/final"
        print()
        print("Distillation complete.")
        print(f"  Model saved to: {final_dir}")
        print()
        print("Benchmark:")
        print(f"  python bench_thermo.py \\")
        print(f"    --target {args.target} \\")
        print(f"    --load-thermo {final_dir} \\")
        print(f"    --block-size {args.block_size} \\")
        print(f"    --train-steps 0 \\")
        print(f"    --eval-prompts 8")
        return 0
    else:
        print(f"\nDistillation failed (exit code {result.returncode})")
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
