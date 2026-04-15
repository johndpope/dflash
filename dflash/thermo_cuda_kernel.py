"""
dflash/thermo_cuda_kernel.py

Fused CUDA kernel for thermodynamic Gibbs sampling attention.

Replaces the pure-Python loop and the broken Triton stub with a single
CUDA kernel compiled at first use via torch.utils.cpp_extension.load_inline.

Kernel design
-------------
Grid  : (B, q_len)  — one block per (batch, query-position) pair
Block : (BLK,)      — BLK = next power-of-2 >= max(kv_len, 32), capped at 1024

Each block:
  1. Loads h_row (external field) into shared memory            O(kv_len)
  2. Initialises m = softmax(h_row) via tree reduction          O(kv_len)
  3. For each Gibbs step:
       a. coupling[j] = Σ_i m[i]·J[b,i,j]                    O(kv_len²)
       b. m[j]        = sigmoid(β·(coupling[j] + h[j]))
       c. renormalise m via tree reduction                       O(kv_len)
  4. out[b,qi,d] = Σ_i m[i]·v[b,i,d]                          O(kv_len·D)

m[] lives entirely in shared memory across all steps — no round-trips to HBM.

Shared memory layout (all float32):
  [ sh_m | sh_h | sh_c | sh_red ]
    Nkv    Nkv    Nkv    BLK
  ≈ 3·kv_len + BLK  floats.  At kv_len=512, BLK=512: 8 KB — well within limits.
"""

from __future__ import annotations

import logging
import os
import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CUDA kernel source
# ---------------------------------------------------------------------------

_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

/*
 * thermo_gibbs_fwd
 *
 * All tensors are float32 and contiguous (caller ensures this).
 *
 * h      : [B, Nq, Nkv]   external field (already has mask added if any)
 * J      : [B, Nkv, Nkv]  coupling matrix  (J[b,i,j] = k[b,i] · J_proj(k)[b,j])
 * v      : [B, Nkv, D]    value matrix
 * out    : [B, Nq, D]     output (written by kernel)
 */
__global__ void thermo_gibbs_fwd(
    const float* __restrict__ h,
    const float* __restrict__ J,
    const float* __restrict__ v,
    float*       __restrict__ out,
    float beta_start,
    float beta_end,
    float beta_offset,
    int n_steps,
    int Nq,
    int Nkv,
    int D
) {
    const int b   = blockIdx.x;
    const int qi  = blockIdx.y;
    const int tid = threadIdx.x;
    const int BLK = blockDim.x;

    /* Shared memory layout:
       [0 .. Nkv-1]        sh_m   : current attention weights
       [Nkv .. 2*Nkv-1]    sh_h   : h_row (constant across steps)
       [2*Nkv .. 3*Nkv-1]  sh_c   : coupling accumulator
       [3*Nkv .. 3*Nkv+BLK-1] sh_red : reduction buffer
    */
    extern __shared__ float shmem[];
    float* sh_m   = shmem;
    float* sh_h   = shmem +     Nkv;
    float* sh_c   = shmem + 2 * Nkv;
    float* sh_red = shmem + 3 * Nkv;

    /* ---- 1. Load h_row into shared memory ---- */
    const float* h_row = h + (b * Nq + qi) * Nkv;
    for (int i = tid; i < Nkv; i += BLK)
        sh_h[i] = h_row[i];
    __syncthreads();

    /* ---- 2. Init m = softmax(sh_h) ---- */

    /* 2a. max reduction — threads beyond Nkv seed with -inf */
    float thr = (tid < Nkv) ? sh_h[tid] : -1e30f;
    for (int i = tid + BLK; i < Nkv; i += BLK)
        thr = fmaxf(thr, sh_h[i]);
    sh_red[tid] = thr;
    __syncthreads();
    for (int s = BLK >> 1; s > 0; s >>= 1) {
        if (tid < s) sh_red[tid] = fmaxf(sh_red[tid], sh_red[tid + s]);
        __syncthreads();
    }
    const float gmax = sh_red[0];

    /* 2b. exp and partial sum */
    thr = 0.0f;
    for (int i = tid; i < Nkv; i += BLK) {
        float e = expf(sh_h[i] - gmax);
        sh_m[i] = e;
        thr    += e;
    }
    sh_red[tid] = thr;
    __syncthreads();
    for (int s = BLK >> 1; s > 0; s >>= 1) {
        if (tid < s) sh_red[tid] += sh_red[tid + s];
        __syncthreads();
    }
    const float inv_s0 = 1.0f / (sh_red[0] + 1e-8f);
    for (int i = tid; i < Nkv; i += BLK) sh_m[i] *= inv_s0;
    __syncthreads();

    /* ---- 3. Gibbs iterations ---- */
    const float* J_base = J + (long)b * Nkv * Nkv;

    for (int step = 0; step < n_steps; step++) {
        const float frac   = (n_steps > 1) ? (float)step / (float)(n_steps - 1) : 0.0f;
        const float beta_t = beta_start + frac * (beta_end - beta_start) + beta_offset;

        /* 3a. coupling[j] = Σ_i m[i] · J[b, i, j]
               Thread j handles coupling[j]; reads column j of J (strided access,
               but J[Nkv×Nkv] typically fits in L2 after first step).         */
        for (int j = tid; j < Nkv; j += BLK) {
            float c = 0.0f;
            for (int i = 0; i < Nkv; i++)
                c += sh_m[i] * J_base[(long)i * Nkv + j];
            sh_c[j] = c;
        }
        __syncthreads();

        /* 3b. Potts TAP update: m = softmax(beta_t * (coupling + h))
               Uses the same online-softmax pattern as the init step.        */
        /* max reduction */
        thr = -1e30f;
        for (int j = tid; j < Nkv; j += BLK) thr = fmaxf(thr, beta_t * (sh_c[j] + sh_h[j]));
        sh_red[tid] = (tid < Nkv) ? thr : -1e30f;
        __syncthreads();
        for (int s = BLK >> 1; s > 0; s >>= 1) {
            if (tid < s) sh_red[tid] = fmaxf(sh_red[tid], sh_red[tid + s]);
            __syncthreads();
        }
        const float step_max = sh_red[0];

        /* exp and sum */
        thr = 0.0f;
        for (int j = tid; j < Nkv; j += BLK) {
            const float e = expf(beta_t * (sh_c[j] + sh_h[j]) - step_max);
            sh_m[j] = e;
            thr    += e;
        }
        sh_red[tid] = thr;
        __syncthreads();
        for (int s = BLK >> 1; s > 0; s >>= 1) {
            if (tid < s) sh_red[tid] += sh_red[tid + s];
            __syncthreads();
        }
        const float inv_s = 1.0f / (sh_red[0] + 1e-8f);
        for (int j = tid; j < Nkv; j += BLK) sh_m[j] *= inv_s;
        __syncthreads();
    }

    /* ---- 4. out[b, qi, d] = Σ_i m[i] · v[b, i, d] ---- */
    const float* v_base = v + (long)b * Nkv * D;
    float*       o_ptr  = out + (long)(b * Nq + qi) * D;
    for (int d = tid; d < D; d += BLK) {
        float s = 0.0f;
        for (int i = 0; i < Nkv; i++)
            s += sh_m[i] * v_base[(long)i * D + d];
        o_ptr[d] = s;
    }
}


/* ---- Python-facing launcher ---- */
torch::Tensor thermo_gibbs_cuda(
    torch::Tensor h,           /* [B, Nq, Nkv]   float32, contiguous */
    torch::Tensor J,           /* [B, Nkv, Nkv]  float32, contiguous */
    torch::Tensor v,           /* [B, Nkv, D]    float32, contiguous */
    float         beta_start,
    float         beta_end,
    float         beta_offset,
    int           n_steps
) {
    TORCH_CHECK(h.is_cuda(),       "h must be a CUDA tensor");
    TORCH_CHECK(h.is_contiguous(), "h must be contiguous");
    TORCH_CHECK(J.is_contiguous(), "J must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");

    const int B   = h.size(0);
    const int Nq  = h.size(1);
    const int Nkv = h.size(2);
    const int D   = v.size(2);

    /* Block size = next power-of-2 >= max(Nkv, D), capped at 1024 */
    int blk = 32;
    int need = (Nkv > D) ? Nkv : D;
    while (blk < need && blk < 1024) blk <<= 1;
    if (blk > 1024) blk = 1024;

    const int shmem_bytes = (3 * Nkv + blk) * sizeof(float);

    auto out = torch::empty({B, Nq, D}, h.options());

    const dim3 grid(B, Nq);
    const dim3 block(blk);

    thermo_gibbs_fwd<<<grid, block, shmem_bytes,
                       at::cuda::getCurrentCUDAStream()>>>(
        h.data_ptr<float>(),
        J.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        beta_start, beta_end, beta_offset,
        n_steps, Nq, Nkv, D
    );

    return out;
}


/* ---- pybind11 module — same translation unit as the kernel launcher ---- */
#include <pybind11/pybind11.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("thermo_gibbs_cuda",
          &thermo_gibbs_cuda,
          "Fused thermodynamic Gibbs attention (CUDA float32)");
}
"""

# ---------------------------------------------------------------------------
# Lazy compilation
# ---------------------------------------------------------------------------

_ext = None          # compiled extension module, or None
_ext_tried = False   # whether we already attempted compilation

# Wipe stale cached build so the compiler picks up any source changes
def _clear_stale_cache():
    import shutil, pathlib
    cache = pathlib.Path.home() / ".cache" / "torch_extensions"
    for d in cache.glob("**/thermo_gibbs_cuda_ext"):
        try:
            shutil.rmtree(d)
        except Exception:
            pass


def _load_ext() -> bool:
    """Compile and load the CUDA extension on first call. Returns True if OK."""
    global _ext, _ext_tried
    if _ext_tried:
        return _ext is not None
    _ext_tried = True

    if not torch.cuda.is_available():
        return False

    try:
        from torch.utils.cpp_extension import load_inline
        # Use functions= only (no cpp_sources) so load_inline generates a single
        # PYBIND11_MODULE automatically — avoids the duplicate-module compile error.
        # Put the PYBIND11 binding directly in cuda_sources alongside the kernel.
        # This is the most reliable approach: one compilation unit, no cross-TU
        # linkage issues that arise when using functions= with separate cpp_sources.
        # cpp_sources is a required positional arg; pass empty string.
        # The PYBIND11_MODULE is defined at the bottom of _CUDA_SRC so that
        # the kernel launcher and its binding are in the same translation unit.
        # Signature: load_inline(name, cpp_sources, cuda_sources=..., ...)
        # PYBIND11_MODULE lives inside _CUDA_SRC (same TU as kernel launcher),
        # so cpp_sources can be empty.
        _ext = load_inline(
            "thermo_gibbs_cuda_ext",   # name
            "",                         # cpp_sources (empty)
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        log.info("thermo_gibbs CUDA kernel compiled and loaded.")
        return True
    except Exception as e:
        log.warning(
            "Could not compile thermo_gibbs CUDA kernel (%s). "
            "Falling back to vectorised PyTorch.",
            e,
        )
        _ext = None
        return False


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

def thermo_gibbs_cuda(
    h: torch.Tensor,          # (B, q_len, kv_len) — mask already applied
    J: torch.Tensor,          # (B, kv_len, kv_len)
    v: torch.Tensor,          # (B, kv_len, D)
    beta_start: float,
    beta_end: float,
    beta_offset: float,
    n_steps: int,
) -> torch.Tensor | None:
    """
    Run the fused CUDA Gibbs kernel.

    Returns the output tensor (B, q_len, D) **in the same dtype as h**,
    or *None* if CUDA compilation failed (caller should use PyTorch fallback).
    """
    if not _load_ext():
        return None

    orig_dtype  = h.dtype
    orig_device = h.device

    # Kernel operates in float32; bfloat16/float16 inputs are up-cast here.
    h32 = h.float().contiguous()
    J32 = J.float().contiguous()
    v32 = v.float().contiguous()

    out32 = _ext.thermo_gibbs_cuda(h32, J32, v32, beta_start, beta_end, beta_offset, n_steps)

    return out32.to(orig_dtype)
