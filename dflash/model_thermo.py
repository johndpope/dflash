"""
dflash/model_thermo.py
Thermodynamic Flash Attention (ThermoFTA) — PyTorch port for DFlash.

Replaces Qwen3DFlashAttention with a mean-field Ising Gibbs sampler:
  - QK^T logits become the external field h in the Ising Hamiltonian
  - A learned coupling matrix J extends beyond pairwise softmax routing
  - Annealed Gibbs steps (beta: beta_start → beta_end) find the low-energy
    attention configuration — more globally coherent than single-pass softmax
  - Compatible with the full dflash_generate / DFlashDraftModel interface

Drop-in swap:
    from dflash.model import DFlashDraftModel, dflash_generate
  →
    from dflash.model_thermo import ThermoDFlashDraftModel as DFlashDraftModel, dflash_generate

The dflash_generate function is re-exported unchanged — only the draft model
internal attention is replaced.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from typing_extensions import Unpack

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers import DynamicCache
from transformers.cache_utils import Cache

# Re-export the unchanged generation loop and utilities
from .model import (
    dflash_generate,
    extract_context_feature,
    sample,
    build_target_layer_ids,
    apply_rotary_pos_emb,
)


# ---------------------------------------------------------------------------
# Core: Mean-field Ising Gibbs sampler
# ---------------------------------------------------------------------------

class ThermoGibbsSampler(nn.Module):
    """
    Mean-field approximation of Ising model Gibbs sampling for attention.

    Ising Hamiltonian for one attention head:
        E(m) = -beta * (m^T J m + h^T m)
    where:
        h  = QK^T / sqrt(d)          (external field — standard attention logits)
        J  = Q (J_proj K)^T / sqrt(d) (learned couplings beyond softmax)
        m  ∈ [0,1]^{q_len × kv_len}   (mean-field magnetisation / soft attention weights)

    TAP mean-field update (each Gibbs step):
        m_i ← sigmoid(beta_t * (Σ_j J_ij·m_j + h_i))

    After n_steps of annealed refinement, output = normalize(m) @ V.

    Args:
        head_dim:    attention head dimension
        n_steps:     number of Gibbs / mean-field refinement steps (default 4)
        beta_start:  initial inverse temperature (default 0.8)
        beta_end:    final inverse temperature (default 1.2)
    """

    def __init__(
        self,
        head_dim: int,
        n_steps: int = 4,
        beta_start: float = 0.8,
        beta_end: float = 1.2,
    ):
        super().__init__()
        self.head_dim   = head_dim
        self.n_steps    = n_steps
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.scale      = head_dim ** -0.5

        # Learned coupling matrix — projects keys into coupling space.
        # Near-zero init: J ≈ 0 at the start so coupling ≈ 0 and the Gibbs
        # sampler's initial output ≈ standard attention.  The model can then
        # gradually learn to use inter-token couplings without disrupting the
        # pretrained Q/K/V weights from the start.
        self.J_proj = nn.Linear(head_dim, head_dim, bias=False)
        nn.init.normal_(self.J_proj.weight, mean=0.0, std=0.01)

        # Learnable log-temperature offset (shared across heads via broadcasting)
        self.log_beta_offset = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q:  (B, q_len, D)   — query (draft/noise positions)
            k:  (B, kv_len, D)  — keys (target_hidden ‖ noise)
            v:  (B, kv_len, D)  — values
        Returns:
            out: (B, q_len, D)
        """
        # External field: standard attention logits
        h = q @ k.transpose(-2, -1) * self.scale          # (B, q_len, kv_len)

        # Coupling matrix: kv_len × kv_len key-key coupling
        # J[b,i,j] = k[b,i] · J_proj(k)[b,j] — models how key i
        # should influence the weight on key j beyond what softmax captures.
        J_k = self.J_proj(k)                              # (B, kv_len, D)
        J   = k @ J_k.transpose(-2, -1) * self.scale      # (B, kv_len, kv_len)
        J   = torch.tanh(J)                               # bound couplings ∈ (-1,1)

        if attention_mask is not None:
            h = h + attention_mask

        beta_offset = (
            float(torch.sigmoid(self.log_beta_offset).detach())
            * (self.beta_end - self.beta_start)
        )

        # Try fused CUDA kernel first (compiled once, cached)
        if q.is_cuda:
            from .thermo_cuda_kernel import thermo_gibbs_cuda
            out = thermo_gibbs_cuda(
                h, J, v,
                self.beta_start, self.beta_end, beta_offset,
                self.n_steps,
            )
            if out is not None:
                return out

        # Vectorised PyTorch fallback (GPU or CPU)
        #
        # Potts mean-field TAP update (softmax form, not sigmoid):
        #   m ← softmax(β · (m @ J + h))
        #
        # At J=0 this reduces to softmax(β·h), which for β=1 is exactly standard
        # attention — so pretrained DFlash weights work without modification.
        # As J_proj learns, the coupling enriches the attention beyond softmax.
        m = torch.softmax(h, dim=-1)                      # (B, q_len, kv_len)

        for step in range(self.n_steps):
            frac   = step / max(self.n_steps - 1, 1)
            beta_t = self.beta_start + frac * (self.beta_end - self.beta_start) + beta_offset

            # coupling[b, q, j] = Σ_i m[b, q, i] · J[b, i, j]
            coupling = m @ J                              # (B, q_len, kv_len)
            m        = torch.softmax(beta_t * (coupling + h), dim=-1)

        return m @ v                                      # (B, q_len, D)


# ---------------------------------------------------------------------------
# ThermoDFlashAttention — replaces Qwen3DFlashAttention
# ---------------------------------------------------------------------------

class ThermoDFlashAttention(nn.Module):
    """
    Thermodynamic Flash Attention for DFlash draft model.

    Same interface as Qwen3DFlashAttention but routes through ThermoGibbsSampler
    instead of softmax. Keeps all Q/K/V projections, RMS norms, and RoPE
    identical to the original — only the attention kernel changes.
    """

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        n_gibbs_steps: int = 4,
        beta_start: float = 0.8,
        beta_end: float = 1.2,
    ):
        super().__init__()
        self.config    = config
        self.layer_idx = layer_idx
        self.head_dim  = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  # DFlash draft is always non-causal
        self.sliding_window = (
            config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        )

        # Same projections as original
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # One Gibbs sampler per head — each head has its own coupling matrix J
        self.gibbs = nn.ModuleList([
            ThermoGibbsSampler(self.head_dim, n_gibbs_steps, beta_start, beta_end)
            for _ in range(self.num_heads)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _  = hidden_states.shape
        ctx_len        = target_hidden.shape[1]

        # Project queries from noise/draft tokens
        q = self.q_proj(hidden_states)                               # (B, q_len, H*D)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)                          # (B, H, q_len, D)

        # Keys and values from [target_hidden ‖ noise] (same as original DFlash)
        k_ctx   = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx   = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)                          # (B, H_kv, ctx+q, D)
        v = v.transpose(1, 2)

        # RoPE (positional encoding — unchanged)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache for decode steps
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # Expand GQA groups (num_kv_groups > 1 means grouped-query attention)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # --- Thermodynamic Gibbs attention (per head) ---
        # Reshape for per-head sampler: each gets (B, q_len, D) / (B, kv_len, D)
        head_outputs = []
        for h_idx, gibbs in enumerate(self.gibbs):
            q_h = q[:, h_idx, :, :]                                 # (B, q_len, D)
            k_h = k[:, h_idx, :, :]                                 # (B, kv_len, D)
            v_h = v[:, h_idx, :, :]                                 # (B, kv_len, D)
            out_h = gibbs(q_h, k_h, v_h, attention_mask)            # (B, q_len, D)
            head_outputs.append(out_h)

        # Concatenate heads and project output
        attn_output = torch.cat(head_outputs, dim=-1)               # (B, q_len, H*D)
        attn_output = self.o_proj(attn_output)
        return attn_output, None                                     # no attn_weights returned


# ---------------------------------------------------------------------------
# ThermoDFlashDecoderLayer — swaps in ThermoDFlashAttention
# ---------------------------------------------------------------------------

class ThermoDFlashDecoderLayer(GradientCheckpointingLayer):
    """Identical to Qwen3DFlashDecoderLayer but uses ThermoDFlashAttention."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        n_gibbs_steps: int = 4,
        beta_start: float = 0.8,
        beta_end: float = 1.2,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = ThermoDFlashAttention(
            config, layer_idx, n_gibbs_steps, beta_start, beta_end
        )
        self.mlp = Qwen3MLP(config)
        self.input_layernorm       = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        residual     = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual     = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# ThermoDFlashDraftModel — full drop-in for DFlashDraftModel
# ---------------------------------------------------------------------------

class ThermoDFlashDraftModel(Qwen3PreTrainedModel):
    """
    Drop-in replacement for DFlashDraftModel with thermodynamic attention.

    Same forward() / spec_generate() API.  Only the internal attention kernel
    differs: softmax → annealed Ising mean-field Gibbs sampler.

    Extra config keys (under config.dflash_config):
        n_gibbs_steps  (int, default 4)
        beta_start     (float, default 0.8)
        beta_end       (float, default 1.2)
    """

    config_class = Qwen3Config
    _no_split_modules = ["ThermoDFlashDecoderLayer"]

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        dflash_cfg    = getattr(config, "dflash_config", {})
        n_gibbs_steps = dflash_cfg.get("n_gibbs_steps", 4)
        beta_start    = dflash_cfg.get("beta_start", 0.8)
        beta_end      = dflash_cfg.get("beta_end",   1.2)

        self.layers = nn.ModuleList([
            ThermoDFlashDecoderLayer(config, layer_idx, n_gibbs_steps, beta_start, beta_end)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.target_layer_ids = dflash_cfg.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm        = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb  = Qwen3RotaryEmbedding(config)
        self.fc          = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size  = config.block_size
        self.mask_token_id = dflash_cfg.get("mask_token_id", None)
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
    ):
        self.eval()
        return dflash_generate(
            self,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )


# ---------------------------------------------------------------------------
# Distillation training helpers
# ---------------------------------------------------------------------------

def thermo_distillation_loss(
    draft_hidden: torch.Tensor,
    target_lm_head: nn.Module,
    target_logits: torch.Tensor,
    temperature: float = 2.0,
    gibbs_entropy_weight: float = 0.01,
    model: Optional[ThermoDFlashDraftModel] = None,
    kl_weight: float = 0.9,
) -> dict[str, torch.Tensor]:
    """
    Compute distillation loss for ThermoDFlashDraftModel.

    Three terms:
    1. KL divergence on soft targets (temperature-scaled) — dense gradient
       signal across the full vocabulary (dominant term, weight=kl_weight)
    2. Hard cross-entropy on argmax targets — ensures top-1 accuracy
       (weight=1-kl_weight)
    3. Gibbs entropy regularisation — keep log_beta_offset near zero

    Soft-target KL divergence (Hinton et al., 2015) converges far faster
    than hard CE because every logit position contributes gradient, not just
    the single argmax token.

    Args:
        draft_hidden:         (B, seq, D)
        target_lm_head:       target model's lm_head (not trained)
        target_logits:        (B, seq, vocab) — detached target logits
        temperature:          distillation temperature (>1 softens targets)
        gibbs_entropy_weight: weight for Gibbs regularisation term
        model:                ThermoDFlashDraftModel (for reg term)
        kl_weight:            fraction of loss from KL vs hard CE (default 0.9)

    Returns:
        dict with 'loss', 'ce_loss', 'kl_loss', 'entropy_reg'
    """
    import torch.nn.functional as F

    draft_logits = target_lm_head(draft_hidden)                      # (B, seq, V)

    # ── 1. KL divergence on soft (temperature-scaled) targets ────────────────
    # Scale both sides by T; KL(p_teacher || p_student) × T² is the standard
    # Hinton formulation that keeps gradient magnitude independent of T.
    T = max(temperature, 1e-5)
    with torch.no_grad():
        p_teacher = F.softmax(target_logits.float() / T, dim=-1)     # (B, seq, V)
    log_p_student = F.log_softmax(draft_logits.float() / T, dim=-1)  # (B, seq, V)
    kl_loss = F.kl_div(
        log_p_student.view(-1, log_p_student.shape[-1]),
        p_teacher.view(-1, p_teacher.shape[-1]),
        reduction="batchmean",
    ) * (T ** 2)

    # ── 2. Hard CE on argmax token (ground-truth token accuracy) ─────────────
    with torch.no_grad():
        target_tokens = target_logits.argmax(dim=-1)                  # (B, seq)
    ce_loss = F.cross_entropy(
        draft_logits.view(-1, draft_logits.shape[-1]),
        target_tokens.view(-1),
        ignore_index=-100,
    )

    # ── 3. Gibbs entropy regularisation ──────────────────────────────────────
    entropy_reg = torch.tensor(0.0, device=draft_hidden.device)
    if model is not None and gibbs_entropy_weight > 0:
        for layer in model.layers:
            for sampler in layer.self_attn.gibbs:
                entropy_reg = entropy_reg + sampler.log_beta_offset.abs()
        entropy_reg = entropy_reg / (len(model.layers) * len(model.layers[0].self_attn.gibbs))

    loss = kl_weight * kl_loss + (1 - kl_weight) * ce_loss + gibbs_entropy_weight * entropy_reg
    return {
        "loss":        loss,
        "ce_loss":     ce_loss,
        "kl_loss":     kl_loss,
        "entropy_reg": entropy_reg,
    }
