"""Causal transformer architecture with paged KV cache attention.

Implements a Qwen3-compatible transformer with:
- Fused QKV and gate/up projections for efficient weight loading
- Paged KV cache with Flash Attention / SDPA backends
- RoPE position encoding with compiled forward passes
- Safetensors weight loading with shard-aware mapping
"""

import os
from functools import lru_cache
from glob import glob

import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import Qwen3Config

_HAS_TRITON = False
_HAS_FLASH_ATTN = False

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    _HAS_FLASH_ATTN = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# KV cache write operations
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    @triton.jit
    def _triton_kv_write(
        key_ptr, key_stride, value_ptr, value_stride,
        k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        offs = tl.arange(0, D)
        tl.store(k_cache_ptr + slot * D + offs, tl.load(key_ptr + idx * key_stride + offs))
        tl.store(v_cache_ptr + slot * D + offs, tl.load(value_ptr + idx * value_stride + offs))


def _torch_kv_write(key, value, k_cache, v_cache, slot_mapping):
    N, num_kv_heads, head_dim = key.shape
    D = num_kv_heads * head_dim
    valid = slot_mapping != -1
    slots = slot_mapping[valid]
    k_cache.reshape(-1, D)[slots] = key.reshape(N, D)[valid]
    v_cache.reshape(-1, D)[slots] = value.reshape(N, D)[valid]


def write_kv_cache(key, value, k_cache, v_cache, slot_mapping):
    """Persist key/value tensors into the paged KV cache."""
    if _HAS_TRITON:
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        _triton_kv_write[(N,)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping, D,
        )
    else:
        _torch_kv_write(key, value, k_cache, v_cache, slot_mapping)


# ---------------------------------------------------------------------------
# SDPA fallback implementations
# ---------------------------------------------------------------------------

def _sdpa_packed_prefill(q, k, v, cu_q, cu_k, scale, n_heads, n_kv):
    results = []
    gqa = n_heads != n_kv
    for i in range(cu_q.shape[0] - 1):
        qs, qe = cu_q[i].item(), cu_q[i + 1].item()
        ks, ke = cu_k[i].item(), cu_k[i + 1].item()
        qi = q[qs:qe].unsqueeze(0).transpose(1, 2)
        ki = k[ks:ke].unsqueeze(0).transpose(1, 2)
        vi = v[ks:ke].unsqueeze(0).transpose(1, 2)
        oi = F.scaled_dot_product_attention(qi, ki, vi, scale=scale, is_causal=True, enable_gqa=gqa)
        results.append(oi.transpose(1, 2).squeeze(0))
    return torch.cat(results, dim=0)


def _sdpa_cached_decode(q, k_cache, v_cache, ctx_lens, block_tbl, scale, n_heads, n_kv):
    blk_sz = k_cache.shape[1]
    results = []
    gqa = n_heads != n_kv
    for i in range(q.shape[0]):
        cl = ctx_lens[i].item()
        nb = (cl + blk_sz - 1) // blk_sz
        idx = block_tbl[i, :nb]
        ki = k_cache[idx].reshape(-1, n_kv, k_cache.shape[-1])[:cl]
        vi = v_cache[idx].reshape(-1, n_kv, v_cache.shape[-1])[:cl]
        qi = q[i].unsqueeze(0).transpose(1, 2)
        ki = ki.unsqueeze(0).transpose(1, 2)
        vi = vi.unsqueeze(0).transpose(1, 2)
        oi = F.scaled_dot_product_attention(qi, ki, vi, scale=scale, is_causal=False, enable_gqa=gqa)
        results.append(oi.transpose(1, 2).squeeze(0))
    return torch.stack(results, dim=0)


# ---------------------------------------------------------------------------
# Attention with paged KV cache
# ---------------------------------------------------------------------------

class PagedAttention(nn.Module):
    """Multi-head attention with paged KV cache, Flash Attention or SDPA backend."""

    def __init__(self, num_heads, head_dim, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self._scale = head_dim ** -0.5
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q, k, v):
        from acestep.customized_vllm import _get_forward_state
        state = _get_forward_state()

        if self.k_cache.numel() and self.v_cache.numel():
            write_kv_cache(k, v, self.k_cache, self.v_cache, state.slot_mapping)

        if _HAS_FLASH_ATTN:
            return self._flash_path(q, k, v, state)
        return self._sdpa_path(q, k, v, state)

    def _flash_path(self, q, k, v, state):
        if state.is_prefill:
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=state.cu_seqlens_q, cu_seqlens_k=state.cu_seqlens_k,
                max_seqlen_q=state.max_seqlen_q, max_seqlen_k=state.max_seqlen_k,
                softmax_scale=self._scale, causal=True,
            )
        return flash_attn_with_kvcache(
            q.unsqueeze(1), self.k_cache, self.v_cache,
            cache_seqlens=state.context_lens, block_table=state.block_tables,
            softmax_scale=self._scale, causal=True,
        )

    def _sdpa_path(self, q, k, v, state):
        if state.is_prefill:
            return _sdpa_packed_prefill(
                q, k, v, state.cu_seqlens_q, state.cu_seqlens_k,
                self._scale, self.num_heads, self.num_kv_heads,
            )
        return _sdpa_cached_decode(
            q.unsqueeze(1), self.k_cache, self.v_cache,
            state.context_lens, state.block_tables,
            self._scale, self.num_heads, self.num_kv_heads,
        )


# ---------------------------------------------------------------------------
# Core neural-network layers
# ---------------------------------------------------------------------------

class NormLayer(nn.Module):
    """Root-mean-square layer normalisation with optional fused residual add."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def _normalize(self, x):
        dt = x.dtype
        x = x.float()
        x.mul_(torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps))
        return x.to(dt).mul_(self.weight)

    @torch.compile
    def _fused_add_normalize(self, x, residual):
        dt = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(dt)
        x.mul_(torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps))
        return x.to(dt).mul_(self.weight), residual

    def forward(self, x, residual=None):
        if residual is None:
            return self._normalize(x)
        return self._fused_add_normalize(x, residual)


def _rotate(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)


class PositionEncoding(nn.Module):
    """Rotary position embedding (RoPE)."""

    def __init__(self, head_size: int, max_position: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2, dtype=torch.float) / head_size))
        freqs = torch.einsum("i,j->ij", torch.arange(max_position, dtype=torch.float), inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(self, positions, query, key):
        cs = self.cos_sin_cache[positions]
        cos, sin = cs.chunk(2, dim=-1)
        return _rotate(query, cos, sin), _rotate(key, cos, sin)


@lru_cache(1)
def get_position_encoder(head_size: int, max_position: int, base: float):
    return PositionEncoding(head_size, max_position, base)


class GatedActivation(nn.Module):
    """SiLU-gated activation: SiLU(x) * y where [x, y] = chunk(input, 2)."""

    @torch.compile
    def forward(self, x):
        a, b = x.chunk(2, -1)
        return F.silu(a) * b


# ---------------------------------------------------------------------------
# Fused projections with shard-aware weight loaders
# ---------------------------------------------------------------------------

def _FusedQKVProjection(hidden_size, num_heads, num_kv_heads, head_dim, bias):
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    proj = nn.Linear(hidden_size, q_size + 2 * kv_size, bias=bias)

    def _load_shard(param, weight, shard_id):
        offsets = {"q": 0, "k": q_size, "v": q_size + kv_size}
        sizes = {"q": q_size, "k": kv_size, "v": kv_size}
        param.data.narrow(0, offsets[shard_id], sizes[shard_id]).copy_(weight)

    proj.weight.weight_loader = _load_shard
    if bias:
        proj.bias.weight_loader = _load_shard
    return proj


def _FusedGateUpProjection(hidden_size, intermediate_size):
    proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)

    def _load_shard(param, weight, shard_id):
        param.data.narrow(0, shard_id * intermediate_size, intermediate_size).copy_(weight)

    proj.weight.weight_loader = _load_shard
    return proj


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class _SelfAttention(nn.Module):
    """Self-attention block with fused QKV, RoPE, and optional QK-norm."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        has_bias = getattr(config, "attention_bias", True)

        self.qkv_proj = _FusedQKVProjection(
            config.hidden_size, self.num_heads, self.num_kv_heads, self.head_dim, has_bias,
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rope = get_position_encoder(
            self.head_dim, config.max_position_embeddings,
            getattr(config, "rope_theta", 1000000),
        )
        self.attn = PagedAttention(self.num_heads, self.head_dim, self.num_kv_heads)
        if not has_bias:
            self.q_norm = NormLayer(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = NormLayer(self.head_dim, eps=config.rms_norm_eps)
        self._has_bias = has_bias

    def forward(self, positions, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self._has_bias:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(positions, q, k)
        return self.o_proj(self.attn(q, k, v).flatten(1, -1))


class _FeedForward(nn.Module):
    """MLP with fused gate/up projection and SiLU gating."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_up_proj = _FusedGateUpProjection(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = GatedActivation()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_up_proj(x)))


class _TransformerBlock(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = _SelfAttention(config)
        self.mlp = _FeedForward(config)
        self.input_layernorm = NormLayer(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NormLayer(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class CausalTransformer(nn.Module):
    """Causal language model for inference (Qwen3-compatible)."""

    WEIGHT_SHARD_MAP = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([_TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = NormLayer(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, positions):
        h = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            h, residual = layer(positions, h, residual)
        h, _ = self.norm(h, residual)
        return h

    def project_to_vocab(self, hidden_states):
        """Extract last-token hidden states (during prefill) and project to vocabulary logits."""
        from acestep.customized_vllm import _get_forward_state
        state = _get_forward_state()
        if state.is_prefill:
            hidden_states = hidden_states[state.cu_seqlens_q[1:] - 1].contiguous()
        return self.lm_head(hidden_states)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _copy_weight(param, loaded_weight):
    param.data.copy_(loaded_weight)


def _resolve_parameter(model, name):
    for candidate in (name, f"model.{name}", name.removeprefix("model.")):
        try:
            return model.get_parameter(candidate)
        except AttributeError:
            continue
    return None


def load_weights(model: nn.Module, path: str):
    """Load safetensors weights into model with shard-aware mapping for fused projections."""
    shard_map = getattr(model, "WEIGHT_SHARD_MAP", {})
    files = glob(os.path.join(path, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {path}")

    for filepath in files:
        with safe_open(filepath, "pt", "cpu") as f:
            for weight_name in f.keys():
                for src_key, (dst_key, shard_id) in shard_map.items():
                    if src_key in weight_name:
                        param_name = weight_name.replace(src_key, dst_key)
                        param = _resolve_parameter(model, param_name)
                        if param is None:
                            continue
                        param.weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = _resolve_parameter(model, weight_name)
                    if param is None:
                        continue
                    loader = getattr(param, "weight_loader", _copy_weight)
                    loader(param, f.get_tensor(weight_name))
