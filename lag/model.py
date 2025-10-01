"""Decoder-only transformer language model for LAG with modern enhancements."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .config import ModelConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def build_norm(config: ModelConfig) -> nn.Module:
    if config.norm_type.lower() == "layernorm":
        return nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
    if config.norm_type.lower() == "rmsnorm":
        return RMSNorm(config.d_model, eps=config.layer_norm_eps)
    raise ValueError(f"Unsupported norm type: {config.norm_type}")


def swiglu(x: torch.Tensor) -> torch.Tensor:
    x, gate = x.chunk(2, dim=-1)
    return F.silu(x) * gate


def geglu(x: torch.Tensor) -> torch.Tensor:
    x, gate = x.chunk(2, dim=-1)
    return F.gelu(x) * gate


def get_activation_fn(name: str):
    name = name.lower()
    if name == "swiglu":
        return swiglu
    if name == "geglu":
        return geglu
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported activation: {name}")


def repeat_kv(hidden: torch.Tensor, n_repeat: int) -> torch.Tensor:
    if n_repeat == 1:
        return hidden
    return hidden.repeat_interleave(n_repeat, dim=1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.config = config
        self.n_heads = config.n_heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.d_model // config.n_heads
        if config.use_rotary and self.head_dim % 2 != 0:
            raise ValueError("Head dimension must be even when using rotary embeddings")
        self.scale = self.head_dim**-0.5

        bias = config.use_bias
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=bias)
        self.k_proj = nn.Linear(config.d_model, self.kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.d_model, self.kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=bias)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        self.use_rotary = config.use_rotary
        self.rotary_base = config.rope_theta
        self.rotary_scaling = config.rope_scaling
        if self.use_rotary:
            self.register_buffer("_cos_cache", torch.empty(0), persistent=False)
            self.register_buffer("_sin_cache", torch.empty(0), persistent=False)
            self._rope_seq_len = 0

    def _build_rope_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._cos_cache.numel() > 0 and self._cos_cache.size(-2) >= seq_len:
            if self._cos_cache.device == device and self._cos_cache.dtype == dtype:
                return
        base = self.rotary_base if self.rotary_scaling is None else self.rotary_base * self.rotary_scaling
        half_dim = self.head_dim // 2
        freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (freq_seq / max(1, half_dim)))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = torch.cos(emb)[None, None, :, :].to(dtype=dtype)
        sin = torch.sin(emb)[None, None, :, :].to(dtype=dtype)
        self._cos_cache = cos
        self._sin_cache = sin
        self._rope_seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary:
            self._build_rope_cache(seq_len, x.device, q.dtype)
            cos = self._cos_cache[..., :seq_len, :]
            sin = self._sin_cache[..., :seq_len, :]
            q = apply_rotary(q, sin, cos)
            k = apply_rotary(k, sin, cos)

        if self.kv_heads != self.n_heads:
            repeat_factor = self.n_heads // self.kv_heads
            k = repeat_kv(k, repeat_factor)
            v = repeat_kv(v, repeat_factor)

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal_mask, torch.finfo(att.dtype).min)
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        y = self.resid_dropout(self.out_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.ff_hidden_size
        self.activation_name = config.activation.lower()
        self.activation = get_activation_fn(config.activation)
        mult = 2 if self.activation_name in {"swiglu", "geglu"} else 1
        bias = config.use_bias
        self.in_proj = nn.Linear(config.d_model, hidden * mult, bias=bias)
        self.out_proj = nn.Linear(hidden, config.d_model, bias=bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.activation(x)
        x = self.out_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = build_norm(config)
        self.attn = CausalSelfAttention(config)
        self.norm2 = build_norm(config)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class LAGLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.use_rotary = config.use_rotary
        self.pos_embedding = (
            None
            if self.use_rotary
            else nn.Embedding(config.max_seq_len, config.d_model)
        )
        self.embed_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = build_norm(config)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=config.use_bias)
        if config.tie_embeddings:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if idx.ndim != 2:
            raise ValueError("Input tensor must be 2D (batch, seq_len)")
        bsz, seq_len = idx.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("Sequence length exceeds model capacity")

        device = idx.device
        tok_emb = self.token_embedding(idx)
        if self.pos_embedding is not None:
            positions = torch.arange(seq_len, device=device, dtype=torch.long)
            pos_emb = self.pos_embedding(positions)[None, :, :]
            x = tok_emb + pos_emb
        else:
            x = tok_emb
        x = self.embed_dropout(x)

        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = activation_checkpoint(block, x)
            else:
                x = block(x)

        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature <= 0:
                temperature = 1.0
                do_sample = False
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                values, indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, indices, values)
                logits = mask
            probs = torch.softmax(logits, dim=-1)
            if top_p is not None and 0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = sorted_probs.cumsum(dim=-1)
                mask = cumulative > top_p
                mask[..., 1:] = mask[..., :-1]
                mask[..., 0] = False
                sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, next_token)
            elif do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
