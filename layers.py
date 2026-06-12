import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention
from typing import Tuple, Optional

flex_attention = torch.compile(flex_attention)

def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.reshape(-1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1]
        ],
        dim=-1
    )

    x_out = x_out.flatten(3)
    x_out = x_out.type_as(x)

    return x_out.transpose(1, 2)


class RoPE(nn.Module):
    def __init__(
            self,
            dim: int,
            max_seq_len: int = 4096,
            base: int = 10_000
    ) -> None:
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
            )
        self.register_buffer("theta", theta, persistent=False)

        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)


# 1. RoPE.
# 2. GQA.
class GKVRoPEAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            num_kv_heads: int
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.depth = d_model // num_heads
        self.n_rep = num_heads // num_kv_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, num_kv_heads * self.depth, bias=False)
        self.wv = nn.Linear(d_model, num_kv_heads * self.depth, bias=False)

        self.fc = nn.Linear(d_model, d_model, bias=False)

    def split_heads(
            self,
            x: torch.Tensor,
            batch_size: int,
            num_heads: int
    ) -> torch.Tensor:

        x = torch.reshape(x, [batch_size, -1, num_heads, self.depth])
        return torch.permute(x, [0, 2, 1, 3])

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                rope_cache: torch.Tensor,
                block_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size, self.num_heads)
        k = self.split_heads(k, batch_size, self.num_kv_heads)
        v = self.split_heads(v, batch_size, self.num_kv_heads)

        q = apply_rope(q, rope_cache)
        k = apply_rope(k, rope_cache)

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        if block_mask is not None:
            scaled_attention = flex_attention(q, k, v, block_mask=block_mask)

        else:
            scaled_attention = F.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )

        scaled_attention = torch.permute(scaled_attention, [0, 2, 1, 3])
        concat_attention = torch.reshape(
            scaled_attention, (batch_size, -1, self.d_model))
        output = self.fc(concat_attention)

        return output


class SwiGLUFeedForward(nn.Module):
    def __init__(
            self,
            d_model: int,
            dff: int,
            multiple_of: int = 4,
            ffn_dim_multiplier: Optional[float] = None
    ):
        super().__init__()

        dff = int(2 * dff / 3)

        if ffn_dim_multiplier is not None:
            dff = int(ffn_dim_multiplier * dff)
        dff = multiple_of * ((dff + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(d_model, dff, bias=False)
        self.w2 = nn.Linear(dff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, dff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
