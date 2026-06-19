import torch
from sympy.physics.quantum.grover import superposition_basis
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from typing import Tuple, Optional, Callable
from .layers import *
from .masking import *


class DecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            num_kv_heads: int,
            ff_factory: Callable[[], nn.Module],
            norm_factory: Callable[[], nn.Module]
    ):
        super().__init__()

        self.self_attention = GKVRoPEAttention(d_model, num_heads, num_kv_heads)

        self.ff_module = ff_factory()
        self.norm1 = norm_factory()
        self.norm2 = norm_factory()

    def forward(
            self,
            x: torch.Tensor,
            rope_cache: torch.Tensor,
            block_mask = None
    ) -> torch.Tensor:

        nx = self.norm1(x)
        self_attention_output = self.self_attention(nx, nx, nx, rope_cache, block_mask)
        out1 = x + self_attention_output
        out2 = out1 + self.ff_module(self.norm2(out1))

        return out2


class Decoder(nn.Module):
    def __init__(
            self,
            num_decoder_layers: int,
            d_model: int,
            num_heads: int,
            num_kv_heads: int,
            vocab_size: int,
            ff_factory: Callable[[], nn.Module],
            norm_factory: Callable[[], nn.Module],
            max_seq_len: int = 4096,
            rope_theta: int = 500_000
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        depth = d_model // num_heads
        self.rope = RoPE(depth, max_seq_len, rope_theta)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads,
                          num_kv_heads, ff_factory, norm_factory)
             for _ in range(num_decoder_layers)]
        )
        self.output_fc = nn.Linear(d_model, vocab_size, bias=False)
        self.final_layer_norm = norm_factory()
        self.d_model = d_model

        # Weight Init.
        self._init_weights()
        # Weight Tying.
        self.output_fc.weight = self.embedding.weight

    def forward(
            self,
            src: torch.Tensor,
            doc_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Фаза суперпозиции.
        if len(src.shape) == 3:
            bs, superposition_len, superposition_bag_size = src.shape

            h = self.embedding(src[..., 0].to(torch.long))
            h_dtype = h.dtype
            h = h.float()

            for i in range(1, superposition_bag_size):
                h = h + self.embedding(src[..., i]).float()

            x = (h / superposition_bag_size).to(h_dtype)

            block_mask = None
            rope_cache = self.rope.cache[:x.size(1)]
            for layer in self.decoder_layers:
                x = layer(x, rope_cache, block_mask)
            x = self.final_layer_norm(x)
            x = self.output_fc(x)
            return x

        # Обычная фаза NTP.
        else:
            x = self.embedding(src.to(torch.long))

            if doc_ids is not None:
                assert doc_ids.shape == src.shape
                doc_ids = doc_ids.to(device=x.device, dtype=torch.long).contiguous()
                position_ids = make_reset_position_ids(doc_ids)
                rope_cache = self.rope.cache[position_ids]

                B, L = doc_ids.size()

                def mask_mod(b, h, q_idx, kv_idx):
                    same_doc = doc_ids[b, q_idx] == doc_ids[b, kv_idx]
                    causal = q_idx >= kv_idx
                    return same_doc & causal

                block_mask = create_block_mask(
                    mask_mod, B=B,H=None, Q_LEN=L, KV_LEN=L, device=doc_ids.device, _compile=True
                )

            else:
                block_mask = None
                rope_cache = self.rope.cache[:x.size(1)]

            for layer in self.decoder_layers:
                x = layer(x, rope_cache, block_mask)

            x = self.final_layer_norm(x)
            x = self.output_fc(x)

            return x

    def _init_weights(self):
        std = self.d_model ** -0.5
        residual_scale = (2 * len(self.decoder_layers)) ** -0.5

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=std)

        for layer in self.decoder_layers:
            layer.self_attention.fc.weight.data.mul_(residual_scale)
            layer.ff_module.w2.weight.data.mul_(residual_scale)
