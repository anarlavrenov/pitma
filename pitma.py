import torch
from torch import nn
from typing import Tuple, Optional, Callable
from .layers import *


class DecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            num_kv_heads: int,
            attn_dropout_rate: float,
            ff_factory: Callable[[], nn.Module],
            norm_factory: Callable[[], nn.Module],
            is_padding: bool
    ):
        super().__init__()

        self.self_attention = GKVRoPEAttention(
            d_model, num_heads, num_kv_heads, attn_dropout_rate, is_padding
        )

        self.ff_module = ff_factory()
        self.norm1 = norm_factory()
        self.norm2 = norm_factory()

    def forward(
            self,
            x: torch.Tensor,
            rope_cache: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        nx = self.norm1(x)
        self_attention_output = self.self_attention(
            nx, nx, nx, rope_cache, tgt_mask, tgt_key_padding_mask
        )
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
            attn_dropout_rate: float,
            ff_factory: Callable[[], nn.Module],
            norm_factory: Callable[[], nn.Module],
            is_padding: bool,
            pad_token_id: int,
            max_seq_len: int = 4096
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        depth = d_model // num_heads
        self.rope = RoPE(depth, max_seq_len)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, num_kv_heads, attn_dropout_rate,
                          ff_factory, norm_factory, is_padding)
             for _ in range(num_decoder_layers)]
        )
        self.output_fc = nn.Linear(d_model, vocab_size, bias=False)
        self.final_layer_norm = norm_factory()
        self.d_model = d_model

    def forward(
            self,
            tgt: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embedding(tgt.to(torch.long))

        rope_cache = self.rope.cache[:x.size(1)]

        for layer in self.decoder_layers:
            x = layer(x, rope_cache, tgt_mask, tgt_key_padding_mask)

        x = self.final_layer_norm(x)
        x = self.output_fc(x)

        return x
