"""Gene-Query JEPA — lightweight CellEncoder (CPU tests / no pretrained ckpt).

THIS IS NOT THE TRAINING MODEL.
The only JEPA we train is Gene-Query JEPA:
  perturbgen/Modules/gene_query_jepa.py
  docs/examples/train_gene_query_jepa.py       (--data toy|full)
  docs/examples/run_gene_query_toy_sweep.sh    (hyperparameter search)

This file exists so tests can run on CPU without loading the 768-d MaskGIT
encoder. Production / toy / sweep always use encoder_type='scmaskgit'
(jepa_scmaskgit.py).

Honesty metric (everywhere): val/gene_gap_vs_copy_src must be > 0.
Index: docs/examples/GENE_QUERY_JEPA.md
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def generate_pad(input_ids: torch.Tensor) -> torch.Tensor:
    return input_ids == 0


def mean_pool_tokens(
    embs: torch.Tensor,
    input_ids: torch.Tensor,
    exclude_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """Mean-pool token embeddings, excluding pad/mask (and optional special IDs)."""
    if exclude_ids is None:
        exclude_ids = [0, 1]  # <pad>, <mask>
    keep = torch.ones_like(input_ids, dtype=torch.bool)
    for token_id in exclude_ids:
        keep = keep & (input_ids != token_id)
    lengths = keep.sum(dim=1).clamp(min=1).unsqueeze(1).float()
    masked = embs.masked_fill(~keep.unsqueeze(-1), 0.0)
    return masked.sum(dim=1) / lengths


def _sinusoidal(length: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class TimePosSinEncoding(nn.Module):
    """Sequence + dummy time sinusoidal encoding for the tiny CellEncoder."""

    def __init__(self, d_model: int, length: int, n_time_steps: int):
        super().__init__()
        self.register_buffer('time_pe', _sinusoidal(n_time_steps + 1, d_model))
        self.register_buffer('pos_pe', _sinusoidal(length, d_model))

    def forward(self, x: torch.Tensor, tgt_time_step: int = 0) -> torch.Tensor:
        time_pe = self.time_pe[:, int(tgt_time_step)].unsqueeze(0)
        pos_pe = self.pos_pe[:, : x.size(1)]
        return x + time_pe + pos_pe


class CellEncoder(nn.Module):
    """Tiny transformer: gene-token sequence -> token embeddings + pooled cell.

    Used only when GeneQueryJEPA(encoder_type='cell'), i.e. unit tests.
    Toy training and hyperparameter search use SCMaskGITCellEncoder instead.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        d_ff: int = 1024,
        max_seq_length: int = 2048,
        n_time_steps: int = 4,
        dropout: float = 0.0,
        pad_token: int = 0,
        pos_encoding_mode: str = 'time_pos_sin',
    ):
        super().__init__()
        del pos_encoding_mode
        self.d_model = d_model
        self.pad_token = pad_token
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token
        )
        self.pos_embedding = TimePosSinEncoding(
            d_model=d_model,
            length=max_seq_length,
            n_time_steps=n_time_steps,
        )
        layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            activation='gelu',
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        nn.init.xavier_uniform_(self.token_embedding.weight)
        with torch.no_grad():
            self.token_embedding.weight[pad_token].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        time_step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        pad_mask = generate_pad(input_ids)
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_embedding(x, tgt_time_step=time_step)
        token_emb = self.transformer(x, src_key_padding_mask=pad_mask)
        cell_emb = mean_pool_tokens(token_emb, input_ids)
        return {
            'token_embedding': token_emb,
            'cell_embedding': cell_emb,
        }
