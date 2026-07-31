"""Cell-trajectory JEPA backbone for PerturbGen replacement research.

Kept free of scmaskgit / PerturbGen imports so JEPA can train as a thin parallel path.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def generate_pad(input_ids: torch.Tensor) -> torch.Tensor:
    return input_ids == 0


def modify_ckpt_state_dict(checkpoint, replace_str: str):
    if 'module' in checkpoint.keys():
        state_dict = checkpoint['module']
    elif 'state_dict' in checkpoint.keys():
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(replace_str):
            k = k.replace(replace_str, '', 1)
        k = k.replace('_orig_mod.', '')
        new_state_dict[k] = v
    return new_state_dict


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
    """Minimal time + position sinusoidal encoding for JEPA cell sequences."""

    def __init__(self, d_model: int, length: int, n_time_steps: int):
        super().__init__()
        self.register_buffer('time_pe', _sinusoidal(n_time_steps + 1, d_model))
        self.register_buffer('pos_pe', _sinusoidal(length, d_model))

    def forward(self, x: torch.Tensor, tgt_time_step: int = 0) -> torch.Tensor:
        time_pe = self.time_pe[:, int(tgt_time_step)].unsqueeze(0)
        pos_pe = self.pos_pe[:, : x.size(1)]
        return x + time_pe + pos_pe


class CellEncoder(nn.Module):
    """Encode a gene-token cell sequence to a single cell embedding."""

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
        del pos_encoding_mode  # reserved for API compatibility with train.py
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


class TimeConditionedPredictor(nn.Module):
    """Predict target cell embedding from context embedding + target time."""

    def __init__(
        self,
        d_model: int,
        n_time_steps: int,
        hidden_multiplier: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_embedding = nn.Embedding(n_time_steps + 1, d_model)
        hidden = d_model * hidden_multiplier
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, z_context: torch.Tensor, time_step: int) -> torch.Tensor:
        t = torch.full(
            (z_context.size(0),),
            int(time_step),
            device=z_context.device,
            dtype=torch.long,
        )
        t_emb = self.time_embedding(t)
        return self.mlp(torch.cat([z_context, t_emb], dim=-1))


class CellTrajectoryJEPA(nn.Module):
    """JEPA: predict future cell latents from source cell latents.

    Context encoder is trainable. Target encoder is an EMA copy (stop-grad).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        d_ff: int = 1024,
        max_seq_length: int = 2048,
        n_total_tps: int = 3,
        pred_tps: Optional[List[int]] = None,
        dropout: float = 0.0,
        ema_decay: float = 0.996,
        pad_token: int = 0,
        pos_encoding_mode: Literal[
            'time_pos_sin', 'comb_sin', 'sin_learnt', 'time_pos_learnt'
        ] = 'time_pos_sin',
        normalize_latents: bool = True,
    ):
        super().__init__()
        self.pred_tps = pred_tps if pred_tps is not None else [1, 2, 3]
        self.n_total_tps = n_total_tps
        self.ema_decay = ema_decay
        self.normalize_latents = normalize_latents
        self.d_model = d_model

        encoder_kwargs = dict(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            n_time_steps=n_total_tps + 1,
            dropout=dropout,
            pad_token=pad_token,
            pos_encoding_mode=pos_encoding_mode,
        )
        self.context_encoder = CellEncoder(**encoder_kwargs)
        self.target_encoder = CellEncoder(**encoder_kwargs)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = TimeConditionedPredictor(
            d_model=d_model,
            n_time_steps=n_total_tps,
            dropout=dropout,
        )

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        for online, target in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target.data.mul_(self.ema_decay).add_(
                online.data, alpha=1.0 - self.ema_decay
            )

    def _maybe_normalize(self, z: torch.Tensor) -> torch.Tensor:
        if self.normalize_latents:
            return F.normalize(z, dim=-1)
        return z

    def encode_context(
        self, input_ids: torch.Tensor, time_step: int = 0
    ) -> Dict[str, torch.Tensor]:
        out = self.context_encoder(input_ids, time_step=time_step)
        out['cell_embedding'] = self._maybe_normalize(out['cell_embedding'])
        return out

    @torch.no_grad()
    def encode_target(
        self, input_ids: torch.Tensor, time_step: int
    ) -> Dict[str, torch.Tensor]:
        was_training = self.target_encoder.training
        self.target_encoder.eval()
        out = self.target_encoder(input_ids, time_step=time_step)
        out['cell_embedding'] = self._maybe_normalize(out['cell_embedding'])
        if was_training:
            self.target_encoder.train()
        return out

    def forward(
        self,
        src_input_ids: torch.Tensor,
        tgt_input_id_dict: Dict[str, torch.Tensor],
        pred_tps: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Predict each requested target time from the source cell embedding."""
        times = pred_tps if pred_tps is not None else self.pred_tps
        z_src = self.encode_context(src_input_ids, time_step=0)['cell_embedding']
        outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for t in times:
            key = f'tgt_input_ids_t{t}'
            if key not in tgt_input_id_dict:
                continue
            z_tgt = self.encode_target(tgt_input_id_dict[key], time_step=t)[
                'cell_embedding'
            ]
            z_hat = self.predictor(z_src, time_step=t)
            if self.normalize_latents:
                z_hat = F.normalize(z_hat, dim=-1)
            outputs[t] = {
                'z_src': z_src,
                'z_tgt': z_tgt,
                'z_hat': z_hat,
            }
        return outputs

    def load_token_embedding_from_masking_ckpt(self, ckpt_path: str) -> List[str]:
        """Initialize token embeddings from a PerturbGen masking checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state = modify_ckpt_state_dict(checkpoint, 'transformer.')
        loaded = []
        weight_key = 'token_embedding.weight'
        if weight_key in state:
            src = state[weight_key]
            dst = self.context_encoder.token_embedding.weight
            n = min(src.shape[0], dst.shape[0])
            d = min(src.shape[1], dst.shape[1])
            with torch.no_grad():
                dst[:n, :d].copy_(src[:n, :d])
                self.target_encoder.token_embedding.weight[:n, :d].copy_(
                    src[:n, :d]
                )
            loaded.append(weight_key)
        return loaded


class JEPACountDecoder(nn.Module):
    """Phase D: map JEPA cell latents to gene expression (counts)."""

    def __init__(
        self,
        jepa: CellTrajectoryJEPA,
        n_genes: int,
        d_model: int,
        dropout: float = 0.0,
        freeze_jepa: bool = True,
    ):
        super().__init__()
        self.jepa = jepa
        if freeze_jepa:
            for param in self.jepa.parameters():
                param.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_genes),
        )
        self.n_genes = n_genes

    def forward(
        self,
        src_input_ids: torch.Tensor,
        tgt_input_id_dict: Dict[str, torch.Tensor],
        pred_tps: Optional[List[int]] = None,
        use_predicted_latent: bool = True,
    ) -> Dict[int, torch.Tensor]:
        times = pred_tps if pred_tps is not None else self.jepa.pred_tps
        with torch.set_grad_enabled(any(p.requires_grad for p in self.jepa.parameters())):
            jepa_out = self.jepa(
                src_input_ids=src_input_ids,
                tgt_input_id_dict=tgt_input_id_dict,
                pred_tps=times,
            )
        counts = {}
        for t, out in jepa_out.items():
            z = out['z_hat'] if use_predicted_latent else out['z_tgt']
            counts[t] = F.relu(self.head(z))
        return counts
