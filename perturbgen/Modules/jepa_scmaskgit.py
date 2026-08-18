"""Gene-Query JEPA — pretrained MaskGIT encoder wrapper (production encoder).

KEEP THIS FILE. Training and the hyperparameter sweep load this encoder
(encoder_type='scmaskgit'). Time is discarded: pos_embedding(x, 1) only.
Early-exit depth n_encoder_layers is a sweep knob (L in {1..6}).

The tiny CPU encoder lives in jepa.py (tests only).
Honesty metric: val/gene_gap_vs_copy_src > 0.
Index: docs/examples/GENE_QUERY_JEPA.md
"""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn

from scmaskgit.Modules.T_model import scmoscf
from scmaskgit.src.utils import generate_pad, mean_nonpadding_embs


def _load_scmoscf_from_ckpt(encoder_path: str) -> scmoscf:
    """Build scmoscf and load weights (same layout as scmaskgitwrapper)."""
    model = scmoscf(
        tgt_vocab_size=19000,
        d_model=768,
        num_heads=8,
        num_layers=12,
        d_ff=96,
        max_seq_length=4096,
        dropout=0.03,
    )
    pretrained_dict = torch.load(encoder_path, map_location='cpu', weights_only=True)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    nested_prefix = 'transformer.encoder_layers.model.'
    if any(k.startswith(nested_prefix) for k in pretrained_dict):
        corrected_dict = {
            k[len(nested_prefix) :]: v
            for k, v in pretrained_dict.items()
            if k.startswith(nested_prefix)
        }
    else:
        corrected_dict = {
            k.replace('transformer.', ''): v for k, v in pretrained_dict.items()
        }
    model.load_state_dict(corrected_dict, strict=False)
    return model


class SCMaskGITCellEncoder(nn.Module):
    """Encode gene-token IDs with the pretrained MaskGIT / scmaskgit backbone.

    Loads the full 12-layer pretrained body, then runs only the first
    ``n_encoder_layers`` transformer blocks (early exit) and mean-pools.
    Heads / width stay as in the ckpt (8 / 768). Freeze is optional.
    """

    def __init__(
        self,
        encoder_path: str,
        freeze: bool = False,
        n_encoder_layers: int = 3,
    ):
        super().__init__()
        if not encoder_path:
            raise ValueError('encoder_path is required for scmaskgit JEPA encoder')
        self.model = _load_scmoscf_from_ckpt(encoder_path)
        self.d_model = int(self.model.d_model)
        self.vocab_size = int(self.model.tgt_vocab_size)
        total = len(self.model.decoder_block)
        if n_encoder_layers < 1 or n_encoder_layers > total:
            raise ValueError(
                f'n_encoder_layers must be in [1, {total}], got {n_encoder_layers}'
            )
        self.n_encoder_layers = int(n_encoder_layers)
        # Unused deeper blocks never run; keep them frozen always.
        for i, block in enumerate(self.model.decoder_block):
            if i >= self.n_encoder_layers:
                for p in block.parameters():
                    p.requires_grad = False
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        time_step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        del time_step  # encode path uses fixed pos encoding as in MaskGIT src
        src_attention_mask = generate_pad(input_ids)
        x = self.model.token_embedding(input_ids)
        x = self.model.pos_embedding(x, 1)
        for block in self.model.decoder_block[: self.n_encoder_layers]:
            x, _ = block(x=x, tgt_mask=src_attention_mask)
        return {
            'token_embedding': x,
            'cell_embedding': mean_nonpadding_embs(
                embs=x, pad=src_attention_mask
            ),
        }

    def clone_as_ema_target(self) -> 'SCMaskGITCellEncoder':
        """Deep-copy weights into a frozen EMA twin (no second ckpt load)."""
        twin = copy.deepcopy(self)
        for param in twin.parameters():
            param.requires_grad = False
        return twin
