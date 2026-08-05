"""Token-ID remaps for JEPA (global pretrain IDs vs remapped HVG / row IDs)."""

from __future__ import annotations

import pickle
from typing import Dict, Optional, Tuple

import torch


def load_tokenid_to_rowid(path: str) -> Dict[int, int]:
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def build_lookup_tables(
    tokenid_to_rowid: Dict[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (global_to_local, local_to_global) LongTensors.

    Unmapped entries stay 0 (pad). Inverse is unique for LPS HVG maps.
    """
    if not tokenid_to_rowid:
        empty = torch.zeros(1, dtype=torch.long)
        return empty, empty
    max_global = max(tokenid_to_rowid.keys())
    max_local = max(tokenid_to_rowid.values())
    global_to_local = torch.zeros(max_global + 1, dtype=torch.long)
    local_to_global = torch.zeros(max_local + 1, dtype=torch.long)
    for g, loc in tokenid_to_rowid.items():
        global_to_local[g] = loc
        local_to_global[loc] = g
    return global_to_local, local_to_global


def apply_id_lookup(
    input_ids: torch.Tensor,
    table: torch.Tensor,
) -> torch.Tensor:
    """Map IDs with a 1D lookup table; IDs outside table range become 0."""
    table = table.to(device=input_ids.device)
    flat = input_ids.long().reshape(-1)
    out = torch.zeros_like(flat)
    in_range = (flat >= 0) & (flat < table.numel())
    out[in_range] = table[flat[in_range]]
    return out.view_as(input_ids)


def maybe_load_maps(
    tokenid_to_rowid_path: Optional[str],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not tokenid_to_rowid_path:
        return None, None
    mapping = load_tokenid_to_rowid(tokenid_to_rowid_path)
    return build_lookup_tables(mapping)
