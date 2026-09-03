"""Toy roster for Gene-Query JEPA: first N cells per type, train/val disjoint.

``--data toy --split false``: take train then val from all cells of each type.
``--data toy --split true``: take train cells from the pickle train pool and
val cells from the pickle val pool (90-10). A type is kept only if both
pools have enough cells.

Honesty metric: val/gene_gap_vs_copy_src > 0.
Index: docs/examples/GENE_QUERY_JEPA.md
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

RosterReport = List[Tuple[str, int, int, str]]


def group_indices_by_type(
    cell_types: Sequence[str],
    pool: Sequence[int],
) -> Dict[str, List[int]]:
    """Preserve ``pool`` order; skip out-of-range indices."""
    n = len(cell_types)
    buckets: Dict[str, List[int]] = {}
    for raw in pool:
        idx = int(raw)
        if idx < 0 or idx >= n:
            continue
        buckets.setdefault(str(cell_types[idx]), []).append(idx)
    return buckets


def pick_toy_train_val(
    cell_types: Sequence[str],
    *,
    train_per_type: int,
    val_per_type: int,
    train_pool: Optional[Sequence[int]] = None,
    val_pool: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int], RosterReport]:
    """First ``train_per_type`` / ``val_per_type`` cells of every eligible type.

    If both pools are ``None``, split each type's dataset-order list:
    ``[:train]`` then ``[train:train+val]``. If both pools are set, take
    the first ``train_per_type`` from ``train_pool`` and the first
    ``val_per_type`` from ``val_pool`` (types need enough in both).
    """
    if train_per_type < 1:
        raise ValueError('train_per_type must be >= 1')
    if val_per_type < 1:
        raise ValueError('val_per_type must be >= 1')
    if (train_pool is None) != (val_pool is None):
        raise ValueError('train_pool and val_pool must both be set or both None')

    types = sorted({str(t) for t in cell_types})
    train_out: List[int] = []
    val_out: List[int] = []
    report: RosterReport = []

    if train_pool is None:
        buckets = group_indices_by_type(cell_types, range(len(cell_types)))
        need = train_per_type + val_per_type
        for cell_type in types:
            cells = buckets.get(cell_type, [])
            if len(cells) < need:
                report.append((cell_type, len(cells), len(cells), f'SKIP (need {need})'))
                continue
            train_out.extend(cells[:train_per_type])
            val_out.extend(cells[train_per_type : train_per_type + val_per_type])
            report.append((cell_type, len(cells), len(cells), 'used'))
    else:
        train_buckets = group_indices_by_type(cell_types, train_pool)
        val_buckets = group_indices_by_type(cell_types, val_pool or ())
        for cell_type in types:
            train_cells = train_buckets.get(cell_type, [])
            val_cells = val_buckets.get(cell_type, [])
            if len(train_cells) < train_per_type or len(val_cells) < val_per_type:
                report.append(
                    (cell_type, len(train_cells), len(val_cells), 'SKIP (too few)')
                )
                continue
            train_out.extend(train_cells[:train_per_type])
            val_out.extend(val_cells[:val_per_type])
            report.append(
                (cell_type, len(train_cells), len(val_cells), 'used')
            )

    overlap = set(train_out).intersection(val_out)
    if overlap:
        raise RuntimeError(f'train/val roster overlap ({len(overlap)} cells)')
    return train_out, val_out, report
