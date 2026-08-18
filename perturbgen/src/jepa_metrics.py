"""Gene-Query JEPA — VICReg variance/covariance (optional anti-collapse).

Used by GeneQueryJEPATrainer when vicreg_var_coeff / vicreg_cov_coeff > 0.
The hyperparameter sweep turns VICReg on/off; leave both coeffs at 0 to disable.

Honesty metric is NOT here: it is val/gene_gap_vs_copy_src in the trainer.
Index: docs/examples/GENE_QUERY_JEPA.md
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def vicreg_var_cov(
    z: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance + covariance (Bardes et al.), no invariance term.

    Variance: hinge so each dim has std >= gamma across the batch.
    Covariance: penalize off-diagonal entries of the batch covariance.
    """
    if z.ndim != 2 or z.size(0) < 2:
        zero = z.sum() * 0.0
        return zero, zero
    z_c = z - z.mean(dim=0)
    std = torch.sqrt(z_c.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(gamma - std).mean()
    n, d = z_c.shape
    cov = (z_c.T @ z_c) / max(n - 1, 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    cov_loss = off_diag / d
    return var_loss, cov_loss
