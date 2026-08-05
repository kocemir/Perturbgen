"""Metrics and analysis helpers for JEPA research phases B–F."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def latent_collapse_stats(z: torch.Tensor) -> Dict[str, float]:
    """Diagnostics for representation collapse."""
    z = z.detach().float()
    if z.ndim != 2 or z.size(0) < 2:
        return {
            'std_mean': float(z.std().item()) if z.numel() else 0.0,
            'mean_cosine': 0.0,
            'norm_mean': float(z.norm(dim=-1).mean().item()) if z.numel() else 0.0,
        }
    z_norm = F.normalize(z, dim=-1)
    sim = z_norm @ z_norm.T
    n = sim.size(0)
    # exclude diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    mean_cosine = sim[mask].mean().item()
    return {
        'std_mean': float(z.std(dim=0).mean().item()),
        'mean_cosine': float(mean_cosine),
        'norm_mean': float(z.norm(dim=-1).mean().item()),
    }


def pairwise_latent_mse(
    z_hat: torch.Tensor, z_tgt: torch.Tensor
) -> Dict[str, float]:
    mse = F.mse_loss(z_hat, z_tgt).item()
    cos = F.cosine_similarity(z_hat, z_tgt, dim=-1).mean().item()
    return {
        'mse': float(mse),
        'cosine': float(cos),
        'cosine_loss': float(1.0 - cos),
    }


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
    # Center across batch (standard VICReg).
    z_c = z - z.mean(dim=0)
    std = torch.sqrt(z_c.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(gamma - std).mean()
    n, d = z_c.shape
    cov = (z_c.T @ z_c) / max(n - 1, 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    cov_loss = off_diag / d
    return var_loss, cov_loss


def linear_probe_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> float:
    """Simple logistic-regression probe accuracy (sklearn if available)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(labels)) < 2:
        return float('nan')
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, multi_class='auto'),
    )
    clf.fit(x_train, y_train)
    return float(clf.score(x_test, y_test))


def silhouette_safe(features: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score

    if len(np.unique(labels)) < 2 or features.shape[0] < 3:
        return float('nan')
    return float(silhouette_score(features, labels, metric='euclidean'))


def neighbor_purity(
    features: np.ndarray,
    labels: np.ndarray,
    k: int = 15,
) -> float:
    """Fraction of kNN neighbors sharing the same label."""
    from sklearn.neighbors import NearestNeighbors

    if features.shape[0] <= k:
        return float('nan')
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(features)
    indices = nn.kneighbors(features, return_distance=False)[:, 1:]
    hits = 0
    total = 0
    for i, neigh in enumerate(indices):
        hits += int((labels[neigh] == labels[i]).sum())
        total += k
    return float(hits / max(total, 1))


def compare_representation_quality(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    labels: Dict[str, np.ndarray],
    name_a: str = 'jepa',
    name_b: str = 'masking',
) -> Dict[str, Dict[str, float]]:
    """Phase B: compare two embedding sets on shared label probes."""
    report: Dict[str, Dict[str, float]] = {}
    for label_name, y in labels.items():
        report[label_name] = {
            f'{name_a}_probe_acc': linear_probe_accuracy(emb_a, y),
            f'{name_b}_probe_acc': linear_probe_accuracy(emb_b, y),
            f'{name_a}_silhouette': silhouette_safe(emb_a, y),
            f'{name_b}_silhouette': silhouette_safe(emb_b, y),
            f'{name_a}_nn_purity': neighbor_purity(emb_a, y),
            f'{name_b}_nn_purity': neighbor_purity(emb_b, y),
        }
    return report


def trajectory_baselines(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    z_hat: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    """Phase C: JEPA predictor vs identity and mean baselines."""
    identity = pairwise_latent_mse(z_src, z_tgt)
    # constant predictor: batch mean of z_src
    mean_pred = z_src.mean(dim=0, keepdim=True).expand_as(z_tgt)
    mean_base = pairwise_latent_mse(mean_pred, z_tgt)
    model = pairwise_latent_mse(z_hat, z_tgt)
    return {
        'identity': identity,
        'mean_src': mean_base,
        'jepa': model,
    }


def simulate_latent_perturbation(
    z_src: torch.Tensor,
    direction: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Phase E: simple additive intervention in latent space."""
    direction = F.normalize(direction.float(), dim=-1)
    return z_src + scale * direction.unsqueeze(0).expand_as(z_src)


def package_comparison_summary(
    phase_b: Optional[Dict] = None,
    phase_c: Optional[Dict] = None,
    phase_d: Optional[Dict] = None,
    phase_e: Optional[Dict] = None,
) -> Dict:
    """Phase F: gather replacement comparison package."""
    return {
        'representation': phase_b or {},
        'trajectory': phase_c or {},
        'generation': phase_d or {},
        'applications': phase_e or {},
    }
