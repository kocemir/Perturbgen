#!/usr/bin/env python3
"""Plot JEPA training curves from Lightning CSV metrics (sod2 logs).

Example:
  python docs/examples/plot_jepa_curves.py
  python docs/examples/plot_jepa_curves.py \\
    --csv /mnt/sod2-project/csb4/stuke1/perturbgen/logs/20260731_1559_cellgen/version_0/metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LOG_ROOT = Path("/mnt/sod2-project/csb4/stuke1/perturbgen/logs")
DEFAULT_CSV = LOG_ROOT / "20260731_1559_cellgen" / "version_0" / "metrics.csv"

METRIC_COLS = (
    "train/jepa_loss_epoch",
    "val/jepa_loss",
    "train/latent_cosine",
    "val/latent_cosine",
    "val/baseline_identity_mse",
    "val/baseline_jepa_mse",
    "val/collapse_mean_cosine",
    "val/collapse_std_mean",
)


def resolve_csv(csv_path: Path) -> Path:
    csv_path = Path(csv_path)
    if csv_path.is_file():
        return csv_path
    candidates = sorted(
        LOG_ROOT.glob("*/version_*/metrics.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No metrics.csv under {LOG_ROOT}")
    return candidates[-1]


def load_epoch_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        raise ValueError(f"No 'epoch' column in {csv_path}")
    cols = [c for c in METRIC_COLS if c in df.columns]
    if not cols:
        raise ValueError(f"No JEPA metric columns in {csv_path}: {list(df.columns)}")
    return df.groupby("epoch", as_index=True)[cols].last()


def plot_curves(by: pd.DataFrame, title: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    ax = axes[0, 0]
    if "train/jepa_loss_epoch" in by:
        ax.plot(by.index, by["train/jepa_loss_epoch"], "o-", label="train", ms=4)
    if "val/jepa_loss" in by:
        ax.plot(by.index, by["val/jepa_loss"], "o-", label="val", ms=4)
    ax.set_title("JEPA loss")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if "train/latent_cosine" in by:
        ax.plot(by.index, by["train/latent_cosine"], "o-", label="train", ms=4)
    if "val/latent_cosine" in by:
        ax.plot(by.index, by["val/latent_cosine"], "o-", label="val", ms=4)
    ax.set_title("Latent cosine")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if "val/baseline_identity_mse" in by:
        ax.plot(
            by.index, by["val/baseline_identity_mse"], "o-", label="identity", ms=4
        )
    if "val/baseline_jepa_mse" in by:
        ax.plot(by.index, by["val/baseline_jepa_mse"], "o-", label="jepa", ms=4)
    ax.set_title("Val baseline MSE")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "val/collapse_mean_cosine" in by:
        ax.plot(
            by.index,
            by["val/collapse_mean_cosine"],
            "o-",
            label="mean_cosine",
            ms=4,
        )
    if "val/collapse_std_mean" in by:
        ax2 = ax.twinx()
        ax2.plot(
            by.index,
            by["val/collapse_std_mean"],
            "s--",
            color="C1",
            label="std_mean",
            ms=4,
        )
        ax2.set_ylabel("std_mean", color="C1")
    ax.set_title("Val collapse")
    ax.set_xlabel("epoch")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to Lightning metrics.csv (default: last JEPA run, or newest under sod2 logs)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: docs/examples/jepa_curves.png)",
    )
    args = parser.parse_args()

    csv_path = resolve_csv(args.csv)
    print(f"Using {csv_path}")
    by = load_epoch_metrics(csv_path)
    out = args.out or (Path(__file__).resolve().parent / "jepa_curves.png")
    plot_curves(by, title=f"JEPA curves\n{csv_path}", out=out)


if __name__ == "__main__":
    main()
