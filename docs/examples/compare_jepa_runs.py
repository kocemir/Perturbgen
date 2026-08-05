#!/usr/bin/env python3
"""Numeric + visual comparison of JEPA Phase A Lightning metrics CSVs.

Created 2026-08-05 for freeze ablation; defaults include kept sod2 runs
(A unfz 1120, B fz 1128, C cell 1622 when metrics exist).
See ``docs/examples/JEPA_README.md``.

Examples:
  python docs/examples/compare_jepa_runs.py
  python docs/examples/compare_jepa_runs.py \\
    --runs unfz=/path/to/1120/.../metrics.csv \\
           fz=/path/to/1128/.../metrics.csv \\
    --out-dir /tmp/jepa_compare
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

LOG_ROOT = Path("/mnt/sod2-project/csb4/stuke1/perturbgen/logs")

DEFAULT_RUNS = {
    "unfz_1120": LOG_ROOT
    / "20260804_1120_cellgen"
    / "version_0"
    / "metrics.csv",
    "fz_1128": LOG_ROOT
    / "20260805_1128_cellgen"
    / "version_0"
    / "metrics.csv",
    "cell_1622": LOG_ROOT
    / "20260805_1622_cellgen"
    / "version_0"
    / "metrics.csv",
}

SUMMARY_COLS = (
    "val/jepa_pred_loss",
    "val/jepa_loss",
    "val/latent_cosine",
    "val/beats_identity",
    "val/collapse_mean_cosine",
    "val/collapse_std_mean",
    "val/baseline_identity_cosine_loss",
    "val/baseline_jepa_cosine_loss",
    "val/jepa_loss_t1",
    "val/jepa_loss_t2",
    "val/jepa_loss_t3",
    "train/jepa_pred_loss_epoch",
    "train/latent_cosine",
    "train/collapse_mean_cosine",
)

PLOT_SPECS: List[Tuple[str, List[str]]] = [
    ("JEPA pred loss (1-cos)", ["train/jepa_pred_loss_epoch", "val/jepa_pred_loss"]),
    ("Latent cosine", ["train/latent_cosine", "val/latent_cosine"]),
    (
        "Val baseline cosine-loss",
        ["val/baseline_identity_cosine_loss", "val/baseline_jepa_cosine_loss"],
    ),
    ("Val beats identity", ["val/beats_identity"]),
    ("Val collapse mean cosine", ["val/collapse_mean_cosine"]),
    (
        "Val per-time pred loss",
        ["val/jepa_loss_t1", "val/jepa_loss_t2", "val/jepa_loss_t3"],
    ),
]


def parse_runs(items: Optional[List[str]]) -> Dict[str, Path]:
    if not items:
        return dict(DEFAULT_RUNS)
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--runs entry must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        out[name] = Path(path)
    return out


def load_epoch_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        raise ValueError(f"No epoch column in {csv_path}")
    cols = [c for c in SUMMARY_COLS if c in df.columns]
    # also keep total train loss if present
    for extra in ("train/jepa_loss_epoch", "val/jepa_loss"):
        if extra in df.columns and extra not in cols:
            cols.append(extra)
    return df.groupby("epoch", as_index=True)[cols].last().sort_index()


def summarize_run(name: str, by: pd.DataFrame) -> pd.Series:
    key = (
        "val/jepa_pred_loss"
        if "val/jepa_pred_loss" in by
        else "val/jepa_loss"
    )
    best_ep = int(by[key].idxmin())
    last_ep = int(by.index.max())
    first_ep = int(by.index.min())

    def get(ep: int, col: str) -> Optional[float]:
        if col not in by.columns:
            return None
        return float(by.loc[ep, col])

    rows = {
        "run": name,
        "epochs": f"{first_ep}-{last_ep}",
        "best_epoch": best_ep,
        "best_val_pred_loss": get(best_ep, "val/jepa_pred_loss"),
        "best_val_latent_cosine": get(best_ep, "val/latent_cosine"),
        "best_val_beats_identity": get(best_ep, "val/beats_identity"),
        "best_val_collapse_mean_cos": get(best_ep, "val/collapse_mean_cosine"),
        "best_identity_cos_loss": get(best_ep, "val/baseline_identity_cosine_loss"),
        "best_jepa_cos_loss": get(best_ep, "val/baseline_jepa_cosine_loss"),
        "best_val_t1": get(best_ep, "val/jepa_loss_t1"),
        "best_val_t2": get(best_ep, "val/jepa_loss_t2"),
        "best_val_t3": get(best_ep, "val/jepa_loss_t3"),
        "epoch0_val_pred_loss": get(first_ep, "val/jepa_pred_loss"),
        "epoch0_val_latent_cosine": get(first_ep, "val/latent_cosine"),
        "epoch0_collapse_mean_cos": get(first_ep, "val/collapse_mean_cosine"),
        "final_val_pred_loss": get(last_ep, "val/jepa_pred_loss"),
        "final_val_latent_cosine": get(last_ep, "val/latent_cosine"),
        "final_collapse_mean_cos": get(last_ep, "val/collapse_mean_cosine"),
    }
    return pd.Series(rows)


def print_numeric(tables: Dict[str, pd.DataFrame], out_csv: Path) -> pd.DataFrame:
    summary = pd.DataFrame([summarize_run(n, t) for n, t in tables.items()])
    print("\n=== Numeric summary (best = min val/jepa_pred_loss) ===")
    with pd.option_context("display.max_columns", 30, "display.width", 140):
        print(summary.to_string(index=False))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    print("\n=== Snapshot: epoch 0 / best / last ===")
    for name, by in tables.items():
        key = (
            "val/jepa_pred_loss"
            if "val/jepa_pred_loss" in by
            else "val/jepa_loss"
        )
        best_ep = int(by[key].idxmin())
        for label, ep in (
            ("epoch0", int(by.index.min())),
            ("best", best_ep),
            ("last", int(by.index.max())),
        ):
            print(f"\n[{name}] {label} @ epoch {ep}")
            for c in (
                "val/jepa_pred_loss",
                "val/latent_cosine",
                "val/beats_identity",
                "val/collapse_mean_cosine",
                "val/baseline_identity_cosine_loss",
                "val/baseline_jepa_cosine_loss",
            ):
                if c in by.columns:
                    print(f"  {c}: {by.loc[ep, c]:.6g}")
    return summary


def plot_compare(tables: Dict[str, pd.DataFrame], out_png: Path) -> None:
    n = len(PLOT_SPECS)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows))
    axes = axes.flatten()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, (title, cols) in zip(axes, PLOT_SPECS):
        for run_i, (name, by) in enumerate(tables.items()):
            color = colors[run_i % len(colors)]
            for col_i, col in enumerate(cols):
                if col not in by.columns:
                    continue
                style = "-" if col_i == 0 or len(cols) == 1 else "--"
                label = f"{name}:{col.split('/')[-1]}"
                ax.plot(
                    by.index,
                    by[col],
                    style,
                    color=color,
                    alpha=0.85 if col_i == 0 else 0.7,
                    label=label,
                    ms=3,
                    marker="o",
                )
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    for ax in axes[len(PLOT_SPECS) :]:
        ax.axis("off")

    fig.suptitle("JEPA Phase A: unfz vs fz comparison", fontsize=13)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_png}")
    plt.close(fig)


def plot_single(name: str, by: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    ax = axes[0, 0]
    if "train/jepa_pred_loss_epoch" in by:
        ax.plot(by.index, by["train/jepa_pred_loss_epoch"], "o-", label="train", ms=3)
    if "val/jepa_pred_loss" in by:
        ax.plot(by.index, by["val/jepa_pred_loss"], "o-", label="val", ms=3)
    ax.set_title("Pred loss (1-cos)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if "train/latent_cosine" in by:
        ax.plot(by.index, by["train/latent_cosine"], "o-", label="train", ms=3)
    if "val/latent_cosine" in by:
        ax.plot(by.index, by["val/latent_cosine"], "o-", label="val", ms=3)
    ax.set_title("Latent cosine")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if "val/baseline_identity_cosine_loss" in by:
        ax.plot(
            by.index,
            by["val/baseline_identity_cosine_loss"],
            "o-",
            label="identity",
            ms=3,
        )
    if "val/baseline_jepa_cosine_loss" in by:
        ax.plot(
            by.index, by["val/baseline_jepa_cosine_loss"], "o-", label="jepa", ms=3
        )
    if "val/beats_identity" in by:
        ax2 = ax.twinx()
        ax2.plot(
            by.index,
            by["val/beats_identity"],
            "s--",
            color="C2",
            label="beats_id",
            ms=3,
        )
        ax2.set_ylabel("beats_identity", color="C2")
        ax2.set_ylim(0, 1.05)
    ax.set_title("Baseline cos-loss / beats identity")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "val/collapse_mean_cosine" in by:
        ax.plot(
            by.index, by["val/collapse_mean_cosine"], "o-", label="collapse_cos", ms=3
        )
    if "val/collapse_std_mean" in by:
        ax2 = ax.twinx()
        ax2.plot(
            by.index,
            by["val/collapse_std_mean"],
            "s--",
            color="C1",
            label="std_mean",
            ms=3,
        )
        ax2.set_ylabel("std_mean", color="C1")
    ax.set_title("Collapse")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"JEPA curves: {name}")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_png}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="name=metrics.csv pairs (default: kept runs 1120/1128/1622)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "jepa_compare_kept_runs",
        help="Output directory for CSV/PNGs",
    )
    args = parser.parse_args()

    runs = parse_runs(args.runs)
    tables: Dict[str, pd.DataFrame] = {}
    for name, path in runs.items():
        if not path.is_file():
            print(f"SKIP {name}: missing {path}")
            continue
        print(f"Loading {name}: {path}")
        tables[name] = load_epoch_table(path)
    if not tables:
        raise FileNotFoundError("No metrics.csv found for any requested run")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print_numeric(tables, out_dir / "summary.csv")
    plot_compare(tables, out_dir / "compare_curves.png")
    for name, by in tables.items():
        plot_single(name, by, out_dir / f"{name}_curves.png")

    print("\nDone.")
    print(f"Open: {out_dir / 'compare_curves.png'}")
    print(f"Open: {out_dir / 'summary.csv'}")
    print(
        "\nTensorBoard (both runs):\n"
        "  tensorboard --logdir_spec "
        "unfz_1120:/mnt/sod2-project/csb4/stuke1/perturbgen/logs/20260804_1120_cellgen,"
        "fz_1128:/mnt/sod2-project/csb4/stuke1/perturbgen/logs/20260805_1128_cellgen "
        "--port 6007 --bind_all --load_fast=false"
    )


if __name__ == "__main__":
    main()
