#!/usr/bin/env python3
"""Plot epoch curves for the kept gene-query JEPA runs (see docs/examples/KEEP.md)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ATLAS = Path(
    "/mnt/sod2-project/csb4/stuke1/perturbgen/T_perturb/res/jepa_gene_query_full_atlas"
)
SRC90 = Path(
    "/mnt/sod2-project/csb4/stuke1/perturbgen/T_perturb/res/jepa_gene_query_90m_src"
)
OUT_ROOT = Path(__file__).resolve().parent

# Only the four runs retained after cleanup.
RUNS = [
    {
        "name": "frozen_mse_contr_maskce_qsrcpos_100ep_bs16_20260926",
        "title": (
            "resting → 90m/6h/10h  ·  frozen enc L3  ·  "
            "MSE + λ_contr=0.1 + λ_maskCE=0.1  ·  notebook 08 ckpt"
        ),
        "metrics": ATLAS
        / (
            "fzT_encL3_predL3_q128_mixed_lg1_lc0_contr0.1_vic0_glmse_bs16_ep100_"
            "lr0.0001_seed0_splitT_poolcls_tp1-2-3_max512_mce0.1_ms0.2_qsrcpos_20260926_232131"
        )
        / "logs"
        / "version_0"
        / "metrics.csv",
        "mark_epoch": None,
        "gene_loss_in_obj": "mse",
        "time_labels": {"1": "90m", "2": "6h", "3": "10h"},
    },
    {
        "name": "source_blind_90m_splitT_20260930",
        "title": (
            "90m → 6h/10h  ·  source_blind qsb96 λ_absent=0.25  ·  "
            "KO JEPA …143041"
        ),
        "metrics": SRC90
        / (
            "fzT_encL3_predL3_q128_mixed_lg1_lc0_contr0.1_vic0_glmse_bs16_ep50_"
            "lr5e-05_seed0_splitT_poolcls_tp1-2_max512_src90m_mce0.1_ms0.2_qsrcpos_"
            "qsb96_nwct_la0.25_20261001_143041"
        )
        / "logs"
        / "version_0"
        / "metrics.csv",
        "mark_epoch": 24,
        "gene_loss_in_obj": "mse",
        "time_labels": {"1": "6h", "2": "10h"},
    },
    {
        "name": "source_blind_normal_bs32_ep100_20261008",
        "title": (
            "normal → 90m/6h/10h  ·  source_blind qsb96 λ_absent=0.25  ·  "
            "bs32  ·  100 ep"
        ),
        "metrics": ATLAS
        / (
            "fzT_encL3_predL3_q128_mixed_lg1_lc0_contr0.1_vic0_glmse_bs32_ep100_"
            "lr0.0001_seed0_splitT_poolcls_tp1-2-3_max512_mce0.1_ms0.2_qsrcpos_"
            "qsb96_nwct_la0.25_20261008_101504"
        )
        / "logs"
        / "version_0"
        / "metrics.csv",
        "mark_epoch": None,
        "gene_loss_in_obj": "mse",
        "time_labels": {"1": "90m", "2": "6h", "3": "10h"},
    },
    {
        "name": "source_blind_90m_qsb64_tc48_contr0_ep30_20261012",
        "title": (
            "90m → 6h/10h  ·  source_blind qsb64 + 48 top-common  ·  "
            "no InfoNCE  ·  current KO JEPA"
        ),
        "metrics": SRC90
        / (
            "fzT_encL3_predL3_q128_mixed_lg1_lc0_contr0_vic0_glmse_bs32_ep30_"
            "lr0.0001_seed0_splitT_poolcls_tp1-2_max512_src90m_mce0.1_ms0.2_"
            "qsrcpos_qsb64_nwct_tc48_la0.25_20261012_160033"
        )
        / "logs"
        / "version_0"
        / "metrics.csv",
        "mark_epoch": None,
        "gene_loss_in_obj": "mse",
        "time_labels": {"1": "6h", "2": "10h"},
    },
]


def epoch_series(raw: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in raw.columns:
        return pd.DataFrame(columns=["epoch", col])
    ep = (
        raw[raw[col].notna()][["epoch", col]]
        .drop_duplicates("epoch")
        .sort_values("epoch")
    )
    return ep


def plot_train_val(raw: pd.DataFrame, out_dir: Path, title: str) -> None:
    pairs = [
        ("train/mask_ce_loss", "val/mask_ce_loss", "mask-CE"),
        ("train/gene_mse", "val/gene_mse", "gene MSE"),
        ("train/contrastive_loss", "val/contrastive_loss", "InfoNCE"),
        ("train/absent_loss", "val/absent_loss", "absent MSE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.ravel()
    for ax, (tr, va, lab) in zip(axes, pairs):
        a = epoch_series(raw, tr)
        b = epoch_series(raw, va)
        if not a.empty:
            ax.plot(a["epoch"], a[tr], label="train", lw=1.8)
        if not b.empty:
            ax.plot(b["epoch"], b[va], label="val", lw=1.8, ls="--")
        ax.set_title(lab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "train_vs_val.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_gene_mse(raw: pd.DataFrame, out_dir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col, lab, ls in (
        ("train/gene_mse", "train", "-"),
        ("val/gene_mse", "val", "--"),
        ("val/gene_loss", "val gene_loss", ":"),
    ):
        s = epoch_series(raw, col)
        if not s.empty:
            ax.plot(s["epoch"], s[col], label=lab, ls=ls, lw=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE / loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gene_mse_and_loss.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_aux(raw: pd.DataFrame, out_dir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col, lab in (
        ("val/contrastive_loss", "val InfoNCE"),
        ("val/mask_ce_loss", "val mask-CE"),
        ("train/mask_ce_loss", "train mask-CE"),
    ):
        s = epoch_series(raw, col)
        if not s.empty:
            ax.plot(s["epoch"], s[col], label=lab, lw=1.8)
    ax.set_xlabel("epoch")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "aux_losses_contr_maskce.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_gene_by_time(raw: pd.DataFrame, out_dir: Path, title: str, labels: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, name in labels.items():
        col = f"val/gene_loss_t{k}"
        s = epoch_series(raw, col)
        if not s.empty:
            ax.plot(s["epoch"], s[col], label=name, lw=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val gene loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gene_loss_by_time.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    # also CSV
    cols = [f"val/gene_loss_t{k}" for k in labels]
    pieces = []
    for k, name in labels.items():
        col = f"val/gene_loss_t{k}"
        s = epoch_series(raw, col)
        if not s.empty:
            s = s.rename(columns={col: name})
            pieces.append(s.set_index("epoch"))
    if pieces:
        pd.concat(pieces, axis=1).to_csv(out_dir / "val_gene_loss_by_time.csv")


def plot_run(run: dict) -> None:
    metrics = Path(run["metrics"])
    out_dir = OUT_ROOT / run["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if not metrics.is_file():
        print("MISSING", metrics)
        return
    raw = pd.read_csv(metrics)
    title = run["title"]
    print("plot", run["name"], "->", out_dir)
    plot_train_val(raw, out_dir, title)
    plot_gene_mse(raw, out_dir, title)
    plot_aux(raw, out_dir, title)
    labels = run.get("time_labels") or {"1": "t1", "2": "t2", "3": "t3"}
    plot_gene_by_time(raw, out_dir, title, labels)
    # epoch metrics snapshot
    keep_cols = [c for c in raw.columns if c.startswith("val/") or c == "epoch"]
    ep = raw[keep_cols].dropna(how="all", subset=[c for c in keep_cols if c != "epoch"])
    ep = ep.drop_duplicates("epoch").sort_values("epoch")
    ep.to_csv(out_dir / "epoch_metrics.csv", index=False)
    if run.get("mark_epoch") is not None:
        print("  mark_epoch", run["mark_epoch"])


def main() -> None:
    for run in RUNS:
        plot_run(run)
    print("done")


if __name__ == "__main__":
    main()
