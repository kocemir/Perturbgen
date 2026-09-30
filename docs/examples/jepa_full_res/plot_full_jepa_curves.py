#!/usr/bin/env python3
"""Plot epoch curves for the gene-only full JEPA runs (not a sweep)."""
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

RUNS = [
    {
        "name": "gene_only_resting_20260920_021803",
        "title": "resting → 90m/6h/10h  ·  λ_gene=1, λ_cell=0, VICReg off",
        "metrics": ATLAS
        / "fzT_encL3_predL3_q128_mixed_lg1_lc0_contr0_vic0_bs16_ep20_lr0.0001_seed0_splitT_poolcls_20260920_021803"
        / "logs"
        / "version_0"
        / "metrics.csv",
        "mark_epoch": 8,
    },
    {
        "name": "gene_only_90m_src_20260921_004618",
        "title": "90m → 6h/10h  ·  same recipe, latest timestamp",
        "metrics": SRC90
        / "fzT_encL3_predL3_q128_mixed_lg1_lc0_contr0_vic0_bs16_ep20_lr0.0001_seed0_splitF_poolcls_tp1-2_src90m_20260921_004618"
        / "logs"
        / "version_0"
        / "metrics.csv",
        "mark_epoch": 8,
    },
    {
        "name": "gene_only_resting_freeze15_20260921_174246",
        "title": "resting → 90m/6h/10h  ·  encoder freeze:15, predictor 1e-4, 15 ep",
        "metrics": ATLAS
        / "fzT_encL3_predL3_q128_mixed_lg1_lc0_contr0_vic0_bs16_ep15_lr0.0001_seed0_splitT_poolcls_tp1-2-3_20260921_174246"
        / "logs"
        / "version_0"
        / "metrics.csv",
        "mark_epoch": None,
    },
    {
        "name": "gene_only_resting_30ep_20260922_133521",
        "title": "resting → 90m/6h/10h  ·  encoder frozen, cosine, 30 ep",
        "metrics": ATLAS
        / "fzT_encL3_predL3_q128_mixed_lg1_lc0_contr0_vic0_bs16_ep30_lr0.0001_seed0_splitT_poolcls_tp1-2-3_20260922_133521"
        / "logs"
        / "version_0"
        / "metrics.csv",
        "mark_epoch": None,
    },
    # --- 2026-09-25 finished full-atlas runs ---
    {
        "name": "frozen_cosine_only_100ep_bs4_20260925",
        "title": (
            "resting → 90m/6h/10h  ·  frozen enc L3  ·  "
            "cosine only (no CE / contrastive / query-src-pos)  ·  bs4 × 8gpu  ·  100 ep"
        ),
        "metrics": ATLAS
        / (
            "fzT_encL3_predL3_q128_mixed_lg1_lc0_contr0_vic0_glcosine_bs4_ep100_"
            "lr0.0001_seed0_splitT_poolcls_tp1-2-3_max512_20260925_112921"
        )
        / "logs"
        / "version_0"
        / "metrics.csv",
        "mark_epoch": None,
        "gene_loss_in_obj": "cosine",
    },
    {
        "name": "frozen_mse_contr_maskce_qsrcpos_50ep_bs16_20260925",
        "title": (
            "resting → 90m/6h/10h  ·  frozen enc L3  ·  "
            "MSE + λ_contr=0.1 + λ_maskCE=0.1 (20% shared) + learnable query-src-pos  ·  "
            "bs16 × 8gpu  ·  50 ep"
        ),
        "metrics": ATLAS
        / (
            "fzT_encL3_predL3_q128_mixed_lg1_lc0_contr0.1_vic0_glmse_bs16_ep50_"
            "lr0.0001_seed0_splitT_poolcls_tp1-2-3_max512_mce0.1_ms0.2_qsrcpos_20260925_203433"
        )
        / "logs"
        / "version_0"
        / "metrics.csv",
        "mark_epoch": None,
        "gene_loss_in_obj": "mse",
    },
    {
        "name": "frozen_mse_contr_maskce_qsrcpos_100ep_bs16_20260926",
        "title": (
            "resting → 90m/6h/10h  ·  frozen enc L3  ·  "
            "MSE + λ_contr=0.1 + λ_maskCE=0.1 (20% shared) + learnable query-src-pos  ·  "
            "bs16 × 8gpu  ·  100 ep"
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
    },
    {
        "name": "frozen_mse_contr_maskce_qsrcpos_100ep_90m_src_20260928",
        "title": (
            "90m → 6h/10h  ·  frozen enc L3  ·  "
            "MSE + λ_contr=0.1 + λ_maskCE=0.1 (20% shared) + learnable query-src-pos  ·  "
            "bs16 × 8gpu  ·  100 ep  ·  --split false (train curves)"
        ),
        "metrics": SRC90
        / (
            "fzT_encL3_predL3_q128_mixed_lg1_lc0_contr0.1_vic0_glmse_bs16_ep100_"
            "lr0.0001_seed0_splitF_poolcls_tp1-2_max512_src90m_mce0.1_ms0.2_qsrcpos_20260928_053236"
        )
        / "logs"
        / "version_0"
        / "metrics.csv",
        "mark_epoch": None,
        "gene_loss_in_obj": "mse",
        "time_labels": {"1": "6h", "2": "10h"},
    },
]


def epoch_table(raw: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    if "val/gene_gap_vs_copy_src" in raw.columns and raw["val/gene_gap_vs_copy_src"].notna().any():
        prefix = "val"
    elif "train/gene_gap_vs_copy_src" in raw.columns and raw["train/gene_gap_vs_copy_src"].notna().any():
        prefix = "train"
    else:
        raise ValueError("no gene_gap_vs_copy_src")
    gap = f"{prefix}/gene_gap_vs_copy_src"
    ep = (
        raw[raw[gap].notna()]
        .copy()
        .sort_values("epoch")
        .drop_duplicates("epoch", keep="last")
        .reset_index(drop=True)
    )
    ep["epoch"] = ep["epoch"].astype(int)
    return prefix, ep


def metric_col(ep: pd.DataFrame, prefix: str, name: str) -> str | None:
    for c in (f"{prefix}/{name}", f"{prefix}/{name}_epoch"):
        if c in ep.columns and ep[c].notna().any():
            return c
    return None


def mark(ax, epoch: int | None) -> None:
    if epoch is None:
        return
    ax.axvline(epoch, color="#c44e52", ls="--", lw=1.0, alpha=0.85, label=f"ckpt epoch={epoch}")


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_run(spec: dict) -> None:
    csv_path = spec["metrics"]
    assert csv_path.is_file(), csv_path
    prefix, ep = epoch_table(pd.read_csv(csv_path))
    out = OUT_ROOT / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    mark_ep = spec.get("mark_epoch")
    if mark_ep is None and f"{prefix}/gene_cos_pred" in ep.columns:
        mark_ep = int(ep.loc[ep[f"{prefix}/gene_cos_pred"].idxmax(), "epoch"])
    x = ep["epoch"].to_numpy()

    keep = [c for c in ep.columns if c == "epoch" or c.startswith(f"{prefix}/")]
    ep[keep].to_csv(out / "epoch_metrics.csv", index=False)

    gap = ep[f"{prefix}/gene_gap_vs_copy_src"].to_numpy()
    cos = ep[f"{prefix}/gene_cos_pred"].to_numpy()
    copy = cos - gap

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(x, cos, color="#c44e52", lw=2.0, label="cos(pred, tgt)")
    ax.plot(x, copy, color="#4c72b0", lw=2.0, label="cos(src, tgt)  copy baseline")
    mark(ax, mark_ep)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cosine")
    ax.set_title(f"{spec['title']}\ngene cosine vs copy-src  ({prefix})")
    ax.legend(frameon=False, loc="best")
    ax.set_xlim(x.min(), x.max())
    save(fig, out / "gene_cosine_vs_copy.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(x, gap, color="#55a868", lw=2.0, label="gap = pred − copy")
    ax.axhline(0.0, color="#666666", lw=0.8)
    ax.axhline(0.05, color="#666666", ls="--", lw=0.8, label="pass 0.05")
    mark(ax, mark_ep)
    ax.set_xlabel("epoch")
    ax.set_ylabel("gene_gap_vs_copy_src")
    ax.set_title(f"{spec['title']}\nhonesty gap ({prefix})")
    ax.legend(frameon=False)
    ax.set_xlim(x.min(), x.max())
    save(fig, out / "gene_gap_vs_copy.png")

    if f"{prefix}/gene_gap_vs_static" in ep.columns:
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        ax.plot(
            x,
            ep[f"{prefix}/gene_gap_vs_static"],
            color="#8c6d31",
            lw=2.0,
            label="gap vs static e(g)",
        )
        ax.axhline(0.0, color="#666666", lw=0.8)
        mark(ax, mark_ep)
        ax.set_xlabel("epoch")
        ax.set_ylabel("gene_gap_vs_static")
        ax.set_title(f"{spec['title']}\nhonesty gap vs gene identity ({prefix})")
        ax.legend(frameon=False)
        ax.set_xlim(x.min(), x.max())
        save(fig, out / "gene_gap_vs_static.png")

    gene_loss_in_obj = spec.get("gene_loss_in_obj", "cosine")
    mse_c = metric_col(ep, prefix, "gene_mse")
    loss_c = metric_col(ep, prefix, "gene_loss")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    if mse_c is not None:
        axes[0].plot(x, ep[mse_c], color="#8172b3", lw=2.0)
        axes[0].set_ylabel("gene_mse")
    axes[0].set_xlabel("epoch")
    if gene_loss_in_obj == "mse":
        axes[0].set_title("gene MSE (= gene objective term)")
    else:
        axes[0].set_title("gene MSE (logged, not in objective)")
    mark(axes[0], mark_ep)
    if loss_c is not None:
        axes[1].plot(x, ep[loss_c], color="#cc8963", lw=2.0)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("gene_loss")
    axes[1].set_title(f"gene loss ({gene_loss_in_obj})")
    mark(axes[1], mark_ep)
    fig.suptitle(spec["title"], y=1.03)
    save(fig, out / "gene_mse_and_loss.png")

    # Extra panel when contrastive / mask-CE were active.
    contr_c = metric_col(ep, prefix, "contrastive_loss")
    mce_c = metric_col(ep, prefix, "mask_ce_loss")
    total_c = metric_col(ep, prefix, "total_loss")
    has_contr = contr_c is not None and float(ep[contr_c].fillna(0).abs().max()) > 1e-8
    has_mce = mce_c is not None and float(ep[mce_c].fillna(0).abs().max()) > 1e-8
    if has_contr or has_mce:
        n = int(has_contr) + int(has_mce) + int(total_c is not None)
        fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.0))
        if n == 1:
            axes = [axes]
        i = 0
        if total_c is not None:
            axes[i].plot(x, ep[total_c], color="#333333", lw=2.0)
            axes[i].set_title("total loss")
            axes[i].set_xlabel("epoch")
            mark(axes[i], mark_ep)
            i += 1
        if has_contr:
            axes[i].plot(x, ep[contr_c], color="#4c72b0", lw=2.0)
            axes[i].set_title("contrastive (InfoNCE)")
            axes[i].set_xlabel("epoch")
            mark(axes[i], mark_ep)
            i += 1
        if has_mce:
            axes[i].plot(x, ep[mce_c], color="#c44e52", lw=2.0)
            axes[i].axhline(
                float(np.log(1860)),
                color="#666666",
                ls="--",
                lw=0.8,
                label="chance ln(1860)",
            )
            axes[i].set_title("mask-CE")
            axes[i].set_xlabel("epoch")
            axes[i].legend(frameon=False, fontsize=9)
            mark(axes[i], mark_ep)
        fig.suptitle(spec["title"], y=1.03)
        save(fig, out / "aux_losses_contr_maskce.png")

    tcols = [
        c
        for c in (
            f"{prefix}/gene_loss_t1",
            f"{prefix}/gene_loss_t2",
            f"{prefix}/gene_loss_t3",
            f"{prefix}/gene_loss_t1_epoch",
            f"{prefix}/gene_loss_t2_epoch",
            f"{prefix}/gene_loss_t3_epoch",
        )
        if c in ep.columns
    ]
    if tcols:
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        labels = spec.get("time_labels") or {"1": "90m", "2": "6h", "3": "10h"}
        for col in tcols:
            tag = col.rsplit("_t", 1)[-1].replace("_epoch", "")
            ax.plot(x, ep[col], lw=1.8, label=labels.get(tag, col))
        mark(ax, mark_ep)
        ax.set_xlabel("epoch")
        ax.set_ylabel("gene_loss by target time")
        ax.set_title(f"{spec['title']}\nper-time gene loss ({prefix})")
        ax.legend(frameon=False)
        save(fig, out / "gene_loss_by_time.png")

    if f"{prefix}/cell_gap_vs_copy_src" in ep.columns and ep[f"{prefix}/cell_gap_vs_copy_src"].notna().any():
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        ax.plot(x, ep[f"{prefix}/cell_gap_vs_copy_src"], color="#937860", lw=2.0, label="cell gap")
        if f"{prefix}/cell_cos_pred" in ep.columns:
            ax.plot(x, ep[f"{prefix}/cell_cos_pred"], color="#da8bc3", lw=1.6, alpha=0.85, label="cell cos(pred,tgt)")
        ax.axhline(0.0, color="#666666", lw=0.8)
        mark(ax, mark_ep)
        ax.set_xlabel("epoch")
        ax.set_title(f"{spec['title']}\ncell pool (λ_cell=0, logged only)  ({prefix})")
        ax.legend(frameon=False)
        save(fig, out / "cell_logged.png")

    best = int(ep.loc[ep[f"{prefix}/gene_cos_pred"].idxmax(), "epoch"])
    last = int(ep["epoch"].iloc[-1])
    print(
        spec["name"],
        f"prefix={prefix}",
        f"epochs={len(ep)}",
        f"best_cos_epoch={best}",
        f"last={last}",
        f"gap@8={float(ep.loc[ep['epoch']==8, f'{prefix}/gene_gap_vs_copy_src'].iloc[0]) if (ep['epoch']==8).any() else 'n/a'}",
        "->",
        out,
    )


def main() -> None:
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "font.size": 11})
    for spec in RUNS:
        plot_run(spec)


if __name__ == "__main__":
    main()
