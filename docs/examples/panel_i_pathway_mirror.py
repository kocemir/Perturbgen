#!/usr/bin/env python3
"""Panel i: pathway mirror + weighted reversal (Parse IL-1β vs in-silico IL1B KO).

Follows PerturbGen Methods (Ly et al. bioRxiv):
  - GT: Parse 10M PBMC cytokine screen, IL-1β vs PBS, myeloid
  - KO: Wilcoxon on predicted KO vs control at 6h / 10h (myeloid)
  - ORA: Reactome_Pathways_2024 via gseapy Enrichr
  - Mirror: pathway scores scaled to [-1, 1] per condition
  - Reversal: GT-weighted fraction of genes with opposite KO sign; permutation null

Parse DEGs are cell-type-wise. We aggregate CD14_Mono + CD16_Mono + cDC
(notebook-07 myeloid; pDC / generic Mono excluded) by mean log_fc across types
present in ≥2 cell types, requiring ≥2 types with adj_p<0.05 and the same sign.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import anndata as ad
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import yaml

REPO = Path(__file__).resolve().parent.parent.parent  # Perturbgen/
EXAMPLES = Path(__file__).resolve().parent
KO_CFG_DIR = EXAMPLES / "configs" / "ko"
PARSE_DEGS = EXAMPLES / "external" / "parse10m" / "degs_myeloid.parquet"
MYELOID_PARSE = ("CD14_Mono", "CD16_Mono", "cDC")
MYELOID_KO = {
    "CD14 monocytes",
    "CD14+ monocytes",
    "CD16 monocytes",
    "CD16+ monocytes",
    "Dendritic cells",
}
CT_MAP = {
    "CD14 monocytes": "CD14+ monocytes",
    "CD16 monocytes": "CD16+ monocytes",
    "B cell": "B cells",
    "NK": "NK cells",
    "NKT": "NKT cells",
    "Plasmocytoid dendritic cell": "Plasmacytoid dendritic cells",
    "hematopoietic stem cell": "Hematopoietic stem cells",
    "platelet": "Platelets",
}
TIMES = ("6h_LPS", "10h_LPS")


def load_registry() -> tuple[dict, dict[str, dict]]:
    with open(KO_CFG_DIR / "comparison.yaml") as f:
        cmp = yaml.safe_load(f)
    exps = {}
    for name, row in cmp["experiments"].items():
        with open(KO_CFG_DIR / row["yaml"]) as f:
            cfg = yaml.safe_load(f)
        cfg["_name"] = name
        exps[name] = cfg
    return cmp, exps


def preferred_h5ad(cfg: dict) -> Path:
    d = Path(cfg["trainer"]["output_dir"])
    pref = cfg["experiment"].get("preferred_dump")
    if pref and (d / pref).is_file():
        return d / pref
    g = cfg["experiment"].get(
        "dump_glob", "*_minference_adata_gENSG00000125538_ssrc_*.h5ad"
    )
    hits = sorted(d.glob(g))
    if not hits:
        raise FileNotFoundError(d)
    return hits[-1]


def aggregate_parse_il1b(
    path: Path = PARSE_DEGS,
    min_types: int = 2,
    lfc_ora: float = 0.2,
    lfc_gt: float = 0.5,
) -> pd.DataFrame:
    """Return one row per gene with aggregated myeloid IL-1β stats."""
    df = pd.read_parquet(path)
    sub = df[
        (df["cytokine"].astype(str) == "IL-1-beta")
        & (df["celltype"].isin(MYELOID_PARSE))
        & (~df["well_biased"].fillna(False))
    ].copy()
    rows = []
    for gene, g in sub.groupby("gene"):
        if len(g) < min_types:
            continue
        mean_lfc = float(g["log_fc"].mean())
        sign = np.sign(mean_lfc) if mean_lfc != 0 else 0.0
        n_sig = int(
            (
                (g["adj_p_value"] < 0.05)
                & (np.sign(g["log_fc"]) == sign)
                & (sign != 0)
            ).sum()
        )
        rows.append(
            {
                "gene": str(gene),
                "mean_log_fc": mean_lfc,
                "median_adj_p": float(g["adj_p_value"].median()),
                "min_adj_p": float(g["adj_p_value"].min()),
                "n_types": int(len(g)),
                "n_sig_same_sign": n_sig,
            }
        )
    out = pd.DataFrame(rows)
    out["sig_ora"] = (
        (out["n_sig_same_sign"] >= min_types)
        & (out["mean_log_fc"].abs() > lfc_ora)
    )
    out["sig_gt"] = (
        (out["n_sig_same_sign"] >= min_types)
        & (out["mean_log_fc"].abs() > lfc_gt)
    )
    out["direction"] = np.sign(out["mean_log_fc"])
    return out.sort_values("mean_log_fc", ascending=False)


def ko_degs_myeloid(
    h5: Path,
    sym_to_ens: dict,
    ens_to_sym: dict,
    token_to_ens: dict,
) -> dict[str, pd.DataFrame]:
    """Wilcoxon KO vs control per timepoint on myeloid cells. Returns symbol-indexed DE."""
    adata = sc.read(h5)
    if {"cell_pairing_index", "time_after_LPS"}.issubset(adata.obs.columns):
        adata = adata[
            ~adata.obs.duplicated(["cell_pairing_index", "time_after_LPS"])
        ].copy()
    if "pred_counts" not in adata.layers:
        raise KeyError(h5)
    pred = adata.copy()
    pred.X = pred.layers["pred_counts"].copy()
    adata.obs["status"] = "perturbed"
    pred.obs["status"] = "unperturbed"
    both = ad.concat(
        {"perturbed": adata, "unperturbed": pred},
        label="batch_status",
        index_unique="-",
    )
    both.obs["cell_type_harmonized"] = (
        both.obs["cell_type_harmonized"]
        .astype(str)
        .map(lambda x: CT_MAP.get(x, x))
    )
    both = both[both.obs["cell_type_harmonized"].isin(MYELOID_KO)].copy()
    sc.pp.normalize_total(both)
    sc.pp.log1p(both)

    # Map var_names -> symbols
    var = both.var_names.astype(str)
    symbols = []
    for v in var:
        if v in ens_to_sym:
            symbols.append(ens_to_sym[v])
        elif v.isdigit() and int(v) in token_to_ens:
            e = token_to_ens[int(v)]
            symbols.append(ens_to_sym.get(e, e))
        elif v in sym_to_ens:
            symbols.append(v)
        else:
            symbols.append(v)
    both.var["gene_symbol"] = symbols
    both = both[:, both.var["gene_symbol"].notna()].copy()
    # drop duplicate symbols keeping first
    _, keep = np.unique(both.var["gene_symbol"].astype(str), return_index=True)
    both = both[:, np.sort(keep)].copy()
    both.var_names = both.var["gene_symbol"].astype(str)

    out = {}
    for tp in TIMES:
        mask = both.obs["time_after_LPS"].astype(str) == tp
        sub = both[mask].copy()
        if sub.obs["status"].nunique() < 2 or sub.n_obs < 20:
            out[tp] = pd.DataFrame(columns=["gene", "logfoldchanges", "pvals_adj"])
            continue
        sub.obs["group"] = sub.obs["status"].astype("category")
        sc.tl.rank_genes_groups(
            sub, groupby="group", reference="unperturbed", method="wilcoxon"
        )
        de = sc.get.rank_genes_groups_df(sub, group="perturbed")
        de = de.rename(columns={"names": "gene"})
        de["gene"] = de["gene"].astype(str)
        out[tp] = de
        print(
            f"  {h5.name} {tp}: myeloid n={sub.n_obs} "
            f"sig02={int(((de.pvals_adj < 0.05) & (de.logfoldchanges.abs() > 0.2)).sum())}"
        )
    return out


def run_ora(
    genes: list[str],
    background: list[str],
    gene_sets: str = "Reactome_Pathways_2024",
) -> pd.DataFrame:
    if len(genes) < 5:
        return pd.DataFrame()
    try:
        enr = gp.enrichr(
            gene_list=list(dict.fromkeys(genes)),
            gene_sets=gene_sets,
            organism="Human",
            background=list(dict.fromkeys(background)),
            outdir=None,
            verbose=False,
        )
        df = enr.results.copy()
    except Exception as e:
        print(f"  enrichr failed ({e}); retry without background")
        enr = gp.enrichr(
            gene_list=list(dict.fromkeys(genes)),
            gene_sets=gene_sets,
            organism="Human",
            outdir=None,
            verbose=False,
        )
        df = enr.results.copy()
    df = df.rename(columns={"Term": "pathway", "Adjusted P-value": "padj"})
    if "Genes" in df.columns:
        df["genes"] = df["Genes"].astype(str).str.replace(";", ",", regex=False)
    elif "Overlap" in df.columns:
        df["genes"] = ""
    keep = (df["padj"] < 0.05) & (df.get("Overlap", "5/5").astype(str).str.split("/").str[0].astype(int) >= 5)
    # Overlap like "12/80"
    if "Overlap" in df.columns:
        n_in = df["Overlap"].astype(str).str.split("/").str[0].astype(int)
        keep = (df["padj"] < 0.05) & (n_in >= 5)
    else:
        keep = df["padj"] < 0.05
    return df.loc[keep].reset_index(drop=True)


def pathway_score(pathway_genes: list[str], de: pd.DataFrame, gene_col: str, lfc_col: str) -> float:
    m = de.set_index(gene_col)
    signs = []
    for g in pathway_genes:
        if g in m.index:
            signs.append(float(np.sign(m.loc[g, lfc_col])))
    if not signs:
        return 0.0
    return float(np.sum(signs))


def mirror_table(
    ora_stim: pd.DataFrame,
    ko_de: pd.DataFrame,
    stim_de: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, r in ora_stim.iterrows():
        genes = [g.strip() for g in str(r.get("genes", "")).split(",") if g.strip()]
        if not genes and "Genes" in r:
            genes = [g.strip() for g in str(r["Genes"]).replace(";", ",").split(",") if g.strip()]
        s_stim = pathway_score(genes, stim_de, "gene", "mean_log_fc")
        s_ko = pathway_score(genes, ko_de, "gene", "logfoldchanges")
        rows.append(
            {
                "pathway": r["pathway"],
                "padj_stim": r["padj"],
                "score_stim_raw": s_stim,
                "score_ko_raw": s_ko,
                "n_genes": len(genes),
                "genes": ",".join(genes),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col, raw in ("score_stim", "score_stim_raw"), ("score_ko", "score_ko_raw"):
        m = df[raw].abs().max()
        df[col] = df[raw] / m if m > 0 else 0.0
    df["reversed"] = np.sign(df["score_stim"]) * np.sign(df["score_ko"]) < 0
    return df.sort_values("score_stim", ascending=False)


def weighted_reversal(
    ora_stim: pd.DataFrame,
    ko_de: pd.DataFrame,
    stim_gt: pd.DataFrame,
    n_perm: int = 200,
    seed: int = 0,
) -> dict:
    """GT-weighted reversal over pathways enriched on GT-up genes."""
    gt = stim_gt.set_index("gene")
    ko = ko_de.set_index("gene")
    # evaluable genes = GT up genes present in KO table
    gt_up = stim_gt[(stim_gt["sig_gt"]) & (stim_gt["mean_log_fc"] > 0)].copy()
    gt_up = gt_up[gt_up["gene"].isin(ko.index)]
    if gt_up.empty or ora_stim.empty:
        return {"accuracy": np.nan, "null_mean": np.nan, "p_emp": np.nan, "n_pathways": 0}

    pathway_scores = []
    for _, r in ora_stim.iterrows():
        genes = [g.strip() for g in str(r.get("genes", r.get("Genes", ""))).replace(";", ",").split(",") if g.strip()]
        genes = [g for g in genes if g in gt_up["gene"].values and g in ko.index]
        if len(genes) < 5:
            continue
        w = gt.loc[genes, "mean_log_fc"].abs().to_numpy()
        gt_sign = np.sign(gt.loc[genes, "mean_log_fc"].to_numpy())
        ko_sign = np.sign(ko.loc[genes, "logfoldchanges"].to_numpy())
        rev = (gt_sign * ko_sign) < 0
        pathway_scores.append(float(np.sum(w[rev]) / np.sum(w)))
    if not pathway_scores:
        return {"accuracy": np.nan, "null_mean": np.nan, "p_emp": np.nan, "n_pathways": 0}
    acc = float(np.mean(pathway_scores))

    rng = np.random.default_rng(seed)
    genes_all = gt_up["gene"].tolist()
    ko_signs = np.sign(ko.loc[genes_all, "logfoldchanges"].to_numpy())
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(ko_signs)
        perm_map = dict(zip(genes_all, perm))
        ps = []
        for _, r in ora_stim.iterrows():
            genes = [g.strip() for g in str(r.get("genes", r.get("Genes", ""))).replace(";", ",").split(",") if g.strip()]
            genes = [g for g in genes if g in perm_map and g in gt.index]
            if len(genes) < 5:
                continue
            w = gt.loc[genes, "mean_log_fc"].abs().to_numpy()
            gt_sign = np.sign(gt.loc[genes, "mean_log_fc"].to_numpy())
            ko_sign = np.array([perm_map[g] for g in genes])
            rev = (gt_sign * ko_sign) < 0
            ps.append(float(np.sum(w[rev]) / np.sum(w)))
        if ps:
            null.append(float(np.mean(ps)))
    null = np.asarray(null) if null else np.array([np.nan])
    p_emp = float(np.mean(null >= acc)) if len(null) else np.nan
    return {
        "accuracy": acc,
        "null_mean": float(np.nanmean(null)),
        "null_std": float(np.nanstd(null)),
        "p_emp": p_emp,
        "n_pathways": len(pathway_scores),
        "n_perm": int(len(null)),
    }


def plot_mirror(df: pd.DataFrame, title: str, path: Path, top_n: int = 20) -> None:
    if df.empty:
        print("  empty mirror, skip", path)
        return
    # Prefer pathways with stim up (positive) for left/right mirror like the paper
    use = df.copy()
    use = use.reindex(use["score_stim"].abs().sort_values(ascending=False).index).head(top_n)
    use = use.sort_values("score_stim")
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.35 * len(use))))
    y = np.arange(len(use))
    ax.barh(y - 0.15, use["score_stim"], height=0.3, color="#d95f02", label="IL-1β stim (Parse)")
    ax.barh(y + 0.15, use["score_ko"], height=0.3, color="#1b9e77", label="IL1B KO (model)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(use["pathway"], fontsize=8)
    ax.set_xlabel("Scaled pathway direction (−1 … +1)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", path)


def plot_reversal_bars(rev_df: pd.DataFrame, path: Path) -> None:
    if rev_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.barplot(data=rev_df, x="time", y="accuracy", hue="experiment", ax=ax)
    # null baseline as dashed
    for i, row in rev_df.iterrows():
        pass
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance ~0.5")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Weighted pathway reversal accuracy")
    ax.set_title("Panel i — reversal accuracy vs Parse IL-1β")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    cmp, exps = load_registry()
    out_dir = Path(cmp["output_dir"]) / "panel_i"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cmp["maps"]["symbol_to_ensembl"], "rb") as f:
        sym_to_ens = pickle.load(f)
    ens_to_sym = {str(v): str(k) for k, v in sym_to_ens.items()}
    with open(cmp["maps"]["token_id_to_ensembl"], "rb") as f:
        token_to_ens = {int(k): str(v) for k, v in pickle.load(f).items()}

    print("Aggregating Parse IL-1β myeloid DEGs…")
    stim = aggregate_parse_il1b()
    stim.to_csv(out_dir / "parse_il1b_myeloid_aggregated.csv", index=False)
    up_ora = stim[stim["sig_ora"] & (stim["mean_log_fc"] > 0)]["gene"].tolist()
    up_gt = stim[stim["sig_gt"] & (stim["mean_log_fc"] > 0)]["gene"].tolist()
    print(f"  genes={len(stim)}  up_ora={len(up_ora)}  up_gt(|lfc|>0.5)={len(up_gt)}")

    # Background ≈ KO HVG symbols (from first dump)
    first_h5 = preferred_h5ad(next(iter(exps.values())))
    a0 = ad.read_h5ad(first_h5, backed="r")
    bg = []
    for v in a0.var_names.astype(str):
        if v in ens_to_sym:
            bg.append(ens_to_sym[v])
        elif v.isdigit() and int(v) in token_to_ens:
            bg.append(ens_to_sym.get(token_to_ens[int(v)], token_to_ens[int(v)]))
        else:
            bg.append(v)
    background = sorted(set(bg) | set(stim["gene"]))
    print("  background", len(background))

    print("ORA on Parse IL-1β upregulated genes…")
    ora_stim = run_ora(up_ora if up_ora else up_gt, background)
    # Prefer ORA on GT-strict set for reversal pathways (Methods: enrichment on GT up)
    ora_gt = run_ora(up_gt, background)
    if ora_gt.empty and not ora_stim.empty:
        ora_gt = ora_stim
    ora_stim.to_csv(out_dir / "ora_parse_il1b_up_ora02.csv", index=False)
    ora_gt.to_csv(out_dir / "ora_parse_il1b_up_gt05.csv", index=False)
    print(f"  pathways ora02={len(ora_stim)}  gt05={len(ora_gt)}")

    rev_rows = []
    for name, cfg in exps.items():
        label = cfg["experiment"]["label"]
        h5 = preferred_h5ad(cfg)
        print(f"\n=== {name} ({label}) ===\n  {h5}")
        ko_by_tp = ko_degs_myeloid(h5, sym_to_ens, ens_to_sym, token_to_ens)
        for tp, ko_de in ko_by_tp.items():
            tag = tp.replace("_LPS", "")
            ko_de.to_csv(out_dir / f"ko_degs_{name}_{tag}.csv", index=False)
            # Restrict KO table to overlapping panel with stim
            overlap = ko_de[ko_de["gene"].isin(stim["gene"])].copy()
            mir = mirror_table(ora_stim if not ora_stim.empty else ora_gt, overlap, stim)
            mir.to_csv(out_dir / f"mirror_{name}_{tag}.csv", index=False)
            plot_mirror(
                mir,
                title=f"Panel i — {label} | {tag}",
                path=out_dir / f"mirror_{name}_{tag}.png",
                top_n=args.top_n,
            )
            rev = weighted_reversal(ora_gt if not ora_gt.empty else ora_stim, overlap, stim, n_perm=args.n_perm)
            rev_rows.append(
                {
                    "name": name,
                    "experiment": label,
                    "time": tag,
                    **{k: rev[k] for k in ("accuracy", "null_mean", "null_std", "p_emp", "n_pathways", "n_perm")},
                }
            )
            print(
                f"  reversal {tag}: acc={rev['accuracy']:.3f} "
                f"null={rev['null_mean']:.3f} p={rev['p_emp']:.4f} n_pw={rev['n_pathways']}"
            )

    rev_df = pd.DataFrame(rev_rows)
    rev_df.to_csv(out_dir / "reversal_accuracy.csv", index=False)
    plot_reversal_bars(rev_df, out_dir / "reversal_accuracy.png")

    # Combined mirror facet: one PNG with models as columns for 6h
    print("\nDone. Outputs in", out_dir)
    print(rev_df.to_string(index=False))


if __name__ == "__main__":
    main()
