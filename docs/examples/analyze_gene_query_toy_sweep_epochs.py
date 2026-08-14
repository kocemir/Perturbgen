#!/usr/bin/env python3
"""Detailed epoch-wise analysis for Gene-Query JEPA toy sweep metrics.

This script complements:
  docs/examples/summarize_gene_query_toy_sweep.py

It builds:
1) plots for each selected metric across epochs with useful groupings,
2) a lightweight text report with coverage and warnings.

Usage examples (commands will be requested later by user):
  python docs/examples/analyze_gene_query_toy_sweep_epochs.py
  python docs/examples/analyze_gene_query_toy_sweep_epochs.py --suite-root <dir>
  python docs/examples/analyze_gene_query_toy_sweep_epochs.py --include-failed
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install it in your environment first."
    ) from exc


DEFAULT_SUITE_ROOT = (
    "/mnt/sod2-project/csb4/stuke1/perturbgen/"
    "gene_query_jepa/toy_runs/systematic_144"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "jepa_toy_res"
RUN_ID_RE = re.compile(r"^fz[TF]_vic[01]_contr(?:0|0\.3)_q(?:64|128|256)_L[1-6]$")

# High-signal metrics for quick reading; script still keeps all numeric columns in CSV.
DEFAULT_PLOT_METRICS = [
    "val/gene_gap_vs_copy_src",
    "val/gene_loss",
    "val/cell_loss",
    "val/contrastive_loss",
    "val/vicreg_loss",
    "train/loss_epoch",
]


def parse_env(env_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not env_path.is_file():
        return out
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def to_float_or_nan(value: str) -> float:
    if value is None:
        return float("nan")
    v = value.strip()
    if not v:
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def bool_flag_from_env(v: str) -> str:
    return "on" if str(v).strip().lower() in {"1", "true", "yes", "y"} else "off"


def collect_metric_rows(metrics_csv: Path) -> List[Dict[str, float]]:
    """Aggregate by epoch, preserving the latest finite value per metric."""
    by_epoch: Dict[int, Dict[str, float]] = {}
    with metrics_csv.open() as f:
        reader = csv.DictReader(f)
        for raw in reader:
            epoch_raw = raw.get("epoch")
            if epoch_raw is None:
                continue
            epoch_f = to_float_or_nan(epoch_raw)
            if not is_finite(epoch_f):
                continue
            epoch_i = int(epoch_f)
            row = by_epoch.setdefault(epoch_i, {"epoch": float(epoch_i)})
            for key, value in raw.items():
                v = to_float_or_nan(value)
                if key == "epoch":
                    row["epoch"] = float(epoch_i)
                elif is_finite(v):
                    # Keep the latest finite value seen for this epoch/metric.
                    row[key] = v
    return [by_epoch[e] for e in sorted(by_epoch.keys())]


def load_run_rows(run_dir: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    """Return run rows (one row per epoch) and list of parse warnings."""
    warnings: List[str] = []
    hparams = parse_env(run_dir / "hparams.env")
    metrics_candidates = sorted(run_dir.glob("toy_logs/**/metrics.csv"))
    if not metrics_candidates:
        return [], [f"{run_dir.name}: missing metrics.csv"]
    metrics_csv = metrics_candidates[-1]
    epochs = collect_metric_rows(metrics_csv)
    if not epochs:
        return [], [f"{run_dir.name}: empty numeric metrics in {metrics_csv}"]

    freeze = hparams.get("FREEZE_ENCODER", "")
    contr_lambda = to_float_or_nan(hparams.get("LAMBDA_CONTRASTIVE", ""))
    vic_var = to_float_or_nan(hparams.get("VICREG_VAR", ""))
    vic_cov = to_float_or_nan(hparams.get("VICREG_COV", ""))
    q = hparams.get("N_QUERIES", "")
    l = hparams.get("ENC_LAYERS", "")
    done = (run_dir / "DONE").is_file()

    out_rows: List[Dict[str, object]] = []
    for ep in epochs:
        row: Dict[str, object] = dict(ep)
        row["run_id"] = run_dir.name
        row["run_dir"] = str(run_dir)
        row["is_done"] = 1 if done else 0
        row["freeze"] = freeze
        row["freeze_group"] = bool_flag_from_env(freeze)
        row["contrastive_lambda"] = contr_lambda
        row["contrastive_group"] = "on" if is_finite(contr_lambda) and contr_lambda > 0 else "off"
        row["vicreg_var"] = vic_var
        row["vicreg_cov"] = vic_cov
        row["vicreg_group"] = (
            "on"
            if (is_finite(vic_var) and vic_var > 0) or (is_finite(vic_cov) and vic_cov > 0)
            else "off"
        )
        row["Q"] = q
        row["L"] = l
        row["q_group"] = f"q{q}" if q else "q?"
        row["l_group"] = f"L{l}" if l else "L?"
        row["cond_group"] = (
            f"fz_{row['freeze_group']}"
            f"__vic_{row['vicreg_group']}"
            f"__ctr_{row['contrastive_group']}"
        )
        out_rows.append(row)
    return out_rows, warnings


def discover_runs(suite_root: Path) -> List[Path]:
    return sorted(
        [
            p for p in suite_root.iterdir()
            if p.is_dir() and RUN_ID_RE.match(p.name)
        ]
    )


def summarise_run_coverage(run_dirs: Sequence[Path]) -> Dict[str, object]:
    total = len(run_dirs)
    done = 0
    with_metrics = 0
    done_with_metrics = 0
    not_done_with_metrics = 0
    no_metrics = 0
    incomplete: List[str] = []

    cond_counts: Dict[str, Dict[str, int]] = {}

    for run_dir in run_dirs:
        has_done = (run_dir / "DONE").is_file()
        has_metrics = bool(sorted(run_dir.glob("toy_logs/**/metrics.csv")))
        if has_done:
            done += 1
        if has_metrics:
            with_metrics += 1
        if has_done and has_metrics:
            done_with_metrics += 1
        if (not has_done) and has_metrics:
            not_done_with_metrics += 1
        if not has_metrics:
            no_metrics += 1
        if (not has_done) or (not has_metrics):
            incomplete.append(run_dir.name)

        cond = "__".join(run_dir.name.split("_")[:3])  # e.g. fzF_vic1_contr0.3
        c = cond_counts.setdefault(cond, {"total": 0, "done": 0, "metrics": 0})
        c["total"] += 1
        c["done"] += int(has_done)
        c["metrics"] += int(has_metrics)

    return {
        "total_runs": total,
        "done_runs": done,
        "with_metrics": with_metrics,
        "done_with_metrics": done_with_metrics,
        "not_done_with_metrics": not_done_with_metrics,
        "no_metrics": no_metrics,
        "incomplete_runs": incomplete,
        "by_condition": cond_counts,
    }


def infer_numeric_columns(rows: Sequence[Dict[str, object]]) -> List[str]:
    if not rows:
        return []
    skip = {
        "run_id",
        "run_dir",
        "freeze",
        "freeze_group",
        "contrastive_group",
        "vicreg_group",
        "Q",
        "L",
        "q_group",
        "l_group",
        "cond_group",
    }
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    numeric_cols: List[str] = []
    for key in sorted(all_keys):
        if key in skip:
            continue
        vals = [r.get(key) for r in rows]
        if any(isinstance(v, float) and is_finite(v) for v in vals):
            numeric_cols.append(key)
    return numeric_cols


def groupby_epoch_stats(
    rows: Sequence[Dict[str, object]],
    group_keys: Sequence[str],
    metric_cols: Sequence[str],
) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in ["epoch", *group_keys])
        buckets.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for key, items in sorted(buckets.items(), key=lambda kv: kv[0]):
        row_out: Dict[str, object] = {"epoch": key[0]}
        for idx, gk in enumerate(group_keys, start=1):
            row_out[gk] = key[idx]

        for metric in metric_cols:
            vals = [
                float(x[metric])
                for x in items
                if isinstance(x.get(metric), float) and is_finite(float(x[metric]))
            ]
            row_out[f"{metric}__count"] = len(vals)
            if vals:
                vals_sorted = sorted(vals)
                row_out[f"{metric}__mean"] = sum(vals) / len(vals)
                if len(vals) > 1:
                    mu = row_out[f"{metric}__mean"]
                    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
                    row_out[f"{metric}__std"] = math.sqrt(var)
                else:
                    row_out[f"{metric}__std"] = 0.0
                row_out[f"{metric}__min"] = vals_sorted[0]
                row_out[f"{metric}__max"] = vals_sorted[-1]
                row_out[f"{metric}__median"] = vals_sorted[len(vals_sorted) // 2]
            else:
                row_out[f"{metric}__mean"] = float("nan")
                row_out[f"{metric}__std"] = float("nan")
                row_out[f"{metric}__min"] = float("nan")
                row_out[f"{metric}__max"] = float("nan")
                row_out[f"{metric}__median"] = float("nan")
        out.append(row_out)
    return out


def list_present_plot_metrics(
    rows: Sequence[Dict[str, object]], selected: Sequence[str]
) -> List[str]:
    numeric_cols = set(infer_numeric_columns(rows))
    return [m for m in selected if m in numeric_cols]


def unique_in_order(values: Iterable[object]) -> List[object]:
    seen = set()
    out: List[object] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def make_plot(
    grouped_rows: Sequence[Dict[str, object]],
    metric: str,
    group_key: str,
    out_png: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10, 6))
    groups = unique_in_order(r[group_key] for r in grouped_rows)
    for g in groups:
        xs: List[float] = []
        ys: List[float] = []
        yerr: List[float] = []
        for row in grouped_rows:
            if row[group_key] != g:
                continue
            mean = row.get(f"{metric}__mean")
            if not isinstance(mean, float) or not is_finite(mean):
                continue
            std = row.get(f"{metric}__std")
            xs.append(float(row["epoch"]))
            ys.append(mean)
            yerr.append(float(std) if isinstance(std, float) and is_finite(std) else 0.0)
        if xs:
            plt.plot(xs, ys, label=str(g))
            # Shade +/- std for visual stability cues.
            lower = [y - s for y, s in zip(ys, yerr)]
            upper = [y + s for y, s in zip(ys, yerr)]
            plt.fill_between(xs, lower, upper, alpha=0.15)

    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", default=DEFAULT_SUITE_ROOT)
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Default: docs/examples/jepa_toy_res",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include runs without DONE marker if metrics.csv exists.",
    )
    parser.add_argument(
        "--plot-metrics",
        default=",".join(DEFAULT_PLOT_METRICS),
        help="Comma-separated metric names to plot.",
    )
    parser.add_argument(
        "--clean-plots",
        action="store_true",
        help="Delete existing PNG files under out-dir/plots before writing new ones.",
    )
    args = parser.parse_args()

    suite_root = Path(args.suite_root)
    if not suite_root.is_dir():
        raise SystemExit(f"suite root not found: {suite_root}")
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_plots:
        for png in plots_dir.glob("**/*.png"):
            png.unlink()

    run_dirs = discover_runs(suite_root)
    coverage = summarise_run_coverage(run_dirs)
    all_rows: List[Dict[str, object]] = []
    warnings: List[str] = []
    for run_dir in run_dirs:
        run_rows, run_warn = load_run_rows(run_dir)
        warnings.extend(run_warn)
        if not run_rows:
            continue
        if not args.include_failed:
            # Keep only complete runs unless user explicitly asks otherwise.
            if not all(int(r["is_done"]) == 1 for r in run_rows):
                continue
        all_rows.extend(run_rows)

    if not all_rows:
        raise SystemExit(
            f"no usable epoch rows under {suite_root}; "
            "try --include-failed if many runs ended early"
        )

    numeric_cols = infer_numeric_columns(all_rows)
    selected_metrics = [m.strip() for m in args.plot_metrics.split(",") if m.strip()]
    present_plot_metrics = list_present_plot_metrics(all_rows, selected_metrics)

    # Groupings designed for the 144-run sweep factors.
    group_specs = [
        ("by_cond_q_l", ["cond_group", "Q", "L"]),
        ("by_cond", ["cond_group"]),
        ("by_q_l", ["Q", "L"]),
        ("by_freeze", ["freeze_group"]),
        ("by_vicreg", ["vicreg_group"]),
        ("by_contrastive", ["contrastive_group"]),
    ]

    grouped_outputs: Dict[str, List[Dict[str, object]]] = {}
    for label, keys in group_specs:
        grouped = groupby_epoch_stats(all_rows, keys, numeric_cols)
        grouped_outputs[label] = grouped

    # Plots from grouped means over epoch.
    for metric in present_plot_metrics:
        # 1) Main condition trend (freeze/vicreg/contrastive combo).
        by_cond = grouped_outputs.get("by_cond", [])
        if by_cond:
            make_plot(
                grouped_rows=by_cond,
                metric=metric,
                group_key="cond_group",
                out_png=plots_dir / "by_cond" / f"{metric.replace('/', '__')}.png",
                title=f"{metric} vs epoch (group: freeze+vicreg+contrastive)",
            )

        # 2) Capacity trend (Q/L combinations).
        by_q_l = grouped_outputs.get("by_q_l", [])
        if by_q_l:
            # Compact group label for q/l lines.
            relabeled: List[Dict[str, object]] = []
            for row in by_q_l:
                r2 = dict(row)
                r2["q_l_group"] = f"q{row['Q']}_L{row['L']}"
                relabeled.append(r2)
            make_plot(
                grouped_rows=relabeled,
                metric=metric,
                group_key="q_l_group",
                out_png=plots_dir / "by_q_l" / f"{metric.replace('/', '__')}.png",
                title=f"{metric} vs epoch (group: Q/L)",
            )

        # 3) Single-factor views.
        for view, group_key in [
            ("by_freeze", "freeze_group"),
            ("by_vicreg", "vicreg_group"),
            ("by_contrastive", "contrastive_group"),
        ]:
            grouped = grouped_outputs.get(view, [])
            if not grouped:
                continue
            make_plot(
                grouped_rows=grouped,
                metric=metric,
                group_key=group_key,
                out_png=plots_dir / view / f"{metric.replace('/', '__')}.png",
                title=f"{metric} vs epoch (group: {group_key})",
            )

    # Lightweight text report.
    report_lines = [
        f"suite_root: {suite_root}",
        f"out_dir: {out_dir}",
        f"runs_seen: {len(run_dirs)}",
        f"runs_done: {coverage['done_runs']}",
        f"runs_with_metrics: {coverage['with_metrics']}",
        f"runs_done_with_metrics: {coverage['done_with_metrics']}",
        f"runs_not_done_with_metrics: {coverage['not_done_with_metrics']}",
        f"runs_no_metrics: {coverage['no_metrics']}",
        f"epoch_rows: {len(all_rows)}",
        f"numeric_metrics: {len(numeric_cols)}",
        f"plot_metrics_requested: {len(selected_metrics)}",
        f"plot_metrics_present: {len(present_plot_metrics)}",
        "",
        "coverage_by_condition:",
    ]
    for cond in sorted(coverage["by_condition"].keys()):
        c = coverage["by_condition"][cond]
        report_lines.append(
            f"- {cond}: done={c['done']}/{c['total']}, metrics={c['metrics']}/{c['total']}"
        )
    report_lines += [
        "",
        "present_plot_metrics:",
    ] + [f"- {m}" for m in present_plot_metrics]
    incomplete_runs: List[str] = coverage["incomplete_runs"]
    if incomplete_runs:
        report_lines += ["", f"incomplete_runs ({len(incomplete_runs)}):"]
        report_lines += [f"- {name}" for name in incomplete_runs]
    if warnings:
        report_lines += ["", "warnings:"] + [f"- {w}" for w in warnings[:200]]
        if len(warnings) > 200:
            report_lines.append(f"- ... {len(warnings) - 200} more")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_report.txt").write_text("\n".join(report_lines) + "\n")

    print(f"Wrote epoch analysis to: {out_dir}")
    print(f"- plots:  {plots_dir}")
    print(f"- report: {out_dir / 'analysis_report.txt'}")


if __name__ == "__main__":
    main()
