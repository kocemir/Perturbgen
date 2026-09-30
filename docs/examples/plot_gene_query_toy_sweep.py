#!/usr/bin/env python3
"""Gene-Query JEPA — grouped mean±std curves (same layout as jepa_toy_res/plots).

Usage:
  python docs/examples/plot_gene_query_toy_sweep.py
  python docs/examples/plot_gene_query_toy_sweep.py --suite-root <dir> --out-dir <dir>

Only runs with metrics.csv are plotted. L in by_q_l is whichever of encoder /
predictor depth actually varies in the suite.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_SUITE = Path(
    '/mnt/sod2-project/csb4/stuke1/perturbgen/'
    'gene_query_jepa/toy_runs/vic_fz5thenU_enc5e-6_q128-256_L1-3_ep20_bpt8'
)
DEFAULT_OUT = Path(
    '/home/stuke1/perturbgen/Perturbgen/docs/examples/'
    'jepa_toy_res/vic_fz5thenU_enc5e-6_q128-256_L1-3_ep20_bpt8/plots'
)

METRICS = [
    'val/gene_gap_vs_copy_src',
    'train/gene_gap_vs_copy_src',
    'val/gene_loss',
    'train/gene_loss',
    'val/gene_mse',
    'train/gene_mse',
    'val/cell_loss',
    'train/cell_loss',
    'val/contrastive_loss',
]

GROUP_SPECS: List[Tuple[str, str, str]] = [
    ('by_cond', 'cond', 'freeze+vicreg+contrastive'),
    ('by_freeze', 'freeze_group', 'freeze'),
    ('by_vicreg', 'vicreg_group', 'vicreg'),
    ('by_contrastive', 'contrastive_group', 'contrastive'),
    ('by_q_l', 'q_l', 'Q/L'),
]


def parse_env(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                out[key.strip()] = value.strip()
    return out


def on_off(flag: bool) -> str:
    return 'on' if flag else 'off'


def labels_from_hparams(hp: Dict[str, str], run_id: str) -> Dict[str, str]:
    freeze = hp.get('FREEZE_ENCODER', '').lower() in {'1', 'true', 'yes'}
    if not hp.get('FREEZE_ENCODER'):
        freeze = run_id.startswith('fzT') or run_id.startswith('fz5')
    vic = float(hp.get('VICREG_VAR') or 0) > 0
    if 'VICREG_VAR' not in hp:
        vic = '_vic1_' in f'_{run_id}_'
    contr = float(hp.get('LAMBDA_CONTRASTIVE') or 0) > 0
    q = hp.get('N_QUERIES') or ''
    enc_l = hp.get('ENC_LAYERS') or ''
    pred_l = hp.get('PREDICTOR_LAYERS') or ''
    if not q:
        m = re.search(r'_q(\d+)_', f'_{run_id}_')
        q = m.group(1) if m else '?'
    if not enc_l:
        m = re.search(r'encL(\d+)', run_id) or re.search(r'_L(\d+)$', run_id)
        enc_l = m.group(1) if m else '?'
    if not pred_l:
        m = re.search(r'_predL(\d+)', run_id)
        pred_l = m.group(1) if m else ''
    fz = on_off(freeze)
    vic_s = on_off(vic)
    ctr_s = on_off(contr)
    return {
        'freeze_group': fz,
        'vicreg_group': vic_s,
        'contrastive_group': ctr_s,
        'cond': f'fz_{fz}__vic_{vic_s}__ctr_{ctr_s}',
        'q': q,
        'enc_l': enc_l,
        'pred_l': pred_l or '?',
        'run_short': f"q{q}_L{enc_l or pred_l}",
        'run_id': run_id,
    }


def load_epoch_metrics(metrics_csv: Path) -> Dict[int, Dict[str, float]]:
    by_epoch: Dict[int, Dict[str, float]] = defaultdict(dict)
    with metrics_csv.open() as f:
        for raw in csv.DictReader(f):
            if not raw.get('epoch'):
                continue
            epoch = int(float(raw['epoch']))
            for key in METRICS:
                value = raw.get(key) or ''
                if value == '':
                    continue
                try:
                    by_epoch[epoch][key] = float(value)
                except ValueError:
                    continue
    return dict(by_epoch)


def collect_runs(suite: Path, *, done_only: bool) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run_dir in sorted(p for p in suite.iterdir() if p.is_dir()):
        if done_only and not (run_dir / 'DONE').is_file():
            continue
        metrics_files = sorted(run_dir.glob('logs/**/metrics.csv')) or sorted(
            run_dir.glob('**/metrics.csv')
        )
        if not metrics_files:
            continue
        series = load_epoch_metrics(metrics_files[-1])
        if not series:
            continue
        hp = parse_env(run_dir / 'hparams.env')
        labels = labels_from_hparams(hp, run_dir.name)
        rows.append({'series': series, 'done': (run_dir / 'DONE').is_file(), **labels})
    if not rows:
        return rows
    encs = {str(r['enc_l']) for r in rows}
    preds = {str(r['pred_l']) for r in rows}
    use_enc = len(encs) >= len(preds)
    for row in rows:
        depth = row['enc_l'] if use_enc else row['pred_l']
        row['q_l'] = f"q{row['q']}_L{depth}"
        row['run_short'] = str(row['q_l'])
    return rows


def group_mean_std(
    runs: Iterable[Dict[str, object]],
    group_key: str,
    metric: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    values: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        label = str(run[group_key])
        series: Dict[int, Dict[str, float]] = run['series']  # type: ignore[assignment]
        for epoch, row in series.items():
            if metric in row:
                values[label][epoch].append(row[metric])
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, by_ep in values.items():
        epochs = np.array(sorted(by_ep))
        mean = np.array([np.mean(by_ep[int(e)]) for e in epochs])
        std = np.array([np.std(by_ep[int(e)], ddof=0) for e in epochs])
        out[label] = (epochs, mean, std)
    return out


def metric_filename(metric: str) -> str:
    return metric.replace('/', '__') + '.png'


def style_gap_axes(ax, metric: str) -> None:
    if metric.endswith('gene_gap_vs_copy_src'):
        ax.axhline(0.0, color='#666666', lw=0.8)
        ax.axhline(0.05, color='#666666', ls='--', lw=0.8)
        ax.axvline(5, color='#999999', ls=':', lw=0.8)


def plot_group(
    grouped: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    metric: str,
    group_title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=120)
    cmap = plt.get_cmap('tab10')
    labels = sorted(grouped)
    for i, label in enumerate(labels):
        epochs, mean, std = grouped[label]
        color = cmap(i % 10)
        ax.plot(epochs, mean, color=color, lw=1.8, label=label)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs epoch (group: {group_title})')
    style_gap_axes(ax, metric)
    ax.grid(True, color='#d0d0d0', lw=0.6)
    ax.set_axisbelow(True)
    ncol = 2 if len(labels) > 8 else 1
    ax.legend(loc='best', fontsize=8, framealpha=0.92, ncol=ncol)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_summary_heatmap(summary_csv: Path, out_path: Path) -> None:
    if not summary_csv.is_file():
        return
    with summary_csv.open() as f:
        rows = list(csv.DictReader(f))
    if not rows or 'Q' not in rows[0] or 'L' not in rows[0]:
        return
    qs = sorted({int(r['Q']) for r in rows if r.get('Q')})
    ls = sorted({int(r['L']) for r in rows if r.get('L')})
    if not qs or not ls:
        return
    grid = np.full((len(ls), len(qs)), np.nan)
    annot = [['' for _ in qs] for _ in ls]
    for r in rows:
        try:
            q_i = qs.index(int(r['Q']))
            l_i = ls.index(int(r['L']))
        except (ValueError, KeyError):
            continue
        gap = float(r['final_gap'])
        grid[l_i, q_i] = gap
        tag = '' if r.get('done', 'True') in {'True', 'true', '1'} else '*'
        annot[l_i][q_i] = f'{gap:+.3f}{tag}'

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=130)
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap='RdYlGn', vmin=-0.02, vmax=0.28, origin='lower', aspect='auto')
    ax.set_xticks(range(len(qs)), [str(q) for q in qs])
    ax.set_yticks(range(len(ls)), [str(l) for l in ls])
    ax.set_xlabel('Q (queries)')
    ax.set_ylabel('L (encoder layers)')
    ax.set_title(f'final val/gene_gap_vs_copy_src\n{summary_csv.parent.name}')
    for i in range(len(ls)):
        for j in range(len(qs)):
            if annot[i][j]:
                ax.text(j, i, annot[i][j], ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='final gap')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_suite(suite: Path, out_dir: Path, *, by_run: bool, done_only: bool) -> int:
    runs = collect_runs(suite, done_only=done_only)
    if not runs:
        print(f'skip (no metrics): {suite}')
        return 0
    print(f'plotting {len(runs)} runs from {suite.name}')
    print(f'  out: {out_dir}')
    for folder, column, title in GROUP_SPECS:
        n_groups = len({r[column] for r in runs})
        print(f'  {folder}: {n_groups} groups')
        for metric in METRICS:
            grouped = group_mean_std(runs, column, metric)
            plot_group(
                grouped,
                metric,
                title,
                out_dir / folder / metric_filename(metric),
            )
    if by_run:
        print(f'  by_run: {len(runs)} series')
        for metric in METRICS:
            grouped = group_mean_std(runs, 'run_short', metric)
            plot_group(
                grouped,
                metric,
                'run',
                out_dir / 'by_run' / metric_filename(metric),
            )
    summary = suite / 'summary.csv'
    local_summary = out_dir.parent / 'summary.csv'
    if local_summary.is_file():
        summary = local_summary
    plot_summary_heatmap(summary, out_dir / 'summary_heatmap_final_gap.png')
    vic_out = out_dir.parent if out_dir.name == 'plots' else out_dir
    plot_vic_grid(suite, vic_out)
    return len(runs)


def _run_label(name: str) -> Tuple[str, str, int, int]:
    q_m = re.search(r'q(\d+)', name)
    q = q_m.group(1) if q_m else '?'
    enc_m = re.search(r'encL(\d+)', name)
    pred_m = re.search(r'predL(\d+)', name)
    enc_l = int(enc_m.group(1)) if enc_m else -1
    pred_l = int(pred_m.group(1)) if pred_m else -1
    if enc_l < 0:
        try:
            enc_l = int(name.split('_L')[-1])
        except ValueError:
            enc_l = -1
    if pred_l >= 0:
        return f'encL{enc_l} predL{pred_l}', q, enc_l, pred_l
    return f'Q{q} encL{enc_l}', q, enc_l, pred_l


def _stage_edges(suite: Path) -> List[int]:
    edges = [5]
    for hp_path in sorted(suite.glob('*/hparams.env')):
        sched = parse_env(hp_path).get('ENCODER_LR_SCHEDULE', '')
        if not sched:
            continue
        t = 0
        out: List[int] = []
        for part in sched.split(','):
            if ':' not in part:
                continue
            try:
                t += int(part.rsplit(':', 1)[-1])
            except ValueError:
                continue
            out.append(t)
        if out:
            return out[:-1] or edges
    return edges


def _stage_lines(ax, suite: Path) -> None:
    for i, x in enumerate(_stage_edges(suite)):
        ax.axvline(x, color='0.7' if i == 0 else '0.85', lw=1, ls='--' if i == 0 else ':')


def plot_vic_grid(suite: Path, out: Path) -> None:
    """Per-run honesty / cosine PNG set used for the Q×L vicreg suites."""
    runs: Dict[str, Dict[int, Dict[str, float]]] = {}
    for run_dir in sorted(p for p in suite.iterdir() if p.is_dir()):
        mets = sorted(run_dir.glob('logs/**/metrics.csv'))
        if not mets:
            continue
        by_ep: Dict[int, Dict[str, float]] = {}
        with mets[-1].open() as f:
            for raw in csv.DictReader(f):
                if not raw.get('epoch'):
                    continue
                ep = int(float(raw['epoch']))
                row = by_ep.setdefault(ep, {})
                for key, value in raw.items():
                    if key == 'epoch' or not value:
                        continue
                    try:
                        row[key] = float(value)
                    except ValueError:
                        continue
        if by_ep:
            runs[run_dir.name] = by_ep
    if not runs:
        return

    max_ep = max(max(series) for series in runs.values())
    finished = {k: v for k, v in runs.items() if max(v) >= max_ep - 1}
    if not finished:
        finished = runs
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
    })

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, by_ep in runs.items():
        lab, q, _enc, _pred = _run_label(name)
        eps = sorted(e for e in by_ep if 'val/gene_gap_vs_copy_src' in by_ep[e])
        if not eps:
            continue
        ax.plot(
            eps,
            [by_ep[e]['val/gene_gap_vs_copy_src'] for e in eps],
            '-' if q == '128' else '--',
            label=lab,
            linewidth=1.8,
        )
    _stage_lines(ax, suite)
    ax.axhline(0, color='0.4', lw=1, label='copy-src')
    ax.axhline(0.05, color='0.5', lw=1, ls=':', label='STABLE 0.05')
    ax.set_xlabel('epoch')
    ax.set_ylabel('val/gene_gap_vs_copy_src')
    ax.set_title(f'{suite.name}: honesty gap vs epoch')
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'gap_vs_epoch_all.png', dpi=150)
    plt.close(fig)

    by_L: Dict[int, List[Tuple[List[int], List[float]]]] = defaultdict(list)
    by_pred: Dict[int, List[Tuple[List[int], List[float]]]] = defaultdict(list)
    by_Q: Dict[str, List[Tuple[List[int], List[float]]]] = defaultdict(list)
    for name, by_ep in finished.items():
        _lab, q, depth, pred = _run_label(name)
        eps = sorted(e for e in by_ep if 'val/gene_gap_vs_copy_src' in by_ep[e])
        if not eps:
            continue
        ys = [by_ep[e]['val/gene_gap_vs_copy_src'] for e in eps]
        by_L[depth].append((eps, ys))
        if pred >= 0:
            by_pred[pred].append((eps, ys))
        by_Q[q].append((eps, ys))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for depth in sorted(by_L):
        series = by_L[depth]
        ep_set = sorted({ep for eps, _ys in series for ep in eps})
        means = [
            sum(ys[eps.index(e)] for eps, ys in series if e in eps)
            / sum(1 for eps, _ys in series if e in eps)
            for e in ep_set
        ]
        ax.plot(ep_set, means, lw=2, label=f'enc L{depth} (mean over rest)')
    _stage_lines(ax, suite)
    ax.axhline(0.05, color='0.5', lw=1, ls=':')
    ax.set_xlabel('epoch')
    ax.set_ylabel('val/gene_gap_vs_copy_src')
    ax.set_title('Honesty gap by encoder depth')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'gap_vs_epoch_by_encL.png', dpi=150)
    plt.close(fig)

    if by_pred:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for pred in sorted(by_pred):
            series = by_pred[pred]
            ep_set = sorted({ep for eps, _ys in series for ep in eps})
            means = [
                sum(ys[eps.index(e)] for eps, ys in series if e in eps)
                / sum(1 for eps, _ys in series if e in eps)
                for e in ep_set
            ]
            ax.plot(ep_set, means, lw=2, label=f'pred L{pred} (mean over enc L)')
        _stage_lines(ax, suite)
        ax.axhline(0.05, color='0.5', lw=1, ls=':')
        ax.set_xlabel('epoch')
        ax.set_ylabel('val/gene_gap_vs_copy_src')
        ax.set_title('Honesty gap by predictor depth')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'gap_vs_epoch_by_predL.png', dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for q in sorted(by_Q):
        series = by_Q[q]
        ep_set = sorted({ep for eps, _ys in series for ep in eps})
        means = [
            sum(ys[eps.index(e)] for eps, ys in series if e in eps)
            / sum(1 for eps, _ys in series if e in eps)
            for e in ep_set
        ]
        ax.plot(ep_set, means, lw=2, label=f'Q {q} (mean over L)')
    _stage_lines(ax, suite)
    ax.axhline(0.05, color='0.5', lw=1, ls=':')
    ax.set_xlabel('epoch')
    ax.set_ylabel('val/gene_gap_vs_copy_src')
    ax.set_title('Honesty gap by query count')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'gap_vs_epoch_by_Q.png', dpi=150)
    plt.close(fig)

    overlays = [
        ('val/gene_loss', 'val/gene_loss', 'gene loss vs epoch', 'gene_loss_vs_epoch.png'),
        ('val/gene_cos_pred', 'val/gene_cos_pred', 'Predicted vs EMA target cosine', 'gene_cos_pred_vs_epoch.png'),
        ('val/gene_gap_vs_static', 'val/gene_gap_vs_static', 'Honesty gap vs static gene identity', 'gene_gap_vs_static.png'),
        ('val/cell_cos_pred', 'val/cell_cos_pred', 'Cell pooled cosine (pred vs EMA target)', 'cell_cos_pred_vs_epoch.png'),
        ('val/cell_loss', 'val/cell_loss', 'Cell loss (1 - cosine)', 'cell_loss_vs_epoch.png'),
        ('val/vicreg_var', 'val/vicreg_var', 'VICReg variance term', 'vicreg_var_vs_epoch.png'),
        ('val/vicreg_cov', 'val/vicreg_cov', 'VICReg covariance term', 'vicreg_cov_vs_epoch.png'),
        ('val/total_loss', 'val/total_loss', 'Total weighted loss', 'total_loss_vs_epoch.png'),
        ('val/gene_mse', 'val/gene_mse', 'Present-query gene MSE (not in objective)', 'gene_mse_vs_epoch.png'),
        ('train/gene_mse', 'train/gene_mse', 'Present-query train gene MSE (not in objective)', 'train_gene_mse_vs_epoch.png'),
        ('train/gene_loss', 'train/gene_loss', 'train gene loss vs epoch', 'train_gene_loss_vs_epoch.png'),
        ('train/cell_loss', 'train/cell_loss', 'train cell loss vs epoch', 'train_cell_loss_vs_epoch.png'),
        ('train/gene_gap_vs_copy_src', 'train/gene_gap_vs_copy_src', 'train honesty gap vs epoch', 'train_gap_vs_epoch.png'),
    ]
    for metric, ylabel, title, fname in overlays:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name, by_ep in finished.items():
            lab, q, _enc, _pred = _run_label(name)
            eps = sorted(e for e in by_ep if metric in by_ep[e])
            if not eps:
                continue
            ax.plot(
                eps,
                [by_ep[e][metric] for e in eps],
                '-' if q == '128' else '--',
                label=lab,
                linewidth=1.8,
            )
        _stage_lines(ax, suite)
        if 'gap' in metric:
            ax.axhline(0, color='0.4', lw=1)
        ax.set_xlabel('epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{suite.name}: {title}')
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)

    pairs = [
        ('gene_mse', 'Present-query gene MSE (not in objective)', 'train_val_gene_mse_vs_epoch.png'),
        ('gene_loss', 'gene loss', 'train_val_gene_loss_vs_epoch.png'),
        ('cell_loss', 'cell loss', 'train_val_cell_loss_vs_epoch.png'),
        ('gene_gap_vs_copy_src', 'honesty gap vs copy-src', 'train_val_gap_vs_epoch.png'),
    ]
    for stem, title, fname in pairs:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name, by_ep in finished.items():
            lab, q, _enc, _pred = _run_label(name)
            ls = '-' if q == '128' else '--'
            for split, lw, alpha in (('val', 1.8, 1.0), ('train', 1.2, 0.7)):
                key = f'{split}/{stem}'
                eps = sorted(e for e in by_ep if key in by_ep[e])
                if not eps:
                    continue
                ax.plot(
                    eps,
                    [by_ep[e][key] for e in eps],
                    ls,
                    lw=lw,
                    alpha=alpha,
                    label=f'{lab} {split}',
                )
        _stage_lines(ax, suite)
        if 'gap' in stem:
            ax.axhline(0, color='0.4', lw=1)
        ax.set_xlabel('epoch')
        ax.set_ylabel(stem)
        ax.set_title(f'{suite.name}: train/val {title}')
        ax.legend(ncol=2, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)

    n_runs = len(finished)
    cols = 3
    rows = (n_runs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.1 * rows), sharex=True)
    axes_list = list(np.array(axes).ravel()) if n_runs > 1 else [axes]
    for i, (name, by_ep) in enumerate(sorted(finished.items())):
        lab, _q, _enc, _pred = _run_label(name)
        ax = axes_list[i]
        for split in ('train', 'val'):
            key = f'{split}/gene_mse'
            eps = sorted(e for e in by_ep if key in by_ep[e])
            if eps:
                ax.plot(eps, [by_ep[e][key] for e in eps], lw=1.6, label=split)
        _stage_lines(ax, suite)
        ax.set_title(lab, fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
        if i >= n_runs - cols:
            ax.set_xlabel('epoch')
        ax.set_ylabel('gene_mse')
    for j in range(n_runs, len(axes_list)):
        axes_list[j].set_visible(False)
    fig.suptitle(f'{suite.name}: train vs val gene MSE')
    fig.tight_layout()
    fig.savefig(out / 'train_val_gene_mse_grid.png', dpi=150)
    plt.close(fig)

    winner_name = max(
        finished,
        key=lambda n: finished[n][max(finished[n])]['val/gene_gap_vs_copy_src'],
    )
    winner = finished[winner_name]
    eps = sorted(winner)
    wlab, _q, _enc, _pred = _run_label(winner_name)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    panels = [
        (axes[0, 0], 'val/gene_cos_pred', 'gene cos(pred, EMA tgt)\n(present queries)'),
        (axes[0, 1], 'val/gene_loss', 'gene loss = 1 - cos (present queries)'),
        (axes[1, 0], 'val/gene_gap_vs_copy_src', 'gap vs copy-src (shared only)'),
        (axes[1, 1], 'val/gene_gap_vs_static', 'gap vs static identity'),
    ]
    for ax, key, title in panels:
        ax.plot(eps, [winner[e].get(key, float('nan')) for e in eps], color='C2', lw=2, label=wlab)
        _stage_lines(ax, suite)
        if 'gap' in key:
            ax.axhline(0, color='0.4', lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(key.replace('val/', ''))
        ax.legend(fontsize=8)
    axes[1, 0].set_xlabel('epoch')
    axes[1, 1].set_xlabel('epoch')
    fig.suptitle(f'Decompose {wlab} — {suite.name}', y=1.01)
    fig.tight_layout()
    fig.savefig(out / f'decompose_{wlab.replace(" ", "_").lower()}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    n = len(finished)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), sharex=True)
    axes_list = list(axes.ravel()) if n > 1 else [axes]
    for i, (name, by_ep) in enumerate(sorted(finished.items())):
        lab, _q, _enc, _pred = _run_label(name)
        ax = axes_list[i]
        e = sorted(by_ep)
        ax.plot(e, [by_ep[x]['val/gene_cos_pred'] for x in e], lw=1.8, label='cos(pred,tgt)')
        ax.plot(e, [by_ep[x]['val/gene_gap_vs_copy_src'] for x in e], lw=1.8, label='gap vs copy-src')
        ax.plot(e, [by_ep[x]['val/gene_gap_vs_static'] for x in e], lw=1.4, ls='--', label='gap vs static')
        _stage_lines(ax, suite)
        ax.axhline(0, color='0.4', lw=0.8)
        ax.set_title(lab)
        if i == 0:
            ax.legend(fontsize=7)
        if i >= n - cols:
            ax.set_xlabel('epoch')
    for j in range(n, len(axes_list)):
        axes_list[j].set_visible(False)
    fig.suptitle(f'{suite.name}: prediction cosine vs honesty gaps')
    fig.tight_layout()
    fig.savefig(out / 'decompose_all_runs.png', dpi=150)
    plt.close(fig)
    print(f'  vic-grid PNGs -> {out}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite-root', type=Path, default=None)
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument(
        '--by-run',
        action='store_true',
        default=True,
        help='plot each run as its own series (default on)',
    )
    parser.add_argument('--no-by-run', action='store_false', dest='by_run')
    parser.add_argument(
        '--include-incomplete',
        action='store_true',
        help='also plot runs that have metrics.csv but no DONE file',
    )
    args = parser.parse_args()

    if args.suite_root is not None:
        suite = args.suite_root
        out = args.out_dir or (Path(
            '/home/stuke1/perturbgen/Perturbgen/docs/examples/jepa_toy_res'
        ) / suite.name / 'plots')
        if not suite.is_dir():
            raise SystemExit(f'suite root not found: {suite}')
        n = plot_suite(suite, out, by_run=args.by_run, done_only=not args.include_incomplete)
        if n == 0:
            raise SystemExit(f'no runs with metrics under {suite}')
        print('done')
        return

    n = plot_suite(
        DEFAULT_SUITE,
        args.out_dir or DEFAULT_OUT,
        by_run=args.by_run,
        done_only=not args.include_incomplete,
    )
    if n == 0:
        raise SystemExit(f'no runs with metrics under {DEFAULT_SUITE}')
    print('done')


if __name__ == '__main__':
    main()
