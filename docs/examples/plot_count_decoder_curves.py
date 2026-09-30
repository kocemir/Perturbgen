#!/usr/bin/env python3
"""Plot epoch curves from a Gene-Query count-decoder metrics.csv.

  python docs/examples/plot_count_decoder_curves.py /path/to/count_run_dir
  python docs/examples/plot_count_decoder_curves.py --csv logs/version_0/metrics.csv --out curves
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

TIME_LABEL = {'1': '90m', '2': '6h', '3': '10h'}
TRAIN_COLOR = '#4c72b0'
VAL_COLOR = '#c44e52'


def _float(value: Optional[str]) -> float:
    if value in (None, ''):
        return float('nan')
    try:
        return float(value)
    except ValueError:
        return float('nan')


def load_epoch_table(metrics_csv: Path) -> List[Dict[str, float]]:
    by_epoch: Dict[int, Dict[str, float]] = {}
    with metrics_csv.open() as handle:
        for row in csv.DictReader(handle):
            if not row.get('epoch'):
                continue
            epoch = int(float(row['epoch']))
            merged = by_epoch.setdefault(epoch, {'epoch': float(epoch)})
            for key, value in row.items():
                if key in ('epoch', 'step') or key.endswith('_step'):
                    continue
                if value in (None, ''):
                    continue
                number = _float(value)
                if math.isnan(number):
                    continue
                canon = key.replace('_epoch', '')
                merged[canon] = number
    return [by_epoch[k] for k in sorted(by_epoch)]


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _xy(rows: List[Dict[str, float]], key: str) -> Tuple[List[int], List[float]]:
    xs, ys = [], []
    for row in rows:
        value = row.get(key)
        if value is None or math.isnan(value):
            continue
        xs.append(int(row['epoch']))
        ys.append(value)
    return xs, ys


def _plot_pair(
    rows: List[Dict[str, float]],
    stem: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> bool:
    t_x, t_y = _xy(rows, f'train/{stem}')
    v_x, v_y = _xy(rows, f'val/{stem}')
    if not t_x and not v_x:
        return False
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    if t_x:
        ax.plot(t_x, t_y, color=TRAIN_COLOR, lw=2.0, label=f'train/{stem}')
    if v_x:
        ax.plot(v_x, v_y, color=VAL_COLOR, lw=2.0, label=f'val/{stem}')
    ax.set_xlabel('epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc='best')
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)
    return True


def plot_count_curves(metrics_csv: str | Path, out_dir: str | Path) -> Path:
    csv_path = Path(metrics_csv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_epoch_table(csv_path)
    if not rows:
        raise SystemExit(f'no epoch rows in {csv_path}')

    keys = [k for k in rows[0].keys() if k != 'epoch']
    for later in rows[1:]:
        for key in later:
            if key != 'epoch' and key not in keys:
                keys.append(key)

    with (out / 'epoch_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', *keys])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in ['epoch', *keys]})

    summary = [
        ('loss', 'ZINB NLL (sum over t)'),
        ('mse', 'MSE(μ, counts)'),
        ('emd', 'EMD (10k cells)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    for ax, (stem, ylabel) in zip(axes, summary):
        t_x, t_y = _xy(rows, f'train/{stem}')
        v_x, v_y = _xy(rows, f'val/{stem}')
        if t_x:
            ax.plot(t_x, t_y, color=TRAIN_COLOR, lw=2.0, label='train')
        if v_x:
            ax.plot(v_x, v_y, color=VAL_COLOR, lw=2.0, label='val')
        ax.set_xlabel('epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(stem)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle('count decoder', y=1.04)
    _save(fig, out / 'summary_loss_mse_emd.png')

    plotted = set()
    for stem, ylabel in summary:
        if _plot_pair(rows, stem, ylabel, f'count decoder / {stem}', out / f'{stem}.png'):
            plotted.add(stem)

    time_stems = []
    for t in ('1', '2', '3'):
        stem = f'loss_t{t}'
        if any(f'train/{stem}' in r or f'val/{stem}' in r for r in rows):
            time_stems.append(stem)
            _plot_pair(
                rows,
                stem,
                'ZINB NLL',
                f'count decoder / loss {TIME_LABEL[t]}',
                out / f'loss_t{t}_{TIME_LABEL[t]}.png',
            )
            plotted.add(stem)

    if time_stems:
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        for stem in time_stems:
            tag = stem.rsplit('t', 1)[-1]
            for split, color, ls in (
                ('val', VAL_COLOR, '-'),
                ('train', TRAIN_COLOR, '--'),
            ):
                xs, ys = _xy(rows, f'{split}/{stem}')
                if xs:
                    ax.plot(
                        xs,
                        ys,
                        color=color,
                        ls=ls,
                        lw=1.8,
                        label=f'{split} {TIME_LABEL.get(tag, tag)}',
                    )
        ax.set_xlabel('epoch')
        ax.set_ylabel('ZINB NLL')
        ax.set_title('count decoder / loss by target time')
        ax.legend(frameon=False, ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3)
        _save(fig, out / 'loss_by_time.png')

    leftovers = []
    for key in keys:
        if '/' not in key:
            continue
        split, stem = key.split('/', 1)
        if split not in ('train', 'val') or stem in plotted:
            continue
        leftovers.append(stem)
    for stem in sorted(set(leftovers)):
        safe = stem.replace('/', '_')
        _plot_pair(rows, stem, stem, f'count decoder / {stem}', out / f'{safe}.png')

    print(f'count-decoder curves -> {out}  ({len(rows)} epochs)')
    return out


def _find_metrics_csv(run_dir: Path) -> Path:
    direct = run_dir / 'logs' / 'version_0' / 'metrics.csv'
    if direct.is_file():
        return direct
    matches = sorted(run_dir.glob('**/metrics.csv'))
    if not matches:
        raise SystemExit(f'no metrics.csv under {run_dir}')
    return matches[0]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='Plot count-decoder epoch curves')
    parser.add_argument('run_dir', nargs='?', help='count decoder run folder')
    parser.add_argument('--csv', default=None, help='metrics.csv path')
    parser.add_argument('--out', default=None, help='output folder (default: <run>/curves)')
    args = parser.parse_args(argv)
    if args.csv:
        csv_path = Path(args.csv)
        out = Path(args.out) if args.out else csv_path.parent / 'curves'
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        csv_path = _find_metrics_csv(run_dir)
        out = Path(args.out) if args.out else run_dir / 'curves'
    else:
        raise SystemExit('pass a run dir or --csv')
    plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 11})
    plot_count_curves(csv_path, out)


if __name__ == '__main__':
    main()
