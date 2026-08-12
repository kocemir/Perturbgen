#!/usr/bin/env python3
"""Summarise the Gene-Query JEPA toy sweep into one ranked table + CSV.

Usage:
  python docs/examples/summarize_gene_query_toy_sweep.py
  python docs/examples/summarize_gene_query_toy_sweep.py --suite-root <dir>

For every run folder it reports the honesty metric val/gene_gap_vs_copy_src:
  peak      highest value over all epochs (early spikes included)
  warm      highest value from epoch >= 5 (ignores the early spike)
  final     last epoch
  min>peak  lowest value after the peak (collapse detector)
Verdict:
  STABLE    final > +0.05 and no dip below -0.01 after the peak
            and final is close to the post-warmup best (no late collapse)
  PASS_WEAK final > +0.05 but the run dipped or decayed on the way
  FAIL      final <= +0.05
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_SUITE = (
    '/mnt/sod2-project/csb4/stuke1/perturbgen/'
    'gene_query_jepa/toy_runs/systematic_144'
)
GAP = 'val/gene_gap_vs_copy_src'
WARMUP_EPOCH = 5


def parse_env(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                out[key.strip()] = value.strip()
    return out


def load_epochs(metrics_csv: Path) -> List[Dict[str, float]]:
    """One dict per epoch (last row wins), floats only."""
    by_epoch: Dict[int, Dict[str, float]] = {}
    with metrics_csv.open() as f:
        for raw in csv.DictReader(f):
            if not raw.get(GAP):
                continue
            row: Dict[str, float] = {}
            for key, value in raw.items():
                if not value:
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    continue
            by_epoch[int(row['epoch'])] = row
    return [by_epoch[e] for e in sorted(by_epoch)]


def summarise_run(run_dir: Path) -> Optional[Dict[str, object]]:
    metrics_files = sorted(run_dir.glob('toy_logs/**/metrics.csv'))
    if not metrics_files:
        return None
    epochs = load_epochs(metrics_files[-1])
    if not epochs:
        return None

    peak = max(epochs, key=lambda r: r[GAP])
    warm_rows = [r for r in epochs if r['epoch'] >= WARMUP_EPOCH] or epochs
    warm = max(warm_rows, key=lambda r: r[GAP])
    final = epochs[-1]
    min_after_peak = min(r[GAP] for r in epochs if r['epoch'] >= peak['epoch'])

    if final[GAP] > 0.05 and min_after_peak > -0.01 \
            and abs(warm[GAP] - final[GAP]) < 0.05:
        verdict = 'STABLE'
    elif final[GAP] > 0.05:
        verdict = 'PASS_WEAK'
    else:
        verdict = 'FAIL'

    hp = parse_env(run_dir / 'hparams.env')
    return {
        'run_id': run_dir.name,
        'freeze': hp.get('FREEZE_ENCODER', ''),
        'Q': hp.get('N_QUERIES', ''),
        'L': hp.get('ENC_LAYERS', ''),
        'contr': hp.get('LAMBDA_CONTRASTIVE', ''),
        'vicreg': f"{hp.get('VICREG_VAR', '')}/{hp.get('VICREG_COV', '')}",
        'n_epochs': len(epochs),
        'peak_gap': peak[GAP],
        'peak_ep': int(peak['epoch']),
        'warm_gap': warm[GAP],
        'warm_ep': int(warm['epoch']),
        'final_gap': final[GAP],
        'min_after_peak': min_after_peak,
        'final_gene_loss': final.get('val/gene_loss', float('nan')),
        'verdict': verdict,
        'done': (run_dir / 'DONE').is_file(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite-root', default=DEFAULT_SUITE)
    parser.add_argument('--csv-out', default='')
    args = parser.parse_args()

    suite = Path(args.suite_root)
    if not suite.is_dir():
        raise SystemExit(f'suite root not found: {suite}')

    rows = []
    for run_dir in sorted(suite.iterdir()):
        if run_dir.is_dir():
            row = summarise_run(run_dir)
            if row:
                rows.append(row)
    if not rows:
        raise SystemExit(f'no finished runs with metrics under {suite}')

    rows.sort(key=lambda r: r['final_gap'], reverse=True)

    print(
        f'{"run_id":<34} {"warm":>8} {"final":>8} {"min>pk":>8} '
        f'{"ep":>3} {"verdict":>10}'
    )
    print('-' * 78)
    for r in rows:
        print(
            f'{r["run_id"]:<34} '
            f'{r["warm_gap"]:>+8.4f} '
            f'{r["final_gap"]:>+8.4f} '
            f'{r["min_after_peak"]:>+8.4f} '
            f'{r["n_epochs"]:>3} '
            f'{r["verdict"]:>10}'
        )

    out = Path(args.csv_out) if args.csv_out else suite / 'summary.csv'
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n{len(rows)} runs -> {out}')


if __name__ == '__main__':
    main()
