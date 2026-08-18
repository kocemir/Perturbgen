"""Gene-Query JEPA — one training entry for toy and full LPS.

Honesty metric: gene_gap_vs_copy_src must be > 0 (val on toy, train on
full-with-no-split). Index: docs/examples/GENE_QUERY_JEPA.md

USAGE
-----
  # toy: N mini-batches from EVERY cell type; train=val=same cells
  python docs/examples/train_gene_query_jepa.py --data toy \\
      --batches-per-type 2 --batch-size 16

  # full: every LPS cell (paper-style, no hold-out)
  python docs/examples/train_gene_query_jepa.py --data full --split false

  python docs/examples/train_gene_query_jepa.py --help
"""

from __future__ import annotations

import os

# Headless-safe defaults BEFORE heavy imports (SSH / screen / no X11).
os.environ.pop('DISPLAY', None)
os.environ.pop('WAYLAND_DISPLAY', None)
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('WANDB_DISABLED', 'true')
os.environ.setdefault('WANDB_DISABLE_CODE', 'true')
os.environ.setdefault('WANDB_CONSOLE', 'off')
os.environ.setdefault('HWLOC_COMPONENTS', '-gl')
os.environ.setdefault('HWLOC_GL_LINUX_NVIDIA_DISABLE', '1')

import argparse
import csv
import json
import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

WORKSPACE = os.environ.get('WORKSPACE', '/home/stuke1/perturbgen')
TOKENIZED = f'{WORKSPACE}/T_perturb/tokenized_data/LPS_all_tps_2k'
PRETRAIN_CKPT = (
    f'{WORKSPACE}/Perturbgen/pretraining_cohort/'
    '20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_'
    'ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt'
)
SOD2 = '/mnt/sod2-project/csb4/stuke1/perturbgen'
DEFAULT_FULL_ROOT = f'{SOD2}/T_perturb/res/jepa_gene_query_full_atlas'
DEFAULT_TOY_ROOT = f'{SOD2}/gene_query_jepa/toy_runs'
GAP = 'gene_gap_vs_copy_src'


def _gpu_ids(value: str) -> List[int]:
    ids = [int(part.strip()) for part in str(value).split(',') if part.strip()]
    if not ids:
        raise argparse.ArgumentTypeError('expected at least one GPU id')
    return ids


def _bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in ('true', '1', 'yes', 'y'):
        return True
    if text in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'expected true/false, got {value!r}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Train Gene-Query JEPA on LPS. --data toy subsets by cell type; '
            '--data full uses every cell. Honesty: gene_gap_vs_copy_src > 0.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group('data')
    data.add_argument(
        '--data',
        choices=('toy', 'full'),
        default='toy',
        help='toy = N batches per cell type; full = all LPS cells',
    )
    data.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='dataloader batch size (also the toy class unit)',
    )
    data.add_argument(
        '--batches-per-type',
        type=int,
        default=2,
        help='toy only: mini-batches kept per cell type. '
        'cells/class = batches-per-type × batch-size',
    )
    data.add_argument(
        '--class-key',
        default='cell_type_harmonized',
        help='toy only: metadata column used as the class',
    )
    data.add_argument(
        '--split',
        type=_bool,
        default=False,
        help='full only: if true, use --split-path hold-out; if false, all cells',
    )
    data.add_argument(
        '--split-path',
        default=None,
        help='full + --split true: pickle with train/val/test_indices',
    )
    data.add_argument('--tokenized', default=TOKENIZED)
    data.add_argument('--encoder-path', default=PRETRAIN_CKPT)

    model = parser.add_argument_group('model')
    model.add_argument('--freeze-encoder', type=_bool, default=False)
    model.add_argument('--encoder-layers', type=int, default=3)
    model.add_argument('--predictor-layers', type=int, default=3)
    model.add_argument('--n-queries', type=int, default=128)
    model.add_argument('--query-mode', choices=('mixed', 'all_shared'), default='mixed')
    model.add_argument('--shared-max-queries', type=int, default=0)
    model.add_argument('--lambda-gene', type=float, default=1.0)
    model.add_argument('--lambda-cell', type=float, default=0.1)
    model.add_argument('--lambda-contrastive', type=float, default=0.0)
    model.add_argument('--contrastive-tau', type=float, default=0.1)
    model.add_argument('--vicreg-var', type=float, default=1.0)
    model.add_argument('--vicreg-cov', type=float, default=0.04)
    model.add_argument('--ema-decay', type=float, default=0.996)
    model.add_argument('--lr', type=float, default=1e-4)
    model.add_argument('--weight-decay', type=float, default=1e-4)

    train = parser.add_argument_group('train')
    train.add_argument('--epochs', type=int, default=None, help='default: toy 40, full 5')
    train.add_argument(
        '--gpu',
        type=_gpu_ids,
        default=[0],
        help='comma-separated GPU ids, e.g. 6,7',
    )
    train.add_argument('--num-workers', type=int, default=2)
    train.add_argument('--seed', type=int, default=0)
    train.add_argument(
        '--early-stop',
        type=_bool,
        default=None,
        help='default: toy true, full false',
    )
    train.add_argument('--early-stop-patience', type=int, default=6)
    train.add_argument('--early-stop-min-delta', type=float, default=0.005)
    train.add_argument(
        '--output-root',
        default=None,
        help='parent folder; default full=jepa_gene_query_full_atlas, toy=toy_runs',
    )
    train.add_argument(
        '--output-dir',
        default=None,
        help='exact run folder (sweep). If omitted: <output-root>/<spec-name>/',
    )

    ckpt = parser.add_argument_group('checkpoint')
    ckpt.add_argument('--save-ckpt', type=_bool, default=True)
    ckpt.add_argument('--ckpt-every', type=int, default=1)
    ckpt.add_argument('--ckpt-top-k', type=int, default=-1, help='-1 keeps all')
    ckpt.add_argument('--ckpt-save-last', type=_bool, default=True)
    ckpt.add_argument('--ckpt-weights-only', type=_bool, default=False)
    return parser


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.epochs is None:
        args.epochs = 40 if args.data == 'toy' else 5
    if args.early_stop is None:
        args.early_stop = args.data == 'toy'
    if args.data == 'toy' and args.batches_per_type < 1:
        raise SystemExit('--batches-per-type must be >= 1 for --data toy')
    if args.data == 'full' and args.split and not args.split_path:
        raise SystemExit('--split true requires --split-path')
    return args


def spec_run_name(args: argparse.Namespace, stamp: str) -> str:
    """Filesystem-safe folder name with the knobs that distinguish runs."""
    freeze = 'fzT' if args.freeze_encoder else 'fzF'
    contr = (
        f'contr{args.lambda_contrastive:g}'
        if args.lambda_contrastive > 0
        else 'contr0'
    )
    if args.vicreg_var > 0 or args.vicreg_cov > 0:
        vic = f'vic{args.vicreg_var:g}_{args.vicreg_cov:g}'
    else:
        vic = 'vic0'
    split_tag = 'splitT' if args.data == 'full' and args.split else 'splitF'
    name = (
        f'{freeze}'
        f'_encL{args.encoder_layers}'
        f'_predL{args.predictor_layers}'
        f'_q{args.n_queries}'
        f'_{args.query_mode}'
        f'_lg{args.lambda_gene:g}'
        f'_lc{args.lambda_cell:g}'
        f'_{contr}_{vic}'
        f'_bs{args.batch_size}'
        f'_ep{args.epochs}'
        f'_lr{args.lr:g}'
        f'_seed{args.seed}'
        f'_{split_tag}'
    )
    if args.data == 'toy':
        name += f'_bpt{args.batches_per_type}'
    return f'{name}_{stamp}'


def default_output_root(args: argparse.Namespace) -> Path:
    if args.output_root:
        return Path(args.output_root)
    return Path(DEFAULT_TOY_ROOT if args.data == 'toy' else DEFAULT_FULL_ROOT)


def resolve_run_dir(args: argparse.Namespace, stamp: str) -> Path:
    """Sweep passes --output-dir (exact leaf). Otherwise parent/spec-name."""
    if args.output_dir:
        return Path(args.output_dir)
    return default_output_root(args) / spec_run_name(args, stamp)


def build_specs(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    run_name: str,
    stamp: str,
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    honesty_metric: Optional[str] = None,
    last_gap: Optional[float] = None,
    metrics_csv: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        'run_name': run_name,
        'created_at': stamp,
        'honesty_metric': honesty_metric,
        'honesty_rule': 'PASS if last gene_gap_vs_copy_src > 0.05',
        'last_gene_gap_vs_copy_src': last_gap,
        'metrics_csv': metrics_csv,
        'n_train': n_train,
        'n_val': n_val,
        'paths': {
            'run_dir': str(run_dir),
            'tokenized': args.tokenized,
            'encoder_path': args.encoder_path,
            'split_path': args.split_path,
        },
        'data': {
            'data': args.data,
            'batch_size': args.batch_size,
            'batches_per_type': args.batches_per_type if args.data == 'toy' else None,
            'cells_per_type': (
                args.batches_per_type * args.batch_size
                if args.data == 'toy'
                else None
            ),
            'class_key': args.class_key,
            'split': bool(args.split) if args.data == 'full' else True,
            'pred_tps': [1, 2, 3],
        },
        'model': {
            'jepa_encoder': 'scmaskgit',
            'freeze_encoder': args.freeze_encoder,
            'encoder_layers': args.encoder_layers,
            'predictor_layers': args.predictor_layers,
            'n_queries': args.n_queries,
            'query_mode': args.query_mode,
            'shared_max_queries': args.shared_max_queries,
            'frac_shared': 0.5,
            'frac_tgt_only': 0.5,
            'absent_queries': False,
            'query_time': 'dct_k2_shared_shift_induced_gain',
            'lambda_gene': args.lambda_gene,
            'lambda_cell': args.lambda_cell,
            'lambda_contrastive': args.lambda_contrastive,
            'contrastive_tau': args.contrastive_tau,
            'vicreg_var': args.vicreg_var,
            'vicreg_cov': args.vicreg_cov,
            'ema_decay': args.ema_decay,
            'normalize_latents': True,
        },
        'train': {
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'gpu': args.gpu,
            'seed': args.seed,
            'early_stop': args.early_stop,
            'early_stop_patience': args.early_stop_patience,
            'early_stop_min_delta': args.early_stop_min_delta,
        },
        'checkpoint': {
            'save_ckpt': args.save_ckpt,
            'ckpt_every': args.ckpt_every,
            'ckpt_top_k': args.ckpt_top_k,
            'ckpt_save_last': args.ckpt_save_last,
            'ckpt_weights_only': args.ckpt_weights_only,
        },
    }


def write_specs(run_dir: Path, specs: Dict[str, Any]) -> Path:
    path = run_dir / 'specs.json'
    path.write_text(json.dumps(specs, indent=2, sort_keys=True) + '\n')
    return path


def pick_toy_indices(
    cell_types: Sequence[str],
    cells_per_type: int,
) -> List[int]:
    """First ``cells_per_type`` row indices of every class (deterministic)."""
    indices_by_type: Dict[str, List[int]] = {}
    for row_index, cell_type in enumerate(cell_types):
        bucket = indices_by_type.setdefault(cell_type, [])
        if len(bucket) < cells_per_type:
            bucket.append(row_index)

    chosen: List[int] = []
    print(f'\nToy roster ({cells_per_type} cells wanted per type):')
    for cell_type, indices in sorted(indices_by_type.items()):
        if len(indices) < cells_per_type:
            print(f'  {cell_type:<40} {len(indices):>4} cells  -> SKIPPED (too few)')
            continue
        print(f'  {cell_type:<40} {len(indices):>4} cells  -> used')
        chosen.extend(indices)
    print(f'  total toy cells: {len(chosen)}\n')
    if not chosen:
        raise SystemExit('Toy roster is empty: lower --batches-per-type or --batch-size.')
    return chosen


def load_split_pickle(path: str) -> Tuple[List[int], List[int], List[int]]:
    with open(path, 'rb') as handle:
        split = pickle.load(handle)
    return (
        list(split['train_indices']),
        list(split['val_indices']),
        list(split['test_indices']),
    )


def select_indices(
    args: argparse.Namespace,
    src_dataset,
) -> Tuple[List[int], Optional[List[int]], List[int], bool]:
    """Return (train, val, test, datamodule_split_flag)."""
    n_cells = len(src_dataset)
    if args.data == 'toy':
        cell_types = src_dataset[args.class_key]
        print(f'Classes in {args.class_key}:', dict(Counter(cell_types)))
        cells_per_type = args.batches_per_type * args.batch_size
        print(
            f'toy: {args.batches_per_type} batches/class × '
            f'batch_size {args.batch_size} = {cells_per_type} cells/class'
        )
        indices = pick_toy_indices(cell_types, cells_per_type)
        return indices, indices, indices, True

    if args.split:
        train_i, val_i, test_i = load_split_pickle(args.split_path)
        print(
            f'full + split: train={len(train_i)} val={len(val_i)} test={len(test_i)}'
        )
        return train_i, val_i, test_i, True

    all_i = list(range(n_cells))
    print(f'full, no split: {n_cells} cells (train metrics only)')
    return all_i, None, all_i, False


def gap_key(has_val: bool) -> str:
    return f'{"val" if has_val else "train"}/{GAP}'


def print_metrics_table(metrics_csv_path: str, has_val: bool) -> float:
    prefix = 'val' if has_val else 'train'
    gap_col = f'{prefix}/{GAP}'
    with open(metrics_csv_path) as handle:
        rows = [r for r in csv.DictReader(handle) if r.get(gap_col)]

    columns = [
        (f'{prefix}/gene_loss', 'gene_loss'),
        (f'{prefix}/contrastive_loss', 'contr'),
        (f'{prefix}/gene_cos_pred', 'cos_pred'),
        (gap_col, 'gap_copy'),
        (f'{prefix}/gene_gap_vs_static', 'gap_static'),
        (f'{prefix}/cell_gap_vs_copy_src', 'cell_gap'),
        (f'{prefix}/vicreg_var', 'vic_var'),
    ]
    header = 'epoch  ' + '  '.join(f'{short:>10}' for _, short in columns)
    print('\n' + header)
    print('-' * len(header))
    last_gap = 0.0
    for row in rows:
        cells = []
        for key, _ in columns:
            value = float(row[key]) if row.get(key) else float('nan')
            cells.append(f'{value:>+10.4f}')
        print(f"{int(float(row['epoch'])):>5}  " + '  '.join(cells))
        if row.get(gap_col):
            last_gap = float(row[gap_col])
    return last_gap


def print_verdict(args: argparse.Namespace, last_gap: float, metrics_path: str) -> None:
    print('\n================ VERDICT ================')
    print(f'data={args.data}  final {GAP} = {last_gap:+.4f}')
    if last_gap > 0.05:
        print('PASS: prediction beats copy-source.')
    else:
        print('FAIL: gap <= 0.05 — copying or collapse, not gene dynamics.')
        if args.data == 'toy':
            print('Even memorisation failed. Fix the design before a full run.')
    print(f'(table: {metrics_path})')


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = apply_mode_defaults(build_parser().parse_args(argv))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = resolve_run_dir(args, stamp)
    run_name = run_dir.name
    args.output_dir = str(run_dir)

    import pytorch_lightning as pl
    from datasets import load_from_disk
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    from perturbgen.Dataloaders.datamodule import PerturbGenDataModule
    from perturbgen.Model.gene_query_jepa_trainer import GeneQueryJEPATrainer
    from perturbgen.src.utils import read_dataset_files

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading tokenised datasets...')
    src_dataset = load_from_disk(f'{args.tokenized}/dataset_2000_hvg_src/normal.dataset')
    tgt_datasets = read_dataset_files(f'{args.tokenized}/dataset_2000_hvg_tgt', 'dataset')
    train_i, val_i, test_i, use_split = select_indices(args, src_dataset)
    has_val = val_i is not None
    monitor = gap_key(has_val)

    specs = build_specs(
        args,
        run_dir=run_dir,
        run_name=run_name,
        stamp=stamp,
        n_train=len(train_i),
        n_val=len(val_i) if val_i is not None else None,
        honesty_metric=monitor,
    )
    specs_path = write_specs(run_dir, specs)
    print(f'Wrote {specs_path}')

    data_module = PerturbGenDataModule(
        src_dataset=src_dataset,
        tgt_datasets=tgt_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        split=use_split,
        pred_tps=[1, 2, 3],
        n_total_tps=3,
        train_indices=train_i,
        val_indices=val_i,
        test_indices=test_i,
        use_weighted_sampler=False,
        seed=args.seed,
    )

    model = GeneQueryJEPATrainer(
        jepa_encoder='scmaskgit',
        encoder_path=args.encoder_path,
        jepa_encoder_layers=args.encoder_layers,
        freeze_jepa_encoder=args.freeze_encoder,
        predictor_layers=args.predictor_layers,
        n_queries=args.n_queries,
        frac_shared=0.5,
        frac_tgt_only=0.5,
        query_mode=args.query_mode,
        shared_max_queries=args.shared_max_queries,
        lambda_gene=args.lambda_gene,
        lambda_cell=args.lambda_cell,
        lambda_contrastive=args.lambda_contrastive,
        contrastive_temperature=args.contrastive_tau,
        vicreg_var_coeff=args.vicreg_var,
        vicreg_cov_coeff=args.vicreg_cov,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        normalize_latents=True,
        pred_tps=[1, 2, 3],
        n_total_tps=3,
        tokenid_to_rowid_path=f'{args.tokenized}/tokenid_to_rowid_2000_hvg.pkl',
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(
        'Run config: '
        f'data={args.data}, freeze_encoder={args.freeze_encoder}, '
        f'predictor_layers={args.predictor_layers}, '
        f'encoder_layers={args.encoder_layers}, '
        f'batch_size={args.batch_size}, '
        f'batches_per_type={args.batches_per_type if args.data == "toy" else "n/a"}, '
        f'lambda_contr={args.lambda_contrastive}, '
        f'vicreg_var={args.vicreg_var}, vicreg_cov={args.vicreg_cov}, '
        f'epochs={args.epochs}, output_dir={args.output_dir}'
    )

    logger = CSVLogger(save_dir=args.output_dir, name='logs')
    callbacks = []
    if args.early_stop:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode='max',
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
                verbose=True,
            )
        )
    if args.save_ckpt:
        ckpt_dir = str(Path(args.output_dir) / 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename='{epoch:02d}',
                monitor=monitor,
                mode='max',
                save_top_k=args.ckpt_top_k,
                every_n_epochs=args.ckpt_every,
                save_last=args.ckpt_save_last,
                save_weights_only=args.ckpt_weights_only,
                verbose=False,
            )
        )

    trainer_kwargs = dict(
        accelerator='gpu',
        devices=args.gpu,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=args.save_ckpt,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    if len(args.gpu) > 1:
        from pytorch_lightning.strategies import DDPStrategy

        trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=True)
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, data_module)

    metrics_path = os.path.join(logger.log_dir, 'metrics.csv')
    last_gap = print_metrics_table(metrics_path, has_val=has_val)
    specs['last_gene_gap_vs_copy_src'] = last_gap
    specs['metrics_csv'] = metrics_path
    specs['checkpoints_dir'] = str(Path(args.output_dir) / 'checkpoints')
    write_specs(run_dir, specs)
    print_verdict(args, last_gap, metrics_path)


if __name__ == '__main__':
    main()
