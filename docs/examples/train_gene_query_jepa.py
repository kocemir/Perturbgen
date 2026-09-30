"""Gene-Query JEPA — one training entry for toy and full LPS.

Honesty metric: gene_gap_vs_copy_src must be > 0 (val on toy, train on
full-with-no-split). Index: docs/examples/GENE_QUERY_JEPA.md

USAGE
-----
  # toy: N train mini-batches per cell type; --val-batches-per-type 0 reuses them
  python docs/examples/train_gene_query_jepa.py --data toy \\
      --batches-per-type 4 --val-batches-per-type 1 --batch-size 16

  # full: every LPS cell (paper-style, no hold-out), last 2 GPUs
  python docs/examples/train_gene_query_jepa.py --data full --split true --gpu 6,7
  bash docs/examples/run_gene_query_full.sh

  # dump embeddings from a checkpoint (1 GPU). Writes embeddings/*.h5ad
  python docs/examples/train_gene_query_jepa.py --eval-ckpt path/to/epoch=02.ckpt

  # count decoder on frozen JEPA. --count-input source_only (z_hat) or copy_target
  python docs/examples/train_gene_query_jepa.py --train-count-decoder true \\
      --jepa-ckpt path/to/last.ckpt --count-input source_only --data full --split true --gpu 4,5,6,7

  # 90m source -> 6h/10h (PerturbGen IL1B pairing). No 2k split pickle: --split false
  python docs/examples/train_gene_query_jepa.py --data full --split false \\
      --tokenized /path/to/lps_90min_perturb --pred-tps 1,2 --gpu 0,1,2,3,4,5,6,7

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
TOKENIZED = os.environ.get(
    'TOKENIZED',
    '/mnt/sod2-project/csb4/stuke1/perturbgen_reproduction/lps/'
    'tokenized_data/LPS_all_tps_2k',
)
TOKENIZED_90M = (
    '/mnt/sod2-project/csb4/stuke1/perturbgen_reproduction/lps/'
    'tokenized_data/lps_90min_perturb'
)
PRETRAIN_CKPT = (
    f'{WORKSPACE}/Perturbgen/pretraining_cohort/'
    '20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_'
    'ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt'
)
SOD2 = '/mnt/sod2-project/csb4/stuke1/perturbgen'
DEFAULT_FULL_ROOT = f'{SOD2}/T_perturb/res/jepa_gene_query_full_atlas'
DEFAULT_90M_ROOT = f'{SOD2}/T_perturb/res/jepa_gene_query_90m_src'
DEFAULT_TOY_ROOT = f'{SOD2}/gene_query_jepa/toy_runs'
DEFAULT_COUNT_ROOT = f'{SOD2}/gene_query_jepa/count_decoder_runs'
GAP = 'gene_gap_vs_copy_src'
TIME_LABEL = {1: '90m_LPS', 2: '6h_LPS', 3: '10h_LPS'}
JEPA_COUNT_CKPT = (
    f'{SOD2}/T_perturb/res/jepa_gene_query_full_atlas/'
    'fzT_encL3_predL3_q128_mixed_lg1_lc0_contr0_vic0_bs16_ep20_'
    'lr0.0001_seed0_splitT_poolcls_20260920_021803/checkpoints/'
    'fzT_encL3_predL3_q128_mixed_lg1_lc0_contr0_vic0_bs16_ep20_'
    'lr0.0001_seed0_splitT_poolcls_20260920_021803-epoch=08-gap=0.1075.ckpt'
)


def _gpu_ids(value: str) -> List[int]:
    ids = [int(part.strip()) for part in str(value).split(',') if part.strip()]
    if not ids:
        raise argparse.ArgumentTypeError('expected at least one GPU id')
    return ids


def _int_list(value: str) -> List[int]:
    ids = [int(part.strip()) for part in str(value).replace(' ', ',').split(',') if part.strip()]
    if not ids:
        raise argparse.ArgumentTypeError('expected at least one integer')
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
        default=4,
        help='toy only: train mini-batches kept per cell type. '
        'train cells/class = batches-per-type × batch-size',
    )
    data.add_argument(
        '--val-batches-per-type',
        type=int,
        default=1,
        help='toy only: held-out val mini-batches per cell type (disjoint). '
        '0 = reuse the train cells as val (legacy).',
    )
    data.add_argument(
        '--class-key',
        default='cell_type_harmonized',
        help='toy only: metadata column used as the class',
    )
    data.add_argument(
        '--split',
        type=_bool,
        default=True,
        help='full only: if true, use --split-path hold-out (val metrics); if false, all cells',
    )
    data.add_argument(
        '--split-path',
        default=(
            f'{TOKENIZED}/splits/'
            'stratified_cell_type_harmonized_seed42_90_10.pkl'
        ),
        help='full + --split true: pickle with train/val/test_indices',
    )
    data.add_argument('--tokenized', default=TOKENIZED)
    data.add_argument(
        '--pred-tps',
        type=_int_list,
        default=[1, 2, 3],
        help='target time indices in the tokenized folder (resting: 1,2,3 = 90m/6h/10h; '
        '90m-source: 1,2 = 6h/10h)',
    )
    data.add_argument(
        '--n-total-tps',
        type=int,
        default=None,
        help='time-embedding size; default = max(--pred-tps)',
    )
    data.add_argument('--encoder-path', default=PRETRAIN_CKPT)
    data.add_argument(
        '--max-len',
        type=int,
        default=512,
        help='pad/truncate source and target gene sequences to this length',
    )

    model = parser.add_argument_group('model')
    model.add_argument('--freeze-encoder', type=_bool, default=False)
    model.add_argument(
        '--freeze-encoder-epochs',
        type=int,
        default=0,
        help='freeze the online encoder for this many epochs, then unfreeze. '
        '0 = no scheduled unfreeze. Implies encoder starts frozen.',
    )
    model.add_argument('--encoder-layers', type=int, default=3)
    model.add_argument('--predictor-layers', type=int, default=3)
    model.add_argument('--n-queries', type=int, default=128)
    model.add_argument('--frac-shared', type=float, default=0.4)
    model.add_argument('--frac-tgt-only', type=float, default=0.5)
    model.add_argument('--query-mode', choices=('mixed', 'all_shared'), default='mixed')
    model.add_argument('--shared-max-queries', type=int, default=0)
    model.add_argument('--lambda-gene', type=float, default=1.0)
    model.add_argument('--lambda-cell', type=float, default=0.0)
    model.add_argument('--lambda-contrastive', type=float, default=0.0)
    model.add_argument('--contrastive-tau', type=float, default=0.1)
    model.add_argument('--vicreg-var', type=float, default=0.0)
    model.add_argument('--vicreg-cov', type=float, default=0.0)
    model.add_argument(
        '--gene-loss',
        choices=('cosine', 'mse'),
        default='cosine',
        help='present-query reconstruction: cosine (default) or unnormalized MSE',
    )
    model.add_argument(
        '--normalize-latents',
        type=_bool,
        default=True,
        help='L2-normalize copies used for cosine / InfoNCE / mask-CE',
    )
    model.add_argument(
        '--dump-raw-latents',
        type=_bool,
        default=False,
        help='unused at eval (dump always writes L2 and raw). Kept for old ckpts.',
    )
    model.add_argument(
        '--lambda-mask-ce',
        type=float,
        default=0.0,
        help='weight for predictor mask-CE on a fraction of shared src genes',
    )
    model.add_argument(
        '--mask-frac-of-shared',
        type=float,
        default=0.0,
        help='fraction of chosen shared queries used as identity-free mask-CE',
    )
    model.add_argument(
        '--query-src-pos',
        type=_bool,
        default=True,
        help='add learnable source-rank PE to every gene query '
        '(last slot = gene absent from source)',
    )
    model.add_argument('--ema-decay', type=float, default=0.996)
    model.add_argument('--lr', type=float, default=1e-4)
    model.add_argument(
        '--encoder-lr',
        type=float,
        default=None,
        help='AdamW lr for the online encoder after unfreeze. Default: same as --lr.',
    )
    model.add_argument(
        '--encoder-lr-schedule',
        default=None,
        help='comma stages kind:epochs, e.g. freeze:5,5e-7:5,1e-6:5,freeze:5. '
        'Overrides --freeze-encoder-epochs when set. '
        'Toy default: freeze:5,5e-7:5,1e-6:5,freeze:5.',
    )
    model.add_argument('--weight-decay', type=float, default=1e-4)
    model.add_argument(
        '--cell-pool',
        choices=('mean', 'cls'),
        default='cls',
        help='cell target/src vector: encoder <cls> (default) or non-pad mean. '
        'Predicted cell vector is always the learned cell_query slot.',
    )
    data.add_argument(
        '--strip-tgt-special-tokens',
        type=_bool,
        default=False,
        help='JEPA default False: keep <cls>/<eos> on target sequences. '
        'PerturbGen train.py still strips. cell-pool=cls requires False.',
    )

    train = parser.add_argument_group('train')
    train.add_argument('--epochs', type=int, default=None, help='default: toy 20, full 5')
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
    ckpt.add_argument('--ckpt-every', type=int, default=10)
    ckpt.add_argument('--ckpt-top-k', type=int, default=-1, help='-1 keeps all')
    ckpt.add_argument('--ckpt-save-last', type=_bool, default=True)
    ckpt.add_argument('--ckpt-weights-only', type=_bool, default=False)

    ev = parser.add_argument_group('eval dump (notebook 08)')
    ev.add_argument(
        '--eval-ckpt',
        default=None,
        help='if set, skip training and dump cell/gene embeddings from this ckpt',
    )
    ev.add_argument(
        '--eval-split',
        choices=('test', 'all'),
        default='all',
        help='all = every cell; test = frozen 10%% pickle',
    )
    ev.add_argument(
        '--eval-split-path',
        default=(
            f'{TOKENIZED}/splits/'
            'stratified_cell_type_harmonized_seed42_90_10.pkl'
        ),
        help='pickle used when --eval-split test',
    )
    ev.add_argument(
        '--eval-force',
        type=_bool,
        default=False,
        help='re-run dump even if jepa_cell_embeddings.h5ad already exists',
    )
    ev.add_argument(
        '--gene-name-id-dict',
        default='/mnt/sod2-project/csb4/stuke1/Geneformer/geneformer/gene_name_id_dict.pkl',
        help='optional Ensembl↔symbol map written into the gene h5ad',
    )

    count = parser.add_argument_group('count decoder')
    count.add_argument(
        '--train-count-decoder',
        type=_bool,
        default=False,
        help='freeze JEPA and train ZINB CountHead',
    )
    count.add_argument(
        '--count-input',
        choices=('source_only', 'copy_target'),
        default='source_only',
        help='source_only = mean of present z_hat (mixed 128 quiz). '
        'copy_target = same z_hat mean, queries = first n_queries target tokens, skip pad.',
    )
    count.add_argument(
        '--jepa-ckpt',
        default=JEPA_COUNT_CKPT,
        help='Lightning ckpt of GeneQueryJEPATrainer (required for count training)',
    )
    count.add_argument('--n-genes', type=int, default=2000)
    count.add_argument('--count-dropout', type=float, default=0.1)
    count.add_argument(
        '--count-ckpt',
        default=None,
        help='CountHead Lightning ckpt. Required with --eval-count-ko',
    )
    count.add_argument(
        '--eval-count-ko',
        type=_bool,
        default=False,
        help='in-silico KO: replace --ko-gene in 90m src, decode control vs KO',
    )
    count.add_argument('--ko-gene', default='ENSG00000125538')
    count.add_argument('--ko-src-name', default='90m_LPS.dataset')
    count.add_argument(
        '--ko-pred-tps',
        nargs='+',
        type=int,
        default=[1, 2],
        help='target folder ids (90m pairing: 1=6h, 2=10h)',
    )
    count.add_argument(
        '--ko-replace',
        choices=('mask', 'pad'),
        default='mask',
        help='source edit for KO: <mask> id 1 (MaskGIT-matched, default) or <pad> id 0',
    )
    count.add_argument(
        '--ko-resync-quiz',
        type=_bool,
        default=False,
        help='KO: keep control gene questions but recompute query src_position '
        'from the edited source (knocked-out gene -> absent/-1). '
        'Default false freezes the full control quiz (old behaviour).',
    )
    count.add_argument('--tgt-h5ad-folder', default=None)
    return parser


def is_90m_src(tokenized: str) -> bool:
    name = Path(tokenized).name.lower()
    return '90min' in name or '90m' in name


def resolve_src_dataset_path(tokenized: str) -> str:
    src_dir = Path(tokenized) / 'dataset_2000_hvg_src'
    normal = src_dir / 'normal.dataset'
    if normal.exists():
        return str(normal)
    cands = sorted(p for p in src_dir.glob('*.dataset') if p.is_dir() or p.exists())
    if len(cands) == 1:
        return str(cands[0])
    names = [p.name for p in cands]
    raise SystemExit(f'cannot pick src dataset in {src_dir} (found {names})')


def time_labels_for(tokenized: str) -> Dict[int, str]:
    labels = dict(TIME_LABEL)
    tgt = Path(tokenized) / 'dataset_2000_hvg_tgt'
    if not tgt.is_dir():
        return labels
    for child in tgt.iterdir():
        if not child.name.endswith('.dataset'):
            continue
        stem = child.name[: -len('.dataset')]
        head, _, rest = stem.partition('_')
        if head.isdigit() and rest:
            labels[int(head)] = rest
    return labels


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.n_total_tps is None:
        args.n_total_tps = max(args.pred_tps)
    if args.n_total_tps < max(args.pred_tps):
        raise SystemExit('--n-total-tps must be >= max(--pred-tps)')
    if (
        is_90m_src(args.tokenized)
        and args.split
        and 'LPS_all_tps_2k' in str(args.split_path)
        and not args.eval_ckpt
    ):
        if 'LPS_all_tps_2k' in str(args.split_path):
            raise SystemExit(
                'lps_90min_perturb has no resting 90/10 pickle; '
                'use --split false (or a pairing-matched --split-path)'
            )
    if args.train_count_decoder:
        if not args.jepa_ckpt:
            raise SystemExit('--train-count-decoder requires --jepa-ckpt')
        if not Path(args.jepa_ckpt).is_file():
            raise SystemExit(f'--jepa-ckpt not found: {args.jepa_ckpt}')
        if args.epochs is None:
            args.epochs = 40
        if args.early_stop is None:
            args.early_stop = False
    elif args.epochs is None:
        args.epochs = 20 if args.data == 'toy' else 5
    if args.early_stop is None:
        args.early_stop = args.data == 'toy'
    if args.data == 'toy' and args.encoder_lr_schedule is None:
        args.encoder_lr_schedule = 'freeze:5,5e-7:5,1e-6:5,freeze:5'
    if args.data == 'toy' and args.batches_per_type < 1:
        raise SystemExit('--batches-per-type must be >= 1 for --data toy')
    if args.data == 'toy' and args.val_batches_per_type < 0:
        raise SystemExit('--val-batches-per-type must be >= 0')
    if args.data == 'full' and args.split and not args.split_path:
        raise SystemExit('--split true requires --split-path')
    if args.cell_pool == 'cls' and args.strip_tgt_special_tokens:
        raise SystemExit('--cell-pool cls requires --strip-tgt-special-tokens false')
    if args.freeze_encoder_epochs < 0:
        raise SystemExit('--freeze-encoder-epochs must be >= 0')
    return args


def spec_run_name(args: argparse.Namespace, stamp: str) -> str:
    """Filesystem-safe folder name with the knobs that distinguish runs."""
    freeze = 'fzT' if args.freeze_encoder else 'fzF'
    if args.freeze_encoder_epochs > 0:
        freeze = f'fz{args.freeze_encoder_epochs}thenU'
    split_tag = 'splitT' if args.data == 'full' and args.split else 'splitF'
    contr = (
        f'contr{args.lambda_contrastive:g}'
        if args.lambda_contrastive > 0
        else 'contr0'
    )
    if args.vicreg_var > 0 or args.vicreg_cov > 0:
        vic = f'vic{args.vicreg_var:g}_{args.vicreg_cov:g}'
    else:
        vic = 'vic0'
    name = (
        f'{freeze}'
        f'_encL{args.encoder_layers}'
        f'_predL{args.predictor_layers}'
        f'_q{args.n_queries}'
        f'_{args.query_mode}'
        f'_lg{args.lambda_gene:g}'
        f'_lc{args.lambda_cell:g}'
        f'_{contr}_{vic}'
        f'_gl{args.gene_loss}'
        f'_bs{args.batch_size}'
        f'_ep{args.epochs}'
        f'_lr{args.lr:g}'
        f'_seed{args.seed}'
        f'_{split_tag}'
        f'_pool{args.cell_pool}'
        f'_tp{"-".join(str(t) for t in args.pred_tps)}'
        f'_max{args.max_len}'
    )
    if is_90m_src(args.tokenized):
        name += '_src90m'
    if args.data == 'toy':
        name += f'_bpt{args.batches_per_type}'
    if args.lambda_mask_ce > 0:
        name += f'_mce{args.lambda_mask_ce:g}_ms{args.mask_frac_of_shared:g}'
    if args.query_src_pos:
        name += '_qsrcpos'
    if args.dump_raw_latents:
        name += '_rawdump'
    if args.train_count_decoder:
        cinp = 'srconly' if args.count_input == 'source_only' else 'copytgt'
        name = f'countdec_{cinp}_{name}'
    return f'{name}_{stamp}'


def default_output_root(args: argparse.Namespace) -> Path:
    if args.output_root:
        return Path(args.output_root)
    if args.train_count_decoder:
        return Path(DEFAULT_COUNT_ROOT)
    if args.data != 'toy' and is_90m_src(args.tokenized):
        return Path(DEFAULT_90M_ROOT)
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
            'val_batches_per_type': (
                args.val_batches_per_type if args.data == 'toy' else None
            ),
            'cells_per_type': (
                args.batches_per_type * args.batch_size
                if args.data == 'toy'
                else None
            ),
            'val_cells_per_type': (
                args.val_batches_per_type * args.batch_size
                if args.data == 'toy' and args.val_batches_per_type > 0
                else None
            ),
            'class_key': args.class_key,
            'split': bool(args.split) if args.data == 'full' else True,
            'pred_tps': list(args.pred_tps),
            'n_total_tps': args.n_total_tps,
            'max_len': args.max_len,
            'strip_tgt_special_tokens': args.strip_tgt_special_tokens,
        },
        'model': {
            'jepa_encoder': 'scmaskgit',
            'cell_pool': args.cell_pool,
            'freeze_encoder': args.freeze_encoder,
            'freeze_encoder_epochs': args.freeze_encoder_epochs,
            'encoder_lr_schedule': args.encoder_lr_schedule,
            'encoder_layers': args.encoder_layers,
            'predictor_layers': args.predictor_layers,
            'n_queries': args.n_queries,
            'query_mode': args.query_mode,
            'shared_max_queries': args.shared_max_queries,
            'frac_shared': args.frac_shared,
            'frac_tgt_only': args.frac_tgt_only,
            'absent_queries': True,
            'query_time': 'learned_embedding',
            'lambda_gene': args.lambda_gene,
            'lambda_cell': args.lambda_cell,
            'lambda_contrastive': args.lambda_contrastive,
            'contrastive_tau': args.contrastive_tau,
            'vicreg_var': args.vicreg_var,
            'vicreg_cov': args.vicreg_cov,
            'gene_loss': args.gene_loss,
            'normalize_latents': args.normalize_latents,
            'dump_raw_latents': args.dump_raw_latents,
            'lambda_mask_ce': args.lambda_mask_ce,
            'mask_frac_of_shared': args.mask_frac_of_shared,
            'use_query_src_pos': args.query_src_pos,
            'ema_decay': args.ema_decay,
            'normalize_latents': args.normalize_latents,
        },
        'train': {
            'epochs': args.epochs,
            'lr': args.lr,
            'encoder_lr': args.encoder_lr,
            'encoder_lr_schedule': args.encoder_lr_schedule,
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


def pick_toy_split(
    cell_types: Sequence[str],
    n_train: int,
    n_val: int,
) -> Tuple[List[int], List[int]]:
    """First ``n_train`` then ``n_val`` row indices per class (disjoint if n_val>0)."""
    need = n_train + max(n_val, 0)
    indices_by_type: Dict[str, List[int]] = {}
    for row_index, cell_type in enumerate(cell_types):
        bucket = indices_by_type.setdefault(cell_type, [])
        if len(bucket) < need:
            bucket.append(row_index)

    train: List[int] = []
    val: List[int] = []
    print(
        f'\nToy roster (train {n_train} + val {n_val} cells wanted per type):'
    )
    for cell_type, indices in sorted(indices_by_type.items()):
        if len(indices) < need:
            print(f'  {cell_type:<40} {len(indices):>4} cells  -> SKIPPED (too few)')
            continue
        print(f'  {cell_type:<40} {len(indices):>4} cells  -> used')
        train.extend(indices[:n_train])
        if n_val > 0:
            val.extend(indices[n_train:n_train + n_val])
    if n_val <= 0:
        val = list(train)
    print(f'  total toy train cells: {len(train)}  val cells: {len(val)}\n')
    if not train:
        raise SystemExit(
            'Toy roster is empty: lower --batches-per-type / --val-batches-per-type '
            'or --batch-size.'
        )
    overlap = set(train) & set(val)
    if n_val > 0 and overlap:
        raise SystemExit(f'toy train/val overlap ({len(overlap)} cells)')
    return train, val


def pick_toy_indices(
    cell_types: Sequence[str],
    cells_per_type: int,
) -> List[int]:
    train, _ = pick_toy_split(cell_types, cells_per_type, 0)
    return train


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
        n_train = args.batches_per_type * args.batch_size
        n_val = args.val_batches_per_type * args.batch_size
        print(
            f'toy: {args.batches_per_type} train batches/class × '
            f'batch_size {args.batch_size} = {n_train} train cells/class; '
            f'{args.val_batches_per_type} val batches/class = {n_val} val cells/class'
        )
        train_i, val_i = pick_toy_split(cell_types, n_train, n_val)
        return train_i, val_i, val_i, True

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
        (f'{prefix}/total_loss', 'total'),
        (f'{prefix}/gene_loss', 'gene_loss'),
        (f'{prefix}/mask_ce_loss', 'mask_ce'),
        (f'{prefix}/contrastive_loss', 'contr'),
        (f'{prefix}/gene_mse', 'gene_mse'),
        (f'{prefix}/gene_cos_pred', 'cos_pred'),
        (gap_col, 'gap_copy'),
        (f'{prefix}/gene_gap_vs_static', 'gap_static'),
        (f'{prefix}/cell_loss', 'cell_loss'),
        (f'{prefix}/vicreg_var', 'vic_var'),
        (f'{prefix}/vicreg_cov', 'vic_cov'),
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


def _csv_float(row: Dict[str, str], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ''):
            try:
                return float(value)
            except ValueError:
                continue
    return float('nan')


def print_count_metrics_table(metrics_csv_path: str, has_val: bool) -> float:
    with open(metrics_csv_path) as handle:
        raw_rows = list(csv.DictReader(handle))
    by_epoch: Dict[int, Dict[str, str]] = {}
    for row in raw_rows:
        if not row.get('epoch'):
            continue
        epoch = int(float(row['epoch']))
        merged = by_epoch.setdefault(epoch, {})
        for key, value in row.items():
            if value not in (None, ''):
                merged[key] = value
    print(
        '\nepoch  train/loss  train/mse  train/emd'
        + ('  val/loss  val/mse  val/emd' if has_val else '')
    )
    print('-' * (48 if not has_val else 88))
    last_mse = float('nan')
    for epoch in sorted(by_epoch):
        row = by_epoch[epoch]
        t_loss = _csv_float(row, 'train/loss_epoch', 'train/loss')
        t_mse = _csv_float(row, 'train/mse_epoch', 'train/mse')
        t_emd = _csv_float(row, 'train/emd', 'train/emd_epoch')
        line = (
            f'{epoch:>5}  {t_loss:>+10.4f}  {t_mse:>+10.4f}  {t_emd:>+10.4f}'
        )
        if has_val:
            v_loss = _csv_float(row, 'val/loss', 'val/loss_epoch')
            v_mse = _csv_float(row, 'val/mse', 'val/mse_epoch')
            v_emd = _csv_float(row, 'val/emd', 'val/emd_epoch')
            line += f'  {v_loss:>+10.4f}  {v_mse:>+10.4f}  {v_emd:>+10.4f}'
            if v_mse == v_mse:
                last_mse = v_mse
        elif t_mse == t_mse:
            last_mse = t_mse
        print(line)
    return last_mse


EVAL_VAR_LIST = ['cell_pairing_index', 'time_after_LPS', 'cell_type_harmonized']


def _ensembl_to_symbol(path: Optional[str]) -> Dict[str, str]:
    if not path or not Path(path).is_file():
        return {}
    with open(path, 'rb') as handle:
        raw = pickle.load(handle)
    out: Dict[str, str] = {}
    for key, value in raw.items():
        key_s, val_s = str(key), str(value)
        if key_s.startswith('ENSG'):
            out[key_s] = val_s
        elif val_s.startswith('ENSG'):
            out[val_s] = key_s
    return out


def write_jepa_h5ads(
    run_dir: Path,
    mapping_path: str,
    gene_name_id_dict: Optional[str] = None,
    time_labels: Optional[Dict[int, str]] = None,
) -> Tuple[Path, Path]:
    import anndata as ad
    import numpy as np
    import pandas as pd
    import torch

    pt_path = run_dir / 'embeddings' / 'gene_query_jepa_embeddings.pt'
    if not pt_path.is_file():
        raise SystemExit(f'missing dump: {pt_path}')
    payload = torch.load(pt_path, map_location='cpu', weights_only=False)
    out_dir = run_dir / 'embeddings'
    out_dir.mkdir(parents=True, exist_ok=True)

    time_labels = time_labels or TIME_LABEL
    z_hat = np.asarray(payload['z_hat_cell'])
    z_src = np.asarray(payload['z_src_cell'])
    z_tgt = np.asarray(payload['z_tgt_cell'])
    time_step = np.asarray(payload['time']).reshape(-1).astype(int)
    n_cells = int(z_hat.shape[0])
    obs = pd.DataFrame(
        {
            'time_step': time_step,
            'time_after_LPS': [time_labels.get(int(t), str(t)) for t in time_step],
        },
        index=[f'row_{i}' for i in range(n_cells)],
    )
    for col in EVAL_VAR_LIST:
        values = payload.get(col)
        if values is None:
            continue
        if len(values) != n_cells:
            print(f'warning: skip obs {col}: len {len(values)} != {n_cells}')
            continue
        obs[col] = list(values)
        if col == 'time_after_LPS':
            obs['time_after_LPS'] = [str(v) for v in values]

    cell = ad.AnnData(X=z_hat.copy(), obs=obs)
    cell.obsm['z_hat_cell'] = z_hat
    cell.obsm['z_src_cell'] = z_src
    cell.obsm['z_tgt_cell'] = z_tgt
    for raw_key in ('z_hat_cell_raw', 'z_src_cell_raw', 'z_tgt_cell_raw'):
        if raw_key in payload and payload[raw_key] is not None:
            cell.obsm[raw_key] = np.asarray(payload[raw_key])
    cell_path = out_dir / 'jepa_cell_embeddings.h5ad'
    cell.write_h5ad(cell_path)
    if 'z_hat_cell_raw' in cell.obsm:
        cell_raw = ad.AnnData(X=cell.obsm['z_hat_cell_raw'].copy(), obs=obs.copy())
        for key in ('z_hat_cell_raw', 'z_src_cell_raw', 'z_tgt_cell_raw'):
            if key in cell.obsm:
                cell_raw.obsm[key] = cell.obsm[key]
        cell_raw_path = out_dir / 'jepa_cell_embeddings_raw.h5ad'
        cell_raw.write_h5ad(cell_raw_path)

    with open(mapping_path, 'rb') as handle:
        id_to_ensembl = pickle.load(handle)
    gene_mean: Dict[str, Any] = payload['gene_mean']
    gene_count: Dict[str, Any] = payload['gene_count']
    vocab = int(next(iter(gene_mean.values())).shape[0])
    hat_counts = np.zeros(vocab, dtype=np.float64)
    for key, counts in gene_count.items():
        if str(key).startswith('hat_') and not str(key).endswith('_raw'):
            hat_counts += np.asarray(counts).reshape(-1)
    keep = hat_counts > 0
    token_ids = np.where(keep)[0]
    ensembl = [
        str(id_to_ensembl.get(int(i), f'local_{int(i)}')) for i in token_ids
    ]
    symbol_map = _ensembl_to_symbol(gene_name_id_dict)
    symbols = [symbol_map.get(e, e) for e in ensembl]
    var = pd.DataFrame(
        {
            'token_id': token_ids,
            'ensembl_id': ensembl,
            'gene_symbol': symbols,
            'n_queries_hat': hat_counts[token_ids],
        },
        index=ensembl,
    )
    gene = ad.AnnData(X=np.zeros((1, len(token_ids)), dtype=np.float32), var=var)
    for key, mat in gene_mean.items():
        arr = np.asarray(mat)[token_ids]
        gene.varm[str(key)] = arr
        is_raw = str(key).endswith('_raw')
        base = str(key)[:-4] if is_raw else str(key)
        if base.startswith('hat_t'):
            step = int(base.split('t')[-1])
            label = time_labels.get(step, str(step))
            if is_raw:
                gene.varm[f'hat_{label}_raw'] = arr
            else:
                gene.varm[label] = arr
                gene.varm[f'hat_{label}'] = arr
        elif base.startswith('tgt_t'):
            step = int(base.split('t')[-1])
            label = time_labels.get(step, str(step))
            gene.varm[f'tgt_{label}{"_raw" if is_raw else ""}'] = arr
        elif base.startswith('src_t'):
            step = int(base.split('t')[-1])
            label = time_labels.get(step, str(step))
            gene.varm[f'src_{label}{"_raw" if is_raw else ""}'] = arr
    for key, counts in gene_count.items():
        gene.var[f'count_{key}'] = np.asarray(counts)[token_ids]
    gene_path = out_dir / 'jepa_gene_embeddings.h5ad'
    gene.write_h5ad(gene_path)
    raw_varm = [k for k in gene.varm.keys() if str(k).endswith('_raw')]
    if raw_varm:
        gene_raw = ad.AnnData(X=gene.X.copy(), var=gene.var.copy())
        for key in raw_varm:
            gene_raw.varm[key] = gene.varm[key]
        gene_raw.write_h5ad(out_dir / 'jepa_gene_embeddings_raw.h5ad')
    print(f'Wrote {cell_path}  ({cell.n_obs} cell-time rows, L2 + obsm *_raw)')
    print(f'Wrote {gene_path}  ({gene.n_vars} genes, varm L2 + *_raw)')
    return cell_path, gene_path


def run_eval(args: argparse.Namespace) -> None:
    args = apply_mode_defaults(args)
    ckpt = Path(args.eval_ckpt)
    if not ckpt.is_file():
        raise SystemExit(f'--eval-ckpt not found: {ckpt}')
    run_dir = ckpt.parent.parent
    if args.output_dir:
        run_dir = Path(args.output_dir)
    cell_h5ad = run_dir / 'embeddings' / 'jepa_cell_embeddings.h5ad'
    if cell_h5ad.is_file() and not args.eval_force:
        print(f'{cell_h5ad} already exists; skip dump (pass --eval-force true to redo)')
        return

    import pytorch_lightning as pl
    from datasets import load_from_disk

    from perturbgen.Dataloaders.datamodule import PerturbGenDataModule
    from perturbgen.Model.gene_query_jepa_trainer import GeneQueryJEPATrainer
    from perturbgen.src.utils import read_dataset_files

    gpu_ids = args.gpu[:1]
    if len(args.gpu) > 1:
        print(f'eval dump is not DDP-safe; using GPU {gpu_ids[0]} only')
    pl.seed_everything(args.seed, workers=True)

    src_dataset = load_from_disk(resolve_src_dataset_path(args.tokenized))
    tgt_datasets = read_dataset_files(f'{args.tokenized}/dataset_2000_hvg_tgt', 'dataset')
    n_cells = len(src_dataset)
    if args.eval_split == 'test':
        _, _, test_i = load_split_pickle(args.eval_split_path)
        print(f'eval split=test: {len(test_i)} cells from {args.eval_split_path}')
        print(
            'NOTE: this JEPA run trained with --split false (all cells). '
            'The frozen test split is for a tractable UMAP, not a clean hold-out.'
        )
        use_split = True
    else:
        test_i = list(range(n_cells))
        print(f'eval split=all: {len(test_i)} cells')
        use_split = False

    data_module = PerturbGenDataModule(
        src_dataset=src_dataset,
        tgt_datasets=tgt_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        split=use_split,
        pred_tps=args.pred_tps,
        n_total_tps=args.n_total_tps,
        context_tps=[1, 2, 3],
        train_indices=test_i,
        val_indices=None,
        test_indices=test_i,
        var_list=EVAL_VAR_LIST,
        use_weighted_sampler=False,
        seed=args.seed,
        strip_tgt_special_tokens=args.strip_tgt_special_tokens,
        max_len=args.max_len,
    )

    model = GeneQueryJEPATrainer.load_from_checkpoint(
        str(ckpt),
        map_location='cpu',
        output_dir=str(run_dir),
        var_list=EVAL_VAR_LIST,
        lambda_mask_ce=0.0,
    )
    model.output_dir = str(run_dir)
    model.var_list = EVAL_VAR_LIST
    model.lambda_mask_ce = 0.0
    model.eval()
    model.freeze()
    (run_dir / 'embeddings').mkdir(parents=True, exist_ok=True)
    info = run_dir / 'embeddings' / 'ckpt_info.txt'
    info_lines = [
        f'ckpt={ckpt}',
        f'eval_split={args.eval_split}',
        f'n_eval_cells={len(test_i)}',
        'latents=l2+raw',
        'mask_ce=off',
    ]
    if args.eval_split == 'test':
        info_lines.append(f'eval_split_path={args.eval_split_path}')
    info.write_text('\n'.join(info_lines) + '\n')
    print(f'Wrote {info}')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=gpu_ids,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )
    trainer.test(model, datamodule=data_module)
    write_jepa_h5ads(
        run_dir,
        f'{args.tokenized}/token_id_to_genename_2000_hvg.pkl',
        args.gene_name_id_dict,
        time_labels=time_labels_for(args.tokenized),
    )


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


def load_tgt_counts(tokenized: str) -> Dict[str, Any]:
    from perturbgen.src.utils import read_dataset_files

    folder = f'{tokenized}/h5ad_pairing_2000_hvg_tgt'
    if not Path(folder).is_dir():
        raise SystemExit(f'count h5ads not found: {folder}')
    tgt_adatas = read_dataset_files(folder, 'h5ad')
    return {key: adata.X for key, adata in tgt_adatas.items()}


def run_count_train(args: argparse.Namespace) -> None:
    import pytorch_lightning as pl
    from datasets import load_from_disk
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    from perturbgen.Dataloaders.datamodule import PerturbGenDataModule
    from perturbgen.Model.gene_query_count_trainer import GeneQueryCountDecoderTrainer
    from perturbgen.src.utils import read_dataset_files

    args = apply_mode_defaults(args)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = resolve_run_dir(args, stamp)
    run_name = run_dir.name
    args.output_dir = str(run_dir)
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading tokenised datasets + count h5ads...')
    src_dataset = load_from_disk(resolve_src_dataset_path(args.tokenized))
    tgt_datasets = read_dataset_files(f'{args.tokenized}/dataset_2000_hvg_tgt', 'dataset')
    tgt_counts_dict = load_tgt_counts(args.tokenized)
    train_i, val_i, test_i, use_split = select_indices(args, src_dataset)
    has_val = val_i is not None
    monitor = 'val/mse' if has_val else 'train/mse'

    specs = build_specs(
        args,
        run_dir=run_dir,
        run_name=run_name,
        stamp=stamp,
        n_train=len(train_i),
        n_val=len(val_i) if val_i is not None else None,
        honesty_metric=monitor,
    )
    specs['task'] = 'jepa_count_decoder'
    specs['jepa_ckpt'] = args.jepa_ckpt
    specs['count_input'] = args.count_input
    specs['honesty_rule'] = (
        'count decoder: lower val/mse is better (not gene_gap). '
        'source_only: mixed quiz z_hat mean; '
        'copy_target: z_hat mean over first n_queries target tokens (unpadded).'
    )
    specs['paths']['jepa_ckpt'] = args.jepa_ckpt
    write_specs(run_dir, specs)
    print(f'Wrote {run_dir / "specs.json"}')

    data_module = PerturbGenDataModule(
        src_dataset=src_dataset,
        tgt_datasets=tgt_datasets,
        tgt_counts_dict=tgt_counts_dict,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        split=use_split,
        pred_tps=args.pred_tps,
        n_total_tps=args.n_total_tps,
        train_indices=train_i,
        val_indices=val_i,
        test_indices=test_i,
        use_weighted_sampler=False,
        seed=args.seed,
        strip_tgt_special_tokens=args.strip_tgt_special_tokens,
        max_len=args.max_len,
    )
    model = GeneQueryCountDecoderTrainer(
        jepa_ckpt=args.jepa_ckpt,
        n_genes=args.n_genes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.count_dropout,
        pred_tps=args.pred_tps,
        output_dir=args.output_dir,
        seed=args.seed,
        count_input=args.count_input,
    )
    print(
        'Count-decoder config: '
        f'count_input={args.count_input}, jepa_ckpt={args.jepa_ckpt}, '
        f'gpu={args.gpu}, bs={args.batch_size}, lr={args.lr}, '
        f'epochs={args.epochs}, output_dir={args.output_dir}'
    )
    logger = CSVLogger(save_dir=args.output_dir, name='logs')
    callbacks = []
    if args.early_stop:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode='min',
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
                verbose=True,
            )
        )
    if args.save_ckpt:
        ckpt_dir = str(Path(args.output_dir) / 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=run_name + '-last',
            save_top_k=0,
            save_last=True,
            save_weights_only=args.ckpt_weights_only,
            verbose=False,
        )
        ckpt_cb.CHECKPOINT_NAME_LAST = f'{run_name}-last'
        callbacks.append(ckpt_cb)
        print('Count decoder checkpoints: last epoch only; curves from metrics.csv')
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

        trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, data_module)
    metrics_path = os.path.join(logger.log_dir, 'metrics.csv')
    last_mse = print_count_metrics_table(metrics_path, has_val=has_val)
    specs['last_val_mse'] = last_mse
    specs['metrics_csv'] = metrics_path
    specs['checkpoints_dir'] = str(Path(args.output_dir) / 'checkpoints')
    curves_dir = Path(args.output_dir) / 'curves'
    specs['curves_dir'] = str(curves_dir)
    write_specs(run_dir, specs)
    if trainer.is_global_zero:
        import importlib.util

        plot_path = Path(__file__).resolve().parent / 'plot_count_decoder_curves.py'
        spec = importlib.util.spec_from_file_location('plot_count_decoder_curves', plot_path)
        plot_mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(plot_mod)
        plot_mod.plot_count_curves(metrics_path, curves_dir)
    print(f'Count decoder finished. last {monitor}={last_mse}')


def _global_gene_token_id(ensembl_id: str) -> int:
    from perturbgen.pp import TOKEN_DICTIONARY_FILE

    with open(TOKEN_DICTIONARY_FILE, 'rb') as handle:
        token_dictionary = pickle.load(handle)
    if ensembl_id not in token_dictionary:
        raise SystemExit(f'{ensembl_id} is not in the pretrain vocabulary')
    return int(token_dictionary[ensembl_id])


def _kept_obs(batch: dict, keep: Any, key: str) -> list:
    if key not in batch:
        return []
    values = batch[key]
    idx = keep.nonzero(as_tuple=False).view(-1).cpu().tolist()
    if hasattr(values, 'detach'):
        values = values.detach().cpu().tolist()
    return [values[i] for i in idx]


def run_count_ko(args: argparse.Namespace) -> None:
    if not args.count_ckpt or not Path(args.count_ckpt).is_file():
        raise SystemExit('--eval-count-ko needs --count-ckpt')
    if not args.jepa_ckpt or not Path(args.jepa_ckpt).is_file():
        raise SystemExit('--eval-count-ko needs --jepa-ckpt')
    args.pred_tps = list(args.ko_pred_tps)
    args = apply_mode_defaults(args)

    import pytorch_lightning as pl
    import torch
    from datasets import load_from_disk

    from perturbgen.Dataloaders.datamodule import PerturbGenDataModule
    from perturbgen.Model.gene_query_count_trainer import GeneQueryCountDecoderTrainer
    from perturbgen.Modules.gene_query_jepa import PAD_TOKEN_ID
    from perturbgen.pp import TOKEN_DICTIONARY_FILE
    from perturbgen.src.utils import read_dataset_files

    tokenized = args.tokenized
    src_path = Path(tokenized) / 'dataset_2000_hvg_src' / args.ko_src_name
    if not src_path.exists():
        raise SystemExit(f'missing 90m source tokens: {src_path}')
    h5ad_dir = args.tgt_h5ad_folder or f'{tokenized}/h5ad_pairing_2000_hvg_tgt'
    gene_token_id = _global_gene_token_id(args.ko_gene)
    with open(TOKEN_DICTIONARY_FILE, 'rb') as handle:
        token_dictionary = pickle.load(handle)
    if args.ko_replace == 'mask':
        replace_id = int(token_dictionary.get('<mask>', 1))
        replace_name = '<mask>'
    else:
        replace_id = int(token_dictionary.get('<pad>', PAD_TOKEN_ID))
        replace_name = '<pad>'
    stamp = datetime.now().strftime('%Y%m%d-%H:%M')
    run_dir = Path(args.output_dir) if args.output_dir else (
        Path(DEFAULT_COUNT_ROOT) / 'il1b_src_90m_retrain'
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(run_dir)
    pred_tps = list(args.ko_pred_tps)
    time_labels = time_labels_for(tokenized)
    pl.seed_everything(args.seed, workers=True)
    device = torch.device(f'cuda:{args.gpu[0]}' if torch.cuda.is_available() else 'cpu')
    print(
        f'JEPA IL1B KO on {device}; token {gene_token_id} ({args.ko_gene}); '
        f'replace={replace_name} id={replace_id}; '
        f'resync_quiz={args.ko_resync_quiz}; pred_tps={pred_tps}'
    )

    src_dataset = load_from_disk(str(src_path))
    tgt_datasets = read_dataset_files(f'{tokenized}/dataset_2000_hvg_tgt', 'dataset')
    tgt_adatas = read_dataset_files(h5ad_dir, 'h5ad')
    n_genes = int(next(iter(tgt_adatas.values())).shape[1])
    tgt_counts_dict = {key: adata.X for key, adata in tgt_adatas.items()}
    n_cells = len(src_dataset)
    data_module = PerturbGenDataModule(
        src_dataset=src_dataset,
        tgt_datasets=tgt_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        split=False,
        pred_tps=pred_tps,
        n_total_tps=args.n_total_tps,
        train_indices=list(range(n_cells)),
        val_indices=None,
        test_indices=list(range(n_cells)),
        use_weighted_sampler=False,
        seed=args.seed,
        tgt_counts_dict=tgt_counts_dict,
        var_list=EVAL_VAR_LIST,
        strip_tgt_special_tokens=args.strip_tgt_special_tokens,
        max_len=args.max_len,
    )
    data_module.setup('test')
    loader = data_module.test_dataloader()

    model = GeneQueryCountDecoderTrainer.load_from_checkpoint(
        str(args.count_ckpt),
        map_location='cpu',
        jepa_ckpt=str(args.jepa_ckpt),
        n_genes=n_genes,
        pred_tps=pred_tps,
        output_dir=args.output_dir,
        strict=False,
    )
    model.eval()
    model.to(device)
    model.jepa.to(device)
    model.jepa.eval()
    model.pred_tps = pred_tps

    pert_chunks, ctrl_chunks, true_chunks = [], [], []
    obs = {
        'cell_type_harmonized': [],
        'cell_pairing_index': [],
        'time_after_LPS': [],
        'cell_idx': [],
    }
    n_kept = 0
    for batch_i, batch in enumerate(loader):
        moved = {
            key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        result = model.predict_control_and_ko(
            moved,
            gene_token_id=gene_token_id,
            replace_id=replace_id,
            resync_quiz=bool(args.ko_resync_quiz),
        )
        if result is None:
            continue
        keep = result['keep']
        for time_step in result['control']:
            pert_chunks.append(result['perturbed'][time_step].cpu())
            ctrl_chunks.append(result['control'][time_step].cpu())
            true_chunks.append(result['true'][time_step].cpu())
            n = int(result['control'][time_step].size(0))
            n_kept += n
            obs['time_after_LPS'].extend(
                [time_labels.get(time_step, f't{time_step}')] * n
            )
            ct = _kept_obs(moved, keep, f'cell_type_harmonized_t{time_step}')
            idx = _kept_obs(moved, keep, f'cell_pairing_index_t{time_step}')
            if not ct:
                ct = _kept_obs(moved, keep, 'cell_type_harmonized')
            if not idx:
                idx = _kept_obs(moved, keep, 'cell_pairing_index')
            obs['cell_type_harmonized'].extend(ct if ct else ['unknown'] * n)
            obs['cell_pairing_index'].extend(
                idx if idx else list(range(n_kept - n, n_kept))
            )
            obs['cell_idx'].extend(idx if idx else list(range(n_kept - n, n_kept)))
        if batch_i % 20 == 0:
            print(f'  batch {batch_i} kept_rows={n_kept}')

    if not pert_chunks:
        raise SystemExit('no cells carried the KO gene in the 90m source')
    import anndata as ad
    import pandas as pd

    pert = torch.cat(pert_chunks).numpy()
    ctrl = torch.cat(ctrl_chunks).numpy()
    true = torch.cat(true_chunks).numpy()
    for key, values in obs.items():
        if len(values) != pert.shape[0]:
            obs[key] = (values + [''] * pert.shape[0])[: pert.shape[0]]
    adata = ad.AnnData(X=pert, obs=pd.DataFrame(obs))
    adata.layers['pred_counts'] = ctrl
    adata.layers['true_counts'] = true
    adata.layers['pert_counts'] = pert
    ref = next(iter(tgt_adatas.values()))
    if ref.n_vars == adata.n_vars:
        adata.var_names = ref.var_names.astype(str)
        if 'gene_symbol' in ref.var.columns:
            adata.var['gene_symbol'] = ref.var['gene_symbol'].astype(str).values
    ko_tag = 'kmask' if args.ko_replace == 'mask' else 'kpad'
    if args.ko_resync_quiz:
        ko_tag = f'{ko_tag}_qresync'
    out_path = run_dir / (
        f'{stamp}_minference_adata_g{args.ko_gene}_ssrc_{ko_tag}.h5ad'
    )
    adata.write_h5ad(out_path)
    print(f'Wrote {out_path}  n_obs={adata.n_obs} x {adata.n_vars}')
    print(adata.obs['time_after_LPS'].value_counts().to_string())


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.eval_ckpt:
        run_eval(args)
        return
    if args.eval_count_ko:
        run_count_ko(args)
        return
    if args.train_count_decoder:
        run_count_train(args)
        return
    args = apply_mode_defaults(args)
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
    src_dataset = load_from_disk(resolve_src_dataset_path(args.tokenized))
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
        pred_tps=args.pred_tps,
        n_total_tps=args.n_total_tps,
        train_indices=train_i,
        val_indices=val_i,
        test_indices=test_i,
        use_weighted_sampler=False,
        seed=args.seed,
        strip_tgt_special_tokens=args.strip_tgt_special_tokens,
        max_len=args.max_len,
    )

    model = GeneQueryJEPATrainer(
        jepa_encoder='scmaskgit',
        encoder_path=args.encoder_path,
        jepa_encoder_layers=args.encoder_layers,
        freeze_jepa_encoder=args.freeze_encoder,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        encoder_lr_schedule=args.encoder_lr_schedule,
        predictor_layers=args.predictor_layers,
        n_queries=args.n_queries,
        frac_shared=args.frac_shared,
        frac_tgt_only=args.frac_tgt_only,
        query_mode=args.query_mode,
        shared_max_queries=args.shared_max_queries,
        lambda_gene=args.lambda_gene,
        lambda_cell=args.lambda_cell,
        lambda_contrastive=args.lambda_contrastive,
        contrastive_temperature=args.contrastive_tau,
        vicreg_var_coeff=args.vicreg_var,
        vicreg_cov_coeff=args.vicreg_cov,
        gene_loss=args.gene_loss,
        lambda_mask_ce=args.lambda_mask_ce,
        mask_frac_of_shared=args.mask_frac_of_shared,
        use_query_src_pos=args.query_src_pos,
        dump_raw_latents=args.dump_raw_latents,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        normalize_latents=args.normalize_latents,
        pred_tps=args.pred_tps,
        n_total_tps=args.n_total_tps,
        tokenid_to_rowid_path=f'{args.tokenized}/tokenid_to_rowid_2000_hvg.pkl',
        output_dir=args.output_dir,
        seed=args.seed,
        cell_pool=args.cell_pool,
        max_seq_length=args.max_len,
    )
    print(
        'Run config: '
        f'data={args.data}, freeze_encoder={args.freeze_encoder}, '
        f'freeze_encoder_epochs={args.freeze_encoder_epochs}, '
        f'encoder_lr_schedule={args.encoder_lr_schedule}, '
        f'lr={args.lr}, encoder_lr={args.encoder_lr}, '
        f'cell_pool={args.cell_pool}, '
        f'strip_tgt_special_tokens={args.strip_tgt_special_tokens}, '
        f'predictor_layers={args.predictor_layers}, '
        f'encoder_layers={args.encoder_layers}, '
        f'batch_size={args.batch_size}, '
        f'max_len={args.max_len}, '
        f'batches_per_type={args.batches_per_type if args.data == "toy" else "n/a"}, '
        f'lambda_contr={args.lambda_contrastive}, '
        f'gene_loss={args.gene_loss}, '
        f'lambda_mask_ce={args.lambda_mask_ce}, '
        f'mask_frac_of_shared={args.mask_frac_of_shared}, '
        f'query_src_pos={args.query_src_pos}, '
        f'dump_raw={args.dump_raw_latents}, '
        f'vicreg_var={args.vicreg_var}, vicreg_cov={args.vicreg_cov}, '
        f'epochs={args.epochs}, output_dir={args.output_dir}, '
        f'tokenized={args.tokenized}, pred_tps={args.pred_tps}'
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
        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=run_name + '-epoch={epoch:02d}-gap={' + monitor + ':.4f}',
            auto_insert_metric_name=False,
            monitor=monitor,
            mode='max',
            save_top_k=args.ckpt_top_k,
            every_n_epochs=args.ckpt_every,
            save_last=args.ckpt_save_last,
            save_weights_only=args.ckpt_weights_only,
            verbose=False,
        )
        ckpt_cb.CHECKPOINT_NAME_LAST = f'{run_name}-last'
        callbacks.append(ckpt_cb)

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

        trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=False)
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
