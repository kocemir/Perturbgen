"""Toy overfit experiment for Gene-Query JEPA.

WHY THIS SCRIPT EXISTS
----------------------
The full-data run collapsed: losses went to ~0 while the identity gaps
(prediction minus copy-baseline) stayed at ~0. Before we start turning
knobs on the big run we need the answer to a more basic question:

    Can this architecture learn ANYTHING when we let it memorise
    a tiny dataset?

So we build the smallest meaningful dataset — TWO mini-batches from EACH
cell type — train on it, and validate on the SAME cells. Overfitting is
not a bug here; it is the experiment. If the model cannot even memorise
its way past the copy-baseline, the problem is the design, not the scale.

HOW TO READ THE RESULT (printed at the end)
-------------------------------------------
PASS  val/gene_gap_vs_copy_src clearly above zero (say > +0.05):
      the architecture CAN learn gene dynamics; the full-run failure is
      about regularisation / hyperparameters (collapse pressure).
FAIL  the gap stays pinned near zero while the loss drops to zero:
      the collapse shortcut wins even with memorisation allowed;
      fix the design (e.g. freeze encoder) before burning GPU-days.

USAGE
-----
    python docs/examples/toy_train_gene_query_jepa.py

Env knobs (all optional):
    TOY_GPU=7  BATCH_SIZE=16  BATCHES_PER_TYPE=2  EPOCHS=40
    FREEZE_ENCODER=false  ENC_LAYERS=3
    VICREG_VAR=0.0  VICREG_COV=0.0          # VICReg optional (off by default)
    LAMBDA_CONTRASTIVE=0.3  CONTRASTIVE_TAU=0.1
    LR=1e-4  N_QUERIES=64  PREDICTOR_LAYERS=1
"""

import os

# Headless-safe defaults BEFORE heavy imports.
# This prevents X11/GL probes that emit "No protocol specified" in screen/SSH.
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
# OpenMPI/hwloc can try OpenGL discovery on this host and trip X11 checks.
os.environ.setdefault('HWLOC_COMPONENTS', '-gl')
os.environ.setdefault('HWLOC_GL_LINUX_NVIDIA_DISABLE', '1')

import csv
from collections import Counter
from pathlib import Path

import pytorch_lightning as pl
from datasets import load_from_disk
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from perturbgen.Dataloaders.datamodule import PerturbGenDataModule
from perturbgen.Model.gene_query_jepa_trainer import GeneQueryJEPATrainer
from perturbgen.src.utils import read_dataset_files

# ---------------------------------------------------------------------------
# Fixed paths (same data as the full run).
# ---------------------------------------------------------------------------
WORKSPACE = '/home/stuke1/perturbgen'
TOKENIZED = f'{WORKSPACE}/T_perturb/tokenized_data/LPS_all_tps_2k'
PRETRAIN_CKPT = (
    f'{WORKSPACE}/Perturbgen/pretraining_cohort/'
    '20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_'
    'ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt'
)
OUTPUT_DIR = os.environ.get(
    'TOY_OUT',
    '/mnt/sod2-project/csb4/stuke1/perturbgen/gene_query_jepa/toy_runs',
)

# ---------------------------------------------------------------------------
# Knobs (env-overridable). Defaults mirror the failed full run on purpose,
# so the toy result explains the full-run behaviour.
# ---------------------------------------------------------------------------
GPU = int(os.environ.get('TOY_GPU', 7))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 16))
BATCHES_PER_TYPE = int(os.environ.get('BATCHES_PER_TYPE', 2))
EPOCHS = int(os.environ.get('EPOCHS', 40))
FREEZE_ENCODER = os.environ.get('FREEZE_ENCODER', 'false').lower() in ('true', '1')
ENC_LAYERS = int(os.environ.get('ENC_LAYERS', 3))
VICREG_VAR = float(os.environ.get('VICREG_VAR', 0.0))
VICREG_COV = float(os.environ.get('VICREG_COV', 0.0))
LAMBDA_CONTRASTIVE = float(os.environ.get('LAMBDA_CONTRASTIVE', 0.3))
CONTRASTIVE_TAU = float(os.environ.get('CONTRASTIVE_TAU', 0.1))
LR = float(os.environ.get('LR', 1e-4))
N_QUERIES = int(os.environ.get('N_QUERIES', 64))
PREDICTOR_LAYERS = int(os.environ.get('PREDICTOR_LAYERS', 1))
USE_EARLY_STOP = os.environ.get('EARLY_STOP', 'true').lower() in ('true', '1')
EARLY_STOP_PATIENCE = int(os.environ.get('EARLY_STOP_PATIENCE', 6))
EARLY_STOP_MIN_DELTA = float(os.environ.get('EARLY_STOP_MIN_DELTA', 0.005))
# Checkpoints (for later gene/cell embedding extraction).
# Defaults keep everything; the sweep sets CKPT_TOP_K=1 to keep only the
# best epoch by val/gene_gap_vs_copy_src and save disk space.
SAVE_CKPT = os.environ.get('SAVE_CKPT', 'true').lower() in ('true', '1')
CKPT_EVERY = int(os.environ.get('CKPT_EVERY', 1))
CKPT_TOP_K = int(os.environ.get('CKPT_TOP_K', -1))        # -1 = keep all
CKPT_SAVE_LAST = os.environ.get('CKPT_SAVE_LAST', 'true').lower() in ('true', '1')
CKPT_WEIGHTS_ONLY = os.environ.get('CKPT_WEIGHTS_ONLY', 'false').lower() in ('true', '1')

# Keep each run in a descriptive subfolder unless user overrides TOY_OUT.
if os.environ.get('TOY_OUT'):
    OUTPUT_DIR = os.environ['TOY_OUT']
else:
    contr_tag = (
        f'contr{LAMBDA_CONTRASTIVE:g}'
        if LAMBDA_CONTRASTIVE > 0
        else 'contr0'
    )
    if VICREG_VAR > 0 or VICREG_COV > 0:
        vic_tag = f'vic{VICREG_VAR:g}_{VICREG_COV:g}'
    else:
        vic_tag = 'vic0'
    run_name = (
        f'freeze_{"true" if FREEZE_ENCODER else "false"}'
        f'_predL{PREDICTOR_LAYERS}'
        f'_encL{ENC_LAYERS}'
        f'_ep{EPOCHS}'
        f'_q{N_QUERIES}'
        f'_{contr_tag}'
        f'_{vic_tag}'
    )
    OUTPUT_DIR = str(Path(OUTPUT_DIR) / run_name)


def pick_toy_indices(cell_types: list, cells_per_type: int) -> list:
    """First ``cells_per_type`` row indices of every cell type.

    Deterministic on purpose (no sampling): re-runs see the same cells.
    Cell types with fewer cells than requested are skipped and reported.
    """
    indices_by_type = {}
    for row_index, cell_type in enumerate(cell_types):
        indices_by_type.setdefault(cell_type, [])
        if len(indices_by_type[cell_type]) < cells_per_type:
            indices_by_type[cell_type].append(row_index)

    chosen = []
    print(f'\nToy roster ({cells_per_type} cells wanted per type):')
    for cell_type, indices in sorted(indices_by_type.items()):
        if len(indices) < cells_per_type:
            print(f'  {cell_type:<40} {len(indices):>4} cells  -> SKIPPED (too few)')
            continue
        print(f'  {cell_type:<40} {len(indices):>4} cells  -> used')
        chosen.extend(indices)
    print(f'  total toy cells: {len(chosen)}\n')
    return chosen


def print_metrics_table(metrics_csv_path: str) -> float:
    """Print one line per epoch; return the last val gene gap."""
    with open(metrics_csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r.get('val/gene_cos_pred')]

    columns = [
        ('val/gene_loss', 'gene_loss'),
        ('val/contrastive_loss', 'contr'),
        ('val/gene_cos_pred', 'cos_pred'),
        ('val/gene_gap_vs_copy_src', 'gap_copy'),
        ('val/gene_gap_vs_static', 'gap_static'),
        ('val/cell_gap_vs_copy_src', 'cell_gap'),
        ('val/vicreg_var', 'vic_var'),
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
        if row.get('val/gene_gap_vs_copy_src'):
            last_gap = float(row['val/gene_gap_vs_copy_src'])
    return last_gap


def main() -> None:
    pl.seed_everything(0, workers=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Step 1: load the tokenised LPS data (same files as full run). ----
    print('Loading tokenised datasets...')
    src_dataset = load_from_disk(f'{TOKENIZED}/dataset_2000_hvg_src/normal.dataset')
    tgt_datasets = read_dataset_files(f'{TOKENIZED}/dataset_2000_hvg_tgt', 'dataset')

    # ---- Step 2: choose the toy cells — 2 batches per cell type. ----------
    cell_types = src_dataset['cell_type_harmonized']
    print('Cell types in the source data:', dict(Counter(cell_types)))
    toy_indices = pick_toy_indices(
        cell_types, cells_per_type=BATCHES_PER_TYPE * BATCH_SIZE
    )

    # ---- Step 3: datamodule. Train and validate on the SAME cells: --------
    # memorisation is exactly what this experiment measures.
    data_module = PerturbGenDataModule(
        src_dataset=src_dataset,
        tgt_datasets=tgt_datasets,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True,
        split=True,
        pred_tps=[1, 2, 3],
        n_total_tps=3,
        train_indices=toy_indices,
        val_indices=toy_indices,
        test_indices=toy_indices,
        use_weighted_sampler=False,
        seed=0,
    )

    # ---- Step 4: the model, with the SAME recipe as the failed full run. --
    model = GeneQueryJEPATrainer(
        jepa_encoder='scmaskgit',
        encoder_path=PRETRAIN_CKPT,
        jepa_encoder_layers=ENC_LAYERS,
        freeze_jepa_encoder=FREEZE_ENCODER,
        predictor_layers=PREDICTOR_LAYERS,
        n_queries=N_QUERIES,
        frac_shared=0.5,
        frac_tgt_only=0.3,
        lambda_gene=1.0,
        lambda_cell=0.1,
        lambda_contrastive=LAMBDA_CONTRASTIVE,
        contrastive_temperature=CONTRASTIVE_TAU,
        vicreg_var_coeff=VICREG_VAR,
        vicreg_cov_coeff=VICREG_COV,
        lr=LR,
        weight_decay=1e-4,
        ema_decay=0.996,
        normalize_latents=True,
        pred_tps=[1, 2, 3],
        n_total_tps=3,
        tokenid_to_rowid_path=f'{TOKENIZED}/tokenid_to_rowid_2000_hvg.pkl',
        output_dir=OUTPUT_DIR,
        seed=0,
    )
    print(
        'Run config: '
        f'freeze_encoder={FREEZE_ENCODER}, '
        f'predictor_layers={PREDICTOR_LAYERS}, '
        f'encoder_layers={ENC_LAYERS}, '
        f'lambda_contr={LAMBDA_CONTRASTIVE}, '
        f'tau={CONTRASTIVE_TAU}, '
        f'vicreg_var={VICREG_VAR}, vicreg_cov={VICREG_COV}, '
        f'epochs={EPOCHS}, '
        f'output_dir={OUTPUT_DIR}'
    )

    # ---- Step 5: train on one GPU, log every step to a CSV. ----------------
    logger = CSVLogger(save_dir=OUTPUT_DIR, name='toy_logs')
    callbacks = []
    if USE_EARLY_STOP:
        callbacks.append(
            EarlyStopping(
                monitor='val/gene_gap_vs_copy_src',
                mode='max',
                patience=EARLY_STOP_PATIENCE,
                min_delta=EARLY_STOP_MIN_DELTA,
                verbose=True,
            )
        )
    if SAVE_CKPT:
        ckpt_dir = str(Path(OUTPUT_DIR) / 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename='{epoch:02d}',
                monitor='val/gene_gap_vs_copy_src',
                mode='max',
                save_top_k=CKPT_TOP_K,
                every_n_epochs=CKPT_EVERY,
                save_last=CKPT_SAVE_LAST,
                save_weights_only=CKPT_WEIGHTS_ONLY,
                verbose=False,
            )
        )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[GPU],
        max_epochs=EPOCHS,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=SAVE_CKPT,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    trainer.fit(model, data_module)

    # ---- Step 6: the verdict. ----------------------------------------------
    metrics_path = os.path.join(logger.log_dir, 'metrics.csv')
    last_gap = print_metrics_table(metrics_path)
    print('\n================ VERDICT ================')
    print(f'final val/gene_gap_vs_copy_src = {last_gap:+.4f}')
    if last_gap > 0.05:
        print('PASS: the architecture CAN beat the copy baseline when it may')
        print('memorise. The full-run failure is a regularisation/scale issue.')
    else:
        print('FAIL: even memorisation does not beat copying. The collapse')
        print('shortcut dominates — fix the design (e.g. FREEZE_ENCODER=true,')
        print('stronger VICReg) before launching long runs.')
    print(f'(full table: {metrics_path})')


if __name__ == '__main__':
    main()
