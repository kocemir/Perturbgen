import argparse
import os
import uuid
import warnings
from datetime import datetime


def _force_headless_env() -> None:
    """Prevent X11 'No protocol specified' on headless/SSH hosts.

    Primary cause here is OpenMPI/mpi4py (via Lightning MPIEnvironment.detect):
    importing MPI starts an `orted` singleton that probes X11/GL and prints
    'No protocol specified', then can hang. Also clear DISPLAY and force Agg.
    """
    os.environ.pop('DISPLAY', None)
    os.environ.pop('WAYLAND_DISPLAY', None)
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('WANDB_DISABLE_CODE', 'true')
    os.environ.setdefault('WANDB_CONSOLE', 'off')
    os.environ.setdefault('WANDB_SILENT', 'true')
    # hwloc (pulled in by OpenMPI) probes GL/X11 unless disabled.
    os.environ.setdefault('HWLOC_COMPONENTS', '-gl')
    os.environ.setdefault('HWLOC_GL_LINUX_NVIDIA_DISABLE', '1')
    # Stop OpenMPI/PMIx from GUI / oob probes that can emit X11 noise.
    os.environ.setdefault('OMPI_MCA_btl', 'self,tcp')
    os.environ.setdefault('OMPI_MCA_orte_base_help_aggregate', '0')
    os.environ.setdefault('PMIX_MCA_gds', 'hash')
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')


def _disable_lightning_mpi_autodetect() -> None:
    """Stop Lightning from importing mpi4py during Trainer construction.

    ``MPIEnvironment.detect()`` calls ``from mpi4py import MPI``, which on this
    host starts OpenMPI ``orted`` and can hang / print 'No protocol specified'.
    Training already forces ``LightningEnvironment`` for multi-GPU DDP; test must
    do the same (or disable detect) for single-GPU ``strategy='auto'``.
    """
    try:
        from lightning_fabric.plugins.environments.mpi import MPIEnvironment

        MPIEnvironment.detect = staticmethod(lambda: False)  # type: ignore[method-assign]
    except Exception:
        pass


_force_headless_env()

import pytorch_lightning as pl
import scanpy as sc
from sympy import limit
import torch
from datasets import concatenate_datasets, load_from_disk
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.plugins.environments import LightningEnvironment

_disable_lightning_mpi_autodetect()

from perturbgen.configs import ROOT
from perturbgen.Dataloaders.datamodule import PerturbGenDataModule
from perturbgen.Model.trainer import CountDecoderTrainer, PerturbGenTrainer
from perturbgen.src.utils import (
    condition_for_count_loss,
    get_idx_for_filtering,
    load_frozen_split,
    randomised_split,
    read_dataset_files,
    str2bool,
    stratified_split,
)

os.chdir(ROOT)
print(f'Current working directory: {os.getcwd()}')


def get_args(argv):
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_mode',
        type=str,
        default='count',
        help='Mode [masking, count]',
    )
    parser.add_argument(
        '--split',
        type=str2bool,
        default=True,
        help='split data for extrapolation',
    )
    parser.add_argument(
        '--split_path',
        type=str,
        default=None,
        help=(
            'Optional path to a frozen split .pkl from notebook 02 '
            '(keys: train_indices, val_indices, test_indices). '
            'When set, overrides stratified/random index generation.'
        ),
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./T_perturb/perturbgen/plt/res/cytoimmgen',
        help='store dataset name',
    )
    parser.add_argument(
        '--splitting_mode',
        type=str,
        default='stratified',
        choices=['random', 'stratified', 'unseen_cond'],
        help='splitting mode',
    )
    parser.add_argument(
        '--train_prop',
        type=float,
        default=0.8,
    )
    parser.add_argument(
        '--test_prop',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '--split_obs',
        type=str,
        nargs='+',
        default=None,
    )
    parser.add_argument('--split_value', type=str, default='D351')
    parser.add_argument('--use_positional_encoding', type=str2bool, default=False)
    parser.add_argument('--layer_norm', type=str2bool, default=False)
    parser.add_argument(
        '--generate',
        type=str2bool,
        default=True,
        help='generate data',
    )
    parser.add_argument(
        '--return_embeddings',
        type=str2bool,
        default=False,
        help='return embedding',
    )
    parser.add_argument(
        '--return_attn',
        type=str2bool,
        default=False,
        help='return attention',
    )
    parser.add_argument(
        '--ckpt_masking_path',
        type=str,
        default=None,
        help='path to checkpoint',
    )
    parser.add_argument(
        '--ckpt_count_path',
        type=str,
        default=None,
        help='path to checkpoint',
    )
    parser.add_argument(
        '--mapping_dict_path',
        type=str,
        default='./T_perturb/tokenized_data/cytoimmgen/token_id_to_genename_hvg.pkl',
    )
    parser.add_argument(
        '--src_dataset',
        type=str,
        default='./T_perturb/tokenized_data/cytoimmgen/dataset_hvg_src/0h.dataset',
        help='path to tokenised resting data',
    )
    parser.add_argument(
        '--tgt_dataset_folder',
        type=str,
        default='./T_perturb/tokenized_data/cytoimmgen/dataset_hvg_tgt/',
        help='path to tokenised activated data',
    )

    parser.add_argument(
        '--src_adata',
        type=str,
        default=(
            './T_perturb/tokenized_data/cytoimmgen/h5ad_pairing_hvg_src/0h.h5ad'
        ),
        help='path to src',
    )
    parser.add_argument(
        '--tgt_adata_folder',
        type=str,
        default=('./T_perturb/tokenized_data/cytoimmgen/h5ad_pairing_hvg_tgt'),
        help='path to tgt',
    )
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--shuffle', type=bool, default=False, help='shuffle')
    parser.add_argument('--num_node', type=int, default=1)
    parser.add_argument(
        '--log_dir', type=str, default='logs', help='path to data directory'
    )
    parser.add_argument(
        '--cellgen_lr', type=float, default=0.0001, help='learning rate'
    )
    parser.add_argument('--count_lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--cellgen_wd', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--count_wd', type=float, default=0.01, help='weight decay')
    parser.add_argument('--n_workers', type=int, default=32, help='number of workers')
    parser.add_argument(
        '--num_layers', type=int, default=6, help='number of decoder layers'
    )
    parser.add_argument('--d_ff', type=int, default=128, help='feed forward dimension')
    parser.add_argument(
        '--loss_mode', type=str, default='mse', help='loss mode [zinb, nb, mse]'
    )
    parser.add_argument('--cellgen_dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--count_dropout', type=float, default=0.0, help='dropout')
    parser.add_argument(
        '--condition_keys',
        nargs='+',
        default=None,
        type=str,
        help='Selection of condition keys to use for model',
    )
    parser.add_argument(
        '--mask_scheduler',
        type=str,
        default='cosine',
        help='mask scheduler [cosine, exp, pow]',
    )
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature')
    parser.add_argument('--sequence_length', type=int, default=150, help='iterations')
    parser.add_argument('--iterations', type=int, default=20, help='iterations')
    parser.add_argument('--conditions', type=dict, default=None, help='conditions')
    parser.add_argument(
        '--conditions_combined', type=list, default=None, help='conditions combined'
    )
    parser.add_argument(
        '--pred_tps',
        type=int,
        nargs='+',
        default=[1, 2, 3],
        help='time steps to include during training',
    )
    parser.add_argument(
        '--context_tps',
        type=int,
        nargs='+',
        default=None,
    )
    parser.add_argument(
        '--var_list',
        nargs='+',
        type=str,
        default=['Cell_population', 'Cell_type', 'Time_point', 'Donor'],
        help='List of variables to keep in the dataset',
    )
    parser.add_argument(
        '--cond_list',
        nargs='+',
        type=str,
        help='List of variables to form condition tokens',
    )
    parser.add_argument(
        '--encoder',
        default='scmaskgit',
        type=str,
        choices=[
            'Transformer_encoder',
            'scmaskgit',
        ],
        help='mode of encoder',
    )
    parser.add_argument(
        '--encoder_path',
        type=str,
        default=None,
        help='path to pre-trained encoder',
    )
    parser.add_argument(
        '--tokenid_to_rowid_path',
        type=str,
        default=None,
        help='path to tokenid to rowid mapping file',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='seed for reproducibility',
    )
    parser.add_argument(
        '--wandb_mode',
        type=str,
        default=os.environ.get('WANDB_MODE', 'online'),
        choices=['online', 'offline', 'disabled'],
        help=(
            'Weights & Biases logging mode. Use "offline" or "disabled" on '
            'machines without internet access to avoid hanging on wandb '
            'authentication. Defaults to the WANDB_MODE env var, or "online".'
        ),
    )
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=os.environ.get('WANDB_ENTITY', None),
        help=(
            'Weights & Biases entity (team/user). Defaults to the WANDB_ENTITY '
            'env var, or your default wandb entity if unset.'
        ),
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=os.environ.get('WANDB_PROJECT', 'perturbgen'),
        help='Weights & Biases project name.',
    )
    parser.add_argument(
        '--context_mode',
        type=str2bool,
        default=False,
        help='context mode for timepoints',
    )
    parser.add_argument(
        '--pos_encoding_mode',
        type=str,
        default='time_pos_sin',
        help='positional encoding',
    )
    parser.add_argument(
        '--d_model',
        type=int,
        default=512,
        help='embedding dimension',
    )
    parser.add_argument(
        '--deg_pkl_path',
        type=str,
        default=None,
        help='path to deg pkl file',
    )
    parser.add_argument(
        '--return_gene_embs',
        type=str2bool,
        default=False,
        help='return gene embeddings',
    )
    parser.add_argument(
        '--gene_embs_condition',
        type=str,
        default=None,
        help='aggregate gene embeddings over condition',
    )
    parser.add_argument(
        '--filter_cond',
        type=str,
        nargs='+',
        default=None,
        help='condition to filter tgt datasets',
    )
    parser.add_argument(
        '--filter_var',
        type=str,
        default=None,
        help='covariate to filter tgt datasets',
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=3,
    )
    args = parser.parse_args(argv)
    return args


def _apply_ckpt_arch_hparams(args) -> dict:
    """Override CLI arch hparams from Lightning checkpoint hyper_parameters.

    extract-embedding builds a fresh module then loads weights; if ``d_model`` /
    ``d_ff`` / etc. differ from training defaults, load_state_dict crashes with
    size mismatches. Prefer the values stored in the .ckpt whenever present.
    """
    path = None
    if args.test_mode == 'masking':
        path = args.ckpt_masking_path
    elif args.test_mode == 'count':
        path = args.ckpt_count_path or args.ckpt_masking_path
    if not path or not str(path).endswith('.ckpt') or not os.path.isfile(path):
        return {}
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location='cpu')
    hp = ckpt.get('hyper_parameters') or {}
    if not hp:
        return {}
    overrides = {
        'd_model': 'd_model',
        'd_ff': 'd_ff',
        'num_layers': 'num_layers',
        'mask_scheduler': 'mask_scheduler',
        'pos_encoding_mode': 'pos_encoding_mode',
        'context_mode': 'context_mode',
        'pred_tps': 'pred_tps',
    }
    for ckpt_key, arg_key in overrides.items():
        if ckpt_key not in hp or hp[ckpt_key] is None:
            continue
        old = getattr(args, arg_key)
        new = hp[ckpt_key]
        if old != new:
            print(f'Overriding --{arg_key} from ckpt: {old} -> {new}')
        setattr(args, arg_key, new)
    args.num_heads = int(hp.get('num_heads', getattr(args, 'num_heads', 8)))
    return dict(hp)


def main(argv=None) -> None:
    """Run training."""
    # Re-apply before any logger/trainer construction (some libs reset DISPLAY).
    _force_headless_env()
    args = get_args(argv)
    ckpt_hp = _apply_ckpt_arch_hparams(args)
    if args.wandb_mode == 'disabled':
        os.environ['WANDB_MODE'] = 'disabled'
    print('positional encoding:', args.pos_encoding_mode)

    # PyTorch Lightning allows to set all necessary seeds in one function call.
    pl.seed_everything(42, workers=True)
    torch.manual_seed(42)
    # Load and preprocess data
    print('Loading and preprocessing data...')
    tgt_datasets = read_dataset_files(
        args.tgt_dataset_folder,
        'dataset',
    )
    tgt_adatas = read_dataset_files(
        args.tgt_adata_folder,
        'h5ad',
    )
    src_dataset = load_from_disk(args.src_dataset)
    src_adata = sc.read_h5ad(args.src_adata)

    # select max input id and max len across all tgt datasets
    max_tgt_input_id = 0
    max_len = 0
    for keys, dataset in tgt_datasets.items():
        # print max input id
        input_id = dataset['input_ids']
        max_tgt_input_id = max(max(max(input_id)), max_tgt_input_id)
        max_len = max(max_len, max([len(x) for x in input_id]))
    max_tgt_input_id = max_tgt_input_id + 1 # add 1 for padding
    max_len = max(max_len, max([len(x) for x in src_dataset['input_ids']]))
    print(
        f'---PerturbGen training --- \n'
        f'Target vocab size: {max_tgt_input_id}, max sequence length: {max_len}'
    )

    # Drop covariate names that are not present in this dataset (CLI defaults are
    # for a different cohort; LPS uses cell_type_harmonized / time_after_LPS).
    if args.var_list:
        ref_ds = next(iter(tgt_datasets.values()))
        available = set(getattr(ref_ds, 'column_names', None) or ref_ds.features.keys())
        missing = [v for v in args.var_list if v not in available]
        kept = [v for v in args.var_list if v in available]
        if missing:
            print(f'Skipping --var_list entries absent from dataset: {missing}')
        args.var_list = kept if kept else None

    V = max_tgt_input_id + 50
    for keys, dataset in tgt_datasets.items():
        k = keys

        ds = dataset
        bad = []
        for i in range(min(5000, len(ds))):
            ids = ds[i]["input_ids"]
            if ids:
                lo, hi = min(ids), max(ids)
                if lo < 0 or hi >= V:
                    bad.append((i, lo, hi, ds[i].get("input_ids", None)))
                if len(bad) > 0:
                    print(k, "bad count:", len(bad), "first:", bad[:10])
                    raise ValueError(f"Dataset {k} has out of bounds token ids")


    if args.loss_mode == 'mse':
        # log normalize data only for mse loss
        sc.pp.normalize_total(src_adata, target_sum=1e4)
        sc.pp.log1p(src_adata)
        for _, tgt_adata in tgt_adatas.items():
            sc.pp.normalize_total(tgt_adata, target_sum=1e4)
            sc.pp.log1p(tgt_adata)

    # Classifier-free guidance pre-processing
    # ---------------------------------------
    # create dictionnary of metadata for classifier-free guidance
    # needs to be completed before any filtering to initialize
    # all condition tokens
    if args.cond_list is not None:
        full_dataset = concatenate_datasets([src_dataset] + list(tgt_datasets.values()))
        condition_dict = {}
        for condition in args.cond_list:
            condition_dict[condition] = {
                cell_type: i + max_tgt_input_id
                for i, cell_type in enumerate(full_dataset.unique(condition))
            }
            max_tgt_input_id += len(condition_dict[condition])
    else:
        condition_dict = None

    # 1. Filter datasets based on condition, if available
    # 2. Extract condition to return gene embeddings
    # ---------------------------------------------------

    # create full dataset to extract metadata for conditioning
    filter_idx = []
    # filter dataset based on condition
    if (args.filter_var is not None) and (args.filter_cond is not None):
        for dataset in tgt_datasets.values():
            idx_ = get_idx_for_filtering(
                dataset,
                args.filter_cond,
                args.filter_var,
            )
            filter_idx.extend(idx_)

    if len(filter_idx) > 0:
        # apply condition filter to all datasets
        filter_idx = list(set(filter_idx))
        for i in range(len(tgt_datasets)):
            t = i + 1
            tgt_dataset = tgt_datasets[f'tgt_dataset_t{t}']
            tgt_adata = tgt_adatas[f'tgt_h5ad_t{t}']
            tgt_dataset = tgt_dataset.select(filter_idx)
            tgt_datasets[f'tgt_dataset_t{t}'] = tgt_dataset
            tgt_adata = tgt_adata[filter_idx, :]
            tgt_adatas[f'tgt_h5ad_t{t}'] = tgt_adata
        src_dataset = src_dataset.select(filter_idx)
        src_adata = src_adata[filter_idx, :]
        if args.gene_embs_condition is not None:
            # filter for keys which are in pred_tps
            pred_dataset = {
                key: tgt_datasets[key]
                for key in tgt_datasets
                if key in {f'tgt_dataset_t{tp}' for tp in args.pred_tps}
            }
            all_pred_dataset = concatenate_datasets(list(pred_dataset.values()))
            # check if filter_cond is the same as all_pred_dataset
            all_pred_condition = all_pred_dataset.unique(args.filter_var)
            if all_pred_condition != args.filter_cond:
                warnings.warn(
                    f'Filtering gene embs for {all_pred_condition} '
                    f'instead of only {args.filter_cond} '
                    f'if this is not the intended behaviour, please select the '
                    f'corresponding pred_tps for the gene embs condition '
                )
            gene_embs_list = all_pred_dataset.unique(args.gene_embs_condition)
            print(
                f'Return gene embs for {gene_embs_list} '
                f'in condition {args.gene_embs_condition} '
                f'filtered for {all_pred_condition}.'
            )
        else:
            gene_embs_list = None
    else:
        if args.gene_embs_condition is not None:
            pred_dataset = {
                key: tgt_datasets[key]
                for key in tgt_datasets
                if key in {f'tgt_dataset_t{tp}' for tp in args.pred_tps}
            }
            all_pred_dataset = concatenate_datasets(list(pred_dataset.values()))
            # check if filter_cond is the same as all_pred_dataset
            gene_embs_list = all_pred_dataset.unique(args.gene_embs_condition)
            print(
                f'Return gene embs for {gene_embs_list} '
                f'in {args.gene_embs_condition}.'
            )
        else:
            gene_embs_list = None
    # use the tmp adata for all operation
    # where the metadata and information is shared across timepoints
    tgt_adata_tmp = tgt_adatas[f'tgt_h5ad_t{args.pred_tps[0]}'].copy()

    # ZINB and NB count loss preprocessing
    # ------------------------------------

    (
        conditions,
        condition_encodings,
        conditions_combined,
        conditions_,
        condition_keys_,
        conditions_combined_,
    ) = condition_for_count_loss(
        args.condition_keys, args.conditions, args.conditions_combined, tgt_adata_tmp
    )

    print('Data loaded and preprocessed.')
    # Preparing train-test split
    # --------------------------
    if args.split:
        if args.split_path:
            train_indices, val_indices, test_indices = load_frozen_split(
                args.split_path
            )
            print(f'Loaded frozen split from {args.split_path}')
        elif args.splitting_mode == 'stratified':
            # start preprocessing to avoid loading anndata into datamodule
            train_indices, val_indices, test_indices = stratified_split(
                tgt_adata=tgt_adata_tmp,
                train_prop=args.train_prop,  # 0.8,0.1,0.1 train, val, test
                test_prop=args.test_prop,
                groups=args.split_obs,
                seed=args.seed,
            )
        elif args.splitting_mode == 'random':
            train_indices, val_indices, test_indices = randomised_split(
                adata=tgt_adata_tmp,
                train_prop=args.train_prop,  # 0.8,0.1,0.1 train, val, test
                test_prop=args.test_prop,
                seed=args.seed,
            )
        else:
            raise ValueError(
                "split is not available, must be either '"
                "random','stratified' or 'unseen_donor'"
            )
        # check that indices are unique to avoid data leakage
        assert len(set(train_indices).intersection(val_indices)) == 0
        assert len(set(train_indices).intersection(test_indices)) == 0
        assert len(set(val_indices).intersection(test_indices)) == 0
        print(
            f'Number of samples in train set: {len(train_indices)}\n'
            f'Number of samples in val set: {len(val_indices)}\n'
            f'Number of samples in test set: {len(test_indices)}'
        )
    else:
        # return all the indices
        train_indices = list(range(len(src_dataset)))
        val_indices = None
        test_indices = list(
            range(len(tgt_datasets[f'tgt_dataset_t{args.pred_tps[0]}']))
        )
    # check if the train indices are the same for both adata and dataset
    subset_adata = tgt_adata_tmp[train_indices]
    subset_dataset = tgt_datasets[f'tgt_dataset_t{args.pred_tps[0]}'].select(
        train_indices
    )
    adata_idx = subset_adata.obs['cell_pairing_index'].astype(str)
    adata_idx = adata_idx.tolist()
    dataset_idx = list(map(str, subset_dataset['cell_pairing_index']))
    assert adata_idx == dataset_idx, (
        'Cell pairing indices do not match ' 
        'between AnnData and Dataset objects'
    )
    # count number of unique timepoints
    n_total_tps = len(tgt_adatas)
    # Initialize model module
    # ----------------------------------------------------------------------------------
    # Prefer checkpoint architecture when available (must match weight shapes).
    max_seq_length = int(ckpt_hp.get('max_seq_length', max_len + 100))
    tgt_vocab_size = int(ckpt_hp.get('tgt_vocab_size', max_tgt_input_id + 50))
    num_heads = int(getattr(args, 'num_heads', 8))
    test_kwargs = {
        'tgt_vocab_size': tgt_vocab_size,
        'd_model': args.d_model,
        'num_heads': num_heads,
        'num_layers': args.num_layers,
        'd_ff': args.d_ff,
        'max_seq_length': max_seq_length,
        'dropout': 0,
        'generate': args.generate,
        'context_tps': args.context_tps,
        'pred_tps': args.pred_tps,
        'n_total_tps': n_total_tps,
        'mask_scheduler': args.mask_scheduler,
        'pos_encoding_mode': args.pos_encoding_mode,
        'output_dir': args.output_dir,
        'encoder': args.encoder,
        'var_list': args.var_list,
        'encoder_path': args.encoder_path,
        'condition_dict': condition_dict,
        'temperature': args.temperature,
        'iterations': args.iterations,
        'sequence_length': args.sequence_length,
        'mapping_dict_path': args.mapping_dict_path,
        'seed': args.seed,
    }
    if args.test_mode == 'masking':
        test_kwargs['weight_decay'] = args.cellgen_wd
        test_kwargs['end_lr'] = args.cellgen_lr
        test_kwargs['return_embeddings'] = args.return_embeddings
        test_kwargs['return_gene_embs'] = args.return_gene_embs
        test_kwargs['gene_names'] = tgt_adata_tmp.var['gene_name']
        test_kwargs['context_mode'] = args.context_mode
        test_kwargs['return_attn'] = args.return_attn
        test_kwargs['tokenid_to_rowid_path'] = args.tokenid_to_rowid_path
        test_kwargs['deg_pkl_path'] = args.deg_pkl_path
        test_kwargs['gene_embs_list'] = gene_embs_list
        test_kwargs['gene_embs_condition'] = args.gene_embs_condition
        # Rouge is only useful for generation; loading evaluate/rouge can hang headless.
        test_kwargs['return_rouge_score'] = bool(args.generate)
        pretrained_module = PerturbGenTrainer(**test_kwargs)

    elif args.test_mode == 'count':
        test_kwargs['ckpt_masking_path'] = args.ckpt_masking_path
        test_kwargs['ckpt_count_path'] = args.ckpt_count_path
        test_kwargs['loss_mode'] = args.loss_mode
        test_kwargs['layer_norm'] = args.layer_norm
        test_kwargs['dropout'] = args.count_dropout
        test_kwargs['use_positional_encoding'] = args.use_positional_encoding
        test_kwargs['weight_decay'] = args.count_wd
        test_kwargs['lr'] = args.count_lr
        test_kwargs['conditions'] = conditions_
        test_kwargs['conditions_combined'] = conditions_combined_
        test_kwargs['tgt_adata'] = tgt_adatas
        test_kwargs['n_samples'] = args.n_samples
        test_kwargs['n_genes'] = src_adata.shape[1]
        decoder_module = CountDecoderTrainer(**test_kwargs)
    else:
        raise ValueError('test_mode not recognised, needs to be masking or count')

    # Initialize data module
    # ----------------------------------------------------------------------------------

    # While there is a wide variety of different augmentation strategies, we simply
    # resort to the supposedly optimal AutoAugment policy.
    # change dataloader and input
    # create count dictionnary
    tgt_counts_dict = {}
    for keys, tgt_adata in tgt_adatas.items():
        tgt_counts_dict[keys] = tgt_adata.X
    src_counts = src_adata.X
    data_module_kwargs = {
        'src_dataset': src_dataset,
        'tgt_datasets': tgt_datasets,
        'batch_size': args.batch_size,
        'num_workers': args.n_workers,
        'shuffle': args.shuffle,
        'max_len': max_len,
        'split': args.split,
        'src_counts': src_counts,
        'tgt_counts_dict': tgt_counts_dict,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'pred_tps': args.pred_tps,
        'context_tps': args.context_tps,
        'n_total_tps': n_total_tps,
        'var_list': args.var_list,
        'condition_keys': condition_keys_,
        'condition_encodings': condition_encodings,
        'conditions': conditions,
        'conditions_combined': conditions_combined,
        'use_weighted_sampler': False,
    }

    data_module = PerturbGenDataModule(
        **data_module_kwargs,
    )
    # Setup trainer
    # ----------------------------------------------------------------------------------
    run_id = datetime.now().strftime('%Y%m%d_%H%M_PerturbGen')
    # Prefer CSV logger for local metrics. Only attach WandB when explicitly enabled.
    # Always creating WandbLogger (even mode=disabled) is what triggered
    # X11 "No protocol specified" during Trainer construction on this host.
    loggers = [CSVLogger(save_dir='logs', name=run_id)]
    if args.wandb_mode != 'disabled':
        log_path = os.path.join('./T_perturb/perturbgen/wandb/wandb', run_id)
        os.makedirs(os.path.join(os.getcwd(), log_path), exist_ok=True)
        wandb_name = (
            f'{run_id}_{str(uuid.uuid4())[:6]}'
            if torch.cuda.device_count() > 1
            else run_id
        )
        loggers.insert(
            0,
            WandbLogger(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=wandb_name,
                save_dir='./T_perturb/perturbgen/wandb/wandb',
                log_model=False,
                mode=args.wandb_mode,
            ),
        )
    print(f'CSV metrics: {loggers[-1].log_dir}/metrics.csv')

    # In this simple example we just check if a GPU is available.
    # For training larger models in a distributed settings, this needs more care.
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    # Force single-process test: multi-GPU DDP on this host reintroduces MPI/X11 hangs.
    n_devices = 1
    print('Using device {} (n_devices={}).'.format(accelerator, n_devices))

    # Instantiate trainer object.
    # Force LightningEnvironment so MPIEnvironment.detect() never imports mpi4py.
    _force_headless_env()
    _disable_lightning_mpi_autodetect()
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=[TQDMProgressBar(refresh_rate=1)],
        accelerator=accelerator,
        num_nodes=1,
        devices=n_devices,
        strategy='auto',
        plugins=[LightningEnvironment()],
        enable_progress_bar=True,
    )
    # Finally, kick of the training process.
    if args.test_mode == 'masking':
        if args.ckpt_masking_path is not None:
            # check if masking_path ends with .bin
            if args.ckpt_masking_path.endswith('.bin'):
                # load the model from the bin file
                state_dict = torch.load(args.ckpt_masking_path, map_location='cpu')
                pretrained_module.load_state_dict(state_dict, strict=False)
                trainer.test(
                    pretrained_module,
                    data_module,
                )
            else:
                trainer.test(
                    pretrained_module,
                    data_module,
                    ckpt_path=args.ckpt_masking_path,
                )

    elif args.test_mode == 'count':
        trainer.test(
            decoder_module,
            data_module,
        )
    else:
        raise ValueError('test_mode not recognised, needs to be masking or count')


if __name__ == '__main__':
    main()
