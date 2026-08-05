#!/usr/bin/env python3
"""Phase A diagnostic overfit / smoke test (2026-08-03).

Debug-only helper before full-data sod2 runs. Prefer
``run_train_jepa_sod2.sh`` for real ablations. See ``JEPA_README.md``.

Matches the student / EMA-teacher / predictor scheme:

  x_src  -> f_theta (student) -> z_src -> p_phi(z_src, t) -> z_hat
  x_tgt  -> f_xi   (EMA teacher, stop-grad)               -> z_tgt
  L = MSE(z_hat, sg(z_tgt));  xi <- beta*xi + (1-beta)*theta

Flexible CLI:
  --batch_size N
  --num_train_batches M   (train M batches each epoch)
  --num_val_batches K
  --sample_mode {contiguous,strided,random}

Example (single GPU):
  CUDA_VISIBLE_DEVICES=1 python docs/examples/jepa_phase_a_overfit.py \\
    --sample_mode per_class --jepa_loss cosine --epochs 200 --device cuda:0

Example (6 GPUs via torchrun):
  CUDA_VISIBLE_DEVICES=0,1,2,5,6,7 torchrun --standalone --nproc_per_node=6 \\
    docs/examples/jepa_phase_a_overfit.py --sample_mode per_class \\
    --jepa_loss cosine --epochs 200
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel as DDP

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO))

from perturbgen.Modules.jepa import CellTrajectoryJEPA  # noqa: E402
from perturbgen.src.gf_utils import pad_tensor_list  # noqa: E402
from perturbgen.src.jepa_metrics import (  # noqa: E402
    latent_collapse_stats,
    pairwise_latent_mse,
    trajectory_baselines,
)
from perturbgen.src.jepa_token_maps import apply_id_lookup, maybe_load_maps  # noqa: E402

TOKENIZED = WORKSPACE / "T_perturb" / "tokenized_data" / "LPS_all_tps_2k"
DEFAULT_CKPT = (
    REPO
    / "pretraining_cohort"
    / "20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt"
)
DEFAULT_OUT = Path(
    "/mnt/sod2-project/csb4/stuke1/perturbgen/T_perturb/res/jepa/phaseA_overfit"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--encoder_path", type=Path, default=DEFAULT_CKPT)
    p.add_argument(
        "--tokenid_to_rowid_path",
        type=Path,
        default=TOKENIZED / "tokenid_to_rowid_2000_hvg.pkl",
    )
    p.add_argument(
        "--src_dataset",
        type=Path,
        default=TOKENIZED / "dataset_2000_hvg_src" / "normal.dataset",
    )
    p.add_argument(
        "--tgt_dataset_folder",
        type=Path,
        default=TOKENIZED / "dataset_2000_hvg_tgt",
    )
    p.add_argument("--pred_tps", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--num_train_batches",
        type=int,
        default=1,
        help="Train minibatches/epoch for contiguous/strided/random. "
        "Ignored for sample_mode=per_class (set by --min_batches_per_class).",
    )
    p.add_argument(
        "--num_val_batches",
        type=int,
        default=1,
        help="Val minibatches for contiguous/strided/random. "
        "Ignored for sample_mode=per_class (set by --val_batches_per_class).",
    )
    p.add_argument(
        "--sample_mode",
        type=str,
        choices=("contiguous", "strided", "random", "per_class"),
        default="per_class",
        help="How to pick rows. per_class: >=N pure batches per cell type "
        "(recommended; avoids all-B-cell contiguous slices).",
    )
    p.add_argument(
        "--min_batches_per_class",
        type=int,
        default=2,
        help="For per_class: train batches per cell_type (pure type batches).",
    )
    p.add_argument(
        "--val_batches_per_class",
        type=int,
        default=1,
        help="For per_class: held-out val batches per cell_type (disjoint rows).",
    )
    p.add_argument(
        "--cell_type_key",
        type=str,
        default="cell_type_harmonized",
        help="Dataset column used for per_class stratification.",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema_decay", type=float, default=0.996)
    p.add_argument(
        "--jepa_loss",
        type=str,
        choices=("mse", "smooth_l1", "cosine"),
        default="mse",
        help="Latent loss: mse | smooth_l1 | cosine (minimize 1-cos ⇔ max cosine).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--train_start",
        type=int,
        default=0,
        help="row offset for first train batch (contiguous mode)",
    )
    p.add_argument(
        "--val_start",
        type=int,
        default=10_000,
        help="row offset for first val batch (contiguous mode)",
    )
    p.add_argument("--log_every", type=int, default=1, help="log/eval every N epochs")
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Used for single-process runs. Ignored under torchrun (uses LOCAL_RANK).",
    )
    p.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="optional; default is trainable encoder (learn)",
    )
    return p.parse_args()


def setup_distributed() -> tuple[int, int, int, bool]:
    """Return (rank, local_rank, world_size, distributed)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 0, 1, False
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, True


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def shard_batch(
    batch: dict, rank: int, world_size: int, device: torch.device
) -> dict:
    """Split tensor fields along dim-0 for this rank; move shard to device."""
    bsz = batch["src_input_ids"].size(0)
    if world_size <= 1:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(device) if torch.is_tensor(v) else v
        return out
    # even split with remainder on lower ranks
    base, rem = divmod(bsz, world_size)
    sizes = [base + (1 if r < rem else 0) for r in range(world_size)]
    start = sum(sizes[:rank])
    end = start + sizes[rank]
    if end <= start:
        # empty shard — give 1 duplicate row so DDP forward still runs
        start, end = 0, 1
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v[start:end].to(device, non_blocking=True)
        elif isinstance(v, list):
            out[k] = v[start:end]
        else:
            out[k] = v
    return out


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _pad_ids(seqs: list[list[int]], pad_id: int = 0) -> torch.Tensor:
    tensors = [torch.as_tensor(s, dtype=torch.long) for s in seqs]
    max_len = max(t.numel() for t in tensors)
    return pad_tensor_list(tensors, max_len, pad_id, max_len)


def _strip_cls_eos(ids: list[int], cls_id: int = 2, eos_id: int = 3) -> list[int]:
    return [x for x in ids if x not in (cls_id, eos_id)]


def make_index_blocks(
    n: int,
    batch_size: int,
    num_batches: int,
    mode: str,
    start: int,
    seed: int,
    exclude: set[int] | None = None,
) -> list[list[int]]:
    """Return list of index lists, each of length batch_size."""
    need = batch_size * num_batches
    if need > n:
        raise ValueError(
            f"Need {need} rows for {num_batches} x batch_size={batch_size}, "
            f"but dataset has only {n}"
        )
    exclude = exclude or set()
    g = torch.Generator().manual_seed(seed)

    if mode == "contiguous":
        if start + need > n:
            raise ValueError(
                f"contiguous block [{start}, {start+need}) exceeds n={n}"
            )
        idxs = list(range(start, start + need))
    elif mode == "strided":
        stride = max(1, n // need)
        idxs = []
        pos = start % n
        while len(idxs) < need:
            if pos not in exclude:
                idxs.append(pos)
            pos = (pos + stride) % n
            if len(idxs) > need * 3:
                for i in range(n):
                    if i not in exclude and i not in idxs:
                        idxs.append(i)
                    if len(idxs) >= need:
                        break
                break
        idxs = idxs[:need]
    elif mode == "random":
        perm = torch.randperm(n, generator=g).tolist()
        idxs = [i for i in perm if i not in exclude][:need]
        if len(idxs) < need:
            raise ValueError("Not enough rows after exclude for random sampling")
    else:
        raise ValueError(mode)

    return [
        idxs[b * batch_size : (b + 1) * batch_size] for b in range(num_batches)
    ]


def make_per_class_blocks(
    cell_types: list[str],
    batch_size: int,
    min_train_per_class: int,
    val_per_class: int,
    seed: int,
) -> tuple[list[list[int]], list[list[int]], dict]:
    """Build pure cell-type batches: >= min_train_per_class train (+ val) per type.

    Returns (train_blocks, val_blocks, report).
    Types with fewer than batch_size cells are skipped.
    If a type has fewer than (min_train+val)*batch_size cells, take as many
    full batches as possible (train first, then val).
    """
    from collections import defaultdict

    by_type: dict[str, list[int]] = defaultdict(list)
    for i, ct in enumerate(cell_types):
        by_type[str(ct)].append(i)

    g = torch.Generator().manual_seed(seed)
    train_blocks: list[list[int]] = []
    val_blocks: list[list[int]] = []
    report: dict = {"per_class": {}, "skipped": {}}

    for ct in sorted(by_type.keys()):
        idxs = by_type[ct]
        # shuffle within class
        perm = torch.randperm(len(idxs), generator=g).tolist()
        idxs = [idxs[j] for j in perm]
        n_full = len(idxs) // batch_size
        if n_full < 1:
            report["skipped"][ct] = {
                "n_cells": len(idxs),
                "reason": f"need >= {batch_size} cells for one batch",
            }
            continue

        want_train = min_train_per_class
        want_val = val_per_class
        # Prefer satisfying train quota, then val, with whatever full batches exist.
        n_train = min(want_train, n_full)
        n_val = min(want_val, max(0, n_full - n_train))
        # If we still have spare full batches and train < want, already handled;
        # if train was reduced but we could trade val for train when want_train>n_train:
        if n_train < want_train and n_val > 0:
            # already prioritized train above
            pass

        cursor = 0
        class_train = []
        class_val = []
        for _ in range(n_train):
            block = idxs[cursor : cursor + batch_size]
            train_blocks.append(block)
            class_train.append(len(block))
            cursor += batch_size
        for _ in range(n_val):
            block = idxs[cursor : cursor + batch_size]
            val_blocks.append(block)
            class_val.append(len(block))
            cursor += batch_size

        report["per_class"][ct] = {
            "n_cells": len(by_type[ct]),
            "n_full_available": n_full,
            "n_train_batches": n_train,
            "n_val_batches": n_val,
            "met_min_train": n_train >= want_train,
        }
        if n_train < want_train:
            print(
                f"WARNING: {ct}: only {n_train}/{want_train} train batches "
                f"(cells={len(by_type[ct])}, bs={batch_size})"
            )

    if not train_blocks:
        raise RuntimeError("per_class sampling produced zero train batches")
    return train_blocks, val_blocks, report


def load_batch_by_indices(
    src_ds,
    tgt_by_t: dict[int, object],
    indices: list[int],
    pred_tps: list[int],
    local_to_global: torch.Tensor | None,
    device: torch.device | None = None,
) -> dict:
    """Build one JEPA batch from explicit row indices; remap tgt local -> global.

    Tensors stay on CPU unless ``device`` is set (single-GPU convenience).
    """
    src_ids = [src_ds[i]["input_ids"] for i in indices]
    batch = {"src_input_ids": _pad_ids(src_ids), "indices": indices}
    for t in pred_tps:
        raw = [_strip_cls_eos(tgt_by_t[t][i]["input_ids"]) for i in indices]
        ids = _pad_ids(raw)
        if local_to_global is not None:
            ids = apply_id_lookup(ids, local_to_global)
        batch[f"tgt_input_ids_t{t}"] = ids
        batch[f"cell_type_t{t}"] = [
            tgt_by_t[t][i]["cell_type_harmonized"] for i in indices
        ]
        batch[f"time_after_LPS_t{t}"] = [
            tgt_by_t[t][i]["time_after_LPS"] for i in indices
        ]
    batch["src_cell_type"] = [src_ds[i]["cell_type_harmonized"] for i in indices]
    batch["src_time"] = [src_ds[i]["time_after_LPS"] for i in indices]
    if device is not None:
        batch = move_batch_to_device(batch, device)
    return batch


def latent_loss(
    z_hat: torch.Tensor, z_tgt: torch.Tensor, loss_type: str
) -> torch.Tensor:
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(z_hat, z_tgt)
    if loss_type == "cosine":
        return (1.0 - F.cosine_similarity(z_hat, z_tgt, dim=-1)).mean()
    return F.mse_loss(z_hat, z_tgt)


def _mean_metric_dicts(dicts: list[dict]) -> dict:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts]
        if isinstance(vals[0], bool):
            out[k] = bool(sum(vals) > len(vals) / 2)
        elif isinstance(vals[0], (int, float)):
            out[k] = float(sum(vals) / len(vals))
        else:
            out[k] = vals[0]
    return out


@torch.no_grad()
def eval_batch(
    model: CellTrajectoryJEPA,
    batch: dict,
    pred_tps: list[int],
    loss_type: str = "mse",
) -> dict:
    model.eval()
    tgt = {f"tgt_input_ids_t{t}": batch[f"tgt_input_ids_t{t}"] for t in pred_tps}
    outs = model(batch["src_input_ids"], tgt, pred_tps=pred_tps)
    losses = []
    z_hats, z_tgts, z_srcs = [], [], []
    per_t = {}
    for t, out in outs.items():
        lt = latent_loss(out["z_hat"], out["z_tgt"], loss_type)
        losses.append(lt)
        z_hats.append(out["z_hat"])
        z_tgts.append(out["z_tgt"])
        z_srcs.append(out["z_src"])
        m = pairwise_latent_mse(out["z_hat"], out["z_tgt"])
        per_t[f"loss_t{t}"] = float(lt.item())
        per_t[f"cosine_t{t}"] = float(m["cosine"])
    loss = torch.stack(losses).mean()
    z_hat = torch.cat(z_hats, dim=0)
    z_tgt = torch.cat(z_tgts, dim=0)
    metrics = pairwise_latent_mse(z_hat, z_tgt)
    collapse = latent_collapse_stats(z_tgt)
    t0 = next(iter(outs))
    bases = trajectory_baselines(
        outs[t0]["z_src"], outs[t0]["z_tgt"], outs[t0]["z_hat"]
    )
    if loss_type == "cosine":
        id_base = bases["identity"]["cosine_loss"]
        jepa_base = bases["jepa"]["cosine_loss"]
    else:
        id_base = bases["identity"]["mse"]
        jepa_base = bases["jepa"]["mse"]
    return {
        "loss": float(loss.item()),
        "latent_cosine": float(metrics["cosine"]),
        "collapse_mean_cosine": float(collapse["mean_cosine"]),
        "collapse_std_mean": float(collapse["std_mean"]),
        "baseline_identity_mse": float(bases["identity"]["mse"]),
        "baseline_jepa_mse": float(bases["jepa"]["mse"]),
        "baseline_identity": float(id_base),
        "baseline_jepa": float(jepa_base),
        "beats_identity": bool(jepa_base < id_base),
        **per_t,
    }


def train_one_batch(
    model: torch.nn.Module,
    batch: dict,
    pred_tps: list[int],
    optimizer: torch.optim.Optimizer,
    loss_type: str = "mse",
    rank: int = 0,
    world_size: int = 1,
    device: torch.device | None = None,
) -> dict:
    model.train()
    if device is None:
        device = batch["src_input_ids"].device
    local = shard_batch(batch, rank, world_size, device)
    tgt = {f"tgt_input_ids_t{t}": local[f"tgt_input_ids_t{t}"] for t in pred_tps}
    outs = model(local["src_input_ids"], tgt, pred_tps=pred_tps)
    # DDP wraps Module; CellTrajectoryJEPA.forward returns dict[int, dict]
    losses = []
    z_hats, z_tgts = [], []
    for t, out in outs.items():
        lt = latent_loss(out["z_hat"], out["z_tgt"], loss_type)
        losses.append(lt)
        z_hats.append(out["z_hat"])
        z_tgts.append(out["z_tgt"])
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    unwrap(model).update_target_encoder()

    with torch.no_grad():
        metrics = pairwise_latent_mse(torch.cat(z_hats), torch.cat(z_tgts))
        collapse = latent_collapse_stats(torch.cat(z_tgts))
    return {
        "loss": float(loss.item()),
        "latent_cosine": float(metrics["cosine"]),
        "collapse_mean_cosine": float(collapse["mean_cosine"]),
        "collapse_std_mean": float(collapse["std_mean"]),
    }


def train_one_epoch(
    model: torch.nn.Module,
    train_batches: list[dict],
    pred_tps: list[int],
    optimizer: torch.optim.Optimizer,
    loss_type: str = "mse",
    rank: int = 0,
    world_size: int = 1,
    device: torch.device | None = None,
) -> dict:
    stats = []
    for batch in train_batches:
        stats.append(
            train_one_batch(
                model,
                batch,
                pred_tps,
                optimizer,
                loss_type,
                rank=rank,
                world_size=world_size,
                device=device,
            )
        )
    return _mean_metric_dicts(stats)


def eval_batches(
    model: torch.nn.Module,
    batches: list[dict],
    pred_tps: list[int],
    loss_type: str = "mse",
    device: torch.device | None = None,
) -> dict:
    """Eval on full batches (no shard). Call on rank 0 only under DDP."""
    core = unwrap(model)
    moved = [
        move_batch_to_device(b, device) if device is not None else b for b in batches
    ]
    return _mean_metric_dicts(
        [eval_batch(core, b, pred_tps, loss_type) for b in moved]
    )


def plot_metrics(csv_path: Path, out_png: Path, title: str) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], label="train", lw=1.5)
    ax.plot(df["epoch"], df["val_loss"], label="val", lw=1.5)
    ax.set_title("JEPA latent loss")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df["epoch"], df["train_latent_cosine"], label="train", lw=1.5)
    ax.plot(df["epoch"], df["val_latent_cosine"], label="val", lw=1.5)
    ax.set_title("Latent cosine (z_hat vs z_tgt)")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    id_col = (
        "val_baseline_identity"
        if "val_baseline_identity" in df.columns
        else "val_baseline_identity_mse"
    )
    jepa_col = (
        "val_baseline_jepa"
        if "val_baseline_jepa" in df.columns
        else "val_baseline_jepa_mse"
    )
    ax.plot(df["epoch"], df[id_col], label="identity", lw=1.5)
    ax.plot(df["epoch"], df[jepa_col], label="jepa", lw=1.5)
    ax.set_title("Val: JEPA vs identity baseline")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(df["epoch"], df["val_collapse_mean_cosine"], label="val collapse cos", lw=1.5)
    ax.plot(
        df["epoch"], df["train_collapse_mean_cosine"], label="train collapse cos", lw=1.5
    )
    ax.set_title("Collapse mean cosine")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote plot: {out_png}")


def _cell_type_hist(batch: dict) -> dict[str, int]:
    from collections import Counter

    return dict(Counter(batch["src_cell_type"]))


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, distributed = setup_distributed()
    is_main = rank == 0

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    tgt_by_t: dict[int, object] = {}
    for p in sorted(args.tgt_dataset_folder.glob("*.dataset")):
        t = int(p.name[0])
        if t in args.pred_tps:
            tgt_by_t[t] = load_from_disk(str(p))
    src_ds = load_from_disk(str(args.src_dataset))
    missing = [t for t in args.pred_tps if t not in tgt_by_t]
    if missing:
        raise FileNotFoundError(f"Missing tgt datasets for t={missing}")

    n = len(src_ds)
    per_class_report = None
    if args.sample_mode == "per_class":
        if is_main:
            print(f"Reading {args.cell_type_key} for per-class sampling...")
        cell_types = list(src_ds[args.cell_type_key])
        train_blocks, val_blocks, per_class_report = make_per_class_blocks(
            cell_types=cell_types,
            batch_size=args.batch_size,
            min_train_per_class=args.min_batches_per_class,
            val_per_class=args.val_batches_per_class,
            seed=args.seed,
        )
        args.num_train_batches = len(train_blocks)
        args.num_val_batches = len(val_blocks)
        if is_main:
            print(
                f"per_class: {args.num_train_batches} train batches, "
                f"{args.num_val_batches} val batches "
                f"(min_train/class={args.min_batches_per_class}, "
                f"val/class={args.val_batches_per_class}, bs={args.batch_size})"
            )
    else:
        train_blocks = make_index_blocks(
            n=n,
            batch_size=args.batch_size,
            num_batches=args.num_train_batches,
            mode=args.sample_mode,
            start=args.train_start,
            seed=args.seed,
        )
        train_idx_set = {i for block in train_blocks for i in block}
        val_blocks = make_index_blocks(
            n=n,
            batch_size=args.batch_size,
            num_batches=args.num_val_batches,
            mode=args.sample_mode,
            start=args.val_start,
            seed=args.seed + 1,
            exclude=train_idx_set,
        )

    _, local_to_global = maybe_load_maps(str(args.tokenid_to_rowid_path))
    if local_to_global is None:
        raise RuntimeError("need tokenid_to_rowid map for scmaskgit")

    if is_main:
        print(
            f"Loading {len(train_blocks)} train + {len(val_blocks)} val "
            f"batches (bs={args.batch_size}, mode={args.sample_mode}, "
            f"world_size={world_size})..."
        )
    train_batches = [
        load_batch_by_indices(
            src_ds, tgt_by_t, block, args.pred_tps, local_to_global, device=None
        )
        for block in train_blocks
    ]
    val_batches = [
        load_batch_by_indices(
            src_ds, tgt_by_t, block, args.pred_tps, local_to_global, device=None
        )
        for block in val_blocks
    ]

    from collections import Counter

    pooled_train_types = Counter()
    for b in train_batches:
        pooled_train_types.update(b["src_cell_type"])
    align_report = {
        "train_src_times": sorted({t for b in train_batches for t in b["src_time"]}),
        "val_src_times": sorted({t for b in val_batches for t in b["src_time"]}),
        "train_cell_type_hist": dict(pooled_train_types),
        "val_cell_type_hist": dict(
            sum((Counter(b["src_cell_type"]) for b in val_batches), Counter())
        ),
        "n_unique_train_types": len(pooled_train_types),
        "train_batches_per_type": dict(
            Counter(b["src_cell_type"][0] for b in train_batches)
        ),
        "val_batches_per_type": dict(
            Counter(b["src_cell_type"][0] for b in val_batches)
        ),
        "per_class_sampling": per_class_report,
        "train_cell_type_mismatch": {},
        "val_cell_type_mismatch": {},
        "world_size": world_size,
    }
    for t in args.pred_tps:
        align_report["train_cell_type_mismatch"][t] = sum(
            sum(a != b for a, b in zip(batch["src_cell_type"], batch[f"cell_type_t{t}"]))
            for batch in train_batches
        )
        align_report["val_cell_type_mismatch"][t] = sum(
            sum(a != b for a, b in zip(batch["src_cell_type"], batch[f"cell_type_t{t}"]))
            for batch in val_batches
        )

    model = CellTrajectoryJEPA(
        vocab_size=19000,
        d_model=768,
        n_total_tps=3,
        pred_tps=args.pred_tps,
        ema_decay=args.ema_decay,
        normalize_latents=True,
        encoder_type="scmaskgit",
        encoder_path=str(args.encoder_path),
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    core = unwrap(model)
    n_student = sum(p.requires_grad for p in core.context_encoder.parameters())
    n_teacher = sum(p.requires_grad for p in core.target_encoder.parameters())
    n_pred = sum(p.requires_grad for p in core.predictor.parameters())
    if is_main:
        print(
            f"EMA scheme: student_trainable_tensors={n_student}, "
            f"teacher_trainable_tensors={n_teacher} (want 0), "
            f"predictor_trainable={n_pred}, freeze_encoder={args.freeze_encoder}, "
            f"ddp={distributed}, world_size={world_size}"
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    csv_path = args.output_dir / "metrics.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_latent_cosine",
        "val_latent_cosine",
        "train_collapse_mean_cosine",
        "val_collapse_mean_cosine",
        "train_collapse_std_mean",
        "val_collapse_std_mean",
        "val_baseline_identity_mse",
        "val_baseline_jepa_mse",
        "val_baseline_identity",
        "val_baseline_jepa",
        "val_beats_identity",
        "epoch_sec",
    ]
    for t in args.pred_tps:
        fieldnames += [f"train_loss_t{t}", f"val_loss_t{t}", f"val_cosine_t{t}"]

    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": str(device),
        "distributed": distributed,
        "world_size": world_size,
        "scheme": "student_encoder + EMA_teacher + time_predictor; latent loss",
        "align_report": align_report,
        "n_student_trainable": n_student,
        "n_teacher_trainable": n_teacher,
        "n_predictor_trainable": n_pred,
        "n_train_rows": args.batch_size * args.num_train_batches,
        "n_val_rows": args.batch_size * args.num_val_batches,
    }
    if is_main:
        with open(args.output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(json.dumps(align_report, indent=2))
        print(
            f"Phase A: bs={args.batch_size}, train_batches={args.num_train_batches}, "
            f"val_batches={args.num_val_batches}, epochs={args.epochs}, "
            f"sample_mode={args.sample_mode}, jepa_loss={args.jepa_loss}, "
            f"gpus={world_size}"
        )

    t0 = time.time()
    csv_file = open(csv_path, "w", newline="") if is_main else None
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames) if is_main else None
    if writer is not None:
        writer.writeheader()

    try:
        for epoch in range(args.epochs):
            ep_t0 = time.time()
            if distributed:
                dist.barrier()
            tr_step = train_one_epoch(
                model,
                train_batches,
                args.pred_tps,
                optimizer,
                args.jepa_loss,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            if epoch % args.log_every != 0 and epoch != args.epochs - 1:
                if distributed:
                    dist.barrier()
                continue
            if is_main:
                tr_full = eval_batches(
                    model, train_batches, args.pred_tps, args.jepa_loss, device=device
                )
                va = eval_batches(
                    model, val_batches, args.pred_tps, args.jepa_loss, device=device
                )
                row = {
                    "epoch": epoch,
                    "train_loss": tr_full["loss"],
                    "val_loss": va["loss"],
                    "train_latent_cosine": tr_full["latent_cosine"],
                    "val_latent_cosine": va["latent_cosine"],
                    "train_collapse_mean_cosine": tr_step["collapse_mean_cosine"],
                    "val_collapse_mean_cosine": va["collapse_mean_cosine"],
                    "train_collapse_std_mean": tr_step["collapse_std_mean"],
                    "val_collapse_std_mean": va["collapse_std_mean"],
                    "val_baseline_identity_mse": va["baseline_identity_mse"],
                    "val_baseline_jepa_mse": va["baseline_jepa_mse"],
                    "val_baseline_identity": va["baseline_identity"],
                    "val_baseline_jepa": va["baseline_jepa"],
                    "val_beats_identity": int(va["beats_identity"]),
                    "epoch_sec": time.time() - ep_t0,
                }
                for t in args.pred_tps:
                    row[f"train_loss_t{t}"] = tr_full.get(f"loss_t{t}")
                    row[f"val_loss_t{t}"] = va.get(f"loss_t{t}")
                    row[f"val_cosine_t{t}"] = va.get(f"cosine_t{t}")
                writer.writerow(row)
                csv_file.flush()
                if (
                    epoch % max(1, args.epochs // 20) == 0
                    or epoch < 5
                    or epoch == args.epochs - 1
                ):
                    print(
                        f"epoch {epoch:4d}  train_loss={row['train_loss']:.6f}  "
                        f"val_loss={row['val_loss']:.6f}  "
                        f"val_cos={row['val_latent_cosine']:.4f}  "
                        f"beats_id={row['val_beats_identity']}  "
                        f"({row['epoch_sec']:.1f}s)"
                    )
            if distributed:
                dist.barrier()
    finally:
        if csv_file is not None:
            csv_file.close()

    if is_main:
        png = args.output_dir / "phaseA_overfit_curves.png"
        plot_metrics(
            csv_path,
            png,
            title=(
                f"Phase A (EMA JEPA, freeze_enc={args.freeze_encoder}, "
                f"bs={args.batch_size}, ntrain={args.num_train_batches}, "
                f"nval={args.num_val_batches}, ep={args.epochs}, "
                f"mode={args.sample_mode}, loss={args.jepa_loss}, "
                f"gpus={world_size})"
            ),
        )
        local_png = Path(__file__).resolve().parent / "jepa_phase_a_overfit_curves.png"
        try:
            local_png.write_bytes(png.read_bytes())
            print(f"Copied plot to {local_png}")
        except OSError as e:
            print(f"Could not copy plot locally: {e}")
        print(f"Done in {(time.time()-t0)/60:.1f} min. metrics={csv_path}")

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
