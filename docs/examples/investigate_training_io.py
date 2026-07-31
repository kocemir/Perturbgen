"""
Investigate what PerturbGen masking training takes as input, and the tensor shapes.

Run from anywhere with the perturbgen venv:

  source /home/stuke1/perturbgen/.venv/bin/activate
  cd /home/stuke1/perturbgen/Perturbgen
  python docs/examples/investigate_training_io.py

Then read the printed shapes and the "Model I/O map" section at the bottom
before diving into Modules/transformer.py component-by-component.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from datasets import load_from_disk

from perturbgen.Dataloaders.datamodule import PerturbGenDataModule
from perturbgen.src.utils import read_dataset_files

# ---------------------------------------------------------------------------
# Paths (same LPS run as notebooks 02 / 03)
# ---------------------------------------------------------------------------
WORKSPACE = Path("/home/stuke1/perturbgen")
TOKENIZED = WORKSPACE / "T_perturb" / "tokenized_data" / "LPS_all_tps_2k"

SRC_DATASET_PATH = TOKENIZED / "dataset_2000_hvg_src" / "normal.dataset"
TGT_DATASET_FOLDER = TOKENIZED / "dataset_2000_hvg_tgt"
MAPPING_PATH = TOKENIZED / "token_id_to_genename_2000_hvg.pkl"

PRED_TPS = [1, 2, 3]
BATCH_SIZE = 4  # small probe batch
D_MODEL = 768   # notebook / paper default


def compute_dims(src_dataset, tgt_datasets: dict) -> dict:
    """Mirror train.py: max_len from src+tgt, tgt_vocab from tgt token ids only."""
    max_tgt_input_id = 0
    max_len = 0
    per_tgt = {}
    for key, dataset in tgt_datasets.items():
        input_ids = dataset["input_ids"]
        lengths = [len(x) for x in input_ids]
        max_id = max(max(x) for x in input_ids)
        max_tgt_input_id = max(max_tgt_input_id, max_id)
        max_len = max(max_len, max(lengths))
        per_tgt[key] = {"n": len(dataset), "max_len": max(lengths), "max_token_id": max_id}

    src_lengths = [len(x) for x in src_dataset["input_ids"]]
    src_max_id = max(max(x) for x in src_dataset["input_ids"])
    max_len = max(max_len, max(src_lengths))

    # train.py: +1 for pad indexing, +50 headroom
    max_tgt_input_id = max_tgt_input_id + 1
    tgt_vocab_size = max_tgt_input_id + 50
    max_seq_length = max_len + 100

    return {
        "n_cells": len(src_dataset),
        "src_max_len": max(src_lengths),
        "src_max_token_id": src_max_id,
        "per_tgt": per_tgt,
        "max_len": max_len,                 # DataModule pad length
        "max_seq_length": max_seq_length,   # positional encoding capacity
        "tgt_vocab_size": tgt_vocab_size,   # decoder embedding / logits dim
        "d_model": D_MODEL,
        "n_total_tps": len(tgt_datasets),
    }


def describe_batch(batch: dict, dims: dict) -> None:
    print("\n=== One training batch (collated) ===")
    print(f"batch keys ({len(batch)}): {sorted(batch.keys())}\n")
    for k in sorted(batch.keys()):
        v = batch[k]
        if torch.is_tensor(v):
            print(f"  {k:30s}  shape={tuple(v.shape)}  dtype={v.dtype}")
        elif v is None:
            print(f"  {k:30s}  None")
        elif isinstance(v, list):
            sample = v[0] if v else None
            print(f"  {k:30s}  list(len={len(v)})  e.g. {sample!r}")
        else:
            print(f"  {k:30s}  type={type(v).__name__}")

    B = batch["src_input_ids"].shape[0]
    L = batch["src_input_ids"].shape[1]
    print("\n=== Shape summary ===")
    print(f"  B (batch)              = {B}")
    print(f"  L (padded seq length)  = {L}   (max_len from data = {dims['max_len']})")
    print(f"  src_input_ids          = [B, L] = [{B}, {L}]")
    for t in PRED_TPS:
        key = f"tgt_input_ids_t{t}"
        print(f"  {key:22s} = [B, L] = {tuple(batch[key].shape)}")
    print(f"  d_model (hidden)       = {dims['d_model']}")
    print(f"  tgt_vocab_size (V)     = {dims['tgt_vocab_size']}")
    print(f"  expected dec_logits    = [B, L, V] = [{B}, {L}, {dims['tgt_vocab_size']}]")
    print(f"  expected dec_embedding = [B, L, d_model] = [{B}, {L}, {dims['d_model']}]")


def print_model_io_map(dims: dict) -> None:
    print(
        f"""
=== Model I/O map (masking) — read this before Modules/transformer.py ===

Data path
  PerturbGenDataset  ->  PerturbGenDataModule.collate  ->  batch dict
  files: perturbgen/Dataloaders/datamodule.py

Trainer path
  PerturbGenTrainer.forward(batch)
    builds tgt_input_id_dict from batch['tgt_input_ids_t{{t}}']
    calls PerturbGen(...)
  files: perturbgen/Model/trainer.py

Model path
  PerturbGen.forward(
      src_input_id,          # [B, L]  Geneformer-scale token ids (source / normal)
      tgt_input_id_dict,     # dict t -> [B, L]  remapped HVG token ids per time
      ...
  )
  files: perturbgen/Modules/transformer.py

What happens (high level)
  1) Encode src_input_id  -> enc_output [B, L, d_model]
  2) Optionally build context from other time points (context_mode)
  3) Mask target tokens (MaskGIT scheduler)
  4) Decode with cross-attention to context
  5) decoder_fc: d_model -> V
     dec_logits [B, L, V], labels [B, L]

Your LPS dims (from this script)
  max_len (L)       = {dims['max_len']}
  max_seq_length    = {dims['max_seq_length']}
  tgt_vocab_size V  = {dims['tgt_vocab_size']}
  d_model           = {dims['d_model']}
  n_total_tps       = {dims['n_total_tps']}
  paired cells      = {dims['n_cells']}

Special tokens (Geneformer dict): pad=0, mask=1, cls=2, eos=3
  (perturbgen/pp/token_dict_gftokens_gc95M.pkl)

Next files to read (component tour)
  1) perturbgen/Dataloaders/datamodule.py     — batch construction
  2) perturbgen/Model/trainer.py             — training_step / loss
  3) perturbgen/Modules/transformer.py       — PerturbGen encode/mask/decode
  4) scmaskgit/Modules/T_model.py            — scmaskgit encoder backbone
"""
    )


def main() -> None:
    print("Loading tokenized LPS datasets...")
    src_dataset = load_from_disk(str(SRC_DATASET_PATH))
    tgt_datasets = read_dataset_files(str(TGT_DATASET_FOLDER), "dataset")
    print("  src:", SRC_DATASET_PATH)
    print("  tgt keys:", sorted(tgt_datasets.keys()))

    with open(MAPPING_PATH, "rb") as f:
        mapping = pickle.load(f)
    print(f"  token_id_to_genename entries: {len(mapping)}")

    dims = compute_dims(src_dataset, tgt_datasets)
    print("\n=== Dimensions computed like train.py ===")
    for k, v in dims.items():
        if k == "per_tgt":
            print("  per_tgt:")
            for tk, tv in v.items():
                print(f"    {tk}: {tv}")
        else:
            print(f"  {k}: {v}")

    n = dims["n_cells"]
    dm = PerturbGenDataModule(
        src_dataset=src_dataset,
        tgt_datasets=tgt_datasets,
        batch_size=BATCH_SIZE,
        num_workers=0,
        max_len=dims["max_len"],
        pred_tps=PRED_TPS,
        context_tps=None,
        n_total_tps=dims["n_total_tps"],
        train_indices=list(range(min(n, 256))),  # small subset for a fast probe
        val_indices=None,
        test_indices=list(range(min(n, 64))),
        var_list=["cell_type_harmonized", "time_after_LPS"],
        use_weighted_sampler=False,
        sampling_keys=None,
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    describe_batch(batch, dims)
    print_model_io_map(dims)

    # Tiny numeric peek at one cell's tokens (unpadded length)
    src_row = batch["src_input_ids"][0]
    src_nnz = int((src_row != 0).sum())
    print("=== Example cell 0 in batch ===")
    print(f"  src non-pad tokens: {src_nnz} / {src_row.numel()}")
    print(f"  src token ids (first 20): {src_row[:20].tolist()}")
    for t in PRED_TPS:
        tgt = batch[f"tgt_input_ids_t{t}"][0]
        nnz = int((tgt != 0).sum())
        print(f"  tgt_t{t} non-pad: {nnz}  first 20 ids: {tgt[:20].tolist()}")


if __name__ == "__main__":
    main()
