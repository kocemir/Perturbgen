# Gene-Query JEPA

One training script covers **toy** and **full**.

**Honesty metric:** `gene_gap_vs_copy_src` must be **> 0**.
Toy with `--val-batches-per-type > 0` logs a held-out `val/...`.
`--val-batches-per-type 0` reuses train cells as val (legacy).

## How to run

From `Perturbgen/` with the venv and `PYTHONPATH` set:

```bash
# toy: 4 train + 1 val mini-batches per cell type, batch size 16
python docs/examples/train_gene_query_jepa.py --data toy \
    --batches-per-type 4 --val-batches-per-type 1 --batch-size 16

# full LPS, frozen 90/10 train/val, last 2 GPUs (DDP)
bash docs/examples/run_gene_query_full.sh
# equivalent:
python docs/examples/train_gene_query_jepa.py --data full --split true --gpu 4,5,6,7 \
    --encoder-layers 3 --predictor-layers 3 --batch-size 16 \
    --freeze-encoder true --encoder-lr-schedule freeze:5,5e-7:5,1e-6:5,freeze:5 \
    --lambda-gene 1.0 --lambda-cell 0 --vicreg-var 0 --vicreg-cov 0 \
    --epochs 20 --early-stop false

# 90m_LPS source -> 6h/10h (IL1B pairing). No matched split pickle: --split false
python docs/examples/train_gene_query_jepa.py --data full --split false \
    --tokenized /mnt/sod2-project/csb4/stuke1/perturbgen_reproduction/lps/tokenized_data/lps_90min_perturb \
    --pred-tps 1,2 --gpu 0,1,2,3,4,5,6,7 --batch-size 16 \
    --encoder-layers 3 --predictor-layers 3 \
    --freeze-encoder true --encoder-lr-schedule freeze:5,5e-7:5,1e-6:5,freeze:5 \
    --lambda-gene 1.0 --lambda-cell 0 --vicreg-var 0 --vicreg-cov 0 \
    --epochs 20 --early-stop false

# one toy sweep (default 4p1, Q=128, enc L 1-3 × pred L 1-3; logs gene_mse)
bash docs/examples/run_gene_query_toy_sweep.sh
python docs/examples/summarize_gene_query_toy_sweep.py
python docs/examples/plot_gene_query_toy_sweep.py
```

```bash
# count decoder, source-only (honest: mean of present z_hat from the predictor)
python docs/examples/train_gene_query_jepa.py --train-count-decoder true \
    --count-input source_only \
    --jepa-ckpt path/to/jepa-last.ckpt \
    --data full --split true --gpu 0,1,2,3,4,5,6,7 --batch-size 64 --epochs 40 --lr 1e-3 \
    --early-stop false

# count decoder, copy-target (z_hat mean; queries = first n_queries target tokens, skip pad)
python docs/examples/train_gene_query_jepa.py --train-count-decoder true \
    --count-input copy_target \
    --jepa-ckpt path/to/jepa-last.ckpt \
    --data full --split true --gpu 0,1,2,3,4,5,6,7 --batch-size 64 --epochs 40 --lr 1e-3 \
    --early-stop false
```

Embedding analysis (cell UMAP + gene programs, like notebooks 04/05):
[08_GeneQuery_JEPA_Embedding_Analysis.ipynb](08_GeneQuery_JEPA_Embedding_Analysis.ipynb).

Side-by-side JEPA (notebook 08 dump; currently 100ep MSE `20260926_232131`) vs MaskGIT programs:
[10_GeneProgram_Comparison.ipynb](10_GeneProgram_Comparison.ipynb).

Dump from a checkpoint (1 GPU). Writes **L2 and raw** cell/gene embeddings in one pass
(mask-CE off; same quiz as count-decoder inference):

```bash
python docs/examples/train_gene_query_jepa.py \
    --eval-ckpt path/to/checkpoints/last.ckpt \
    --eval-split all --gpu 0
```

`--eval-split all` dumps every LPS cell (default). `--eval-split test` is the
frozen 10% pickle only. Outputs under the ckpt run's `embeddings/`:
`jepa_cell_embeddings.h5ad` (X = L2; `obsm` also has `*_raw`),
`jepa_cell_embeddings_raw.h5ad`, `jepa_gene_embeddings.h5ad` (`varm` L2 + `*_raw`),
`jepa_gene_embeddings_raw.h5ad`.

Full-run curves: [09_GeneQuery_JEPA_Full_Curves.ipynb](09_GeneQuery_JEPA_Full_Curves.ipynb).

`--data toy` takes `--batches-per-type` train batches and `--val-batches-per-type`
held-out val batches per `cell_type_harmonized` class (disjoint if val > 0).
`--data full` uses the frozen 90/10 pickle (`--split true`) so Lightning logs held-out `val/...`. `--split false` trains on every cell (no val).

JEPA keeps `<cls>` / `<eos>` on target sequences (`--strip-tgt-special-tokens false`).
Cell **target/src**: encoder `--cell-pool cls` (default; `--cell-pool mean` is ablation).
Objective is gene cosine only (`--lambda-cell 0`; VICReg off because it was on `z_hat_cell`).
Cell metrics are still logged. Encoder schedule `freeze:5,5e-7:5,1e-6:5,freeze:5`.

Full runs write everything under
`.../T_perturb/res/jepa_gene_query_full_atlas/<spec-name>_{timestamp}/`
(`specs.json`, `checkpoints/`, `logs/`).

```bash
python docs/examples/train_gene_query_jepa.py --help
pytest perturbgen/tests/test_gene_query_jepa.py -v
```

## Files

| File | Role |
|------|------|
| `docs/examples/train_gene_query_jepa.py` | Unified train (toy or full) |
| `docs/examples/run_gene_query_full.sh` | Full LPS + val split, last 2 GPUs (DDP) |
| `docs/examples/run_gene_query_toy_sweep.sh` | Toy grid (Q × enc L × pred L; default 4p1) |
| `docs/examples/plot_gene_query_toy_sweep.py` | Curves including present-query gene MSE |
| `docs/examples/summarize_gene_query_toy_sweep.py` | Rank sweep runs |
| `docs/examples/08_GeneQuery_JEPA_Embedding_Analysis.ipynb` | Cell/gene analysis of a JEPA ckpt |
| `docs/examples/10_GeneProgram_Comparison.ipynb` | Side-by-side JEPA vs MaskGIT gene programs (atlas) |
| `docs/examples/09_GeneQuery_JEPA_Full_Curves.ipynb` | Full-run training curves |
| `perturbgen/Modules/gene_query_jepa.py` | Model |
| `perturbgen/Model/gene_query_jepa_trainer.py` | Lightning trainer |
| `perturbgen/Model/gene_query_count_trainer.py` | Frozen JEPA + ZINB CountHead (present gene mean) |
| `perturbgen/Modules/jepa_scmaskgit.py` | Pretrained MaskGIT encoder |
| `perturbgen/Modules/jepa.py` | Tiny `CellEncoder` for CPU tests |
| `perturbgen/src/jepa_token_maps.py` | GLOBAL ↔ LOCAL ids |
| `perturbgen/src/jepa_metrics.py` | VICReg |
| `perturbgen/tests/test_gene_query_jepa.py` | CPU smoke tests |
