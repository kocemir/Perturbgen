# Gene-Query JEPA — the current (and only) JEPA design

Everything older (cell-level JEPA "Phase A", compare notebooks, sweep scripts)
has been deleted. This page is the single entry point.

## What the model does

Predict **target-time GENE embeddings** with gene queries, instead of only a
pooled cell vector. Full plain-language description at the top of
`perturbgen/Modules/gene_query_jepa.py`.

## Code

| Path | Role |
|------|------|
| `perturbgen/Modules/gene_query_jepa.py` | Model (encoders + population context + gene-query predictor) |
| `perturbgen/Model/gene_query_jepa_trainer.py` | Lightning trainer, query sampler, losses, honesty metrics |
| `perturbgen/tests/test_gene_query_jepa.py` | Unit tests (CPU, no checkpoint needed) |
| `docs/examples/toy_train_gene_query_jepa.py` | **Toy run** (~480 cells, overfit on purpose) |
| `docs/examples/run_gene_query_toy_sweep.sh` | **144-run toy sweep** (freeze × VICReg × contrastive × Q × L, 8 GPUs) |
| `docs/examples/summarize_gene_query_toy_sweep.py` | Ranked summary table of a sweep |
| `docs/examples/gene_query_jepa_architecture.pdf` | Architecture sketch |

## Results

```
/mnt/sod2-project/csb4/stuke1/perturbgen/gene_query_jepa/toy_runs/
```

One subfolder per run (auto-named `freeze_*_predL*_encL*_ep*_q*_*`), each with
`toy_logs/version_*/metrics.csv` and `checkpoints/`.

## Run the toy example

```bash
cd /home/stuke1/perturbgen/Perturbgen
source /home/stuke1/perturbgen/.venv/bin/activate
export PYTHONPATH=/home/stuke1/perturbgen/Perturbgen

# stable baseline (frozen encoder)
TOY_GPU=7 EPOCHS=30 FREEZE_ENCODER=true ENC_LAYERS=1 PREDICTOR_LAYERS=1 \
EARLY_STOP=false LAMBDA_CONTRASTIVE=0 VICREG_VAR=0 VICREG_COV=0 \
python -u docs/examples/toy_train_gene_query_jepa.py
```

Knobs (env): `FREEZE_ENCODER`, `ENC_LAYERS`, `PREDICTOR_LAYERS`, `N_QUERIES`,
`LAMBDA_CONTRASTIVE`, `CONTRASTIVE_TAU`, `VICREG_VAR`, `VICREG_COV`,
`EPOCHS`, `LR`, `BATCH_SIZE`, `TOY_GPU`, `SAVE_CKPT`, `TOY_OUT`.

## Run the 144-alternative toy sweep (all 8 GPUs)

```bash
cd /home/stuke1/perturbgen/Perturbgen
source /home/stuke1/perturbgen/.venv/bin/activate
export PYTHONPATH=/home/stuke1/perturbgen/Perturbgen

bash docs/examples/run_gene_query_toy_sweep.sh
```

Grid: freeze {T,F} × VICReg {off,1.0/0.04} × contrastive {off,0.3} ×
Q {64,128,256} × L {1..6}. One job per GPU, 8 in parallel.
Results land in `toy_runs/systematic_144/<run_id>/`; per run the best-epoch
checkpoint (by `val/gene_gap_vs_copy_src`) is kept for embedding extraction.

Resume: `SKIP_DONE=1 bash ...`  Subset: `Q_FILTER=64 L_FILTER=1,2 ...`
Summary: `python docs/examples/summarize_gene_query_toy_sweep.py`

## Unit tests

```bash
python -m pytest perturbgen/tests/test_gene_query_jepa.py -v
```

## How to read the result

Primary honesty metric: **`val/gene_gap_vs_copy_src`**
(cos(prediction, target) − cos(source, target), shared genes).

- Clearly positive and stable → the model learned dynamics, not copying.
- Pinned at ~0 while losses go to 0 → collapse; fix the recipe.
