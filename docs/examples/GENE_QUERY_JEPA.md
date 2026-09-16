# Gene-Query JEPA

One training script covers **toy** and **full**.

**Honesty metric:** `gene_gap_vs_copy_src` must be **> 0**.
Toy logs it as `val/...` (train=val). Full with `--split false` logs `train/...`.
Do not trust a dropping `total_loss` alone.

## How to run

From `Perturbgen/` with the venv and `PYTHONPATH` set:

```bash
# toy: 2 mini-batches per cell type, batch size 16 → 32 cells / class
python docs/examples/train_gene_query_jepa.py --data toy \
    --batches-per-type 2 --batch-size 16

# full LPS, every cell, no hold-out
python docs/examples/train_gene_query_jepa.py --data full --split false

# hyperparameter search (always toy)
bash docs/examples/run_gene_query_toy_sweep.sh
python docs/examples/summarize_gene_query_toy_sweep.py
```

Embedding analysis (cell UMAP + gene programs, like notebooks 04/05):
[08_GeneQuery_JEPA_Embedding_Analysis.ipynb](08_GeneQuery_JEPA_Embedding_Analysis.ipynb).

Dump from a checkpoint (1 GPU):

```bash
python docs/examples/train_gene_query_jepa.py \
    --eval-ckpt path/to/checkpoints/epoch=02.ckpt \
    --eval-split all --gpu 7
```

`--eval-split all` dumps every LPS cell (default). `--eval-split test` is the
frozen 10% pickle only.

Full-run curves: [09_GeneQuery_JEPA_Full_Curves.ipynb](09_GeneQuery_JEPA_Full_Curves.ipynb).

`--data toy` uses `--batches-per-type × --batch-size` cells from every
`cell_type_harmonized` class, and trains/validates on those same cells.
`--data full` ignores `--batches-per-type`. `--split true` needs `--split-path`.

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
| `docs/examples/run_gene_query_toy_sweep.sh` | 144-run toy HPO |
| `docs/examples/summarize_gene_query_toy_sweep.py` | Rank sweep runs |
| `docs/examples/08_GeneQuery_JEPA_Embedding_Analysis.ipynb` | Cell/gene analysis of a JEPA ckpt |
| `docs/examples/09_GeneQuery_JEPA_Full_Curves.ipynb` | Full-run training curves |
| `perturbgen/Modules/gene_query_jepa.py` | Model |
| `perturbgen/Model/gene_query_jepa_trainer.py` | Lightning trainer |
| `perturbgen/Modules/jepa_scmaskgit.py` | Pretrained MaskGIT encoder |
| `perturbgen/Modules/jepa.py` | Tiny `CellEncoder` for CPU tests |
| `perturbgen/src/jepa_token_maps.py` | GLOBAL ↔ LOCAL ids |
| `perturbgen/src/jepa_metrics.py` | VICReg |
| `perturbgen/tests/test_gene_query_jepa.py` | CPU smoke tests |
