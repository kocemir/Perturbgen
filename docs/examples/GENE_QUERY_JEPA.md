# Gene-Query JEPA (LPS)

The model, the figures, and what each notebook does: [../GENE_QUERY.md](../GENE_QUERY.md).

One script trains the JEPA, trains the pooled count head, dumps embeddings, and runs an IL1B knockout.

From `Perturbgen/` with the Poetry env and `PYTHONPATH` set:

```bash
python docs/examples/train_gene_query_jepa.py --help
```

## What is kept

See [KEEP.md](KEEP.md). In short:

- **Notebook 08** uses the atlas JEPA `…qsrcpos_20260926_232131-last.ckpt` and its `embeddings/`.
- **KO comparison** uses MaskGIT plus two 90m source-blind JEPAs (`…20261001_143041` and `…20261012_160033`) and their count heads.
- Checkpoints and `.h5ad` dumps live on disk, not in git.

## Analyses

| Notebook / script | Role |
|---|---|
| [08_GeneQuery_JEPA_Embedding_Analysis.ipynb](08_GeneQuery_JEPA_Embedding_Analysis.ipynb) | Cell UMAP and gene programs from the atlas checkpoint |
| [07_PostPerturbation_Analyses.ipynb](07_PostPerturbation_Analyses.ipynb) | MaskGIT-only DEG bar chart |
| [KO_comparison.ipynb](KO_comparison.ipynb) | MaskGIT vs both JEPA KOs (gene effects, panel f, panel i) |
| [panel_i_pathway_mirror.py](panel_i_pathway_mirror.py) | Pathway mirror and reversal vs Parse IL-1β |
| [jepa_full_res/plot_full_jepa_curves.py](jepa_full_res/plot_full_jepa_curves.py) | Curves for the four kept training runs |

Panel i reads `external/parse10m/degs_myeloid.parquet` (not in git; ~200 MB). How to obtain it: [external/parse10m/README.md](external/parse10m/README.md).

KO configs: `configs/ko/comparison.yaml` plus the three experiment YAMLs next to it. Figures from a notebook run land in `ko_comparison_res_maskgit_vs_sourceblind/` (gitignored).

## Count head

The count decoder averages the present gene rows of one 128-query quiz, then a ZINB `CountHead` emits all 2000 genes. Loss is ZINB NLL; MSE is logged only.

```bash
python docs/examples/train_gene_query_jepa.py --train-count-decoder true \
    --count-input source_only \
    --jepa-ckpt path/to/jepa-last.ckpt \
    --data full --split true --pred-tps 1,2 \
    --batch-size 32 --epochs 25 --lr 0.001
```

`--pred-tps` must be a single comma token (`1,2` or `1,2,3`). On the 90m corpus, `1,2` is 6h and 10h.

## Embedding dump

```bash
python docs/examples/train_gene_query_jepa.py \
    --eval-ckpt path/to/checkpoints/last.ckpt \
    --eval-split all --gpu 0
```

Writes L2 and raw cell/gene embeddings under that run’s `embeddings/`.

## Code

| File | Role |
|---|---|
| `docs/examples/train_gene_query_jepa.py` | Train, eval dump, count head, KO |
| `perturbgen/Modules/gene_query_jepa.py` | Model |
| `perturbgen/Model/gene_query_jepa_trainer.py` | Lightning trainer |
| `perturbgen/Model/gene_query_count_trainer.py` | Frozen JEPA + pooled CountHead |
| `perturbgen/tests/test_gene_query_jepa.py` | CPU smoke tests |
