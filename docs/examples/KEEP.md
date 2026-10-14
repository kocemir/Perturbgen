# KEEP list (cleanup decision — 2026-10-14)

**Status: cleanup applied 2026-10-14.** Log: `cleanup_deleted_20261014.log`.

Rule: if it is not named below, it was treated as delete-candidate.

## Notebook 08 (embedding / gene programs)

Uses **exactly one** JEPA run:

```
/mnt/sod2-project/csb4/stuke1/perturbgen/T_perturb/res/jepa_gene_query_full_atlas/
  fzT_encL3_predL3_q128_mixed_lg1_lc0_contr0.1_vic0_glmse_bs16_ep100_
  lr0.0001_seed0_splitT_poolcls_tp1-2-3_max512_mce0.1_ms0.2_qsrcpos_20260926_232131/
```

Keep inside that folder:
- `checkpoints/…-last.ckpt` (required)
- `embeddings/` (cell/gene h5ad + figures; skip re-dump)
- `logs/version_0/metrics.csv` (optional but tiny)

Delete other checkpoints in that folder (`epoch=*.ckpt`) if you only need `last`.
Delete **all other** atlas run directories (the rest of ~166G).

Also needed for notebook 08 (shared data, not JEPA):
- `/mnt/sod2-project/csb4/stuke1/perturbgen_reproduction/lps/tokenized_data/LPS_all_tps_2k/`
- Geneformer `gene_name_id_dict.pkl`

## KO story (MaskGIT + JEPA) — keep these only

### JEPA encoders
| Role | Path stamp |
|---|---|
| Best 90m KO line | `…jepa_gene_query_90m_src/…qsb64_nwct_tc48…20261012_160033` (`last.ckpt`) |
| Older 90m KO line | `…qsb96…20261001_143041` (`last.ckpt`) |
| Normal→rest (report curves) | `…full_atlas/…qsb96…20261008_101504` (`last.ckpt` + enough epochs if you want resume) |

Delete every other folder under `jepa_gene_query_90m_src/` and `jepa_gene_query_full_atlas/`.

### Count heads
| Role | Path stamp |
|---|---|
| Current KO count | `…countdec_…20261013_172837` (`last.ckpt`) |
| Older source_blind KO count | `…20261001_202210` (`last.ckpt`) |

Delete other `count_decoder_runs/countdec_*` and abandoned `…20261013_104857`.

### KO dumps (h5ad)
| Role | Dir |
|---|---|
| MaskGIT | `…/il1b_src_90m_retrain/` (preferred h5ad) |
| JEPA qsb64 | `…/il1b_src_90m_qsb64_tc48_count25_mask_qresync/` |
| JEPA old SB | `…/il1b_src_90m_sourceblind_srconly100_mask_qresync/` |

Delete `il1b_src_90m_jepa100_*`, `paperko`, unused JEPA `il1b_src_90m_retrain`, etc.

### Docs / code under `docs/examples/`
**Keep**
- `08_GeneQuery_JEPA_Embedding_Analysis.ipynb`
- `KO_comparison.ipynb`
- `07_PostPerturbation_Analyses.ipynb` (MaskGIT-only post)
- `panel_i_pathway_mirror.py`
- `configs/ko/{comparison.yaml, maskgit_*.yaml, jepa_*sourceblind*.yaml, jepa_*qsb64*.yaml}`
- `external/parse10m/degs_myeloid.parquet` (+ README)
- `ko_comparison_res_maskgit_vs_sourceblind/` (panel e/f/i figures)
- `train_gene_query_jepa.py` + library code it imports
- `jepa_full_res/` **trimmed** (see below)

**Delete**
- Other `ko_comparison_res_*`
- Archived KO yamls (`*pad*`, `*copytgt*`, generic `jepa_count_il1b_src_90m.yaml` if unused)
- `09_*`, `collapse_check*`, toy sweep scripts, `launch_logs`, `_recovered_bytecode`, `__pycache__`
- Optional: `lps_otar.h5ad` only if Enrichr maps are covered elsewhere

## `jepa_full_res/` — decision

This folder is **plots only** (~5MB). It does not hold checkpoints.

### Keep these plot subdirs + script
- `plot_full_jepa_curves.py`
- `source_blind_90m_qsb64_tc48_contr0_ep30_20261012/`  ← current KO JEPA
- `source_blind_90m_splitT_20260930/`                 ← older KO JEPA (143041)
- `source_blind_normal_bs32_ep100_20261008/`           ← normal→rest curves
- `frozen_mse_contr_maskce_qsrcpos_100ep_bs16_20260926/` ← **notebook 08’s run** (same stamp as 20260926_232131)

### Delete from `jepa_full_res/`
- `frozen_mse_contr_maskce_qsrcpos_100ep_90m_src_20260928/` (splitF train curves; not in KO KEEP)
- Any leftover older names if regenerated (`gene_only_*`, cosine-only, 50ep, etc.)
- Trim `plot_full_jepa_curves.py` `RUNS` list to the four kept names above

## Not needed for “08 + KO report”
- `toy_runs/`, most of `full_runs/`
- Intermediate epoch ckpts once `last.ckpt` exists (except if you want resume)
- Duplicate effect CSVs / old comparison figure trees
