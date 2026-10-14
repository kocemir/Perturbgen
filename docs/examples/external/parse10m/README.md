# Parse 10M PBMC cytokine screen (panel i)

Source: [Parse Biosciences — 10 Million Human PBMCs](https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment/) (PerturbGen paper ref. 67).

`degs_myeloid.parquet` is a streamed subset of the public `DEGs.csv`: myeloid cell types only (`CD14_Mono`, `CD16_Mono`, `cDC`, `pDC`, `Mono`), all cytokines. ~200 MB.

Panel i uses **IL-1-beta** and aggregates **`CD14_Mono` + `CD16_Mono` + `cDC`** (aligned with notebook 07 myeloid; `pDC` / generic `Mono` excluded). See `docs/examples/panel_i_pathway_mirror.py`.
