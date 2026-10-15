# Run Gene-Query JEPA

Everything below is a shell command. Run them from the `Perturbgen/` directory, after Poetry has created the environment:

```bash
cd Perturbgen
poetry env use python3.11
poetry install
source "$(poetry env info -p)/bin/activate"
```

What each stage is, and why the two datasets are separate, is in [GENE_QUERY.md](GENE_QUERY.md).

## 0. Paths for this shell

`LPS_DATA` is where tokenization writes. The training script does not follow it on its own: every training and dump command below passes `--tokenized`, `--split-path`, and `--output-dir`. Without those, the script looks for this server's `/mnt/sod2-project/...` tree.

```bash
export LPS_DATA="$HOME/lps"          # anywhere with a few hundred GB free
export WANDB_MODE=offline
export GPU=0                         # or 0,1,2,3
mkdir -p "$LPS_DATA"

export REPO="$PWD"
export ATLAS="$LPS_DATA/tokenized_data/LPS_all_tps_2k"
export SRC90="$LPS_DATA/tokenized_data/lps_90min_perturb"
export ENCODER="$REPO/pretraining_cohort/20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt"
```

`ENCODER` is the pretrained MaskGIT checkpoint. It is Git LFS; if that file is only a pointer, pull it or take the copy from [lotfollahi-lab/PerturbGen on Hugging Face](https://huggingface.co/lotfollahi-lab/PerturbGen/tree/main) and point `ENCODER` at the real file. Gene-name and token dictionaries used by tokenization ship in the repo under `perturbgen/pp/`.

`--pred-tps` is one comma-separated token. Write `1,2,3`, not `1, 2, 3`.

## 1. Tokenize

The raw atlas is not in git. Download it, then tokenize twice. The two runs read the same h5ad and write two folders, because the source time is different.

```bash
aws --endpoint-url https://cog.sanger.ac.uk --no-sign-request \
  s3 cp s3://perturbgen/Manuscript/lps_otar.h5ad \
  "$REPO/docs/examples/lps_otar.h5ad"
```

Resting source. `normal` is the source; targets are 90 min, 6 h, and 10 h. This is the data for section 3 and for the embedding dump.

```bash
python -m perturbgen tokenise \
  --h5ad_path "$REPO/docs/examples/lps_otar.h5ad" \
  --dataset LPS_all_tps_2k \
  --gene_filtering_mode hvg \
  --hvg_mode before_tokenisation \
  --var_list cell_type_harmonized time_after_LPS \
  --pairing_mode stratified \
  --time_obs time_after_LPS \
  --main_pairing_obs cell_type_harmonized \
  --nproc 8 \
  --n_hvg 2000 \
  --reference_time normal \
  --time_point_order normal 90m_LPS 6h_LPS 10h_LPS \
  --gene_median_path "$REPO/perturbgen/pp/gene_median_dict_gftokens_gc95M.pkl" \
  --token_dict_path "$REPO/perturbgen/pp/token_dict_gftokens_gc95M.pkl" \
  --gene_mapping_path "$REPO/perturbgen/pp/ensembl_mapping_dict_gc95M.pkl"
```

90 min source. `90m_LPS` is the source; targets are 6 h and 10 h. This is the data for section 5 and for the knockout. IL1B is essentially absent at `normal`, so a source knockout has to start here.

```bash
python -m perturbgen tokenise \
  --h5ad_path "$REPO/docs/examples/lps_otar.h5ad" \
  --dataset lps_90min_perturb \
  --gene_filtering_mode hvg \
  --hvg_mode before_tokenisation \
  --var_list cell_type_harmonized time_after_LPS \
  --pairing_mode stratified \
  --time_obs time_after_LPS \
  --main_pairing_obs cell_type_harmonized \
  --nproc 8 \
  --n_hvg 2000 \
  --reference_time 90m_LPS \
  --time_point_order 90m_LPS 6h_LPS 10h_LPS \
  --gene_median_path "$REPO/perturbgen/pp/gene_median_dict_gftokens_gc95M.pkl" \
  --token_dict_path "$REPO/perturbgen/pp/token_dict_gftokens_gc95M.pkl" \
  --gene_mapping_path "$REPO/perturbgen/pp/ensembl_mapping_dict_gc95M.pkl"
```

Each command writes under `$LPS_DATA/tokenized_data/<dataset>/`:

| Path | What it is |
|---|---|
| `dataset_2000_hvg_src/` | Hugging Face dataset of the source cells |
| `dataset_2000_hvg_tgt/` | One dataset per target time. The leading number is the `--pred-tps` index |
| `h5ad_pairing_2000_hvg_src/` and `..._tgt/` | Same cells, with counts |
| `token_id_to_genename_2000_hvg.pkl` | Local token id → Ensembl |
| `tokenid_to_rowid_2000_hvg.pkl` | Token id → row in the 2,000-gene panel |

On the 90 min corpus, `1_6h_LPS` and `2_10h_LPS` mean `--pred-tps 1,2` is 6 h and 10 h. On the atlas, `1,2,3` is 90 min, 6 h, and 10 h. The index is the slot this tokenization created, not a universal clock.

## 2. Freeze the split

Tokenization does not write the split. The training script does not compute one either: `--split true` only reads a pickle of row numbers. The block below is the same recipe as notebook 02. It writes the pickle next to the data it just tokenized. Another machine's pickle will not match these rows.

Resting atlas, one row at a time, stratified on `cell_type_harmonized`, seed 42, 90% train / 10% val:

```bash
TOK="$ATLAS" GROUP_BY_SOURCE=0 python - <<'PY'
import datetime, os, pickle
from pathlib import Path
import anndata as ad
import numpy as np
from perturbgen.src.utils import stratified_split

root = Path(os.environ["TOK"])
group = os.environ.get("GROUP_BY_SOURCE", "0") == "1"
tgt = sorted((root / "h5ad_pairing_2000_hvg_tgt").glob("*.h5ad"))[0]
adata = ad.read_h5ad(tgt)
obs = ["cell_type_harmonized"]
if group:
    dedup = adata.obs.drop_duplicates("cell_pairing_index").reset_index(drop=True)
    dedup_ad = ad.AnnData(X=np.zeros((len(dedup), 1), np.float32), obs=dedup)
    tr, va, te = stratified_split(dedup_ad, 0.9, 0.0, obs, seed=42)
    pairs = dedup["cell_pairing_index"].to_numpy()
    all_pairs = adata.obs["cell_pairing_index"].to_numpy()
    train = np.where(np.isin(all_pairs, pairs[list(tr)]))[0].astype(np.int64)
    val = np.where(np.isin(all_pairs, pairs[list(va)]))[0].astype(np.int64)
    test = np.where(np.isin(all_pairs, pairs[list(te)]))[0].astype(np.int64)
    stem = "stratified_cell_type_harmonized_srcgroup_seed42_90_10"
else:
    tr, va, te = stratified_split(adata, 0.9, 0.0, obs, seed=42)
    train, val, test = (np.asarray(x, np.int64) for x in (tr, va, te))
    stem = "stratified_cell_type_harmonized_seed42_90_10"
out = root / "splits"
out.mkdir(parents=True, exist_ok=True)
path = out / f"{stem}.pkl"
with path.open("wb") as handle:
    pickle.dump({
        "train_indices": train, "val_indices": val, "test_indices": test,
        "seed": 42, "train_prop": 0.9, "test_prop": 0.0,
        "group_by_source": group, "n_rows": int(adata.n_obs),
        "reference_h5ad": str(tgt),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(path)
print("train", len(train), "val", len(val), "test", len(test))
PY
```

90 min corpus. The same source cell is repeated across many target rows, so the split is grouped by `cell_pairing_index`: a source cell is entirely in train or entirely in val.

```bash
TOK="$SRC90" GROUP_BY_SOURCE=1 python - <<'PY'
import datetime, os, pickle
from pathlib import Path
import anndata as ad
import numpy as np
from perturbgen.src.utils import stratified_split

root = Path(os.environ["TOK"])
group = os.environ.get("GROUP_BY_SOURCE", "0") == "1"
tgt = sorted((root / "h5ad_pairing_2000_hvg_tgt").glob("*.h5ad"))[0]
adata = ad.read_h5ad(tgt)
obs = ["cell_type_harmonized"]
if group:
    dedup = adata.obs.drop_duplicates("cell_pairing_index").reset_index(drop=True)
    dedup_ad = ad.AnnData(X=np.zeros((len(dedup), 1), np.float32), obs=dedup)
    tr, va, te = stratified_split(dedup_ad, 0.9, 0.0, obs, seed=42)
    pairs = dedup["cell_pairing_index"].to_numpy()
    all_pairs = adata.obs["cell_pairing_index"].to_numpy()
    train = np.where(np.isin(all_pairs, pairs[list(tr)]))[0].astype(np.int64)
    val = np.where(np.isin(all_pairs, pairs[list(va)]))[0].astype(np.int64)
    test = np.where(np.isin(all_pairs, pairs[list(te)]))[0].astype(np.int64)
    stem = "stratified_cell_type_harmonized_srcgroup_seed42_90_10"
else:
    tr, va, te = stratified_split(adata, 0.9, 0.0, obs, seed=42)
    train, val, test = (np.asarray(x, np.int64) for x in (tr, va, te))
    stem = "stratified_cell_type_harmonized_seed42_90_10"
out = root / "splits"
out.mkdir(parents=True, exist_ok=True)
path = out / f"{stem}.pkl"
with path.open("wb") as handle:
    pickle.dump({
        "train_indices": train, "val_indices": val, "test_indices": test,
        "seed": 42, "train_prop": 0.9, "test_prop": 0.0,
        "group_by_source": group, "n_rows": int(adata.n_obs),
        "reference_h5ad": str(tgt),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(path)
print("train", len(train), "val", len(val), "test", len(test))
PY
```

Leaving `--split-path` off does not build this file. The flag then points at the atlas pickle baked into the script, and a 90 min run stops before training.

## 3. Train the resting JEPA

Source `normal`, targets 90 min / 6 h / 10 h. These flags are the notebook 08 checkpoint (`20260926_232131`): encoder frozen for the whole run, gene MSE, contrastive weight 0.1, mask-CE on 20% of the source-side questions, source-rank on the query, 100 epochs, the frozen 90/10 split. The quiz is the legacy `target` sampler (`--query-sampling target`). The current script default is `source_blind`; omitting the flag trains a different model.

```bash
mkdir -p "$LPS_DATA/res/jepa_atlas"
python docs/examples/train_gene_query_jepa.py \
  --data full \
  --tokenized "$ATLAS" \
  --pred-tps 1,2,3 \
  --split true \
  --split-path "$ATLAS/splits/stratified_cell_type_harmonized_seed42_90_10.pkl" \
  --encoder-path "$ENCODER" \
  --freeze-encoder true \
  --encoder-layers 3 \
  --predictor-layers 3 \
  --n-queries 128 \
  --query-sampling target \
  --query-mode mixed \
  --gene-loss mse \
  --lambda-gene 1 \
  --lambda-cell 0 \
  --lambda-contrastive 0.1 \
  --lambda-mask-ce 0.1 \
  --mask-frac-of-shared 0.2 \
  --lambda-absent 0.25 \
  --query-src-pos true \
  --cell-pool cls \
  --max-len 512 \
  --batch-size 16 \
  --lr 0.0001 \
  --epochs 100 \
  --seed 0 \
  --early-stop false \
  --gpu "$GPU" \
  --output-dir "$LPS_DATA/res/jepa_atlas"
```

The checkpoint lands at `$LPS_DATA/res/jepa_atlas/checkpoints/*-last.ckpt`. Training logs `gene_gap_vs_copy_src`. A run that is learning the transition, rather than copying the source embedding, has that gap above 0.

## 4. Embedding extraction

Reads the resting checkpoint and writes cell and gene embeddings for every paired cell. One GPU. Mask-CE is off for the dump. If `embeddings/jepa_cell_embeddings.h5ad` is already there, the command exits; `--eval-force true` writes it again.

```bash
python docs/examples/train_gene_query_jepa.py \
  --eval-ckpt "$LPS_DATA/res/jepa_atlas/checkpoints/"*-last.ckpt \
  --tokenized "$ATLAS" \
  --pred-tps 1,2,3 \
  --split true \
  --split-path "$ATLAS/splits/stratified_cell_type_harmonized_seed42_90_10.pkl" \
  --eval-split all \
  --gpu "${GPU%%,*}" \
  --batch-size 16
```

Files under `$LPS_DATA/res/jepa_atlas/embeddings/`:

| File | Contents |
|---|---|
| `jepa_cell_embeddings.h5ad` | One row per cell-time. `X` is L2-normalized; `obsm` also has the raw vectors |
| `jepa_cell_embeddings_raw.h5ad` | Raw cell vectors only |
| `jepa_gene_embeddings.h5ad` | Gene vectors in `varm`, L2 and raw |
| `jepa_gene_embeddings_raw.h5ad` | Raw gene vectors only |

Notebook 08 reads this directory. `--gene-name-id-dict` is optional; a missing file leaves gene symbols blank and the dump still finishes.

## 5. Train 90 min → 6 h / 10 h

This is the JEPA behind the IL1B knockout. Source is `90m_LPS`. `--pred-tps 1,2` is 6 h and 10 h on this corpus.

The quiz is source-blind: 64 genes drawn from the source list, 48 genes absent from the source and chosen as the most common at that cell type and time, and 16 weighted-random from the rest. Contrastive weight is 0. Mask-CE is on. The encoder stays frozen. The split is the source-grouped pickle from section 2, which is what `splitT` on this corpus means. Passing the atlas pickle here makes the script exit.

```bash
mkdir -p "$LPS_DATA/res/jepa_90m"
python docs/examples/train_gene_query_jepa.py \
  --data full \
  --tokenized "$SRC90" \
  --pred-tps 1,2 \
  --split true \
  --split-path "$SRC90/splits/stratified_cell_type_harmonized_srcgroup_seed42_90_10.pkl" \
  --encoder-path "$ENCODER" \
  --freeze-encoder true \
  --encoder-layers 3 \
  --predictor-layers 3 \
  --n-queries 128 \
  --query-sampling source_blind \
  --n-src-queries 64 \
  --n-top-common-queries 48 \
  --nonsrc-weighting celltype_time \
  --gene-loss mse \
  --lambda-gene 1 \
  --lambda-cell 0 \
  --lambda-contrastive 0 \
  --lambda-mask-ce 0.1 \
  --mask-frac-of-shared 0.2 \
  --lambda-absent 0.25 \
  --query-src-pos true \
  --cell-pool cls \
  --max-len 512 \
  --batch-size 32 \
  --lr 0.0001 \
  --epochs 30 \
  --seed 0 \
  --early-stop false \
  --gpu "$GPU" \
  --output-dir "$LPS_DATA/res/jepa_90m"
```

Checkpoint: `$LPS_DATA/res/jepa_90m/checkpoints/*-last.ckpt`.

An embedding dump of this model is the section 4 command with three changes: `--eval-ckpt` points at this checkpoint, `--tokenized "$SRC90"`, and `--pred-tps 1,2` together with the source-grouped `--split-path`. The knockout in section 6 does not use that embedding file. It runs the count head.

## 6. Count head, then the IL1B knockout dump

The JEPA predicts embeddings. Counts come from a second training step. The JEPA weights stay frozen. One 128-question quiz is drawn by the checkpoint's own sampler, present rows are averaged, and a ZINB head emits all 2,000 genes. Mask-CE is turned off for this step, so the quiz is unmasked. Loss is ZINB negative log-likelihood. The run directory's name may say `qsb96` because that string is built from the count-training defaults; the loaded checkpoint is still the qsb64 model from section 5.

```bash
mkdir -p "$LPS_DATA/res/count_90m"
python docs/examples/train_gene_query_jepa.py \
  --train-count-decoder true \
  --count-input source_only \
  --jepa-ckpt "$LPS_DATA/res/jepa_90m/checkpoints/"*-last.ckpt \
  --data full \
  --tokenized "$SRC90" \
  --pred-tps 1,2 \
  --split true \
  --split-path "$SRC90/splits/stratified_cell_type_harmonized_srcgroup_seed42_90_10.pkl" \
  --encoder-path "$ENCODER" \
  --batch-size 32 \
  --lr 0.001 \
  --epochs 25 \
  --seed 0 \
  --early-stop false \
  --gpu "$GPU" \
  --output-dir "$LPS_DATA/res/count_90m"
```

The knockout dump replaces IL1B (`ENSG00000125538`) with `<mask>` in the 90 min source list, resynchronizes the quiz so that gene's source rank becomes −1, and writes paired control and knockout counts for every cell. `--split false` is required on this command: the dump itself uses every row, and the default split path is the atlas pickle, which this corpus rejects.

```bash
mkdir -p "$LPS_DATA/res/ko_il1b"
python docs/examples/train_gene_query_jepa.py \
  --eval-count-ko true \
  --tokenized "$SRC90" \
  --jepa-ckpt "$LPS_DATA/res/jepa_90m/checkpoints/"*-last.ckpt \
  --count-ckpt "$LPS_DATA/res/count_90m/checkpoints/"*-last.ckpt \
  --ko-gene ENSG00000125538 \
  --ko-src-name 90m_LPS.dataset \
  --ko-pred-tps 1 2 \
  --ko-replace mask \
  --ko-resync-quiz true \
  --split false \
  --gpu "${GPU%%,*}" \
  --batch-size 64 \
  --output-dir "$LPS_DATA/res/ko_il1b"
```

The h5ad in that directory has:

| Slot | Contents |
|---|---|
| `X` | Knockout predicted counts |
| `layers['pred_counts']` | Control predicted counts, same cells |
| `layers['true_counts']` | Measured target counts |
| `layers['pert_counts']` | Same matrix as `X` |

`KO_comparison.ipynb` does not run this command. `RERUN_INFERENCE` stays `False`, and the notebook opens whatever path is written in the YAML. Section 6 produces the h5ad. The YAML is only the pointer the notebook and `panel_i_pathway_mirror.py` follow.

The files in the repo still point at this server's sod2 tree. On another machine, after the dump finishes, edit them to the paths you just wrote. The same absolute path is copied in more than one place; changing only `cli.args` leaves the notebook on the old file.

`docs/examples/configs/ko/comparison.yaml`

| Field | Set it to |
|---|---|
| `maps.token_id_to_ensembl` | `$SRC90/token_id_to_genename_2000_hvg.pkl` |
| `maps.symbol_to_ensembl` | a Geneformer `gene_name_id_dict.pkl`, if you have one. Missing symbols do not block the dump |
| `output_dir` | where the comparison figures should be written |

`docs/examples/configs/ko/jepa_count_il1b_src_90m_qsb64_tc48_count25_mask.yaml`

| Field | Set it to |
|---|---|
| `data.tokenized`, `data.src_dataset_file`, `data.tgt_dataset_folder`, `data.src_adata`, `data.tgt_adata_folder` | the `$SRC90` tree from section 1 |
| `trainer.output_dir` | `$LPS_DATA/res/ko_il1b` |
| `trainer.mapping_dict_path`, `trainer.tokenid_to_rowid_path` | the two pickles in `$SRC90` |
| `model.jepa_ckpt` | the section 5 `*-last.ckpt` |
| `model.count_ckpt` | the count-head `*-last.ckpt` |
| `cli.args` | the same command as section 6, with those paths |

That YAML has no `preferred_dump`. The notebook takes the newest file under `trainer.output_dir` matching `dump_glob` (`*_ssrc_*qresync.h5ad`). A mask + quiz-resync dump is named `..._ssrc_kmask_qresync.h5ad`, so the glob matches. If you set `preferred_dump`, it must be that filename or the notebook ignores the new file and keeps the pinned one.

The older source-blind column is `jepa_count_il1b_src_90m_sourceblind_srconly_mask.yaml`. Same fields, its own checkpoints.

MaskGIT does not use `train_gene_query_jepa.py`. Its YAML is `configs/ko/maskgit_il1b_src_90m.yaml`, and the run command in that file's header is `perturbgen/Perturb/val.py --config` that YAML. `experiment.preferred_dump` is set there, so a new MaskGIT h5ad stays unused until that filename is updated. Notebook 07 reads the MaskGIT directory only.
