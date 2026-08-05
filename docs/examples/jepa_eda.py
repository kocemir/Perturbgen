#!/usr/bin/env python3
"""JEPA data / PE EDA (2026-08-05).

Sanity-check LPS src↔tgt pairing and visualize TimePosSin encodings.
Output figure: docs/examples/jepa_eda.png (same dir as this script).
See docs/examples/JEPA_README.md.
"""
from datasets import load_from_disk
from pathlib import Path

ROOT = Path("/home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k")
OUT_PNG = Path(__file__).resolve().parent / "jepa_eda.png"
src = load_from_disk(ROOT / "dataset_2000_hvg_src/normal.dataset")
t1  = load_from_disk(ROOT / "dataset_2000_hvg_tgt/1_90m_LPS.dataset")
t2  = load_from_disk(ROOT / "dataset_2000_hvg_tgt/2_6h_LPS.dataset")
t3  = load_from_disk(ROOT / "dataset_2000_hvg_tgt/3_10h_LPS.dataset")

print(len(src), len(t1), src.column_names)
row = src[0]
print(type(row["input_ids"]), len(row["input_ids"]), min(row["input_ids"]), max(row["input_ids"]))
print(t1[0].keys())
print(t1[0]["cell_pairing_index"], src[0]["cell_pairing_index"])  # should match for same i

i = 0
print("types", src[i]["cell_type_harmonized"], t1[i]["cell_type_harmonized"])
print("times", src[i]["time_after_LPS"], t1[i]["time_after_LPS"])
print("pairing ids", src[i]["cell_pairing_index"], t1[i]["cell_pairing_index"])

i = 0  # change to 1, 100, ...

sample = {
    "src": src[i],
    "t1": t1[i],
    "t2": t2[i],
    "t3": t3[i],
}

print("=== Step B: getitem-style sample at i =", i, "===")
print(
    "types:",
    sample["src"]["cell_type_harmonized"],
    sample["t1"]["cell_type_harmonized"],
    sample["t2"]["cell_type_harmonized"],
    sample["t3"]["cell_type_harmonized"],
)
print(
    "times:",
    sample["src"]["time_after_LPS"],
    sample["t1"]["time_after_LPS"],
    sample["t2"]["time_after_LPS"],
    sample["t3"]["time_after_LPS"],
)

for name, row in sample.items():
    ids = row["input_ids"]
    print(
        f"{name}: L={len(ids)}  min={min(ids)} max={max(ids)}  "
        f"pairing={row['cell_pairing_index']}"
    )




import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from perturbgen.pp import TOKEN_DICTIONARY_FILE

with open(TOKEN_DICTIONARY_FILE, "rb") as f:
    gene_token_dict = pickle.load(f)

PAD = gene_token_dict["<pad>"]   # usually 0
CLS = gene_token_dict["<cls>"]
EOS = gene_token_dict["<eos>"]
print("PAD/CLS/EOS =", PAD, CLS, EOS)


def ids_to_tensor(ids, strip_special=False):
    t = torch.tensor(ids, dtype=torch.long)
    if strip_special:
        t = t[(t != CLS) & (t != EOS)]
    return t


def make_batch(indices, ds, strip_special=False):
    seqs = [ids_to_tensor(ds[i]["input_ids"], strip_special=strip_special) for i in indices]
    batch = pad_sequence(seqs, batch_first=True, padding_value=PAD)
    lengths = torch.tensor([len(s) for s in seqs])
    return batch, lengths


B_IDX = list(range(8))  # batch of 8 cells

src_batch, src_len = make_batch(B_IDX, src, strip_special=False)
t1_batch, t1_len = make_batch(B_IDX, t1, strip_special=True)
t2_batch, t2_len = make_batch(B_IDX, t2, strip_special=True)
t3_batch, t3_len = make_batch(B_IDX, t3, strip_special=True)

print("=== Step C: padded batch ===")
for name, batch, lengths in [
    ("src", src_batch, src_len),
    ("t1", t1_batch, t1_len),
    ("t2", t2_batch, t2_len),
    ("t3", t3_batch, t3_len),
]:
    print(
        f"{name}: shape={tuple(batch.shape)} dtype={batch.dtype} "
        f"max_id={int(batch.max())} lengths={lengths.tolist()}"
    )



import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def sinusoidal(length, d_model):
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (length, d_model)

D = 768
N_TIME = 5          # JEPA: n_time_steps+1 with n_time_steps=4
L_SHOW = 64         # first 64 positions (full max_seq is huge)
D_SHOW = 64         # first 64 dims (768 is hard to read)

time_pe = sinusoidal(N_TIME, D)
pos_pe = sinusoidal(2048, D)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1) time_pe heatmap
im0 = axes[0, 0].imshow(time_pe[:, :D_SHOW].numpy(), aspect="auto", cmap="coolwarm")
axes[0, 0].set_title(f"time_pe  shape=({N_TIME},{D})  showing [:, :{D_SHOW}]")
axes[0, 0].set_ylabel("time index k")
axes[0, 0].set_xlabel("dim")
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

# 2) pos_pe heatmap
im1 = axes[0, 1].imshow(pos_pe[:L_SHOW, :D_SHOW].numpy(), aspect="auto", cmap="coolwarm")
axes[0, 1].set_title(f"pos_pe  showing [:{L_SHOW}, :{D_SHOW}]")
axes[0, 1].set_ylabel("position i")
axes[0, 1].set_xlabel("dim")
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

# 3) time cosine similarity
tn = F.normalize(time_pe, dim=1)
im2 = axes[1, 0].imshow(torch.mm(tn, tn.t()).numpy(), vmin=0, vmax=1, cmap="viridis")
axes[1, 0].set_title("cosine between time rows")
axes[1, 0].set_xlabel("time j"); axes[1, 0].set_ylabel("time i")
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

# 4) one dim over positions (classic PE wave)
axes[1, 1].plot(pos_pe[:L_SHOW, 0].numpy(), label="dim 0 (sin)")
axes[1, 1].plot(pos_pe[:L_SHOW, 1].numpy(), label="dim 1 (cos)")
axes[1, 1].plot(pos_pe[:L_SHOW, 20].numpy(), label="dim 20")
axes[1, 1].set_title("pos_pe waves along sequence")
axes[1, 1].set_xlabel("position i"); axes[1, 1].legend()

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print("Wrote", OUT_PNG)
plt.show()