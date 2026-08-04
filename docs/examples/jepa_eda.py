from datasets import load_from_disk
from pathlib import Path

ROOT = Path("/home/stuke1/perturbgen/T_perturb/tokenized_data/LPS_all_tps_2k")
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