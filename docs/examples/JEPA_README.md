# JEPA Phase A — examples & artifacts

Index of training / analysis helpers under `docs/examples/` (and related docs).
Dates are **first useful version** in this LPS JEPA effort (YYYYMMDD).

## Kept training runs (sod2)

| Alt | Run ID | Encoder | Freeze | Notes |
|-----|--------|---------|--------|-------|
| A | `20260804_1120` | scmaskgit 3L | unfz | Best learning so far; prefer for Phase B |
| B | `20260805_1128` | scmaskgit 3L | fz | Low pred loss but **collapsed** |
| C | `20260805_1622` | cell 3L + token warm-start | unfz | In progress / fresh encoder ablation |

Paths:
- Metrics / TB: `/mnt/sod2-project/csb4/stuke1/perturbgen/logs/<run>_cellgen/`
- Tee stdout: `logs/jepa_phaseA_*.log` (one per kept run)
- Checkpoints: `/mnt/sod2-project/csb4/stuke1/perturbgen/T_perturb/res/jepa/checkpoints/`

**Not kept (deleted):** aborted fz launches (`1110`/`1112`/`1113`), DDP empty sidecars (`1129`/`1623`), older fz ckpt folder `1708`, offline `wandb/`.

## Scripts & notebooks

| File | ~Date | Why it exists |
|------|-------|----------------|
| `run_train_jepa_sod2.sh` | 20260801+ | **Main launch** for Phase A on sod2 (DDP, cosine+VICReg). Env: `JEPA_ENCODER`, `FREEZE_ENCODER`, `WARMSTART_TOKEN_EMB`, … |
| `08_train_jepa.ipynb` | 20260801 | Interactive / documented train entry (paths, notes) |
| `09_compare_jepa_freeze.ipynb` | 20260805 | Side-by-side fz vs unfz: loss, cosine, collapse, baselines, VICReg |
| `compare_jepa_runs.py` | 20260805 | CLI numeric+plot compare (default 1120 vs 1128; pass `--runs` for 1622) |
| `run_tensorboard_jepa.sh` | 20260805 | TB on port 6007 for kept runs |
| `plot_jepa_curves.py` | 20260801 | Single-run CSV curve plotter |
| `jepa_eda.py` | 20260805 | Data / PE sanity checks for LPS pairing + positional encodings → `jepa_eda.png` |
| `jepa_phase_a_overfit.py` + `run_jepa_phase_a_overfit.sh` | 20260803 | **Debug only**: tiny overfit / smoke before full-data runs |

## Docs (repo `docs/`)

| File | Why |
|------|-----|
| `jepa_phaseA_architecture_ablations.md` (+ `.pdf`) | Handoff: architecture, freeze/fresh ablations, eval checklist |
| `jepa_research_plan.md` | High-level Phase A→D plan |
| `jepa_research_trajectory.plan.md` | Earlier planning notes |

## Quick commands

```bash
# Full-data train (examples)
JEPA_ENCODER=scmaskgit FREEZE_ENCODER=false bash docs/examples/run_train_jepa_sod2.sh
JEPA_ENCODER=cell FREEZE_ENCODER=false WARMSTART_TOKEN_EMB=true bash docs/examples/run_train_jepa_sod2.sh

# Compare / TB
python docs/examples/compare_jepa_runs.py
bash docs/examples/run_tensorboard_jepa.sh
```

Compare plots land in `jepa_compare_kept_runs/` (regenerate anytime; safe to delete).
