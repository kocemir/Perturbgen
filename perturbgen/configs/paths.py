import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# Tokenized LPS + PerturbGen results live on sod2, not home T_perturb.
LPS_DATA = Path(
    os.environ.get(
        'LPS_DATA',
        '/mnt/sod2-project/csb4/stuke1/perturbgen_reproduction/lps',
    )
)
PROJECT_DIR = LPS_DATA
DATA_DIR = ROOT / 'data'
RESULTS_DIR = PROJECT_DIR / 'res'
TOKENIZED_DIR = PROJECT_DIR / 'tokenized_data'
