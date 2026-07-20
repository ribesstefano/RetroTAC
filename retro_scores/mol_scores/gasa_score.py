"""
gasa_score.py
=============
GASA (Graph Attention Synthetic Accessibility; Yu et al., 2022).
Output: binary prediction (0=ES easy, 1=HS hard) + ES probability [0,1].

Run from the cloned GASA repo (cadd-synthetic/GASA). GASA reads files
relative to its own directory, so we chdir into it for the call and
restore afterwards.
"""

import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem

COLUMNS = ["gasa_pred", "gasa_es_prob"]


def compute(smiles: list) -> pd.DataFrame:
    print("Computing GASA")
    from . import GASA_DIR  # lazy to avoid circular import

    # GASA's own SMILES loop (Chem.MolFromSmiles + Chem.SanitizeMol) has no
    # None-check and raises on the first invalid SMILES, killing the whole
    # batch — pre-filter here and reinsert NaN at the same positions so one
    # bad molecule doesn't cost every other score in the run.
    valid_idx = [
        i for i, smi in enumerate(smiles)
        if isinstance(smi, str) and Chem.MolFromSmiles(smi) is not None
    ]
    valid_smiles = [smiles[i] for i in valid_idx]

    original_dir  = os.getcwd()
    original_argv = sys.argv.copy()
    pred_valid, pos_valid = [], []
    try:
        os.chdir(str(GASA_DIR))
        sys.path.insert(0, str(GASA_DIR))
        sys.argv = ["gasa.py"]  # override argv to avoid argparse conflict
        from gasa import GASA
        if valid_smiles:
            pred_valid, pos_valid, _ = GASA(valid_smiles)
    finally:
        os.chdir(original_dir)
        if str(GASA_DIR) in sys.path:
            sys.path.remove(str(GASA_DIR))
        sys.argv = original_argv

    pred = np.full(len(smiles), np.nan)
    pos = np.full(len(smiles), np.nan)
    for j, i in enumerate(valid_idx):
        pred[i] = pred_valid[j]
        pos[i] = pos_valid[j]
    return pd.DataFrame({"gasa_pred": pred, "gasa_es_prob": pos})
