"""
gasa_score.py
=============
GASA (Graph Attention Synthetic Accessibility; Yu et al., 2022).
Output: binary prediction (0=ES easy, 1=HS hard) + ES probability [0,1].

Run from the cloned GASA repo (cadd-synthetic/GASA). GASA reads files
relative to its own directory, so we chdir into it for the call and
restore afterwards. Path comes from retro_scores._paths.
"""

import os
import sys
import pandas as pd


def compute(smiles: list) -> pd.DataFrame:
    print("Computing GASA")
    from . import GASA_DIR  # lazy to avoid circular import
    original_dir  = os.getcwd()
    original_argv = sys.argv.copy()
    try:
        os.chdir(str(GASA_DIR))
        sys.path.insert(0, str(GASA_DIR))
        sys.argv = ["gasa.py"]  # override argv to avoid argparse conflict
        from gasa import GASA
        pred, pos, neg = GASA(smiles)
    finally:
        os.chdir(original_dir)
        if str(GASA_DIR) in sys.path:
            sys.path.remove(str(GASA_DIR))
        sys.argv = original_argv

    return pd.DataFrame({"gasa_pred": pred, "gasa_es_prob": pos})