"""
syba_score.py
=============
SYBA (Bayesian synthesizability classifier; Voršilák et al., 2020).
Raw: unbounded (higher = easier).  Scaled: sigmoid -> [0,1].

Installed as a pip/git package (lich-uct/syba).
"""

import numpy as np
import pandas as pd

from ._utils import sigmoid, safe_score

COLUMNS = ["syba_score", "syba_score_scaled"]


def compute(smiles: list) -> pd.DataFrame:
    print("Computing SYBA")
    from syba.syba import SybaClassifier
    syba = SybaClassifier()
    syba.fitDefaultScore()
    raw = np.array([safe_score(syba.predict, smi) for smi in smiles])
    scaled = sigmoid(raw)
    return pd.DataFrame({"syba_score": raw, "syba_score_scaled": scaled})
