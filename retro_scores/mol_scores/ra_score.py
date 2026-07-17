"""
ra_score.py
===========
RAscore (Retrosynthetic Accessibility Score; Thakkar et al., 2021).
Raw: [0,1] (higher = easier).  No scaling needed.

Installed as a pip/git package (reymond-group/RAscore). Uses the XGB model.
"""

import numpy as np
import pandas as pd

from ._utils import safe_score

COLUMNS = ["ra_score"]


def compute(smiles: list) -> pd.DataFrame:
    print("Computing RAscore")
    from RAscore import RAscore_XGB
    scorer = RAscore_XGB.RAScorerXGB()
    raw = np.array([safe_score(scorer.predict, smi) for smi in smiles])
    return pd.DataFrame({"ra_score": raw})
