"""
sc_score.py
===========
SCScore (Synthetic Complexity Score; Coley et al., 2018).
Raw: 1-5 (lower = easier).  Scaled: [0,1] (higher = easier).

Run from the cloned SCScore repo (CatSci fork ships the .json.gz weights).
"""

import sys
import numpy as np
import pandas as pd

from ._utils import safe_score

COLUMNS = ["sc_score", "sc_score_scaled"]


def compute(smiles: list) -> pd.DataFrame:
    print("Computing SCScore")
    from . import SCSCORE_DIR, SCSCORE_WEIGHTS  # lazy to avoid circular import
    sys.path.append(str(SCSCORE_DIR))
    from scscore.standalone_model_numpy import SCScorer

    scorer = SCScorer()
    scorer.restore(str(SCSCORE_WEIGHTS))

    raw = np.array([
        safe_score(lambda s: scorer.get_score_from_smi(s)[1], smi)
        for smi in smiles
    ], dtype=np.float64)
    scaled = 1 - (raw - 1) / (5 - 1)
    return pd.DataFrame({"sc_score": raw, "sc_score_scaled": scaled})
