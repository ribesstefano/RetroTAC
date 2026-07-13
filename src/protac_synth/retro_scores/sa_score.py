"""
sa_score.py
===========
SA score (Synthetic Accessibility Score; Ertl & Schuffenhauer, 2009).
Raw: 1-10 (lower = easier).  Scaled: [0,1] (higher = easier).

Uses RDKit's contrib sascorer (ships with RDKit).
"""

import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDConfig


def compute(smiles: list) -> pd.DataFrame:
    print("Computing SAscore")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer

    raw = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        raw.append(sascorer.calculateScore(mol) if mol else np.nan)
    raw = np.array(raw)
    scaled = 1 - (raw - 1) / (10 - 1)
    return pd.DataFrame({"sa_score": raw, "sa_score_scaled": scaled})