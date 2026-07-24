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

from ._utils import safe_score

COLUMNS = ["sa_score", "sa_score_scaled"]


def compute(smiles: list) -> pd.DataFrame:
    print("Computing SAscore")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer

    def _score_one(smi):
        # MolFromSmiles('') returns a valid 0-atom Mol (not None), which
        # sascorer then divides by zero on — guard both that and non-string
        # (e.g. NaN) input explicitly rather than relying on `if mol`.
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None or mol.GetNumAtoms() == 0:
            return np.nan
        return sascorer.calculateScore(mol)

    raw = np.array([safe_score(_score_one, smi) for smi in smiles])
    scaled = 1 - (raw - 1) / (10 - 1)
    return pd.DataFrame({"sa_score": raw, "sa_score_scaled": scaled})
