"""
scoring.py
==========
Orchestrator: computes all synthesizability scores for a DataFrame of
SMILES and adds them as columns. Each scorer lives in its own module and
is called here; if one fails (missing repo, missing checkpoint, bad env)
its columns are filled with NaN and the rest still run.

Scores
------
SA score : raw 1-10 (lower=easier)   -> scaled [0,1] (higher=easier)
SCScore  : raw 1-5  (lower=easier)   -> scaled [0,1] (higher=easier)
RAscore  : raw [0,1] (higher=easier)  -- no scaling
SYBA     : raw unbounded (higher=easier) -> sigmoid [0,1]
GASA     : binary 0=ES/1=HS + ES probability [0,1]
FSscore  : raw [0,1] (higher=easier) -> sigmoid(raw/10)

Usage
-----
    from protac_synth.retro_scores.scoring import compute_scores
    df = compute_scores(df)               # df needs a 'molecule' column
    # or from the command line:
    python -m protac_synth.retro_scores.scoring in.csv out.csv
"""

import numpy as np
import pandas as pd

from . import sa_score, sc_score, ra_score, syba_score, gasa_score, fs_score


# (name, module, output columns)  -- columns listed so we can NaN-fill on failure
SCORERS = [
    ("SA score", sa_score,   ["sa_score", "sa_score_scaled"]),
    ("SCScore",  sc_score,   ["sc_score", "sc_score_scaled"]),
    ("RAscore",  ra_score,   ["ra_score"]),
    ("SYBA",     syba_score, ["syba_score", "syba_score_scaled"]),
    ("GASA",     gasa_score, ["gasa_pred", "gasa_es_prob"]),
    ("FSscore",  fs_score,   ["fs_score", "fs_score_scaled"]),
]

SCORE_COLUMNS = [c for _, _, cols in SCORERS for c in cols]


def compute_scores(df: pd.DataFrame, smiles_col: str = "molecule") -> pd.DataFrame:
    """Compute all synthesizability scores and add them as columns to df."""
    smiles = df[smiles_col].tolist()
    n = len(smiles)
    print(f"Computing synthesizability scores for {n} molecules...")

    for name, module, columns in SCORERS:
        print(f"  Computing {name}...", end=" ", flush=True)
        try:
            scores_df = module.compute(smiles)
            for col in scores_df.columns:
                df[col] = scores_df[col].values
            print("done.")
        except Exception as e:
            print(f"FAILED ({e})")
            print(f"  WARNING: {name} scores will be NaN.")
            for col in columns:
                df[col] = np.nan

    print("All scores computed.")
    return df