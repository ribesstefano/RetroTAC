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
FSscore  : raw unbounded (higher=easier) -> batch min-max normalized [0,1]

Usage
-----
    from mol_scores.scoring import compute_scores
    df = compute_scores(df)               # df needs a 'molecule' column
    # or from the command line:
    python -m mol_scores.scoring in.csv out.csv
"""

import numpy as np
import pandas as pd

from . import sa_score, sc_score, ra_score, syba_score, gasa_score, fs_score

# Each scorer module owns its output column names via a module-level
# COLUMNS constant, so the NaN-fallback path below can't drift out of sync
# with what compute() actually returns.
SCORERS = [
    ("SA score", sa_score),
    ("SCScore",  sc_score),
    ("RAscore",  ra_score),
    ("SYBA",     syba_score),
    ("GASA",     gasa_score),
    ("FSscore",  fs_score),
]

SCORE_COLUMNS = [col for _, module in SCORERS for col in module.COLUMNS]


def compute_scores(
    df: pd.DataFrame,
    smiles_col: str = "molecule",
    device: str = "auto",
    batch_size: int = 128,
    num_workers: int = 4,
) -> pd.DataFrame:
    """Compute all synthesizability scores and add them as columns to df.

    Args:
        df: DataFrame with a smiles_col column.
        smiles_col: Name of the SMILES column.
        device: Forwarded to fs_score.compute only -- see its module
            docstring for the login-node GPU-crash caveat ('auto' grabs a
            GPU if one is visible, which fails on a login node that has one
            it cannot actually use; pass 'cpu' there). Every other scorer
            here takes no device argument.
        batch_size: Forwarded to fs_score.compute only -- molecules per
            forward pass through FSscore's RankNet.
        num_workers: Forwarded to fs_score.compute only -- FSscore's
            dataloader worker count.
    """
    smiles = df[smiles_col].tolist()
    n = len(smiles)
    print(f"Computing synthesizability scores for {n} molecules...")

    for name, module in SCORERS:
        print(f"  Computing {name}...", end=" ", flush=True)
        try:
            if module is fs_score:
                scores_df = module.compute(
                    smiles, device=device, batch_size=batch_size, num_workers=num_workers
                )
            else:
                scores_df = module.compute(smiles)
            for col in scores_df.columns:
                df[col] = scores_df[col].values
            n_nan = int(scores_df.isna().any(axis=1).sum())
            print(f"done ({n_nan}/{n} molecules NaN)." if n_nan else "done.")
        except Exception as e:
            print(f"FAILED ({e})")
            print(f"  WARNING: {name} scores will be NaN.")
            for col in module.COLUMNS:
                df[col] = np.nan

    print("All scores computed.")
    return df
