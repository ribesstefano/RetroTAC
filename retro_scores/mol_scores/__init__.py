"""
mol_scores
==========
Per-molecule synthesizability scoring for PROTAC molecules — six independent
scorers (SA score, SCScore, RAscore, SYBA, GASA, FSscore), each with a raw
and a scaled value. Each scorer only needs a SMILES string, no retrosynthesis
route (contrast with the sibling `route_scores` package).

Installed and run separately from the main `retrotac` package (see
README.md in this directory): the dependency stack here (TF 2.8 +
torch 2.0 + dgl 2.1 + old xgboost, three git-cloned/manually-fetched
models) is fragile and Python-3.10-pinned, so it lives in its own
`scoring_env` rather than the main uv-managed environment.

Public API:
    compute_scores(df, smiles_col="molecule")  -- all scores at once
    SCORE_COLUMNS                              -- list of score column names

Individual scorers are also importable, e.g.:
    from mol_scores import sa_score
    sa_score.compute(smiles_list)

Paths to the external scorer repos / checkpoints also live here so there's
a single place to edit them. Layout assumed (repo root):
    PROTAC-Synthesizability/
    ├── retro_scores/mol_scores/   <- this package
    └── external/
        ├── SCScore/                <- git clone CatSci/SCScore
        └── GASA/                   <- git clone cadd-synthetic/GASA

Set PROTAC_EXTERNAL_DIR to override the external/ location instead (used by
apptainer/scoring.def, which bakes these into the image at a path outside
the live-code bind mount so they aren't shadowed by a host checkout that
doesn't have them -- see apptainer/README.md).
"""

import os
from pathlib import Path

# mol_scores/ (package) -> retro_scores/ (sub-project root) -> PROJECT_ROOT
# (adjust the index if your nesting differs)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = Path(os.environ["PROTAC_EXTERNAL_DIR"]) if "PROTAC_EXTERNAL_DIR" in os.environ \
    else PROJECT_ROOT / "external"

SCSCORE_DIR = EXTERNAL_DIR / "SCScore"
GASA_DIR    = EXTERNAL_DIR / "GASA"

# CatSci fork ships the .json.gz weights that sc_score.py loads
SCSCORE_WEIGHTS = (
    SCSCORE_DIR / "models" / "full_reaxys_model_1024bool" /
    "model.ckpt-10654.as_numpy.json.gz"
)

# FSscore checkpoint — downloaded separately (figshare).
# Point this at wherever you save it, and match the real filename.
FSSCORE_MODEL = EXTERNAL_DIR / "fsscore" / "models" / "pretrain_graph_GGLGGL_ep242_best_valloss.ckpt"


from .scoring import compute_scores, SCORE_COLUMNS

__all__ = [
    "compute_scores", "SCORE_COLUMNS",
    "SCSCORE_DIR", "GASA_DIR", "SCSCORE_WEIGHTS", "FSSCORE_MODEL",
]
