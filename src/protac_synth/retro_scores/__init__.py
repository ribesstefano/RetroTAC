"""
retro_scores
============
Synthesizability / retrosynthesis scoring for PROTAC molecules.

Public API:
    compute_scores(df, smiles_col="molecule")  -- all scores at once
    SCORE_COLUMNS                              -- list of score column names

Individual scorers are also importable, e.g.:
    from protac_synth.retro_scores import sa_score
    sa_score.compute(smiles_list)

Paths to the external scorer repos / checkpoints also live here so there's
a single place to edit them. Layout assumed (repo root):
    PROTAC-Synthesizability/
    ├── src/protac_synth/retro_scores/   <- this package
    └── external/
        ├── SCScore/                     <- git clone CatSci/SCScore
        └── GASA/                        <- git clone cadd-synthetic/GASA
"""

from pathlib import Path

# retro_scores/ -> protac_synth/ -> src/ -> PROJECT_ROOT
# (adjust the index if your nesting differs)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_DIR = PROJECT_ROOT / "external"

SCSCORE_DIR = EXTERNAL_DIR / "SCScore"
GASA_DIR    = EXTERNAL_DIR / "GASA"

# CatSci fork ships the .json.gz weights that sc_score.py loads
SCSCORE_WEIGHTS = (
    SCSCORE_DIR / "models" / "full_reaxys_model_1024bool" /
    "model.ckpt-10654.as_numpy.json.gz"
)

# FSscore checkpoint — downloaded separately (gdown / Google Drive).
# Point this at wherever you save it, and match the real filename.
FSSCORE_MODEL = EXTERNAL_DIR / "fsscore" / "models" / "pretrain_graph_GGLGGL_ep242_best_valloss.ckpt"


from .scoring import compute_scores, SCORE_COLUMNS

# your existing personalized score
from . import hac_score

__all__ = [
    "compute_scores", "SCORE_COLUMNS", "hac_score",
    "SCSCORE_DIR", "GASA_DIR", "SCSCORE_WEIGHTS", "FSSCORE_MODEL",
]