"""
route_scores
============
Scoring that needs a full retrosynthesis route (an AiZynthFinder search
tree), not just a molecule — contrast with the sibling `mol_scores` package.
Currently: the HAC-weighted score (`hac_score.py`), computed from the
precursor list and route length of a searched tree. More route-based scores
will be added here later.

Unlike `mol_scores`, this package's only heavy dependency is `aizynthfinder`
itself, which the main `retrotac` environment already installs — so this
is meant to run there, not in `scoring_env`.

Public API:
    compute_hac_weighted_score(precursors, route_length=None, ...)
    extract_route_data(route, finder, original_smiles, search_time)
    init_finder(config_path, stock_db, ...)
    append_df(path, df)
"""

from .hac_score import compute_hac_weighted_score
from .aizynthfinder_utils import append_df, extract_route_data, init_finder

__all__ = [
    "compute_hac_weighted_score",
    "append_df", "extract_route_data", "init_finder",
]
