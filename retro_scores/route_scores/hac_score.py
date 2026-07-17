"""
scoring_utils.py
================
Shared scoring helpers used by both the component and PROTAC AiZynthFinder scripts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_hac_weighted_score(
    precursors: List[Dict[str, Any]],
    route_length: Optional[int] = None,
    alpha: float = 0.95,
    max_route_length: int = 20,
) -> float:
    """Compute a heavy-atom-count weighted stock-availability score.

    Weights each precursor by its HAC so that large in-stock building blocks
    contribute more to the score than small ones. This avoids over-rewarding
    routes that trivially fragment the target into many tiny, cheap pieces.

    An optional route-length penalty is blended in with weight ``(1 - alpha)``
    to penalise unnecessarily long synthesis paths.

    Args:
        precursors: List of precursor dicts, each with keys ``precursor_smiles``
            (str) and ``in_stock`` (bool).
        route_length: Number of synthesis steps. If ``None``, no penalty is
            applied and the function returns pure availability.
        alpha: Blending coefficient between availability (1.0) and brevity (0.0).
            Default 0.95 keeps availability as the dominant signal.
        max_route_length: Normalisation constant for the route-length penalty.
            Routes longer than this are not penalised further.

    Returns:
        Score in [0, 1]. 0 means no heavy atoms are in stock; 1 means all are.
    """
    # Deferred to avoid the RDKit startup cost when the script is called with bad args.
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    hac_solved, hac_total = 0, 0
    for p in precursors:
        mol = Chem.MolFromSmiles(p["precursor_smiles"])
        if mol is None:
            logger.warning("Skipping malformed precursor SMILES: %s", p["precursor_smiles"])
            continue
        hac = Descriptors.HeavyAtomCount(mol)
        hac_total += hac
        if p["in_stock"]:
            hac_solved += hac
        logger.debug(
            "    precursor %-40s  hac=%2d  in_stock=%s",
            p["precursor_smiles"][:40],
            hac,
            p["in_stock"],
        )

    # Guard against the degenerate case where all precursor SMILES are invalid.
    if hac_total == 0:
        logger.debug("All precursor SMILES invalid or empty — score=0.0")
        return 0.0

    availability = hac_solved / hac_total
    logger.debug("HAC: %d/%d in stock  availability=%.4f", hac_solved, hac_total, availability)

    if route_length is not None:
        # Linear penalty: shorter routes score closer to `availability`.
        route_penalty = 1 - (route_length / max_route_length)
        score = alpha * availability + (1 - alpha) * route_penalty
        logger.debug(
            "Route-length penalty: steps=%d  penalty=%.4f  final=%.4f",
            route_length, route_penalty, score,
        )
        return score

    return availability
