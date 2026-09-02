"""
aizynthfinder_utils.py
======================
Shared AiZynthFinder helpers used by both the component-scoring and
PROTAC-scoring retrosynthesis scripts.

Callers must have the repo root on ``sys.path`` before importing (not just
``src/``) — ``init_finder`` reaches across to ``src.retrotac.stock_utils``,
which is an implicit namespace package rooted one level above ``src/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from aizynthfinder.aizynthfinder import AiZynthFinder

from .hac_score import compute_hac_weighted_score

logger = logging.getLogger(__name__)


def extract_route_data(
    route: Dict[str, Any],
    finder: "AiZynthFinder",
    original_smiles: str,
    search_time: float,
) -> Dict[str, Any]:
    """Extract score, precursors, and synthesis steps from the best AiZynthFinder route.

    Traverses the reaction tree to collect leaf precursors, checks each against
    the configured stock, and recursively extracts reaction steps. The HAC-weighted
    score is computed from the extracted precursor list and route length.

    Args:
        route: The top-ranked route object returned by AiZynthFinder (``finder.routes[0]``).
        finder: The active AiZynthFinder instance — needed to query stock availability.
        original_smiles: SMILES of the target molecule, used as the record key.
        search_time: Wall-clock seconds spent on the tree search, stored for profiling.

    Returns:
        Dict with keys: ``target_smiles``, ``route_score``, ``hac_weighted_score``,
        ``search_time_seconds``, ``precursors`` (list of dicts), ``synthesis_steps``
        (list of dicts).
    """
    reaction_tree = route.get("reaction_tree")
    # "state score" is AiZynthFinder's default MCTS evaluation of the route quality.
    score = route.get("score", {}).get("state score", 0.0)
    logger.debug("Best route state score: %.4f", score or 0.0)

    precursors: List[Dict[str, Any]] = []
    if reaction_tree:
        stock = finder.config.stock
        for leaf in reaction_tree.leafs():
            smi = leaf.smiles
            in_stock = stock.smiles_in_stock(smi)
            # availability_list returns source names (e.g. "zinc", "vendor_stock").
            stock_source = (
                ",".join(s.upper() for s in stock.availability_list(leaf)) if in_stock else None
            )
            precursors.append(
                {"precursor_smiles": smi, "in_stock": bool(in_stock), "stock_source": stock_source}
            )
            logger.debug(
                "  leaf: %-50s  in_stock=%-5s  source=%s",
                smi[:50], bool(in_stock), stock_source or "—",
            )

    logger.debug("Extracted %d leaf precursors", len(precursors))

    steps: List[Dict[str, Any]] = []
    # Use a list as a mutable counter so the nested closure can increment it
    # without a `nonlocal` declaration (nonlocal requires Python 3, but this
    # pattern also avoids re-binding the name, which some linters flag).
    step_counter = [0]

    def _extract_steps(node: Optional[Dict[str, Any]]) -> None:
        """Recursively walk the reaction-tree dict and collect reaction steps."""
        if node is None:
            return
        node_type = node.get("type")
        if node_type == "mol" and node.get("children"):
            for reaction in node["children"]:
                if reaction.get("type") != "reaction":
                    continue
                meta = reaction.get("metadata", {})
                reactants = [
                    c["smiles"] for c in reaction.get("children", []) if c.get("type") == "mol"
                ]
                product = node.get("smiles")
                if reactants and product:
                    step_counter[0] += 1
                    steps.append(
                        {
                            "step_number": step_counter[0],
                            "reactants": reactants,
                            "product": product,
                            "template_code": meta.get("template_code"),
                            "probability": meta.get("policy_probability", 0.0),
                        }
                    )
                    logger.debug(
                        "  step %d: [%s] → %s  (p=%.3f)",
                        step_counter[0],
                        " + ".join(r[:25] for r in reactants),
                        product[:25],
                        meta.get("policy_probability", 0.0),
                    )
                for child in reaction.get("children", []):
                    _extract_steps(child)
        elif node_type == "reaction":
            for child in node.get("children", []):
                _extract_steps(child)

    if reaction_tree:
        # Some AiZynthFinder versions return an object rather than a plain dict;
        # to_dict() normalises it before we recurse.
        tree_dict = (
            reaction_tree.to_dict() if hasattr(reaction_tree, "to_dict") else reaction_tree
        )
        _extract_steps(tree_dict)

    logger.debug("Extracted %d synthesis steps", len(steps))
    hac_score = compute_hac_weighted_score(precursors, route_length=len(steps))

    return {
        "target_smiles": original_smiles,
        "route_score": float(score or 0.0),
        "hac_weighted_score": float(hac_score or 0.0),
        "search_time_seconds": float(search_time),
        "precursors": precursors,
        "synthesis_steps": steps,
    }


def append_df(path: Path, df: pd.DataFrame) -> None:
    """Append *df* to a CSV at *path*, writing the header only on first write.

    Args:
        path: Destination CSV path.  Created if it does not exist.
        df: DataFrame to append.
    """
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def init_finder(
    config_path: Path,
    stock_db: Path,
    extra_stock_tables: Optional[List[str]] = None,
    max_transforms: Optional[int] = None,
    max_iterations: Optional[int] = None,
    time_limit: Optional[int] = None,
) -> "AiZynthFinder":
    """Initialise AiZynthFinder with expansion/filter policies and all custom SQLite stocks.

    Loads the base domain-specific stock tables (fragments, amines, acids, vendor
    catalogues, solved components) plus any caller-specified extras.  Search
    hyperparameters stay at their config-YAML defaults unless overrides are passed.

    The heavy imports (AiZynthFinder → TensorFlow/PyTorch, SQLiteStock) are
    deferred here so that argument parsing and data validation run at full speed.

    Args:
        config_path: Path to the AiZynthFinder config YAML.
        stock_db: Path to the SQLite database holding all custom stock tables.
        extra_stock_tables: Additional table names to load on top of the base set
            (e.g. ``["warhead", "e3_ligase", "linker", "capped_smiles"]`` for
            component scoring).
        max_transforms: Override ``finder.config.max_transforms`` when set.
        max_iterations: Override ``finder.config.max_iterations`` when set.
        time_limit: Override ``finder.config.time_limit`` (seconds) when set.

    Returns:
        A fully initialised ``AiZynthFinder`` instance ready for ``tree_search()``.
    """
    # Deferred imports: AiZynthFinder triggers TensorFlow/PyTorch initialisation
    # (CUDA device scan, kernel JIT), which can take 10-30 s.  Keeping them here
    # means --help and bad-argument errors complete instantly.
    from aizynthfinder.aizynthfinder import AiZynthFinder
    from src.retrotac.stock_utils.sqlite_stock import SQLiteStock

    logger.info("Initialising AiZynthFinder from config: %s", config_path)
    finder = AiZynthFinder(configfile=str(config_path))
    finder.config.expansion_policy.select("uspto")
    finder.config.filter_policy.select("uspto")
    logger.debug("Expansion policy: uspto  |  Filter policy: uspto")

    base_tables = [
        "fragments",
        "aliphatic_carboxylic_acids",
        "aliphatic_primary_amines",
        "aromatic_carboxylic_acids",
        "aromatic_primary_amines",
        "secondary_amines",
        "terminal_acetylenes",
        "vendor_stock",
        "solved_stock",
    ]
    all_tables = base_tables + (extra_stock_tables or [])

    db_path = str(stock_db)
    for table in all_tables:
        logger.debug("Loading stock table: %s", table)
        finder.config.stock.load(SQLiteStock(db_path, table), table)

    finder.config.stock.select(["zinc"] + all_tables)
    logger.info("Loaded %d custom stock tables + zinc", len(all_tables))

    if max_transforms is not None:
        finder.config.max_transforms = max_transforms
    if max_iterations is not None:
        finder.config.max_iterations = max_iterations
    if time_limit is not None:
        finder.config.time_limit = time_limit
    logger.debug(
        "Search config: max_transforms=%s  max_iterations=%s  time_limit=%s",
        finder.config.max_transforms,
        finder.config.max_iterations,
        finder.config.time_limit,
    )

    return finder
