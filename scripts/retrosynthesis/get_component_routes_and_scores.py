"""
get_component_routes_and_scores.py
===================================
Stage 2, step 1 — run AiZynthFinder retrosynthesis on capped components.

Loads the capped components, filters to a single ``(component_type, cap)`` slice,
and runs AiZynthFinder on each unique capped SMILES, recording the default route
score, the HAC-weighted score, route length, and per-precursor in-stock flags.
Because the cap sweep is large (warhead/E3: 6 caps each; linker: 36 cap pairs),
each slice is run as its own job — the data orchestrator expands all 48 slices
automatically. Output is resumable (already-scored molecules are skipped).

Usage
-----
    python scripts/retrosynthesis/get_component_routes_and_scores.py \\
        --config config/config.yml \\
        --stock_db data/external/aizynthfinder_stock.db \\
        --csv data/processed/component_capped.csv \\
        --component_type warhead --cap OH \\
        --outdir data/processed/scores/components

I/O
---
    in :  data/processed/component_capped.csv, data/external/aizynthfinder_stock.db
    out:  <outdir>/<component_type>_cap_<cap>_results_{summary,precursors,steps}.csv
          (+ a .jsonl checkpoint)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add src/ to the path so aizynthfinder_utils, scoring_utils, etc. are importable.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from aizynthfinder_utils import append_df, extract_route_data, init_finder  # noqa: E402

logger = logging.getLogger(__name__)

# Maps the user-facing --component_type choice to the component label derived from component_id.
# "e3" is shortened for convenience on the command line.
_TARGET_MAP: Dict[str, str] = {
    "warhead": "warhead",
    "linker": "linker",
    "e3": "e3",
}

# Extra stock tables loaded only for component scoring (isolated fragments, not full PROTACs).
_COMPONENT_STOCK_TABLES: List[str] = ["warhead", "e3_ligase", "linker", "capped_smiles"]


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Returns:
        Populated ``argparse.Namespace`` with all arguments as typed attributes.
    """
    ap = argparse.ArgumentParser(
        description="Run AiZynthFinder on a filtered slice of capped components."
    )
    ap.add_argument("--config", required=True, type=Path, help="AiZynthFinder config YAML.")
    ap.add_argument(
        "--stock_db",
        required=True,
        type=Path,
        help="SQLite database containing all custom stock tables (e.g. aizynthfinder_stock.db).",
    )
    ap.add_argument(
        "--csv", required=True, type=Path, help="Capped components CSV (component_capped.csv)."
    )
    ap.add_argument(
        "--component_type",
        required=True,
        choices=list(_TARGET_MAP),
        help="Component type to score (warhead, linker, e3).",
    )
    ap.add_argument(
        "--outdir", required=True, type=Path, help="Output directory for result CSVs and JSONL."
    )
    ap.add_argument(
        "--cap",
        required=True,
        help="Cap group applied to the component, e.g. H, OH, NH2, or 'H|NH2' for linkers.",
    )
    ap.add_argument(
        "--smiles_col",
        default="cap_smiles",
        help="Column containing canonical capped SMILES.",
    )
    ap.add_argument(
        "--error_col", default="error", help="Column flagging capping errors to skip."
    )
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="Molecules per SLURM array chunk (0 = run all).",
    )
    ap.add_argument(
        "--chunk_idx",
        type=int,
        default=0,
        help="Zero-based index of the chunk to process (used with --chunk_size).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Smoke-test cap: process only the first N molecules (0 = no limit).",
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging (precursor HAC, reaction steps, stock lookups).",
    )
    return ap.parse_args()


def main() -> None:
    """Entry point: load data, filter to a cap slice, and run AiZynthFinder molecule by molecule."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== AiZynthFinder component scoring ===")
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Linker caps are always a pair (one cap per attachment point), e.g. "H|NH2".
    if args.component_type == "linker" and "|" not in args.cap:
        raise ValueError(
            f"Linker cap must contain '|' to separate both attachment-point caps, "
            f"e.g. 'H|H'. Got: {args.cap}"
        )

    # Replace characters that are illegal or ambiguous in filenames.
    # "|" separates multi-cap linker values; "=" appears in SMARTS-style caps like "=O".
    safe_cap = (
        args.cap.replace("|", "-").replace("=", "eq").replace("/", "_").replace(" ", "")
    )
    prefix = f"{args.component_type}_cap_{safe_cap}"
    if args.chunk_size > 0:
        prefix = f"{prefix}_chunk{args.chunk_idx}"

    # Four output streams: full JSONL checkpoint, summary row per molecule,
    # one row per precursor, and one row per synthesis step.
    out_jsonl = args.outdir / f"{prefix}_results.jsonl"
    out_sum   = args.outdir / f"{prefix}_results_summary.csv"
    out_precs = args.outdir / f"{prefix}_results_precursors.csv"
    out_steps = args.outdir / f"{prefix}_results_steps.csv"

    logger.info("Input CSV:  %s", args.csv)
    logger.info("Output dir: %s  (prefix=%s)", args.outdir, prefix)
    logger.debug(
        "Output files:\n  %s\n  %s\n  %s\n  %s",
        out_jsonl, out_sum, out_precs, out_steps,
    )

    df = pd.read_csv(args.csv)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # cap_type is the fixed column name for the capping group in component_capped.csv.
    for col in ("component_id", args.smiles_col, "cap_type"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Available: {list(df.columns)}")

    # Rows where capping failed carry a non-empty error string; exclude them
    # before building the SMILES list so we never send bad structures to AiZynthFinder.
    if args.error_col in df.columns:
        n_before = len(df)
        df = df[
            df[args.error_col].isna() | (df[args.error_col].astype(str).str.strip() == "")
        ].copy()
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.info("Dropped %d rows with capping errors", n_dropped)

    # Derive a short component label from the component_id prefix (e.g. "WH_001" → "warhead").
    df["component"] = df["component_id"].str.split("_").str[0]
    df["component"] = df["component"].replace({"E3": "e3", "LK": "linker", "WH": "warhead"})
    # Normalise before string comparison to avoid trailing-whitespace mismatches.
    df["cap_type"] = df["cap_type"].astype(str).str.strip()

    component_val = _TARGET_MAP[args.component_type]
    n_before = len(df)
    df = df[
        (df["component"] == component_val) & (df["cap_type"] == args.cap)
    ].copy()
    logger.info(
        "Filtered to component_type=%s  cap=%s: %d/%d rows",
        component_val, args.cap, len(df), n_before,
    )

    # AiZynthFinder is slow (~seconds/molecule); run once per unique capped SMILES
    # rather than once per row, since many rows share the same capped structure.
    smiles_list: List[str] = (
        df[args.smiles_col]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .tolist()
    )
    total_unique = len(smiles_list)
    logger.info("Unique capped SMILES: %d", total_unique)

    # Build a lookup so we can tag each result row with the component_id.
    # drop_duplicates ensures the first component_id wins when multiple rows
    # share the same capped SMILES (e.g. identical warheads from different PROTACs).
    smiles_to_comp_id: Dict[str, str] = (
        df.drop_duplicates(subset=["component_id", args.smiles_col])
        .set_index(args.smiles_col)["component_id"]
        .to_dict()
    )

    if args.chunk_size > 0:
        start_idx = args.chunk_idx * args.chunk_size
        # ceiling division without importing math
        n_chunks = -(- total_unique // args.chunk_size)
        smiles_list = smiles_list[start_idx : start_idx + args.chunk_size]
        logger.info(
            "Chunk %d/%d: indices %d–%d (%d molecules in this chunk)",
            args.chunk_idx, n_chunks - 1,
            start_idx, start_idx + len(smiles_list) - 1,
            len(smiles_list),
        )

    if args.limit > 0:
        smiles_list = smiles_list[: args.limit]
        logger.info("Smoke-test limit applied: processing first %d molecules", len(smiles_list))

    # Resume support: skip any SMILES already present in the summary CSV so the
    # job can be safely restarted after a SLURM timeout.
    if out_sum.exists():
        done_df = pd.read_csv(out_sum)
        if "molecule" in done_df.columns:
            done = set(done_df["molecule"].astype(str).str.strip())
            n_before_resume = len(smiles_list)
            smiles_list = [s for s in smiles_list if s not in done]
            logger.info(
                "Resume: skipped %d already-scored molecules (%d remaining)",
                n_before_resume - len(smiles_list),
                len(smiles_list),
            )

    if not smiles_list:
        logger.info("Nothing to do — all molecules already scored.")
        return

    finder = init_finder(
        args.config,
        args.stock_db,
        extra_stock_tables=_COMPONENT_STOCK_TABLES,
        max_transforms=12,
        max_iterations=400,
        time_limit=120,
    )
    logger.info("Starting route extraction for %d molecules", len(smiles_list))

    for i, smiles in enumerate(smiles_list, 1):
        comp_id = smiles_to_comp_id.get(smiles, "UNKNOWN")
        logger.info("[%d/%d] %s  |  %s", i, len(smiles_list), comp_id, smiles[:80])

        try:
            t0 = time.time()
            finder.target_smiles = smiles
            logger.debug("  Preparing search tree")
            finder.prepare_tree()
            logger.debug("  Running MCTS tree search")
            finder.tree_search()
            logger.debug("  Building routes from search tree")
            finder.build_routes()
            search_time = time.time() - t0

            if finder.routes:
                logger.debug("  %d routes found in %.1fs", len(finder.routes), search_time)
                # routes[0] is sorted by score descending; always take the best route.
                route_data = extract_route_data(finder.routes[0], finder, smiles, search_time)

                # JSONL checkpoint: full route data for post-hoc re-analysis without re-running.
                with open(out_jsonl, "a") as fh:
                    fh.write(json.dumps(route_data) + "\n")

                append_df(
                    out_sum,
                    pd.DataFrame(
                        [
                            {
                                "component_id": comp_id,
                                "molecule": route_data["target_smiles"],
                                "aizynthfinder_score": route_data["route_score"],
                                "hac_weighted_score": route_data["hac_weighted_score"],
                                "search_time": route_data["search_time_seconds"],
                                "precursors": len(route_data["precursors"]),
                                "in_stock": sum(
                                    1 for p in route_data["precursors"] if p["in_stock"]
                                ),
                                "steps": len(route_data["synthesis_steps"]),
                            }
                        ]
                    ),
                )

                prec_rows = pd.DataFrame(
                    [
                        {
                            "component_id": comp_id,
                            "molecule": route_data["target_smiles"],
                            "score": route_data["route_score"],
                            "search_time": route_data["search_time_seconds"],
                            "precursor": p["precursor_smiles"],
                            "in_stock": p["in_stock"],
                            "stock": p["stock_source"] or "N/A",
                        }
                        for p in route_data["precursors"]
                    ]
                )
                if not prec_rows.empty:
                    append_df(out_precs, prec_rows)

                step_rows = pd.DataFrame(
                    [
                        {
                            "component_id": comp_id,
                            "molecule": route_data["target_smiles"],
                            "score": route_data["route_score"],
                            "search_time": route_data["search_time_seconds"],
                            "step": s["step_number"],
                            "reactants": " + ".join(s["reactants"]),
                            "product": s["product"],
                            "template": s["template_code"],
                            "probability": s["probability"],
                        }
                        for s in route_data["synthesis_steps"]
                    ]
                )
                if not step_rows.empty:
                    append_df(out_steps, step_rows)

                n_in_stock = sum(1 for p in route_data["precursors"] if p["in_stock"])
                logger.info(
                    "  → score=%.4f  hac=%.4f  steps=%d  precursors=%d/%d in stock  time=%.1fs",
                    route_data["route_score"],
                    route_data["hac_weighted_score"],
                    len(route_data["synthesis_steps"]),
                    n_in_stock,
                    len(route_data["precursors"]),
                    search_time,
                )

            else:
                logger.warning("  No routes found (time=%.1fs)", search_time)
                # No route found: record a zero-score row so the molecule isn't retried.
                append_df(
                    out_sum,
                    pd.DataFrame(
                        [
                            {
                                "component_id": comp_id,
                                "molecule": smiles,
                                "aizynthfinder_score": 0.0,
                                "hac_weighted_score": 0.0,
                                "search_time": float(search_time),
                                "precursors": 0,
                                "in_stock": 0,
                                "steps": 0,
                            }
                        ]
                    ),
                )

            # Explicitly release the search tree between molecules; without this,
            # AiZynthFinder accumulates memory across iterations until OOM.
            finder._tree = None
            finder.routes = []
            logger.debug("  Search tree released")

        except Exception as e:
            logger.error("  Failed on %s: %s", comp_id, e, exc_info=args.verbose)

    logger.info(
        "Done. Saved:\n  %s\n  %s\n  %s\n  %s",
        out_jsonl, out_sum, out_precs, out_steps,
    )


if __name__ == "__main__":
    main()
