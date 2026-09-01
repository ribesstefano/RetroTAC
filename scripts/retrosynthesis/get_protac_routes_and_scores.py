"""
get_protac_routes_and_scores.py
================================
Stage 3, step 1 — run AiZynthFinder retrosynthesis on full PROTAC molecules.

Loads SMILES from a CSV, assigns each SLURM array task an interleaved chunk
of the molecule list, and runs AiZynthFinder molecule by molecule, recording
the best route's default score, HAC-weighted score, route length, and
per-precursor in-stock flags. Results are written incrementally so that
interrupted runs can be resumed without reprocessing already-scored molecules.

Usage
-----
    # Single job
    python scripts/retrosynthesis/get_protac_routes_and_scores.py \\
        --config config/aizynthfinder_protac_config.yaml \\
        --stock_db data/external/aizynthfinder_stock.db \\
        --input_csv data/processed/protac_smiles.csv \\
        --outdir data/processed/scores/protacs

    # SLURM array task (task 2 of 8)
    python scripts/retrosynthesis/get_protac_routes_and_scores.py \\
        --config config/aizynthfinder_protac_config.yaml \\
        --stock_db data/external/aizynthfinder_stock.db \\
        --input_csv data/processed/protac_smiles.csv \\
        --outdir data/processed/scores/protacs \\
        --total_tasks 8 --task_id 2

I/O
---
    in :  <input_csv>, <stock_db>, <config>
    out:  <outdir>/<prefix>[_task<N>].jsonl           full route data (one JSON record per line)
          <outdir>/<prefix>[_task<N>]_summary.csv     one row per molecule
          <outdir>/<prefix>[_task<N>]_precursors.csv  one row per precursor
          <outdir>/<prefix>[_task<N>]_steps.csv       one row per synthesis step
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

# route_scores/ isn't pip-installed by the main uv-managed env (its heavy-dep
# sibling mol_scores/ needs a separate scoring_env — see retro_scores/README.md),
# so reach it directly via sys.path. Also needs the repo root itself on
# sys.path for its own src.protac_synth.stock_utils reach-across.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_REPO_ROOT))
sys.path.append(str(_REPO_ROOT / "retro_scores"))

from route_scores.aizynthfinder_utils import append_df, extract_route_data, init_finder  # noqa: E402

logger = logging.getLogger(__name__)


def load_smiles_from_csv(csv_path: Path, smiles_col: Optional[str] = None) -> List[str]:
    """Load SMILES strings from a CSV, auto-detecting the SMILES column if needed.

    Args:
        csv_path: Path to the input CSV file.
        smiles_col: Name of the SMILES column. If ``None``, common column names
            are tried in order before raising an error.

    Returns:
        List of non-null SMILES strings in their original order.

    Raises:
        ValueError: If the specified column is absent, or if auto-detection fails.
    """
    df = pd.read_csv(csv_path)

    if smiles_col:
        if smiles_col not in df.columns:
            raise ValueError(
                f"Column '{smiles_col}' not found. Available: {list(df.columns)}"
            )
        return df[smiles_col].dropna().unique().tolist()

    for candidate in ("smiles", "SMILES", "Smiles", "smi", "SMI", "canonical_smiles"):
        if candidate in df.columns:
            logger.info("Auto-detected SMILES column: '%s'", candidate)
            return df[candidate].dropna().unique().tolist()

    raise ValueError(
        f"Could not auto-detect SMILES column. Available: {list(df.columns)}\n"
        "Specify with --smiles_col <column_name>."
    )


def get_chunk(molecules: List[str], task_id: int, total_tasks: int) -> List[str]:
    """Return the interleaved subset of *molecules* assigned to *task_id*.

    Interleaved (round-robin) assignment spreads molecules evenly across tasks
    and avoids any ordering bias that would arise from contiguous block splits.

    Args:
        molecules: Full list of SMILES to distribute.
        task_id: Zero-based index of the current task.
        total_tasks: Total number of parallel tasks.

    Returns:
        Molecules at positions task_id, task_id + total_tasks, task_id + 2*total_tasks, …
    """
    return molecules[task_id::total_tasks]


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Returns:
        Populated ``argparse.Namespace`` with all arguments as typed attributes.
    """
    ap = argparse.ArgumentParser(
        description="Run AiZynthFinder retrosynthesis on full PROTAC molecules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--config", required=True, type=Path,
        help="AiZynthFinder config YAML (config/aizynthfinder_protac_config.yaml).",
    )
    ap.add_argument(
        "--stock_db", required=True, type=Path,
        help="SQLite database containing all custom stock tables.",
    )
    ap.add_argument(
        "--input_csv", required=True, type=Path,
        help="CSV file containing SMILES to score.",
    )
    ap.add_argument(
        "--outdir", required=True, type=Path,
        help="Output directory for result CSVs and JSONL checkpoint.",
    )
    ap.add_argument(
        "--prefix", default="protac",
        help="Filename stem for output files.",
    )
    ap.add_argument(
        "--smiles_col", default=None,
        help="SMILES column name (auto-detected if omitted).",
    )
    ap.add_argument(
        "--total_tasks", type=int, default=None,
        help="Total number of SLURM array tasks. Overrides SLURM_ARRAY_TASK_COUNT.",
    )
    ap.add_argument(
        "--task_id", type=int, default=None,
        help="This task's 0-based index. Overrides SLURM_ARRAY_TASK_ID.",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Smoke-test cap: process only the first N molecules (0 = no limit).",
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging (precursor HAC, reaction steps, stock lookups).",
    )
    return ap.parse_args()


def main() -> None:
    """Entry point: load molecules, assign chunk, and run AiZynthFinder molecule by molecule."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== AiZynthFinder PROTAC scoring ===")
    args.outdir.mkdir(parents=True, exist_ok=True)

    # SLURM_ARRAY_TASK_ID is 1-based; convert to 0-based task_id.
    task_id = (
        args.task_id if args.task_id is not None
        else int(os.environ.get("SLURM_ARRAY_TASK_ID", 1)) - 1
    )
    total_tasks = (
        args.total_tasks if args.total_tasks is not None
        else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    )

    all_molecules = load_smiles_from_csv(args.input_csv, args.smiles_col)
    logger.info("Loaded %d molecules from %s", len(all_molecules), args.input_csv)

    my_molecules = get_chunk(all_molecules, task_id, total_tasks)
    logger.info(
        "Task %d/%d — %d total | %d in this chunk",
        task_id, total_tasks - 1, len(all_molecules), len(my_molecules),
    )

    # Output file stems: include task suffix when running as an array job so
    # concurrent tasks never write to the same file.
    stem = f"{args.prefix}_task{task_id}" if total_tasks > 1 else args.prefix
    out_jsonl = args.outdir / f"{stem}.jsonl"
    out_sum = args.outdir / f"{stem}_summary.csv"
    out_precs = args.outdir / f"{stem}_precursors.csv"
    out_steps = args.outdir / f"{stem}_steps.csv"

    logger.info("Output prefix: %s/%s", args.outdir, stem)

    # Resume: skip molecules already present in the summary CSV so the job can
    # be safely restarted after a SLURM timeout without re-running work.
    if out_sum.exists():
        done_df = pd.read_csv(out_sum)
        if "molecule" in done_df.columns:
            done = set(done_df["molecule"].astype(str).str.strip())
            n_before = len(my_molecules)
            my_molecules = [s for s in my_molecules if s not in done]
            logger.info(
                "Resume: skipped %d already-scored molecules (%d remaining)",
                n_before - len(my_molecules), len(my_molecules),
            )

    if args.limit > 0:
        my_molecules = my_molecules[: args.limit]
        logger.info("Smoke-test limit applied: processing first %d molecules", len(my_molecules))

    if not my_molecules:
        logger.info("Nothing to do — all molecules already scored.")
        return

    finder = init_finder(args.config, args.stock_db)
    logger.info("Starting route extraction for %d molecules", len(my_molecules))

    for i, smiles in enumerate(my_molecules, 1):
        logger.info("[%d/%d] %s", i, len(my_molecules), smiles[:80])

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
                                "molecule": route_data["target_smiles"],
                                "aizynthfinder_score": route_data["route_score"],
                                "hac_weighted_score": route_data["hac_weighted_score"],
                                "search_time": route_data["search_time_seconds"],
                                "precursors": len(route_data["precursors"]),
                                "in_stock": sum(1 for p in route_data["precursors"] if p["in_stock"]),
                                "steps": len(route_data["synthesis_steps"]),
                            }
                        ]
                    ),
                )

                prec_rows = pd.DataFrame(
                    [
                        {
                            "molecule": route_data["target_smiles"],
                            "score": route_data["route_score"],
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
                            "molecule": route_data["target_smiles"],
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
                # Record a zero-score row so the molecule is not retried on resume.
                append_df(
                    out_sum,
                    pd.DataFrame(
                        [
                            {
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
            logger.error("  Failed on %s: %s", smiles[:60], e, exc_info=args.verbose)

    logger.info(
        "Done. Saved:\n  %s\n  %s\n  %s\n  %s",
        out_jsonl, out_sum, out_precs, out_steps,
    )


if __name__ == "__main__":
    main()
