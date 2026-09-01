"""Merge routes.csv-aligned scoring outputs into one wide CSV: the base
routes.csv columns plus any combination of
scripts/chem_mol_scoring/molecule_scores.py's per-target molecule scores,
scripts/chem_route_scoring/route_scores.py's route-level scores, and an LLM
judge CSV from scripts/llm_scoring/llm_scoring.py.

Every input is joined by ROW POSITION, not by a SMILES key: routes.csv has a
handful of duplicate SMILES (convergent building blocks reused across
targets), and a SMILES-keyed join would multiply those rows. Row-position
joining is safe as long as every file was produced by scoring the exact same
routes CSV with no `--limit`/reordering -- which is checked (each input's
SMILES column must match the base file's, in order) before concatenating,
so a mismatched run fails loudly instead of silently misaligning columns.

Usage
-----
    python scripts/analysis/aggregate_route_scores.py \\
        --molecule-scores data/processed/chem_mol_scoring/routes_molecule_scores.csv \\
        --route-scores data/processed/chem_route_scoring/routes_route_scores.csv \\
        --llm-scores data/llm_scoring/routes_with_score_20260711142743.csv \\
        --output data/processed/analysis/routes_aggregated.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Original routes.csv columns, present verbatim in every scoring output below
# (each script reads routes.csv and appends its own columns) -- excluded from
# every input but the base file so they aren't duplicated in the merged output.
_BASE_COLUMNS = ["SMILES", "score", "resolved", "route", "BBs",
                 "search_duration", "resolved_depth", "error", "pool"]


def _load_new_columns(path: Path, base_smiles: pd.Series, label: str) -> pd.DataFrame:
    """Read a routes.csv-aligned CSV and return only the columns it added.

    Args:
        path: CSV produced by scoring the same routes CSV as `base_smiles`
            (same rows, same order) -- e.g. molecule_scores.py, route_scores.py,
            or llm_scoring.py's output.
        base_smiles: The base routes CSV's `SMILES` column, used to verify
            row alignment before a blind column-wise concat.
        label: Human-readable name for the error message.

    Returns:
        DataFrame with only the columns not already in `_BASE_COLUMNS`,
        row-aligned to `base_smiles`.

    Raises:
        ValueError: If `path` has a different row count or SMILES order than
            `base_smiles`.
    """
    df = pd.read_csv(path)
    if len(df) != len(base_smiles) or not (df["SMILES"].values == base_smiles.values).all():
        raise ValueError(
            f"{label} ({path}) is not row-aligned with the base routes CSV -- "
            "re-run it on the exact same input CSV, with no --limit or reordering.")
    new_cols = [c for c in df.columns if c not in _BASE_COLUMNS]
    return df[new_cols]


def aggregate_route_scores(routes_path: Path, output_path: Path,
                           molecule_scores_path: Optional[Path] = None,
                           route_scores_path: Optional[Path] = None,
                           llm_scores_path: Optional[Path] = None) -> None:
    """Concatenate the base routes CSV with any of its scoring outputs.

    Args:
        routes_path: Base routes CSV (default data/routes/routes.csv).
        output_path: Where to write the merged CSV.
        molecule_scores_path: Output of chem_mol_scoring/molecule_scores.py.
        route_scores_path: Output of chem_route_scoring/route_scores.py.
        llm_scores_path: Output of llm_scoring/llm_scoring.py.

    Raises:
        ValueError: If none of the three optional inputs are given, or if any
            given input is not row-aligned with `routes_path` (see
            `_load_new_columns`).
    """
    inputs = [
        (molecule_scores_path, "molecule scores"),
        (route_scores_path, "route scores"),
        (llm_scores_path, "LLM scores"),
    ]
    if not any(path is not None for path, _ in inputs):
        raise ValueError("pass at least one of --molecule-scores, --route-scores, --llm-scores")

    base = pd.read_csv(routes_path)
    parts: List[pd.DataFrame] = [base]
    for path, label in inputs:
        if path is not None:
            parts.append(_load_new_columns(path, base["SMILES"], label))

    out_df = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"merged {len(parts) - 1} scoring output(s) onto {len(out_df)} routes "
         f"({len(out_df.columns)} columns total) -> {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the aggregation CLI.

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--routes", type=Path, default=_PROJECT_ROOT / "data" / "routes" / "routes.csv",
                    help="base routes CSV all other inputs were scored from (default: %(default)s)")
    ap.add_argument("--molecule-scores", type=Path, default=None,
                    help="output of scripts/chem_mol_scoring/molecule_scores.py")
    ap.add_argument("--route-scores", type=Path, default=None,
                    help="output of scripts/chem_route_scoring/route_scores.py")
    ap.add_argument("--llm-scores", type=Path, default=None,
                    help="output of scripts/llm_scoring/llm_scoring.py")
    ap.add_argument("--output", type=Path,
                    default=_PROJECT_ROOT / "data" / "processed" / "analysis" / "routes_aggregated.csv",
                    help="merged output CSV (default: %(default)s)")
    return ap.parse_args()


def main() -> None:
    """CLI entry point: parse arguments and run the aggregation."""
    args = parse_args()
    aggregate_route_scores(args.routes, args.output, molecule_scores_path=args.molecule_scores,
                           route_scores_path=args.route_scores, llm_scores_path=args.llm_scores)


if __name__ == "__main__":
    main()
