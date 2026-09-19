"""
dedupe_routes.py
─────────────────
Collapses duplicate rows in a scored-routes CSV down to one row per unique
whole-molecule SMILES.

Rows sharing the same SMILES typically come from independent route searches
for the same molecule (e.g. reruns, or the same molecule surfacing under
different search configs), not genuine repeats -- see the analysis in the
conversation that produced this script. Among duplicates, this keeps:
    1. a row with a resolved route (--resolved-col == True) over one without,
       so a failed search never displaces a successful one, then
    2. the highest --target-col value among the remaining ties, so the
       best-scoring route wins when several searches all resolved.

Usage
-----
    python scripts/dataset/dedupe_routes.py data/sets/routes_train_val.csv \\
        --smiles-col smiles --resolved-col resolved --target-col synthesizability \\
        --output data/sets/routes_train_val_deduped.csv

Arguments:
    input_csv       Input CSV to deduplicate.
    --smiles-col    Column identifying duplicate molecules (default: "smiles").
    --resolved-col  Boolean column, True where a route was found
                    (default: "resolved").
    --target-col    Numeric column used to break ties among same-resolved
                    duplicates; higher is kept (default: "synthesizability").
    --output        Output CSV path (default: "<input_csv stem>_deduped.csv"
                    next to input_csv).
"""

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_csv", type=str, help="Input CSV to deduplicate.")
    p.add_argument("--smiles-col", type=str, default="smiles", help="Column identifying duplicate molecules.")
    p.add_argument(
        "--resolved-col", type=str, default="resolved", help="Boolean column, True where a route was found."
    )
    p.add_argument(
        "--target-col",
        type=str,
        default="synthesizability",
        help="Numeric column used to break ties among same-resolved duplicates; higher is kept.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: '<input_csv stem>_deduped.csv' next to input_csv).",
    )
    return p.parse_args()


def dedupe_by_smiles(
    df: pd.DataFrame, smiles_col: str, resolved_col: str, target_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse duplicate SMILES to one row each: resolved route first, then highest score.

    Args:
        df: Input DataFrame.
        smiles_col: Column identifying duplicate molecules.
        resolved_col: Boolean column, True where a route was found. Compared as a
            lowercased string rather than relied on for a bool dtype, since a CSV
            round-trip can leave it as "True"/"False" text instead.
        target_col: Numeric column to break ties among same-resolved duplicates
            (higher is kept); NaN is treated as worse than any real value.

    Returns:
        ``(deduped_df, dropped_df)``: the deduplicated DataFrame (original row
        order preserved) and the rows dropped as duplicates, for reporting.
    """
    resolved_priority = df[resolved_col].astype(str).str.strip().str.lower().eq("true")
    ranked = df.assign(_resolved_priority=resolved_priority).sort_values(
        by=[smiles_col, "_resolved_priority", target_col],
        ascending=[True, False, False],
        na_position="last",
        kind="stable",
    )
    keep_index = ranked.index[~ranked.duplicated(subset=[smiles_col], keep="first")]
    deduped = df.loc[keep_index].sort_index()
    dropped = df.drop(index=keep_index)
    return deduped, dropped


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    for col in (args.smiles_col, args.resolved_col, args.target_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {args.input_csv}. Available: {df.columns.tolist()}")

    n_unique = df[args.smiles_col].nunique()
    print(f"Loaded {len(df)} rows from {args.input_csv} ({n_unique} unique '{args.smiles_col}' values).")

    deduped, dropped = dedupe_by_smiles(df, args.smiles_col, args.resolved_col, args.target_col)

    n_dropped_resolved = dropped[args.resolved_col].astype(str).str.strip().str.lower().eq("true").sum()
    print(f"Removed {len(dropped)} duplicate rows -> {len(deduped)} rows remain.")
    if len(dropped):
        print(
            f"  Of those, {n_dropped_resolved} had a resolved route themselves but lost to a same-molecule "
            f"duplicate with an equal-or-higher '{args.target_col}' (only the best kept)."
        )

    output_path = (
        Path(args.output) if args.output else Path(args.input_csv).with_name(f"{Path(args.input_csv).stem}_deduped.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
