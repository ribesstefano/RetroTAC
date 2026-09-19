"""
add_component_splits.py
─────────────────────────
Adds PROTAC component SMILES (warhead, linker, E3 ligand) to a target CSV by
joining in the matching row from a PROTAC-Splitter output CSV (the "split
CSV"), matched on a whole-molecule SMILES column -- the column name can
differ between the two files.

Each matched row's `default_pred_n0` prediction (a dot-separated SMILES
string, one fragment per component, each tagged with RDKit dummy-atom labels
[*:1] = warhead attachment, [*:2] = E3 attachment) is split on "." and each
fragment classified by which labels it contains:
    e3_smiles     - fragment containing [*:2] only
    linker_smiles - fragment containing both [*:1] and [*:2]
    wh_smiles     - fragment containing [*:1] only

These three columns are appended to the target CSV and written to the
output CSV. A target row whose SMILES has no match in the split CSV, or
whose `default_pred_n0` doesn't decompose into exactly one fragment of each
type, gets NaN in all three new columns (not dropped) and is counted in a
warning printed at the end.

Usage
-----
    python scripts/dataset/add_component_splits.py \\
        data/routes/routes_scored.csv data/tack/tack_smiles_split.csv \\
        --target-smiles-col smiles --split-smiles-col SMILES \\
        --output data/routes/routes_scored_with_components.csv

Arguments:
    target_csv           Target CSV to add e3_smiles/linker_smiles/wh_smiles to.
    split_csv             PROTAC-Splitter output CSV holding `--pred-col`.
    --target-smiles-col   Whole-molecule SMILES column name in target_csv
                          (default: "smiles").
    --split-smiles-col    Whole-molecule SMILES column name in split_csv
                          (default: "SMILES").
    --pred-col            Dot-separated, dummy-atom-tagged fragment column in
                          split_csv (default: "default_pred_n0").
    --output              Output CSV path (default: "<target_csv stem>_with_components.csv"
                          next to target_csv).
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target_csv", type=str, help="Target CSV to add component-SMILES columns to.")
    p.add_argument(
        "split_csv", type=str, help="PROTAC-Splitter output CSV holding the dot-split prediction column."
    )
    p.add_argument(
        "--target-smiles-col", type=str, default="smiles", help="Whole-molecule SMILES column name in target_csv."
    )
    p.add_argument(
        "--split-smiles-col", type=str, default="SMILES", help="Whole-molecule SMILES column name in split_csv."
    )
    p.add_argument(
        "--pred-col",
        type=str,
        default="default_pred_n0",
        help="Dot-separated, dummy-atom-tagged fragment column in split_csv.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: '<target_csv stem>_with_components.csv' next to target_csv).",
    )
    return p.parse_args()


def classify_fragments(pred: str) -> Optional[Tuple[str, str, str]]:
    """Dot-split a `default_pred_n0`-style prediction into its three tagged fragments.

    Args:
        pred: Dot-separated PROTAC-Splitter prediction string, each fragment
            tagged with RDKit dummy-atom labels [*:1] (warhead attachment)
            and/or [*:2] (E3 attachment).

    Returns:
        (wh_smiles, linker_smiles, e3_smiles), or None if the fragments don't
        decompose into exactly one warhead ([*:1] only), one linker (both
        [*:1] and [*:2]), and one E3 fragment ([*:2] only).
    """
    wh = linker = e3 = None
    for frag in pred.split("."):
        has1, has2 = "[*:1]" in frag, "[*:2]" in frag
        if has1 and has2:
            if linker is not None:
                return None
            linker = frag
        elif has2:
            if e3 is not None:
                return None
            e3 = frag
        elif has1:
            if wh is not None:
                return None
            wh = frag
    if wh is None or linker is None or e3 is None:
        return None
    return wh, linker, e3


def add_component_splits(
    target_df: pd.DataFrame,
    split_df: pd.DataFrame,
    target_smiles_col: str,
    split_smiles_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """Append e3_smiles/linker_smiles/wh_smiles columns to a copy of target_df.

    Args:
        target_df: Target DataFrame to extend.
        split_df: PROTAC-Splitter output DataFrame holding `pred_col`.
        target_smiles_col: Whole-molecule SMILES column name in target_df.
        split_smiles_col: Whole-molecule SMILES column name in split_df.
        pred_col: Dot-separated fragment-prediction column name in split_df.

    Returns:
        Copy of target_df with e3_smiles, linker_smiles, wh_smiles columns
        appended (NaN where the SMILES has no match in split_df, or the
        match's `pred_col` doesn't decompose cleanly -- see
        `classify_fragments`).
    """
    dup = split_df[split_smiles_col].duplicated()
    if dup.any():
        print(f"Warning: {dup.sum()} duplicate '{split_smiles_col}' values in split_csv; keeping the first of each.")
        split_df = split_df[~dup]

    fragments: Dict[str, Optional[Tuple[str, str, str]]] = {
        smi: classify_fragments(pred) for smi, pred in zip(split_df[split_smiles_col], split_df[pred_col])
    }

    wh_col, linker_col, e3_col = [], [], []
    n_unmatched, n_bad_split = 0, 0
    for smi in target_df[target_smiles_col]:
        if smi not in fragments:
            n_unmatched += 1
            parsed = None
        else:
            parsed = fragments[smi]
            if parsed is None:
                n_bad_split += 1
        if parsed is None:
            wh_col.append(None)
            linker_col.append(None)
            e3_col.append(None)
        else:
            wh, linker, e3 = parsed
            wh_col.append(wh)
            linker_col.append(linker)
            e3_col.append(e3)

    if n_unmatched:
        print(f"Warning: {n_unmatched}/{len(target_df)} target rows had no matching SMILES in split_csv.")
    if n_bad_split:
        print(
            f"Warning: {n_bad_split}/{len(target_df)} matched rows had a '{pred_col}' that didn't decompose "
            "into exactly one warhead/linker/E3 fragment."
        )

    out_df = target_df.copy()
    out_df["e3_smiles"] = e3_col
    out_df["linker_smiles"] = linker_col
    out_df["wh_smiles"] = wh_col
    return out_df


def main() -> None:
    args = parse_args()
    target_df = pd.read_csv(args.target_csv)
    split_df = pd.read_csv(args.split_csv)

    for col, df, path in (
        (args.target_smiles_col, target_df, args.target_csv),
        (args.split_smiles_col, split_df, args.split_csv),
    ):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {path}. Available: {df.columns.tolist()}")
    if args.pred_col not in split_df.columns:
        raise ValueError(f"Column '{args.pred_col}' not found in {args.split_csv}. Available: {split_df.columns.tolist()}")

    print(f"Loaded {len(target_df)} rows from {args.target_csv}, {len(split_df)} rows from {args.split_csv}")

    out_df = add_component_splits(
        target_df, split_df, args.target_smiles_col, args.split_smiles_col, args.pred_col
    )

    output_path = (
        Path(args.output)
        if args.output
        else Path(args.target_csv).with_name(f"{Path(args.target_csv).stem}_with_components.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
