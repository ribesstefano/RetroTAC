"""
plot_dataset.py
────────────────
Plots a diverse subset of molecules from a SMILES CSV, selected via one of RDKit's
SimDivFilters diversity pickers
(https://www.rdkit.org/docs/source/rdkit.SimDivFilters.rdSimDivPickers.html):

    maxmin       - MaxMinPicker (default): greedy max-min diversity picking over Morgan
                   fingerprints via Tanimoto distance.
    leader       - LeaderPicker: picks molecules whose Tanimoto distance to every
                   already-picked molecule exceeds a fixed threshold (LEADER_THRESHOLD
                   below); can return fewer than --n-mols if too few molecules clear it.
    hierarchical - HierarchicalClusterPicker (Ward linkage, HIERARCHICAL_CLUSTER_METHOD
                   below): hierarchically clusters all molecules by pairwise Tanimoto
                   distance, then picks a diverse subset from the resulting tree.

Renders the picked molecules as a single grid image (rdkit.Chem.Draw.MolsToGridImage)
to --output. If --target-col is given, each molecule's legend also shows
"<target-col>: <value>".

Usage
-----
    python plot_dataset.py data/routes/routes_scored.csv \
        --smiles-col smiles --target-col synthesizability \
        --method maxmin --n-mols 20 --output figures/routes_scored_maxmin.png

Arguments:
    input_csv       Input CSV path.
    --smiles-col    SMILES column name (default: "smiles").
    --target-col    Optional target column; if given, shown as "<target-col>: <value>"
                    below each molecule.
    --method        Diversity-picker method: "maxmin", "leader", or "hierarchical"
                    (default: "maxmin").
    --n-mols        Number of molecules to pick and plot (default: 20).
    --fp-radius     Morgan fingerprint radius (default: 8).
    --fp-bits       Morgan fingerprint bit-vector length (default: 1024).
    --output        Output image path (default: "figures/<input_csv stem>_<method>.png").
    --seed          Random seed for MaxMinPicker's first pick; ignored by leader/
                     hierarchical, neither of which takes a seed (default: 42).
"""

import argparse
from array import array
from functools import partial
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger
from rdkit.Chem import Draw
from rdkit.SimDivFilters import rdSimDivPickers
from tqdm import tqdm

from retrotac.chem_utils import morgan_fp, papply, smiles_to_mol

RDLogger.DisableLog("rdApp.*")

# Tanimoto-distance threshold below which LeaderPicker refuses to pick a molecule as a
# new "leader" (i.e. it's too similar to an already-picked one). Same order of
# magnitude as the Butina cutoff used elsewhere in the repo (isolate_heldout.py).
LEADER_THRESHOLD = 0.65
HIERARCHICAL_CLUSTER_METHOD = rdSimDivPickers.ClusterMethod.WARD


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_csv", type=str, help="Input CSV path.")
    p.add_argument("--smiles-col", type=str, default="smiles", help="SMILES column name.")
    p.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Optional target column; if given, shown as '<target-col>: <value>' below each molecule.",
    )
    p.add_argument(
        "--method",
        type=str,
        choices=["maxmin", "leader", "hierarchical"],
        default="maxmin",
        help="Diversity-picker method (default: maxmin).",
    )
    p.add_argument("--n-mols", type=int, default=20, help="Number of molecules to pick and plot (default: 20).")
    p.add_argument("--fp-radius", type=int, default=8, help="Morgan fingerprint radius (default: 8).")
    p.add_argument("--fp-bits", type=int, default=1024, help="Morgan fingerprint bit-vector length (default: 1024).")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: 'figures/<input_csv stem>_<method>.png').",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for MaxMinPicker's first pick (default: 42)."
    )
    return p.parse_args()


def _pairwise_tanimoto_condensed(fps: List) -> array:
    """Compute the condensed (lower-triangle) pairwise Tanimoto-distance list.

    This is the flat 1D format ``HierarchicalClusterPicker.Pick`` expects as ``distMat``.

    Args:
        fps: RDKit ``ExplicitBitVect`` Morgan fingerprints, one per row.

    Returns:
        ``array('f')`` of length n*(n-1)/2 holding 1 - Tanimoto similarity for each
        pair, in row-major lower-triangle order.
    """
    n = len(fps)
    dists = array("f")
    for i in tqdm(range(1, n), desc="Pairwise Tanimoto distances"):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    return dists


def pick_indices_maxmin(fps: List, pick_size: int, seed: int) -> List[int]:
    """Pick a diverse subset via MaxMinPicker.

    Args:
        fps: RDKit ``ExplicitBitVect`` Morgan fingerprints, one per row.
        pick_size: Number of molecules to pick.
        seed: Random seed for the picker's first choice.

    Returns:
        Row indices (into *fps*) of the picked molecules.
    """
    n = len(fps)
    picker = rdSimDivPickers.MaxMinPicker()
    picked = picker.LazyBitVectorPick(fps, n, pick_size, seed=seed)
    return list(picked)


def pick_indices_leader(fps: List, pick_size: int) -> List[int]:
    """Pick a diverse subset via LeaderPicker.

    Molecules are picked greedily as new "leaders" as long as their Tanimoto distance
    to every already-picked leader exceeds ``LEADER_THRESHOLD``, so the number
    actually picked can be smaller than *pick_size* if too few molecules clear it.

    Args:
        fps: RDKit ``ExplicitBitVect`` Morgan fingerprints, one per row.
        pick_size: Max number of molecules to pick.

    Returns:
        Row indices (into *fps*) of the picked molecules; length <= pick_size.
    """
    n = len(fps)
    picker = rdSimDivPickers.LeaderPicker()
    picked = picker.LazyBitVectorPick(fps, n, LEADER_THRESHOLD, pickSize=pick_size)
    return list(picked)


def pick_indices_hierarchical(fps: List, pick_size: int) -> List[int]:
    """Pick a diverse subset via HierarchicalClusterPicker (Ward linkage).

    Args:
        fps: RDKit ``ExplicitBitVect`` Morgan fingerprints, one per row.
        pick_size: Number of molecules to pick.

    Returns:
        Row indices (into *fps*) of the picked molecules.
    """
    n = len(fps)
    # Pick() rejects the array.array from _pairwise_tanimoto_condensed outright
    # ("distance mat argument must be a numpy matrix"), unlike Butina.ClusterData.
    dists = np.asarray(_pairwise_tanimoto_condensed(fps), dtype=np.double)
    picker = rdSimDivPickers.HierarchicalClusterPicker(HIERARCHICAL_CLUSTER_METHOD)
    picked = picker.Pick(dists, n, pick_size)
    return list(picked)


def _format_legend_value(v: Any) -> str:
    """Format a single target-column value for a grid-image legend.

    Args:
        v: Raw value from the target column.

    Returns:
        ``"NA"`` for missing values, a 3-decimal float string when *v* is numeric
        (or numeric-looking), otherwise ``str(v)``.
    """
    if pd.isna(v):
        return "NA"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def build_legends(df: pd.DataFrame, target_col: Optional[str]) -> Optional[List[str]]:
    """Build per-molecule grid-image legends from the target column, if any.

    Args:
        df: DataFrame of the picked rows, row-aligned with the mols being plotted.
        target_col: Target column name, or None to skip legends entirely.

    Returns:
        ``["<target_col>: <value>", ...]`` aligned with *df*, or None if *target_col*
        is None.
    """
    if target_col is None:
        return None
    return [f"{target_col}: {_format_legend_value(v)}" for v in df[target_col]]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    out_path = Path(args.output) if args.output else Path("figures") / f"{input_path.stem}_{args.method}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    # Fail fast on a typo'd column name, not minutes into fingerprinting.
    cols_to_check = [args.smiles_col] + ([args.target_col] if args.target_col else [])
    for col in cols_to_check:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {args.input_csv}. Available: {df.columns.tolist()}")

    print(f"Loaded {len(df)} rows from {args.input_csv}")
    df["_mol"] = papply(df[args.smiles_col], smiles_to_mol, desc="Parsing SMILES")
    n_invalid = df["_mol"].isna().sum()
    if n_invalid:
        print(f"Dropping {n_invalid} rows with unparseable SMILES")
        df = df[df["_mol"].notna()].reset_index(drop=True)

    fp_fn = partial(morgan_fp, radius=args.fp_radius, nbits=args.fp_bits)
    df["_fp"] = papply(
        df["_mol"], fp_fn, desc=f"Computing Morgan fingerprints (radius={args.fp_radius}, nbits={args.fp_bits})"
    )

    fps = df["_fp"].tolist()
    n = len(fps)
    pick_size = min(args.n_mols, n)
    if pick_size < 1:
        raise ValueError(f"No molecules available to plot (n={n}).")
    if pick_size < args.n_mols:
        print(f"Requested --n-mols={args.n_mols} exceeds available {n} molecules; plotting all {pick_size}.")

    print(f"Picking {pick_size} molecules via {args.method} …")
    if args.method == "maxmin":
        picked = pick_indices_maxmin(fps, pick_size, args.seed)
    elif args.method == "leader":
        picked = pick_indices_leader(fps, pick_size)
    else:
        picked = pick_indices_hierarchical(fps, pick_size)
    print(f"Picked {len(picked)} molecules (requested {pick_size}).")

    picked_df = df.iloc[list(picked)]
    legends = build_legends(picked_df, args.target_col)

    img = Draw.MolsToGridImage(picked_df["_mol"].tolist(), molsPerRow=5, subImgSize=(250, 250), legends=legends)
    img.save(out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
