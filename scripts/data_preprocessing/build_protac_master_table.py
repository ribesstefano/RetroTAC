"""
Stage 1, step 2 — build the standardized PROTAC master table.

Combines the held-out curated set with PROTAC-Splitter output (step 1),
standardizes SMILES, extracts components, and assigns two kinds of stable IDs:
  - sequential (WH_001, LK_001, E3_001, PROTAC_001) — order-dependent but human-readable
  - hash-based  (WH_a1b2c3d4, ...)                  — order-independent, stable across runs

Output columns:
    protac_seq_id, protac_id, datasource,
    protac_smiles, protac_smiles_index, protac_canon_smiles, protac_smiles_std,
    warhead_smiles,          warhead_seq_id,          warhead_id,
    linker_smiles,           linker_seq_id,           linker_id,
    e3_ligase_ligand_smiles, e3_ligase_ligand_seq_id, e3_ligase_ligand_id
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from chem_utils import (  # noqa: E402
    canon_smiles,
    make_hash_ids,
    make_sequential_ids,
    papply,
    smiles_hash,
    std_smiles,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def extract_components(elem: Any) -> pd.Series:
    """Split dot-separated PROTAC-Splitter output into (warhead, linker, e3_ligase).

    Attachment-point tags determine the role of each fragment:
      ``[*:1]`` only        → warhead
      ``[*:2]`` only        → E3-ligase ligand
      ``[*:1]`` and ``[*:2]``   → linker

    Args:
        elem: Dot-separated SMILES string from PROTAC-Splitter, or NaN.

    Returns:
        pd.Series of ``[warhead_smiles, linker_smiles, e3_ligase_ligand_smiles]``.
    """
    warhead = linker = e3_ligase = None
    if not pd.isna(elem):
        for frag in str(elem).split("."):
            has1, has2 = "[*:1]" in frag, "[*:2]" in frag
            if   has1 and not has2: warhead  = frag
            elif has2 and not has1: e3_ligase = frag
            elif has1 and has2:     linker   = frag
    return pd.Series([warhead, linker, e3_ligase])


def load_held_out(path: Path) -> pd.DataFrame:
    """Load and standardize the curated held-out PROTAC set with pre-split components.

    Args:
        path: Path to the held-out CSV (must contain ``PROTAC SMILES``,
            ``Warhead SMILES``, ``Linker SMILES``, ``E3 Ligase Ligand SMILES``).

    Returns:
        DataFrame with columns ``protac_smiles``, ``warhead_smiles``,
        ``linker_smiles``, ``e3_ligase_ligand_smiles``, ``datasource``.
    """
    df = (
        pd.read_csv(path, usecols=[
            "PROTAC SMILES", "Warhead SMILES",
            "Linker SMILES", "E3 Ligase Ligand SMILES",
        ])
        .rename(columns={
            "PROTAC SMILES":           "protac_smiles",
            "Warhead SMILES":          "warhead_smiles",
            "Linker SMILES":           "linker_smiles",
            "E3 Ligase Ligand SMILES": "e3_ligase_ligand_smiles",
        })
    )
    df["protac_smiles"] = papply(
        papply(df["protac_smiles"], canon_smiles, "held-out: canon protac"),
        std_smiles, "held-out: std protac",
    )
    df = df.drop_duplicates(subset="protac_smiles").dropna(subset="protac_smiles")

    df["warhead_smiles"]          = papply(
        papply(df["warhead_smiles"], canon_smiles, "held-out: canon warhead"),
        std_smiles, "held-out: std warhead",
    )
    df["linker_smiles"]           = papply(
        papply(df["linker_smiles"], canon_smiles, "held-out: canon linker"),
        std_smiles, "held-out: std linker",
    )
    df["e3_ligase_ligand_smiles"] = papply(
        papply(df["e3_ligase_ligand_smiles"], canon_smiles, "held-out: canon e3"),
        std_smiles, "held-out: std e3",
    )

    df["datasource"] = "held-out"
    return df


def load_tack(path: Path, exclude_smiles: pd.Series) -> pd.DataFrame:
    """Load TACK PROTAC-Splitter output, extract components, and drop held-out overlaps.

    Args:
        path: Path to the TACK split CSV (must contain ``SMILES`` and
            ``default_pred_n0`` columns).
        exclude_smiles: Series of already-seen standardized SMILES to filter out.

    Returns:
        DataFrame with the same columns as ``load_held_out``, datasource ``"tack"``.
    """
    raw = pd.read_csv(path)
    if "default_pred_n0" not in raw.columns:
        raise ValueError("'default_pred_n0' not found — check step 1 output.")
    raw[["warhead_smiles", "linker_smiles", "e3_ligase_ligand_smiles"]] = (
        papply(raw["default_pred_n0"], extract_components, "tack: extract components")
    )
    df = (
        raw[["SMILES", "warhead_smiles", "linker_smiles", "e3_ligase_ligand_smiles"]]
        .rename(columns={"SMILES": "protac_smiles"})
    )
    df["protac_smiles"] = papply(
        papply(df["protac_smiles"], canon_smiles, "tack: canon protac"),
        std_smiles, "tack: std protac",
    )
    df = df.drop_duplicates(subset="protac_smiles").dropna(subset="protac_smiles")

    df["warhead_smiles"]          = papply(
        papply(df["warhead_smiles"], canon_smiles, "tack: canon warhead"),
        std_smiles, "tack: std warhead",
    )
    df["linker_smiles"]           = papply(
        papply(df["linker_smiles"], canon_smiles, "tack: canon linker"),
        std_smiles, "tack: std linker",
    )
    df["e3_ligase_ligand_smiles"] = papply(
        papply(df["e3_ligase_ligand_smiles"], canon_smiles, "tack: canon e3"),
        std_smiles, "tack: std e3",
    )

    df = df[~df["protac_smiles"].isin(exclude_smiles)].dropna(subset="protac_smiles").copy()
    df["datasource"] = "tack"
    return df


def main(held_path: Path, tack_path: Path, output_path: Path) -> None:
    """Build and write the merged, standardized PROTAC master table.

    Args:
        held_path: Path to the curated held-out CSV.
        tack_path: Path to the TACK PROTAC-Splitter output CSV.
        output_path: Destination CSV path.
    """
    COLS = [
        "protac_smiles", "warhead_smiles", "linker_smiles",
        "e3_ligase_ligand_smiles", "datasource",
    ]
    OUT_COLS = [
        "protac_seq_id", "protac_id", "datasource",
        "protac_smiles", "protac_smiles_index", "protac_canon_smiles", "protac_smiles_std",
        "warhead_smiles",          "warhead_seq_id",          "warhead_id",
        "linker_smiles",           "linker_seq_id",           "linker_id",
        "e3_ligase_ligand_smiles", "e3_ligase_ligand_seq_id", "e3_ligase_ligand_id",
    ]
    held = load_held_out(held_path)
    print(f"Held-out rows                       : {len(held)}")

    tack = load_tack(tack_path, exclude_smiles=held["protac_smiles"])
    print(f"TACK rows (after overlap removal)  : {len(tack)}")

    df = pd.concat([held[COLS], tack[COLS]], ignore_index=True)

    df["protac_canon_smiles"] = papply(df["protac_smiles"], canon_smiles, "canon protac (merged)")
    df["protac_smiles_std"]   = papply(df["protac_smiles"], std_smiles,   "std protac (merged)")
    df["protac_smiles_index"] = df["protac_smiles"].duplicated(keep=False).astype(int)

    df["warhead_seq_id"]          = make_sequential_ids(df["warhead_smiles"],          "WH")
    df["linker_seq_id"]           = make_sequential_ids(df["linker_smiles"],           "LK")
    df["e3_ligase_ligand_seq_id"] = make_sequential_ids(df["e3_ligase_ligand_smiles"], "E3")

    df["warhead_id"]          = make_hash_ids(df["warhead_smiles"],          "WH")
    df["linker_id"]           = make_hash_ids(df["linker_smiles"],           "LK")
    df["e3_ligase_ligand_id"] = make_hash_ids(df["e3_ligase_ligand_smiles"], "E3")

    df["protac_seq_id"] = pd.factorize(df["protac_smiles"])[0] + 1
    df["protac_id"] = df["protac_smiles"].map(
        lambda s: f"PROTAC_{smiles_hash(s)}" if pd.notna(s) else None
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[OUT_COLS].to_csv(output_path, index=False)

    print(f"\nSaved → {output_path}  ({len(df)} rows)")
    print(f"  invalid protac_canon_smiles : {df['protac_canon_smiles'].isna().sum()}")
    print(f"  invalid protac_smiles_std   : {df['protac_smiles_std'].isna().sum()}")
    print(f"  unique raw SMILES           : {df['protac_smiles'].nunique()}")
    print(f"  unique std SMILES           : {df['protac_smiles_std'].nunique(dropna=True)}")
    print()
    print(df["datasource"].value_counts().to_string())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Populated ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(description="Build standardized PROTAC master table.")
    p.add_argument("--held-out", type=Path,
                   default=_PROJECT_ROOT / "data/raw/dataset-curated-held-out.csv",
                   help="Curated held-out CSV with pre-split components.")
    p.add_argument("--tack", type=Path,
                   default=_PROJECT_ROOT / "data/raw/tpddb_split_raw_complete.csv",
                   help="TACK dataset split via PROTAC-Splitter.")
    p.add_argument("--output", type=Path,
                   default=_PROJECT_ROOT / "data/processed/protac_smiles_master_std.csv",
                   help="Destination CSV.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.held_out, args.tack, args.output)
