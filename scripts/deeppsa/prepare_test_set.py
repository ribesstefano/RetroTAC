"""prepare_test_set.py
=====================
Builds a held-out test set shared between RetroTAC and DeepPSA
(Zhang-Ran-0119/DeepPSA, see apptainer/deeppsa.def) for a fair head-to-head
comparison: any molecule DeepPSA saw during its own training would give it an
unfair advantage, so this script standardizes SMILES on both sides and drops
every RetroTAC held-out molecule that also appears in DeepPSA's training set.
Duplicate SMILES within the held-out set itself are also collapsed to one row
each, so every molecule in the output is scored exactly once.

Needs rdkit/pandas -- run inside apptainer/training.sif (see CLAUDE.md
"Containers"):
    apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \\
        python scripts/deeppsa/prepare_test_set.py

Usage
-----
    python scripts/deeppsa/prepare_test_set.py \\
        --held-out data/sets/routes_test.csv \\
        --deeppsa-train data/deeppsa/deeppsa_train.csv \\
        --output-dir data/deeppsa/shared_test_set
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from retrotac.chem_utils import std_smiles

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_STD_COL = "_std_smiles"


def prepare_test_set(held_out_path: Path, deeppsa_train_path: Path, output_dir: Path,
                     smiles_col: str = "smiles") -> None:
    """Build the RetroTAC/DeepPSA shared test set and write it + stats to disk.

    Standardizes `smiles_col` on both inputs via `retrotac.chem_utils.std_smiles`,
    drops held-out rows that fail standardization (their overlap with DeepPSA's
    training set can't be verified), collapses duplicate standardized SMILES
    within the held-out set to one row each, then drops any row whose
    standardized SMILES also appears in DeepPSA's training set.

    Args:
        held_out_path: Our held-out test CSV (must have `smiles_col`).
        deeppsa_train_path: DeepPSA's training CSV (must have `smiles_col`).
        output_dir: Directory to write `shared_test_set.csv`,
            `excluded_overlap.csv`, and `stats.json` into (created if missing).
        smiles_col: SMILES column name, shared by both inputs.
    """
    held_out = pd.read_csv(held_out_path)
    deeppsa_train = pd.read_csv(deeppsa_train_path)

    n_held_out_total = len(held_out)
    n_deeppsa_train_total = len(deeppsa_train)

    held_out[_STD_COL] = held_out[smiles_col].apply(std_smiles)
    deeppsa_train[_STD_COL] = deeppsa_train[smiles_col].apply(std_smiles)

    failed_std = held_out[_STD_COL].isna()
    n_failed_standardization = int(failed_std.sum())
    held_out = held_out[~failed_std].copy()

    n_deeppsa_train_failed = int(deeppsa_train[_STD_COL].isna().sum())
    deeppsa_smiles_set = set(deeppsa_train[_STD_COL].dropna())
    n_deeppsa_train_unique = len(deeppsa_smiles_set)

    is_dup = held_out[_STD_COL].duplicated(keep="first")
    n_duplicates_removed = int(is_dup.sum())
    held_out = held_out[~is_dup].copy()
    n_held_out_after_dedup = len(held_out)

    is_overlap = held_out[_STD_COL].isin(deeppsa_smiles_set)
    n_overlap_removed = int(is_overlap.sum())
    excluded_overlap = held_out.loc[is_overlap].drop(columns=_STD_COL)
    shared_test_set = held_out.loc[~is_overlap].drop(columns=_STD_COL)
    # DeepPSA's data loader requires a `labels` column even for pure inference;
    # -1 marks it as a dummy (no real label, not one of DeepPSA's 0/1 classes).
    shared_test_set["labels"] = -1

    output_dir.mkdir(parents=True, exist_ok=True)
    shared_path = output_dir / "shared_test_set.csv"
    excluded_path = output_dir / "excluded_overlap.csv"
    stats_path = output_dir / "stats.json"

    shared_test_set.to_csv(shared_path, index=False)
    excluded_overlap.to_csv(excluded_path, index=False)

    overlap_fraction = n_overlap_removed / n_held_out_after_dedup if n_held_out_after_dedup else 0.0
    stats = {
        "held_out_input": str(held_out_path),
        "deeppsa_train_input": str(deeppsa_train_path),
        "n_held_out_total": n_held_out_total,
        "n_held_out_failed_standardization": n_failed_standardization,
        "n_held_out_duplicates_removed": n_duplicates_removed,
        "n_held_out_after_dedup": n_held_out_after_dedup,
        "n_deeppsa_train_total": n_deeppsa_train_total,
        "n_deeppsa_train_failed_standardization": n_deeppsa_train_failed,
        "n_deeppsa_train_unique_standardized": n_deeppsa_train_unique,
        "n_removed_overlap_with_deeppsa_train": n_overlap_removed,
        "overlap_fraction_of_deduped_held_out": overlap_fraction,
        "n_shared_test_set": len(shared_test_set),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("held-out set: %d rows (%s)", n_held_out_total, held_out_path)
    logger.info("  -> %d dropped: failed SMILES standardization", n_failed_standardization)
    logger.info("  -> %d dropped: duplicate SMILES within held-out set", n_duplicates_removed)
    logger.info("  -> %d rows remain after dedup", n_held_out_after_dedup)
    logger.info("DeepPSA train set: %d rows, %d unique standardized SMILES (%s)",
                n_deeppsa_train_total, n_deeppsa_train_unique, deeppsa_train_path)
    logger.info("  -> %d dropped: seen in DeepPSA training data (%.1f%% overlap)",
                n_overlap_removed, overlap_fraction * 100)
    logger.info("shared test set: %d rows -> %s", len(shared_test_set), shared_path)
    logger.info("excluded-overlap rows written to %s", excluded_path)
    logger.info("stats written to %s", stats_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the shared-test-set CLI.

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--held-out", type=Path, default=_PROJECT_ROOT / "data" / "sets" / "routes_test.csv",
                    help="our held-out test CSV (default: %(default)s)")
    ap.add_argument("--deeppsa-train", type=Path,
                    default=_PROJECT_ROOT / "data" / "deeppsa" / "deeppsa_train.csv",
                    help="DeepPSA's training CSV (default: %(default)s)")
    ap.add_argument("--output-dir", type=Path,
                    default=_PROJECT_ROOT / "data" / "deeppsa" / "shared_test_set",
                    help="output directory for the shared test set + stats (default: %(default)s)")
    ap.add_argument("--smiles-col", default="smiles",
                    help="SMILES column name, shared by both inputs (default: %(default)s)")
    return ap.parse_args()


def main() -> None:
    """CLI entry point: parse arguments and build the shared test set."""
    args = parse_args()
    prepare_test_set(args.held_out, args.deeppsa_train, args.output_dir, smiles_col=args.smiles_col)


if __name__ == "__main__":
    main()
