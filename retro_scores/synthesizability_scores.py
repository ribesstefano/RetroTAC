"""
synthesizability_scores.py
==========================
CLI entry point for computing synthesizability scores over a CSV of SMILES.

Thin wrapper around mol_scores.compute_scores — all the scoring logic
lives in the library; this just handles I/O and arguments. Requires the
standalone `mol_scores` package (part of retro_scores/, see its
README.md), installed separately from the main retrotac environment.

Example
-------
    python retro_scores/synthesizability_scores.py \
        data/retrotac_data.csv \
        data/retrotac_scores.csv \
        --smiles-col molecule

Typically invoked from slurm/submit_protac_scores.sh as a batch job.
"""

import argparse
import sys

import pandas as pd

from mol_scores import compute_scores, SCORE_COLUMNS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute synthesizability scores for a CSV of SMILES."
    )
    parser.add_argument("input_csv", help="Input CSV containing a SMILES column")
    parser.add_argument("output_csv", help="Path to write the scored CSV")
    parser.add_argument("--smiles-col", default="molecule",
                        help="Name of the SMILES column (default: molecule)")
    parser.add_argument("--keep-all-columns", action="store_true",
                        help="Keep the original input columns in the output too")
    parser.add_argument("--device", default="auto",
                        help="Compute device for FSscore's Lightning trainer (default: auto, which "
                             "grabs a GPU if one is visible -- crashes on a login node that has a "
                             "GPU it cannot use; pass cpu there). Only FSscore uses this.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Molecules per forward pass for FSscore (default: 128). Only FSscore "
                             "batches; every other score is unaffected.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Dataloader worker processes for FSscore (default: 4).")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    if args.smiles_col not in df.columns:
        sys.exit(f"ERROR: column '{args.smiles_col}' not found in {args.input_csv}. "
                 f"Available columns: {list(df.columns)}")

    df = compute_scores(
        df, smiles_col=args.smiles_col,
        device=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    if args.keep_all_columns:
        out_cols = list(df.columns)
    else:
        out_cols = [args.smiles_col] + SCORE_COLUMNS

    df[out_cols].to_csv(args.output_csv, index=False)
    print(f"Saved scores to {args.output_csv}")


if __name__ == "__main__":
    main()