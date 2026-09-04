"""Score a CSV of SMILES with the pretrained RetroTAC Caruana ensemble.

Loads retrotac.models.ensemble.RetroTAC via RetroTAC.from_pretrained from the
ailab-bio/RetroTAC Hugging Face Hub repo, scores the input CSV's SMILES
column in batches, and writes the input CSV back out with an appended
"prediction" column.

Usage
-----
    python -m retrotac.cli --input data/molecules.csv --output data/molecules_scored.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from retrotac.models.ensemble import RetroTAC

HF_REPO = "ailab-bio/RetroTAC"


def score_csv(
    in_path: Path,
    out_path: Path,
    smiles_col: str = "SMILES",
    device: str = "cpu",
    batch_size: int = 64,
    load_strategy: str = "eager",
    n_jobs: int = 1,
    verbose: bool = False,
) -> None:
    """Load the RetroTAC ensemble and append its predictions to a CSV.

    Args:
        in_path: Input CSV containing at least `smiles_col`.
        out_path: Output CSV path (input columns + appended "prediction" column).
        smiles_col: Name of the SMILES column in `in_path`.
        device: Compute device forwarded to the ensemble's gnn members.
        batch_size: Number of SMILES scored per RetroTAC.predict() call.
        load_strategy: Forwarded to RetroTAC.from_pretrained -- "eager", "lazy",
            or "stream" (see retrotac.models.ensemble.RetroTAC).
        n_jobs: Forwarded to RetroTAC.predict() for parallelizing its internal
            fingerprint/descriptor computation; see papply's docstring
            (1 = sequential, -1 = one worker per CPU).
        verbose: If True, enable INFO-level logging (e.g. from_pretrained's
            local-dir-vs-Hub resolution messages).
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    ensemble = RetroTAC.from_pretrained(HF_REPO, device=device, load_strategy=load_strategy)

    df = pd.read_csv(in_path)
    smiles_list = df[smiles_col].tolist()
    batches = [smiles_list[i:i + batch_size] for i in range(0, len(smiles_list), batch_size)]

    preds = [
        ensemble.predict(batch, n_jobs=n_jobs)
        for batch in tqdm(batches, desc="Scoring batches", unit="batch")
    ]
    df["prediction"] = np.concatenate(preds) if preds else np.array([], dtype=float)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"scored {len(df)} molecules -> {out_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI args for scoring a CSV of SMILES with the RetroTAC ensemble.

    Returns:
        argparse.Namespace with the parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True, type=Path, help="CSV with a SMILES column")
    ap.add_argument("--output", required=True, type=Path,
                    help="input columns + appended \"prediction\" column")
    ap.add_argument("--smiles-col", default="SMILES", help="SMILES column name (default: SMILES)")
    ap.add_argument("--device", default="cpu", help="compute device forwarded to gnn members (default: cpu)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="SMILES scored per RetroTAC.predict() call (default: 64)")
    ap.add_argument("--load-strategy", choices=["eager", "lazy", "stream"], default="eager",
                    help="RetroTAC.from_pretrained member loading strategy (default: eager)")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="worker processes for fingerprint/descriptor computation; "
                         "-1 = one per CPU, 1 = sequential (default: -1)")
    ap.add_argument("--verbose", action="store_true", help="enable INFO-level logging")
    return ap.parse_args()


def main() -> None:
    """Parse CLI args and run score_csv end-to-end."""
    args = parse_args()
    score_csv(args.input, args.output, smiles_col=args.smiles_col, device=args.device,
              batch_size=args.batch_size, load_strategy=args.load_strategy,
              n_jobs=args.n_jobs, verbose=args.verbose)


if __name__ == "__main__":
    main()
