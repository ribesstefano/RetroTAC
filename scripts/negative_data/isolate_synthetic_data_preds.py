"""Analyze and filter RetroTAC predictions on the synthetic PROTAC dataset.

Reads --predictions-csv (the predictions.csv written by predict_synthetic_data.py:
every scored molecule with its "prediction" (Caruana-weighted ensemble mean) and
"uncertainty" (unweighted std across ensemble members) columns), writes summary
statistics and distribution plots, then isolates four "interesting" subsets. The
predictions are first split by ensemble uncertainty (--uncertainty-threshold,
default median) into a confident and an uncertain pool, then each pool is
independently reduced to its bottom/top --n-tail rows by "prediction":

- confident_low_predictions.csv / confident_high_predictions.csv: low ensemble
  uncertainty -- likely-correct hard negatives/positives, candidates for
  negative-data mining.
- uncertain_low_predictions.csv / uncertain_high_predictions.csv: high ensemble
  uncertainty -- candidates for manual curation (cf.
  dataset-curated-held-out.csv) rather than automated mining.

Usage:
    python scripts/negative_data/isolate_synthetic_data_preds.py \\
        --predictions-csv outputs/negative_data/predictions.csv \\
        --output-dir outputs/negative_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from matplotlib import pyplot as plt


def predictions_summary(df: pd.DataFrame, outdir: Path) -> None:
    """Write summary statistics of the predictions/uncertainty to a JSON file.

    Args:
        df: Scored DataFrame with "prediction" and "uncertainty" columns.
        outdir: Directory to write predictions_summary.json to.
    """
    quantiles = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)

    def _stats(s: pd.Series) -> dict:
        return {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "quantiles": {str(q): float(s.quantile(q)) for q in quantiles},
        }

    summary = {
        "n_molecules": len(df),
        "prediction": _stats(df["prediction"]),
        "uncertainty": _stats(df["uncertainty"]),
        "pearson_prediction_uncertainty": float(df["prediction"].corr(df["uncertainty"])),
    }
    if "source_file" in df.columns:
        summary["by_source_file"] = (
            df.groupby("source_file")["prediction"].agg(["count", "mean", "std"]).to_dict("index")
        )

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "predictions_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {out_path}")


def plot_predictions(df: pd.DataFrame, outdir: Path) -> None:
    """Plot and save prediction/uncertainty distributions.

    Args:
        df: Scored DataFrame with "prediction" and "uncertainty" columns.
        outdir: Directory to write the PNG plots to.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["prediction"].dropna(), bins=50, color="#4C72B0")
    ax.set_xlabel("Predicted synthesizability")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of predicted synthesizability")
    fig.tight_layout()
    fig.savefig(outdir / "prediction_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["uncertainty"].dropna(), bins=50, color="#DD8452")
    ax.set_xlabel("Ensemble uncertainty (std across members)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of ensemble uncertainty")
    fig.tight_layout()
    fig.savefig(outdir / "uncertainty_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["prediction"], df["uncertainty"], s=4, alpha=0.3, color="#55A868")
    ax.set_xlabel("Predicted synthesizability")
    ax.set_ylabel("Ensemble uncertainty")
    ax.set_title("Prediction vs. uncertainty")
    fig.tight_layout()
    fig.savefig(outdir / "prediction_vs_uncertainty.png", dpi=150)
    plt.close(fig)

    print(f"  wrote plots to {outdir}")


def _tail_masks(df: pd.DataFrame, n_tail: int) -> Tuple[pd.Series, pd.Series]:
    """Boolean masks selecting the bottom and top n_tail rows by "prediction".

    Unlike a quantile cut, this fixes the absolute size of each tail
    regardless of len(df) -- e.g. n_tail=10000 always isolates 10000 lowest-
    and 10000 highest-scoring molecules. n_tail is clipped to len(df) // 2 so
    the two tails never overlap.

    Returns:
        (lo_mask, hi_mask): each true for up to n_tail rows, selecting the
        lowest- and highest-scoring molecules respectively.
    """
    n_tail = min(n_tail, len(df) // 2)
    lo_mask = pd.Series(False, index=df.index)
    hi_mask = pd.Series(False, index=df.index)
    if n_tail <= 0:
        return lo_mask, hi_mask
    lo_mask.loc[df["prediction"].nsmallest(n_tail).index] = True
    hi_mask.loc[df["prediction"].nlargest(n_tail).index] = True
    return lo_mask, hi_mask


def split_by_confidence(
    df: pd.DataFrame, threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split predictions into confident/uncertain pools by ensemble uncertainty.

    Args:
        df: Scored DataFrame with "prediction" and "uncertainty" columns.
        threshold: Uncertainty split point -- confident keeps rows <= this,
            uncertain keeps rows >= this (a row exactly at the threshold
            lands in both). If None, defaults to the median uncertainty in df.

    Returns:
        (confident, uncertain): row subsets of df.
    """
    threshold = float(df["uncertainty"].median()) if threshold is None else threshold
    confident = df[df["uncertainty"] <= threshold]
    uncertain = df[df["uncertainty"] >= threshold]
    return confident, uncertain


def isolate_tails(df: pd.DataFrame, n_tail: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Isolate the bottom/top n_tail rows of df by "prediction".

    These are the rows the ensemble is most decisive about (extreme predicted
    score) within df -- e.g. within a confident pool, good candidates for
    confidently-labeled negative/positive data; within an uncertain pool,
    candidates for manual curation.

    Args:
        df: Scored DataFrame with a "prediction" column (e.g. a confident or
            uncertain pool from split_by_confidence).
        n_tail: Number of molecules to take from each tail (bottom and top)
            of the prediction distribution.

    Returns:
        (low, high): the bottom/top n_tail rows, each sorted by "prediction".
    """
    lo_mask, hi_mask = _tail_masks(df, n_tail)
    low = df[lo_mask].sort_values("prediction", ascending=True).reset_index(drop=True)
    high = df[hi_mask].sort_values("prediction", ascending=False).reset_index(drop=True)
    return low, high


def parse_args() -> argparse.Namespace:
    """Parse CLI args for analyzing/filtering synthetic-data predictions.

    Returns:
        argparse.Namespace with the parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--predictions-csv", type=Path,
                     default=Path("outputs/negative_data/predictions.csv"),
                     help="predictions.csv written by predict_synthetic_data.py")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/negative_data"),
                     help="directory to write summary, plots, and isolated subsets to")
    ap.add_argument("--n-tail", type=int, default=5_000,
                     help="number of molecules to isolate from each tail (bottom and top) "
                          "of the prediction distribution")
    ap.add_argument("--uncertainty-threshold", type=float, default=None,
                     help="uncertainty split point for confident/uncertain subsets; default: median")
    return ap.parse_args()


def main() -> None:
    """Parse CLI args and run the prediction analysis/filtering pipeline end-to-end."""
    args = parse_args()

    print(f"Loading predictions from {args.predictions_csv}...")
    df = pd.read_csv(args.predictions_csv)
    print(f"  {len(df):,} rows loaded")

    print("Writing predictions summary...")
    predictions_summary(df, args.output_dir)

    print("Plotting predictions...")
    plot_predictions(df, args.output_dir)

    print(f"Splitting by confidence (threshold={args.uncertainty_threshold})...")
    confident_df, uncertain_df = split_by_confidence(df, args.uncertainty_threshold)
    print(f"  {len(confident_df):,} confident, {len(uncertain_df):,} uncertain rows")

    print(f"Isolating tails within each pool (n_tail={args.n_tail:,})...")
    confident_low, confident_high = isolate_tails(confident_df, args.n_tail)
    uncertain_low, uncertain_high = isolate_tails(uncertain_df, args.n_tail)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    confident_low.to_csv(args.output_dir / "confident_low_predictions.csv", index=False)
    confident_high.to_csv(args.output_dir / "confident_high_predictions.csv", index=False)
    uncertain_low.to_csv(args.output_dir / "uncertain_low_predictions.csv", index=False)
    uncertain_high.to_csv(args.output_dir / "uncertain_high_predictions.csv", index=False)
    print(f"  confident: {len(confident_low):,} low + {len(confident_high):,} high; "
          f"uncertain: {len(uncertain_low):,} low + {len(uncertain_high):,} high -> {args.output_dir}")


if __name__ == "__main__":
    main()
