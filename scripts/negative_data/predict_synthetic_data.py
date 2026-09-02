"""Score the synthetic PROTAC dataset with the RetroTAC Caruana ensemble.

Loads every CSV under --data-dir whose name matches --pattern (by default the
three `dataset-synthetic-{train,test,validation}.csv` splits under
data/synthetic/smiles/ -- `dataset-curated-held-out.csv` is deliberately
excluded: it already carries a "Manually Curated" label and isn't a
prediction target), standardizes their SMILES columns, drops any molecule
also present in the route-scored training set (data/sets/routes_train_val.csv)
so predictions can't leak train data back in as "new" negative-data
candidates, then scores the remainder with `retrotac.models.ensemble.RetroTAC`
(the 27-member xgb/mlp/gnn Caruana ensemble). Predictions are written
incrementally so a crash only loses at most one --batch-size worth of work,
and a rerun resumes automatically by skipping molecules already scored.

Two "interesting" subsets are isolated from the tails of the prediction
distribution: confident extreme predictions (low ensemble uncertainty --
likely-correct hard positives/negatives, candidates for negative-data mining)
and uncertain ones (high ensemble uncertainty -- candidates for manual
curation, cf. dataset-curated-held-out.csv).

Only the gnn ensemble members (13 of 27) honor --device; the mlp members
always run on CPU (TorchMLPRegressor.load() has no device parameter) and the
xgb members run on whatever device they were saved with (CPU; see CLAUDE.md's
note that XGBoost's GPU-enabled wheel cannot even load on the Berzelius login
node). --device defaults to cuda when available since it is the ensemble's
only inference-time GPU lever.

Usage:
    python scripts/negative_data/predict_synthetic_data.py \\
        --model-path retrotac_staged \\
        --output-dir outputs/negative_data
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
import torch
from matplotlib import pyplot as plt

from retrotac.chem_utils import papply, std_smiles

if TYPE_CHECKING:
    from retrotac.models.ensemble import RetroTAC

PROTAC_COL = "PROTAC SMILES"
SMILES_COLS = [PROTAC_COL, "Warhead SMILES", "Linker SMILES", "E3 Ligase Ligand SMILES"]


def load_synthetic_data(
    data_dir: Path, pattern: str = "*synthetic*.csv", n_jobs: int = 1
) -> pd.DataFrame:
    """Load, standardize, and deduplicate every synthetic-dataset CSV in a directory.

    Args:
        data_dir: Directory holding the synthetic-dataset CSVs (e.g.
            data/synthetic/smiles), each with at least a "PROTAC SMILES" column
            and typically also "Warhead SMILES"/"Linker SMILES"/
            "E3 Ligase Ligand SMILES".
        pattern: Glob pattern (relative to data_dir) selecting which CSVs to
            load. Default matches the three dataset-synthetic-* splits and
            skips dataset-curated-held-out.csv (see module docstring).
        n_jobs: Forwarded to papply for SMILES standardization; see papply's
            docstring (1 = sequential, -1 = one worker per CPU).

    Returns:
        Concatenated DataFrame with every SMILES column standardized (RDKit
        canonical form), an added "source_file" column recording provenance,
        and duplicate "PROTAC SMILES" rows dropped (first occurrence kept).

    Raises:
        FileNotFoundError: If no file under data_dir matches pattern.
    """
    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} under {data_dir}")

    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["source_file"] = f.name
        print(f"  loaded {len(d):,} rows from {f.name}")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    for col in SMILES_COLS:
        if col in df.columns:
            df[col] = papply(df[col], std_smiles, desc=f"Standardizing {col}", n_jobs=n_jobs)

    before = len(df)
    df = df.dropna(subset=[PROTAC_COL])
    df = df.drop_duplicates(subset=[PROTAC_COL]).reset_index(drop=True)
    print(f"  {len(df):,}/{before:,} rows kept after standardization + dedup on {PROTAC_COL!r}")
    return df


def remove_train_data(
    df: pd.DataFrame, train_df: pd.DataFrame, train_smiles_col: str = "smiles", n_jobs: int = 1
) -> pd.DataFrame:
    """Drop synthetic rows whose PROTAC SMILES also appears in the training set.

    Args:
        df: Synthetic dataset, standardized (see load_synthetic_data), with a
            "PROTAC SMILES" column.
        train_df: Training set (e.g. data/sets/routes_train_val.csv) with a
            SMILES column named train_smiles_col.
        train_smiles_col: Name of the SMILES column in train_df.
        n_jobs: Forwarded to papply for SMILES standardization; see papply's
            docstring (1 = sequential, -1 = one worker per CPU).

    Returns:
        df filtered to rows not present (by standardized SMILES) in train_df.
    """
    train_smiles = set(
        papply(
            train_df[train_smiles_col], std_smiles, desc="Standardizing train SMILES", n_jobs=n_jobs
        ).dropna()
    )

    before = len(df)
    filtered = df[~df[PROTAC_COL].isin(train_smiles)].reset_index(drop=True)
    print(f"  removed {before - len(filtered):,}/{before:,} rows overlapping with the training set")
    return filtered


def get_predictions(
    df: pd.DataFrame, model: "RetroTAC", batch_size: int, outdir: Path, resume: bool = True
) -> pd.DataFrame:
    """Score df's PROTAC SMILES with model, checkpointing after every batch.

    Writes/appends to outdir/"predictions.csv" one batch at a time so a crash
    loses at most one batch_size worth of work. When resume=True and that
    file already exists, molecules it already covers are skipped.

    Args:
        df: DataFrame with a "PROTAC SMILES" column to score.
        model: Loaded RetroTAC ensemble (see RetroTAC.from_pretrained).
        batch_size: Number of molecules scored per model.predict() call and
            per checkpoint flush.
        outdir: Directory to write predictions.csv to (created if missing).
        resume: If True and outdir/"predictions.csv" exists, skip molecules
            already present in it. If False, any existing file is discarded.

    Returns:
        df with "prediction" (Caruana-weighted mean) and "uncertainty"
        (unweighted std across the 27 members) columns added, restricted to
        rows actually requested via df (stale checkpoint rows outside df's
        current scope, e.g. from a differently-filtered previous run, are
        excluded).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "predictions.csv"

    already = set()
    if resume and out_path.exists():
        already = set(pd.read_csv(out_path, usecols=[PROTAC_COL])[PROTAC_COL])
    elif out_path.exists():
        out_path.unlink()

    todo = df[~df[PROTAC_COL].isin(already)].reset_index(drop=True) if already else df.reset_index(drop=True)
    print(f"  {len(already):,}/{len(df):,} molecules already scored; {len(todo):,} remaining")

    write_header = not out_path.exists()
    n_batches = math.ceil(len(todo) / batch_size) if len(todo) else 0
    for i in range(n_batches):
        print(f"  batch {i + 1}/{n_batches}")
        batch = todo.iloc[i * batch_size:(i + 1) * batch_size].copy()
        result = model.predict(batch[PROTAC_COL].tolist(), return_details=True)
        batch["prediction"] = result["mean"]
        batch["uncertainty"] = result["std"]
        batch.to_csv(out_path, mode="a", header=write_header, index=False)
        write_header = False

    scored = pd.read_csv(out_path)
    return scored[scored[PROTAC_COL].isin(set(df[PROTAC_COL]))].reset_index(drop=True)


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


def _tail_mask(df: pd.DataFrame, pct_tail: float) -> pd.Series:
    """Boolean mask selecting rows in the bottom/top pct_tail of "prediction"."""
    lo = df["prediction"].quantile(pct_tail)
    hi = df["prediction"].quantile(1 - pct_tail)
    return (df["prediction"] <= lo) | (df["prediction"] >= hi)


def isolate_confident_predictions(
    df: pd.DataFrame, threshold: Optional[float] = None, pct_tail: float = 0.1
) -> pd.DataFrame:
    """Isolate low-uncertainty predictions in the tails of the score distribution.

    These are the rows the ensemble both agrees on (low uncertainty) and is
    most decisive about (extreme predicted score) -- good candidates for
    confidently-labeled negative/positive data.

    Args:
        df: Scored DataFrame with "prediction" and "uncertainty" columns.
        threshold: Uncertainty must be <= this to count as confident. If
            None, defaults to the median uncertainty in df.
        pct_tail: Fraction defining each tail of the prediction distribution
            (e.g. 0.1 keeps the bottom and top deciles).

    Returns:
        Rows satisfying both conditions, sorted by "prediction".
    """
    threshold = float(df["uncertainty"].median()) if threshold is None else threshold
    mask = _tail_mask(df, pct_tail) & (df["uncertainty"] <= threshold)
    return df[mask].sort_values("prediction").reset_index(drop=True)


def isolate_uncertain_predictions(
    df: pd.DataFrame, threshold: Optional[float] = None, pct_tail: float = 0.1
) -> pd.DataFrame:
    """Isolate high-uncertainty predictions in the tails of the score distribution.

    These are rows the ensemble is decisive about on average but disagrees on
    internally -- good candidates for manual curation (cf.
    dataset-curated-held-out.csv) rather than automated negative-data mining.

    Args:
        df: Scored DataFrame with "prediction" and "uncertainty" columns.
        threshold: Uncertainty must be >= this to count as uncertain. If
            None, defaults to the median uncertainty in df.
        pct_tail: Fraction defining each tail of the prediction distribution.

    Returns:
        Rows satisfying both conditions, sorted by "prediction".
    """
    threshold = float(df["uncertainty"].median()) if threshold is None else threshold
    mask = _tail_mask(df, pct_tail) & (df["uncertainty"] >= threshold)
    return df[mask].sort_values("prediction").reset_index(drop=True)


def resolve_device(requested: Optional[str]) -> str:
    """Resolve the compute device for the gnn ensemble members.

    torch.cuda.is_available() only checks that the driver sees a GPU -- on
    the Berzelius login node it returns True even though the GPU's
    compute_mode=Prohibited rejects any actual context creation (see
    CLAUDE.md's login-node GPU note, previously observed for xgboost's
    .fit() and, per this function, equally true of any CUDA allocation
    including GNN checkpoint loading). So when no --device is given, this
    probes with a real small allocation and falls back to cpu on failure
    instead of trusting is_available() alone -- inside a SLURM
    `--gpus=1` allocation the probe succeeds and cuda is used.

    Args:
        requested: The user's explicit --device value, or None to auto-detect.

    Returns:
        "cuda" or "cpu" (or `requested` verbatim if given).
    """
    if requested is not None:
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda")
        return "cuda"
    except RuntimeError as e:
        print(f"  cuda reports available but is not usable ({e}); falling back to cpu")
        return "cpu"


def parse_args() -> argparse.Namespace:
    """Parse CLI args for scoring the synthetic dataset with the RetroTAC ensemble.

    Returns:
        argparse.Namespace with the parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data/synthetic/smiles"),
                     help="directory holding the synthetic-dataset CSVs")
    ap.add_argument("--pattern", default="*synthetic*.csv",
                     help="glob (within --data-dir) selecting which CSVs to score")
    ap.add_argument("--train-csv", type=Path, default=Path("data/sets/routes_train_val.csv"),
                     help="training set to exclude from the synthetic data by SMILES overlap")
    ap.add_argument("--train-smiles-col", default="smiles", help="SMILES column name in --train-csv")
    ap.add_argument("--n-jobs", type=int, default=-1,
                     help="worker processes for SMILES standardization; -1 = one per CPU, 1 = sequential")
    ap.add_argument("--model-path", default="retrotac_staged",
                     help="local RetroTAC ensemble directory or Hugging Face Hub repo id")
    ap.add_argument("--load-strategy", default="eager", choices=["eager", "lazy", "stream"],
                     help="ensemble member residency strategy (see RetroTAC.from_pretrained)")
    ap.add_argument("--device", default=None,
                     help="compute device for the gnn ensemble members; default: cuda if available, else cpu")
    ap.add_argument("--batch-size", type=int, default=512, help="molecules per predict()/checkpoint")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/negative_data"),
                     help="directory to write predictions, summary, plots, and isolated subsets to")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                     help="ignore any existing predictions.csv and rescore everything")
    ap.add_argument("--pct-tail", type=float, default=0.1,
                     help="fraction of each tail of the prediction distribution to isolate")
    ap.add_argument("--uncertainty-threshold", type=float, default=None,
                     help="uncertainty split point for confident/uncertain subsets; default: median")
    ap.add_argument("--limit", type=int, default=None,
                     help="only score the first N molecules after filtering (for smoke-testing)")
    return ap.parse_args()


def main() -> None:
    """Parse CLI args and run the synthetic-data scoring pipeline end-to-end."""
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device={device!r} for gnn ensemble members")

    print(f"Loading synthetic data from {args.data_dir} (pattern={args.pattern!r})...")
    df = load_synthetic_data(args.data_dir, args.pattern, n_jobs=args.n_jobs)

    print(f"Removing rows overlapping with {args.train_csv}...")
    train_df = pd.read_csv(args.train_csv, usecols=[args.train_smiles_col])
    df = remove_train_data(df, train_df, args.train_smiles_col, n_jobs=args.n_jobs)

    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)
        print(f"  --limit applied: scoring only the first {len(df):,} rows")

    print(f"Loading RetroTAC ensemble from {args.model_path!r}...")
    from retrotac import RetroTAC

    model = RetroTAC.from_pretrained(
        args.model_path, load_strategy=args.load_strategy, device=device
    )

    print(f"Scoring {len(df):,} molecules in batches of {args.batch_size}...")
    df = get_predictions(df, model, args.batch_size, args.output_dir, resume=args.resume)

    print("Writing predictions summary...")
    predictions_summary(df, args.output_dir)

    print("Plotting predictions...")
    plot_predictions(df, args.output_dir)

    print("Isolating confident and uncertain predictions...")
    confident = isolate_confident_predictions(df, args.uncertainty_threshold, args.pct_tail)
    uncertain = isolate_uncertain_predictions(df, args.uncertainty_threshold, args.pct_tail)
    confident.to_csv(args.output_dir / "confident_predictions.csv", index=False)
    uncertain.to_csv(args.output_dir / "uncertain_predictions.csv", index=False)
    print(f"  {len(confident):,} confident, {len(uncertain):,} uncertain rows -> {args.output_dir}")


if __name__ == "__main__":
    main()
