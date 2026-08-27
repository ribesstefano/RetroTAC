"""
isolate_heldout.py
───────────────────
Splits a scored-routes CSV into a train/val set and a diverse held-out
(test) set, using RDKit's MaxMinPicker on Morgan fingerprints to select the
held-out rows: the *most chemically diverse* subset of the input, which
makes for a harder, less redundant generalization test than a random split.

Also reports whether the target column's distribution is similar between
the two resulting sets (descriptive stats + a two-sample statistical test),
and can optionally render a handful of example molecules from each set.

Usage
-----
    python isolate_heldout.py data/routes/routes_scored.csv \
        --smiles-col smiles --target-col synthesizability \
        --heldout-pct 10 --output-dir data/routes --make-figures

Arguments:
    input_csv            Input CSV path.
    --smiles-col          SMILES column name (default: "smiles").
    --target-col          Target column name (default: "synthesizability").
    --heldout-pct         Percentage (0-100) of rows to isolate into the
                          held-out set via MaxMin diversity picking
                          (default: 10.0).
    --output-dir          Output directory (default: "data/routes").
    --train-val-filename  Filename for the non-held-out split
                          (default: "routes_train_val.csv").
    --test-filename       Filename for the held-out split
                          (default: "routes_test.csv").
    --seed                Random seed for MaxMinPicker and figure sampling
                          (default: 42).
    --make-figures        If set, also render up to --n-figure-mols example
                          molecules from each set as grid images, plus an
                          overlapping train_val-vs-test target distribution
                          plot, under <output-dir>/smiles_figures/.
    --n-figure-mols       Max molecules per figure (default: 20).
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import Draw
from rdkit.SimDivFilters import rdSimDivPickers
from scipy import stats
from tabulate import tabulate

from protac_synth.chem_utils import morgan_fp, papply, smiles_to_mol

RDLogger.DisableLog("rdApp.*")

MAX_CATEGORICAL_UNIQUE = 10


def _is_categorical(series: pd.Series) -> bool:
    """Heuristic: non-numeric dtype, or few enough unique values to be a class label."""
    return series.dtype == object or series.nunique() <= MAX_CATEGORICAL_UNIQUE


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_csv", type=str, help="Input CSV path.")
    p.add_argument("--smiles-col", type=str, default="smiles", help="SMILES column name.")
    p.add_argument("--target-col", type=str, default="synthesizability", help="Target column name.")
    p.add_argument(
        "--heldout-pct",
        type=float,
        default=10.0,
        help="Percentage (0-100) of rows to isolate into the held-out set (default: 10.0).",
    )
    p.add_argument("--output-dir", type=str, default="data/routes", help="Output directory.")
    p.add_argument(
        "--train-val-filename", type=str, default="routes_train_val.csv", help="Non-held-out split filename."
    )
    p.add_argument("--test-filename", type=str, default="routes_test.csv", help="Held-out split filename.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for MaxMinPicker and figure sampling.")
    p.add_argument(
        "--make-figures",
        action="store_true",
        help="Render up to --n-figure-mols example molecules per set under <output-dir>/smiles_figures/.",
    )
    p.add_argument("--n-figure-mols", type=int, default=20, help="Max molecules per figure (default: 20).")
    return p.parse_args()


def pick_heldout_indices(fps: List, pct: float, seed: int) -> List[int]:
    """Select the most chemically diverse subset of fingerprints via MaxMinPicker.

    Args:
        fps: RDKit ``ExplicitBitVect`` Morgan fingerprints, one per row.
        pct: Percentage (0-100) of rows to select.
        seed: Random seed for the picker's first choice.

    Returns:
        Row indices (into *fps*) of the selected, maximally diverse subset.
    """
    n = len(fps)
    pick_size = round(n * pct / 100.0)
    if pick_size < 1 or pick_size >= n:
        raise ValueError(f"--heldout-pct={pct} yields pick_size={pick_size} for n={n}; must be in [1, {n - 1}].")
    picker = rdSimDivPickers.MaxMinPicker()
    picked = picker.LazyBitVectorPick(fps, n, pick_size, seed=seed)
    return list(picked)


def describe_distribution(train_val: pd.Series, test: pd.Series, target_col: str) -> str:
    """Compare the target column's distribution between the two splits.

    Auto-detects continuous vs. categorical (<= MAX_CATEGORICAL_UNIQUE unique
    values, or non-numeric dtype) and reports descriptive stats plus a
    two-sample statistical test (Kolmogorov-Smirnov for continuous,
    chi-square for categorical).

    Args:
        train_val: Target column values for the non-held-out split.
        test: Target column values for the held-out split.
        target_col: Column name, used only for the report header.

    Returns:
        Human-readable report string.
    """
    train_val = train_val.dropna()
    test = test.dropna()

    lines = [f"=== Target column '{target_col}' distribution: train_val vs. held-out test ==="]
    if _is_categorical(train_val):
        counts = pd.concat(
            [train_val.value_counts().rename("train_val"), test.value_counts().rename("test")], axis=1
        ).fillna(0)
        fracs = counts / counts.sum(axis=0)
        table = pd.concat(
            [counts.astype(int).add_suffix("_count"), fracs.round(4).add_suffix("_frac")], axis=1
        ).sort_index()
        lines.append(tabulate(table, headers="keys", tablefmt="github"))
        chi2, p_value, _, _ = stats.chi2_contingency(counts)
        lines.append(f"\nChi-square test of independence: statistic={chi2:.4f}, p-value={p_value:.4g}")
    else:
        summary = pd.DataFrame({"train_val": train_val.describe(), "test": test.describe()})
        lines.append(tabulate(summary, headers="keys", tablefmt="github"))
        ks_stat, p_value = stats.ks_2samp(train_val, test)
        lines.append(f"\nTwo-sample Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={p_value:.4g}")

    verdict = "similar" if p_value >= 0.05 else "significantly different"
    lines.append(f"-> p-value {'>=' if p_value >= 0.05 else '<'} 0.05: distributions look {verdict} (alpha=0.05).")
    return "\n".join(lines)


def plot_target_overlap(train_val: pd.Series, test: pd.Series, target_col: str, out_path: Path) -> None:
    """Save a plot overlapping the target column's distribution between the two splits.

    Continuous targets get overlaid, density-normalized histograms (raw counts
    would be misleading since the two splits have very different sizes);
    categorical targets get a grouped bar chart of per-class fractions.

    Args:
        train_val: Target column values for the non-held-out split.
        test: Target column values for the held-out split.
        target_col: Column name, used for axis/title labels.
        out_path: PNG file path to write (parent dir created if missing).
    """
    train_val = train_val.dropna()
    test = test.dropna()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    if _is_categorical(train_val):
        categories = sorted(set(train_val) | set(test))
        x = np.arange(len(categories))
        width = 0.35
        train_frac = train_val.value_counts(normalize=True).reindex(categories, fill_value=0)
        test_frac = test.value_counts(normalize=True).reindex(categories, fill_value=0)
        ax.bar(x - width / 2, train_frac, width, alpha=0.7, label="train_val")
        ax.bar(x + width / 2, test_frac, width, alpha=0.7, label="test (held-out)")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Fraction")
    else:
        bins = np.histogram_bin_edges(pd.concat([train_val, test]), bins=30)
        ax.hist(train_val, bins=bins, alpha=0.5, density=True, label="train_val")
        ax.hist(test, bins=bins, alpha=0.5, density=True, label="test (held-out)")
        ax.set_ylabel("Density")

    ax.set_xlabel(target_col)
    ax.set_title(f"'{target_col}' distribution: train_val vs. held-out test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_figures(
    df: pd.DataFrame, smiles_col: str, target_col: str, out_dir: Path, n: int, seed: int, name: str
) -> None:
    """Render a grid image of up to *n* sampled molecules from *df*.

    Args:
        df: DataFrame with a ``_mol`` column of pre-parsed RDKit Mols.
        smiles_col: SMILES column name, used only for the sub-image legend fallback.
        target_col: Target column name, shown as the sub-image legend.
        out_dir: Directory the PNG is written into (created if missing).
        n: Max number of molecules to sample and render.
        seed: Random seed for sampling.
        name: Output filename (without directory).
    """
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    legends = [f"{v:.3f}" if pd.notna(v) else "NA" for v in sample[target_col]]
    img = Draw.MolsToGridImage(
        sample["_mol"].tolist(), molsPerRow=5, subImgSize=(250, 250), legends=legends
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / name)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    for col in (args.smiles_col, args.target_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {args.input_csv}. Available: {df.columns.tolist()}")

    print(f"Loaded {len(df)} rows from {args.input_csv}")
    df["_mol"] = papply(df[args.smiles_col], smiles_to_mol, desc="Parsing SMILES")
    n_invalid = df["_mol"].isna().sum()
    if n_invalid:
        print(f"Dropping {n_invalid} rows with unparseable SMILES")
        df = df[df["_mol"].notna()].reset_index(drop=True)

    df["_fp"] = papply(df["_mol"], morgan_fp, desc="Computing Morgan fingerprints")

    print(f"Picking the {args.heldout_pct}% most diverse rows as the held-out set (seed={args.seed}) …")
    heldout_idx = pick_heldout_indices(df["_fp"].tolist(), args.heldout_pct, args.seed)
    is_heldout = np.zeros(len(df), dtype=bool)
    is_heldout[heldout_idx] = True

    test_df = df[is_heldout].drop(columns=["_mol", "_fp"])
    train_val_df = df[~is_heldout].drop(columns=["_mol", "_fp"])
    print(f"train_val: {len(train_val_df)} rows | test (held-out): {len(test_df)} rows")

    train_val_path = out_dir / args.train_val_filename
    test_path = out_dir / args.test_filename
    train_val_df.to_csv(train_val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved → {train_val_path}")
    print(f"Saved → {test_path}")

    report = describe_distribution(train_val_df[args.target_col], test_df[args.target_col], args.target_col)
    print("\n" + report)
    report_path = out_dir / "heldout_split_report.txt"
    report_path.write_text(report + "\n")
    print(f"\nSaved → {report_path}")

    if args.make_figures:
        fig_dir = out_dir / "smiles_figures"
        save_figures(
            df[~is_heldout], args.smiles_col, args.target_col, fig_dir, args.n_figure_mols, args.seed,
            "train_val_sample.png",
        )
        save_figures(
            df[is_heldout], args.smiles_col, args.target_col, fig_dir, args.n_figure_mols, args.seed,
            "test_heldout_sample.png",
        )
        plot_target_overlap(
            train_val_df[args.target_col], test_df[args.target_col], args.target_col,
            fig_dir / "target_distribution_overlap.png",
        )
        print(f"Saved figures → {fig_dir}")


if __name__ == "__main__":
    main()
