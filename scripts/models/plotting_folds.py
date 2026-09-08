"""
scripts/models/plotting_folds.py
=================================
Box plot of the synthesizability label's distribution per outer CV fold,
split into train/val (hue), to sanity-check that the scaffold-grouped
5x5 nested CV (retrotac.models.training.get_fold_indices) doesn't produce
folds with a skewed label distribution. Optionally also plots a held-out
test set's label distribution (its own hue level, one box past the last
fold) plus a dashed horizontal line at its mean.

Reuses the same input CSV + row order + --config as train.py so the folds
plotted here are exactly the ones train.py trains/evaluates on. Each of the
len(cross_validation.seeds) * cross_validation.n_folds outer (seed, fold_idx)
pairs is assigned a single sequential integer (1, 2, 3, ...) for the x-axis,
in the same order train.py's own seed/fold loop would visit them.

Usage:
    python scripts/models/plotting_folds.py --input data.csv
    python scripts/models/plotting_folds.py --input data.csv --test-csv data/sets/routes_test.csv
"""
import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from retrotac.chem_utils import get_scaffold  # noqa: E402
from retrotac.models.config import ModelsConfig  # noqa: E402
from retrotac.models.training import get_fold_indices  # noqa: E402

OUT_DIR = Path("figures")
HELD_OUT_LABEL = "held-out"


def build_fold_label_frame(
    df_train: pd.DataFrame, target: str, seeds: list, n_folds: int,
) -> pd.DataFrame:
    """Build a long frame of [fold_number, split, target] for every outer fold.

    Args:
        df_train: Full training DataFrame with a "scaffolds" column and the
            target column.
        target: Name of the target (label) column.
        seeds: Outer CV seeds, visited in order.
        n_folds: Outer folds per seed, visited in order.

    Returns:
        Long DataFrame with columns "fold_number" (int, 1-indexed, sequential
        over the (seed, fold_idx) pairs in visiting order), "split" ("train"
        or "val"), and `target`.
    """
    rows = []
    fold_number = 0
    for seed in seeds:
        for fold_idx in range(n_folds):
            fold_number += 1
            fold_train_idx, fold_val_idx, _, _ = get_fold_indices(
                df_train, seed, fold_idx, n_folds)
            for split, idx in (("train", fold_train_idx), ("val", fold_val_idx)):
                rows.append(pd.DataFrame({
                    "fold_number": fold_number,
                    "split": split,
                    target: df_train[target].iloc[idx].values,
                }))
    return pd.concat(rows, ignore_index=True)


def append_held_out(df_long: pd.DataFrame, held_out_values, target: str) -> pd.DataFrame:
    """Append the held-out set as one extra box, past the last fold_number.

    Args:
        df_long: Long frame as returned by build_fold_label_frame.
        held_out_values: 1-D array-like of the held-out set's target values.
        target: Name of the target (label) column.

    Returns:
        `df_long` with held-out rows appended; their "fold_number" is one past
        the last CV fold, and their "split" is HELD_OUT_LABEL, so they render
        as their own box/hue level at the right of the plot.
    """
    held_out_fold_number = df_long["fold_number"].max() + 1
    held_out_rows = pd.DataFrame({
        "fold_number": held_out_fold_number,
        "split": HELD_OUT_LABEL,
        target: held_out_values,
    })
    return pd.concat([df_long, held_out_rows], ignore_index=True)


def plot_fold_label_boxplot(
    df_long: pd.DataFrame, target: str, out_dir: Path = OUT_DIR, prefix: str = "",
    held_out_mean: Optional[float] = None,
) -> None:
    """Save a box plot of `target`, grouped by fold_number and hue-ed by split.

    Args:
        df_long: Long frame as returned by build_fold_label_frame, optionally
            extended with append_held_out.
        target: Name of the target (label) column.
        out_dir: Directory to save the figure into.
        prefix: Optional filename prefix for the saved figure.
        held_out_mean: If given, draws a dashed horizontal line at this value
            (the held-out set's mean label) across the whole plot.
    """
    n_boxes = df_long["fold_number"].nunique()
    fig, ax = plt.subplots(figsize=(max(8, n_boxes * 0.5), 5))
    sns.boxplot(data=df_long, x="fold_number", y=target, hue="split", ax=ax)
    ax.set_xlabel("Fold")
    ax.set_ylabel(target)
    ax.set_title("Label distribution per outer CV fold")

    handles, labels = ax.get_legend_handles_labels()
    if held_out_mean is not None:
        line = ax.axhline(held_out_mean, color="black", linestyle="--", linewidth=1.5)
        handles.append(line)
        labels.append(f"Held-out mean ({held_out_mean:.3f})")
        # HELD_OUT_LABEL's fold_number is the last (highest) category -> last xtick
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        xtick_labels[-1] = "Held-out"
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(xtick_labels)
    ax.legend(handles=handles, labels=labels, title=None)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{prefix}fold_label_boxplot.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {out_dir / f'{prefix}fold_label_boxplot.png'}")


def main() -> None:
    """CLI entry point: build the outer-fold splits and save the box plot."""
    ap = argparse.ArgumentParser(
        description="Box plot of the synthesizability label per outer CV fold, hue-ed by train/val.")
    ap.add_argument("--input", required=True, type=Path,
                    help="training CSV; must hold the SMILES + target columns named in --config")
    ap.add_argument("--config", type=Path, default=Path("config") / "models_config.yaml",
                    help="YAML with target / molecule_col / cross_validation "
                         "(default: config/models_config.yaml)")
    ap.add_argument("--scaffold-col", default=None,
                    help="SMILES column to compute scaffolds from; defaults to the config's molecule_col")
    ap.add_argument("--test-csv", default=None,
                    help="optional held-out test CSV; its label distribution is plotted as an "
                         "extra box past the last fold, with a dashed line at its mean")
    ap.add_argument("--target-col", default=None,
                    help="target column in --test-csv (default: --config's target)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="directory to save the figure into (default: figures)")
    ap.add_argument("--prefix", default="",
                    help="optional filename prefix for the saved figure")
    args = ap.parse_args()

    cfg = ModelsConfig.load(args.config)
    molecule_col = cfg.molecule_col
    target = cfg.target
    scaffold_col = args.scaffold_col or molecule_col

    df_train = pd.read_csv(args.input)
    # same fixed row order train.py uses, so get_fold_indices reproduces the
    # exact same folds train.py trains/evaluates on
    df_train = df_train.sort_values(molecule_col, kind="stable").reset_index(drop=True)
    df_train["scaffolds"] = df_train[scaffold_col].apply(get_scaffold)

    df_long = build_fold_label_frame(
        df_train, target, cfg.cross_validation.seeds, cfg.cross_validation.n_folds)

    held_out_mean = None
    if args.test_csv:
        target_col = args.target_col or target
        held_out_values = pd.read_csv(args.test_csv)[target_col]
        df_long = append_held_out(df_long, held_out_values, target)
        held_out_mean = float(held_out_values.mean())

    plot_fold_label_boxplot(
        df_long, target, out_dir=args.out_dir, prefix=args.prefix, held_out_mean=held_out_mean)


if __name__ == "__main__":
    main()
