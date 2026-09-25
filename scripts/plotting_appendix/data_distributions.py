"""
scripts/plotting_appendix/data_distributions.py
===============================================
The two data-side appendix items: how the Butina cutoff was chosen for the
held-out split, and what the 25 outer cross-validation folds look like.

Butina
------
`data/sets/adaptive_cluster_metrics.csv` is the sweep written while isolating
the held-out set (`scripts/dataset/isolate_heldout.py`): one row per Tanimoto
cutoff with the three internal cluster-quality indices, the composite quality
score, the cluster-size summary and the held-out fraction actually attainable
at that cutoff. This script turns it into the appendix table; the figure it
accompanies is the existing `figures/butina/adaptive_metrics_vs_threshold.pdf`
(`scripts/dataset/plot_butina_clustering.py`).

Folds
-----
The outer folds are *recomputed here* with the same function and the same
seeds `train.py` used (`retrotac.models.training.get_fold_indices`, grouping
on Bemis-Murcko scaffolds of the warhead via `--scaffold-col wh_smiles`), so
the fold statistics reported in the appendix are the folds the models were
actually evaluated on rather than a fresh random partition.

Usage
-----
    python scripts/plotting_appendix/data_distributions.py \\
        --train-val data/sets/routes_train_val.csv \\
        --test data/sets/routes_test.csv \\
        --cluster-metrics data/sets/adaptive_cluster_metrics.csv

Outputs (see --out-dir / --fig-dir):
    outputs/appendix/butina_cutoff_sweep.csv
    outputs/appendix/fold_statistics.csv
    figures/appendix/fold_distributions.{pdf,svg,png}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from retrotac.chem_utils import get_scaffold
from retrotac.models.config import ModelsConfig
from retrotac.models.training import get_fold_indices

import pubstyle as ps

ps.apply_style()

#: Columns kept for the appendix's Butina table, with display labels. The
#: remaining sweep columns (cluster-size skewness, min/median size, average
#: cluster-to-dataset ratio) stay in the CSV only.
BUTINA_COLUMNS = {
    "cutoff": "Cutoff",
    "num_clusters": "Clusters",
    "avg_cluster_size": "Mean size",
    "max_cluster_size": "Largest",
    "silhouette": "Silhouette",
    "davies_bouldin": "Davies--Bouldin",
    "calinski_harabasz": "Calinski--Harabasz",
    "quality_score": "Quality",
    "achieved_pct": "Held-out (\\%)",
}


def butina_table(path: Path, out_dir: Path) -> pd.DataFrame:
    """Load and tidy the Butina cutoff sweep.

    Args:
        path: `adaptive_cluster_metrics.csv`.
        out_dir: Directory the tidied CSV is written to.

    Returns:
        The tidied frame.
    """
    df = pd.read_csv(path)
    tidy = df[list(BUTINA_COLUMNS)].rename(columns=BUTINA_COLUMNS)
    tidy.to_csv(out_dir / "butina_cutoff_sweep.csv", index=False)
    return tidy


def fold_frame(df: pd.DataFrame, target: str, seeds: List[int],
               n_folds: int) -> pd.DataFrame:
    """Per-(seed, fold) statistics of the outer validation partitions.

    Args:
        df: Development set, already carrying a "scaffolds" column.
        target: Label column.
        seeds: Outer CV seeds, visited in `train.py`'s order.
        n_folds: Outer folds per seed.

    Returns:
        One row per outer fold, with size, group count and label summary.
    """
    values = df[target].to_numpy(float)
    rows = []
    cycle = 0
    for seed in seeds:
        for fold_idx in range(n_folds):
            train_idx, val_idx, inner_train, inner_val = get_fold_indices(
                df, seed, fold_idx, n_folds=n_folds)
            val = values[val_idx]
            groups = df["scaffolds"].iloc[val_idx]
            largest = groups.value_counts().iloc[0]
            cycle += 1
            rows.append({
                "cv_cycle": cycle, "seed": seed, "fold_idx": fold_idx,
                "n_train": len(train_idx), "n_val": len(val_idx),
                "n_inner_train": len(inner_train), "n_inner_val": len(inner_val),
                "val_frac": len(val_idx) / len(df),
                "n_groups_val": int(groups.nunique()),
                "largest_group_val": int(largest),
                "largest_group_val_pct": 100.0 * largest / len(val_idx),
                "label_mean": float(val.mean()), "label_std": float(val.std(ddof=1)),
                "label_q1": float(np.percentile(val, 25)),
                "label_median": float(np.median(val)),
                "label_q3": float(np.percentile(val, 75)),
                "label_zero_pct": float(100.0 * np.mean(val == 0.0)),
                "label_one_pct": float(100.0 * np.mean(val == 1.0)),
                # Against the development set the fold was drawn from: large
                # values would mean the grouped split is producing folds that
                # are not exchangeable with the whole.
                "ks_vs_development": float(stats.ks_2samp(val, values).statistic),
            })
    return pd.DataFrame(rows)


def plot_folds(folds: pd.DataFrame, df: pd.DataFrame, target: str,
               held_out: np.ndarray, seeds: List[int], n_folds: int,
               out_stem: Path) -> List[Path]:
    """Fold sizes, per-fold label spread, and the label ECDF of every fold.

    Args:
        folds: Output of `fold_frame`.
        df: Development set with the "scaffolds" column.
        target: Label column.
        held_out: Held-out label values, drawn as a reference.
        seeds: Outer CV seeds.
        n_folds: Outer folds per seed.
        out_stem: Output path without extension.

    Returns:
        Paths written.
    """
    values = df[target].to_numpy(float)
    width, _ = ps.set_size()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(width * 1.2, width * 0.40),
                                           layout="constrained")

    # (a) fold sizes: the greedy scaffold-to-fold assignment balances sizes
    # without ever splitting a group, so the spread here is the cost of that
    # constraint. What each dashed reference line means is left to the
    # caption -- in panels this dense an in-plot label lands on the data.
    # Dots, not bars: the whole range is a few percent around n/5, and a bar
    # chart would have to be drawn on a truncated baseline to show it.
    target_size = len(df) / n_folds
    ax_a.scatter(folds["cv_cycle"], folds["n_val"], s=10,
                 color=ps.PALETTE["blue"], zorder=3, linewidths=0)
    ax_a.axhline(target_size, color=ps.STATS["reference_line"],
                 linestyle="--", linewidth=1, alpha=0.7, zorder=4)
    ax_a.set_xlabel("Outer fold index")
    ax_a.set_ylabel("Validation fold size")
    ax_a.margins(y=0.25)

    # (b) label spread per fold, as median with the interquartile range.
    ax_b.errorbar(
        folds["cv_cycle"], folds["label_median"],
        yerr=[folds["label_median"] - folds["label_q1"],
              folds["label_q3"] - folds["label_median"]],
        fmt="o", markersize=2.5, elinewidth=0.8, capsize=1.2,
        color=ps.PALETTE["blue"], zorder=3)
    development_median = float(np.median(values))
    ax_b.axhline(development_median, color=ps.STATS["reference_line"],
                 linestyle="--", linewidth=1, alpha=0.7, zorder=4)
    ax_b.set_xlabel("Outer fold index")
    ax_b.set_ylabel("Route-derived score spread")

    # (c) every fold's label ECDF on one axis, against the development set and
    # the held-out set. Overlapping grey curves are the point: the folds are
    # interchangeable, and the held-out curve is the one that is not.
    grid = np.linspace(0, 1, 401)
    for seed in seeds:
        for fold_idx in range(n_folds):
            _, val_idx, _, _ = get_fold_indices(df, seed, fold_idx, n_folds=n_folds)
            ax_c.plot(grid, _ecdf(values[val_idx], grid), color="#B8B8B8",
                      linewidth=0.5, zorder=2)
    ax_c.plot(grid, _ecdf(values, grid), color=ps.PALETTE["blue"],
              linewidth=1.6, zorder=4, label="Development")
    ax_c.plot(grid, _ecdf(held_out, grid), color=ps.PALETTE["dark_orange"],
              linewidth=1.6, zorder=5, label="Held-out")
    ax_c.plot([], [], color="#B8B8B8", linewidth=0.8, label="Outer folds (25)")
    ax_c.set_xlabel("Route-derived score")
    ax_c.set_ylabel("Cumulative fraction")
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.legend(loc="upper left", fontsize=ps.ANNOT_FONTSIZE)

    for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)")):
        ax.text(-0.285, 1.12, tag, transform=ax.transAxes, fontweight="bold",
                fontsize=ps.PANEL_LABEL_FONTSIZE, va="top", ha="left", clip_on=False)
    return ps.save_figure(fig, out_stem)


def _ecdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Empirical CDF of `values` evaluated on `grid`.

    Args:
        values: Observations.
        grid: Points to evaluate at.

    Returns:
        Cumulative fraction at each grid point.
    """
    ordered = np.sort(values)
    return np.searchsorted(ordered, grid, side="right") / len(ordered)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-val", type=Path, default=Path("data/sets/routes_train_val.csv"),
                   help="Development set, the same CSV train.py was given.")
    p.add_argument("--test", type=Path, default=Path("data/sets/routes_test.csv"),
                   help="Held-out set.")
    p.add_argument("--cluster-metrics", type=Path,
                   default=Path("data/sets/adaptive_cluster_metrics.csv"),
                   help="Butina cutoff sweep written by the held-out isolation step.")
    p.add_argument("--config", type=Path, default=Path("config/models_config.yaml"),
                   help="Model config, read for the CV seeds, folds and target.")
    p.add_argument("--scaffold-col", default="wh_smiles",
                   help="Column the grouping scaffold is computed from; the SLURM "
                        "arrays pass wh_smiles (the warhead).")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/appendix"),
                   help="Directory for the tables.")
    p.add_argument("--fig-dir", type=Path, default=Path("figures/appendix"),
                   help="Directory for the figure.")
    return p.parse_args()


def main() -> None:
    """Write the Butina sweep table, the fold statistics and the fold figure."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    butina = butina_table(args.cluster_metrics, args.out_dir)
    print("=== Butina cutoff sweep ===")
    print(butina.round(4).to_string(index=False))

    cfg = ModelsConfig.load(args.config)
    seeds = list(cfg.cross_validation.seeds)
    n_folds = int(cfg.cross_validation.n_folds)
    target = cfg.target

    df = pd.read_csv(args.train_val)
    df["scaffolds"] = df[args.scaffold_col].apply(get_scaffold)
    held_out = pd.read_csv(args.test)[target].to_numpy(float)
    print(f"\ndevelopment n = {len(df)}, held-out n = {len(held_out)}; "
          f"{df['scaffolds'].nunique()} {args.scaffold_col} scaffold groups, "
          f"largest {df['scaffolds'].value_counts().iloc[0]} "
          f"({100 * df['scaffolds'].value_counts().iloc[0] / len(df):.1f}%)")

    folds = fold_frame(df, target, seeds, n_folds)
    folds.to_csv(args.out_dir / "fold_statistics.csv", index=False)
    print(f"\n=== outer folds ({len(folds)} = {len(seeds)} seeds x {n_folds} folds) ===")
    summary = folds[["n_val", "val_frac", "n_groups_val", "largest_group_val_pct",
                     "label_mean", "label_std", "label_zero_pct", "label_one_pct",
                     "ks_vs_development"]].describe().loc[["mean", "std", "min", "max"]]
    print(summary.round(4).to_string())

    written = plot_folds(folds, df, target, held_out, seeds, n_folds,
                         args.fig_dir / "fold_distributions")
    print("\nwrote:", *[str(p) for p in written], sep="\n  ")


if __name__ == "__main__":
    main()
