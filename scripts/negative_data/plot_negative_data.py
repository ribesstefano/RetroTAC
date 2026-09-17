"""Plot route-tree synthesizability scores for the negative-data candidate pools.

Reads the four route-scored CSVs written by running
``retro_scores/route_scores/route_tree_score.py`` over the confident/uncertain
x low/high tail subsets isolated by ``isolate_synthetic_data_preds.py``
(``data/negative_data/routes_{confident,uncertain}_{low,high}_scored.csv`` by
default -- each column-named ``synthesizability`` per that script's tiered
scoring), and draws two histogram panels side by side: the low-score pool on
the left, the high-score pool on the right, each hued by ensemble confidence
(confident vs. uncertain).

Usage:
    python scripts/negative_data/plot_negative_data.py \\
        --output-dir outputs/negative_data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

import pubstyle as ps

SCORE_COL = "synthesizability"
CONFIDENT_LABEL = "Confident"
UNCERTAIN_LABEL = "Uncertain"


def load_scores(path: Path, score_col: str = SCORE_COL) -> pd.Series:
    """Load one route-scored CSV and return its score column.

    Args:
        path: CSV written by route_tree_score.py (must contain score_col).
        score_col: Name of the synthesizability score column.

    Returns:
        The score column as a float Series, with missing values dropped.
    """
    df = pd.read_csv(path, usecols=[score_col])
    return df[score_col].dropna()


def plot_score_panel(
    ax: plt.Axes,
    confident: pd.Series,
    uncertain: pd.Series,
    bins: np.ndarray,
    xlim: tuple[float, float],
    xlabel: str,
    show_ylabel: bool = True,
) -> None:
    """Draw overlaid confident/uncertain histograms of synthesizability on one axes.

    Each series also gets a darker same-hue KDE curve (scaled to the count
    axis, see the styling-publication-figures skill's histogram recipe) and a
    dashed vertical rule at its mean, so the two rules are distinguishable by
    hue (which series) and by line style (KDE vs. mean) rather than needing a
    legend entry of their own.

    Args:
        ax: Target axes.
        confident: Scores from the confident pool.
        uncertain: Scores from the uncertain pool.
        bins: Shared bin edges, so the two panels stay comparable.
        xlim: Domain the KDE curves are evaluated over.
        xlabel: x-axis label for this panel (carries the low/high distinction
            since the figure has no title).
        show_ylabel: Whether to draw the "Count" y-axis label on this panel.
    """
    bin_width = bins[1] - bins[0]
    x = np.linspace(*xlim, 200)
    for values, color, label in (
        (confident, ps.CONTRAST[0], CONFIDENT_LABEL),
        (uncertain, ps.CONTRAST[1], UNCERTAIN_LABEL),
    ):
        ax.hist(values, bins=bins, range=xlim, color=color, edgecolor="white",
                 alpha=0.7, label=label, zorder=2)
        if len(values) > 10:
            kde = stats.gaussian_kde(values)
            ax.plot(x, kde(x) * len(values) * bin_width, color=ps.darken(color),
                     linewidth=2, zorder=5)
        ax.axvline(values.mean(), color=ps.darken(color), linestyle="--",
                   linewidth=1, alpha=0.9, zorder=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count" if show_ylabel else "")
    ax.grid(False)


def make_figure(
    confident_low: pd.Series,
    confident_high: pd.Series,
    uncertain_low: pd.Series,
    uncertain_high: pd.Series,
    n_bins: int,
) -> plt.Figure:
    """Build the two-panel (low score | high score) confidence-hued histogram figure.

    Args:
        confident_low: Confident-pool scores from the low-score tail.
        confident_high: Confident-pool scores from the high-score tail.
        uncertain_low: Uncertain-pool scores from the low-score tail.
        uncertain_high: Uncertain-pool scores from the high-score tail.
        n_bins: Number of bins spanning [0, 1], shared by both panels.

    Returns:
        The assembled figure, with one shared legend and no title.
    """
    ps.apply_style()
    xlim = (0.0, 1.0)
    bins = np.linspace(*xlim, n_bins + 1)

    # subplots=(1, 1.4) rather than the actual (1, 2) grid: set_size scales height
    # by nrows/ncols, and dividing by the true 2 columns flattens the figure to
    # ~1.7in tall -- too short for the bold axis labels and legend strip to read
    # at proportion. 1.4 lands between that and a full golden-ratio cell (3.4in),
    # wider/flatter as requested while still using the helper rather than a
    # by-eye figsize.
    fig, axes = plt.subplots(1, 2, figsize=ps.set_size(subplots=(1, 1.4)), layout="constrained")
    plot_score_panel(axes[0], confident_low, uncertain_low, bins, xlim, "Synthesizability (low-score pool)")
    plot_score_panel(axes[1], confident_high, uncertain_high, bins, xlim, "Synthesizability (high-score pool)",
                      show_ylabel=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    return fig


def parse_args() -> argparse.Namespace:
    """Parse CLI args for plotting the negative-data confidence/score histograms.

    Returns:
        argparse.Namespace with the parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--confident-low-csv", type=Path,
                     default=Path("data/negative_data/routes_confident_low_scored.csv"),
                     help="route_tree_score.py output for the confident, low-score tail")
    ap.add_argument("--confident-high-csv", type=Path,
                     default=Path("data/negative_data/routes_confident_high_scored.csv"),
                     help="route_tree_score.py output for the confident, high-score tail")
    ap.add_argument("--uncertain-low-csv", type=Path,
                     default=Path("data/negative_data/routes_uncertain_low_scored.csv"),
                     help="route_tree_score.py output for the uncertain, low-score tail")
    ap.add_argument("--uncertain-high-csv", type=Path,
                     default=Path("data/negative_data/routes_uncertain_high_scored.csv"),
                     help="route_tree_score.py output for the uncertain, high-score tail")
    ap.add_argument("--score-col", default=SCORE_COL,
                     help="synthesizability score column shared by all four CSVs")
    ap.add_argument("--n-bins", type=int, default=20, help="histogram bins spanning [0, 1]")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/negative_data"),
                     help="directory to save negative_data_scores.{pdf,svg,png} into")
    return ap.parse_args()


def main() -> None:
    """Parse CLI args, load the four score pools, and save the two-panel figure."""
    args = parse_args()

    print("Loading scores...")
    confident_low = load_scores(args.confident_low_csv, args.score_col)
    confident_high = load_scores(args.confident_high_csv, args.score_col)
    uncertain_low = load_scores(args.uncertain_low_csv, args.score_col)
    uncertain_high = load_scores(args.uncertain_high_csv, args.score_col)
    print(f"  confident: {len(confident_low):,} low, {len(confident_high):,} high")
    print(f"  uncertain: {len(uncertain_low):,} low, {len(uncertain_high):,} high")

    print("Plotting...")
    fig = make_figure(confident_low, confident_high, uncertain_low, uncertain_high, args.n_bins)

    out_path = args.output_dir / "negative_data_scores"
    written = ps.save_figure(fig, out_path)
    print(f"  wrote {', '.join(str(p) for p in written)}")


if __name__ == "__main__":
    main()
