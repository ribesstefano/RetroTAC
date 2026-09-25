"""
scripts/plotting_appendix/normality_diagnostics.py
==================================================
Normal Q-Q plots and the test-selection diagnostics behind the AutoRank
comparison, for the appendix.

The manuscript forces AutoRank's parametric branch (repeated-measures ANOVA +
Tukey HSD) whenever the fold-score variances are homogeneous, *regardless* of
the per-population normality diagnostic. That is a deliberate choice, so the
appendix has to show the diagnostics it overrides: one Q-Q panel per (model,
metric) over the 25 paired outer folds, annotated with Shapiro-Wilk, plus the
Levene statistic and max/min variance ratio that actually drive the branch.

Reads the per-fold metric tables that `train.py --aggregate` wrote
(`outputs/models/<run>/<run>_cv_metrics.csv`), i.e. exactly the 25
`score_seed*_fold*.json` rows the statistical comparison consumed.

Usage
-----
    python scripts/plotting_appendix/normality_diagnostics.py \\
        --runs xgb_20260828_182305 mlp_20260828_182305 gnn_20260828_182305

Outputs (see --out-dir / --fig-dir):
    outputs/appendix/normality_tests.csv
    outputs/appendix/variance_homogeneity.csv
    figures/appendix/qq_cv_metrics.{pdf,svg,png}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import pubstyle as ps

ps.apply_style()

#: Metrics carried through the comparison, in the order Figure 3a shows them,
#: with the display label and whether a larger value is better.
METRICS: Sequence[tuple] = (
    ("r2", "$R^2$", True),
    ("rmse", "RMSE", False),
    ("mae", "MAE", False),
    ("spearman_rho", "Spearman $\\rho$", True),
)

#: Variance-ratio cut above which the manuscript stops forcing the parametric
#: branch and falls back to AutoRank's own test selection (see Sec. 3.4).
VARIANCE_RATIO_CUT = 9.0


def load_runs(cv_root: Path, runs: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """Load one per-fold metric table per run.

    Args:
        cv_root: Directory holding `<run>/<run>_cv_metrics.csv`.
        runs: Run identifiers, e.g. `xgb_20260828_182305`.

    Returns:
        Display model name (upper-cased backend prefix) -> metric frame.
    """
    frames = {}
    for run in runs:
        path = cv_root / run / f"{run}_cv_metrics.csv"
        if not path.exists():
            raise SystemExit(f"missing per-fold metrics: {path}")
        frames[run.split("_")[0].upper()] = pd.read_csv(path)
    return frames


def normality_table(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Shapiro-Wilk and shape statistics per (model, metric).

    Args:
        frames: Model name -> per-fold metric frame.

    Returns:
        Long frame with one row per (model, metric).
    """
    rows = []
    for model, frame in frames.items():
        for column, label, _ in METRICS:
            values = frame[column].to_numpy(float)
            shapiro = stats.shapiro(values)
            rows.append({
                "model": model, "metric": label, "n": len(values),
                "mean": values.mean(), "std": values.std(ddof=1),
                "skew": float(stats.skew(values, bias=False)),
                "kurtosis": float(stats.kurtosis(values, bias=False)),
                "shapiro_W": float(shapiro.statistic),
                "shapiro_p": float(shapiro.pvalue),
            })
    return pd.DataFrame(rows)


def variance_table(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Levene's test and the max/min fold-variance ratio per metric.

    These two together decide the branch: the parametric procedure is forced
    unless Levene's p < 0.05 *and* the ratio exceeds `VARIANCE_RATIO_CUT`.

    Args:
        frames: Model name -> per-fold metric frame.

    Returns:
        One row per metric.
    """
    rows = []
    for column, label, _ in METRICS:
        samples = [frame[column].to_numpy(float) for frame in frames.values()]
        levene = stats.levene(*samples, center="median")
        variances = [float(np.var(s, ddof=1)) for s in samples]
        ratio = max(variances) / min(variances)
        forced = not (levene.pvalue < 0.05 and ratio > VARIANCE_RATIO_CUT)
        rows.append({
            "metric": label,
            "levene_W": float(levene.statistic), "levene_p": float(levene.pvalue),
            "var_ratio_max_min": ratio,
            "parametric_branch_forced": forced,
        })
    return pd.DataFrame(rows)


def plot_qq(frames: Dict[str, pd.DataFrame], normality: pd.DataFrame,
            out_stem: Path) -> List[Path]:
    """Grid of normal Q-Q plots, one row per model and one column per metric.

    Args:
        frames: Model name -> per-fold metric frame.
        normality: Table from `normality_table`, for the Shapiro-Wilk annotation.
        out_stem: Output path without extension.

    Returns:
        Paths written.
    """
    models = list(frames)
    width, _ = ps.set_size()
    fig, axes = plt.subplots(
        len(models), len(METRICS),
        figsize=(width, width * 0.26 * len(models)), layout="constrained")
    axes = np.atleast_2d(axes)

    # One limit for every panel, so the twelve are read against each other
    # rather than each against its own rescaled axis.
    lim = 2.6
    for frame in frames.values():
        for column, _, _ in METRICS:
            values = frame[column].to_numpy(float)
            z = (values - values.mean()) / values.std(ddof=1)
            lim = max(lim, float(np.abs(z).max()) * 1.12)

    lookup = normality.set_index(["model", "metric"])
    for row, model in enumerate(models):
        for col, (column, label, _) in enumerate(METRICS):
            ax = axes[row, col]
            values = frames[model][column].to_numpy(float)
            # Standardise so every panel shares one theoretical axis; the
            # reference line is then y = x and the panels are comparable.
            z = (values - values.mean()) / values.std(ddof=1)
            osm, osr = stats.probplot(z, dist="norm", fit=False)
            # A fold whose score is an outlier is the thing a reader is
            # looking for, so it is marked rather than left to the eye.
            extreme = np.abs(osr) > 2.0
            ax.scatter(osm[~extreme], osr[~extreme], s=6,
                       color=ps.PALETTE["blue"], zorder=3, linewidths=0, alpha=0.8)
            ax.scatter(osm[extreme], osr[extreme], s=10,
                       color=ps.PALETTE["dark_orange"], zorder=4, linewidths=0, alpha=0.8)
            ax.plot([-lim, lim], [-lim, lim], color=ps.STATS["reference_line"],
                    linestyle="--", linewidth=0.7, alpha=0.6, zorder=2)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_box_aspect(1)

            pvalue = float(lookup.loc[(model, label), "shapiro_p"])
            # Bold only where the diagnostic rejects: that is the case the
            # forced-parametric choice has to be defended against.
            ax.text(0.08, 0.87, f"$p$ = {pvalue:.3f}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=ps.ANNOT_FONTSIZE,
                    color=ps.darken(ps.PALETTE["dark_orange"], 0.7) if pvalue < 0.05 else "#555555",
                    fontweight="bold" if pvalue < 0.05 else "normal")
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"{model}\nstd. fold score")
            
            ax.tick_params(axis="both", which="major", labelsize=ps.ANNOT_FONTSIZE - 1)
            ax.grid(False)
    
    # One shared x label: four copies of it would not fit the text width
    # without the rightmost one running off the page.
    fig.supxlabel("Theoretical normal quantile", fontsize=ps.LABEL_FONTSIZE)
    return ps.save_figure(fig, out_stem)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cv-root", type=Path, default=Path("outputs/models"),
                   help="Directory holding <run>/<run>_cv_metrics.csv.")
    p.add_argument("--runs", nargs="+", default=[
        "xgb_20260828_182305", "mlp_20260828_182305", "gnn_20260828_182305"],
        help="Run identifiers, in the order the rows should appear.")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/appendix"),
                   help="Directory for the CSV tables.")
    p.add_argument("--fig-dir", type=Path, default=Path("figures/appendix"),
                   help="Directory for the figures.")
    return p.parse_args()


def main() -> None:
    """Write the normality/variance tables and the Q-Q grid."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    frames = load_runs(args.cv_root, args.runs)
    for model, frame in frames.items():
        print(f"{model}: {len(frame)} folds "
              f"({frame['seed'].nunique()} seeds x {frame['fold_idx'].nunique()} folds)")

    normality = normality_table(frames)
    normality.to_csv(args.out_dir / "normality_tests.csv", index=False)
    print("\n=== Shapiro-Wilk per (model, metric), 25 paired outer folds ===")
    print(normality.round(4).to_string(index=False))

    variance = variance_table(frames)
    variance.to_csv(args.out_dir / "variance_homogeneity.csv", index=False)
    print("\n=== Levene + variance ratio per metric (drives the test branch) ===")
    print(variance.round(4).to_string(index=False))

    written = plot_qq(frames, normality, args.fig_dir / "qq_cv_metrics")
    print("\nwrote:", *[str(p) for p in written], sep="\n  ")


if __name__ == "__main__":
    main()
