"""
plotting.py
========
Figures for the model comparison, reading the artifacts exported by
evaluation.py:
  - cv_fold_scores.pkl   -> per-model, per-metric list of fold values
                            (only the "r2" sub-dict is used, for boxplots)
  - cv_scores_long.csv   -> long format [method, seed, fold, cv_cycle, <every metric>]
                            (used for the simultaneous Tukey HSD CI grids)

Usage:
    python plotting.py --results data/outputs/results/comparison
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import autorank
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, spearmanr  # noqa: F401  (spearmanr handy if you extend)
from statsmodels.sandbox.stats.multicomp import MultiComparison

import pubstyle as ps

ps.apply_style()

# ── palette ─────────────────────────────────────────────────────────────────
# Backend -> hue, held fixed across every figure in this file. Per explicit
# request: XGB blue, GNN orange (the pubstyle contrast pair), MLP violet.
# NOTE: pubstyle reserves purple for "worse/significantly different" in the
# Tukey CI panels below -- MLP being violet here as well as (often) purple
# there is a deliberate departure from that reservation, not an oversight.
COLOR_MAP = {
    "XGB": ps.PALETTE["blue"],
    "GNN": ps.PALETTE["dark_orange"],
    "MLP": ps.PALETTE["purple"],
}
# A couple of ensemble strategies (see plot_ensemble_scatter) get their own
# deliberate accent in their single-panel scatter, keyed by the pretty name
# _pretty_strategy() produces. Caruana's green doubles as its usual "winner"
# meaning, since it is in fact the best strategy in ensemble_strategies.csv.
STRATEGY_COLOR_MAP = {
    "Best Single": ps.PALETTE["dark_orange"],
    "Caruana": ps.PALETTE["green"],
}
# Fallback for labels outside COLOR_MAP/STRATEGY_COLOR_MAP.
DEFAULT_COLOR = ps.PALETTE["blue"]

OUT_DIR = Path("figures")


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save a figure as pdf/svg/png under OUT_DIR.

    Args:
        fig: Figure to save.
        name: Base filename (without extension).
    """
    paths = ps.save_figure(fig, OUT_DIR / name)
    print(f"saved {paths[0]}")


# ── Fig 1: CV fold R2 boxplots (one subplot per model) ──────────────────────
def plot_cv_boxplots(fold_scores: Dict[str, List[float]], prefix: str = "") -> None:
    """Save one boxplot subplot per model, showing per-fold CV R2 scores.

    Args:
        fold_scores: Model label -> list of per-fold R2 scores.
        prefix: Optional filename prefix for the saved figure.
    """
    n     = len(fold_scores)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    # Not ps.set_size(subplots=...): its golden-ratio height is per *figure*,
    # so a single row of three panels comes out barely an inch tall and the
    # boxes flatten into lines. Size the row directly -- a box needs about
    # 1.35 in of plot height, plus a strip for the two-line title.
    fig_width_in = ps.set_size()[0]
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_width_in, 1.35 * nrows + 0.52 * nrows),
                             layout="constrained", squeeze=False)
    axes = axes.flatten()

    all_scores = [s for scores in fold_scores.values() for s in scores]
    pad  = (max(all_scores) - min(all_scores)) * 0.1
    ylim = (max(0.0, min(all_scores) - pad), min(1.0, max(all_scores) + pad))

    for i, (label, scores) in enumerate(fold_scores.items()):
        ax    = axes[i]
        color = next((v for k, v in COLOR_MAP.items() if k in label), DEFAULT_COLOR)
        edge  = ps.darken(color)
        bp = ax.boxplot([scores], positions=[1], patch_artist=True, widths=0.6,
                        medianprops=dict(color=edge, linewidth=1.2),
                        whiskerprops=dict(color=edge, linewidth=0.8),
                        capprops=dict(color=edge, linewidth=0.8),
                        flierprops=dict(marker="o", markersize=2,
                                        markerfacecolor=edge, alpha=0.4,
                                        markeredgecolor="none"))
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][0].set_edgecolor(edge)
        ax.scatter(np.ones(len(scores)), scores, color=edge, zorder=5, s=8, alpha=0.6)
        ax.set_ylim(ylim); ax.set_xlim(0.5, 1.5); ax.set_xticks([])
        ax.set_ylabel("R²")
        ax.set_title(f"{label}\nmean={np.mean(scores):.3f}  std={np.std(scores):.3f}",
                     fontsize=ps.TITLE_FONTSIZE, pad=3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    save_fig(fig, f"{prefix}cv_boxplots")


# Metrics where a LOWER value is better; every other metric (r2, spearman_rho,
# the clf_* rates/AUCs, ...) defaults to higher-is-better.
LOWER_IS_BETTER_METRICS = {"rmse", "mae", "medae", "max_error"}

#: Largest side, in inches, for one square parity panel. See plot_test_scatter.
MAX_PANEL_IN = 2.6

# metric column -> display name, for grid axis labels/titles
METRIC_DISPLAY_NAMES = {
    "r2": "R²", "rmse": "RMSE", "mae": "MAE", "spearman_rho": "Spearman ρ",
    "clf_roc_auc": "ROC-AUC", "clf_pr_auc": "PR-AUC", "clf_mcc": "MCC",
    "clf_recall": "Recall", "clf_precision": "Precision", "clf_f1": "F1",
    "clf_balanced_accuracy": "Balanced Accuracy", "clf_accuracy": "Accuracy",
}


# ── Fig 2: simultaneous Tukey HSD confidence intervals, one grid per metric ──
def _is_stat_color(color, target: str) -> bool:
    """True if `color` (str or RGBA array, as statsmodels hands back) is `target`."""
    if isinstance(color, str):
        return color == target
    if isinstance(color, np.ndarray):
        return np.allclose(color.flatten(), mcolors.to_rgba(target))
    return False


def recolor_tukey(ax: plt.Axes) -> None:
    """Swap statsmodels' hard-coded b/r/0.5 for the reserved stats palette.

    `plot_simultaneous` draws the reference (best) method's interval in blue,
    significantly different methods in red, and non-significant ones in grey
    0.5 -- remap those onto `ps.STATS['better']/['worse']/['nonsignificant']`
    so green/purple/grey keep their reserved, document-wide meaning.

    Args:
        ax: Axes just drawn by `plot_simultaneous`, recolored in place.
    """
    cmap = {"b": ps.STATS["better"], "r": ps.STATS["worse"], "0.5": ps.STATS["nonsignificant"]}
    for container in ax.containers:
        for child in container.get_children():
            cur = child.get_color() if hasattr(child, "get_color") else None
            for src, dst in cmap.items():
                if cur is not None and _is_stat_color(cur, src):
                    child.set_color(dst)
    for line in ax.get_lines():
        if line.get_linestyle() == "--" and _is_stat_color(line.get_color(), "0.7"):
            line.set_color(ps.STATS["reference_line"])
            line.set_linewidth(1.0)
            line.set_alpha(0.5)
        else:
            for src, dst in cmap.items():
                if _is_stat_color(line.get_color(), src):
                    line.set_color(dst)


def run_anova(df_in: pd.DataFrame, col: str, group_col: str = "method") -> float:
    """Run a one-way ANOVA on `col`, grouped by `group_col`.

    Args:
        df_in: Long-format frame holding `col` and `group_col`.
        col: Metric column to test.
        group_col: Name of the grouping column.

    Returns:
        The one-way ANOVA's (uncorrected) p-value for `group_col`.
    """
    groups = [g[col].values for _, g in df_in.groupby(group_col)]
    return float(f_oneway(*groups)[1])


def _compute_tukey_results(
    df_cv: pd.DataFrame, metrics: List[str], group_col: str = "method", alpha: float = 0.05,
) -> Dict[str, Tuple[object, str]]:
    """Run pairwise Tukey HSD per metric, ordered worst-to-best per group mean.

    Shared by make_simultaneous_ci_plot and make_fig3 so both draw from the
    same statistics. Worst-to-best group order (direction per
    LOWER_IS_BETTER_METRICS) makes `plot_simultaneous` read bottom-to-top
    toward the winner.

    Args:
        df_cv: Long-format CV scores frame with `group_col` and every column
            named in `metrics` (see `build_cv_frames` in evaluation.py).
        metrics: Metric columns to test. A metric missing from `df_cv` or
            containing any NaN (e.g. clf_roc_auc on a single-class fold) is
            skipped.
        group_col: Name of the grouping column.
        alpha: Family-wise significance threshold for the Tukey HSD intervals.

    Returns:
        metric -> (TukeyHSDResults, best_method), in the order given.
    """
    tukey_results = {}
    for metric in metrics:
        if metric not in df_cv.columns:
            print(f"_compute_tukey_results: {metric!r} not in df_cv -- skipping.")
            continue
        if df_cv[metric].isna().any():
            print(f"_compute_tukey_results: metric {metric!r} contains NaN values, skipping...")
            continue
        higher_is_better = metric not in LOWER_IS_BETTER_METRICS
        order = (df_cv.groupby(group_col)[metric].mean()
                .sort_values(ascending=higher_is_better).index.tolist())
        mc = MultiComparison(df_cv[metric], df_cv[group_col], group_order=order)
        tukey_results[metric] = (mc.tukeyhsd(alpha=alpha), order[-1])
    return tukey_results


def _draw_tukey_panel(
    ax: plt.Axes, metric: str, tukey_result, best_method: str,
    df_cv: pd.DataFrame, group_col: str = "method",
) -> None:
    """Draw one metric's simultaneous-CI panel into `ax` and recolor/label it.

    Args:
        ax: Target axes.
        metric: Metric column name (used for the omnibus ANOVA and the axis label).
        tukey_result: TukeyHSDResults for this metric, from _compute_tukey_results.
        best_method: The reference group plot_simultaneous compares every other to.
        df_cv: Long-format CV scores frame, for the omnibus ANOVA.
        group_col: Name of the grouping column.
    """
    # plot_simultaneous runs `fig.set_size_inches(figsize)` unconditionally --
    # even when handed an existing `ax`, so it silently resizes OUR figure to
    # its (10, 6) default. Restore the caller's size afterwards, or every
    # figure containing one of these panels renders at 10x6 regardless of
    # what ps.set_size computed (which also starves a fixed-aspect panel
    # sharing the figure -- see make_fig3).
    fig = ax.figure
    size_before = fig.get_size_inches().copy()
    tukey_result.plot_simultaneous(comparison_name=best_method, ax=ax)
    fig.set_size_inches(size_before)
    recolor_tukey(ax)
    p_omnibus = run_anova(df_cv, metric, group_col=group_col)
    ax.set_xlabel(METRIC_DISPLAY_NAMES.get(metric, metric.upper()), fontsize=ps.LABEL_FONTSIZE)
    ax.set_title(f"p = {p_omnibus:.2e}", fontsize=ps.TITLE_FONTSIZE)
    ax.tick_params(axis="y", labelsize=ps.TICK_FONTSIZE)
    # A metric whose groups differ in the third decimal (MAE here) gets
    # matplotlib's default ~8 ticks, and eight 5-character labels do not fit
    # across a half-width panel -- they overlap into one grey smear. Cap the
    # count instead of shrinking the type below everything else in the figure.
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))
    ax.grid(False)     # no horizontal rule at each model's tick -- just the CI bars


def make_simultaneous_ci_plot(
    df_cv: pd.DataFrame,
    metrics: List[str],
    group_col: str = "method",
    alpha: float = 0.05,
    prefix: str = "",
) -> None:
    """Save a grid of simultaneous Tukey HSD confidence-interval plots, one per metric.

    Uses statsmodels' `MultiComparison`/`tukeyhsd` (a one-way, independent-
    samples ANOVA -- not the repeated-measures/paired-fold design of the CV
    data) via _compute_tukey_results, and _draw_tukey_panel for each panel.
    Every group's CI is drawn relative to the metric's best group, and two
    groups' intervals overlapping means they are NOT significantly different
    -- this reads directly off the plot for every pair, not just against the
    best, unlike a vs-best-only comparison.

    Args:
        df_cv: Long-format CV scores frame with `group_col` and every column
            named in `metrics` (see `build_cv_frames` in evaluation.py).
        metrics: Metric columns to plot, one subplot each. A metric missing
            from `df_cv` or containing any NaN (e.g. clf_roc_auc on a
            single-class fold) is skipped.
        group_col: Name of the grouping column.
        alpha: Family-wise significance threshold for the Tukey HSD intervals.
        prefix: Optional filename prefix for the saved figure.
    """
    tukey_results = _compute_tukey_results(df_cv, metrics, group_col, alpha)
    if not tukey_results:
        print("make_simultaneous_ci_plot: no plottable metrics -- skipping.")
        return

    n     = len(tukey_results)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=ps.set_size(subplots=(nrows, ncols)),
                             layout="constrained", squeeze=False)
    axes = axes.flatten()

    for i, (metric, (tukey_result, best_method)) in enumerate(tukey_results.items()):
        _draw_tukey_panel(axes[i], metric, tukey_result, best_method, df_cv, group_col)
        print(f"  Best method for {metric}: {best_method}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    save_fig(fig, f"{prefix}simultaneous_ci_grid")


# ── Fig 3: held-out test set, predicted vs. measured (one panel per model) ──
def _draw_parity_panel(
    ax: plt.Axes, y_pred: np.ndarray, y_true: np.ndarray, color: str,
    thr: float, m: pd.Series, title: str, tick_fs: float = ps.TICK_FONTSIZE,
) -> None:
    """Draw one predicted-vs-measured parity panel into `ax`, with a metrics textbox.

    Shared by plot_test_scatter (a grid, one label per panel) and make_fig3
    (two of these panels in the row below the Tukey CI grid).

    Args:
        ax: Target axes.
        y_pred: Predicted values.
        y_true: Measured values.
        color: Marker color for this panel (see COLOR_MAP/STRATEGY_COLOR_MAP).
        thr: Classification threshold, for the reference cross-hairs.
        m: This label's row of metrics (mae, rmse, r2, spearman_rho,
            clf_precision, clf_recall, clf_roc_auc), e.g. metrics_df.loc[label].
        title: Panel title (the model or strategy name).
        tick_fs: Tick label font size.
    """
    ax.scatter(y_pred, y_true, color=color, alpha=0.35, s=5, edgecolor="none",
              rasterized=True, zorder=1)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8, zorder=2)
    ax.axhline(thr, color=ps.STATS["reference_line"], linestyle="--", linewidth=0.6,
              alpha=0.7, zorder=3)
    ax.axvline(thr, color=ps.STATS["reference_line"], linestyle="--", linewidth=0.6,
              alpha=0.7, zorder=3)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.25)); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.tick_params(labelsize=tick_fs)
    ax.set_title(title, fontsize=ps.TITLE_FONTSIZE, fontweight="bold")
    ax.grid(alpha=0.3)          # both axes here, unlike the y-only default
    ax.set_box_aspect(1)        # square: a parity plot must not be stretched

    text = (
        f"MAE: {m['mae']:.3f}\n"
        f"RMSE: {m['rmse']:.3f}\n"
        f"$R^2$: {m['r2']:.2f}\n"
        f"$\\rho$: {m['spearman_rho']:.2f}\n"
        f"Precision: {m['clf_precision']:.2f}\n"
        f"Recall: {m['clf_recall']:.2f}\n"
        f"AUC: {m['clf_roc_auc']:.2f}"
    )
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=ps.ANNOT_FONTSIZE, linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black",
                      linewidth=0.4, alpha=0.9))


def plot_test_scatter(
    predictions: Dict[str, Dict[str, np.ndarray]], metrics_df: pd.DataFrame, prefix: str = "",
) -> None:
    """Save one predicted-vs-measured scatter subplot per model, with a metrics textbox.

    Args:
        predictions: Model label -> {"y_pred": array, "y_true": array}, as
            written by evaluation.py's `evaluate_test` into test_predictions.pkl.
        metrics_df: Per-model metrics, indexed by label (e.g. test_metrics.csv
            loaded with `index_col=0`); must have the columns referenced in the
            textbox (mae, rmse, r2, spearman_rho, clf_precision, clf_recall,
            clf_roc_auc) and, if present, `clf_threshold` for the reference lines.
        prefix: Optional filename prefix for the saved figure.
    """
    labels = list(predictions.keys())
    n      = len(labels)
    ncols  = min(3, n)
    nrows  = int(np.ceil(n / ncols))
    # A multi-panel grid (the ncols>1 case, e.g. figures/test_scatter.pdf)
    # is tight on room for three panels' worth of ticks/labels -- shrink the
    # axis text there; a single reused panel (plot_ensemble_scatter) has
    # plenty of space and keeps the normal, larger sizing.
    compact       = ncols > 1
    tick_fs       = ps.TICK_FONTSIZE - 1 if compact else ps.TICK_FONTSIZE
    axis_label_fs = ps.LABEL_FONTSIZE
    # ps.set_size's golden-ratio height assumes rectangular cells; these
    # panels are forced square by set_box_aspect(1) below, so size per-panel
    # at aspect 1 instead. Overhead is a per-row title (scales with nrows)
    # plus one shared bottom label strip (fixed, not per-row) -- a flat
    # height multiplier over-grows for a single reused panel (see
    # plot_ensemble_scatter) and under-grows a tall multi-row grid.
    fig_width_in, _ = ps.set_size(fraction=1.0)
    # The squares share one y-label strip on the left, so the width they
    # actually divide up is the figure minus that strip -- sizing off the full
    # width over-estimates the square and leaves a band of dead space under
    # the row (the page is saved at exactly figsize, so nothing crops it).
    # MAX_PANEL_IN keeps the one-panel case (plot_ensemble_scatter) from
    # swelling into a 5 in square: every figure in the project is authored at
    # \linewidth so that it can be imported at 100%, and a lone panel is
    # centered in that width rather than stretched to fill it.
    panel_in      = min((fig_width_in - 0.55) / ncols, MAX_PANEL_IN)
    fig_height_in = panel_in * nrows + 0.30 * nrows + 0.45
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_width_in, fig_height_in),
                             layout="constrained", squeeze=False,
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i, label in enumerate(labels):
        ax     = axes[i]
        y_true = np.asarray(predictions[label]["y_true"], dtype=float).ravel()
        y_pred = np.asarray(predictions[label]["y_pred"], dtype=float).ravel()
        color  = COLOR_MAP.get(label, STRATEGY_COLOR_MAP.get(label, DEFAULT_COLOR))
        m      = metrics_df.loc[label]
        thr    = float(m["clf_threshold"]) if "clf_threshold" in metrics_df.columns else 0.7

        _draw_parity_panel(ax, y_pred, y_true, color, thr, m, title=label, tick_fs=tick_fs)
        if i % ncols != 0:                        # shared x/y scale: axis labels live once,
            ax.tick_params(labelleft=False)        # at the figure level, not per panel

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    # One shared axis-label pair for the whole grid: every panel is the same
    # x/y quantity for a different model (named by its own title), so a
    # label per panel is repeated ink that also overflows a narrow column.
    fig.supxlabel("Predicted synthesizability", fontsize=axis_label_fs, fontweight="bold")
    fig.supylabel("Measured synthesizability", fontsize=axis_label_fs, fontweight="bold")
    save_fig(fig, f"{prefix}test_scatter")


# ── Fig 4: ensemble strategies (best_single / uniform / best_n / Caruana) ───
def _pretty_strategy(name: str) -> str:
    """Display label for a strategy key, e.g. 'best_5' -> 'Best-5', 'best_single' -> 'Best Single'."""
    tail = name.split("_")[-1]
    if name.startswith("best_") and tail.isdigit():
        return f"Best-{tail}"
    return name.replace("_", " ").title()


def plot_ensemble_strategies(results_dir: Path, prefix: str = "") -> None:
    """Save a 3-panel figure comparing ensemble strategies: RMSE, R², and backend composition.

    Reads the artifacts evaluation.py's `--ensemble` writes: `ensemble_strategies.csv`
    (one row per strategy, from `compute_all_metrics` plus `n_models`) and
    `ensemble_weights_{strategy}.json` (per-model weights, for the composition
    panel -- which backend's fold models each strategy actually selected).

    Args:
        results_dir: Directory holding `ensemble_strategies.csv` and
            `ensemble_weights_*.json` (an evaluation.py `--out` folder).
        prefix: Optional filename prefix for the saved figure.
    """
    csv_path = results_dir / "ensemble_strategies.csv"
    if not csv_path.exists():
        print(f"No {csv_path} -- skipping ensemble-strategy plot "
              f"(run evaluation.py with --ensemble first).")
        return

    df = pd.read_csv(csv_path, index_col=0)
    weights: Dict[str, Dict[str, float]] = {}
    for name in df.index:
        p = results_dir / f"ensemble_weights_{name}.json"
        if p.exists():
            with open(p) as f:
                weights[name] = json.load(f)["weights"]

    labels = [_pretty_strategy(s) for s in df.index]
    best   = df["rmse"].idxmin()
    # Best-vs-rest is a statistical outcome (winner), not a data series --
    # the reserved green/grey pair is the right one here.
    bar_colors = [ps.STATS["better"] if s == best else ps.STATS["nonsignificant"] for s in df.index]

    fig_w, _  = ps.set_size(fraction=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, max(2.0, len(df) * 0.42 + 0.95)),
                             layout="constrained", sharey=True)
    # constrained_layout reserves height for a title but not width: a title
    # centered on the rightmost axes simply overhangs the canvas, and with no
    # tight bbox to grow into it is clipped. Inset the layout a little so the
    # overhang has somewhere to go, and keep the strings short below.
    fig.get_layout_engine().set(rect=(0.004, 0, 0.992, 1))

    ax = axes[0]
    bars = ax.barh(labels, df["rmse"], color=bar_colors, edgecolor="white")
    for bar, rmse, n_models in zip(bars, df["rmse"], df["n_models"]):
        ax.text(rmse, bar.get_y() + bar.get_height() / 2, f"  {rmse:.4f} (n={int(n_models)})",
                va="center", fontsize=ps.ANNOT_FONTSIZE)
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Test RMSE", fontsize=ps.TITLE_FONTSIZE)

    ax = axes[1]
    bars = ax.barh(labels, df["r2"], color=bar_colors, edgecolor="white")
    for bar, r2 in zip(bars, df["r2"]):
        ax.text(r2, bar.get_y() + bar.get_height() / 2, f"  {r2:.3f}", va="center", fontsize=ps.ANNOT_FONTSIZE)
    ax.set_xlabel("R² (higher is better)")
    ax.set_title("Test R²", fontsize=ps.TITLE_FONTSIZE)

    ax = axes[2]
    backends = sorted({k.split("_seed")[0] for w in weights.values() for k in w})
    bottoms  = np.zeros(len(df))
    for backend in backends:
        counts = np.array([sum(1 for k in weights.get(s, {}) if k.split("_seed")[0] == backend)
                           for s in df.index])
        ax.barh(labels, counts, left=bottoms, color=COLOR_MAP.get(backend, DEFAULT_COLOR),
                edgecolor="white", label=backend)
        bottoms += counts
    ax.set_xlabel("Fold models")
    ax.set_title("Composition", fontsize=ps.TITLE_FONTSIZE)
    ax.legend(fontsize=ps.LEGEND_FONTSIZE, loc="lower right")
    # Once, not per panel: the three share a y axis, so each call would flip
    # the whole row back over. Best strategy first, reading top to bottom.
    axes[0].invert_yaxis()

    fig.suptitle("Ensemble Strategies", fontsize=ps.PANEL_LABEL_FONTSIZE,
                 fontweight="bold")
    save_fig(fig, f"{prefix}ensemble_strategies")


# name evaluation.py's --ensemble caches its per-fold-model raw predictions
# under (see its own FOLD_PRED_FILENAME); duplicated here since these are two
# standalone sibling scripts with no shared import.
FOLD_PRED_FILENAME = "cv_fold_test_predictions.pkl"


def _reconstruct_ensemble_prediction(
    results_dir: Path, weights: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Rebuild one ensemble strategy's prediction from its saved fold weights.

    evaluation.py's --ensemble scores each strategy on a held-out eval split
    but never persists the strategy's own y_pred array -- only its fold
    weights (ensemble_weights_{strategy}.json) and every individual fold
    model's raw prediction over the FULL test set (cv_fold_test_predictions.pkl).
    Re-weighting those raw predictions reconstructs the strategy's prediction
    over the whole test set (weights already sum to 1, see save_ensemble_weights).

    Args:
        results_dir: Directory holding cv_fold_test_predictions.pkl (an
            evaluation.py --ensemble --out folder).
        weights: fold_model_key -> weight, as saved in
            ensemble_weights_{strategy}.json's "weights" field.

    Returns:
        Tuple (y_pred, y_true) over the full test set.
    """
    with open(results_dir / FOLD_PRED_FILENAME, "rb") as f:
        cached = pickle.load(f)
    fold_predictions, y_true = cached["fold_predictions"], cached["y_true"]
    y_pred = sum(w * fold_predictions[k] for k, w in weights.items())
    return np.asarray(y_pred, dtype=float), np.asarray(y_true, dtype=float)


def plot_ensemble_scatter(results_dir: Path, prefix: str = "") -> None:
    """Save one predicted-vs-measured scatter plot per ensemble strategy.

    Unlike plot_test_scatter (one grid, every model as a subplot in a single
    file), each strategy here gets its own PNG/PDF pair, via a 1-label call
    to plot_test_scatter per strategy -- reusing its scatter/metrics-textbox
    drawing rather than duplicating it.

    Args:
        results_dir: Directory holding ensemble_strategies.csv,
            ensemble_weights_*.json, and cv_fold_test_predictions.pkl (an
            evaluation.py --ensemble --out folder).
        prefix: Optional filename prefix shared by every saved figure.
    """
    csv_path = results_dir / "ensemble_strategies.csv"
    fold_pred_path = results_dir / FOLD_PRED_FILENAME
    if not csv_path.exists() or not fold_pred_path.exists():
        print(f"No {csv_path} / {fold_pred_path} -- skipping per-strategy ensemble "
              f"scatter plots (run evaluation.py with --ensemble first).")
        return

    metrics_df = pd.read_csv(csv_path, index_col=0)
    for name in metrics_df.index:
        weights_path = results_dir / f"ensemble_weights_{name}.json"
        if not weights_path.exists():
            print(f"No {weights_path} -- skipping scatter for strategy {name!r} "
                  f"(only the best_single/uniform/best_backend/caruana strategies "
                  f"have saved fold weights, not e.g. an --evidential-model row).")
            continue
        with open(weights_path) as f:
            weights = json.load(f)["weights"]
        y_pred, y_true = _reconstruct_ensemble_prediction(results_dir, weights)

        label = _pretty_strategy(name)
        predictions = {label: {"y_pred": y_pred, "y_true": y_true}}
        row = metrics_df.loc[[name]].rename(index={name: label})
        plot_test_scatter(predictions, row, prefix=f"{prefix}{name}_")


# ── Fig 3 (combined): regression Tukey CI grid + two ensemble parity plots ──
def make_fig3(df_cv: pd.DataFrame, results_dir: Path, prefix: str = "") -> None:
    """Composite two-block figure: (a) regression Tukey CI grid, (b) Best Single & Caruana parity plots.

    Combines make_simultaneous_ci_plot's regression metrics (r2, rmse, mae,
    spearman_rho) with two of plot_ensemble_scatter's per-strategy panels
    into one lettered figure, reusing the same underlying computations
    (_compute_tukey_results/_draw_tukey_panel,
    _reconstruct_ensemble_prediction/_draw_parity_panel) rather than
    duplicating them.

    Args:
        df_cv: Long-format CV scores frame (see make_simultaneous_ci_plot).
        results_dir: Directory holding ensemble_strategies.csv,
            ensemble_weights_*.json, and cv_fold_test_predictions.pkl (an
            evaluation.py --ensemble --out folder).
        prefix: Optional filename prefix for the saved figure.
    """
    reg_metrics   = ["r2", "rmse", "mae", "spearman_rho"]
    tukey_results = _compute_tukey_results(df_cv, reg_metrics)
    if not tukey_results:
        print("make_fig3: no plottable CI metrics -- skipping.")
        return

    csv_path = results_dir / "ensemble_strategies.csv"
    fold_pred_path = results_dir / FOLD_PRED_FILENAME
    if not csv_path.exists() or not fold_pred_path.exists():
        print(f"make_fig3: no {csv_path} / {fold_pred_path} -- skipping "
              f"(run evaluation.py with --ensemble first).")
        return
    metrics_df = pd.read_csv(csv_path, index_col=0)

    panels_b = []   # (title, y_pred, y_true, metrics_row, threshold, color)
    for name, title in (("best_single", "Best Single"), ("caruana", "Caruana")):
        weights_path = results_dir / f"ensemble_weights_{name}.json"
        if not weights_path.exists() or name not in metrics_df.index:
            print(f"make_fig3: no {weights_path} / row -- skipping the {title!r} panel.")
            continue
        with open(weights_path) as f:
            weights = json.load(f)["weights"]
        y_pred, y_true = _reconstruct_ensemble_prediction(results_dir, weights)
        m   = metrics_df.loc[name]
        thr = float(m["clf_threshold"]) if "clf_threshold" in metrics_df.columns else 0.7
        panels_b.append((title, y_pred, y_true, m, thr, STRATEGY_COLOR_MAP.get(title, DEFAULT_COLOR)))

    if not panels_b:
        print("make_fig3: no ensemble-scatter panels available -- skipping.")
        return

    # Two stacked blocks, not two side-by-side ones. The figure is written at
    # the document's \linewidth and imported at 100% (see pubstyle.set_size),
    # so width is a hard budget: six panels in one row would each get under an
    # inch. Stacking also gives both blocks their natural aspect -- a CI panel
    # with three groups wants to be wide and short, a parity plot wants to be
    # square -- instead of splitting the difference.
    fig_width_in = ps.set_size()[0]
    # (a): two rows of CI panels. Each needs room for three group rows, an
    # x label and the p-value title; below ~1.2 in per row the group labels
    # start colliding with the intervals.
    height_a = 1.16 * int(np.ceil(len(tukey_results) / 2))
    # (b): the squares are set_box_aspect(1) and share the full width, so each
    # is about (width - the left labels) / n wide; give the row that much
    # height plus a strip for the tick labels, the title and the shared x
    # label, or the squares get height-limited and leave dead space either
    # side of the row.
    height_b = (fig_width_in - 0.55) / len(panels_b) + 0.62
    # Headroom for the (a) letter. It is placed with in_layout=False (see
    # below), so constrained_layout does not reserve for it -- and the page is
    # written at exactly figsize, with no tight bbox to rescue anything that
    # lands outside. Without this strip the letter is sliced off at the top.
    letter_pad_in = 0.16
    fig_height_in = height_a + height_b + letter_pad_in

    fig = plt.figure(figsize=(fig_width_in, fig_height_in), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - letter_pad_in / fig_height_in))
    gs  = fig.add_gridspec(2, 1, height_ratios=[height_a, height_b], hspace=0.06)

    gs_a   = gs[0].subgridspec(2, 2)
    axes_a = [fig.add_subplot(gs_a[i // 2, i % 2]) for i in range(len(tukey_results))]
    for ax, (metric, (tukey_result, best_method)) in zip(axes_a, tukey_results.items()):
        _draw_tukey_panel(ax, metric, tukey_result, best_method, df_cv)

    gs_b   = gs[1].subgridspec(1, len(panels_b))
    axes_b = [fig.add_subplot(gs_b[0, i]) for i in range(len(panels_b))]
    for j, (ax, (title, y_pred, y_true, m, thr, color)) in enumerate(zip(axes_b, panels_b)):
        _draw_parity_panel(ax, y_pred, y_true, color, thr, m, title=title)
        if j == 0:
            ax.set_ylabel("Measured synthesizability")
        else:
            ax.tick_params(labelleft=False)
    # Both (b) panels show the same quantity on x, so the label belongs to the
    # row, not to each panel. (b) is the bottom row and spans the full width,
    # so the figure-level label lands centered under it -- and constrained
    # layout reserves its strip, unlike a label placed after the fact.
    fig.supxlabel("Predicted synthesizability", fontsize=ps.LABEL_FONTSIZE,
                  fontweight="bold")

    # Panel letters aligned on one left margin even though (a)'s and (b)'s
    # axes start at different x (their y tick labels differ in width): blend
    # the transform, taking x from the FIGURE and y from each block's own top
    # axes. in_layout=False because the letter sits outside its axes, and a
    # reserved-for decoration there would push the whole block inward.
    for ax, letter in ((axes_a[0], "(a)"), (axes_b[0], "(b)")):
        trans = mtransforms.blended_transform_factory(fig.transFigure, ax.transAxes)
        # Lifted a few points clear of the axes: in (b) the topmost y tick
        # label sits exactly at the axes top, and a letter on that line runs
        # into it. The offset puts both letters on their block's title line.
        trans = mtransforms.offset_copy(trans, fig=fig, y=6, units="points")
        ax.text(0.008, 1.0, letter, transform=trans, fontsize=ps.PANEL_LABEL_FONTSIZE,
                fontweight="bold", va="bottom", ha="left", clip_on=False,
                in_layout=False)

    save_fig(fig, f"{prefix}fig_3")


def main() -> None:
    """CLI entry point: load exported CV artifacts and save the comparison figures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/outputs/results/comparison",
                    help="folder with cv_fold_scores.pkl and cv_scores_long.csv")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"),
                    help="directory to save figures into (default: figures)")
    args = ap.parse_args()
    res = Path(args.results)

    global OUT_DIR
    OUT_DIR = args.out_dir

    fold_metrics = pickle.load(open(res / "cv_fold_scores.pkl", "rb"))   # {model: {metric: [values]}}
    fold_scores  = {label: metrics["r2"] for label, metrics in fold_metrics.items()}
    df_cv        = pd.read_csv(res / "cv_scores_long.csv")
    wide         = pd.read_csv(res / "cv_scores_wide.csv", header=[0, 1], index_col=[0, 1])   # (seed, fold) index

    # sanity: each model must contribute the same number of folds (balanced comparison)
    counts = df_cv.groupby("method")["r2"].count()
    print("folds per model:\n", counts.to_string())
    if counts.nunique() != 1:
        print("WARNING: unbalanced folds — the ANOVA needs equal counts per model.")

    plot_cv_boxplots(fold_scores)
    make_simultaneous_ci_plot(df_cv, prefix="reg_", metrics=["r2", "rmse", "mae", "spearman_rho"])
    make_simultaneous_ci_plot(df_cv, prefix="clf_", metrics=["clf_roc_auc", "clf_pr_auc", "clf_mcc", "clf_recall"])

    test_pred_path, test_metrics_path = res / "test_predictions.pkl", res / "test_metrics.csv"
    if test_pred_path.exists() and test_metrics_path.exists():
        with open(test_pred_path, "rb") as f:
            predictions = pickle.load(f)
        metrics_df = pd.read_csv(test_metrics_path, index_col=0)
        plot_test_scatter(predictions, metrics_df)
    else:
        print("No test_predictions.pkl/test_metrics.csv in --results -- "
              "skipping test-set scatter plots (run evaluation.py with --test-csv first).")

    plot_ensemble_strategies(res)
    plot_ensemble_scatter(res)
    make_fig3(df_cv, res)


if __name__ == "__main__":
    main()