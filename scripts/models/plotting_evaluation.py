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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, spearmanr  # noqa: F401  (spearmanr handy if you extend)
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ── palette ─────────────────────────────────────────────────────────────────
ROYAL_PURPLE = "#6A4C93"
DARK_SLATE   = "#2F4F4F"
DEEP_TEAL    = "#1B7F79"
FOREST_GREEN = "#2E7D32"
SIG_RED      = "#CC3333"

COLOR_MAP = {"XGB": DEEP_TEAL, "MLP": FOREST_GREEN, "GNN": ROYAL_PURPLE}

OUT_DIR = Path("figures")


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save a figure as both PNG and PDF under OUT_DIR.

    Args:
        fig: Figure to save.
        name: Base filename (without extension).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {OUT_DIR / f'{name}.png'}")


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
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             gridspec_kw={"hspace": 0.5, "wspace": 0.4},
                             squeeze=False)
    axes = axes.flatten()

    all_scores = [s for scores in fold_scores.values() for s in scores]
    pad  = (max(all_scores) - min(all_scores)) * 0.1
    ylim = (max(0.0, min(all_scores) - pad), min(1.0, max(all_scores) + pad))

    for i, (label, scores) in enumerate(fold_scores.items()):
        ax    = axes[i]
        color = next((v for k, v in COLOR_MAP.items() if k in label), DARK_SLATE)
        bp = ax.boxplot([scores], positions=[1], patch_artist=True, widths=0.5,
                        medianprops=dict(color=DARK_SLATE, linewidth=1.8),
                        whiskerprops=dict(color=DARK_SLATE, linewidth=1.2),
                        capprops=dict(color=DARK_SLATE, linewidth=1.2),
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor=DARK_SLATE, alpha=0.4,
                                        markeredgecolor="none"))
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][0].set_edgecolor(DARK_SLATE)
        ax.scatter(np.ones(len(scores)), scores, color=DARK_SLATE, zorder=5, s=20, alpha=0.6)
        ax.set_ylim(ylim); ax.set_xlim(0.4, 1.6); ax.set_xticks([])
        ax.set_ylabel("R²")
        ax.set_title(f"{label}\nmean={np.mean(scores):.3f}  std={np.std(scores):.3f}",
                     fontsize=9, pad=3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    save_fig(fig, f"{prefix}cv_boxplots")


# Metrics where a LOWER value is better; every other metric (r2, spearman_rho,
# the clf_* rates/AUCs, ...) defaults to higher-is-better.
LOWER_IS_BETTER_METRICS = {"rmse", "mae", "medae", "max_error"}

# metric column -> display name, for grid axis labels/titles
METRIC_DISPLAY_NAMES = {
    "r2": "R²", "rmse": "RMSE", "mae": "MAE", "spearman_rho": "Spearman ρ",
    "clf_roc_auc": "ROC-AUC", "clf_pr_auc": "PR-AUC", "clf_mcc": "MCC",
    "clf_recall": "Recall", "clf_precision": "Precision", "clf_f1": "F1",
    "clf_balanced_accuracy": "Balanced Accuracy", "clf_accuracy": "Accuracy",
}


# ── Fig 2: simultaneous Tukey HSD confidence intervals, one grid per metric ──
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


def make_simultaneous_ci_plot(
    df_cv: pd.DataFrame,
    metrics: List[str],
    group_col: str = "method",
    alpha: float = 0.05,
    prefix: str = "",
) -> None:
    """Save a grid of simultaneous Tukey HSD confidence-interval plots, one per metric.

    Uses statsmodels' `pairwise_tukeyhsd` (a one-way, independent-samples
    ANOVA -- not the repeated-measures/paired-fold design of the CV data) and
    its own `plot_simultaneous`: every group's CI is drawn relative to the
    metric's best group (direction per LOWER_IS_BETTER_METRICS), and two
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
    tukey_results = {}
    for metric in metrics:
        if metric not in df_cv.columns:
            print(f"make_simultaneous_ci_plot: {metric!r} not in df_cv -- skipping.")
            continue
        if df_cv[metric].isna().any():
            print(f"make_simultaneous_ci_plot: metric {metric!r} contains NaN values, skipping...")
            continue
        tukey_results[metric] = pairwise_tukeyhsd(
            endog=df_cv[metric], groups=df_cv[group_col], alpha=alpha)

    if not tukey_results:
        print("make_simultaneous_ci_plot: no plottable metrics -- skipping.")
        return

    fig, axes = plt.subplots(1, len(tukey_results), figsize=(7 * len(tukey_results), 5), squeeze=False)
    axes = axes[0]

    for ax, (metric, tukey_result) in zip(axes, tukey_results.items()):
        higher_is_better = metric not in LOWER_IS_BETTER_METRICS
        best_method = (df_cv.groupby(group_col)[metric].mean()
                      .sort_values(ascending=not higher_is_better).index[0])
        tukey_result.plot_simultaneous(comparison_name=best_method, ax=ax)
        p_omnibus = run_anova(df_cv, metric, group_col=group_col)
        ax.set_xlabel(METRIC_DISPLAY_NAMES.get(metric, metric.upper()), fontsize=11)
        ax.set_title(f"p = {p_omnibus:.2e}", fontsize=11)
        print(f"  Best method for {metric}: {best_method}")

    for ax in axes:
        ax.tick_params(axis="y", labelsize=10)
    fig.suptitle(f"Simultaneous Confidence Intervals\nTukey HSD, FWER={alpha}",
                fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save_fig(fig, f"{prefix}simultaneous_ci_grid")


# ── Fig 3: held-out test set, predicted vs. measured (one panel per model) ──
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.2 * nrows),
                             gridspec_kw={"hspace": 0.4, "wspace": 0.35}, squeeze=False)
    axes = axes.flatten()

    for i, label in enumerate(labels):
        ax     = axes[i]
        y_true = np.asarray(predictions[label]["y_true"], dtype=float).ravel()
        y_pred = np.asarray(predictions[label]["y_pred"], dtype=float).ravel()
        color  = COLOR_MAP.get(label, DARK_SLATE)

        ax.scatter(y_pred, y_true, color=color, alpha=0.35, s=14, edgecolor="none")
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, zorder=1)

        m = metrics_df.loc[label]
        thr = float(m["clf_threshold"]) if "clf_threshold" in metrics_df.columns else 0.7
        ax.axhline(thr, color=SIG_RED, linestyle="--", linewidth=0.9, alpha=0.8)
        ax.axvline(thr, color=SIG_RED, linestyle="--", linewidth=0.9, alpha=0.8)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted synthesizability")
        ax.set_ylabel("Measured synthesizability")
        ax.set_title(label, fontsize=11, fontweight="bold")

        text = (
            f"MAE: {m['mae']:.3f}\n"
            f"RMSE: {m['rmse']:.3f}\n"
            f"$R^2$: {m['r2']:.2f}\n"
            f"$\\rho$: {m['spearman_rho']:.2f}\n"
            f"Precision: {m['clf_precision']:.2f}\n"
            f"Recall: {m['clf_recall']:.2f}\n"
            f"AUC: {m['clf_roc_auc']:.2f}"
        )
        ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor=DARK_SLATE, alpha=0.9))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
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

    labels    = [_pretty_strategy(s) for s in df.index]
    best      = df["rmse"].idxmin()
    bar_colors = [ROYAL_PURPLE if s == best else DARK_SLATE for s in df.index]

    fig, axes = plt.subplots(1, 3, figsize=(15, max(3, len(df) * 0.7 + 1)), gridspec_kw={"wspace": 0.55})

    ax = axes[0]
    bars = ax.barh(labels, df["rmse"], color=bar_colors, alpha=0.85)
    for bar, rmse, n_models in zip(bars, df["rmse"], df["n_models"]):
        ax.text(rmse, bar.get_y() + bar.get_height() / 2, f"  {rmse:.4f} (n={int(n_models)})",
                va="center", fontsize=8)
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Test RMSE", fontsize=10)
    ax.invert_yaxis()

    ax = axes[1]
    bars = ax.barh(labels, df["r2"], color=bar_colors, alpha=0.85)
    for bar, r2 in zip(bars, df["r2"]):
        ax.text(r2, bar.get_y() + bar.get_height() / 2, f"  {r2:.3f}", va="center", fontsize=8)
    ax.set_xlabel("R² (higher is better)")
    ax.set_title("Test R²", fontsize=10)
    ax.invert_yaxis()

    ax = axes[2]
    backends = sorted({k.split("_seed")[0] for w in weights.values() for k in w})
    bottoms  = np.zeros(len(df))
    for backend in backends:
        counts = np.array([sum(1 for k in weights.get(s, {}) if k.split("_seed")[0] == backend)
                           for s in df.index])
        ax.barh(labels, counts, left=bottoms, color=COLOR_MAP.get(backend, DARK_SLATE),
                alpha=0.85, label=backend)
        bottoms += counts
    ax.set_xlabel("Fold models in ensemble")
    ax.set_title("Composition by backend", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()

    fig.suptitle("Ensemble Strategies", fontsize=12, fontweight="bold")
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


if __name__ == "__main__":
    main()