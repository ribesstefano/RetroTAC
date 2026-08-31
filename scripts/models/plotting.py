"""
plotting.py
========
Figures for the model comparison, reading the artifacts exported by
evaluation.py:
  - cv_fold_scores.pkl   -> per-model, per-metric list of fold values
                            (only the "r2" sub-dict is used, for boxplots)
  - cv_scores_long.csv   -> long format [method, seed, fold, cv_cycle, <every metric>]
                            (only "r2" is used here, for Tukey)

Usage:
    python plotting.py --results data/outputs/results/comparison
"""
import argparse
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import autorank
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from matplotlib.lines import Line2D
from scipy.stats import spearmanr  # noqa: F401  (handy if you extend)
from statsmodels.stats.libqsturng import psturng, qsturng

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
    print(f"saved figures/{name}.png")


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


# ── Fig 2: repeated-measures Tukey HSD, mean ± CI ───────────────────────────
def rm_tukey_hsd(df: pd.DataFrame, metric: str, group_col: str, alpha: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run a repeated-measures ANOVA + Tukey HSD post-hoc pairwise comparison.

    Args:
        df: Long-format frame with `metric`, `group_col`, and a `cv_cycle`
            paired-sample id column (see `build_cv_frames` in
            models_evaluation.py).
        metric: Name of the column to compare (e.g. "r2").
        group_col: Name of the grouping column (e.g. "method").
        alpha: Unused; kept for interface symmetry with `plot_multiple_comparisons`.

    Returns:
        Tuple of (df_means, pc, p_omnibus):
            df_means: Per-group mean of `metric`, indexed by `group_col`.
            pc: Symmetric matrix of Tukey-adjusted pairwise p-values, indexed
                and columned by group label.
            p_omnibus: The repeated-measures ANOVA's overall (uncorrected)
                p-value for `group_col`.
    """
    df_means = df.groupby(group_col).mean(numeric_only=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        aov = pg.rm_anova(dv=metric, within=group_col, subject="cv_cycle",
                          data=df, detailed=True)
    mse, df_resid = aov.loc[1, "MS"], aov.loc[1, "DF"]
    p_omnibus = float(aov.loc[0, "p_unc"])
    methods = df_means.index
    n_per   = df[group_col].value_counts().mean()
    tukey_se = np.sqrt(2 * mse / n_per)

    pc = pd.DataFrame(index=methods, columns=methods, data=1.0)
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j:
                diff  = df[df[group_col] == m1][metric].mean() - df[df[group_col] == m2][metric].mean()
                sr    = np.abs(diff) / tukey_se
                adj_p = psturng(sr * np.sqrt(2), len(methods), df_resid)
                adj_p = adj_p[0] if isinstance(adj_p, np.ndarray) else adj_p
                pc.loc[m1, m2] = pc.loc[m2, m1] = adj_p
    return df_means, pc.astype(float), p_omnibus


def _draw_comparison_panel(ax: plt.Axes, df_cv: pd.DataFrame, metric: str, alpha: float = 0.05, xlabel: str = None) -> None:
    """Draw one mean ± 95% CI Tukey-HSD comparison panel onto `ax`.

    Shared by `plot_multiple_comparisons` (single metric, its own figure) and
    `plot_multiple_comparisons_grid` (many metrics, one subplot each).

    Args:
        ax: Axes to draw into.
        df_cv: Long-format CV scores frame (see `rm_tukey_hsd`).
        metric: Name of the column to compare.
        alpha: Significance threshold for marking a method "significantly worse".
        xlabel: X-axis label; defaults to `f"Mean CV {metric.upper()}"`.
    """
    df_means, pc, p_omnibus = rm_tukey_hsd(df_cv, metric, group_col="method", alpha=alpha)
    df_means = df_means.sort_values(metric, ascending=True)
    labels = df_means.index.tolist()
    means  = df_means[metric].values
    best   = labels[-1]

    ci_low, ci_high = [], []
    for label in labels:
        s  = df_cv[df_cv["method"] == label][metric].values
        se = s.std(ddof=1) / np.sqrt(len(s))
        ci_low.append(s.mean() - 1.96 * se)
        ci_high.append(s.mean() + 1.96 * se)

    sig_worse = [l for l in labels if l != best and pc.loc[best, l] < alpha]
    colors = [ROYAL_PURPLE if l == best else SIG_RED if l in sig_worse else DARK_SLATE
              for l in labels]

    for i, (m, lo, hi, c) in enumerate(zip(means, ci_low, ci_high, colors)):
        ax.plot([lo, hi], [i, i], color=c, linewidth=2.5)
        ax.plot(m, i, "o", color=c, markersize=6)
    bi = labels.index(best)
    ax.axvline(ci_low[bi],  color=ROYAL_PURPLE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(ci_high[bi], color=ROYAL_PURPLE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel or f"Mean CV {metric.upper()}")
    ax.set_title(f"p = {p_omnibus:.2e}", fontsize=10)


def plot_multiple_comparisons(df_cv: pd.DataFrame, metric: str = "r2", prefix: str = "", alpha: float = 0.05) -> None:
    """Save a mean ± 95% CI comparison plot with Tukey-HSD significance coloring.

    Args:
        df_cv: Long-format CV scores frame (see `rm_tukey_hsd`).
        metric: Name of the column to compare.
        prefix: Optional filename prefix for the saved figure.
        alpha: Significance threshold for marking a method "significantly worse".
    """
    n_labels = df_cv["method"].nunique()
    fig, ax = plt.subplots(figsize=(7, max(3, n_labels * 0.8)))
    _draw_comparison_panel(ax, df_cv, metric, alpha=alpha)
    ax.set_title("Multiple Comparisons (Tukey HSD, FWER=0.05)")
    ax.legend(handles=[
        Line2D([0], [0], color=ROYAL_PURPLE, lw=2, label="Best method"),
        Line2D([0], [0], color=SIG_RED,      lw=2, label="Significantly worse"),
        Line2D([0], [0], color=DARK_SLATE,   lw=2, label="Similar"),
    ], fontsize=8, loc="lower right")
    plt.tight_layout()
    save_fig(fig, f"{prefix}reg_multiple_comparisons")


# metric column -> display name, for grid axis labels/titles
METRIC_DISPLAY_NAMES = {
    "r2": "R²", "rmse": "RMSE", "mae": "MAE", "spearman_rho": "Spearman ρ",
    "clf_roc_auc": "ROC-AUC", "clf_pr_auc": "PR-AUC", "clf_mcc": "MCC",
    "clf_recall": "Recall", "clf_precision": "Precision", "clf_f1": "F1",
    "clf_balanced_accuracy": "Balanced Accuracy", "clf_accuracy": "Accuracy",
}


def plot_multiple_comparisons_grid(
    df_cv: pd.DataFrame,
    metrics: List[str] = ("clf_roc_auc", "clf_pr_auc", "clf_mcc", "clf_recall"),
    prefix: str = "", alpha: float = 0.05, ncols: int = 2,
) -> None:
    """Save a grid of mean ± 95% CI Tukey-HSD comparison panels, one per metric.

    Args:
        df_cv: Long-format CV scores frame (see `rm_tukey_hsd`); must contain
            every column named in `metrics`.
        metrics: Metric columns to plot, one subplot each.
        prefix: Optional filename prefix for the saved figure.
        alpha: Significance threshold for marking a method "significantly worse".
        ncols: Number of subplot columns.
    """
    metrics = [m for m in metrics if m in df_cv.columns]
    if not metrics:
        print("plot_multiple_comparisons_grid: none of the requested metrics "
              "are present in df_cv -- skipping.")
        return
    n_labels = df_cv["method"].nunique()
    ncols = min(ncols, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, max(2.5, n_labels * 0.7) * nrows),
                             gridspec_kw={"hspace": 0.6, "wspace": 0.4}, squeeze=False)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        _draw_comparison_panel(ax, df_cv, metric, alpha=alpha,
                               xlabel=METRIC_DISPLAY_NAMES.get(metric, metric.upper()))
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    fig.legend(handles=[
        Line2D([0], [0], color=ROYAL_PURPLE, lw=2, label="Best method"),
        Line2D([0], [0], color=SIG_RED,      lw=2, label="Significantly worse"),
        Line2D([0], [0], color=DARK_SLATE,   lw=2, label="Similar"),
    ], fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    save_fig(fig, f"{prefix}multi_metric_comparisons")


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


def main() -> None:
    """CLI entry point: load exported CV artifacts and save the comparison figures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/outputs/results/comparison",
                    help="folder with cv_fold_scores.pkl and cv_scores_long.csv")
    args = ap.parse_args()
    res = Path(args.results)

    fold_metrics = pickle.load(open(res / "cv_fold_scores.pkl", "rb"))   # {model: {metric: [values]}}
    fold_scores  = {label: metrics["r2"] for label, metrics in fold_metrics.items()}
    df_cv        = pd.read_csv(res / "cv_scores_long.csv")
    wide         = pd.read_csv(res / "cv_scores_wide.csv", header=[0, 1], index_col=[0, 1])   # (seed, fold) index

    # sanity: each model must contribute the same number of folds (balanced for rm-ANOVA)
    counts = df_cv.groupby("method")["r2"].count()
    print("folds per model:\n", counts.to_string())
    if counts.nunique() != 1:
        print("WARNING: unbalanced folds — rm_anova needs equal counts per model.")

    plot_cv_boxplots(fold_scores)
    plot_multiple_comparisons(df_cv, metric="r2")
    plot_multiple_comparisons_grid(df_cv)

    test_pred_path, test_metrics_path = res / "test_predictions.pkl", res / "test_metrics.csv"
    if test_pred_path.exists() and test_metrics_path.exists():
        with open(test_pred_path, "rb") as f:
            predictions = pickle.load(f)
        metrics_df = pd.read_csv(test_metrics_path, index_col=0)
        plot_test_scatter(predictions, metrics_df)
    else:
        print("No test_predictions.pkl/test_metrics.csv in --results -- "
              "skipping test-set scatter plots (run evaluation.py with --test-csv first).")


if __name__ == "__main__":
    main()