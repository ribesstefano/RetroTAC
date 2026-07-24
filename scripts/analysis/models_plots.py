"""
models_plots.py
========
Figures for the model comparison, reading the artifacts exported by
models_evaluation.py:
  - cv_fold_scores.pkl   -> per-model list of fold R2 (for boxplots)
  - cv_scores_long.csv   -> long format [method, seed, fold, cv_cycle, r2] (for Tukey)

Usage:
    python models_plots.py --results data/outputs/results/comparison
"""
import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import autorank
import pingouin as pg
from scipy.stats import spearmanr  # noqa: F401  (handy if you extend)
from statsmodels.stats.libqsturng import psturng, qsturng
from matplotlib.lines import Line2D

# ── palette ─────────────────────────────────────────────────────────────────
ROYAL_PURPLE = "#6A4C93"
DARK_SLATE   = "#2F4F4F"
DEEP_TEAL    = "#1B7F79"
FOREST_GREEN = "#2E7D32"
SIG_RED      = "#CC3333"

COLOR_MAP = {"XGB": DEEP_TEAL, "MLP": FOREST_GREEN, "GNN": ROYAL_PURPLE}

OUT_DIR = Path("figures")


def save_fig(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved figures/{name}.png")


# ── Fig 1: CV fold R2 boxplots (one subplot per model) ──────────────────────
def plot_cv_boxplots(fold_scores, prefix=""):
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
def rm_tukey_hsd(df, metric, group_col, alpha=0.05):
    df_means = df.groupby(group_col).mean(numeric_only=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        aov = pg.rm_anova(dv=metric, within=group_col, subject="cv_cycle",
                          data=df, detailed=True)
    mse, df_resid = aov.loc[1, "MS"], aov.loc[1, "DF"]
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
    return df_means, pc.astype(float)


def plot_multiple_comparisons(df_cv, metric="r2", prefix="", alpha=0.05):
    df_means, pc = rm_tukey_hsd(df_cv, metric, group_col="method", alpha=alpha)
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

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.8)))
    for i, (m, lo, hi, c) in enumerate(zip(means, ci_low, ci_high, colors)):
        ax.plot([lo, hi], [i, i], color=c, linewidth=2.5)
        ax.plot(m, i, "o", color=c, markersize=6)
    bi = labels.index(best)
    ax.axvline(ci_low[bi],  color=ROYAL_PURPLE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(ci_high[bi], color=ROYAL_PURPLE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"Mean CV {metric.upper()}")
    ax.set_title("Multiple Comparisons (Tukey HSD, FWER=0.05)")
    ax.legend(handles=[
        Line2D([0], [0], color=ROYAL_PURPLE, lw=2, label="Best method"),
        Line2D([0], [0], color=SIG_RED,      lw=2, label="Significantly worse"),
        Line2D([0], [0], color=DARK_SLATE,   lw=2, label="Similar"),
    ], fontsize=8, loc="lower right")
    plt.tight_layout()
    save_fig(fig, f"{prefix}reg_multiple_comparisons")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/outputs/results/comparison",
                    help="folder with cv_fold_scores.pkl and cv_scores_long.csv")
    args = ap.parse_args()
    res = Path(args.results)

    fold_scores = pickle.load(open(res / "cv_fold_scores.pkl", "rb"))
    df_cv       = pd.read_csv(res / "cv_scores_long.csv")
    wide        = pd.read_csv(res / "cv_scores_wide.csv", index_col=[0, 1])   # (seed, fold) index

    # sanity: each model must contribute the same number of folds (balanced for rm-ANOVA)
    counts = df_cv.groupby("method")["r2"].count()
    print("folds per model:\n", counts.to_string())
    if counts.nunique() != 1:
        print("WARNING: unbalanced folds — rm_anova needs equal counts per model.")

    plot_cv_boxplots(fold_scores)
    plot_multiple_comparisons(df_cv, metric="r2")


if __name__ == "__main__":
    main()