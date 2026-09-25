"""
scripts/plotting_appendix/remaining_performance.py
==================================================
The metrics `compute_all_metrics` records that the two main-text tables do
not show, as one appendix table plus one figure.

Tables 1 and 3 of the manuscript report R^2, RMSE, MAE, Spearman rho, ROC-AUC,
precision and recall. Every other metric written per fold
(`outputs/cv/<run>/score_seed*_fold*.json`, stacked into
`outputs/models/<run>/<run>_cv_metrics.csv`) is collected here, so the
appendix can state that the ordering GNN > MLP > XGB is not an artifact of
which metrics were chosen for the main text.

The figure shows the 25 paired outer folds as grouped boxes, with the
held-out value of the model refit on the full development set overlaid as a
marker -- the CV spread and the single held-out number are different
quantities and the figure keeps them visibly distinct.

Model hues follow `scripts/models/plotting_evaluation.py` (XGB blue, GNN
orange, MLP violet) so a reader meets the same colour for the same backend
across the whole manuscript.

Usage
-----
    python scripts/plotting_appendix/remaining_performance.py \\
        --runs xgb_20260828_182305 mlp_20260828_182305 gnn_20260828_182305 \\
        --test-metrics outputs/results/results_20260828_182305/test_metrics.csv

Outputs (see --out-dir / --fig-dir):
    outputs/appendix/remaining_performance.csv
    outputs/appendix/remaining_performance_table.tex
    figures/appendix/remaining_performance.{pdf,svg,png}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pubstyle as ps

ps.apply_style()

#: Backend -> hue, matching scripts/models/plotting_evaluation.py's COLOR_MAP
#: so the same backend keeps the same colour throughout the manuscript.
COLOR_MAP = {
    "XGB": ps.PALETTE["blue"],
    "MLP": ps.PALETTE["purple"],
    "GNN": ps.PALETTE["dark_orange"],
}

#: Metrics already in the main text (Tables 1 and 3), excluded from the
#: appendix table so the two do not restate each other.
MAIN_TEXT_METRICS = ("r2", "rmse", "mae", "spearman_rho", "clf_roc_auc",
                     "clf_precision", "clf_recall")

#: Bookkeeping columns and per-fold constants, not performance measurements.
NON_METRIC = ("seed", "fold_idx", "clf_threshold", "n_samples",
              "objective", "objective_alpha", "clf_pos_rate",
              "clf_tp", "clf_fp", "clf_tn", "clf_fn")

#: Display names, and whether a larger value is better. `None` marks a metric
#: with no preferred direction (bias is best at 0).
METRIC_LABELS: Dict[str, tuple] = {
    "pearson_r": ("Pearson $r$", True),
    "kendall_tau": ("Kendall $\\tau$", True),
    "explained_variance": ("Expl. variance", True),
    "medae": ("MedAE", False),
    "max_error": ("Max error", False),
    "bias": ("Bias", None),
    "clf_pr_auc": ("PR-AUC", True),
    "clf_mcc": ("MCC", True),
    "clf_f1": ("$F_1$", True),
    "clf_specificity": ("Specificity", True),
    "clf_balanced_accuracy": ("Balanced acc.", True),
    "clf_accuracy": ("Accuracy", True),
    "clf_cohen_kappa": ("Cohen's $\\kappa$", True),
    "clf_pred_pos_rate": ("Pred. pos. rate", None),
}

#: The subset plotted, in panel order: one row of regression/ranking metrics
#: and one of thresholded-classification metrics. The rest stay table-only.
PLOTTED: Sequence[str] = (
    "pearson_r", "kendall_tau", "explained_variance", "medae", "bias",
    "clf_pr_auc", "clf_mcc", "clf_f1", "clf_balanced_accuracy", "clf_cohen_kappa",
)


def load_cv(cv_root: Path, runs: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """Load the per-fold metric table for each run.

    Args:
        cv_root: Directory holding `<run>/<run>_cv_metrics.csv`.
        runs: Run identifiers.

    Returns:
        Model name -> per-fold metric frame.
    """
    frames = {}
    for run in runs:
        path = cv_root / run / f"{run}_cv_metrics.csv"
        if not path.exists():
            raise SystemExit(f"missing per-fold metrics: {path}")
        frames[run.split("_")[0].upper()] = pd.read_csv(path)
    return frames


def build_table(frames: Dict[str, pd.DataFrame], test: pd.DataFrame) -> pd.DataFrame:
    """Assemble CV mean +/- s.d. and the held-out value for every extra metric.

    Args:
        frames: Model name -> per-fold metric frame.
        test: Held-out metrics, one row per model (index = model name).

    Returns:
        Frame indexed by metric, with a `cv_<model>` and `test_<model>`
        column per model.
    """
    first = next(iter(frames.values()))
    metrics = [c for c in first.columns
               if c not in NON_METRIC and c not in MAIN_TEXT_METRICS]
    rows = []
    for metric in metrics:
        row = {"metric": metric,
               "label": METRIC_LABELS.get(metric, (metric, None))[0]}
        for model, frame in frames.items():
            values = frame[metric].to_numpy(float)
            row[f"cv_{model}_mean"] = values.mean()
            row[f"cv_{model}_std"] = values.std(ddof=1)
            row[f"test_{model}"] = (float(test.loc[model, metric])
                                    if metric in test.columns else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("metric")


def write_latex(table: pd.DataFrame, models: Sequence[str], path: Path) -> None:
    """Write the appendix table as a LaTeX tabular.

    The best value per (metric, evaluation) is bolded, using the direction
    recorded in `METRIC_LABELS`; metrics with no preferred direction are left
    unbolded rather than given an arbitrary winner.

    Args:
        table: Output of `build_table`.
        models: Column order.
        path: Destination `.tex` file.
    """
    lines = [
        r"\begin{tabular}{l" + "r" * (2 * len(models)) + "}",
        r"\toprule",
        " & " + " & ".join([rf"\multicolumn{{{len(models)}}}{{c}}{{5$\times$5 CV}}",
                            rf"\multicolumn{{{len(models)}}}{{c}}{{Held-out}}"]) + r" \\",
        rf"\cmidrule(lr){{2-{1 + len(models)}}}\cmidrule(lr){{{2 + len(models)}-{1 + 2 * len(models)}}}",
        "Metric & " + " & ".join(list(models) + list(models)) + r" \\",
        r"\midrule",
    ]
    for metric, row in table.iterrows():
        label, higher_better = METRIC_LABELS.get(metric, (metric, None))
        cv_means = {m: row[f"cv_{m}_mean"] for m in models}
        tests = {m: row[f"test_{m}"] for m in models}
        cv_best = _best(cv_means, higher_better)
        test_best = _best(tests, higher_better)
        cells = []
        for model in models:
            cell = f"${row[f'cv_{model}_mean']:.3f}\\pm{row[f'cv_{model}_std']:.3f}$"
            cells.append(rf"$\mathbf{{{row[f'cv_{model}_mean']:.3f}\pm{row[f'cv_{model}_std']:.3f}}}$"
                         if model == cv_best else cell)
        for model in models:
            value = row[f"test_{model}"]
            cell = "--" if not np.isfinite(value) else f"{value:.3f}"
            cells.append(rf"\textbf{{{cell}}}" if model == test_best and cell != "--" else cell)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def _best(values: Dict[str, float], higher_better) -> str:
    """Name of the winning model, or an empty string when there is no winner.

    Args:
        values: Model name -> value.
        higher_better: True, False, or None for "no preferred direction".

    Returns:
        The winning model name, or "" when `higher_better` is None or every
        value is missing.
    """
    finite = {k: v for k, v in values.items() if np.isfinite(v)}
    if higher_better is None or not finite:
        return ""
    return max(finite, key=finite.get) if higher_better else min(finite, key=finite.get)


def plot_remaining(frames: Dict[str, pd.DataFrame], test: pd.DataFrame,
                   out_stem: Path, n_cols: int = 5) -> List[Path]:
    """Grouped fold boxes per metric, with the held-out value overlaid.

    Args:
        frames: Model name -> per-fold metric frame.
        test: Held-out metrics indexed by model name.
        out_stem: Output path without extension.
        n_cols: Panels per row.

    Returns:
        Paths written.
    """
    models = list(frames)
    n_rows = int(np.ceil(len(PLOTTED) / n_cols))
    width, _ = ps.set_size()
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(width, width * 0.30 * n_rows),
                             layout="constrained")
    axes = np.atleast_2d(axes)

    for idx, metric in enumerate(PLOTTED):
        ax = axes[idx // n_cols, idx % n_cols]
        data = [frames[m][metric].to_numpy(float) for m in models]
        boxes = ax.boxplot(data, patch_artist=True, widths=0.6,
                           medianprops={"color": "white", "linewidth": 1.0},
                           flierprops={"marker": ".", "markersize": 2.5,
                                       "alpha": 0.4, "markeredgewidth": 0})
        for patch, model in zip(boxes["boxes"], models):
            patch.set_facecolor(COLOR_MAP[model])
            patch.set_edgecolor(ps.darken(COLOR_MAP[model], 0.75))
            patch.set_linewidth(0.5)
        for key in ("whiskers", "caps"):
            for artist in boxes[key]:
                artist.set_linewidth(0.6)
        if metric in test.columns:
            for pos, model in enumerate(models, start=1):
                ax.scatter([pos], [float(test.loc[model, metric])], marker="D",
                           s=12, facecolor="white", zorder=6,
                           edgecolor=ps.darken(COLOR_MAP[model], 0.7), linewidths=0.9)
        ax.set_title(METRIC_LABELS.get(metric, (metric, None))[0], fontsize=ps.LEGEND_FONTSIZE)
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels(models, fontsize=ps.ANNOT_FONTSIZE)
        ax.tick_params(axis="x", length=0)
        # Held-out markers can land past the fold spread; give them room so
        # they are not drawn on the frame itself.
        ax.margins(y=0.10)
        ax.tick_params(axis="both", which="major", labelsize=ps.ANNOT_FONTSIZE - 1)

    for leftover in range(len(PLOTTED), n_rows * n_cols):
        axes[leftover // n_cols, leftover % n_cols].set_visible(False)

    # The backends are already named on every x axis, so the legend carries
    # only the mark whose meaning the panels cannot state themselves.
    handles = [plt.Line2D([], [], marker="D", linestyle="none",
                          markerfacecolor="white", markeredgecolor="#555555",
                          label="Held-out model refit on the full development set")]
    fig.legend(handles=handles, loc="outside lower center", frameon=False,
               fontsize=ps.LEGEND_FONTSIZE)
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
        help="Run identifiers, in column order.")
    p.add_argument("--test-metrics", type=Path,
                   default=Path("outputs/results/results_20260828_182305/test_metrics.csv"),
                   help="Held-out metrics written by evaluation.py --test-csv.")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/appendix"),
                   help="Directory for the tables.")
    p.add_argument("--fig-dir", type=Path, default=Path("figures/appendix"),
                   help="Directory for the figure.")
    return p.parse_args()


def main() -> None:
    """Write the remaining-metrics table and figure."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    frames = load_cv(args.cv_root, args.runs)
    test = pd.read_csv(args.test_metrics, index_col=0)
    print(f"held-out metrics for {list(test.index)} "
          f"on n = {int(test['n_samples'].iloc[0])} molecules")

    table = build_table(frames, test)
    table.to_csv(args.out_dir / "remaining_performance.csv")
    models = list(frames)
    show = ["label"] + [f"cv_{m}_mean" for m in models] + [f"test_{m}" for m in models]
    print("\n=== metrics not shown in the main text ===")
    print(table[show].round(4).to_string())

    tex_path = args.out_dir / "remaining_performance_table.tex"
    write_latex(table, models, tex_path)
    print(f"\nwrote {tex_path}")

    written = plot_remaining(frames, test, args.fig_dir / "remaining_performance")
    print("wrote:", *[str(p) for p in written], sep="\n  ")


if __name__ == "__main__":
    main()
