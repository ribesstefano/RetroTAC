"""
evaluation.py
=============
Statistical comparison of the surrogate models (xgb / mlp / gnn) from their
per-fold CV scores, and (optionally) evaluation of the final models on a
held-out test set. Exports the artifacts a separate plotting script/notebook
consumes. Final-model test predictions/metrics are cached to
`{out}/test_predictions.pkl` / `test_metrics.csv` and reused on rerun (pass
`--force-recompute` to redo).

Every metric written per fold by train.py (see retrotac.models.metrics
.compute_all_metrics) is loaded and reported, not just R2:
    {CV_DIR}/{prefix}/score_seed{seed}_fold{fold}.json
        -> {"seed", "fold_idx", "r2", "rmse", ..., "clf_*", "n_samples", "objective", ...}

With `--ensemble` (requires `--test-csv`), every individual 5x5-CV fold model
(not just the final aggregated one) is also loaded and predicted on the test
set -- see `predict_fold_models` -- and combined into best_single/uniform/
best_backend/Caruana ensemble strategies, each also scored on uncertainty
calibration (Spearman rho between spread and error, k-sigma coverage, ECE/MCE)
-- see `run_ensemble_strategies`. Fold predictions are cached to
`{out}/cv_fold_test_predictions.pkl` and reused on rerun (pass
`--force-predict` to recompute).

`--evidential-model <prefix>` adds the other way of getting an uncertainty:
an evidential-head GNN (see retrotac/models/gnn/model.py) predicts the
parameters of a Normal-Inverse-Gamma per molecule, so ONE model reports its own
aleatoric/epistemic sigma instead of inferring one from ensemble disagreement.
Its predictions are scored through the same metrics on the same evaluation
rows, landing as extra rows in `ensemble_strategies.csv` and in the focused
`uncertainty_comparison.csv`, so "does a bigger sigma really mean a bigger
error?" is answered on equal terms for both.

Usage
-----
    # CV comparison only (no test set yet):
    python evaluation.py --models xgb_v1 mlp_v1 gnn_v1

    # also evaluate final models on a held-out test CSV, against the config
    # the runs were actually trained with (sets target/molecule_col/fp params):
    python evaluation.py --models xgb_v1 mlp_v1 gnn_v1 \
        --config config/models_config_routes.yaml \
        --test-csv data/sets/routes_test.csv --output-root outputs

    # also score ensemble strategies over all 5x5-CV fold models:
    python evaluation.py --models xgb_v1 mlp_v1 gnn_v1 \
        --test-csv data/sets/routes_test.csv --output-root outputs --ensemble

    # ... and compare those ensembles' uncertainty against an evidential GNN's:
    python evaluation.py --models xgb_v1 mlp_v1 gnn_v1 \
        --test-csv data/sets/routes_test.csv --output-root outputs --ensemble \
        --evidential-model gnn_routes_evidential --out uncertainty
"""
import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import autorank
import numpy as np
import pandas as pd
from scipy.stats import levene, spearmanr
from sklearn.metrics import log_loss, mean_squared_error, r2_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[2]))   # -> chem_utils

from retrotac.models.config import ModelsConfig
from retrotac.models.loading import compute_features, load_backend
from retrotac.models.metrics import compute_all_metrics

# ── paths (anchored to repo root, same as train.py's own defaults) ──────────
_ROOT           = Path(__file__).parents[3]                 # RetroTAC/
DEFAULT_CONFIG  = _ROOT / "config" / "models_config.yaml"
DEFAULT_OUTPUT_ROOT = _ROOT / "data" / "outputs"


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the model comparison CLI.

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description="Compare surrogate models from CV fold scores.")
    ap.add_argument("--models", nargs="+", required=True,
                    help="model prefixes to compare, e.g. xgb_v1 mlp_v1 gnn_v1")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="models_config.yaml the runs were trained with -- sets target/"
                         "molecule_col/fp params (default: config/models_config.yaml); "
                         "mirrors train.py's --config")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                    help="root holding cv/ and models/ (default: data/outputs; pass "
                         "'outputs' to read straight from train.py's own --output-root "
                         "default without symlinking)")
    ap.add_argument("--test-csv", default=None,
                    help="optional held-out test CSV (molecule + target columns)")
    ap.add_argument("--smiles-col", default=None,
                    help="SMILES column in --test-csv (default: --config's molecule_col)")
    ap.add_argument("--target-col", default=None,
                    help="target column in --test-csv (default: --config's target)")
    ap.add_argument("--force-recompute", action="store_true",
                    help="recompute final-model test predictions/metrics even if "
                         "test_predictions.pkl/test_metrics.csv from a previous run already "
                         "exist in --out (see also --force-predict, the equivalent toggle "
                         "for the --ensemble fold-model cache)")
    ap.add_argument("--rank-metric", default="r2",
                    help="metric driving the paired AutoRank comparison (default: r2); "
                         "every metric is still reported in cv_scores_wide/long.csv "
                         "regardless of this choice")
    ap.add_argument("--out", default="comparison",
                    help="output subfolder name under results/")
    ap.add_argument("--ensemble", action="store_true",
                    help="also load every 5x5-CV fold model, predict on --test-csv, and "
                         "score best_single/uniform/best_backend/Caruana ensemble "
                         "strategies (requires --test-csv; expensive -- loads up to "
                         "len(seeds)*n_folds*len(models) models, each needing a GPU "
                         "allocation, see CLAUDE.md)")
    ap.add_argument("--force-predict", action="store_true",
                    help="recompute fold-model test predictions even if a cache from a "
                         "previous --ensemble run already exists at "
                         f"{{out}}/{FOLD_PRED_FILENAME}")
    ap.add_argument("--ensemble-caruana-frac", type=float, default=0.2,
                    help="fraction of --test-csv reserved for Caruana greedy selection; "
                         "the remainder is the shared evaluation split every strategy is "
                         "scored on (default: 0.2)")
    ap.add_argument("--ensemble-iterations", type=int, default=100,
                    help="greedy selection iterations for the Caruana strategy (default: 100)")
    ap.add_argument("--evidential-model", default=None,
                    help="prefix of an evidential-head GNN run (e.g. gnn_routes_evidential, "
                         "see slurm/train_final_gnn_evidential.sh) whose final model predicts "
                         "its own uncertainty; scored on the same evaluation rows as the "
                         "ensemble strategies so the two uncertainties can be compared "
                         "(requires --ensemble). Do NOT also list it in --models: it has no "
                         "CV folds to load")
    ap.add_argument("--device", default="cpu",
                    help="compute device for loading/predicting with the models "
                         "(cpu or cuda; only the gnn backend uses it). Default: cpu")
    args = ap.parse_args()
    if args.ensemble and not args.test_csv:
        ap.error("--ensemble requires --test-csv")
    if args.evidential_model and not args.ensemble:
        ap.error("--evidential-model requires --ensemble (it is scored against the "
                 "ensemble strategies, on their evaluation split)")
    return args


def _label(prefix: str) -> str:
    """Human label from a prefix, e.g. 'xgb_v1' -> 'XGB'."""
    return prefix.split("_")[0].upper()


# ── Step 1: load per-fold CV metrics from JSON ──────────────────────────────
def load_fold_metrics(prefix: str, cv_dir: Path, seeds: List[int], n_folds: int) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Load every per-(seed, fold) metric for one model prefix.

    Files are flat under cv/{prefix}/ (distinguished by name/extension:
    score_*.json, best_params_*.json, trials_*.csv, *.db).

    Args:
        prefix: Model prefix, e.g. "xgb_v1".
        cv_dir: Directory holding one subfolder per prefix.
        seeds: CV seeds to look for.
        n_folds: Number of folds per seed.

    Returns:
        Mapping from (seed, fold) to that fold's full metric dict (everything
        train.py wrote except "seed"/"fold_idx", which become the key), for
        every fold whose score JSON exists on disk.
    """
    base = cv_dir / prefix
    scores = {}
    for seed in seeds:
        for fold in range(n_folds):
            p = base / f"score_seed{seed}_fold{fold}.json"
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                scores[(seed, fold)] = {k: v for k, v in d.items() if k not in ("seed", "fold_idx")}
    return scores


def build_cv_frames(
    model_prefixes: List[str], cv_dir: Path, seeds: List[int], n_folds: int, rank_metric: str = "r2",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the paired wide frame (for AutoRank) and a long frame (for plotting).

    Args:
        model_prefixes: Model prefixes to compare, e.g. ["xgb_v1", "mlp_v1"].
        cv_dir: Directory holding one subfolder per prefix.
        seeds: CV seeds to look for.
        n_folds: Number of folds per seed.
        rank_metric: Metric to print the per-model summary for; also validated
            against the metrics actually present.

    Returns:
        Tuple of (wide, long):
            wide: MultiIndex columns (model, metric), rows (seed, fold)-indexed,
                restricted to folds present for every model (paired comparison).
            long: One row per (method, seed, fold) with every metric as its own
                column plus a `cv_cycle` id linking paired samples across
                methods, for rm-Tukey / boxplots.

    Raises:
        RuntimeError: If no model has any fold scores, or if no (seed, fold)
            combination is shared across all models.
        ValueError: If `rank_metric` isn't among the metrics found.
    """
    per_model = {}                    # label -> {(seed,fold): {metric: value}}
    for prefix in model_prefixes:
        s = load_fold_metrics(prefix, cv_dir, seeds, n_folds)
        if not s:
            print(f"  WARNING: no fold scores found for '{prefix}' — skipping.")
            continue
        per_model[_label(prefix)] = s

    if not per_model:
        raise RuntimeError("No fold scores found for any model.")

    # common (seed,fold) keys present in ALL models -> proper paired comparison
    common = sorted(set.intersection(*(set(s.keys()) for s in per_model.values())))
    if not common:
        raise RuntimeError("No (seed,fold) folds are shared across all models.")

    # union of metric keys across every model/fold, so a metric missing from
    # one backend's json doesn't crash the others (filled with NaN instead).
    metric_keys = sorted({m for s in per_model.values() for d in s.values() for m in d})
    if rank_metric not in metric_keys:
        raise ValueError(f"--rank-metric {rank_metric!r} not found; available: {metric_keys}")

    frames = [
        pd.DataFrame(
            [{m: per_model[label][k].get(m, float("nan")) for m in metric_keys} for k in common],
            index=pd.MultiIndex.from_tuples(common, names=["seed", "fold"]),
        )
        for label in per_model
    ]
    wide = pd.concat(frames, axis=1, keys=list(per_model.keys()), names=["model", "metric"])

    # long: for rm-Tukey / boxplots, with a paired-sample id (cv_cycle)
    long_rows = []
    for cycle, (seed, fold) in enumerate(common):
        for label in per_model:
            row = {"method": label, "seed": seed, "fold": fold, "cv_cycle": cycle}
            row.update(per_model[label][(seed, fold)])
            long_rows.append(row)
    long = pd.DataFrame(long_rows)

    print(f"\nLoaded {len(per_model)} models over {len(common)} paired folds.")
    for label in per_model:
        vals = wide[(label, rank_metric)]
        print(f"  {label:<6} mean {rank_metric} = {vals.mean():.4f} ± {vals.std():.4f}")
    return wide, long


def build_cv_metrics_summary(wide: pd.DataFrame) -> pd.DataFrame:
    """Tidy mean/std/n per (model, metric), for every metric in `wide`.

    Args:
        wide: MultiIndex-column frame (model, metric) from `build_cv_frames`.

    Returns:
        DataFrame with columns [model, metric, mean, std, n].
    """
    summary = pd.DataFrame({"mean": wide.mean(), "std": wide.std(), "n": wide.count()})
    summary.index = summary.index.set_names(["model", "metric"])
    return summary.reset_index()


# (metric key, LaTeX column header, higher-is-better, decimal places)
LATEX_METRICS: List[Tuple[str, str, bool, int]] = [
    ("r2", "$R^2$", True, 3),
    ("rmse", "RMSE", False, 3),
    ("mae", "MAE", False, 3),
    ("spearman_rho", r"Spearman $\rho$", True, 3),
    ("clf_roc_auc", "ROC-AUC", True, 3),
    ("clf_precision", "Precision", True, 2),
    ("clf_recall", "Recall", True, 2),
]


def _ordered_labels(available: List[str], model_order: Tuple[str, ...]) -> List[str]:
    """`available`, sorted to match `model_order` with any extras appended after."""
    return [m for m in model_order if m in available] + [m for m in available if m not in model_order]


def _fmt_cv_cell(mean: float, std: float, decimals: int, bold: bool) -> str:
    """Render one CV cell as `$mean\\pm std$`, or `---` if `mean` is NaN."""
    if pd.isna(mean):
        return "---"
    std = 0.0 if pd.isna(std) else std
    body = f"{mean:.{decimals}f}\\pm{std:.{decimals}f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def _fmt_test_cell(value: Optional[float], decimals: int, bold: bool) -> str:
    """Render one held-out cell as a plain number, or `---` if `value` is missing/NaN."""
    if value is None or pd.isna(value):
        return "---"
    body = f"{value:.{decimals}f}"
    return f"\\textbf{{{body}}}" if bold else body


def build_latex_summary_table(
    cv_wide: pd.DataFrame,
    test_results: Optional[Dict[str, Dict[str, float]]] = None,
    model_order: Tuple[str, ...] = ("XGB", "MLP", "GNN"),
    clf_threshold: float = 0.7,
    n_test: Optional[int] = None,
    metrics: List[Tuple[str, str, bool, int]] = LATEX_METRICS,
    caption: Optional[str] = None,
    label: str = "tab:performance",
) -> str:
    """Render CV (+ optional held-out) results as a LaTeX `table` environment.

    CV cells are mean $\\pm$ s.d. over every (seed, fold) row in `cv_wide`; held-out cells
    (when `test_results` is given) are single values, e.g. from `evaluate_test`'s `results`
    dict or a reloaded `test_metrics.csv`. The best model per column is bolded within each
    section separately (min for rmse/mae, max for everything else).

    Args:
        cv_wide: MultiIndex-column frame (model, metric) from `build_cv_frames`.
        test_results: Optional label -> {metric: value}. None omits the "Held-out" block
            entirely rather than filling it with placeholders.
        model_order: Preferred row order; any model not listed is appended after, in the
            order it appears in `cv_wide`.
        clf_threshold: Binarisation cut used for precision/recall, quoted in the caption.
        n_test: Held-out set size, quoted in the caption when `test_results` is given.
        metrics: (key, header, higher_is_better, decimals) columns to render, in order.
        caption: Full `\\caption{}` text; auto-generated from `clf_threshold`/`n_test`/the
            fold count when None.
        label: `\\label{}` value.

    Returns:
        The LaTeX source for a full `table` environment (also see `report_latex_summary`,
        which prints and saves this).
    """
    labels = _ordered_labels(list(cv_wide.columns.get_level_values("model").unique()), model_order)

    cv_stats = {
        lab: {m: (cv_wide[(lab, m)].mean(), cv_wide[(lab, m)].std()) for m, *_ in metrics}
        for lab in labels
    }
    cv_best = {
        m: max((lab for lab in labels if not pd.isna(cv_stats[lab][m][0])),
               key=lambda lab: cv_stats[lab][m][0], default=None)
        if higher else
        min((lab for lab in labels if not pd.isna(cv_stats[lab][m][0])),
            key=lambda lab: cv_stats[lab][m][0], default=None)
        for m, _, higher, _ in metrics
    }

    test_best: Dict[str, Optional[str]] = {}
    if test_results:
        for m, _, higher, _ in metrics:
            candidates = [lab for lab in labels
                          if not pd.isna(test_results.get(lab, {}).get(m, float("nan")))]
            test_best[m] = (
                (max if higher else min)(candidates, key=lambda lab: test_results[lab][m])
                if candidates else None
            )

    if caption is None:
        parts = [
            "Regression and thresholded-classification performance.",
            f"CV values are mean $\\pm$ s.d. over {len(cv_wide)} paired outer evaluations.",
        ]
        if test_results:
            parts.append(
                f"Held-out models are evaluated on {n_test} held-out molecules."
                if n_test else "Held-out models are evaluated on a held-out test set."
            )
        parts.append(
            f"Precision and recall use threshold {clf_threshold:g} for both labels and "
            "predictions; ROC-AUC uses continuous predictions."
        )
        caption = " ".join(parts)

    col_spec = "ll" + "r" * len(metrics)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\footnotesize",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Evaluation & Model & " + " & ".join(h for _, h, _, _ in metrics) + r"\\",
        r"\midrule",
    ]

    for i, lab in enumerate(labels):
        row_label = "Grouped CV & " if i == 0 else "& "
        cells = [_fmt_cv_cell(*cv_stats[lab][m], decimals, cv_best[m] == lab) for m, _, _, decimals in metrics]
        lines.append(row_label + f"{lab} & " + " & ".join(cells) + r"\\")

    if test_results:
        lines.append(r"\midrule")
        for i, lab in enumerate(labels):
            row_label = "Held-out & " if i == 0 else "& "
            cells = [
                _fmt_test_cell(test_results.get(lab, {}).get(m), decimals, test_best[m] == lab)
                for m, _, _, decimals in metrics
            ]
            lines.append(row_label + f"{lab} & " + " & ".join(cells) + r"\\")

    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def report_latex_summary(
    cv_wide: pd.DataFrame,
    out_dir: Path,
    test_results: Optional[Dict[str, Dict[str, float]]] = None,
    model_order: Tuple[str, ...] = ("XGB", "MLP", "GNN"),
    clf_threshold: float = 0.7,
    n_test: Optional[int] = None,
) -> str:
    """Build the LaTeX summary table (see `build_latex_summary_table`), print it, and save it.

    Args:
        cv_wide: MultiIndex-column frame (model, metric) from `build_cv_frames`.
        out_dir: Directory to write `summary_table.tex` into.
        test_results: See `build_latex_summary_table`.
        model_order: See `build_latex_summary_table`.
        clf_threshold: See `build_latex_summary_table`.
        n_test: See `build_latex_summary_table`.

    Returns:
        The LaTeX source (same as `build_latex_summary_table`).
    """
    table = build_latex_summary_table(
        cv_wide, test_results=test_results, model_order=model_order,
        clf_threshold=clf_threshold, n_test=n_test,
    )
    print("\n── LaTeX summary table ─────────────────────────────────────────────────")
    print(table)
    path = out_dir / "summary_table.tex"
    path.write_text(table + "\n")
    print(f"  (saved to {path})")
    return table


def _fold_scores_by_metric(wide: pd.DataFrame) -> Dict[str, Dict[str, List[float]]]:
    """Reshape `wide` into {model: {metric: [values, ...]}} for pickling.

    Args:
        wide: MultiIndex-column frame (model, metric) from `build_cv_frames`.

    Returns:
        Nested dict, fold order matching `wide`'s row order.
    """
    labels = wide.columns.get_level_values("model").unique()
    return {label: {metric: wide[(label, metric)].tolist() for metric in wide[label].columns} for label in labels}


# ── Step 2: AutoRank statistical comparison ─────────────────────────────────
def check_parametric(flat: pd.DataFrame, variance_ratio_threshold: float = 9.0) -> bool:
    """Check whether per-model fold variances are homogeneous enough for a parametric test.

    Combines Levene's test with a variance-ratio heuristic, since Levene's p-value
    alone gets oversensitive to trivial differences as sample size grows: variances
    are only treated as unequal when Levene rejects homogeneity AND the max/min
    variance ratio exceeds `variance_ratio_threshold`.

    Args:
        flat: Paired wide frame (columns = model labels), one metric's values.
        variance_ratio_threshold: Max/min fold-variance ratio above which a
            significant Levene result is treated as a real violation (default 9,
            a common rule of thumb).

    Returns:
        True if variances are homogeneous enough to force AutoRank's parametric path.
    """
    variances = flat.var()
    var_ratio = variances.max() / variances.min()
    _, pvalue = levene(*(flat[col].values for col in flat.columns))

    is_parametric = not (pvalue < 0.05 and var_ratio > variance_ratio_threshold)
    verdict = "homogeneous" if is_parametric else "NOT equal -> non-parametric required"
    print(f"  Levene's test: p={pvalue:.4f} | max/min fold-variance ratio={var_ratio:.4f} "
          f"-> variances {verdict}")
    return is_parametric


def run_autorank(wide: pd.DataFrame, rank_metric: str, out_dir: Path) -> Any:
    """Run AutoRank's statistical comparison (on `rank_metric`) and persist its ranking table.

    Args:
        wide: MultiIndex-column frame (model, metric) from `build_cv_frames`.
        rank_metric: Which metric to extract and rank models on.
        out_dir: Directory to write `autorank_rankdf.csv` into.

    Returns:
        The `autorank.autorank` result object.
    """
    print("\n── AutoRank ─────────────────────────────────────────────────────────")
    flat = wide.xs(rank_metric, axis=1, level="metric")
    force_mode = "parametric" if check_parametric(flat) else None
    result = autorank.autorank(flat, alpha=0.05, verbose=False, force_mode=force_mode)
    autorank.create_report(result)

    # persist the ranking table for the plotting script (figures are made in plots.py)
    if getattr(result, "rankdf", None) is not None:
        result.rankdf.to_csv(out_dir / "autorank_rankdf.csv")
    return result


# ── Step 3 (optional): evaluate final models on a held-out test set ─────────
def _load_model_from_path(prefix: str, base: str, device: str = "cpu") -> Any:
    """Load a saved model from an explicit base path using its class's loader.

    Shared by `_load_model` (the aggregated `{prefix}_final`) and
    `predict_fold_models` (each individual `model_seed{seed}_fold{fold}`) --
    both are the same backend classes, saved via the same `save(path)`
    convention, just under different base paths. Thin wrapper around
    `retrotac.models.loading.load_backend`, which also backs
    `retrotac.models.ensemble.RetroTAC` -- both call sites share one
    definition of "how a fold model is loaded".

    Args:
        prefix: Model prefix, e.g. "xgb_v1"; the leading token before "_"
            selects the loader class.
        base: Base path (no extension) passed to the class's own `load()`.
        device: Compute device; only the gnn backend's `load()` takes one.

    Returns:
        The loaded model instance.

    Raises:
        ValueError: If the prefix's leading token isn't xgb/mlp/gnn.
    """
    return load_backend(prefix.split("_")[0], base, device=device)


def _load_model(prefix: str, models_dir: Path, device: str = "cpu") -> Any:
    """Load the saved final model for a prefix (see `_load_model_from_path`).

    Args:
        prefix: Model prefix, e.g. "xgb_v1".
        models_dir: Directory holding one subfolder per prefix.
        device: Compute device forwarded to the backend's loader.

    Returns:
        The loaded model instance.
    """
    return _load_model_from_path(
        prefix, str(models_dir / prefix / f"{prefix}_final"), device=device)


def _test_features(smiles: List[str], cfg: ModelsConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Standardize test-set SMILES and compute the model input features.

    Args:
        smiles: Raw SMILES strings from the test CSV.
        cfg: Config the models were trained with (fp size/radius, which
            feature groups are enabled).

    Returns:
        Tuple of (fingerprints, descriptors); either may be None if disabled
        via the config's features.use_fingerprints / use_descriptors.
    """
    return compute_features(
        smiles, cfg.features.fp_size, cfg.features.fp_radius,
        cfg.features.use_fingerprints, cfg.features.use_descriptors,
    )


def evaluate_test(
    model_prefixes: List[str], test_csv: str, models_dir: Path, cfg: ModelsConfig,
    smiles_col: str, target_col: str, clf_threshold: float,
    X_fp: Optional[np.ndarray] = None, X_desc: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, float]]]:
    """Load each final model and score it on a held-out test CSV.

    Args:
        model_prefixes: Model prefixes to evaluate, e.g. ["xgb_v1", "mlp_v1"].
        test_csv: Path to a CSV with a SMILES column and a target column.
        models_dir: Directory holding one subfolder per prefix.
        cfg: Config the models were trained with (fp params for feature
            computation).
        smiles_col: SMILES column name in `test_csv`.
        target_col: Target column name in `test_csv`.
        clf_threshold: Binarisation cut forwarded to `compute_all_metrics`.
        X_fp: Precomputed fingerprints for `test_csv`'s SMILES (see
            `_test_features`), or None to compute internally. Pass this (and
            `X_desc`) when the caller already has them -- e.g. main() reusing
            them for `predict_fold_models` too -- to avoid recomputing
            fingerprints/descriptors for the same molecules twice.
        X_desc: Precomputed descriptors, or None to compute internally.
        device: Compute device forwarded to the backend loaders (gnn only).

    Returns:
        Tuple of (predictions, results):
            predictions: label -> {"y_pred": array, "y_true": array}.
            results: label -> full metric dict from `compute_all_metrics`.
        A model whose load or predict raises is skipped from both dicts
        (a warning is printed instead of aborting the whole comparison).
    """
    df = pd.read_csv(test_csv)
    smiles = df[smiles_col].tolist()
    y_true = df[[target_col]].values
    if X_fp is None and X_desc is None:
        X_fp, X_desc = _test_features(smiles, cfg)

    predictions, results = {}, {}
    for prefix in model_prefixes:
        label = _label(prefix)
        try:
            model = _load_model(prefix, models_dir, device=device)
            y_pred = model.predict(smiles, X_fp=X_fp, X_desc=X_desc)   # GNN ignores X_fp/X_desc
            predictions[label] = {"y_pred": y_pred, "y_true": y_true}
            results[label] = compute_all_metrics(y_true, y_pred, threshold=clf_threshold)
            m = results[label]
            print(f"  {label:<6} r2={m['r2']:.3f}  rmse={m['rmse']:.3f}  mae={m['mae']:.3f}  "
                  f"spearman_rho={m['spearman_rho']:.3f}  clf_roc_auc={m['clf_roc_auc']:.3f}")
        except Exception as e:
            print(f"  WARNING: test eval failed for '{prefix}': {e}")
    return predictions, results


# ── Step 4 (optional): predict every 5x5-CV fold model, for ensembling ──────
FOLD_PRED_FILENAME = "cv_fold_test_predictions.pkl"


def predict_fold_models(
    model_prefixes: List[str], test_csv: str, cv_dir: Path, cfg: ModelsConfig,
    smiles_col: str, target_col: str, seeds: List[int], n_folds: int,
    out_dir: Path, force: bool = False,
    X_fp: Optional[np.ndarray] = None, X_desc: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Predict a held-out test CSV with every 5x5-CV fold model, caching to disk.

    Loading and predicting with all `len(model_prefixes) * len(seeds) * n_folds`
    fold models is expensive (every xgb/mlp/gnn call needs a GPU allocation,
    see CLAUDE.md's "Model training cannot run on the Berzelius login node"),
    so results are cached to `{out_dir}/cv_fold_test_predictions.pkl` and
    reused on rerun unless `force`.

    Args:
        model_prefixes: Model prefixes to load fold models for, e.g. ["xgb_v1"].
        test_csv: Path to the held-out test CSV (same one `evaluate_test` uses).
        cv_dir: Directory holding one subfolder per prefix (train.py's cv/).
        cfg: Config the models were trained with (fp params for feature
            computation).
        smiles_col: SMILES column name in `test_csv`.
        target_col: Target column name in `test_csv`.
        seeds: CV seeds to look for.
        n_folds: Number of folds per seed.
        out_dir: Directory to read/write the prediction cache in.
        force: Recompute and overwrite the cache even if it already exists.
        X_fp: Precomputed fingerprints for `test_csv`'s SMILES (see
            `_test_features`), or None to compute internally. Pass this (and
            `X_desc`) when the caller already has them -- e.g. main() reusing
            them from `evaluate_test` -- to avoid recomputing
            fingerprints/descriptors for the same molecules twice. Ignored
            when the prediction cache already exists.
        X_desc: Precomputed descriptors, or None to compute internally.
        device: Compute device forwarded to the backend loaders (gnn only).

    Returns:
        Tuple of (fold_predictions, y_true):
            fold_predictions: "{label}_seed{seed}_fold{fold}" -> 1-D prediction
                array, for every fold model that loaded and predicted cleanly
                (a missing/broken fold is skipped with a warning, not fatal).
            y_true: 1-D ground-truth array from `test_csv`, shared by every
                fold model.
    """
    cache_path = out_dir / FOLD_PRED_FILENAME
    if cache_path.exists() and not force:
        print(f"  Found cached fold predictions at {cache_path} -- skipping load/predict.")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        return cached["fold_predictions"], cached["y_true"]

    df = pd.read_csv(test_csv)
    smiles = df[smiles_col].tolist()
    y_true = df[target_col].to_numpy(dtype=float)
    if X_fp is None and X_desc is None:
        X_fp, X_desc = _test_features(smiles, cfg)

    fold_predictions: Dict[str, np.ndarray] = {}
    for prefix in model_prefixes:
        label = _label(prefix)
        for seed in seeds:
            for fold in range(n_folds):
                base = cv_dir / prefix / f"model_seed{seed}_fold{fold}"
                try:
                    model = _load_model_from_path(prefix, str(base), device=device)
                    y_pred = model.predict(smiles, X_fp=X_fp, X_desc=X_desc)
                    fold_predictions[f"{label}_seed{seed}_fold{fold}"] = np.asarray(y_pred, dtype=float).reshape(-1)
                except Exception as e:
                    print(f"  WARNING: predict failed for {base}: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"fold_predictions": fold_predictions, "y_true": y_true}, f)
    print(f"  Predicted with {len(fold_predictions)} fold models -> cached at {cache_path}")
    return fold_predictions, y_true


# ── Step 4b (optional): the evidential GNN's own predicted uncertainty ──────
EVIDENTIAL_PRED_FILENAME = "evidential_test_predictions.pkl"


def predict_evidential(
    prefix: str, test_csv: str, models_dir: Path, smiles_col: str, target_col: str,
    out_dir: Path, device: str = "cpu", force: bool = False, batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    """Predict a held-out test CSV with an evidential GNN, uncertainty included.

    The counterpart to `predict_fold_models`: where an ensemble reads its
    uncertainty off the disagreement between many models, an evidential head
    predicts the parameters of a Normal-Inverse-Gamma per molecule and reports
    its own -- from ONE model, one forward pass (see
    retrotac.models.gnn.model.CheMeleonRegressor.predict_uncertainty). Scoring
    the two side by side on the same rows is the point of `--evidential-model`.

    Cached to `{out_dir}/evidential_test_predictions.pkl` and reused on rerun
    unless `force`.

    Args:
        prefix: Prefix of the evidential run, e.g. "gnn_routes_evidential"; its
            final model is read from `models_dir/{prefix}/{prefix}_final`.
        test_csv: Path to the held-out test CSV (the same one the ensemble uses).
        models_dir: Directory holding one subfolder per prefix.
        smiles_col: SMILES column name in `test_csv`.
        target_col: Target column name in `test_csv`.
        out_dir: Directory to read/write the prediction cache in.
        device: Compute device for the GNN forward pass.
        force: Recompute and overwrite the cache even if it already exists.
        batch_size: Molecules per forward pass.

    Returns:
        Dict of 1-D arrays over the test set, in row order: "mean", "std"
        (total predictive sigma), "aleatoric_std", "epistemic_std", "y_true".

    Raises:
        ValueError: If `prefix` names a model that has no evidential head, in
            which case there is no predicted uncertainty to compare.
    """
    cache_path = out_dir / EVIDENTIAL_PRED_FILENAME
    if cache_path.exists() and not force:
        print(f"  Found cached evidential predictions at {cache_path} -- skipping predict.")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv(test_csv)
    smiles = df[smiles_col].tolist()
    model = _load_model(prefix, models_dir, device=device)
    if not getattr(model, "is_evidential", False):
        raise ValueError(
            f"--evidential-model {prefix!r} was trained with a point-estimate head; "
            "train it with a config whose gnn.head is 'evidential' (see "
            "config/models_config_routes_evidential.yaml) to get an uncertainty to compare."
        )
    u = model.predict_uncertainty(smiles, batch_size=batch_size)
    out = {
        "mean":           np.asarray(u["mean"], dtype=float).reshape(-1),
        "std":            np.asarray(u["std"], dtype=float).reshape(-1),
        "aleatoric_std":  np.sqrt(np.asarray(u["aleatoric_var"], dtype=float)).reshape(-1),
        "epistemic_std":  np.sqrt(np.asarray(u["epistemic_var"], dtype=float)).reshape(-1),
        "y_true":         df[target_col].to_numpy(dtype=float),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    print(f"  Predicted {len(out['mean'])} molecules with {prefix} -> cached at {cache_path}")
    return out


def evidential_predictors(evidential: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split one evidential prediction into the three sigmas worth comparing.

    All three share the same point estimate and differ only in which part of the
    NIG variance they call uncertainty: the total (the like-for-like counterpart
    of an ensemble's spread), the epistemic part alone (model ignorance, which
    is what ensemble disagreement actually measures), and the aleatoric part
    alone (irreducible noise, which an ensemble cannot see at all).

    Args:
        evidential: The dict `predict_evidential` returns.

    Returns:
        Row name -> (prediction, sigma), ready for `run_ensemble_strategies`'
        `extra_predictors`.
    """
    return {
        "evidential_total":     (evidential["mean"], evidential["std"]),
        "evidential_epistemic": (evidential["mean"], evidential["epistemic_std"]),
        "evidential_aleatoric": (evidential["mean"], evidential["aleatoric_std"]),
    }


# ── Step 5 (optional): ensemble-selection strategies over the fold pool ─────
class EnsembleSelector:
    """Caruana-style greedy forward ensemble selection over a pool of predictions.

    Repeatedly adds whichever pool model most improves the running average
    prediction (with replacement, so a strong model can be added more than
    once), optionally bagged over random subsets of the pool for stability.
    """

    def __init__(self, metric: str = "rmse") -> None:
        """
        Args:
            metric: Metric to minimise: 'rmse', 'mse', 'r2', 'roc_auc',
                'log_loss', or 'brier' ('r2'/'roc_auc' are negated so that,
                like the others, lower is always better).
        """
        self.metric = metric

    def compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Metric value for (y_true, y_pred); lower is better for every option.

        Raises:
            ValueError: If `self.metric` isn't one of the supported names.
        """
        if self.metric == "rmse":
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        if self.metric == "mse":
            return float(mean_squared_error(y_true, y_pred))
        if self.metric == "r2":
            return -float(r2_score(y_true, y_pred))
        if self.metric == "roc_auc":
            return -float(roc_auc_score(y_true, y_pred))
        if self.metric == "log_loss":
            return float(log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7)))
        if self.metric == "brier":
            return float(np.mean((y_true - np.clip(y_pred, 1e-7, 1 - 1e-7)) ** 2))
        raise ValueError(f"Unknown metric: {self.metric!r}")

    def greedy_selection(
        self, predictions_dict: Dict[str, np.ndarray], y_true: np.ndarray,
        n_iterations: int = 100, sorted_init: int = 5, n_bags: int = 10, bag_fraction: float = 0.5,
    ) -> Dict[str, float]:
        """Bagged Caruana-style greedy forward selection.

        Args:
            predictions_dict: model name -> prediction array (shape [n_samples]).
            y_true: Ground-truth values (shape [n_samples]).
            n_iterations: Max greedy iterations per bag.
            sorted_init: Number of top individually-scoring models to seed each
                bag's ensemble with (0 disables, starting from empty instead).
            n_bags: Number of bagged reruns whose selections are merged by
                averaging their (normalized) weights; use 1 to disable bagging.
            bag_fraction: Fraction of the model pool sampled (without
                replacement) into each bag.

        Returns:
            model name -> normalized ensemble weight (sums to 1), merged
            across bags. Empty if the pool is empty.
        """
        model_names = list(predictions_dict.keys())
        if n_bags <= 1:
            return self._single_selection(predictions_dict, y_true, n_iterations, sorted_init)

        n_models = len(model_names)
        bag_ensembles = []
        for bag_idx in range(n_bags):
            rng = np.random.RandomState(bag_idx)
            bag_models = rng.choice(model_names, size=max(1, int(n_models * bag_fraction)), replace=False)
            bag_preds = {k: v for k, v in predictions_dict.items() if k in bag_models}
            bag_ensembles.append(self._single_selection(bag_preds, y_true, n_iterations, sorted_init))

        merged: Dict[str, float] = defaultdict(float)
        for ensemble in bag_ensembles:
            for model, weight in ensemble.items():
                merged[model] += weight
        total = sum(merged.values())
        return {k: v / total for k, v in merged.items()} if total > 0 else {}

    def _single_selection(
        self, predictions_dict: Dict[str, np.ndarray], y_true: np.ndarray,
        n_iterations: int, sorted_init: int,
    ) -> Dict[str, float]:
        """One (unbagged) greedy forward selection run; see `greedy_selection`."""
        model_names = list(predictions_dict.keys())

        if sorted_init > 0:
            initial_scores = {name: self.compute_metric(y_true, predictions_dict[name]) for name in model_names}
            init_models = [name for name, _ in sorted(initial_scores.items(), key=lambda x: x[1])[:sorted_init]]
            ensemble = {m: 1 for m in init_models}
            current_pred = np.mean([predictions_dict[m] for m in init_models], axis=0)
            current_score = self.compute_metric(y_true, current_pred)
        else:
            ensemble = {}
            current_pred = np.zeros_like(y_true, dtype=float)
            current_score = float("inf")

        best_overall_score = current_score
        best_overall_ensemble = ensemble.copy()

        for _ in range(n_iterations):
            best_model, best_score = None, current_score
            for model_name in model_names:
                ensemble_size = sum(ensemble.values())
                if ensemble_size == 0:
                    new_pred = predictions_dict[model_name]
                else:
                    new_pred = (current_pred * ensemble_size + predictions_dict[model_name]) / (ensemble_size + 1)
                score = self.compute_metric(y_true, new_pred)
                if score < best_score:
                    best_score, best_model = score, model_name

            if best_model is None:
                continue   # no improving model this round; kept iterating (with replacement) regardless

            ensemble[best_model] = ensemble.get(best_model, 0) + 1
            ensemble_size = sum(ensemble.values())
            current_pred = (current_pred * (ensemble_size - 1) + predictions_dict[best_model]) / ensemble_size
            current_score = best_score
            if current_score < best_overall_score:
                best_overall_score = current_score
                best_overall_ensemble = ensemble.copy()

        total = sum(best_overall_ensemble.values())
        return {k: v / total for k, v in best_overall_ensemble.items()} if total > 0 else {}


def save_ensemble_weights(
    weights: Dict[str, float], method_name: str, metric_value: float, out_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist one ensemble strategy's model weights to JSON, for provenance/reuse.

    Args:
        weights: model_key -> weight, as returned by `EnsembleSelector` or
            built directly for the fixed strategies (best_single/uniform/best_backend).
        method_name: Strategy name, e.g. "caruana"; used in the filename.
        metric_value: The strategy's RMSE on its evaluation split (see
            `run_ensemble_strategies`), recorded alongside the weights.
        out_dir: Directory to write into.
        metadata: Optional extra fields to record (e.g. split sizes).

    Returns:
        Path written to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"ensemble_weights_{method_name}.json"
    doc = {
        "created_at": datetime.now().isoformat(),
        "method": method_name,
        "metric_name": "rmse",
        "metric_value": float(metric_value),
        "n_models": len(weights),
        "weights": {k: float(v) for k, v in sorted(weights.items(), key=lambda x: -x[1])},
    }
    if metadata:
        doc["metadata"] = metadata
    with open(filepath, "w") as f:
        json.dump(doc, f, indent=2)
    return filepath


def _uncertainty_metrics(
    std: Optional[np.ndarray], pred: np.ndarray, y_eval: np.ndarray,
    clf_threshold: float, n_bins: int = 10,
) -> Dict[str, float]:
    """Uncertainty and calibration metrics for one predictor's per-sample sigma.

    The scoring is deliberately agnostic about where `std` came from -- an
    ensemble's spread across members or an evidential head's own predicted
    variance -- so the two are directly comparable on the same rows. A sigma is
    checked two ways: rank correlation with the actual error (does a larger
    sigma mean a bigger error?) and k-sigma coverage (does `|error| <= k*sigma`
    happen about as often as a calibrated k-sigma interval predicts --
    ~68.3/95.5/99.7% for k=1/2/3?). Separately, Expected and Maximum
    Calibration Error treat the raw [0, 1] prediction as a probability for the
    `clf_threshold`-derived binary label.

    Args:
        std: Per-sample predictive standard deviation, shape (n_eval,), or None
            when undefined (e.g. the spread of a one-model "ensemble"), which
            makes every sigma-based metric NaN.
        pred: The predictor's point estimate, shape (n_eval,).
        y_eval: Ground truth, same order/length as `pred`.
        clf_threshold: Cut used to derive the binary label for ECE/MCE.
        n_bins: Number of confidence bins for ECE/MCE.

    Returns:
        Dict with `mean_std`, `uncertainty_error_rho`, `coverage_1sigma`,
        `coverage_2sigma`, `coverage_3sigma` (all NaN when `std` is None), plus
        `clf_ece` and `clf_mce` (always computed -- calibration of the point
        estimate needs no sigma at all).
    """
    abs_error = np.abs(y_eval - pred)
    out: Dict[str, float] = {}

    if std is None:
        out["mean_std"] = float("nan")
        out["uncertainty_error_rho"] = float("nan")
        for k in (1, 2, 3):
            out[f"coverage_{k}sigma"] = float("nan")
    else:
        out["mean_std"] = float(std.mean())
        out["uncertainty_error_rho"] = (
            float(spearmanr(std, abs_error)[0]) if np.ptp(std) > 0 else float("nan")
        )
        for k in (1, 2, 3):
            out[f"coverage_{k}sigma"] = float(np.mean(abs_error <= k * std))

    y_bin = (y_eval >= clf_threshold).astype(float)
    conf = np.clip(pred, 1e-7, 1 - 1e-7)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    for i in range(n_bins):
        mask = (conf >= bin_edges[i]) & (conf < bin_edges[i + 1])
        if not mask.any():
            continue
        gap = abs(y_bin[mask].mean() - conf[mask].mean())
        ece += (mask.sum() / len(conf)) * gap
        mce = max(mce, gap)
    out["clf_ece"] = float(ece)
    out["clf_mce"] = float(mce)
    return out


def _ensemble_uncertainty_metrics(
    member_preds: np.ndarray, ensemble_pred: np.ndarray, y_eval: np.ndarray,
    clf_threshold: float, n_bins: int = 10,
) -> Dict[str, float]:
    """`_uncertainty_metrics` with the sigma read off an ensemble's member spread.

    Args:
        member_preds: One row per distinct member model's raw prediction on
            the evaluation split, shape (n_members, n_eval). Unweighted --
            spread is read across the base learners themselves, not weighted
            by how many times e.g. Caruana repeated one.
        ensemble_pred: The strategy's final (weighted) prediction, shape (n_eval,).
        y_eval: Ground truth, same order/length as `ensemble_pred`.
        clf_threshold: Cut used to derive the binary label for ECE/MCE.
        n_bins: Number of confidence bins for ECE/MCE.

    Returns:
        See `_uncertainty_metrics`; the sigma-based entries are NaN when fewer
        than 2 distinct members contribute, since spread is undefined for one.
    """
    std = member_preds.std(axis=0) if member_preds.shape[0] >= 2 else None
    return _uncertainty_metrics(std, ensemble_pred, y_eval, clf_threshold, n_bins)


def _format_composition(weights: Dict[str, float]) -> str:
    """Compact per-backend model-count string, e.g. 'GNN:13,MLP:6,XGB:7'."""
    counts = Counter(k.split("_seed")[0] for k in weights)
    return ",".join(f"{b}:{n}" for b, n in sorted(counts.items(), key=lambda x: -x[1]))


def _print_ensemble_table(df: pd.DataFrame, compositions: Dict[str, str]) -> None:
    """Print a compact command-line summary table for `run_ensemble_strategies`'s result.

    Args:
        df: The metrics DataFrame `run_ensemble_strategies` returns (indexed
            by strategy name).
        compositions: strategy name -> per-backend model-count string, for the
            composition column (see `_format_composition`).
    """
    display = pd.DataFrame({
        "n_models":    df["n_models"].astype(int),
        "composition": [compositions.get(s, "") for s in df.index],
        "rmse":        df["rmse"].round(4),
        "r2":          df["r2"].round(3),
        "mean_std":    df["mean_std"].round(4),
        "unc_err_rho": df["uncertainty_error_rho"].round(3),
        "cov_1sig_%":  (df["coverage_1sigma"] * 100).round(1),
        "cov_2sig_%":  (df["coverage_2sigma"] * 100).round(1),
        "cov_3sig_%":  (df["coverage_3sigma"] * 100).round(1),
        "ece_%":       (df["clf_ece"] * 100).round(1),
        "mce_%":       (df["clf_mce"] * 100).round(1),
    }, index=df.index)
    print("\n" + display.to_string())
    print("  (expected coverage: 68.3% / 95.5% / 99.7% at 1σ/2σ/3σ; NaN = undefined for a "
          "1-model ensemble; ECE/MCE treat the raw prediction as P(synthesizable))")


def run_ensemble_strategies(
    fold_predictions: Dict[str, np.ndarray], y_true: np.ndarray, out_dir: Path,
    clf_threshold: float, caruana_sel_frac: float = 0.2,
    caruana_iterations: int = 100, random_seed: int = 42,
    extra_predictors: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    smiles: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Score best_single / uniform / best_backend / Caruana ensembles over the fold-model pool.

    Mirrors the standard "fair ensemble comparison" recipe: every strategy is
    scored on the same held-out evaluation split of the test set; only
    Caruana's greedy weight *selection* (which can overfit whatever data it
    sees) is restricted to a separate, smaller selection split, so its
    reported score isn't inflated relative to the other strategies. Each
    strategy is also scored on uncertainty/calibration -- see
    `_ensemble_uncertainty_metrics`.

    Args:
        fold_predictions: "{label}_seed{seed}_fold{fold}" -> 1-D prediction
            array (see `predict_fold_models`).
        y_true: 1-D ground-truth array, same order/length as each prediction.
        out_dir: Directory to write `ensemble_weights_*.json` and the summary
            CSV into.
        clf_threshold: Binarisation cut forwarded to `compute_all_metrics` and
            `_ensemble_uncertainty_metrics`'s ECE/MCE.
        caruana_sel_frac: Fraction of the test set reserved for Caruana's
            greedy selection; the rest is the shared evaluation split every
            strategy (including best_single/uniform/best_backend) is scored on.
        caruana_iterations: Greedy selection iterations (see `EnsembleSelector`).
        random_seed: Seed for the selection/evaluation split.
        extra_predictors: Optional single models that carry their own
            uncertainty, as name -> (prediction, sigma) over the FULL test set
            (see `evidential_predictors`). They take no part in the ensembles;
            they are sliced onto the same evaluation split and scored through
            the same metrics, so their uncertainty lands in the same table --
            on the same rows -- as the ensembles' member spread.
        smiles: SMILES strings, same order/length as `y_true`, i.e. the same
            row order as the test CSV `fold_predictions`/`y_true` were built
            from. When given, the rows Caruana's greedy selection actually saw
            (`sel_idx` below) are written to `{out_dir}/caruana_selection_smiles.csv`
            for provenance. None skips this (e.g. no test CSV available).

    Returns:
        DataFrame of metrics (one row per strategy and per extra predictor,
        from `compute_all_metrics` plus `n_models` and the
        `_uncertainty_metrics` columns), indexed by strategy name; also written
        to `{out_dir}/ensemble_strategies.csv`, reduced to
        `{out_dir}/uncertainty_comparison.csv`, and printed as a table.

    Raises:
        RuntimeError: If `fold_predictions` is empty.
        ValueError: If an `extra_predictors` array doesn't cover the same rows
            as `y_true`, or if `smiles` is given but doesn't cover the same
            rows as `y_true`.
    """
    if not fold_predictions:
        raise RuntimeError("No fold-model predictions to build ensembles from.")

    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    perm = rng.permutation(n)
    n_sel = max(1, int(n * caruana_sel_frac))
    sel_idx, eval_idx = perm[:n_sel], perm[n_sel:]
    print(f"\n  Caruana selection split: {len(sel_idx)} selection / {len(eval_idx)} evaluation samples "
          f"(every strategy is scored on the {len(eval_idx)}-sample evaluation split)")

    if smiles is not None:
        if len(smiles) != n:
            raise ValueError(
                f"smiles covers {len(smiles)} rows but fold predictions cover {n}; "
                "both must come from the same test CSV."
            )
        sel_smiles_path = out_dir / "caruana_selection_smiles.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "row_index": sel_idx,
            "smiles": np.asarray(smiles)[sel_idx],
            "y_true": y_true[sel_idx],
        }).sort_values("row_index").to_csv(sel_smiles_path, index=False)
        print(f"  Caruana selection SMILES ({len(sel_idx)} rows) -> {sel_smiles_path}")

    y_eval = y_true[eval_idx]
    eval_preds = {k: v[eval_idx] for k, v in fold_predictions.items()}

    def _score(pred: np.ndarray) -> Dict[str, float]:
        return compute_all_metrics(y_eval, pred, threshold=clf_threshold)

    individual_rmse = {k: _score(v)["rmse"] for k, v in eval_preds.items()}
    ranked = sorted(individual_rmse, key=individual_rmse.get)

    # best_backend: not the individually-best N fold models (which tends to just
    # be one backend's folds anyway), but the full 5x5 fold set of whichever
    # single backend scores best as a group, uniformly averaged.
    backends = sorted({k.split("_seed")[0] for k in eval_preds})
    backend_keys = {b: [k for k in eval_preds if k.split("_seed")[0] == b] for b in backends}
    backend_rmse = {b: _score(np.mean([eval_preds[k] for k in ks], axis=0))["rmse"]
                    for b, ks in backend_keys.items()}
    best_backend = min(backend_rmse, key=backend_rmse.get)
    backend_summary = ", ".join(f"{b}={r:.4f}" for b, r in sorted(backend_rmse.items(), key=lambda x: x[1]))
    print(f"  Backend RMSE (uniform avg of all its fold models): {backend_summary} "
          f"-> best_backend picks {best_backend}")

    strategies: Dict[str, Dict[str, float]] = {
        "best_single": {ranked[0]: 1.0},
        "uniform": {k: 1.0 / len(eval_preds) for k in eval_preds},
        "best_backend": {k: 1.0 / len(backend_keys[best_backend]) for k in backend_keys[best_backend]},
    }

    selector = EnsembleSelector(metric="rmse")
    sel_preds = {k: v[sel_idx] for k, v in fold_predictions.items()}
    strategies["caruana"] = selector.greedy_selection(sel_preds, y_true[sel_idx], n_iterations=caruana_iterations)

    print("\n── Ensemble strategies (scored on the shared evaluation split) ─────────")
    rows, compositions = [], {}
    for name, weights in strategies.items():
        pred = np.zeros(len(y_eval))
        for k, w in weights.items():
            pred += eval_preds[k] * w
        metrics = _score(pred)
        member_preds = np.array([eval_preds[k] for k in weights])
        metrics.update(_ensemble_uncertainty_metrics(member_preds, pred, y_eval, clf_threshold))
        metrics["n_models"] = len(weights)
        metrics["strategy"] = name
        metrics["uncertainty_source"] = "member spread"
        rows.append(metrics)
        compositions[name] = _format_composition(weights)

        by_backend = Counter(k.split("_seed")[0] for k in weights)
        print(f"  {name:<12} n_models={len(weights):<3} rmse={metrics['rmse']:.4f}  "
              f"r2={metrics['r2']:.4f}  composition={dict(by_backend)}")

        save_ensemble_weights(weights, name, metrics["rmse"], out_dir, metadata={
            "n_selection_samples": len(sel_idx), "n_evaluation_samples": len(eval_idx),
        })

    # single models with a predicted uncertainty of their own (the evidential
    # GNN), scored on the very same evaluation rows as the ensembles above
    for name, (full_pred, full_std) in (extra_predictors or {}).items():
        if len(full_pred) != n or len(full_std) != n:
            raise ValueError(
                f"extra predictor {name!r} covers {len(full_pred)} rows but the fold "
                f"predictions cover {n}; both must come from the same test CSV."
            )
        pred, std = np.asarray(full_pred)[eval_idx], np.asarray(full_std)[eval_idx]
        metrics = _score(pred)
        metrics.update(_uncertainty_metrics(std, pred, y_eval, clf_threshold))
        metrics["n_models"] = 1
        metrics["strategy"] = name
        metrics["uncertainty_source"] = "predicted"
        rows.append(metrics)
        compositions[name] = "evidential:1"
        print(f"  {name:<12} n_models=1   rmse={metrics['rmse']:.4f}  "
              f"r2={metrics['r2']:.4f}  uncertainty=predicted ({compositions[name]})")

    df = pd.DataFrame(rows).set_index("strategy")
    df.to_csv(out_dir / "ensemble_strategies.csv")
    _print_ensemble_table(df, compositions)

    # focused view of the same rows: how well does each sigma -- however it was
    # obtained -- track the error it is supposed to predict?
    unc_cols = ["uncertainty_source", "n_models", "rmse", "mean_std",
                "uncertainty_error_rho", "coverage_1sigma", "coverage_2sigma",
                "coverage_3sigma", "clf_ece", "clf_mce"]
    df[unc_cols].sort_values("uncertainty_error_rho", ascending=False).to_csv(
        out_dir / "uncertainty_comparison.csv")
    return df


# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    """CLI entry point: load CV fold metrics, run AutoRank, and optionally test-eval."""
    args    = parse_args()
    cfg     = ModelsConfig.load(args.config)
    cv_dir     = args.output_root / "cv"
    models_dir = args.output_root / "models"
    out_dir    = args.output_root / "results" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading CV fold metrics...")
    wide, long = build_cv_frames(
        args.models, cv_dir, cfg.cross_validation.seeds, cfg.cross_validation.n_folds, args.rank_metric,
    )
    if wide.columns.get_level_values("model").nunique() >= 2:
        run_autorank(wide, args.rank_metric, out_dir)   # AutoRank needs >=2 models to compare
    else:
        print("Only one model — skipping AutoRank (needs >=2 to compare).")

    # artifacts for the plotting script/notebook
    with open(out_dir / "cv_fold_scores.pkl", "wb") as f:
        pickle.dump(_fold_scores_by_metric(wide), f)
    long.to_csv(out_dir / "cv_scores_long.csv", index=False)
    wide.to_csv(out_dir / "cv_scores_wide.csv")
    build_cv_metrics_summary(wide).to_csv(out_dir / "cv_metrics_summary.csv", index=False)

    test_results: Optional[Dict[str, Dict[str, float]]] = None
    n_test: Optional[int] = None
    if args.test_csv:
        smiles_col = args.smiles_col or cfg.molecule_col
        target_col = args.target_col or cfg.target

        test_pred_path    = out_dir / "test_predictions.pkl"
        test_metrics_path = out_dir / "test_metrics.csv"
        test_cache_hit    = (
            test_pred_path.exists() and test_metrics_path.exists() and not args.force_recompute
        )
        fold_cache_hit = args.ensemble and (out_dir / FOLD_PRED_FILENAME).exists() and not args.force_predict

        # evaluate_test/predict_fold_models are the only two consumers of
        # fingerprints/descriptors below -- compute them (a ~minute-long step)
        # only if at least one of the two will actually run rather than hit its
        # own cache, and share the one computation between whichever do.
        will_run_evaluate_test      = not test_cache_hit
        will_run_predict_fold_models = args.ensemble and not fold_cache_hit
        need_features = will_run_evaluate_test or will_run_predict_fold_models
        X_fp, X_desc = (
            _test_features(pd.read_csv(args.test_csv)[smiles_col].tolist(), cfg)
            if need_features else (None, None)
        )

        if test_cache_hit:
            print(f"\n[2/3] Found cached final-model test predictions at {test_pred_path} "
                  f"-- skipping load/predict (pass --force-recompute to redo).")
            with open(test_pred_path, "rb") as f:
                predictions = pickle.load(f)
        else:
            print(f"\n[2/3] Evaluating final models on held-out test set "
                  f"(smiles_col={smiles_col!r}, target_col={target_col!r})...")
            predictions, results = evaluate_test(
                args.models, args.test_csv, models_dir, cfg,
                smiles_col, target_col, cfg.hpo.classification_threshold,
                X_fp=X_fp, X_desc=X_desc, device=args.device,
            )
            with open(test_pred_path, "wb") as f:
                pickle.dump(predictions, f)
            pd.DataFrame(results).T.round(4).to_csv(test_metrics_path)

        test_results = pd.read_csv(test_metrics_path, index_col=0).to_dict(orient="index")
        n_test = len(next(iter(predictions.values()))["y_true"]) if predictions else None

        if args.ensemble:
            print("\n[2b/3] Predicting test set with every 5x5-CV fold model...")
            fold_predictions, y_true_folds = predict_fold_models(
                args.models, args.test_csv, cv_dir, cfg, smiles_col, target_col,
                cfg.cross_validation.seeds, cfg.cross_validation.n_folds,
                out_dir, force=args.force_predict, X_fp=X_fp, X_desc=X_desc,
                device=args.device,
            )

            extra_predictors = None
            if args.evidential_model:
                print(f"\n[2b'/3] Predicting test set with the evidential model "
                      f"'{args.evidential_model}' (its own uncertainty)...")
                evidential = predict_evidential(
                    args.evidential_model, args.test_csv, models_dir,
                    smiles_col, target_col, out_dir,
                    device=args.device, force=args.force_predict,
                )
                extra_predictors = evidential_predictors(evidential)

            print("\n[2c/3] Scoring ensemble strategies...")
            ensemble_smiles = pd.read_csv(args.test_csv)[smiles_col].tolist()
            run_ensemble_strategies(
                fold_predictions, y_true_folds, out_dir, cfg.hpo.classification_threshold,
                caruana_sel_frac=args.ensemble_caruana_frac,
                caruana_iterations=args.ensemble_iterations,
                extra_predictors=extra_predictors,
                smiles=ensemble_smiles,
            )
    else:
        print("\n[2/3] No --test-csv given; skipping test-set evaluation.")

    print("\n[2d/3] Building LaTeX summary table...")
    report_latex_summary(
        wide, out_dir, test_results=test_results,
        clf_threshold=cfg.hpo.classification_threshold, n_test=n_test,
    )

    print(f"\n[3/3] Done. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
