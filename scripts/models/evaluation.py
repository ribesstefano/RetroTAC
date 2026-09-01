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
from retrotac.models.metrics import compute_all_metrics

# ── paths (anchored to repo root, same as train.py's own defaults) ──────────
_ROOT           = Path(__file__).parents[3]                 # PROTAC-Synthesizability/
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
    args = ap.parse_args()
    if args.ensemble and not args.test_csv:
        ap.error("--ensemble requires --test-csv")
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
def _load_model_from_path(prefix: str, base: str) -> Any:
    """Load a saved model from an explicit base path using its class's loader.

    Shared by `_load_model` (the aggregated `{prefix}_final`) and
    `predict_fold_models` (each individual `model_seed{seed}_fold{fold}`) --
    both are the same backend classes, saved via the same `save(path)`
    convention, just under different base paths.

    Args:
        prefix: Model prefix, e.g. "xgb_v1"; the leading token before "_"
            selects the loader class.
        base: Base path (no extension) passed to the class's own `load()`.

    Returns:
        The loaded model instance.

    Raises:
        ValueError: If the prefix's leading token isn't xgb/mlp/gnn.
    """
    kind = prefix.split("_")[0]
    if kind == "xgb":
        from retrotac.models.xgb.model import XGBoostRegressor
        return XGBoostRegressor.load(base)
    if kind == "mlp":
        from retrotac.models.mlp.model import TorchMLPRegressor
        return TorchMLPRegressor.load(base)
    if kind == "gnn":
        from retrotac.models.gnn.model import CheMeleonRegressor
        return CheMeleonRegressor.load(base)
    raise ValueError(f"Unknown model kind: {kind}")


def _load_model(prefix: str, models_dir: Path) -> Any:
    """Load the saved final model for a prefix (see `_load_model_from_path`).

    Args:
        prefix: Model prefix, e.g. "xgb_v1".
        models_dir: Directory holding one subfolder per prefix.

    Returns:
        The loaded model instance.
    """
    return _load_model_from_path(prefix, str(models_dir / prefix / f"{prefix}_final"))


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
    from retrotac.chem_utils import compute_descriptors, compute_fingerprints, standardize_all
    mols   = standardize_all(smiles)
    X_fp   = compute_fingerprints(mols, cfg.features.fp_size, cfg.features.fp_radius) if cfg.features.use_fingerprints else None
    X_desc = compute_descriptors(mols) if cfg.features.use_descriptors else None
    return X_fp, X_desc


def evaluate_test(
    model_prefixes: List[str], test_csv: str, models_dir: Path, cfg: ModelsConfig,
    smiles_col: str, target_col: str, clf_threshold: float,
    X_fp: Optional[np.ndarray] = None, X_desc: Optional[np.ndarray] = None,
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
            model = _load_model(prefix, models_dir)
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
                    model = _load_model_from_path(prefix, str(base))
                    y_pred = model.predict(smiles, X_fp=X_fp, X_desc=X_desc)
                    fold_predictions[f"{label}_seed{seed}_fold{fold}"] = np.asarray(y_pred, dtype=float).reshape(-1)
                except Exception as e:
                    print(f"  WARNING: predict failed for {base}: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"fold_predictions": fold_predictions, "y_true": y_true}, f)
    print(f"  Predicted with {len(fold_predictions)} fold models -> cached at {cache_path}")
    return fold_predictions, y_true


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


def _ensemble_uncertainty_metrics(
    member_preds: np.ndarray, ensemble_pred: np.ndarray, y_eval: np.ndarray,
    clf_threshold: float, n_bins: int = 10,
) -> Dict[str, float]:
    """Uncertainty and calibration metrics for one ensemble's predictions.

    Per-sample spread across the distinct member models stands in for
    predictive uncertainty, checked two ways: rank correlation with the actual
    error (does higher spread mean bigger error?) and k-sigma coverage (does
    `|error| <= k * std` happen about as often as a calibrated k-sigma
    interval predicts -- ~68.3/95.5/99.7% for k=1/2/3?). Separately, Expected
    and Maximum Calibration Error treat the ensemble's raw [0, 1] prediction
    as a probability for the `clf_threshold`-derived binary label.

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
        Dict with `mean_std`, `uncertainty_error_rho`, `coverage_1sigma`,
        `coverage_2sigma`, `coverage_3sigma` (all NaN when fewer than 2 member
        models -- spread is undefined for a single model), plus `clf_ece` and
        `clf_mce` (always computed -- calibration doesn't need an ensemble).
    """
    abs_error = np.abs(y_eval - ensemble_pred)
    out: Dict[str, float] = {}

    if member_preds.shape[0] >= 2:
        std = member_preds.std(axis=0)
        out["mean_std"] = float(std.mean())
        out["uncertainty_error_rho"] = (
            float(spearmanr(std, abs_error)[0]) if np.ptp(std) > 0 else float("nan")
        )
        for k in (1, 2, 3):
            out[f"coverage_{k}sigma"] = float(np.mean(abs_error <= k * std))
    else:
        out["mean_std"] = float("nan")
        out["uncertainty_error_rho"] = float("nan")
        for k in (1, 2, 3):
            out[f"coverage_{k}sigma"] = float("nan")

    y_bin = (y_eval >= clf_threshold).astype(float)
    conf = np.clip(ensemble_pred, 1e-7, 1 - 1e-7)
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


def _format_composition(weights: Dict[str, float]) -> str:
    """Compact per-backend model-count string, e.g. 'GNN:13,MLP:6,XGB:7'."""
    counts = Counter(k.split("_seed")[0] for k in weights)
    return ",".join(f"{b}:{n}" for b, n in sorted(counts.items(), key=lambda x: -x[1]))


def _print_ensemble_table(df: pd.DataFrame, strategies: Dict[str, Dict[str, float]]) -> None:
    """Print a compact command-line summary table for `run_ensemble_strategies`'s result.

    Args:
        df: The metrics DataFrame `run_ensemble_strategies` returns (indexed
            by strategy name).
        strategies: strategy name -> {model_key: weight}, for the composition column.
    """
    display = pd.DataFrame({
        "n_models":    df["n_models"].astype(int),
        "composition": [_format_composition(strategies[s]) for s in df.index],
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
          "1-model ensemble; ECE/MCE treat the ensemble's raw prediction as P(synthesizable))")


def run_ensemble_strategies(
    fold_predictions: Dict[str, np.ndarray], y_true: np.ndarray, out_dir: Path,
    clf_threshold: float, caruana_sel_frac: float = 0.2,
    caruana_iterations: int = 100, random_seed: int = 42,
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

    Returns:
        DataFrame of metrics (one row per strategy, from `compute_all_metrics`
        plus `n_models` and the `_ensemble_uncertainty_metrics` columns),
        indexed by strategy name; also written to
        `{out_dir}/ensemble_strategies.csv` and printed as a table.

    Raises:
        RuntimeError: If `fold_predictions` is empty.
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
    rows = []
    for name, weights in strategies.items():
        pred = np.zeros(len(y_eval))
        for k, w in weights.items():
            pred += eval_preds[k] * w
        metrics = _score(pred)
        member_preds = np.array([eval_preds[k] for k in weights])
        metrics.update(_ensemble_uncertainty_metrics(member_preds, pred, y_eval, clf_threshold))
        metrics["n_models"] = len(weights)
        metrics["strategy"] = name
        rows.append(metrics)

        by_backend = Counter(k.split("_seed")[0] for k in weights)
        print(f"  {name:<12} n_models={len(weights):<3} rmse={metrics['rmse']:.4f}  "
              f"r2={metrics['r2']:.4f}  composition={dict(by_backend)}")

        save_ensemble_weights(weights, name, metrics["rmse"], out_dir, metadata={
            "n_selection_samples": len(sel_idx), "n_evaluation_samples": len(eval_idx),
        })

    df = pd.DataFrame(rows).set_index("strategy")
    df.to_csv(out_dir / "ensemble_strategies.csv")
    _print_ensemble_table(df, strategies)
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
                X_fp=X_fp, X_desc=X_desc,
            )
            with open(test_pred_path, "wb") as f:
                pickle.dump(predictions, f)
            pd.DataFrame(results).T.round(4).to_csv(test_metrics_path)

        if args.ensemble:
            print("\n[2b/3] Predicting test set with every 5x5-CV fold model...")
            fold_predictions, y_true_folds = predict_fold_models(
                args.models, args.test_csv, cv_dir, cfg, smiles_col, target_col,
                cfg.cross_validation.seeds, cfg.cross_validation.n_folds,
                out_dir, force=args.force_predict, X_fp=X_fp, X_desc=X_desc,
            )
            print("\n[2c/3] Scoring ensemble strategies...")
            run_ensemble_strategies(
                fold_predictions, y_true_folds, out_dir, cfg.hpo.classification_threshold,
                caruana_sel_frac=args.ensemble_caruana_frac,
                caruana_iterations=args.ensemble_iterations,
            )
    else:
        print("\n[2/3] No --test-csv given; skipping test-set evaluation.")

    print(f"\n[3/3] Done. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
