"""
evaluation.py
=============
Statistical comparison of the surrogate models (xgb / mlp / gnn) from their
per-fold CV scores, and (optionally) evaluation of the final models on a
held-out test set. Exports the artifacts a separate plotting script/notebook
consumes.

Every metric written per fold by train.py (see protac_synth.models.metrics
.compute_all_metrics) is loaded and reported, not just R2:
    {CV_DIR}/{prefix}/score_seed{seed}_fold{fold}.json
        -> {"seed", "fold_idx", "r2", "rmse", ..., "clf_*", "n_samples", "objective", ...}

Usage
-----
    # CV comparison only (no test set yet):
    python evaluation.py --models xgb_v1 mlp_v1 gnn_v1

    # also evaluate final models on a held-out test CSV, against the config
    # the runs were actually trained with (sets target/molecule_col/fp params):
    python evaluation.py --models xgb_v1 mlp_v1 gnn_v1 \
        --config config/models_config_routes.yaml \
        --test-csv data/sets/routes_test.csv --output-root outputs
"""
import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import autorank
import numpy as np
import pandas as pd
from scipy.stats import levene

sys.path.append(str(Path(__file__).resolve().parents[2]))   # -> chem_utils

from protac_synth.models.config import ModelsConfig
from protac_synth.models.metrics import compute_all_metrics

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
    ap.add_argument("--rank-metric", default="r2",
                    help="metric driving the paired AutoRank comparison (default: r2); "
                         "every metric is still reported in cv_scores_wide/long.csv "
                         "regardless of this choice")
    ap.add_argument("--out", default="comparison",
                    help="output subfolder name under results/")
    return ap.parse_args()


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
def _load_model(prefix: str, models_dir: Path) -> Any:
    """Load the saved final model for a prefix using its class's loader.

    Args:
        prefix: Model prefix, e.g. "xgb_v1"; the leading token before "_"
            selects the loader class.
        models_dir: Directory holding one subfolder per prefix.

    Returns:
        The loaded model instance.

    Raises:
        ValueError: If the prefix's leading token isn't xgb/mlp/gnn.
    """
    kind = prefix.split("_")[0]
    base = str(models_dir / prefix / f"{prefix}_final")
    if kind == "xgb":
        from protac_synth.models.xgb.model import XGBoostRegressor
        return XGBoostRegressor.load(base)
    if kind == "mlp":
        from protac_synth.models.mlp.model import TorchMLPRegressor
        return TorchMLPRegressor.load(base)
    if kind == "gnn":
        from protac_synth.models.gnn.model import CheMeleonRegressor
        return CheMeleonRegressor.load(base)
    raise ValueError(f"Unknown model kind: {kind}")


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
    from protac_synth.chem_utils import compute_descriptors, compute_fingerprints, standardize_all
    mols   = standardize_all(smiles)
    X_fp   = compute_fingerprints(mols, cfg.features.fp_size, cfg.features.fp_radius) if cfg.features.use_fingerprints else None
    X_desc = compute_descriptors(mols) if cfg.features.use_descriptors else None
    return X_fp, X_desc


def evaluate_test(
    model_prefixes: List[str], test_csv: str, models_dir: Path, cfg: ModelsConfig,
    smiles_col: str, target_col: str, clf_threshold: float,
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
    X_fp, X_desc = _test_features(smiles, cfg)

    predictions, results = {}, {}
    for prefix in model_prefixes:
        label = _label(prefix)
        try:
            model  = _load_model(prefix, models_dir)
            y_pred = model.predict(smiles, X_fp=X_fp, X_desc=X_desc)   # GNN ignores X_fp/X_desc
            predictions[label] = {"y_pred": y_pred, "y_true": y_true}
            results[label]     = compute_all_metrics(y_true, y_pred, threshold=clf_threshold)
            m = results[label]
            print(f"  {label:<6} r2={m['r2']:.3f}  rmse={m['rmse']:.3f}  mae={m['mae']:.3f}  "
                  f"spearman_rho={m['spearman_rho']:.3f}  clf_roc_auc={m['clf_roc_auc']:.3f}")
        except Exception as e:
            print(f"  WARNING: test eval failed for '{prefix}': {e}")
    return predictions, results


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
        print(f"\n[2/3] Evaluating final models on held-out test set "
              f"(smiles_col={smiles_col!r}, target_col={target_col!r})...")
        predictions, results = evaluate_test(
            args.models, args.test_csv, models_dir, cfg,
            smiles_col, target_col, cfg.hpo.classification_threshold,
        )
        with open(out_dir / "test_predictions.pkl", "wb") as f:
            pickle.dump(predictions, f)
        pd.DataFrame(results).T.round(4).to_csv(out_dir / "test_metrics.csv")
    else:
        print("\n[2/3] No --test-csv given; skipping test-set evaluation.")

    print(f"\n[3/3] Done. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
