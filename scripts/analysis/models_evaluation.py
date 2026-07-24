"""
models_evaluation.py
=============
Statistical comparison of the surrogate models (xgb / mlp / gnn) from their
per-fold CV scores, and (optionally) evaluation of the final models on a
held-out test set. Exports the artifacts a separate plotting script/notebook
consumes.

Fold scores are read from the per-fold JSON files written by train.py:
    {CV_DIR}/{prefix}/json/score_seed{seed}_fold{fold}.json   -> {"seed","fold_idx","r2"}

Usage
-----
    # CV comparison only (no test set yet):
    python evaluation.py --models xgb_v1 mlp_v1 gnn_v1

    # also evaluate final models on a held-out test CSV:
    python evaluation.py --models xgb_v1 mlp_v1 gnn_v1 --test-csv data/test.csv
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import autorank
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).resolve().parents[2]))   # -> chem_utils

# ── paths + config (anchored to repo root, same as train.py) ────────────────
_ROOT       = Path(__file__).parents[3]                 # PROTAC-Synthesizability/
OUTPUT_ROOT = _ROOT / "data" / "outputs"
CV_DIR      = OUTPUT_ROOT / "cv"
MODELS_DIR  = OUTPUT_ROOT / "models"
RESULTS_DIR = OUTPUT_ROOT / "results"

with open(Path(__file__).resolve().parents[2] / "config" / "models_config.yaml") as f:
    _CFG = yaml.safe_load(f)
CV_SEEDS         = _CFG["cross_validation"]["seeds"]
N_FOLDS          = _CFG["cross_validation"]["n_folds"]
TARGET           = _CFG["target"]
FP_SIZE          = _CFG["features"]["fp_size"]
FP_RADIUS        = _CFG["features"]["fp_radius"]
USE_FINGERPRINTS = _CFG["features"]["use_fingerprints"]
USE_DESCRIPTORS  = _CFG["features"]["use_descriptors"]


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Compare surrogate models from CV fold scores.")
    ap.add_argument("--models", nargs="+", required=True,
                    help="model prefixes to compare, e.g. xgb_v1 mlp_v1 gnn_v1")
    ap.add_argument("--test-csv", default=None,
                    help="optional held-out test CSV (molecule + target columns)")
    ap.add_argument("--out", default="comparison",
                    help="output subfolder name under results/")
    return ap.parse_args()


def _label(prefix: str) -> str:
    """Human label from a prefix, e.g. 'xgb_v1' -> 'XGB'."""
    return prefix.split("_")[0].upper()


# ── Step 1: load per-fold CV scores from JSON ───────────────────────────────
def load_fold_scores(prefix: str) -> dict:
    """Return {(seed, fold): r2} for one model prefix.
    Files are flat under cv/{prefix}/ (distinguished by name/extension:
    score_*.json, best_params_*.json, trials_*.csv, *.db)."""
    base   = CV_DIR / prefix
    scores = {}
    for seed in CV_SEEDS:
        for fold in range(N_FOLDS):
            p = base / f"score_seed{seed}_fold{fold}.json"
            if p.exists():
                with open(p) as f:
                    scores[(seed, fold)] = json.load(f)["r2"]
    return scores


def build_cv_frames(model_prefixes):
    """Build the paired wide frame (for AutoRank) and a long frame (for plotting)."""
    per_model = {}                    # label -> {(seed,fold): r2}
    for prefix in model_prefixes:
        s = load_fold_scores(prefix)
        if not s:
            print(f"  WARNING: no fold scores found for '{prefix}' — skipping.")
            continue
        per_model[_label(prefix)] = s

    if not per_model:
        raise RuntimeError("No fold scores found for any model.")

    # common (seed,fold) keys present in ALL models -> proper paired comparison
    common = set.intersection(*(set(s.keys()) for s in per_model.values()))
    common = sorted(common)
    if not common:
        raise RuntimeError("No (seed,fold) folds are shared across all models.")

    # wide: columns = models, rows = folds (paired)
    wide = pd.DataFrame(
        {label: [per_model[label][k] for k in common] for label in per_model},
        index=pd.MultiIndex.from_tuples(common, names=["seed", "fold"]),
    )

    # long: for rm-Tukey / boxplots, with a paired-sample id (cv_cycle)
    long_rows = []
    for cycle, (seed, fold) in enumerate(common):
        for label in per_model:
            long_rows.append({"method": label, "seed": seed, "fold": fold,
                              "cv_cycle": cycle, "r2": per_model[label][(seed, fold)]})
    long = pd.DataFrame(long_rows)

    print(f"\nLoaded {len(per_model)} models over {len(common)} paired folds.")
    for label in wide.columns:
        print(f"  {label:<6} mean R2 = {wide[label].mean():.4f} ± {wide[label].std():.4f}")
    return wide, long


# ── Step 2: AutoRank statistical comparison ─────────────────────────────────
def run_autorank(wide: pd.DataFrame, out_dir: Path):
    print("\n── AutoRank ─────────────────────────────────────────────────────────")
    result = autorank.autorank(wide, alpha=0.05, verbose=False)
    autorank.create_report(result)

    # persist the ranking table for the plotting script (figures are made in plots.py)
    if getattr(result, "rankdf", None) is not None:
        result.rankdf.to_csv(out_dir / "autorank_rankdf.csv")
    return result


# ── Step 3 (optional): evaluate final models on a held-out test set ─────────
def _load_model(prefix: str):
    """Load the saved final model for a prefix using its class's loader."""
    kind = prefix.split("_")[0]
    base = str(MODELS_DIR / prefix / f"{prefix}_final")
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


def _test_features(smiles):
    from protac_synth.chem_utils import standardize_all, compute_fingerprints, compute_descriptors
    mols   = standardize_all(smiles)
    X_fp   = compute_fingerprints(mols, FP_SIZE, FP_RADIUS) if USE_FINGERPRINTS else None
    X_desc = compute_descriptors(mols) if USE_DESCRIPTORS else None
    return X_fp, X_desc


def _metrics(y_true, y_pred):
    n = y_true.shape[1]
    return {
        "R2":   float(np.mean([r2_score(y_true[:, i], y_pred[:, i]) for i in range(n)])),
        "RMSE": float(np.mean([np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(n)])),
        "MAE":  float(np.mean([mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(n)])),
    }


def evaluate_test(model_prefixes, test_csv):
    df = pd.read_csv(test_csv)
    smiles = df["molecule"].tolist()
    y_true = df[[TARGET]].values
    X_fp, X_desc = _test_features(smiles)

    predictions, results = {}, {}
    for prefix in model_prefixes:
        label = _label(prefix)
        try:
            model  = _load_model(prefix)
            y_pred = model.predict(smiles, X_fp=X_fp, X_desc=X_desc)   # GNN ignores X_fp/X_desc
            predictions[label] = {"y_pred": y_pred, "y_true": y_true}
            results[label]     = _metrics(y_true, y_pred)
            m = results[label]
            print(f"  {label:<6} R2={m['R2']:.3f}  RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}")
        except Exception as e:
            print(f"  WARNING: test eval failed for '{prefix}': {e}")
    return predictions, results


# ── main ────────────────────────────────────────────────────────────────────
def main():
    args    = parse_args()
    out_dir = RESULTS_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading CV fold scores...")
    wide, long = build_cv_frames(args.models)
    if wide.shape[1] >= 2:
        run_autorank(wide, out_dir)               # AutoRank needs >=2 models to compare
    else:
        print("Only one model — skipping AutoRank (needs >=2 to compare).")

    # artifacts for the plotting script/notebook
    with open(out_dir / "cv_fold_scores.pkl", "wb") as f:
        pickle.dump({c: wide[c].tolist() for c in wide.columns}, f)
    long.to_csv(out_dir / "cv_scores_long.csv", index=False)
    wide.to_csv(out_dir / "cv_scores_wide.csv")

    if args.test_csv:
        print("\n[2/3] Evaluating final models on held-out test set...")
        predictions, results = evaluate_test(args.models, args.test_csv)
        with open(out_dir / "test_predictions.pkl", "wb") as f:
            pickle.dump(predictions, f)
        pd.DataFrame(results).T.round(4).to_csv(out_dir / "test_metrics.csv")
    else:
        print("\n[2/3] No --test-csv given; skipping test-set evaluation.")

    print(f"\n[3/3] Done. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()