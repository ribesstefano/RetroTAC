"""
train.py
========
5x5 nested scaffold cross-validation for the PROTAC synthesizability
surrogate models (xgb / mlp / gnn), model-agnostic.

Two run modes (dispatched in main):
  * single-fold : --seed S --fold F  -> tune + evaluate ONE outer fold, save JSON
  * aggregate   : --aggregate        -> collect the 25 fold JSONs, retrain final

Run one fold:
    python train.py --model xgb --input data.csv --seed 0 --fold 0 --prefix v1
Aggregate after all 25 folds:
    python train.py --model xgb --input data.csv --aggregate --prefix v1
"""
import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
import numpy as np
import optuna
import pandas as pd
from optuna.trial import FixedTrial

from mol_utils import (
    standardize_all, compute_fingerprints, compute_descriptors, get_scaffold,
)

with open(Path(__file__).parent / "models_config.yaml") as f:
    _CFG = yaml.safe_load(f)

FP_SIZE          = _CFG["features"]["fp_size"]
FP_RADIUS        = _CFG["features"]["fp_radius"]
USE_FINGERPRINTS = _CFG["features"]["use_fingerprints"]
USE_DESCRIPTORS  = _CFG["features"]["use_descriptors"]
TARGET           = _CFG["target"]
CV_SEEDS         = _CFG["cross_validation"]["seeds"]
N_FOLDS          = _CFG["cross_validation"]["n_folds"]

# ── Paths (constructed only — no I/O at import; dirs created in main) ────────
_ROOT       = Path(__file__).parents[3]        # PROTAC-Synthesizability/ (repo root)
OUTPUT_ROOT = _ROOT / "data" / "outputs"
CV_DIR      = OUTPUT_ROOT / "cv"
MODELS_DIR  = OUTPUT_ROOT / "models"


# ── Model dispatch (lazy import so a run pulls in only its backend) ──────────
def get_build_fn(model: str):
    """Return the model's `build_*` factory, importing only that backend.

    The import is deferred so a run never pulls in the other models' heavy deps
    (e.g. an xgb run doesn't import torch).

    Args:
        model: Model key, one of "xgb", "mlp", or "gnn".

    Returns:
        The corresponding build callable (build_xgb / build_mlp / build_gnn).

    Raises:
        ValueError: If `model` is not one of the supported names.
    """
    if model == "xgb":
        from xgb.hpo import build_xgb
        return build_xgb
    if model == "mlp":
        from mlp.hpo import build_mlp
        return build_mlp
    if model == "gnn":
        from gnn.hpo import build_gnn
        return build_gnn
    raise ValueError(f"Invalid model name: {model}")


# ── Feature caching ─────────────────────────────────────────────────────────
def feature_paths(input_path):
    """Build the cache file paths for fingerprints/descriptors, next to the input CSV.

    FP params are baked into the name so a config change can't reuse a stale cache.

    Args:
        input_path: Path to the input CSV; its stem is reused for the cache names.

    Returns:
        Tuple (fp_path, desc_path) of Path objects for the .npy caches.
    """
    stem = str(Path(input_path).with_suffix(""))
    fp_path   = Path(f"{stem}_fp_r{FP_RADIUS}_{FP_SIZE}.npy")
    desc_path = Path(f"{stem}_desc.npy")
    return fp_path, desc_path


def _load_or_compute(path, compute_fn, n_expected):
    """Load a cached .npy if present (and row-count matches), else compute + save.

    Args:
        path: Destination/source .npy cache path.
        compute_fn: Zero-arg callable returning the array when no valid cache exists.
        n_expected: Expected number of rows; a mismatch means a stale cache.

    Returns:
        The loaded or freshly computed numpy array.

    Raises:
        ValueError: If a cached array's row count differs from `n_expected`.
    """
    if path.exists():
        arr = np.load(path)
        if arr.shape[0] != n_expected:
            raise ValueError(
                f"Cached {path.name} has {arr.shape[0]} rows but the dataset has "
                f"{n_expected}. Delete the stale cache and recompute."
            )
        return arr
    arr = compute_fn()
    np.save(path, arr)
    return arr


def cache_features(df_train, input_path):
    """Model-agnostic: load (or compute+cache) fp/desc from the config flags.

    SMILES are standardized ONCE and the mols reused for both feature types.

    Args:
        df_train: DataFrame with a "molecule" column of SMILES.
        input_path: Path to the input CSV, used to locate the feature caches.

    Returns:
        Tuple (X_fp, X_desc); each is a numpy array, or None when disabled by
        USE_FINGERPRINTS / USE_DESCRIPTORS.
    """
    smiles = df_train["molecule"].tolist()
    mols   = standardize_all(smiles)                    # standardize once, reuse for both
    fp_path, desc_path = feature_paths(input_path)

    X_fp = _load_or_compute(
        fp_path, lambda: compute_fingerprints(mols, FP_SIZE, FP_RADIUS), len(df_train)
    ) if USE_FINGERPRINTS else None
    X_desc = _load_or_compute(
        desc_path, lambda: compute_descriptors(mols), len(df_train)
    ) if USE_DESCRIPTORS else None
    return X_fp, X_desc


def prepare_inputs(model, df_train, input_path):
    """Assemble training-time inputs for the chosen model.

    Tabular models load the shared feature cache; the GNN builds graphs inside
    its own fit, so it gets no tabular features.

    Args:
        model: Model key ("xgb", "mlp", or "gnn").
        df_train: DataFrame with a "molecule" column of SMILES.
        input_path: Path to the input CSV, used to locate the feature caches.

    Returns:
        Tuple (X_fp, X_desc); (None, None) for the GNN, otherwise the cached
        feature arrays (each possibly None per the config flags).
    """
    if model == "gnn":
        return None, None
    return cache_features(df_train, input_path)


# ── Scaffold-grouped fold indices ───────────────────────────────────────────
def get_fold_indices(df_train, seed, fold_idx, n_folds=N_FOLDS, inner_val_frac=0.2):
    """Build scaffold-grouped index arrays for one outer fold of nested CV.

    Scaffolds are shuffled (seeded) and greedily balanced across `n_folds`; the
    chosen fold becomes the outer-val set and the rest the outer-train set, which
    is then split again into an inner train/val by scaffold. Grouping guarantees
    no scaffold spans a train/val boundary.

    Args:
        df_train: DataFrame with a "scaffolds" column (one scaffold per row).
        seed: Seed for the outer scaffold shuffle (inner uses seed+fold_idx+1000).
        fold_idx: Which fold (0..n_folds-1) serves as the outer-val set.
        n_folds: Number of outer folds. Defaults to N_FOLDS.
        inner_val_frac: Fraction of the outer-train rows held out for inner-val.

    Returns:
        Tuple (fold_train_idx, fold_val_idx, inner_train_idx, inner_val_idx) of
        integer position arrays.
    """
    scaffold_to_indices = defaultdict(list)
    for pos, scaffold in enumerate(df_train["scaffolds"]):
        scaffold_to_indices[scaffold].append(pos)

    rng = np.random.default_rng(seed)
    scaffolds = list(scaffold_to_indices.keys())
    rng.shuffle(scaffolds)

    folds = [[] for _ in range(n_folds)]
    for scaffold in scaffolds:
        target = min(range(n_folds), key=lambda f: len(folds[f]))
        folds[target].extend(scaffold_to_indices[scaffold])

    fold_val_idx   = np.array(folds[fold_idx], dtype=int)
    fold_train_idx = np.array([i for f in range(n_folds) if f != fold_idx
                                 for i in folds[f]], dtype=int)

    inner_scaffold_to_indices = defaultdict(list)
    for pos in fold_train_idx:
        inner_scaffold_to_indices[df_train["scaffolds"].iloc[pos]].append(pos)

    inner_rng = np.random.default_rng(seed + fold_idx + 1000)
    inner_scaffolds = list(inner_scaffold_to_indices.keys())
    inner_rng.shuffle(inner_scaffolds)

    n_inner_val = int(np.floor(inner_val_frac * len(fold_train_idx)))
    inner_val_set = set()
    for scaffold in inner_scaffolds:
        if len(inner_val_set) >= n_inner_val:
            break
        inner_val_set.update(inner_scaffold_to_indices[scaffold])

    inner_val_idx   = np.array([p for p in fold_train_idx if p in inner_val_set], dtype=int)
    inner_train_idx = np.array([p for p in fold_train_idx if p not in inner_val_set], dtype=int)
    return fold_train_idx, fold_val_idx, inner_train_idx, inner_val_idx


# ── Inner Optuna tuning ─────────────────────────────────────────────────────
def tune_inner_fold(build_fn, df_train, X_fp, X_desc,
                    inner_train_idx, inner_val_idx,
                    seed, fold_idx, prefix, n_trials=25) -> dict:
    """Run the inner Optuna loop for one outer fold and return the best params.

    Each trial builds a model on the inner-train split and is scored (R2) on the
    inner-val split. The study is backed by a node-local SQLite DB (reliable on
    the compute node) with a whole-file snapshot copied to /proj after every
    trial, so a preempted job resumes instead of restarting. Also writes the
    trials dataframe and best_params JSON under CV_DIR/prefix.

    Args:
        build_fn: Model factory from get_build_fn; called once per trial.
        df_train: Full training DataFrame ("molecule" + TARGET columns).
        X_fp: Fingerprint feature matrix, or None.
        X_desc: Descriptor feature matrix, or None.
        inner_train_idx: Position array for the inner-train rows.
        inner_val_idx: Position array for the inner-val rows.
        seed: Outer seed; also seeds the TPE sampler (seed+fold_idx).
        fold_idx: Outer fold index, used in filenames and the sampler seed.
        prefix: Run/model prefix used for the study name and output dir.
        n_trials: Number of Optuna trials. Defaults to 25.

    Returns:
        The best hyperparameters (Optuna study.best_params dict).
    """
    smiles_itr  = df_train.iloc[inner_train_idx]["molecule"].tolist()
    smiles_ival = df_train.iloc[inner_val_idx]["molecule"].tolist()
    y_itr  = df_train.iloc[inner_train_idx][[TARGET]].values
    y_ival = df_train.iloc[inner_val_idx][[TARGET]].values
    X_fp_itr    = X_fp[inner_train_idx]  if X_fp  is not None else None
    X_fp_ival   = X_fp[inner_val_idx]    if X_fp  is not None else None
    X_desc_itr  = X_desc[inner_train_idx] if X_desc is not None else None
    X_desc_ival = X_desc[inner_val_idx]   if X_desc is not None else None
    fp_radius = FP_RADIUS

    def objective(trial):
        m = build_fn(
            trial, fp_radius,
            smiles_itr, y_itr, X_fp_itr, X_desc_itr,
            smiles_val=smiles_ival, y_val=y_ival,
            X_fp_val=X_fp_ival, X_desc_val=X_desc_ival,
        )
        return float(m.score(smiles_ival, y_ival, X_fp=X_fp_ival, X_desc=X_desc_ival))

    # ── node-local DB (reliable SQLite) + durable /proj snapshot (resume) ───
    tmp_dir    = os.environ.get("TMPDIR", "/tmp")
    local_db   = f"{tmp_dir}/{prefix}_seed{seed}_fold{fold_idx}.db"
    out_dir    = CV_DIR / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    persist_db = out_dir / f"{prefix}_seed{seed}_fold{fold_idx}.db"

    if persist_db.exists():
        shutil.copy(persist_db, local_db)

    study = optuna.create_study(
        study_name     = f"{prefix}_{seed}_fold{fold_idx}",
        direction      = "maximize",
        storage        = f"sqlite:///{local_db}",
        load_if_exists = True,
        sampler        = optuna.samplers.TPESampler(seed=seed + fold_idx),
        pruner         = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    def _snapshot(study, trial):
        shutil.copy(local_db, persist_db)     # whole-file copy -> reliable on VAST

    study.optimize(objective, n_trials=n_trials, callbacks=[_snapshot])

    study.trials_dataframe().to_csv(
        out_dir / f"trials_seed{seed}_fold{fold_idx}.csv", index=False)
    with open(out_dir / f"best_params_seed{seed}_fold{fold_idx}.json", "w") as f:
        json.dump({**study.best_params, "best_r2_inner": study.best_value}, f, indent=2)

    return study.best_params


# ── One outer fold: tune -> refit -> score -> save ──────────────────────────
def run_single_fold(build_fn, df_train, X_fp, X_desc, seed, fold_idx, prefix, n_trials=25):
    """Evaluate one outer fold end-to-end: tune on the inner split, refit on the
    full fold-train with the best params, score (R2) on the held-out fold-val,
    and persist the result to score_seed{seed}_fold{fold_idx}.json.

    Idempotent: if that score JSON already exists the fold is skipped and its
    cached R2 returned, so re-running a partially-finished sweep is safe.

    Args:
        build_fn: Model factory from get_build_fn.
        df_train: Full training DataFrame ("molecule", "scaffolds", TARGET).
        X_fp: Fingerprint feature matrix, or None.
        X_desc: Descriptor feature matrix, or None.
        seed: Outer seed for this fold.
        fold_idx: Outer fold index (0..N_FOLDS-1).
        prefix: Run/model prefix used for output paths.
        n_trials: Inner-tuning trial budget. Defaults to 25.

    Returns:
        The outer-val R2 (float) for this fold.
    """
    score_path = CV_DIR / prefix / f"score_seed{seed}_fold{fold_idx}.json"
    if score_path.exists():
        print(f"[{prefix}] seed {seed} fold {fold_idx} already done, skipping")
        with open(score_path) as f:
            return json.load(f)["r2"]

    fold_train_idx, fold_val_idx, inner_train_idx, inner_val_idx = \
        get_fold_indices(df_train, seed, fold_idx)

    best_params = tune_inner_fold(
        build_fn, df_train, X_fp, X_desc,
        inner_train_idx, inner_val_idx,
        seed, fold_idx, prefix, n_trials,
    )

    smiles_tr  = df_train.iloc[fold_train_idx]["molecule"].tolist()
    smiles_val = df_train.iloc[fold_val_idx]["molecule"].tolist()
    y_tr       = df_train.iloc[fold_train_idx][[TARGET]].values
    y_val      = df_train.iloc[fold_val_idx][[TARGET]].values
    X_fp_tr    = X_fp[fold_train_idx]  if X_fp  is not None else None
    X_fp_val   = X_fp[fold_val_idx]    if X_fp  is not None else None
    X_desc_tr  = X_desc[fold_train_idx] if X_desc is not None else None
    X_desc_val = X_desc[fold_val_idx]   if X_desc is not None else None

    fp_radius   = best_params.pop("fp_radius", FP_RADIUS)
    fixed_trial = FixedTrial(best_params)
    m = build_fn(
        fixed_trial, fp_radius,
        smiles_tr, y_tr, X_fp_tr, X_desc_tr,
        smiles_val=smiles_val, y_val=y_val,
        X_fp_val=X_fp_val, X_desc_val=X_desc_val,
    )
    score = float(m.score(smiles_val, y_val, X_fp=X_fp_val, X_desc=X_desc_val))

    score_path.parent.mkdir(parents=True, exist_ok=True)
    with open(score_path, "w") as f:
        json.dump({"seed": seed, "fold_idx": fold_idx, "r2": score}, f, indent=2)

    print(f"[{prefix}] seed {seed} fold {fold_idx} | R2 = {score:.3f}")
    return score


# ── Aggregate the 25 folds -> mean/std + final model ────────────────────────
def aggregate_results(build_fn, df_train, X_fp, X_desc, prefix):
    """Collect all 25 outer-fold JSONs, report mean +/- std R2, and retrain the
    final model on the full dataset.

    The final hyperparameters are the ones with the best inner-val R2 across
    folds (chosen by inner, never outer, score to avoid leakage). Saves the
    fitted model and a *_hparams.yaml CV summary under MODELS_DIR/prefix.

    Args:
        build_fn: Model factory from get_build_fn.
        df_train: Full training DataFrame ("molecule" + TARGET columns).
        X_fp: Fingerprint feature matrix, or None.
        X_desc: Descriptor feature matrix, or None.
        prefix: Run/model prefix identifying the fold JSONs and output dir.

    Returns:
        Tuple (mean_score, scores): the mean outer R2 and the list of per-fold R2s.

    Raises:
        RuntimeError: If any of the CV_SEEDS x N_FOLDS score JSONs is missing.
    """
    scores, best_params_all, missing = [], [], []
    for seed in CV_SEEDS:
        for fold_idx in range(N_FOLDS):
            sp = CV_DIR / prefix / f"score_seed{seed}_fold{fold_idx}.json"
            pp = CV_DIR / prefix / f"best_params_seed{seed}_fold{fold_idx}.json"
            if sp.exists():
                with open(sp) as f:
                    scores.append(json.load(f)["r2"])
            else:
                missing.append(f"seed{seed}_fold{fold_idx}")
            if pp.exists():
                with open(pp) as f:
                    best_params_all.append(json.load(f))

    if missing:
        raise RuntimeError(f"Missing fold results, cannot aggregate: {missing}")

    mean_score, std_score = float(np.mean(scores)), float(np.std(scores))
    print(f"[{prefix}] mean R2 = {mean_score:.3f} +/- {std_score:.3f} (n={len(scores)})")

    # overall-best params chosen by inner score (not outer -> no leak)
    best_entry = max(best_params_all, key=lambda d: d.get("best_r2_inner", -np.inf))
    best_inner = best_entry.get("best_r2_inner")
    fp_radius  = best_entry.pop("fp_radius", FP_RADIUS)
    best_entry.pop("best_r2_inner", None)                 # best_entry now = the hyperparameters

    final_model = build_fn(
        FixedTrial(best_entry), fp_radius,
        df_train["molecule"].tolist(), df_train[[TARGET]].values, X_fp, X_desc,
    )
    model_path = MODELS_DIR / prefix / f"{prefix}_final"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(str(model_path))
    print(f"[{prefix}] final model saved to {model_path}")

    # save the hyperparameters + CV summary as YAML next to the model
    hparams_out = {
        "model":           prefix,
        "hyperparameters": best_entry,
        "fp_radius":       fp_radius,
        "cv_mean_r2":      round(mean_score, 4),
        "cv_std_r2":       round(std_score, 4),
        "best_inner_r2":   round(best_inner, 4) if best_inner is not None else None,
        "n_folds":         len(scores),
    }
    with open(model_path.parent / f"{prefix}_hparams.yaml", "w") as f:
        yaml.safe_dump(hparams_out, f, default_flow_style=False, sort_keys=False)
    print(f"[{prefix}] hyperparameters saved to {model_path.parent / f'{prefix}_hparams.yaml'}")

    return mean_score, scores


# ── Entry point ─────────────────────────────────────────────────────────────
def parse_args():
    """Parse CLI args.

    Selects the run mode (--split / --precompute / --aggregate / single-fold)
    plus the model, input CSV, prefix, and tuning budget.

    Returns:
        argparse.Namespace with the parsed arguments.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xgb", "mlp", "gnn"])
    ap.add_argument("--input", required=True, help="path to the training CSV")
    ap.add_argument("--prefix", default="default")
    ap.add_argument("--n_trials", type=int, default=25)
    ap.add_argument("--seed", type=int, help="single-fold mode: outer seed")
    ap.add_argument("--fold", type=int, help="single-fold mode: outer fold index")
    ap.add_argument("--aggregate", action="store_true",
                    help="aggregate the saved folds and retrain the final model")
    ap.add_argument("--precompute", action="store_true",
                    help="compute + cache fp/desc once, then exit (run before the folds)")
    ap.add_argument("--split", action="store_true",
                    help="scaffold train/test split -> write *_train.csv / *_test.csv, then exit")
    ap.add_argument("--test-size", type=float, default=0.2)
    return ap.parse_args()


def main():
    """Dispatch on the parsed args to one of the run modes.

    --split and --precompute are model-agnostic one-shots that write their output
    and exit. Otherwise --model is required and output dirs are created lazily:
    --aggregate collects the folds and retrains the final model, while the default
    single-fold mode (requires --seed and --fold) tunes and evaluates one fold.

    Args:
        None. Reads from the command line via parse_args().

    Returns:
        None. Side effects only (writes CSVs, caches, fold JSONs, or the final model).

    Raises:
        SystemExit: If --model is missing for training/aggregation, or --seed/--fold
            are missing in single-fold mode.
    """
    args     = parse_args()
    df_train = pd.read_csv(args.input)

    # split: model-agnostic, one-time -> write CSVs and exit
    if args.split:
        from mol_utils import scaffold_train_test_split
        labels = scaffold_train_test_split(df_train["molecule"].tolist(),
                                           test_size=args.test_size, random_state=42)
        stem = Path(args.input).with_suffix("")
        df_train[labels == "train"].to_csv(f"{stem}_train.csv", index=False)
        df_train[labels == "test"].to_csv(f"{stem}_test.csv",  index=False)
        print(pd.Series(labels).value_counts())
        return

    # precompute: model-agnostic feature caching -> do it and exit
    if args.precompute:
        cache_features(df_train, args.input)
        return

    if args.model is None:
        ap_error = "--model is required for training/aggregation"
        raise SystemExit(ap_error)

    # create output dirs only when actually training/aggregating (not on import)
    CV_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    build_fn     = get_build_fn(args.model)
    X_fp, X_desc = prepare_inputs(args.model, df_train, args.input)

    if args.aggregate:
        aggregate_results(build_fn, df_train, X_fp, X_desc, args.prefix)
    else:
        if args.seed is None or args.fold is None:
            raise SystemExit("single-fold mode requires --seed and --fold (or use --aggregate)")
        df_train["scaffolds"] = df_train["molecule"].apply(get_scaffold)
        run_single_fold(build_fn, df_train, X_fp, X_desc,
                        args.seed, args.fold, args.prefix, args.n_trials)


if __name__ == "__main__":
    main()