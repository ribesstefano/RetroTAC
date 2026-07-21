"""
train.py
========
5x5 nested scaffold cross-validation for the PROTAC synthesizability
surrogate models (xgb / mlp / gnn), model-agnostic.

Library only — no argparse/main and no module-level paths. The CLI entry
point is scripts/models/train.py, which supplies every path and config
value as an explicit argument.
"""
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
import numpy as np
import optuna
from optuna.trial import FixedTrial

from protac_synth.chem_utils import standardize_all, compute_fingerprints, compute_descriptors


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
        from protac_synth.models.xgb.hpo import build_xgb
        return build_xgb
    if model == "mlp":
        from protac_synth.models.mlp.hpo import build_mlp
        return build_mlp
    if model == "gnn":
        from protac_synth.models.gnn.hpo import build_gnn
        return build_gnn
    raise ValueError(f"Invalid model name: {model}")


# ── Feature caching ─────────────────────────────────────────────────────────
def feature_paths(input_path, fp_radius, fp_size):
    """Build the cache file paths for fingerprints/descriptors, next to the input CSV.

    FP params are baked into the name so a config change can't reuse a stale cache.

    Args:
        input_path: Path to the input CSV; its stem is reused for the cache names.
        fp_radius: Morgan fingerprint radius, baked into the fp cache filename.
        fp_size: Morgan fingerprint bit size, baked into the fp cache filename.

    Returns:
        Tuple (fp_path, desc_path) of Path objects for the .npy caches.
    """
    stem = str(Path(input_path).with_suffix(""))
    fp_path   = Path(f"{stem}_fp_r{fp_radius}_{fp_size}.npy")
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


def cache_features(df_train, input_path, fp_radius, fp_size, use_fingerprints, use_descriptors):
    """Model-agnostic: load (or compute+cache) fp/desc per the given feature flags.

    SMILES are standardized ONCE and the mols reused for both feature types.

    Args:
        df_train: DataFrame with a "molecule" column of SMILES.
        input_path: Path to the input CSV, used to locate the feature caches.
        fp_radius: Morgan fingerprint radius.
        fp_size: Morgan fingerprint bit size.
        use_fingerprints: Whether to compute/cache fingerprints.
        use_descriptors: Whether to compute/cache RDKit descriptors.

    Returns:
        Tuple (X_fp, X_desc); each is a numpy array, or None when disabled by
        use_fingerprints / use_descriptors.
    """
    smiles = df_train["molecule"].tolist()
    mols   = standardize_all(smiles)                    # standardize once, reuse for both
    fp_path, desc_path = feature_paths(input_path, fp_radius, fp_size)

    X_fp = _load_or_compute(
        fp_path, lambda: compute_fingerprints(mols, fp_size, fp_radius), len(df_train)
    ) if use_fingerprints else None
    X_desc = _load_or_compute(
        desc_path, lambda: compute_descriptors(mols), len(df_train)
    ) if use_descriptors else None
    return X_fp, X_desc


def prepare_inputs(model, df_train, input_path, fp_radius, fp_size, use_fingerprints, use_descriptors):
    """Assemble training-time inputs for the chosen model.

    Tabular models load the shared feature cache; the GNN builds graphs inside
    its own fit, so it gets no tabular features.

    Args:
        model: Model key ("xgb", "mlp", or "gnn").
        df_train: DataFrame with a "molecule" column of SMILES.
        input_path: Path to the input CSV, used to locate the feature caches.
        fp_radius: Morgan fingerprint radius.
        fp_size: Morgan fingerprint bit size.
        use_fingerprints: Whether to compute/cache fingerprints.
        use_descriptors: Whether to compute/cache RDKit descriptors.

    Returns:
        Tuple (X_fp, X_desc); (None, None) for the GNN, otherwise the cached
        feature arrays (each possibly None per the feature flags).
    """
    if model == "gnn":
        return None, None
    return cache_features(df_train, input_path, fp_radius, fp_size, use_fingerprints, use_descriptors)


# ── Scaffold-grouped fold indices ───────────────────────────────────────────
def get_fold_indices(df_train, seed, fold_idx, n_folds, inner_val_frac=0.2):
    """Build scaffold-grouped index arrays for one outer fold of nested CV.

    Scaffolds are shuffled (seeded) and greedily balanced across `n_folds`; the
    chosen fold becomes the outer-val set and the rest the outer-train set, which
    is then split again into an inner train/val by scaffold. Grouping guarantees
    no scaffold spans a train/val boundary.

    Args:
        df_train: DataFrame with a "scaffolds" column (one scaffold per row).
        seed: Seed for the outer scaffold shuffle (inner uses seed+fold_idx+1000).
        fold_idx: Which fold (0..n_folds-1) serves as the outer-val set.
        n_folds: Number of outer folds.
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
                    seed, fold_idx, prefix, cv_dir, target, fp_radius,
                    n_trials=25, build_kwargs=None) -> dict:
    """Run the inner Optuna loop for one outer fold and return the best params.

    Each trial builds a model on the inner-train split and is scored (R2) on the
    inner-val split. The study is backed by a node-local SQLite DB (reliable on
    the compute node) with a whole-file snapshot copied to /proj after every
    trial, so a preempted job resumes instead of restarting. Also writes the
    trials dataframe and best_params JSON under cv_dir/prefix.

    Args:
        build_fn: Model factory from get_build_fn; called once per trial.
        df_train: Full training DataFrame ("molecule" + target columns).
        X_fp: Fingerprint feature matrix, or None.
        X_desc: Descriptor feature matrix, or None.
        inner_train_idx: Position array for the inner-train rows.
        inner_val_idx: Position array for the inner-val rows.
        seed: Outer seed; also seeds the TPE sampler (seed+fold_idx).
        fold_idx: Outer fold index, used in filenames and the sampler seed.
        prefix: Run/model prefix used for the study name and output dir.
        cv_dir: Root directory for per-fold CV outputs.
        target: Name of the target column in df_train.
        fp_radius: Morgan fingerprint radius, forwarded to build_fn.
        n_trials: Number of Optuna trials. Defaults to 25.
        build_kwargs: Model-specific config forwarded to build_fn (e.g. fp_size,
            use_fingerprints, use_descriptors for tabular models; max_epochs,
            patience, chemeleon_weights for the GNN).

    Returns:
        The best hyperparameters (Optuna study.best_params dict).
    """
    build_kwargs = build_kwargs or {}
    smiles_itr  = df_train.iloc[inner_train_idx]["molecule"].tolist()
    smiles_ival = df_train.iloc[inner_val_idx]["molecule"].tolist()
    y_itr  = df_train.iloc[inner_train_idx][[target]].values
    y_ival = df_train.iloc[inner_val_idx][[target]].values
    X_fp_itr    = X_fp[inner_train_idx]  if X_fp  is not None else None
    X_fp_ival   = X_fp[inner_val_idx]    if X_fp  is not None else None
    X_desc_itr  = X_desc[inner_train_idx] if X_desc is not None else None
    X_desc_ival = X_desc[inner_val_idx]   if X_desc is not None else None

    def objective(trial):
        m = build_fn(
            trial, fp_radius,
            smiles_itr, y_itr, X_fp_itr, X_desc_itr,
            smiles_val=smiles_ival, y_val=y_ival,
            X_fp_val=X_fp_ival, X_desc_val=X_desc_ival,
            **build_kwargs,
        )
        return float(m.score(smiles_ival, y_ival, X_fp=X_fp_ival, X_desc=X_desc_ival))

    # ── node-local DB (reliable SQLite) + durable /proj snapshot (resume) ───
    tmp_dir    = os.environ.get("TMPDIR", "/tmp")
    local_db   = f"{tmp_dir}/{prefix}_seed{seed}_fold{fold_idx}.db"
    out_dir    = cv_dir / prefix
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
def run_single_fold(build_fn, df_train, X_fp, X_desc, seed, fold_idx, prefix,
                    cv_dir, target, fp_radius, n_folds, n_trials=25, build_kwargs=None):
    """Evaluate one outer fold end-to-end: tune on the inner split, refit on the
    full fold-train with the best params, score (R2) on the held-out fold-val,
    and persist the result to score_seed{seed}_fold{fold_idx}.json.

    Idempotent: if that score JSON already exists the fold is skipped and its
    cached R2 returned, so re-running a partially-finished sweep is safe.

    Args:
        build_fn: Model factory from get_build_fn.
        df_train: Full training DataFrame ("molecule", "scaffolds", target).
        X_fp: Fingerprint feature matrix, or None.
        X_desc: Descriptor feature matrix, or None.
        seed: Outer seed for this fold.
        fold_idx: Outer fold index (0..n_folds-1).
        prefix: Run/model prefix used for output paths.
        cv_dir: Root directory for per-fold CV outputs.
        target: Name of the target column in df_train.
        fp_radius: Morgan fingerprint radius, forwarded to build_fn.
        n_folds: Total number of outer folds.
        n_trials: Inner-tuning trial budget. Defaults to 25.
        build_kwargs: Model-specific config forwarded to build_fn (e.g. fp_size,
            use_fingerprints, use_descriptors for tabular models; max_epochs,
            patience, chemeleon_weights for the GNN).

    Returns:
        The outer-val R2 (float) for this fold.
    """
    score_path = cv_dir / prefix / f"score_seed{seed}_fold{fold_idx}.json"
    if score_path.exists():
        print(f"[{prefix}] seed {seed} fold {fold_idx} already done, skipping")
        with open(score_path) as f:
            return json.load(f)["r2"]

    build_kwargs = build_kwargs or {}
    fold_train_idx, fold_val_idx, inner_train_idx, inner_val_idx = \
        get_fold_indices(df_train, seed, fold_idx, n_folds)

    best_params = tune_inner_fold(
        build_fn, df_train, X_fp, X_desc,
        inner_train_idx, inner_val_idx,
        seed, fold_idx, prefix, cv_dir, target, fp_radius, n_trials,
        build_kwargs=build_kwargs,
    )

    smiles_tr  = df_train.iloc[fold_train_idx]["molecule"].tolist()
    smiles_val = df_train.iloc[fold_val_idx]["molecule"].tolist()
    y_tr       = df_train.iloc[fold_train_idx][[target]].values
    y_val      = df_train.iloc[fold_val_idx][[target]].values
    X_fp_tr    = X_fp[fold_train_idx]  if X_fp  is not None else None
    X_fp_val   = X_fp[fold_val_idx]    if X_fp  is not None else None
    X_desc_tr  = X_desc[fold_train_idx] if X_desc is not None else None
    X_desc_val = X_desc[fold_val_idx]   if X_desc is not None else None

    fold_fp_radius = best_params.pop("fp_radius", fp_radius)
    fixed_trial    = FixedTrial(best_params)
    m = build_fn(
        fixed_trial, fold_fp_radius,
        smiles_tr, y_tr, X_fp_tr, X_desc_tr,
        smiles_val=smiles_val, y_val=y_val,
        X_fp_val=X_fp_val, X_desc_val=X_desc_val,
        **build_kwargs,
    )
    score = float(m.score(smiles_val, y_val, X_fp=X_fp_val, X_desc=X_desc_val))

    score_path.parent.mkdir(parents=True, exist_ok=True)
    with open(score_path, "w") as f:
        json.dump({"seed": seed, "fold_idx": fold_idx, "r2": score}, f, indent=2)

    print(f"[{prefix}] seed {seed} fold {fold_idx} | R2 = {score:.3f}")
    return score


# ── Aggregate the folds -> mean/std + final model ───────────────────────────
def aggregate_results(build_fn, df_train, X_fp, X_desc, prefix,
                      cv_dir, models_dir, cv_seeds, n_folds, target, fp_radius,
                      build_kwargs=None):
    """Collect all cv_seeds x n_folds outer-fold JSONs, report mean +/- std R2,
    and retrain the final model on the full dataset.

    The final hyperparameters are the ones with the best inner-val R2 across
    folds (chosen by inner, never outer, score to avoid leakage). Saves the
    fitted model and a *_hparams.yaml CV summary under models_dir/prefix.

    Args:
        build_fn: Model factory from get_build_fn.
        df_train: Full training DataFrame ("molecule" + target columns).
        X_fp: Fingerprint feature matrix, or None.
        X_desc: Descriptor feature matrix, or None.
        prefix: Run/model prefix identifying the fold JSONs and output dir.
        cv_dir: Root directory the fold JSONs were written under.
        models_dir: Root directory the final model is saved under.
        cv_seeds: Outer seeds used during single-fold runs.
        n_folds: Total number of outer folds per seed.
        target: Name of the target column in df_train.
        fp_radius: Fallback Morgan fingerprint radius if a fold's best params omit it.
        build_kwargs: Model-specific config forwarded to build_fn (e.g. fp_size,
            use_fingerprints, use_descriptors for tabular models; max_epochs,
            patience, chemeleon_weights for the GNN).

    Returns:
        Tuple (mean_score, scores): the mean outer R2 and the list of per-fold R2s.

    Raises:
        RuntimeError: If any of the cv_seeds x n_folds score JSONs is missing.
    """
    build_kwargs = build_kwargs or {}
    scores, best_params_all, missing = [], [], []
    for seed in cv_seeds:
        for fold_idx in range(n_folds):
            sp = cv_dir / prefix / f"score_seed{seed}_fold{fold_idx}.json"
            pp = cv_dir / prefix / f"best_params_seed{seed}_fold{fold_idx}.json"
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
    best_fp_radius = best_entry.pop("fp_radius", fp_radius)
    best_entry.pop("best_r2_inner", None)                 # best_entry now = the hyperparameters

    final_model = build_fn(
        FixedTrial(best_entry), best_fp_radius,
        df_train["molecule"].tolist(), df_train[[target]].values, X_fp, X_desc,
        **build_kwargs,
    )
    model_path = models_dir / prefix / f"{prefix}_final"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(str(model_path))
    print(f"[{prefix}] final model saved to {model_path}")

    # save the hyperparameters + CV summary as YAML next to the model
    hparams_out = {
        "model":           prefix,
        "hyperparameters": best_entry,
        "fp_radius":       best_fp_radius,
        "cv_mean_r2":      round(mean_score, 4),
        "cv_std_r2":       round(std_score, 4),
        "best_inner_r2":   round(best_inner, 4) if best_inner is not None else None,
        "n_folds":         len(scores),
    }
    with open(model_path.parent / f"{prefix}_hparams.yaml", "w") as f:
        yaml.safe_dump(hparams_out, f, default_flow_style=False, sort_keys=False)
    print(f"[{prefix}] hyperparameters saved to {model_path.parent / f'{prefix}_hparams.yaml'}")

    return mean_score, scores
