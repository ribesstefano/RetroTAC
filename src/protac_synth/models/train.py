import argparse
import json
import numpy as np
import optuna
from collections import defaultdict
from pathlib import Path
import pandas as pd
from model_config import (FP_SIZE, FP_RADIUS, USE_FINGERPRINTS,
                    USE_DESCRIPTORS, TARGET, CV_SEEDS)
from optuna.trial import FixedTrial
from mol_utils import compute_fingerprints, get_scaffold

_ROOT      = Path(__file__).parent
OUTPUT_ROOT = _ROOT / "outputs"
OPTUNA_DB  = OUTPUT_ROOT / "optuna" / "studies.db"
CV_DIR     = OUTPUT_ROOT / "cv"
MODELS_DIR = OUTPUT_ROOT / "models"
INPUT_CSV  = _ROOT / "data" / "protac_train.csv"

for _p in (OPTUNA_DB.parent, CV_DIR, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

def get_build_fn(model: str)->callable:
    if model == "xgb":
        from xgb.hpo import build_xgb
        return build_xgb
    elif model == "mlp":
        from mlp.hpo import build_mlp
        return build_mlp
    elif model == "gnn":
        from gnn.hpo import build_gnn
        return build_gnn
    else:
        raise ValueError('Invalid model name')

    
def prepare_inputs(model, df_train):
    if model in ("xgb", "mlp"):
        if USE_FINGERPRINTS:
            X_fp = compute_fingerprints(df_train["molecule"].tolist(), FP_SIZE, FP_RADIUS)
        else:
            X_fp = None
        if USE_DESCRIPTORS:
            # descriptor columns (precomputed)
        else:
            X_desc = None
    elif model == 'gnn':
        X_fp = None
        X_desc = None
    return X_fp, X_desc

def get_fold_indices(df_train, seed, fold_idx, n_folds=5, inner_val_frac=0.2):
    """Return (fold_train_idx, fold_val_idx, inner_train_idx, inner_val_idx)
    as integer position arrays into df_train, respecting scaffold groups."""

    # ── 1. Group row positions by scaffold ──────────────────────────────────
    scaffold_to_indices = defaultdict(list)
    for pos, scaffold in enumerate(df_train["scaffolds"]):
        scaffold_to_indices[scaffold].append(pos)

    # ── 2. Shuffle scaffolds (seeded) and assign whole groups to K folds ─────
    rng = np.random.default_rng(seed)
    scaffolds = list(scaffold_to_indices.keys())
    rng.shuffle(scaffolds)

    folds = [[] for _ in range(n_folds)]          # each is a list of row positions
    for scaffold in scaffolds:
        # TODO: drop this scaffold's indices into the CURRENTLY-SMALLEST fold
        #   - find the fold with fewest indices  (min over len)
        #   - extend it with scaffold_to_indices[scaffold]
        target = min(range(n_folds), key=lambda f: len(folds[f]))
        folds[target].extend(scaffold_to_indices[scaffold])

    # ── 3. Outer split: fold_idx is validation, the rest is training ─────────
    fold_val_idx   = np.array(folds[fold_idx])
    fold_train_idx = np.array([i for f in range(n_folds) if f != fold_idx
                                 for i in folds[f]])

    # ── 4. Inner split: carve inner-val out of fold_train_idx (scaffold-aware)
    #   Re-group the OUTER-TRAIN positions by scaffold, shuffle, and peel off
    #   ~inner_val_frac of the molecules (whole scaffolds) as inner-val.
    # TODO:
    #   - build scaffold -> indices map, but only over fold_train_idx
    #   - shuffle those scaffolds with a seeded rng
    #   - accumulate indices into inner_val until it reaches
    #     inner_val_frac * len(fold_train_idx); the rest is inner_train
    inner_scaffold_to_indices = defaultdict(list)
    for pos in fold_train_idx:
        scaffold = df_train["scaffolds"].iloc[pos]
        inner_scaffold_to_indices[scaffold].append(pos)

    inner_rng = np.random.default_rng(seed + fold_idx + 1000)
    inner_scaffolds = list(inner_scaffold_to_indices.keys())
    inner_rng.shuffle(inner_scaffolds)

    n_inner_val = int(np.floor(inner_val_frac * len(fold_train_idx)))
    inner_val_set = set()
    for scaffold in inner_scaffolds:
        if len(inner_val_set) >= n_inner_val:
            break
        inner_val_set.update(inner_scaffold_to_indices[scaffold])

    inner_val_idx   = np.array([p for p in fold_train_idx if p in inner_val_set])
    inner_train_idx = np.array([p for p in fold_train_idx if p not in inner_val_set])

    return fold_train_idx, fold_val_idx, inner_train_idx, inner_val_idx

def tune_inner_fold(build_fn, df_train, X_fp, X_desc, inner_train_idx, inner_val_idx, seed, fold_idx, prefix, n_trials = 25) -> dict:
    smiles_itr  = df_train.iloc[inner_train_idx]["molecule"].tolist()
    smiles_ival = df_train.iloc[inner_val_idx]["molecule"].tolist()

    y_itr  = df_train.iloc[inner_train_idx][[TARGET]].values   # [[..]] -> shape (n, 1)
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

    study_name = f"{prefix}_seed{seed}_fold{fold_idx}"
    study = optuna.create_study(
        study_name     = study_name,
        direction      = "maximize",                       # R2: higher is better
        storage        = f"sqlite:///{OPTUNA_DB}",
        load_if_exists = True,                             # resume a crashed run
        sampler        = optuna.samplers.TPESampler(seed=seed + fold_idx),
        pruner         = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=n_trials)

    out = CV_DIR / prefix / f"best_params_seed{seed}_fold{fold_idx}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({**study.best_params, "best_r2_inner": study.best_value}, f, indent=2)

    return study.best_params

def run_single_fold(build_fn, df_train, X_fp, X_desc,
                    seed, fold_idx, prefix, n_trials=25) -> float:

    # ── 1. Get the four index sets for this (seed, fold) ────────────────────
    fold_train_idx, fold_val_idx, inner_train_idx, inner_val_idx = \
        get_fold_indices(df_train, seed, fold_idx)

    # ── 2. Inner tuning -> best hyperparameters ─────────────────────────────
    best_params = tune_inner_fold(
        build_fn, df_train, X_fp, X_desc,
        inner_train_idx, inner_val_idx,
        seed, fold_idx, prefix, n_trials,
    )

    # ── 3. Slice the OUTER train / val by index ─────────────────────────────
    smiles_tr  = df_train.iloc[fold_train_idx]["molecule"].tolist()
    smiles_val = df_train.iloc[fold_val_idx]["molecule"].tolist()
    y_tr       = df_train.iloc[fold_train_idx][[TARGET]].values
    y_val      = df_train.iloc[fold_val_idx][[TARGET]].values
    X_fp_tr    = X_fp[fold_train_idx]  if X_fp  is not None else None
    X_fp_val   = X_fp[fold_val_idx]    if X_fp  is not None else None
    X_desc_tr  = X_desc[fold_train_idx] if X_desc is not None else None
    X_desc_val = X_desc[fold_val_idx]   if X_desc is not None else None

    # ── 4. Refit on outer-train with best params (replayed via FixedTrial) ──
    fp_radius   = FP_RADIUS                       # fixed radius for now
    fixed_trial = FixedTrial(best_params)
    m = build_fn(
        fixed_trial, fp_radius,
        smiles_tr, y_tr, X_fp_tr, X_desc_tr,
        smiles_val=smiles_val, y_val=y_val,
        X_fp_val=X_fp_val, X_desc_val=X_desc_val,
    )

    # ── 5. Score on outer-val, save, return ─────────────────────────────────
    score = float(m.score(smiles_val, y_val, X_fp=X_fp_val, X_desc=X_desc_val))

    score_path = CV_DIR / prefix / f"score_seed{seed}_fold{fold_idx}.json"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with open(score_path, "w") as f:
        json.dump({"seed": seed, "fold_idx": fold_idx, "r2": score}, f, indent=2)

    return score

def run_nested_cv(build_fn, df_train, X_fp, X_desc, prefix, input_path=None):

    # ── 1. Run all 25 folds, collecting outer scores ────────────────────────
    scores = []
    for seed in CV_SEEDS:
        for fold_idx in range(5):
            # (optional resumability: if the score JSON exists, load it instead)
            score = run_single_fold(build_fn, df_train, X_fp, X_desc,
                                    seed, fold_idx, prefix)
            scores.append(score)

    mean_score = float(np.mean(scores))
    std_score  = float(np.std(scores))
    print(f"[{prefix}] mean R2 = {mean_score:.3f} ± {std_score:.3f}")

    # ── 2. Pick overall-best params across the 25 folds ─────────────────────
    #   read every best_params JSON, choose the one with highest best_r2_inner
    best_params_all = []
    for seed in CV_SEEDS:
        for fold_idx in range(5):
            p = CV_DIR / prefix / f"best_params_seed{seed}_fold{fold_idx}.json"
            if p.exists():
                with open(p) as f:
                    best_params_all.append(json.load(f))

    best_entry = max(best_params_all, key=lambda d: d.get("best_r2_inner", -np.inf))
    best_entry = {k: v for k, v in best_entry.items() if k != "best_r2_inner"}
    # (pop fp_radius here too if you tune it)

    # ── 3. Refit FINAL model on ALL of df_train with best params ────────────
    fp_radius   = FP_RADIUS
    fixed_trial = FixedTrial(best_entry)
    smiles_all  = df_train["molecule"].tolist()
    y_all       = df_train[[TARGET]].values
    final_model = build_fn(fixed_trial, fp_radius,
                           smiles_all, y_all, X_fp, X_desc)

    # ── 4. Save the final model ─────────────────────────────────────────────
    model_path = MODELS_DIR / prefix / f"{prefix}_final"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(str(model_path))
    print(f"[{prefix}] final model saved to {model_path}")

    return mean_score, scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["xgb", "mlp", "gnn"])
    ap.add_argument("--n-trials", type=int, default=25)
    args = ap.parse_args()

    df_train = pd.read_csv(INPUT_CSV)
    df_train["scaffolds"] = df_train["molecule"].apply(get_scaffold)   # needed by get_fold_indices

    build_fn      = get_build_fn(args.model)
    X_fp, X_desc  = prepare_inputs(args.model, df_train)
    run_nested_cv(build_fn, df_train, X_fp, X_desc, prefix=args.model)

if __name__ == "__main__":
    main()