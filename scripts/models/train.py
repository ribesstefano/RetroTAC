"""
scripts/models/train.py
========================
CLI entry point for 5x5 nested scaffold cross-validation of the PROTAC
synthesizability surrogate models (xgb / mlp / gnn). The library logic lives
in protac_synth.models.training; this script only parses args, loads config,
and wires paths + config values through.

All modelling knobs (SMILES column, target, seeds, fp params, feature flags)
come from --config, so the CLI carries only the run-mode selectors and I/O paths.

Three run modes (dispatched in main):
  * --precompute : model-agnostic; compute + cache fp/desc, then exit
  * --aggregate  : collect the fold JSONs, retrain the final model
  * single-fold  : the default; --seed S --fold F tunes + evaluates ONE outer fold

Precompute the feature caches once (tabular models):
    python scripts/models/train.py --model xgb --input data.csv --precompute
Run one fold:
    python scripts/models/train.py --model xgb --input data.csv --seed 42 --fold 0 --prefix v1
Aggregate after all folds:
    python scripts/models/train.py --model xgb --input data.csv --aggregate --prefix v1
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from protac_synth.chem_utils import get_scaffold  # noqa: E402
from protac_synth.models.training import (  # noqa: E402
    aggregate_results, cache_features, get_build_fn, prepare_inputs, run_single_fold,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI args and select the run mode.

    Only run-mode selectors and I/O paths live here; every modelling value is
    read from --config (see the module docstring for the three run modes).

    Returns:
        argparse.Namespace with the parsed arguments.
    """
    ap = argparse.ArgumentParser(
        description="5x5 nested scaffold CV for the PROTAC synthesizability surrogates.")

    # ── what & where ────────────────────────────────────────────────────────
    ap.add_argument("--model", choices=["xgb", "mlp", "gnn"],
                    help="surrogate to train; optional for --precompute (feature caching is model-agnostic)")
    ap.add_argument("--input", required=True, type=Path,
                    help="training CSV; must hold the SMILES + target columns named in the config")
    ap.add_argument("--config", type=Path,
                    default=Path("config") / "models_config.yaml",
                    help="YAML with features / target / molecule_col / cross_validation "
                         "(default: config/models_config.yaml)")
    ap.add_argument("--output-root", type=Path,
                    default="outputs",
                    help="root for outputs: fold JSONs under <root>/cv, final model under <root>/models "
                         "(default: data/outputs)")
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs") / "feature_cache",
                    help="dedicated directory for the fp/desc .npy caches, named only by feature "
                         "params (default: outputs/feature_cache); never written next to --input")
    ap.add_argument("--prefix", default="default",
                    help="run label; namespaces the CV and model output dirs (default: default)")
    ap.add_argument("--device", default="cpu",
                    help="compute device forwarded to the model (e.g. cpu, cuda); default: cpu")

    # ── run-mode selectors ──────────────────────────────────────────────────
    ap.add_argument("--seed", type=int,
                    help="single-fold mode: outer CV seed (one of the config seeds)")
    ap.add_argument("--fold", type=int,
                    help="single-fold mode: outer fold index (0..n_folds-1)")
    ap.add_argument("--aggregate", action="store_true",
                    help="aggregate the saved folds and retrain the final model")
    ap.add_argument("--precompute", action="store_true",
                    help="compute + cache fp/desc once, then exit (run before the folds)")
    ap.add_argument("--no-save-fold-models", dest="save_fold_models", action="store_false",
                    help="do not save the per-fold refit model (saved by default under <root>/cv/<run_id>/)")

    # ── tuning ──────────────────────────────────────────────────────────────
    ap.add_argument("--n_trials", type=int, default=25,
                    help="Optuna trials per inner-fold tuning loop (default: 25)")
    return ap.parse_args()


def main():
    """Dispatch on the parsed args to one of the run modes.

    --precompute is a model-agnostic one-shot that caches features and exits.
    Otherwise --model is required and output dirs are created lazily:
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
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    molecule_col = cfg["molecule_col"]
    fp_size = cfg["features"]["fp_size"]
    fp_radius = cfg["features"]["fp_radius"]
    use_fingerprints = cfg["features"]["use_fingerprints"]
    use_descriptors = cfg["features"]["use_descriptors"]
    target = cfg["target"]
    cv_seeds = cfg["cross_validation"]["seeds"]
    n_folds = cfg["cross_validation"]["n_folds"]

    # Create output directories
    cv_dir = args.output_root / "cv"
    models_dir = args.output_root / "models"
    cv_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(args.input)

    # precompute: model-agnostic feature caching -> do it and exit
    if args.precompute:
        cache_features(df_train, args.cache_dir, fp_radius, fp_size,
                       use_fingerprints, use_descriptors, molecule_col)
        return

    if args.model is None:
        raise SystemExit("--model is required for training/aggregation")

    build_fn = get_build_fn(args.model)
    X_fp, X_desc = prepare_inputs(args.model, df_train, args.cache_dir,
                                  fp_radius, fp_size, use_fingerprints, use_descriptors,
                                  molecule_col)

    # model-specific config forwarded to build_fn (absorbed by its **kwargs)
    if args.model == "gnn":
        build_kwargs = {
            "max_epochs": cfg["torch"]["max_epochs"],
            "patience": cfg["torch"]["patience"],
            "chemeleon_weights": cfg.get("gnn", {}).get("chemeleon_weights", "chemeleon_mp.pt"),
            "device": args.device,
        }
    else:
        build_kwargs = {
            "fp_size": fp_size,
            "use_fingerprints": use_fingerprints,
            "use_descriptors": use_descriptors,
            "device": args.device,
        }

    # batch_size is a fixed torch training param (not Optuna-tuned) for the NN models
    if args.model in ("mlp", "gnn"):
        build_kwargs["batch_size"] = cfg["torch"]["batch_size"]

    # Run ID namespaces every output (study name, CV/model dirs, DBs); it must
    # include the model type so different models sharing a --prefix don't collide.
    run_id = f"{args.model}_{args.prefix}"

    if args.aggregate:
        aggregate_results(build_fn, df_train, X_fp, X_desc, run_id,
                          cv_dir, models_dir, cv_seeds, n_folds, target, fp_radius,
                          molecule_col=molecule_col, build_kwargs=build_kwargs)
    else:
        if args.seed is None or args.fold is None:
            raise SystemExit("single-fold mode requires --seed and --fold (or use --aggregate)")
        df_train["scaffolds"] = df_train[molecule_col].apply(get_scaffold)
        run_single_fold(build_fn, df_train, X_fp, X_desc, args.seed, args.fold, run_id,
                        cv_dir, target, fp_radius, n_folds, args.n_trials,
                        save_fold_model=args.save_fold_models,
                        molecule_col=molecule_col, build_kwargs=build_kwargs)


if __name__ == "__main__":
    main()
