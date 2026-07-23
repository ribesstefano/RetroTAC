"""
scripts/models/train.py
========================
CLI entry point for 5x5 nested scaffold cross-validation of the PROTAC
synthesizability surrogate models (xgb / mlp / gnn). The library logic lives
in protac_synth.models.train; this script only parses args, loads config,
and wires paths through.

Two run modes (dispatched in main):
  * single-fold : --seed S --fold F  -> tune + evaluate ONE outer fold, save JSON
  * aggregate   : --aggregate        -> collect the fold JSONs, retrain final

Run one fold:
    python scripts/models/train.py --model xgb --input data.csv --seed 0 --fold 0 --prefix v1
Aggregate after all folds:
    python scripts/models/train.py --model xgb --input data.csv --aggregate --prefix v1
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_PROJECT_ROOT))

from protac_synth.chem_utils import get_scaffold, scaffold_train_test_split  # noqa: E402
from protac_synth.models.train import (  # noqa: E402
    aggregate_results, cache_features, get_build_fn, prepare_inputs, run_single_fold,
)


def parse_args():
    """Parse CLI args.

    Selects the run mode (--split / --precompute / --aggregate / single-fold)
    plus the model, input CSV, prefix, tuning budget, and output paths.

    Returns:
        argparse.Namespace with the parsed arguments.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xgb", "mlp", "gnn"])
    ap.add_argument("--input", required=True, type=Path, help="path to the training CSV")
    ap.add_argument("--config", type=Path, default=_PROJECT_ROOT / "config" / "models_config.yaml")
    ap.add_argument("--output-root", type=Path, default=_PROJECT_ROOT / "data" / "outputs")
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
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    fp_size          = cfg["features"]["fp_size"]
    fp_radius        = cfg["features"]["fp_radius"]
    use_fingerprints = cfg["features"]["use_fingerprints"]
    use_descriptors  = cfg["features"]["use_descriptors"]
    target           = cfg["target"]
    cv_seeds         = cfg["cross_validation"]["seeds"]
    n_folds          = cfg["cross_validation"]["n_folds"]

    cv_dir     = args.output_root / "cv"
    models_dir = args.output_root / "models"

    df_train = pd.read_csv(args.input)

    # split: model-agnostic, one-time -> write CSVs and exit
    if args.split:
        labels = scaffold_train_test_split(df_train["molecule"].tolist(),
                                           test_size=args.test_size, random_state=42)
        stem = args.input.with_suffix("")
        df_train[labels == "train"].to_csv(f"{stem}_train.csv", index=False)
        df_train[labels == "test"].to_csv(f"{stem}_test.csv",  index=False)
        print(pd.Series(labels).value_counts())
        return

    # precompute: model-agnostic feature caching -> do it and exit
    if args.precompute:
        cache_features(df_train, args.input, fp_radius, fp_size, use_fingerprints, use_descriptors)
        return

    if args.model is None:
        raise SystemExit("--model is required for training/aggregation")

    # create output dirs only when actually training/aggregating (not on import)
    cv_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    build_fn     = get_build_fn(args.model)
    X_fp, X_desc = prepare_inputs(args.model, df_train, args.input,
                                  fp_radius, fp_size, use_fingerprints, use_descriptors)

    # model-specific config forwarded to build_fn (absorbed by its **kwargs)
    if args.model == "gnn":
        build_kwargs = {
            "max_epochs":        cfg["torch"]["max_epochs"],
            "patience":          cfg["torch"]["patience"],
            "chemeleon_weights": cfg.get("gnn", {}).get("chemeleon_weights", "chemeleon_mp.pt"),
        }
    else:
        build_kwargs = {
            "fp_size":          fp_size,
            "use_fingerprints": use_fingerprints,
            "use_descriptors":  use_descriptors,
        }

    if args.aggregate:
        aggregate_results(build_fn, df_train, X_fp, X_desc, args.prefix,
                          cv_dir, models_dir, cv_seeds, n_folds, target, fp_radius,
                          build_kwargs=build_kwargs)
    else:
        if args.seed is None or args.fold is None:
            raise SystemExit("single-fold mode requires --seed and --fold (or use --aggregate)")
        df_train["scaffolds"] = df_train["molecule"].apply(get_scaffold)
        run_single_fold(build_fn, df_train, X_fp, X_desc, args.seed, args.fold, args.prefix,
                        cv_dir, target, fp_radius, n_folds, args.n_trials,
                        build_kwargs=build_kwargs)


if __name__ == "__main__":
    main()
