"""Run inference with a saved PROTAC synthesizability surrogate (xgb / mlp / gnn)
over a CSV of SMILES. Loads a model saved by scripts/models/train.py's
--aggregate step (protac_synth.models.{xgb,mlp,gnn}.model's Backend.save()),
either from a local path or a Hugging Face Hub repo, and writes predictions
appended to the input CSV.

Every backend shares the same load/predict contract (see
protac_synth.models.training.get_build_fn's siblings): Backend.load(path) /
Backend.from_hf(hf_repo, hf_model_id) restores a fitted model, and
.predict(smiles_list) returns an (n_molecules, n_targets) array. This script
is a thin CLI over that contract -- all featurization/preprocessing is
handled internally by the loaded model, exactly as during training.

Usage
-----
    # Local model (base path without extension, e.g. outputs/models/xgb_default)
    python scripts/models/predict.py --model xgb --model-path outputs/models/xgb_default \\
        --input data/new_molecules.csv --output data/new_molecules_scored.csv

    # From Hugging Face Hub
    python scripts/models/predict.py --model gnn --hf-repo org/protac-synth-gnn \\
        --hf-model-id gnn_default --input data/new_molecules.csv --output out.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd


def _load_model(model: str, model_path: Optional[Path], hf_repo: Optional[str],
                hf_model_id: Optional[str], device: str) -> Any:
    """Import the requested backend and load a fitted model from disk or the Hub.

    Args:
        model: Backend key, one of "xgb", "mlp", "gnn".
        model_path: Local base path (without extension) to a saved model.
            Ignored (may be None) when hf_repo is given.
        hf_repo: Hugging Face Hub repo to pull the model from instead of disk.
        hf_model_id: Model id (base filename) within hf_repo; required with hf_repo.
        device: Compute device forwarded to the gnn backend only.

    Returns:
        A fitted Backend instance (XGBoostRegressor / TorchMLPRegressor /
        CheMeleonRegressor) exposing the shared load/predict contract.

    Raises:
        ValueError: If `model` is not one of the supported names.
    """
    if model == "xgb":
        from protac_synth.models.xgb.model import XGBoostRegressor as Backend
    elif model == "mlp":
        from protac_synth.models.mlp.model import TorchMLPRegressor as Backend
    elif model == "gnn":
        from protac_synth.models.gnn.model import CheMeleonRegressor as Backend
    else:
        raise ValueError(f"Invalid model name: {model}")

    # Only the gnn backend's load/from_hf take a device kwarg.
    kwargs = {"device": device} if model == "gnn" else {}
    if hf_repo:
        return Backend.from_hf(hf_repo, hf_model_id, **kwargs)
    return Backend.load(str(model_path), **kwargs)


def predict_csv(model: str, in_path: Path, out_path: Path, smiles_col: str = "SMILES",
                model_path: Optional[Path] = None, hf_repo: Optional[str] = None,
                hf_model_id: Optional[str] = None, device: str = "cpu",
                target_names: Optional[List[str]] = None) -> None:
    """Load a saved model and append its predictions to every row of `in_path`.

    Args:
        model: Backend key, one of "xgb", "mlp", "gnn".
        in_path: Input CSV containing at least `smiles_col`.
        out_path: Output CSV path (input columns + appended prediction column(s)).
        smiles_col: Name of the SMILES column in `in_path`.
        model_path: Local base path (without extension) to a model saved by
            train.py's --aggregate step. Mutually exclusive with hf_repo.
        hf_repo: Hugging Face Hub repo to pull the model from instead of disk.
        hf_model_id: Model id (base filename) within `hf_repo`; required with hf_repo.
        device: Compute device forwarded to the gnn backend (xgb/mlp always run
            their (cheap) inference on CPU).
        target_names: Column name(s) for the output prediction(s); defaults to
            "prediction" for a single target or "prediction_0", "prediction_1", ... .
    """
    if not model_path and not hf_repo:
        raise ValueError("one of model_path or hf_repo is required")

    backend = _load_model(model, model_path, hf_repo, hf_model_id, device)

    df = pd.read_csv(in_path)
    preds = backend.predict(df[smiles_col].tolist())

    n_targets = preds.shape[1]
    if target_names is None:
        target_names = ["prediction"] if n_targets == 1 else [f"prediction_{i}" for i in range(n_targets)]
    if len(target_names) != n_targets:
        raise ValueError(f"expected {n_targets} target name(s), got {len(target_names)}")

    for i, name in enumerate(target_names):
        df[name] = preds[:, i]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"scored {len(df)} molecules ({n_targets} target(s)) -> {out_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI args for scoring a CSV of SMILES with a saved model.

    Returns:
        argparse.Namespace with the parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=["xgb", "mlp", "gnn"])
    ap.add_argument("--input", required=True, type=Path, help="CSV with a SMILES column")
    ap.add_argument("--output", required=True, type=Path,
                    help="input columns + appended prediction column(s)")
    ap.add_argument("--smiles-col", default="SMILES", help="SMILES column name (default: SMILES)")
    ap.add_argument("--model-path", type=Path, default=None,
                    help="local base path (no extension) to a model saved by train.py --aggregate")
    ap.add_argument("--hf-repo", default=None, help="Hugging Face Hub repo id, e.g. org/protac-synth-xgb")
    ap.add_argument("--hf-model-id", default=None, help="model id (base filename) within --hf-repo")
    ap.add_argument("--device", default="cpu", help="compute device forwarded to gnn (default: cpu)")
    ap.add_argument("--target-names", default=None,
                    help="comma-separated output column name(s); default: prediction[_i]")
    return ap.parse_args()


def main() -> None:
    """Parse CLI args and run predict_csv end-to-end.

    Returns:
        None. Side effects only (writes the scored CSV to --output).
    """
    args = parse_args()
    target_names = args.target_names.split(",") if args.target_names else None
    predict_csv(args.model, args.input, args.output, smiles_col=args.smiles_col,
               model_path=args.model_path, hf_repo=args.hf_repo, hf_model_id=args.hf_model_id,
               device=args.device, target_names=target_names)


if __name__ == "__main__":
    main()
