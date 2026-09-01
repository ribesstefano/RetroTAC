"""Capture and replay golden predictions from the saved final models.

Guards the retrotac -> retrotac rename: the learned weights must not move.
Run --capture before renaming anything, --verify after every phase.

    apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \\
        python scripts/maintenance/verify_model_artifacts.py --capture
"""

import argparse
import json
import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
GOLDEN_PATH = Path("outputs/golden_predictions.json")
RUN = "20260828_182305"
BACKENDS = {
    "xgb": ("models.xgb.model", "XGBoostRegressor"),
    "mlp": ("models.mlp.model", "TorchMLPRegressor"),
    "gnn": ("models.gnn.model", "CheMeleonRegressor"),
}


def collect(package: str) -> dict:
    """Load each final model through `package` and predict on SMILES."""
    chem = import_module(f"{package}.chem_utils")
    x_desc = chem.compute_descriptors(chem.standardize_all(SMILES))
    out = {}
    for name, (submodule, clsname) in BACKENDS.items():
        backend = getattr(import_module(f"{package}.{submodule}"), clsname)
        model = backend.load(f"outputs/models/{name}_{RUN}/{name}_{RUN}_final")
        # Only the gnn backend featurizes internally; the other two take descriptors.
        preds = (
            model.predict(SMILES)
            if name == "gnn"
            else model.predict(SMILES, X_desc=x_desc)
        )
        out[name] = [round(float(v), 6) for v in np.asarray(preds).ravel()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true", help="Write the golden file.")
    mode.add_argument("--verify", action="store_true", help="Compare against it.")
    ap.add_argument("--package", default="retrotac", help="Package to import from.")
    args = ap.parse_args()

    actual = collect(args.package)

    if args.capture:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        print(f"captured -> {GOLDEN_PATH}")
        for name, preds in sorted(actual.items()):
            print(f"  {name}: {preds}")
        return 0

    if not GOLDEN_PATH.exists():
        print(f"FAIL: no golden file at {GOLDEN_PATH}; run --capture first.")
        return 1

    expected = json.loads(GOLDEN_PATH.read_text())
    ok = True
    for name in sorted(BACKENDS):
        if actual.get(name) != expected.get(name):
            ok = False
            print(f"  MISMATCH {name}: expected {expected.get(name)}, got {actual.get(name)}")
        else:
            print(f"  ok {name}: {actual[name]}")
    print("PASS: all backends match." if ok else "FAIL: predictions moved.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
