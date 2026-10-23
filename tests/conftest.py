"""Shared fixtures: a fake ensemble, and the paths the plugin tests need.

`OMP_NUM_THREADS=1` is set at module top, before anything imports torch or
xgboost: the two bundle separate libomp copies and segfault when both run
multi-threaded in one process (the same reason `retrotac.serve.__main__`
sets it before its own imports).

Nothing here loads the real 27-member ensemble -- it is 1.4 GiB and the
server logic under test is the alignment/validation/locking layer around it,
not the models. `FakeEnsemble` implements exactly the surface
`ScoringService` uses.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path  # noqa: E402
from typing import Dict, List  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
REINVENT_DIR = REPO_ROOT / "reinvent"
STUB_DIR = Path(__file__).resolve().parent / "stub"


class FakeEnsemble:
    """Stand-in for `retrotac.models.ensemble.RetroTAC`.

    Scores a molecule as a deterministic function of its SMILES length, which
    is enough to tell alignment apart from misalignment, and records every
    call so a test can assert what the service passed down.
    """

    def __init__(self, keys=("mean", "std"), n_values=None, non_finite=False):
        """Configure the fake's response shape.

        Args:
            keys: Output keys to return, standing in for the ensemble's own.
            n_values: Return this many values regardless of how many molecules
                were passed, to simulate the width/alignment faults the
                service is supposed to catch. None (default) returns one per
                molecule.
            non_finite: If True, make the first value NaN.
        """
        self.keys = keys
        self.n_values = n_values
        self.non_finite = non_finite
        self.calls: List[List[str]] = []

    members = {
        "GNN_seed42_fold0": {"backend": "gnn", "seed": 42, "fold": 0, "path": "members/a"},
        "MLP_seed42_fold1": {"backend": "mlp", "seed": 42, "fold": 1, "path": "members/b"},
        "XGB_seed7_fold2": {"backend": "xgb", "seed": 7, "fold": 2, "path": "members/c"},
    }
    feature_config = {
        "fp_size": 2048,
        "fp_radius": 3,
        "use_fingerprints": True,
        "use_descriptors": True,
    }
    device = "cpu"
    load_strategy = "eager"

    def predict(self, smiles_list, return_details=False, **kwargs) -> Dict[str, np.ndarray]:
        """Return per-key arrays plus a member block, like the real ensemble."""
        self.calls.append(list(smiles_list))
        n = self.n_values if self.n_values is not None else len(smiles_list)
        mean = np.array([0.01 * len(s) for s in smiles_list], dtype=float)[:n]
        if len(mean) < n:  # n_values larger than the batch: pad, to stay the wrong shape
            mean = np.resize(mean, n)
        if self.non_finite and mean.size:
            mean = mean.copy()
            mean[0] = np.nan
        details = {key: mean + (0.0 if key == "mean" else 0.5) for key in self.keys}
        details["members"] = {name: mean for name in self.members}
        return details if return_details else details["mean"]


@pytest.fixture
def service():
    """A `ScoringService` over a healthy `FakeEnsemble`."""
    from retrotac.serve.app import ScoringService

    return ScoringService(FakeEnsemble(), source="tests/fake")


@pytest.fixture
def client(service):
    """A fastapi `TestClient` over the app built from `service`."""
    from fastapi.testclient import TestClient

    from retrotac.serve.app import create_app

    return TestClient(create_app(service))
