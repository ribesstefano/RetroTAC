"""
loading.py
==========
Shared model-loading and featurization helpers used by both
scripts/models/evaluation.py (predicting with individual 5x5-CV fold models)
and retrotac/models/ensemble.py (the Caruana ensemble over those same fold
models). Extracted so both call sites share one definition of "how a fold
model is loaded" and "how its inputs are featurized" -- the exact contract
the ensemble's Caruana weights were fit against (see
scripts/models/evaluation.py's predict_fold_models / run_ensemble_strategies).
"""
import re
from functools import partial
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

_FOLD_MODEL_NAME_RE = re.compile(r"^([A-Za-z]+)_seed(\d+)_fold(\d+)$")


def load_backend(kind: str, base_path: str, device: str = "cpu") -> Any:
    """Load a fitted model by backend kind from a base path (no extension).

    Args:
        kind: One of "xgb", "mlp", "gnn" (case-sensitive, lowercase).
        base_path: Base path forwarded to the backend's own `load()`.
        device: Compute device; forwarded only to the gnn backend, the only
            one whose `load()` accepts it.

    Returns:
        The loaded backend instance.

    Raises:
        ValueError: If `kind` isn't one of xgb/mlp/gnn.
    """
    if kind == "xgb":
        from retrotac.models.xgb.model import XGBoostRegressor
        return XGBoostRegressor.load(base_path)
    if kind == "mlp":
        from retrotac.models.mlp.model import TorchMLPRegressor
        return TorchMLPRegressor.load(base_path)
    if kind == "gnn":
        from retrotac.models.gnn.model import CheMeleonRegressor
        return CheMeleonRegressor.load(base_path, device=device)
    raise ValueError(f"Unknown model kind: {kind!r} (expected xgb/mlp/gnn)")


def compute_features(
    smiles_list: List[str],
    fp_size: int,
    fp_radius: int,
    use_fingerprints: bool = True,
    use_descriptors: bool = True,
    n_jobs: int = 1,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Standardize SMILES and compute the model input features.

    Args:
        smiles_list: Raw SMILES strings.
        fp_size: Morgan fingerprint bit-vector length.
        fp_radius: Morgan fingerprint bond-hop radius.
        use_fingerprints: Whether to compute the fingerprint block.
        use_descriptors: Whether to compute the RDKit descriptor block.
        n_jobs: Parallelism forwarded to retrotac.chem_utils.papply for each
            standardize/parse/featurize step; 1 (default) runs sequentially
            in-process. See papply's own docstring for the -1/positive-int
            semantics.

    Returns:
        Tuple of (fingerprints, descriptors); either is None when its
        corresponding use_* flag is False.
    """
    from retrotac.chem_utils import (
        compute_descriptors,
        compute_fingerprints,
        papply,
        smiles_to_mol,
        std_smiles,
    )

    smiles_series = pd.Series(smiles_list)
    std_series = papply(smiles_series, std_smiles, desc="Standardizing SMILES", n_jobs=n_jobs)
    mols = papply(std_series, smiles_to_mol, desc="Parsing molecules", n_jobs=n_jobs)

    X_fp = None
    if use_fingerprints:
        fp_series = papply(
            mols,
            partial(compute_fingerprints, fp_size=fp_size, fp_radius=fp_radius),
            desc="Computing fingerprints",
            n_jobs=n_jobs,
        )
        X_fp = np.vstack(fp_series.tolist())

    X_desc = None
    if use_descriptors:
        desc_series = papply(mols, compute_descriptors, desc="Computing descriptors", n_jobs=n_jobs)
        X_desc = np.vstack(desc_series.tolist())

    return X_fp, X_desc


def parse_fold_model_name(name: str) -> Tuple[str, int, int]:
    """Parse a Caruana-weights-file key into (backend_kind, seed, fold).

    Args:
        name: e.g. "MLP_seed42_fold2" (see ensemble_weights_caruana.json).

    Returns:
        (backend_kind, seed, fold), e.g. ("mlp", 42, 2) -- backend_kind is
        lowercased to match load_backend's `kind` and outputs/cv/'s
        `{kind}_<run>` directory naming.

    Raises:
        ValueError: If `name` doesn't match the "{BACKEND}_seed{N}_fold{N}" pattern.
    """
    m = _FOLD_MODEL_NAME_RE.match(name)
    if not m:
        raise ValueError(
            f"Model name {name!r} doesn't match '<BACKEND>_seed<N>_fold<N>'"
        )
    backend, seed, fold = m.groups()
    return backend.lower(), int(seed), int(fold)
