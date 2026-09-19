from typing import Any, List, Optional

import numpy as np

from retrotac.models.xgb.model import XGBoostRegressor


def build_xgb(
    trial: Any,
    fp_radius: int,
    smiles_tr: List[str],
    y_tr: np.ndarray,
    X_fp_tr: Optional[np.ndarray],
    X_desc_tr: Optional[np.ndarray],
    smiles_val: Optional[List[str]] = None,
    y_val: Optional[np.ndarray] = None,
    X_fp_val: Optional[np.ndarray] = None,
    X_desc_val: Optional[np.ndarray] = None,
    fp_size: int = 512,
    use_fingerprints: bool = True,
    use_descriptors: bool = True,
    device: str = "cpu",
    objective: str = "reg:pseudohubererror",
    **kwargs: Any,
) -> XGBoostRegressor:
    """Build and fit an XGBoost regressor.

    Args:
        trial: Optuna trial or FixedTrial with hyperparameter values.
        fp_radius: Morgan fingerprint radius.
        smiles_tr: Training SMILES strings.
        y_tr: Training targets.
        X_fp_tr: Training fingerprint matrix or None.
        X_desc_tr: Training descriptor matrix or None.
        smiles_val: Validation SMILES for early stopping (optional).
        y_val: Validation targets for early stopping (optional).
        X_fp_val: Validation fingerprint matrix (optional).
        X_desc_val: Validation descriptor matrix (optional).
        fp_size: Morgan fingerprint bit size.
        use_fingerprints: Whether the feature matrix includes fingerprints.
        use_descriptors: Whether the feature matrix includes RDKit descriptors.
        device: Compute device for XGBoost (e.g. "cpu", "cuda").
        objective: XGBoost regression objective, e.g. "reg:pseudohubererror"
            (default) or "reg:tweedie". When "reg:tweedie", tweedie_variance_power
            is also added to the Optuna search space (range (1, 2), bounded to
            [1.01, 1.99] to stay inside XGBoost's valid open interval).

    Returns:
        Fitted XGBoostRegressor.
    """
    xgb_params = {
        "objective": objective,
        "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }
    if objective == "reg:tweedie":
        xgb_params["tweedie_variance_power"] = trial.suggest_float(
            "tweedie_variance_power", 1.01, 1.99
        )

    return XGBoostRegressor(
        fp_size=fp_size,
        fp_radius=fp_radius,
        svd_components=trial.suggest_int("svd_components", 0, 128),
        use_fingerprints=use_fingerprints,
        use_descriptors=use_descriptors,
        device=device,
        xgb_params=xgb_params,
    ).fit(
        smiles_tr, y_tr,
        X_fp=X_fp_tr, X_desc=X_desc_tr,
        smiles_val=smiles_val, y_val=y_val,
        X_fp_val=X_fp_val, X_desc_val=X_desc_val,
    )
