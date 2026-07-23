from protac_synth.models.xgb.model import XGBoostRegressor


def build_xgb(trial, fp_radius, smiles_tr, y_tr, X_fp_tr, X_desc_tr,
              smiles_val=None, y_val=None, X_fp_val=None, X_desc_val=None,
              fp_size=512, use_fingerprints=True, use_descriptors=True, **kwargs):
    """Build and fit an XGBoost regressor.

    Args:
        trial:            Optuna trial or FixedTrial with hyperparameter values.
        fp_radius:        Morgan fingerprint radius.
        smiles_tr:        Training SMILES strings.
        y_tr:             Training targets.
        X_fp_tr:          Training fingerprint matrix or None.
        X_desc_tr:        Training descriptor matrix or None.
        smiles_val:       Validation SMILES for early stopping (optional).
        y_val:            Validation targets for early stopping (optional).
        X_fp_val:         Validation fingerprint matrix (optional).
        X_desc_val:       Validation descriptor matrix (optional).
        fp_size:          Morgan fingerprint bit size.
        use_fingerprints: Whether the feature matrix includes fingerprints.
        use_descriptors:  Whether the feature matrix includes RDKit descriptors.

    Returns:
        Fitted XGBoostRegressor.
    """
    return XGBoostRegressor(
        fp_size          = fp_size,
        fp_radius        = fp_radius,
        svd_components   = trial.suggest_int("svd_components", 0, 128),
        use_fingerprints = use_fingerprints,
        use_descriptors  = use_descriptors,
        xgb_params       = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 3000),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        },
    ).fit(smiles_tr, y_tr,
        X_fp=X_fp_tr, X_desc=X_desc_tr,
        smiles_val=smiles_val, y_val=y_val,
        X_fp_val=X_fp_val, X_desc_val=X_desc_val,
    )
