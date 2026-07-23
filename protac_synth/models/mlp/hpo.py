import optuna

from protac_synth.models.mlp.model import TorchMLPRegressor


def build_mlp(trial, fp_radius, smiles_tr, y_tr, X_fp_tr, X_desc_tr,
              smiles_val=None, y_val=None, X_fp_val=None, X_desc_val=None,
              fp_size=512, use_fingerprints=True, use_descriptors=True, **kwargs):
    """Build and fit a Torch MLP regressor.

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
        Fitted TorchMLPRegressor.
    """
    mlp_params = {
        "n_layers":      trial.suggest_int("n_layers", 1, 4),
        "hidden_dim":    trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
        "dropout":       trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }
    model = TorchMLPRegressor(
        fp_size=fp_size, fp_radius=fp_radius,
        svd_components=trial.suggest_int("svd_components", 0, 128),
        use_fingerprints=use_fingerprints, use_descriptors=use_descriptors,
        mlp_params=mlp_params,
    )
    # only a real Optuna Trial supports report()/should_prune()
    prune_trial = trial if isinstance(trial, optuna.Trial) else None
    return model.fit(
        smiles_tr, y_tr, X_fp=X_fp_tr, X_desc=X_desc_tr,
        smiles_val=smiles_val, y_val=y_val,
        X_fp_val=X_fp_val, X_desc_val=X_desc_val,
        trial=prune_trial,
    )
