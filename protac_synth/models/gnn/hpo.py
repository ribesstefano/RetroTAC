"""
gnn/hpo.py
==========
build_gnn: tune the CheMeleon fine-tuning head + training. The backbone
(pretrained D-MPNN) is fixed, so only the FFN head + optimizer are tuned.
"""
from protac_synth.models.gnn.model import CheMeleonRegressor


def build_gnn(trial, fp_radius, smiles_tr, y_tr, X_fp_tr, X_desc_tr,
              smiles_val=None, y_val=None, X_fp_val=None, X_desc_val=None,
              max_epochs=500, patience=20, chemeleon_weights="chemeleon_mp.pt", **kwargs):
    """Build and fit the CheMeleon GNN regressor (fixed backbone, tuned FFN head).

    The GNN featurizes graphs directly from SMILES, so fp_radius / X_fp* / X_desc*
    are accepted for a uniform build signature but ignored.

    Args:
        trial:             Optuna trial or FixedTrial with hyperparameter values.
        smiles_tr:         Training SMILES strings.
        y_tr:              Training targets.
        smiles_val:        Validation SMILES for early stopping (optional).
        y_val:             Validation targets for early stopping (optional).
        max_epochs:        Maximum training epochs.
        patience:          Early-stopping patience in epochs.
        chemeleon_weights: Path to the pretrained CheMeleon backbone checkpoint.

    Returns:
        Fitted CheMeleonRegressor.
    """
    gnn_params = {
        "ffn_hidden_dim": trial.suggest_categorical("ffn_hidden_dim", [128, 300, 512]),
        "ffn_n_layers":   trial.suggest_int("ffn_n_layers", 1, 3),
        "dropout":        trial.suggest_float("dropout", 0.0, 0.4),
        "max_lr":         trial.suggest_float("max_lr", 1e-4, 5e-3, log=True),
        "batch_size":     trial.suggest_categorical("batch_size", [32, 64, 128]),
    }
    model = CheMeleonRegressor(
        gnn_params=gnn_params,
        chemeleon_weights=chemeleon_weights,
        max_epochs=max_epochs,
        patience=patience,
    )
    return model.fit(
        smiles_tr, y_tr, X_fp=X_fp_tr, X_desc=X_desc_tr,
        smiles_val=smiles_val, y_val=y_val,
        X_fp_val=X_fp_val, X_desc_val=X_desc_val,
    )
