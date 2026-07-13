"""
gnn/hpo.py
==========
build_gnn: tune the CheMeleon fine-tuning head + training. The backbone
(pretrained D-MPNN) is fixed, so only the FFN head + optimizer are tuned.
"""
from pathlib import Path
import yaml
 
from gnn.model import CheMeleonRegressor
 
CONFIG_PATH = Path(__file__).parent.parent / "models_config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)
 
MAX_EPOCHS        = CFG["torch"]["max_epochs"]
PATIENCE          = CFG["torch"]["patience"]
CHEMELEON_WEIGHTS = CFG.get("gnn", {}).get("chemeleon_weights", "chemeleon_mp.pt")
 
 
def build_gnn(trial, fp_radius, smiles_tr, y_tr, X_fp_tr, X_desc_tr,
              smiles_val=None, y_val=None, X_fp_val=None, X_desc_val=None, **kwargs):
    gnn_params = {
        "ffn_hidden_dim": trial.suggest_categorical("ffn_hidden_dim", [128, 300, 512]),
        "ffn_n_layers":   trial.suggest_int("ffn_n_layers", 1, 3),
        "dropout":        trial.suggest_float("dropout", 0.0, 0.4),
        "max_lr":         trial.suggest_float("max_lr", 1e-4, 5e-3, log=True),
        "batch_size":     trial.suggest_categorical("batch_size", [32, 64, 128]),
    }
    model = CheMeleonRegressor(
        gnn_params=gnn_params,
        chemeleon_weights=CHEMELEON_WEIGHTS,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
    )
    return model.fit(
        smiles_tr, y_tr, X_fp=X_fp_tr, X_desc=X_desc_tr,
        smiles_val=smiles_val, y_val=y_val,
        X_fp_val=X_fp_val, X_desc_val=X_desc_val,
    )