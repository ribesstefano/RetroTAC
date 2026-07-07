import optuna
from mlp.model import TorchMLPRegressor
from pathlib import Path
import yaml
CONFIG_PATH = Path(__file__).parent.parent / "models_config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

FP_SIZE          = CFG["features"]["fp_size"]
FP_RADIUS        = CFG["features"]["fp_radius"]
USE_FINGERPRINTS = CFG["features"]["use_fingerprints"]
USE_DESCRIPTORS  = CFG["features"]["use_descriptors"]

def build_mlp(trial, fp_radius, smiles_tr, y_tr, X_fp_tr, X_desc_tr,
              smiles_val=None, y_val=None, X_fp_val=None, X_desc_val=None, **kwargs):
    mlp_params = {
        "n_layers":      trial.suggest_int("n_layers", 1, 4),
        "hidden_dim":    trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
        "dropout":       trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }
    model = TorchMLPRegressor(
        fp_size=FP_SIZE, fp_radius=fp_radius,
        svd_components=trial.suggest_int("svd_components", 0, 128),
        use_fingerprints=USE_FINGERPRINTS, use_descriptors=USE_DESCRIPTORS,
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