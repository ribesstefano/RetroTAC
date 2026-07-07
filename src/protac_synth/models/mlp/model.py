import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score          # for score()
from pathlib import Path
import skops.io as sio
from typing import List, Optional
from mol_utils import compute_fingerprints, sanitize_matrix, make_preprocessor
from torch_common import get_device, seed_everything, EarlyStopper, TabularDataset

class MLPNet(nn.Module):
    def __init__(self, input_dim, n_layers, hidden_dim, n_targets,
                 dropout=0.2, funnel=True):
        super().__init__()
        layers = []
        in_dim = input_dim                      # first block reads the feature vector
        width  = hidden_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, width))     # learnable transform
            layers.append(nn.BatchNorm1d(width))        # normalize activations
            layers.append(nn.ReLU())                    # nonlinearity
            layers.append(nn.Dropout(dropout))          # regularization
            in_dim = width                              # next block's input = this width
            if funnel:
                width = max(width // 2, n_targets)       # halve width each layer (optional)

        layers.append(nn.Linear(in_dim, n_targets))     # output head: no BN/activation

        self.net = nn.Sequential(*layers)               # pack the list into a module

    def forward(self, x):
        return self.net(x)

class TorchMLPRegressor():
    def __init__(self, fp_size=512, fp_radius=2, svd_components=64,
                 use_fingerprints=True, use_descriptors=True,
                 mlp_params=None, max_epochs=500, patience=20,
                 device="auto", uncharge=False, random_state=42):
        if not use_fingerprints and not use_descriptors:
            raise ValueError("At least one of use_fingerprints/use_descriptors must be True.")
        self.fp_size          = fp_size
        self.fp_radius        = fp_radius
        self.svd_components    = svd_components
        self.use_fingerprints = use_fingerprints
        self.use_descriptors  = use_descriptors
        self.mlp_params       = mlp_params or {}
        self.max_epochs       = max_epochs
        self.patience         = patience
        self.device           = device
        self.uncharge         = uncharge
        self.random_state     = random_state

    def _featurize(
        self,
        smiles_list: List[str],
        X_fp: Optional[np.ndarray] = None,
        X_desc: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.use_fingerprints:
            if X_fp is None:
                # Compute on-the-fly — used during HPO when radius is being tuned
                X_fp = compute_fingerprints(smiles_list, self.fp_size, self.fp_radius)
            X_fp = sanitize_matrix(X_fp)
        if self.use_descriptors and X_desc is None:
            raise ValueError(
                "use_descriptors=True but no descriptor matrix (X_desc) was provided."
            )

        parts = []
        if self.use_fingerprints:
            parts.append(X_fp)
        if self.use_descriptors:
            parts.append(sanitize_matrix(X_desc))

        return np.hstack(parts) if len(parts) > 1 else parts[0]

    def fit(self, smiles_list, y, X_fp=None, X_desc=None,
            smiles_val=None, y_val=None, X_fp_val=None, X_desc_val=None,
            trial=None) -> 'TorchMLPRegressor':
        """Fit TorchMLPRegressor to featurized SMILES.

        Args:
            smiles_list (List[str]): Training SMILES strings.
            y (np.ndarray): Training targets, shape [n_samples, n_targets].
            X_fp (np.ndarray, optional): Pre-computed training fingerprints,
                required when use_fingerprints=True.
            smiles_val (List[str], optional): Validation SMILES for early stopping.
            y_val (np.ndarray, optional): Validation targets for early stopping.
            X_fp_val (np.ndarray, optional): Pre-computed validation fingerprints,
                required when use_fingerprints=True and smiles_val is provided.

        Returns:
            self
        """
        X = self._featurize(smiles_list, X_fp, X_desc)
        n_fp_cols   = self.fp_size if self.use_fingerprints else 0
        n_desc_cols = X_desc.shape[1] if (self.use_descriptors and X_desc is not None) else 0

        self.preprocessor_ = make_preprocessor(
            self.use_fingerprints, self.use_descriptors,
            n_fp_cols, n_desc_cols,
            self.svd_components, self.random_state,
        )

        self.target_transformer_ = QuantileTransformer(output_distribution='normal', random_state=self.random_state)
        X_proc = self.preprocessor_.fit_transform(X)
        y_proc = self.target_transformer_.fit_transform(y)

        # ── read hyperparameters (tuned) ────────────────────────────────
        p             = self.mlp_params
        n_layers      = p.get("n_layers", 2)
        hidden_dim    = p.get("hidden_dim", 256)
        dropout       = p.get("dropout", 0.2)
        learning_rate = p.get("learning_rate", 1e-3)
        weight_decay  = p.get("weight_decay", 1e-4)
        batch_size    = p.get("batch_size", 64)

        # ── dims (data-derived) ─────────────────────────────────────────
        input_dim = X_proc.shape[1]
        n_targets = y_proc.shape[1]

        # ── torch objects ───────────────────────────────────────────────
        seed_everything(self.random_state)          # seed BEFORE building the net
        self.device_ = get_device(self.device)

        self.arch_ = dict(input_dim=input_dim, n_layers=n_layers,
                          hidden_dim=hidden_dim, n_targets=n_targets, dropout=dropout)
        model     = MLPNet(**self.arch_).to(self.device_)

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.SmoothL1Loss()
        stopper   = EarlyStopper(patience=self.patience)

        # ── train / val split ────────────────────────────────────────────
        if smiles_val is not None:
            X_val_proc = self.preprocessor_.transform(
                self._featurize(smiles_val, X_fp_val, X_desc_val))
            y_val_proc = self.target_transformer_.transform(y_val)
            X_tr_proc, y_tr_proc = X_proc, y_proc
        else:
            rng     = np.random.default_rng(self.random_state)
            idx     = rng.permutation(len(X_proc))
            n_val   = max(1, int(0.1 * len(X_proc)))
            v, t    = idx[:n_val], idx[n_val:]
            X_tr_proc, y_tr_proc   = X_proc[t], y_proc[t]
            X_val_proc, y_val_proc = X_proc[v], y_proc[v]

        loader = DataLoader(TabularDataset(X_tr_proc, y_tr_proc),
                    batch_size=batch_size, shuffle=True, drop_last=True)
        Xv = torch.as_tensor(X_val_proc, dtype=torch.float32).to(self.device_)
        yv = torch.as_tensor(y_val_proc, dtype=torch.float32).to(self.device_)

        # ── epoch loop ───────────────────────────────────────────────
        for epoch in range(self.max_epochs):
            model.train()
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device_), yb.to(self.device_)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(Xv), yv).item()

            if trial is not None:
                trial.report(-val_loss, epoch)      # negate: higher = better, matches "maximize"
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if stopper.step(val_loss, model):
                break

        stopper.restore(model)
        self.model_ = model
        return self
    
    def predict(self, smiles_list, X_fp=None, X_desc=None):
        X_proc = self.preprocessor_.transform(self._featurize(smiles_list, X_fp, X_desc))
        self.model_.eval()
        with torch.no_grad():
            X_t    = torch.as_tensor(X_proc, dtype=torch.float32).to(self.device_)
            y_pred = self.model_(X_t).cpu().numpy()
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        return self.target_transformer_.inverse_transform(y_pred)

    def score(self, smiles_list, y, X_fp=None, X_desc=None):
        y_pred = self.predict(smiles_list, X_fp, X_desc)
        return float(np.mean([r2_score(y[:, i], y_pred[:, i]) for i in range(y.shape[1])]))

    def save(self, path):
        if not hasattr(self, 'model_'):
            raise ValueError("Cannot save an unfitted model. Call fit() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # network weights + architecture (to rebuild) -> .pt
        torch.save({"state_dict": self.model_.state_dict(), "arch": self.arch_}, f"{path}.pt")
        # sklearn side (preprocessor + target transformer + config) -> .skops
        net, dev = self.model_, self.device_
        self.model_, self.device_ = None, None
        sio.dump(self, f"{path}.skops")
        self.model_, self.device_ = net, dev   
    
    @classmethod
    def load(cls, path):
        unknown = sio.get_untrusted_types(file=f"{path}.skops")
        allowed = ('TorchMLPRegressor', 'numpy.dtype')
        for t in unknown:
            if not any(a in t for a in allowed):
                raise ValueError(f"Untrusted type '{t}' in skops file. Aborting load.")
        instance = sio.load(f"{path}.skops", trusted=unknown)
        ckpt  = torch.load(f"{path}.pt", map_location="cpu")
        model = MLPNet(**ckpt["arch"])
        model.load_state_dict(ckpt["state_dict"])
        instance.device_ = get_device(instance.device)
        instance.model_  = model.to(instance.device_)
        return instance