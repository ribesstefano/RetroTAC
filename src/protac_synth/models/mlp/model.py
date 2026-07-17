"""
mlp/model.py  (Lightning version)
=================================
MLP model (MLPNet architecture, AdamW, SmoothL1, early stopping,
best-weight restore, Optuna pruning), the training loop is
replaced by a LightningModule + pl.Trainer.
"""
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import optuna
import torch
import torch.nn as nn
import skops.io as sio
from torch.utils.data import Dataset, DataLoader
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score

from mol_utils import compute_fingerprints, sanitize_matrix, make_preprocessor


# ── tabular dataset ─────────────────────────────────────────────────────────
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── architecture ─────────────────────────────────────────────────────────────
class MLPNet(nn.Module):
    def __init__(self, input_dim, n_layers, hidden_dim, n_targets, dropout=0.2):
        super().__init__()
        layers, in_dim = [], input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── LightningModule: the training/validation/optimizer logic ────────────────
class LitMLP(pl.LightningModule):
    def __init__(self, input_dim, n_layers, hidden_dim, n_targets, dropout,
                 learning_rate, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.net       = MLPNet(input_dim, n_layers, hidden_dim, n_targets, dropout)
        self.criterion = nn.SmoothL1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", self.criterion(self(x), y), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.hparams.learning_rate,
                                 weight_decay=self.hparams.weight_decay)


# ── sklearn-style wrapper matching the harness ──────────────────────────────
class TorchMLPRegressor:
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

    def _featurize(self, smiles_list, X_fp=None, X_desc=None):
        if self.use_fingerprints:
            if X_fp is None:
                X_fp = compute_fingerprints(smiles_list, self.fp_size, self.fp_radius)
            X_fp = sanitize_matrix(X_fp)
        if self.use_descriptors and X_desc is None:
            raise ValueError("use_descriptors=True but no descriptor matrix was provided.")
        parts = []
        if self.use_fingerprints:
            parts.append(X_fp)
        if self.use_descriptors:
            parts.append(sanitize_matrix(X_desc))
        return np.hstack(parts) if len(parts) > 1 else parts[0]

    def fit(self, smiles_list, y, X_fp=None, X_desc=None,
            smiles_val=None, y_val=None, X_fp_val=None, X_desc_val=None, trial=None):
        # ── preprocess (same as before) ────────────────────────────────────
        X = self._featurize(smiles_list, X_fp, X_desc)
        n_fp_cols   = self.fp_size if self.use_fingerprints else 0
        n_desc_cols = X_desc.shape[1] if (self.use_descriptors and X_desc is not None) else 0
        self.preprocessor_ = make_preprocessor(
            self.use_fingerprints, self.use_descriptors,
            n_fp_cols, n_desc_cols, self.svd_components, self.random_state)
        self.target_transformer_ = QuantileTransformer(
            output_distribution='normal', random_state=self.random_state)
        X_proc = self.preprocessor_.fit_transform(X)
        y_proc = self.target_transformer_.fit_transform(y)

        # ── hyperparameters + dims ─────────────────────────────────────────
        p             = self.mlp_params
        n_layers      = p.get("n_layers", 2)
        hidden_dim    = p.get("hidden_dim", 256)
        dropout       = p.get("dropout", 0.2)
        learning_rate = p.get("learning_rate", 1e-3)
        weight_decay  = p.get("weight_decay", 1e-4)
        batch_size    = p.get("batch_size", 64)

        input_dim = X_proc.shape[1]
        n_targets = y_proc.shape[1]
        self.arch_ = dict(input_dim=input_dim, n_layers=n_layers, hidden_dim=hidden_dim,
                          n_targets=n_targets, dropout=dropout,
                          learning_rate=learning_rate, weight_decay=weight_decay)

        pl.seed_everything(self.random_state, workers=True)
        lit = LitMLP(**self.arch_)

        # ── train / val split ──────────────────────────────────────────────
        if smiles_val is not None:
            X_val_proc = self.preprocessor_.transform(
                self._featurize(smiles_val, X_fp_val, X_desc_val))
            y_val_proc = self.target_transformer_.transform(y_val)
            X_tr_proc, y_tr_proc = X_proc, y_proc
        else:
            rng   = np.random.default_rng(self.random_state)
            idx   = rng.permutation(len(X_proc))
            n_val = max(1, int(0.1 * len(X_proc)))
            v, t  = idx[:n_val], idx[n_val:]
            X_tr_proc, y_tr_proc   = X_proc[t], y_proc[t]
            X_val_proc, y_val_proc = X_proc[v], y_proc[v]

        train_loader = DataLoader(TabularDataset(X_tr_proc, y_tr_proc),
                                  batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(TabularDataset(X_val_proc, y_val_proc),
                                  batch_size=batch_size, shuffle=False)

        # ── callbacks: early stopping + best-weights + optional pruning ────
        tmpdir  = tempfile.mkdtemp()
        early   = EarlyStopping(monitor="val_loss", patience=self.patience, mode="min")
        ckpt_cb = ModelCheckpoint(dirpath=tmpdir, monitor="val_loss", mode="min",
                                  save_top_k=1, filename="best")
        callbacks = [early, ckpt_cb]
        if isinstance(trial, optuna.Trial):
            try:
                from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
            except ImportError:
                from optuna.integration import PyTorchLightningPruningCallback
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss"))

        trainer = pl.Trainer(
            accelerator=("cpu" if self.device == "cpu" else "auto"), devices=1,
            max_epochs=self.max_epochs, logger=False,
            enable_checkpointing=True, enable_progress_bar=False, callbacks=callbacks)
        trainer.fit(lit, train_loader, val_loader)

        if ckpt_cb.best_model_path:
            lit = LitMLP.load_from_checkpoint(ckpt_cb.best_model_path)
        shutil.rmtree(tmpdir, ignore_errors=True)

        self.model_ = lit
        return self

    def predict(self, smiles_list, X_fp=None, X_desc=None):
        X_proc = self.preprocessor_.transform(self._featurize(smiles_list, X_fp, X_desc))
        self.model_.eval()
        device = next(self.model_.parameters()).device
        with torch.no_grad():
            X_t    = torch.as_tensor(X_proc, dtype=torch.float32).to(device)
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
        torch.save({"state_dict": self.model_.state_dict(), "arch": self.arch_}, f"{path}.pt")
        net = self.model_
        self.model_ = None
        sio.dump(self, f"{path}.skops")
        self.model_ = net

    @classmethod
    def load(cls, path):
        unknown = sio.get_untrusted_types(file=f"{path}.skops")
        allowed = ('TorchMLPRegressor', 'numpy.dtype')
        for t in unknown:
            if not any(a in t for a in allowed):
                raise ValueError(f"Untrusted type '{t}' in skops file. Aborting load.")
        instance = sio.load(f"{path}.skops", trusted=unknown)
        ckpt = torch.load(f"{path}.pt", map_location="cpu")
        lit  = LitMLP(**ckpt["arch"])
        lit.load_state_dict(ckpt["state_dict"])
        lit.eval()
        instance.model_ = lit
        return instance