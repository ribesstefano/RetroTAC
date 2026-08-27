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
from typing import Dict, List, Optional, Tuple

import huggingface_hub as hf
import numpy as np
import optuna
import skops.io as sio
import torch
import torch.nn as nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, Dataset

from protac_synth.chem_utils import (  # noqa: E402
    compute_fingerprints,
    make_preprocessor,
    sanitize_matrix,
    standardize_all,
)


# ── tabular dataset ─────────────────────────────────────────────────────────
class TabularDataset(Dataset):
    """Wraps a preprocessed feature/target matrix pair as a torch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ── architecture ─────────────────────────────────────────────────────────────
class MLPNet(nn.Module):
    """Plain feed-forward net: n_layers of Linear-BatchNorm-ReLU-Dropout + a
    final linear regression head."""

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        hidden_dim: int,
        n_targets: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Stack n_layers of Linear→BatchNorm→ReLU→Dropout, then a final head.
        layers, in_dim = [], input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── LightningModule: the training/validation/optimizer logic ────────────────
class LitMLP(pl.LightningModule):
    """LightningModule wrapping MLPNet with the train/val loop and optimizer.

    save_hyperparameters() persists the constructor args so
    load_from_checkpoint() can rebuild the exact same architecture without
    external bookkeeping (see TorchMLPRegressor.load).
    """

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        hidden_dim: int,
        n_targets: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
    ):
        super().__init__()
        # Persist ctor args into self.hparams so load_from_checkpoint() can
        # rebuild the exact same architecture without external bookkeeping.
        self.save_hyperparameters()
        self.net = MLPNet(input_dim, n_layers, hidden_dim, n_targets, dropout)
        self.criterion = nn.SmoothL1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        # Logged as "val_loss" — the metric early stopping / checkpoint watch.
        self.log("val_loss", self.criterion(self(x), y), prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )


# ── sklearn-style wrapper matching the harness ──────────────────────────────
class TorchMLPRegressor:
    """Sklearn-style wrapper around a Lightning-trained MLP (LitMLP/MLPNet).

    Handles feature assembly (Morgan fingerprints + RDKit descriptors),
    preprocessing (SVD/imputation/scaling via make_preprocessor), target
    scaling (QuantileTransformer), and the Lightning train/early-stop loop
    behind a fit/predict/score/save/load contract shared with the xgb/gnn
    backends (see training.py).
    """

    def __init__(
        self,
        fp_size: int = 512,
        fp_radius: int = 2,
        svd_components: int = 64,
        use_fingerprints: bool = True,
        use_descriptors: bool = True,
        mlp_params: Optional[Dict] = None,
        max_epochs: int = 500,
        patience: int = 20,
        device: str = "auto",
        uncharge: bool = False,
        random_state: int = 42,
    ):
        """Store config; the LitMLP itself is built lazily in fit().

        Args:
            fp_size: Morgan fingerprint bit-vector length.
            fp_radius: Morgan fingerprint bond-hop radius.
            svd_components: TruncatedSVD output dims for the fingerprint
                block passed to make_preprocessor; 0 disables SVD.
            use_fingerprints: Whether the feature matrix includes fingerprints.
            use_descriptors: Whether the feature matrix includes RDKit descriptors.
            mlp_params: Architecture/optimizer hyperparameters (n_layers,
                hidden_dim, dropout, learning_rate, weight_decay, batch_size);
                missing keys fall back to defaults inside fit().
            max_epochs: Maximum training epochs.
            patience: Early-stopping patience in epochs.
            device: Lightning accelerator selector ("auto", "cpu", or GPU).
            uncharge: Unused by this backend; kept for config parity.
            random_state: Seed for Lightning's global RNG, the fallback
                train/val split, and the target QuantileTransformer.

        Raises:
            ValueError: If both use_fingerprints and use_descriptors are False.
        """
        if not use_fingerprints and not use_descriptors:
            raise ValueError(
                "At least one of use_fingerprints/use_descriptors must be True."
            )
        self.fp_size = fp_size
        self.fp_radius = fp_radius
        self.svd_components = svd_components
        self.use_fingerprints = use_fingerprints
        self.use_descriptors = use_descriptors
        self.mlp_params = mlp_params or {}
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.uncharge = uncharge
        self.random_state = random_state

    def _featurize(
        self,
        smiles_list: List[str],
        X_fp: Optional[np.ndarray] = None,
        X_desc: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Assemble the [fingerprint | descriptor] feature matrix.

        Args:
            smiles_list: SMILES strings; used to compute fingerprints only
                when X_fp is not already supplied.
            X_fp: Pre-computed fingerprint matrix, or None to compute on demand.
            X_desc: Pre-computed descriptor matrix; required when
                use_descriptors=True.

        Returns:
            Horizontally stacked, sanitized feature matrix (whichever blocks
            are enabled).

        Raises:
            ValueError: If use_descriptors=True but X_desc is None.
        """
        if self.use_fingerprints:
            # Compute fingerprints on demand only when not pre-supplied.
            if X_fp is None:
                X_fp = compute_fingerprints(
                    standardize_all(smiles_list), self.fp_size, self.fp_radius
                )
            X_fp = sanitize_matrix(X_fp)
        if self.use_descriptors and X_desc is None:
            raise ValueError(
                "use_descriptors=True but no descriptor matrix was provided."
            )
        parts = []
        if self.use_fingerprints:
            parts.append(X_fp)
        if self.use_descriptors:
            parts.append(sanitize_matrix(X_desc))
        return np.hstack(parts) if len(parts) > 1 else parts[0]

    def fit(
        self,
        smiles_list: List[str],
        y: np.ndarray,
        X_fp: Optional[np.ndarray] = None,
        X_desc: Optional[np.ndarray] = None,
        smiles_val: Optional[List[str]] = None,
        y_val: Optional[np.ndarray] = None,
        X_fp_val: Optional[np.ndarray] = None,
        X_desc_val: Optional[np.ndarray] = None,
        trial: Optional[optuna.Trial] = None,
    ) -> "TorchMLPRegressor":
        """Fit the MLP to featurized SMILES with early stopping.

        Args:
            smiles_list (List[str]): Training SMILES strings.
            y (np.ndarray): Training targets, shape [n_samples, n_targets].
            X_fp (np.ndarray, optional): Pre-computed training fingerprints.
            X_desc (np.ndarray, optional): Pre-computed training descriptors.
            smiles_val (List[str], optional): Validation SMILES; when omitted a
                10% split is carved from the training data instead.
            y_val (np.ndarray, optional): Validation targets.
            X_fp_val (np.ndarray, optional): Pre-computed validation fingerprints.
            X_desc_val (np.ndarray, optional): Pre-computed validation descriptors.
            trial (optuna.Trial, optional): Live Optuna trial enabling pruning.

        Returns:
            self
        """
        # ── preprocess: feature matrix + fitted feature/target transformers ─
        X = self._featurize(smiles_list, X_fp, X_desc)
        n_fp_cols = self.fp_size if self.use_fingerprints else 0
        n_desc_cols = (
            X_desc.shape[1] if (self.use_descriptors and X_desc is not None) else 0
        )
        self.preprocessor_ = make_preprocessor(
            self.use_fingerprints,
            self.use_descriptors,
            n_fp_cols,
            n_desc_cols,
            self.svd_components,
            self.random_state,
        )
        # Map targets to a normal distribution — stabilizes the SmoothL1 loss
        # and is inverted at predict time.
        self.target_transformer_ = QuantileTransformer(
            output_distribution="normal", random_state=self.random_state
        )
        X_proc = self.preprocessor_.fit_transform(X)
        y_proc = self.target_transformer_.fit_transform(y)

        # ── hyperparameters + dims ─────────────────────────────────────────
        p = self.mlp_params
        n_layers = p.get("n_layers", 2)
        hidden_dim = p.get("hidden_dim", 256)
        dropout = p.get("dropout", 0.2)
        learning_rate = p.get("learning_rate", 1e-3)
        weight_decay = p.get("weight_decay", 1e-4)
        batch_size = p.get("batch_size", 64)

        input_dim = X_proc.shape[1]
        n_targets = y_proc.shape[1]
        # Stash the architecture so save()/load() can rebuild the net exactly.
        self.arch_ = dict(
            input_dim=input_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            n_targets=n_targets,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        pl.seed_everything(self.random_state, workers=True)
        lit = LitMLP(**self.arch_)

        # ── train / val split ──────────────────────────────────────────────
        if smiles_val is not None:
            # Caller-provided validation set — transform it with the *fitted*
            # preprocessor (no refit) so train/val features stay comparable.
            X_val_proc = self.preprocessor_.transform(
                self._featurize(smiles_val, X_fp_val, X_desc_val)
            )
            y_val_proc = self.target_transformer_.transform(y_val)
            X_tr_proc, y_tr_proc = X_proc, y_proc
        else:
            # No validation set: hold out a random 10% for early stopping.
            rng = np.random.default_rng(self.random_state)
            idx = rng.permutation(len(X_proc))
            n_val = max(1, int(0.1 * len(X_proc)))
            v, t = idx[:n_val], idx[n_val:]
            X_tr_proc, y_tr_proc = X_proc[t], y_proc[t]
            X_val_proc, y_val_proc = X_proc[v], y_proc[v]

        train_loader = DataLoader(
            TabularDataset(X_tr_proc, y_tr_proc),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            TabularDataset(X_val_proc, y_val_proc), batch_size=batch_size, shuffle=False
        )

        # ── callbacks: early stopping + best-weights + optional pruning ────
        tmpdir = tempfile.mkdtemp()
        early = EarlyStopping(monitor="val_loss", patience=self.patience, mode="min")
        ckpt_cb = ModelCheckpoint(
            dirpath=tmpdir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best",
        )
        callbacks = [early, ckpt_cb]
        # Only a real Optuna Trial (not a FixedTrial) supports pruning; the
        # integration import path moved between Optuna versions, hence the fallback.
        if isinstance(trial, optuna.Trial):
            try:
                from optuna_integration.pytorch_lightning import (
                    PyTorchLightningPruningCallback,
                )
            except ImportError:
                from optuna.integration import PyTorchLightningPruningCallback
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss"))

        trainer = pl.Trainer(
            accelerator=("cpu" if self.device == "cpu" else "auto"),
            devices=1,
            max_epochs=self.max_epochs,
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=False,
            callbacks=callbacks,
        )
        trainer.fit(lit, train_loader, val_loader)

        # Restore the best-val-loss checkpoint (not the last epoch's weights).
        if ckpt_cb.best_model_path:
            lit = LitMLP.load_from_checkpoint(ckpt_cb.best_model_path)
        shutil.rmtree(tmpdir, ignore_errors=True)

        self.model_ = lit
        return self

    def predict(
        self,
        smiles_list: List[str],
        X_fp: Optional[np.ndarray] = None,
        X_desc: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict targets for SMILES not seen during fit().

        Args:
            smiles_list: SMILES strings to predict on.
            X_fp: Pre-computed fingerprints, or None to compute on demand.
            X_desc: Pre-computed descriptors, required when use_descriptors=True.

        Returns:
            Array of shape [len(smiles_list), n_targets] in real (unscaled)
            units — the target QuantileTransformer fit during fit() is
            inverted before returning.
        """
        X_proc = self.preprocessor_.transform(
            self._featurize(smiles_list, X_fp, X_desc)
        )
        self.model_.eval()
        device = next(self.model_.parameters()).device
        with torch.no_grad():
            X_t = torch.as_tensor(X_proc, dtype=torch.float32).to(device)
            y_pred = self.model_(X_t).cpu().numpy()
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        # Undo the target QuantileTransformer to return predictions in real units.
        return self.target_transformer_.inverse_transform(y_pred)

    def score(
        self,
        smiles_list: List[str],
        y: np.ndarray,
        X_fp: Optional[np.ndarray] = None,
        X_desc: Optional[np.ndarray] = None,
    ) -> float:
        """Mean per-target R2 on (smiles_list, y).

        Args:
            smiles_list: SMILES strings to score.
            y: True targets, shape [n_samples, n_targets].
            X_fp: Pre-computed fingerprints, or None to compute on demand.
            X_desc: Pre-computed descriptors, required when use_descriptors=True.

        Returns:
            Mean R2 across target columns.
        """
        y_pred = self.predict(smiles_list, X_fp, X_desc)
        return float(
            np.mean([r2_score(y[:, i], y_pred[:, i]) for i in range(y.shape[1])])
        )

    def save(self, path: str) -> None:
        """Save to two files: {path}.pt (Torch weights) + {path}.skops (sklearn).

        The network weights are saved in Torch's native format while the
        preprocessor/target transformer go through skops, which cannot
        serialize the nn.Module reliably.

        Args:
            path (str): Base path without extension.
        """
        if not hasattr(self, "model_"):
            raise ValueError("Cannot save an unfitted model. Call fit() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.model_.state_dict(), "arch": self.arch_}, f"{path}.pt"
        )
        # Temporarily detach the net so skops only sees sklearn objects.
        net = self.model_
        self.model_ = None
        sio.dump(self, f"{path}.skops")
        self.model_ = net

    @classmethod
    def load(cls, path: str) -> "TorchMLPRegressor":
        """Load from {path}.skops and {path}.pt.

        Args:
            path (str): Base path without extension.

        Returns:
            Restored TorchMLPRegressor instance.
        """
        unknown = sio.get_untrusted_types(file=f"{path}.skops")
        allowed = ("TorchMLPRegressor", "numpy.dtype")
        for t in unknown:
            if not any(a in t for a in allowed):
                raise ValueError(f"Untrusted type '{t}' in skops file. Aborting load.")

        instance = sio.load(f"{path}.skops", trusted=unknown)
        ckpt = torch.load(f"{path}.pt", map_location="cpu")
        lit = LitMLP(**ckpt["arch"])
        lit.load_state_dict(ckpt["state_dict"])
        lit.eval()
        instance.model_ = lit
        return instance

    @classmethod
    def from_hf(cls, hf_path: str, model_id: str) -> "TorchMLPRegressor":
        """Load a model from HuggingFace Hub.

        Args:
            hf_path (str): Path to the HuggingFace Hub repository.
            model_id (str): Model ID on HF Hub, e.g. "model_name".

        Returns:
            Restored TorchMLPRegressor instance.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            hf.snapshot_download(hf_path, local_dir=tmpdir)
            path = Path(tmpdir) / model_id
            return cls.load(str(path))
