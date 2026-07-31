"""
gnn/model.py
============
CheMeleon foundation-model regressor (ChemProp D-MPNN, pretrained on Mordred
descriptors, fine-tuned on our data). Graph-based; ignores X_fp / X_desc.
Matches the fit/predict/score/save/load contract used by train.py.

Requires: chemprop>=2.2.0, lightning. Download the CheMeleon weights once:
    urlretrieve("https://zenodo.org/records/15460715/files/chemeleon_mp.pt", "chemeleon_mp.pt")
"""

import huggingface_hub as hf
import numpy as np
import torch

print("cuda available:", torch.cuda.is_available(), flush=True)
import tempfile
from pathlib import Path

from chemprop import data as cpdata
from chemprop import featurizers
from chemprop import models as cpmodels
from chemprop import nn as cpnn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score


class CheMeleonRegressor:
    def __init__(
        self,
        gnn_params=None,
        chemeleon_weights="chemeleon_mp.pt",
        max_epochs=100,
        patience=15,
        random_state=42,
        device="cpu",
    ):
        self.gnn_params = gnn_params or {}
        self.chemeleon_weights = chemeleon_weights
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def _datapoints(self, smiles_list, y=None):
        if y is None:
            return [cpdata.MoleculeDatapoint.from_smi(s) for s in smiles_list]
        return [
            cpdata.MoleculeDatapoint.from_smi(s, yi) for s, yi in zip(smiles_list, y)
        ]

    def fit(
        self,
        smiles_list,
        y,
        X_fp=None,
        X_desc=None,
        smiles_val=None,
        y_val=None,
        X_fp_val=None,
        X_desc_val=None,
        trial=None,
    ):
        pl.seed_everything(self.random_state, workers=True)

        p = self.gnn_params
        dropout = p.get("dropout", 0.0)
        ffn_hidden = p.get("ffn_hidden_dim", 300)
        ffn_layers = p.get("ffn_n_layers", 1)
        max_lr = p.get("max_lr", 1e-3)
        batch_size = p.get("batch_size", 64)

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        # ── datasets (ChemProp scales targets internally) ──────────────────
        train_dset = cpdata.MoleculeDataset(
            self._datapoints(smiles_list, y), featurizer
        )
        scaler = train_dset.normalize_targets()
        print("built datasets...", flush=True)

        if smiles_val is not None:
            val_dset = cpdata.MoleculeDataset(
                self._datapoints(smiles_val, y_val), featurizer
            )
            val_dset.normalize_targets(scaler)
        else:
            all_dp = self._datapoints(smiles_list, y)
            rng = np.random.default_rng(self.random_state)
            idx = rng.permutation(len(smiles_list))
            n_val = max(1, int(0.1 * len(smiles_list)))
            v, t = idx[:n_val], idx[n_val:]
            tr_dp = [all_dp[i] for i in t]
            va_dp = [all_dp[i] for i in v]
            train_dset = cpdata.MoleculeDataset(tr_dp, featurizer)
            scaler = train_dset.normalize_targets()
            val_dset = cpdata.MoleculeDataset(va_dp, featurizer)
            val_dset.normalize_targets(scaler)

        train_loader = cpdata.build_dataloader(
            train_dset, batch_size=batch_size, num_workers=0
        )
        val_loader = cpdata.build_dataloader(
            val_dset, batch_size=batch_size, num_workers=0, shuffle=False
        )

        # ── pretrained CheMeleon backbone + fresh regression head ──────────
        ckpt = torch.load(
            self.chemeleon_weights,
            weights_only=True,
            map_location=torch.device(self.device),
        )
        mp = cpnn.BondMessagePassing(**ckpt["hyper_parameters"])
        mp.load_state_dict(ckpt["state_dict"])
        agg = cpnn.MeanAggregation()

        print("loaded CheMeleon backbone...", flush=True)

        output_transform = cpnn.UnscaleTransform.from_standard_scaler(scaler)
        ffn = cpnn.RegressionFFN(
            output_transform=output_transform,
            input_dim=mp.output_dim,  # 2048 — must match CheMeleon
            hidden_dim=ffn_hidden,
            n_layers=ffn_layers,
            dropout=dropout,
        )
        mpnn = cpmodels.MPNN(
            mp, agg, ffn, batch_norm=False, metrics=[cpnn.metrics.RMSE()], max_lr=max_lr
        )

        # ── Lightning training with early stopping ─────────────────────────
        early = EarlyStopping(monitor="val_loss", patience=self.patience, mode="min")
        tmpdir = tempfile.mkdtemp()
        ckpt_cb = ModelCheckpoint(
            dirpath=tmpdir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best",
        )
        trainer = pl.Trainer(
            accelerator=self.device,
            devices=1,
            max_epochs=self.max_epochs,
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=False,
            callbacks=[early, ckpt_cb],
        )
        print("starting trainer.fit...", flush=True)
        trainer.fit(mpnn, train_loader, val_loader)

        # restore the BEST-val weights (EarlyStopping alone leaves the last epoch)
        if ckpt_cb.best_model_path:
            mpnn = cpmodels.MPNN.load_from_checkpoint(ckpt_cb.best_model_path)

        self.model_ = mpnn
        self.trainer_ = trainer
        return self

    def predict(self, smiles_list, X_fp=None, X_desc=None):
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        dset = cpdata.MoleculeDataset(self._datapoints(smiles_list), featurizer)
        loader = cpdata.build_dataloader(dset, num_workers=0, shuffle=False)
        preds = self.trainer_.predict(self.model_, loader)  # list of [batch, n_targets]
        return torch.cat(preds).numpy()  # already unscaled

    def score(self, smiles_list, y, X_fp=None, X_desc=None):
        y_pred = self.predict(smiles_list, X_fp, X_desc)
        return float(
            np.mean([r2_score(y[:, i], y_pred[:, i]) for i in range(y.shape[1])])
        )

    def save(self, path):
        if not hasattr(self, "trainer_"):
            raise ValueError("Cannot save an unfitted model. Call fit() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.trainer_.save_checkpoint(f"{path}.ckpt")

    @classmethod
    def load(cls, path, device="cpu"):
        instance = cls(device=device)
        instance.model_ = cpmodels.MPNN.load_from_checkpoint(
            f"{path}.ckpt", map_location=torch.device(device)
        )
        instance.trainer_ = pl.Trainer(
            accelerator=device, devices=1, logger=False, enable_progress_bar=False
        )
        return instance

    @classmethod
    def from_hf(
        cls, hf_path: str, model_id: str, device: str = "cpu"
    ) -> "CheMeleonRegressor":
        """Load a model from HuggingFace Hub.

        Args:
            hf_path (str): Path to the HuggingFace Hub repository.
            model_id (str): Model ID on HF Hub, e.g. "model_name".
            device (str): Compute device to load the checkpoint onto.

        Returns:
            Restored CheMeleonRegressor instance.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            hf.snapshot_download(hf_path, local_dir=tmpdir)
            path = Path(tmpdir) / model_id
            return cls.load(str(path), device=device)
