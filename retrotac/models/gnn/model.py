"""
gnn/model.py
============
CheMeleon foundation-model regressor (ChemProp D-MPNN, pretrained on Mordred
descriptors, fine-tuned on our data). Graph-based; ignores X_fp / X_desc.
Matches the fit/predict/score/save/load contract used by train.py.

Two predictor heads are available, selected by `head`:
  * "regression" (default) -- chemprop's RegressionFFN, a plain point estimate.
  * "evidential"           -- chemprop's EvidentialFFN, which predicts the four
    parameters (mean, v, alpha, beta) of a Normal-Inverse-Gamma distribution per
    task and is trained with EvidentialLoss. predict() still returns only the
    mean, so an evidential model is a drop-in for every existing caller;
    predict_uncertainty() exposes the aleatoric/epistemic variance decomposition
    that the extra outputs buy (see Amini et al. 2020 / Soleimany et al. 2021).

The head is recorded inside the saved Lightning checkpoint (chemprop stores the
predictor class in its hparams), so load() reconstructs either head without
being told which -- checkpoints written before this option existed keep loading
as plain RegressionFFN models.

Requires: chemprop>=2.2.0, lightning. Download the CheMeleon weights once:
    urlretrieve("https://zenodo.org/records/15460715/files/chemeleon_mp.pt", "chemeleon_mp.pt")
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import huggingface_hub as hf
import numpy as np
import torch
from chemprop import data as cpdata
from chemprop import featurizers
from chemprop import models as cpmodels
from chemprop import nn as cpnn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)

#: Predictor heads `CheMeleonRegressor` can put on the CheMeleon backbone.
HEADS = ("regression", "evidential")


class CheMeleonRegressor:
    """Sklearn-style wrapper around a CheMeleon D-MPNN fine-tuned for regression.

    Graph-based: featurizes molecules directly from SMILES via ChemProp, so
    the X_fp/X_desc arguments accepted by fit/predict/score exist only to
    match the tabular models' call signature and are otherwise ignored.
    """

    def __init__(
        self,
        gnn_params: Optional[Dict[str, Any]] = None,
        chemeleon_weights: str = "chemeleon_mp.pt",
        head: str = "regression",
        max_epochs: int = 100,
        patience: int = 15,
        random_state: int = 42,
        device: str = "cpu",
    ):
        """Store config; the network itself is built lazily in fit().

        Args:
            gnn_params: FFN head / optimizer hyperparameters (dropout,
                ffn_hidden_dim, ffn_n_layers, max_lr, batch_size, batch_norm,
                and -- evidential head only -- v_kl); missing keys fall back to
                defaults inside fit().
            chemeleon_weights: Path to the pretrained CheMeleon backbone
                checkpoint (BondMessagePassing state dict).
            head: Predictor head, one of HEADS: "regression" (default, a point
                estimate) or "evidential" (a Normal-Inverse-Gamma head whose
                extra outputs give per-molecule uncertainty, see
                `predict_uncertainty`). Ignored by `load`, which reads the head
                back out of the checkpoint instead.
            max_epochs: Maximum training epochs.
            patience: Early-stopping patience in epochs.
            random_state: Seed for Lightning's global RNG and the fallback
                train/val split.
            device: Compute device for the Lightning trainer ("cpu" or GPU).

        Raises:
            ValueError: If `head` isn't one of HEADS.
        """
        if head not in HEADS:
            raise ValueError(f"Unknown head: {head!r} (expected one of {HEADS})")
        self.gnn_params = gnn_params or {}
        self.chemeleon_weights = chemeleon_weights
        self.head = head
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.device = device

    @property
    def is_evidential(self) -> bool:
        """Whether this model's head emits NIG parameters instead of a point estimate.

        Read off the fitted/loaded network when there is one (chemprop records
        the predictor class in the checkpoint, so this is also correct for a
        model restored by `load`), and off the requested `head` otherwise.
        """
        model = getattr(self, "model_", None)
        if model is not None:
            return getattr(model.predictor, "n_targets", 1) > 1
        return self.head == "evidential"

    def _datapoints(
        self, smiles_list: List[str], y: Optional[np.ndarray] = None
    ) -> List[Any]:
        """Wrap SMILES (and optional targets) into ChemProp MoleculeDatapoints."""
        if y is None:
            return [cpdata.MoleculeDatapoint.from_smi(s) for s in smiles_list]
        return [
            cpdata.MoleculeDatapoint.from_smi(s, yi) for s, yi in zip(smiles_list, y)
        ]

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
        trial: Optional[Any] = None,
    ) -> "CheMeleonRegressor":
        """Fine-tune the CheMeleon backbone + a fresh head of the configured type.

        The head is whichever `head` the instance was built with -- a plain
        RegressionFFN, or an EvidentialFFN trained against chemprop's
        EvidentialLoss instead of MSE. Only the loss and the head's output width
        change; the data pipeline, target scaling and early stopping are shared.

        Args:
            smiles_list: Training SMILES strings.
            y: Training targets, shape [n_samples, n_targets].
            X_fp: Ignored; accepted for signature parity with tabular models.
            X_desc: Ignored; accepted for signature parity with tabular models.
            smiles_val: Validation SMILES for early stopping; when omitted a
                10% split is carved from the training data instead.
            y_val: Validation targets.
            X_fp_val: Ignored; accepted for signature parity with tabular models.
            X_desc_val: Ignored; accepted for signature parity with tabular models.
            trial: Unused (no pruning support here, unlike the MLP backend);
                accepted for a uniform build_fn call signature.

        Returns:
            self
        """
        pl.seed_everything(self.random_state, workers=True)

        p = self.gnn_params
        dropout = p.get("dropout", 0.1)
        ffn_hidden = p.get("ffn_hidden_dim", 300)
        ffn_layers = p.get("ffn_n_layers", 1)
        max_lr = p.get("max_lr", 1e-3)
        batch_size = p.get("batch_size", 64)
        # BatchNorm over the 2048-d CheMeleon fingerprint before the head. Off
        # by default, which is both chemprop's own default and what every model
        # trained before this key existed used, so an absent key reproduces the
        # old architecture exactly.
        batch_norm = p.get("batch_norm", False)

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        # ── datasets (ChemProp scales targets internally) ──────────────────
        train_dset = cpdata.MoleculeDataset(
            self._datapoints(smiles_list, y), featurizer
        )
        scaler = train_dset.normalize_targets()
        logger.debug("built datasets...")

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

        logger.debug("loaded CheMeleon backbone...")

        output_transform = cpnn.UnscaleTransform.from_standard_scaler(scaler)
        ffn_kwargs = dict(
            output_transform=output_transform,
            input_dim=mp.output_dim,  # 2048 — must match CheMeleon
            hidden_dim=ffn_hidden,
            n_layers=ffn_layers,
            dropout=dropout,
        )
        if self.head == "evidential":
            # 4 outputs per task (mean, v, alpha, beta) trained with chemprop's
            # EvidentialLoss; v_kl weights its evidence regularizer against the
            # NIG negative log-likelihood (chemprop's own default is 0.2).
            ffn = cpnn.EvidentialFFN(
                criterion=cpnn.metrics.EvidentialLoss(v_kl=p.get("v_kl", 0.2)),
                **ffn_kwargs,
            )
        else:
            ffn = cpnn.RegressionFFN(**ffn_kwargs)
        # RMSE is reported on the mean for either head (chemprop slices the NIG
        # parameters down to their mean before the non-loss metrics); val_loss,
        # what early stopping/checkpointing watch, stays the head's own
        # criterion -- MSE for regression, the NIG loss for evidential.
        mpnn = cpmodels.MPNN(
            mp, agg, ffn, batch_norm=batch_norm,
            metrics=[cpnn.metrics.RMSE()], max_lr=max_lr,
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
        logger.debug("starting trainer.fit...")
        trainer.fit(mpnn, train_loader, val_loader)

        # restore the BEST-val weights (EarlyStopping alone leaves the last epoch)
        if ckpt_cb.best_model_path:
            mpnn = cpmodels.MPNN.load_from_checkpoint(ckpt_cb.best_model_path)

        self.model_ = mpnn
        self.trainer_ = trainer
        return self

    def _raw_predict(self, smiles_list: List[str], batch_size: int = 64) -> np.ndarray:
        """Run the network over SMILES and return its unreduced output.

        Args:
            smiles_list: SMILES strings to predict on.
            batch_size: Molecules per forward pass (see `predict`).

        Returns:
            [n_samples, n_targets] for the regression head, or
            [n_samples, n_targets, 4] — the NIG (mean, v, alpha, beta) — for
            the evidential head. Already in real (unscaled) units.
        """
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        dset = cpdata.MoleculeDataset(self._datapoints(smiles_list), featurizer)
        loader = cpdata.build_dataloader(dset, batch_size=batch_size, num_workers=0, shuffle=False)
        preds = self.trainer_.predict(self.model_, loader)  # list of [batch, n_targets(, 4)]
        return torch.cat(preds).numpy()  # already unscaled

    def predict(
        self,
        smiles_list: List[str],
        X_fp: Optional[np.ndarray] = None,
        X_desc: Optional[np.ndarray] = None,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Predict targets for SMILES not seen during fit().

        Args:
            smiles_list: SMILES strings to predict on.
            X_fp: Ignored; accepted for signature parity with tabular models.
            X_desc: Ignored; accepted for signature parity with tabular models.
            batch_size: Molecules per GPU forward pass, forwarded to
                chemprop's build_dataloader. Previously hardcoded to
                chemprop's own default (64, still the default here) by
                omission -- entirely decoupled from any outer batching a
                caller does over smiles_list. Raising it (e.g. 256-512;
                watch GPU memory, since D-MPNN batch cost scales with total
                atoms/bonds, not molecule count) is the actual lever for
                GPU utilization during inference.

        Returns:
            Array of shape [len(smiles_list), n_targets] in real (unscaled)
            units — ChemProp's UnscaleTransform undoes the target scaling
            applied during fit() inside the model's forward pass. An
            evidential head returns only the mean of its NIG output here, so
            it stays a drop-in for every caller of the regression head; its
            uncertainty is reached through `predict_uncertainty`.
        """
        raw = self._raw_predict(smiles_list, batch_size=batch_size)
        return raw[..., 0] if self.is_evidential else raw

    def predict_uncertainty(
        self, smiles_list: List[str], batch_size: int = 64
    ) -> Dict[str, np.ndarray]:
        """Predict targets *and* their evidential uncertainty (evidential head only).

        The head parameterises a Normal-Inverse-Gamma per (molecule, task), from
        which chemprop's own estimators read the variance decomposition used
        here: aleatoric (irreducible noise) `beta / (alpha - 1)`, epistemic
        (model ignorance) `beta / (v * (alpha - 1))`, and their sum as the total
        predictive variance. All of them come back in real target units, since
        `EvidentialFFN` pushes beta through the same UnscaleTransform as the mean.

        Args:
            smiles_list: SMILES strings to predict on.
            batch_size: Molecules per forward pass (see `predict`).

        Returns:
            Dict of [n_samples, n_targets] arrays: "mean", "aleatoric_var",
            "epistemic_var", "total_var" and "std" (= sqrt(total_var), the
            per-molecule predictive standard deviation directly comparable to
            an ensemble's spread across members).

        Raises:
            ValueError: If this model wasn't built with the evidential head.
        """
        if not self.is_evidential:
            raise ValueError(
                "predict_uncertainty needs the evidential head; this model has "
                f"head={self.head!r} (a point estimate carries no uncertainty)."
            )
        raw = self._raw_predict(smiles_list, batch_size=batch_size)
        mean, v, alpha, beta = (raw[..., i] for i in range(4))
        aleatoric = beta / (alpha - 1.0)
        epistemic = aleatoric / v
        total = aleatoric + epistemic
        return {
            "mean": mean,
            "aleatoric_var": aleatoric,
            "epistemic_var": epistemic,
            "total_var": total,
            "std": np.sqrt(total),
        }

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
            X_fp: Ignored; accepted for signature parity with tabular models.
            X_desc: Ignored; accepted for signature parity with tabular models.

        Returns:
            Mean R2 across target columns.
        """
        y_pred = self.predict(smiles_list, X_fp, X_desc)
        return float(
            np.mean([r2_score(y[:, i], y_pred[:, i]) for i in range(y.shape[1])])
        )

    def save(self, path: str) -> None:
        """Save the fitted model as a Lightning checkpoint at {path}.ckpt.

        Args:
            path: Base path without extension.
        """
        if not hasattr(self, "trainer_"):
            raise ValueError("Cannot save an unfitted model. Call fit() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.trainer_.save_checkpoint(f"{path}.ckpt")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CheMeleonRegressor":
        """Load a fitted model from {path}.ckpt.

        Args:
            path: Base path without extension.
            device: Compute device to load the checkpoint onto.

        Returns:
            Restored CheMeleonRegressor instance, with `head` set from the
            predictor the checkpoint records (so checkpoints written before the
            evidential head existed come back as "regression" ones).
        """
        instance = cls(device=device)
        instance.model_ = cpmodels.MPNN.load_from_checkpoint(
            f"{path}.ckpt", map_location=torch.device(device)
        )
        instance.head = "evidential" if instance.is_evidential else "regression"
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
