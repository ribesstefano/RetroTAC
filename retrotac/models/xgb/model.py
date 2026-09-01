"""
xgb/model.py
============
Sklearn-style XGBoost regressor wrapping fingerprint/descriptor featurization,
preprocessing, and target scaling behind a fit/predict/score/save/load
contract shared with the mlp/gnn backends (see training.py).
"""

import tempfile
from pathlib import Path
from typing import List, Optional

import huggingface_hub as hf
import numpy as np
import skops.io as sio
import xgboost as xgb
from rdkit import RDLogger
from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer

from retrotac.chem_utils import (  # noqa: E402
    compute_fingerprints,
    make_preprocessor,
    sanitize_matrix,
    standardize_all,
)

RDLogger.DisableLog("rdApp.*")


class XGBoostRegressor:
    """Sklearn-style wrapper around XGBoost with fp/descriptor featurization.

    Handles feature assembly (Morgan fingerprints + RDKit descriptors),
    preprocessing (SVD/imputation/scaling via make_preprocessor), and target
    scaling (QuantileTransformer) around a plain xgboost.XGBRegressor.
    """

    def __init__(
        self,
        fp_size: int = 512,
        fp_radius: int = 2,
        svd_components: int = 64,
        use_fingerprints: bool = True,
        use_descriptors: bool = True,
        xgb_params: Optional[dict] = None,
        uncharge: bool = False,
        random_state: int = 42,
        device: str = "cpu",
    ):
        """Store config; the XGBRegressor itself is built lazily in fit().

        Args:
            fp_size: Morgan fingerprint bit-vector length.
            fp_radius: Morgan fingerprint bond-hop radius.
            svd_components: TruncatedSVD output dims for the fingerprint
                block passed to make_preprocessor; 0 disables SVD.
            use_fingerprints: Whether the feature matrix includes fingerprints.
            use_descriptors: Whether the feature matrix includes RDKit descriptors.
            xgb_params: Extra/overriding kwargs merged into the default
                XGBRegressor params at fit time.
            uncharge: Unused by this backend; kept for config parity.
            random_state: Seed for SVD and the target QuantileTransformer.
            device: XGBoost compute device ("cpu" or "cuda").

        Raises:
            ValueError: If both use_fingerprints and use_descriptors are False.
        """
        if not use_fingerprints and not use_descriptors:
            raise ValueError(
                "At least one of use_fingerprints or use_descriptors must be True."
            )
        self.fp_size = fp_size
        self.fp_radius = fp_radius
        self.svd_components = svd_components
        self.use_fingerprints = use_fingerprints
        self.use_descriptors = use_descriptors
        self.xgb_params = xgb_params or {}
        self.uncharge = uncharge
        self.random_state = random_state
        self.device = device

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
            if X_fp is None:
                X_fp = compute_fingerprints(
                    standardize_all(smiles_list), self.fp_size, self.fp_radius
                )
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
    ) -> "XGBoostRegressor":
        """Fit XGBoost to featurized SMILES.

        Args:
            smiles_list: Training SMILES strings.
            y: Training targets, shape [n_samples, n_targets].
            X_fp: Pre-computed training fingerprints; computed on demand when
                None and use_fingerprints=True.
            X_desc: Pre-computed training descriptors, required when
                use_descriptors=True.
            smiles_val: Validation SMILES for early stopping (optional).
            y_val: Validation targets for early stopping (optional).
            X_fp_val: Pre-computed validation fingerprints (optional).
            X_desc_val: Pre-computed validation descriptors, required when
                use_descriptors=True and smiles_val is provided.

        Returns:
            self
        """
        X = self._featurize(smiles_list, X_fp, X_desc)

        # NOTE: preprocessor and target_transformer are kept separate (not
        # fused into one Pipeline) so that validation data can be transformed
        # independently for XGBoost's eval_set. A fused pipeline would only
        # transform training data, breaking early stopping.
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
        self.target_transformer_ = QuantileTransformer(
            output_distribution="normal", random_state=self.random_state
        )
        X_proc = self.preprocessor_.fit_transform(X)
        y_transf = self.target_transformer_.fit_transform(y)

        default_xgb = dict(
            tree_method="hist",
            device=self.device,
            objective="reg:pseudohubererror",
            multi_strategy="one_output_per_tree",
            n_estimators=2000,
            early_stopping_rounds=50 if smiles_val is not None else None,
            random_state=self.random_state,
        )
        print("-" * 80)
        print(default_xgb)
        print(self.xgb_params)
        print("-" * 80)
        self.model_ = xgb.XGBRegressor(**{**default_xgb, **self.xgb_params})

        if smiles_val is not None:
            X_val_proc = self.preprocessor_.transform(
                self._featurize(smiles_val, X_fp_val, X_desc_val)
            )
            y_val_t = self.target_transformer_.transform(y_val)
            self.model_.fit(
                X_proc,
                y_transf,
                eval_set=[(X_val_proc, y_val_t)],
                verbose=False,
            )
        else:
            self.model_.fit(X_proc, y_transf)

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
        y_transf = self.model_.predict(X_proc)
        if y_transf.ndim == 1:
            y_transf = y_transf.reshape(-1, 1)
        return self.target_transformer_.inverse_transform(y_transf)

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
        """Save to two files: {path}.skops (sklearn) + {path}.ubj (XGBoost).

        The XGBoost model is saved separately in its native binary format
        because skops cannot reliably serialize it across versions.

        Args:
            path (str): Base path without extension, e.g. 'models/xgb_regressor'.
        """
        if not hasattr(self, "model_"):
            raise ValueError("Cannot save an unfitted model. Call fit() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Temporarily detach XGBoost model so skops only sees sklearn objects.
        xgb_model = self.model_
        self.model_ = None
        sio.dump(self, f"{path}.skops")
        self.model_ = xgb_model

        xgb_model.save_model(f"{path}.ubj")

    @classmethod
    def load(cls, path: str) -> "XGBoostRegressor":
        """Load from {path}.skops and {path}.ubj.

        Args:
            path (str): Base path without extension.

        Returns:
            Restored XGBoostRegressor instance.
        """
        unknown_types = sio.get_untrusted_types(file=f"{path}.skops")
        allowed = ("XGBoostRegressor", "numpy.dtype")
        for t in unknown_types:
            if not any(a in t for a in allowed):
                raise ValueError(
                    f"Untrusted type '{t}' found in skops file. Aborting load for safety."
                )
        instance = sio.load(f"{path}.skops", trusted=unknown_types)
        instance.model_ = xgb.XGBRegressor()
        instance.model_.load_model(f"{path}.ubj")
        return instance

    @classmethod
    def from_hf(cls, hf_path: str, model_id: str) -> "XGBoostRegressor":
        """Load a model from HuggingFace Hub.

        Args:
            hf_path (str): Path to the HuggingFace Hub repository.
            model_id (str): Model ID on HF Hub, e.g. "model_name".

        Returns:
            Restored XGBoostRegressor instance.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            hf.snapshot_download(hf_path, local_dir=tmpdir)
            path = Path(tmpdir) / model_id
            return cls.load(str(path))
