"""
ensemble.py
===========
Caruana-weighted ensemble over per-fold xgb/mlp/gnn surrogate models (see
scripts/models/evaluation.py's EnsembleSelector.greedy_selection /
run_ensemble_strategies). Loads member models from a local directory or a
Hugging Face Hub repo laid out by scripts/models/push_to_hf.py
(ensemble.json + config.json + members/), featurizes each input SMILES list
once via retrotac.models.loading.compute_features, and returns the
Caruana-weighted mean -- the same prediction the "caruana" row in
ensemble_strategies.csv reports.
"""
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import huggingface_hub as hf
import numpy as np
from tqdm import tqdm

from retrotac.models.loading import compute_features, load_backend

logger = logging.getLogger(__name__)

# Silence PyTorch Lightning's rank_zero_info banners ("GPU available: ...",
# litlogger/litmodels upsell tips) and its "does not have many workers"
# PossibleUserWarning plus torch's internal LeafSpec deprecation warning --
# both fire once per GNN ensemble member loaded/predicted (see gnn/model.py's
# CheMeleonRegressor.load/predict) and are pure noise for inference. Import
# lightning.pytorch here (retrotac.models already pulls in torch/lightning
# eagerly regardless of backend, see retrotac/models/__init__.py) so its
# own import-time logger setup runs before -- not after -- we override the
# level, rather than relying on retrotac.models.loading's lazy per-backend
# imports to have already run it.
import lightning.pytorch  # noqa: F401,E402

for _lightning_logger in ("lightning", "lightning.pytorch", "lightning.fabric", "pytorch_lightning"):
    logging.getLogger(_lightning_logger).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r".*does not have many workers.*")
warnings.filterwarnings("ignore", message=r".*LeafSpec.*")

MANIFEST_FILENAME = "ensemble.json"
CONFIG_FILENAME = "config.json"

_LOAD_STRATEGIES = ("eager", "lazy", "stream")


class RetroTAC:
    """Caruana ensemble of per-fold xgb/mlp/gnn PROTAC-synthesizability models.

    Construct via `from_pretrained` rather than directly -- it resolves and
    validates the manifest/config this class needs from a local directory or
    Hub repo. See `from_pretrained` and `predict`.
    """

    def __init__(
        self,
        member_specs: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
        feature_config: Dict[str, Any],
        model_dir: Path,
        load_strategy: str = "eager",
        max_resident: Optional[int] = None,
        device: str = "cpu",
    ):
        """Store resolved member/weight/feature info; eager-load if requested.

        Args:
            member_specs: model name -> {"backend", "seed", "fold", "path"},
                as read from the manifest's "members" block.
            weights: model name -> normalized weight (sums to 1); same key
                set as `member_specs`.
            feature_config: fp_size/fp_radius/use_fingerprints/use_descriptors,
                as read from config.json.
            model_dir: Local directory holding the manifest, config, and
                `members/` (a Hub snapshot dir when loaded via a repo id).
            load_strategy: "eager" loads all members immediately and keeps
                them resident; "lazy" loads a member on first use and caches
                it (bounded by `max_resident`, evicting least-recently-used);
                "stream" never caches -- every predict() call reloads every
                member from disk. See `from_pretrained`.
            max_resident: For load_strategy="lazy" only: max number of
                member models kept in memory at once (None = unbounded, i.e.
                keep every member ever loaded).
            device: Compute device forwarded to gnn members.

        Raises:
            ValueError: If load_strategy isn't one of eager/lazy/stream.
        """
        if load_strategy not in _LOAD_STRATEGIES:
            raise ValueError(
                f"Unknown load_strategy: {load_strategy!r} (expected one of {_LOAD_STRATEGIES})"
            )
        self._member_specs = member_specs
        self._weights = weights
        self._feature_config = feature_config
        self._model_dir = Path(model_dir)
        self._load_strategy = load_strategy
        self._max_resident = max_resident
        self._device = device

        self._cache: Dict[str, Any] = {}
        self._lru: List[str] = []

        if load_strategy == "eager":
            for name in tqdm(self._member_specs, desc="Loading ensemble members", unit="model"):
                self._cache[name] = self._load_member(name)

    def _load_member(self, name: str) -> Any:
        """Instantiate one member model from its manifest spec (no caching)."""
        spec = self._member_specs[name]
        base_path = str(self._model_dir / spec["path"])
        return load_backend(spec["backend"], base_path, device=self._device)

    def _get_member(self, name: str) -> Any:
        """Return one member model per this instance's load_strategy."""
        if self._load_strategy == "stream":
            return self._load_member(name)

        if name in self._cache:
            if self._load_strategy == "lazy":
                self._lru.remove(name)
                self._lru.append(name)
            return self._cache[name]

        model = self._load_member(name)
        if self._load_strategy == "lazy":
            self._cache[name] = model
            self._lru.append(name)
            if self._max_resident is not None:
                while len(self._lru) > self._max_resident:
                    evict = self._lru.pop(0)
                    del self._cache[evict]
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        load_strategy: str = "eager",
        max_resident: Optional[int] = None,
        device: str = "cpu",
    ) -> "RetroTAC":
        """Load the Caruana ensemble from a local directory or the Hub.

        Mirrors the resolution HuggingFace's own `from_pretrained` uses: a
        path that exists locally is used as-is; anything else is treated as
        a Hub repo id and downloaded. The repo (see scripts/models/
        push_to_hf.py) holds exactly the members the ensemble needs, so the
        whole snapshot is fetched either way -- `load_strategy` only governs
        how many of those members are kept as loaded model *objects* in
        memory, not what's downloaded to disk.

        Args:
            pretrained_model_name_or_path: Local directory (containing
                ensemble.json/config.json/members/), or a Hub repo id, e.g.
                "ribesstefano/retrotac".
            revision: Hub branch/tag/commit; ignored for a local path.
            cache_dir: Hub download cache dir; ignored for a local path.
            token: Hub auth token for private repos; ignored for a local
                path. Falls back to huggingface_hub's normal resolution
                (HF_TOKEN env var, cached `huggingface-cli login`, ...) when
                omitted -- see scripts/models/push_to_hf.py for how HF_TOKEN
                is read from .env when *pushing* a repo.
            load_strategy: "eager" (default, load all members now and keep
                them resident), "lazy" (load on first use, cache up to
                `max_resident`), or "stream" (load/predict/discard every call).
            max_resident: For load_strategy="lazy" only; None = unbounded.
            device: Compute device forwarded to gnn members.

        Returns:
            A ready-to-predict RetroTAC instance.

        Raises:
            FileNotFoundError: If the resolved directory is missing
                ensemble.json or config.json.
        """
        local_dir = Path(pretrained_model_name_or_path)
        if local_dir.is_dir():
            model_dir = local_dir
            logger.info("Loading RetroTAC ensemble from local directory: %s", model_dir)
        else:
            logger.info("Downloading RetroTAC ensemble from Hub repo: %s", pretrained_model_name_or_path)
            snapshot = hf.snapshot_download(
                repo_id=pretrained_model_name_or_path,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )
            model_dir = Path(snapshot)

        manifest_path = model_dir / MANIFEST_FILENAME
        config_path = model_dir / CONFIG_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing {MANIFEST_FILENAME} under {model_dir}")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing {CONFIG_FILENAME} under {model_dir}")

        manifest = json.loads(manifest_path.read_text())
        feature_config = json.loads(config_path.read_text())

        member_specs = manifest["members"]
        raw_weights = {name: float(spec["weight"]) for name, spec in member_specs.items()}
        total = sum(raw_weights.values())
        if total <= 0:
            raise ValueError("Ensemble weights sum to <= 0; cannot normalize.")
        weights = {name: w / total for name, w in raw_weights.items()}

        return cls(
            member_specs=member_specs,
            weights=weights,
            feature_config=feature_config,
            model_dir=model_dir,
            load_strategy=load_strategy,
            max_resident=max_resident,
            device=device,
        )

    def predict(
        self, smiles_list: List[str], return_details: bool = False
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """Predict synthesizability as the Caruana-weighted mean over members.

        Features (fingerprints/descriptors) are computed once and fanned out
        to every member (mirrors scripts/models/evaluation.py's
        predict_fold_models); gnn members ignore the fingerprint/descriptor
        arguments and featurize SMILES internally.

        Args:
            smiles_list: SMILES strings to score.
            return_details: If True, return a dict with the per-member
                spread and raw predictions instead of just the mean.

        Returns:
            If return_details=False (default): a 1-D np.ndarray of shape
            (len(smiles_list),) -- the weighted mean.

            If return_details=True: a dict with:
                "mean": the same weighted-mean array,
                "std": 1-D np.ndarray, the UNWEIGHTED std across the
                    distinct member models' raw predictions (spread across
                    base learners, not weighted by how often Caruana picked
                    one -- matches evaluation.py's _ensemble_uncertainty_
                    metrics / the "mean_std" column in ensemble_strategies.csv),
                "members": {model name: 1-D np.ndarray}, every member's raw
                    prediction.
        """
        X_fp, X_desc = compute_features(
            smiles_list,
            self._feature_config["fp_size"],
            self._feature_config["fp_radius"],
            self._feature_config.get("use_fingerprints", True),
            self._feature_config.get("use_descriptors", True),
        )

        member_preds: Dict[str, np.ndarray] = {}
        mean = np.zeros(len(smiles_list), dtype=float)
        for name, weight in tqdm(self._weights.items(), desc="Ensemble members", unit="model"):
            model = self._get_member(name)
            y_pred = model.predict(smiles_list, X_fp=X_fp, X_desc=X_desc)
            y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
            member_preds[name] = y_pred
            mean += weight * y_pred

        if not return_details:
            return mean

        stacked = np.stack(list(member_preds.values()), axis=0)  # [n_members, n_smiles]
        std = stacked.std(axis=0) if stacked.shape[0] >= 2 else np.zeros_like(mean)
        return {"mean": mean, "std": std, "members": member_preds}
