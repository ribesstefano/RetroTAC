"""
config.py
=========
Typed loader for config/models_config.yaml, the single schema shared by
scripts/models/train.py and scripts/analysis/models_evaluation.py so both
read the same keys instead of each re-extracting them from a raw dict by hand.

Only fields actually consumed by those two scripts are exposed here. A few
keys present in the YAML are read by no code and are intentionally left off
this schema: features.tune_fp_radius, tracking.wandb_project,
tracking.study_prefix (run naming actually comes from train.py's --prefix
CLI flag), and torch.device (inference/training device actually comes from
--device instead).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import yaml


@dataclass(frozen=True)
class FeaturesConfig:
    use_fingerprints: bool
    use_descriptors: bool
    fp_radius: int
    fp_size: int


@dataclass(frozen=True)
class CrossValidationConfig:
    seeds: List[int]
    n_folds: int


@dataclass(frozen=True)
class TorchConfig:
    max_epochs: int
    patience: int
    batch_size: int


@dataclass(frozen=True)
class GNNConfig:
    chemeleon_weights: str


@dataclass(frozen=True)
class ModelsConfig:
    """Everything scripts/models/train.py / scripts/analysis/models_evaluation.py
    read out of config/models_config.yaml."""
    target: str
    molecule_col: str
    features: FeaturesConfig
    cross_validation: CrossValidationConfig
    torch: TorchConfig
    gnn: GNNConfig

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelsConfig":
        """Parse a models_config.yaml file into a typed, immutable config.

        Args:
            path: Path to the YAML config file.

        Returns:
            A populated ModelsConfig.

        Raises:
            KeyError: If a key this schema requires is missing from the YAML.
        """
        with open(path) as f:
            raw = yaml.safe_load(f)

        feat = raw["features"]
        cv = raw["cross_validation"]
        torch_cfg = raw["torch"]
        gnn_cfg = raw.get("gnn", {})

        return cls(
            target=raw["target"],
            molecule_col=raw["molecule_col"],
            features=FeaturesConfig(
                use_fingerprints=feat["use_fingerprints"],
                use_descriptors=feat["use_descriptors"],
                fp_radius=feat["fp_radius"],
                fp_size=feat["fp_size"],
            ),
            cross_validation=CrossValidationConfig(
                seeds=cv["seeds"],
                n_folds=cv["n_folds"],
            ),
            torch=TorchConfig(
                max_epochs=torch_cfg["max_epochs"],
                patience=torch_cfg["patience"],
                batch_size=torch_cfg["batch_size"],
            ),
            gnn=GNNConfig(
                chemeleon_weights=gnn_cfg.get("chemeleon_weights", "chemeleon_mp.pt"),
            ),
        )
