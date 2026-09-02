"""Build a RetroTAC Caruana-ensemble repo layout and push it to the Hugging
Face Hub (or just stage it locally). Reads a Caruana ensemble-selection
weights file (scripts/models/evaluation.py's save_ensemble_weights output,
e.g. ensemble_weights_caruana.json), resolves each weighted member back to
its per-fold model files under --cv-dir (outputs/cv/'s
{backend}_<run>/model_seed{seed}_fold{fold}.* convention), and stages:

    ensemble.json   -- manifest: weight + backend/seed/fold + path per member
    config.json     -- feature config (fp_size/fp_radius/use_*) to reproduce inputs
    README.md       -- auto-generated summary + usage snippet
    members/        -- the member model files themselves

This is exactly the layout retrotac.models.ensemble.RetroTAC.from_pretrained
expects, whether given a local directory or a Hub repo id.

Usage
-----
    # Push straight to the Hub (HF_TOKEN read from .env):
    python scripts/models/push_to_hf.py \\
        --weights outputs/results/results_20260828_182305/ensemble_weights_caruana.json \\
        --cv-dir outputs/cv --repo-id ribesstefano/retrotac

    # Stage locally only, no network -- e.g. to sanity check with
    # RetroTAC.from_pretrained(local_dir) before actually pushing:
    python scripts/models/push_to_hf.py \\
        --weights outputs/results/results_20260828_182305/ensemble_weights_caruana.json \\
        --cv-dir outputs/cv --local-dir /tmp/retrotac_staged --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from retrotac.models.config import ModelsConfig
from retrotac.models.loading import parse_fold_model_name

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "models_config.yaml"

MANIFEST_FILENAME = "ensemble.json"
CONFIG_FILENAME = "config.json"

# Extensions each backend's save(path) writes alongside a base path (no ext).
_EXTENSIONS = {
    "xgb": [".skops", ".ubj"],
    "mlp": [".skops", ".pt"],
    "gnn": [".ckpt"],
}


def resolve_members(weights: Dict[str, float], cv_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Resolve each weighted member name to its source files under cv_dir.

    Args:
        weights: model name -> raw weight (as read from the weights JSON;
            need not already sum to 1).
        cv_dir: Directory holding one `{backend}_<run>` subdirectory per
            backend (train.py's cv/ output root).

    Returns:
        model name -> {"backend", "seed", "fold", "weight" (normalized to
        sum to 1), "src_base" (Path, no extension), "rel_path" (str, the
        member's path inside the staged repo, also no extension)}.

    Raises:
        ValueError: If `weights` is empty or sums to <= 0.
        FileNotFoundError: If any backend directory can't be resolved
            unambiguously under cv_dir, or any member's file(s) are missing
            -- every such problem is collected and reported together, not
            one at a time.
    """
    if not weights:
        raise ValueError("weights is empty -- nothing to resolve.")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"weights sum to {total} (<= 0); cannot normalize.")

    backend_dirs: Dict[str, Path] = {}
    resolved: Dict[str, Dict[str, Any]] = {}
    dir_errors = []
    missing_files = []

    for name, weight in weights.items():
        backend, seed, fold = parse_fold_model_name(name)

        if backend not in backend_dirs:
            candidates = sorted(p for p in cv_dir.glob(f"{backend}_*") if p.is_dir())
            if len(candidates) != 1:
                dir_errors.append(
                    f"  expected exactly one '{backend}_*' directory under {cv_dir}, "
                    f"found {len(candidates)}: {[str(c) for c in candidates]}"
                )
                continue
            backend_dirs[backend] = candidates[0]

        base = backend_dirs[backend] / f"model_seed{seed}_fold{fold}"
        for ext in _EXTENSIONS[backend]:
            if not base.with_suffix(ext).exists():
                missing_files.append(str(base.with_suffix(ext)))

        resolved[name] = {
            "backend": backend,
            "seed": seed,
            "fold": fold,
            "weight": weight / total,
            "src_base": base,
            "rel_path": f"members/{backend}_seed{seed}_fold{fold}",
        }

    if dir_errors:
        raise FileNotFoundError("Cannot resolve backend directories:\n" + "\n".join(dir_errors))
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} member file(s) missing under {cv_dir}:\n"
            + "\n".join(f"  {m}" for m in missing_files)
        )
    return resolved


def build_manifest(weights_doc: Dict[str, Any], resolved: Dict[str, Dict[str, Any]], weights_path: Path) -> Dict[str, Any]:
    """Assemble the ensemble.json manifest RetroTAC.from_pretrained reads.

    Args:
        weights_doc: Parsed weights JSON (method/metric_name/metric_value/metadata).
        resolved: Output of `resolve_members`.
        weights_path: Path the weights JSON was read from (recorded for provenance).

    Returns:
        The manifest dict, ready to json.dump.
    """
    members = {
        name: {
            "backend": r["backend"],
            "seed": r["seed"],
            "fold": r["fold"],
            "weight": r["weight"],
            "path": r["rel_path"],
        }
        for name, r in resolved.items()
    }
    manifest: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "source_weights_file": weights_path.name,
        "method": weights_doc.get("method", "caruana"),
        "metric_name": weights_doc.get("metric_name"),
        "metric_value": weights_doc.get("metric_value"),
        "n_models": len(members),
        "members": members,
    }
    if "metadata" in weights_doc:
        manifest["metadata"] = weights_doc["metadata"]
    return manifest


def build_feature_config(cfg: ModelsConfig, config_path: Path) -> Dict[str, Any]:
    """Assemble the config.json feature config RetroTAC.predict reads.

    Args:
        cfg: The ModelsConfig the fold models were trained with.
        config_path: Path it was loaded from (recorded for provenance).

    Returns:
        Dict with the fields retrotac.models.loading.compute_features needs,
        plus target/molecule_col/source_config for human provenance only.
    """
    return {
        "fp_size": cfg.features.fp_size,
        "fp_radius": cfg.features.fp_radius,
        "use_fingerprints": cfg.features.use_fingerprints,
        "use_descriptors": cfg.features.use_descriptors,
        "target": cfg.target,
        "molecule_col": cfg.molecule_col,
        "source_config": str(config_path),
    }


def build_readme(manifest: Dict[str, Any], repo_id: Optional[str]) -> str:
    """Render a short auto-generated README.md for the staged/pushed repo."""
    counts = Counter(m["backend"].upper() for m in manifest["members"].values())
    composition = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
    metric_value = manifest.get("metric_value")
    metric_str = f"{metric_value:.4f}" if isinstance(metric_value, (int, float)) else str(metric_value)
    load_target = repo_id or "<repo_id_or_local_dir>"
    return f"""# {repo_id or "RetroTAC ensemble"}

Caruana-weighted ensemble of {manifest["n_models"]} per-fold PROTAC
synthesizability surrogate models ({composition}), selected by greedy
forward ensemble selection over 5x5-CV fold models -- see
`scripts/models/evaluation.py`'s `EnsembleSelector.greedy_selection` /
`run_ensemble_strategies` in the [RetroTAC](https://github.com/ribesstefano/RetroTAC) repo.

- method: `{manifest["method"]}`
- {manifest["metric_name"]}: {metric_str}

## Usage

```python
from retrotac.models.ensemble import RetroTAC

model = RetroTAC.from_pretrained("{load_target}")
predictions = model.predict(["CCO", "c1ccccc1"])
```
"""


def _link_or_copy(src: Path, dst: Path) -> None:
    """Stage one file at dst, symlinking to src (falls back to a copy)."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_repo(
    staging_dir: Path,
    manifest: Dict[str, Any],
    feature_config: Dict[str, Any],
    readme: str,
    resolved: Dict[str, Dict[str, Any]],
) -> None:
    """Materialize the repo layout at staging_dir (members via symlink).

    Args:
        staging_dir: Destination directory (created if missing).
        manifest: Output of `build_manifest`.
        feature_config: Output of `build_feature_config`.
        readme: Output of `build_readme`.
        resolved: Output of `resolve_members` (gives each member's real
            source files to link/copy into staging_dir/members/).
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    members_dir = staging_dir / "members"
    members_dir.mkdir(exist_ok=True)

    (staging_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
    (staging_dir / CONFIG_FILENAME).write_text(json.dumps(feature_config, indent=2))
    (staging_dir / "README.md").write_text(readme)

    for name, r in resolved.items():
        base_name = Path(manifest["members"][name]["path"]).name
        for ext in _EXTENSIONS[r["backend"]]:
            _link_or_copy(r["src_base"].with_suffix(ext), members_dir / f"{base_name}{ext}")


def push(staging_dir: Path, repo_id: str, private: bool, token: Optional[str], commit_message: str) -> None:
    """Create (if needed) and push staging_dir's contents to a Hub repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )


def _print_summary(manifest: Dict[str, Any], staging_dir: Path) -> None:
    counts = Counter(m["backend"].upper() for m in manifest["members"].values())
    composition = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
    total_bytes = sum(p.stat().st_size for p in (staging_dir / "members").glob("*") if p.is_file())
    metric_value = manifest.get("metric_value")
    metric_str = f"{metric_value:.4f}" if isinstance(metric_value, (int, float)) else str(metric_value)
    print(f"  method={manifest['method']} {manifest['metric_name']}={metric_str} "
          f"n_models={manifest['n_models']} ({composition})")
    print(f"  staged at {staging_dir} ({total_bytes / 1e9:.2f} GB)")


def parse_args() -> argparse.Namespace:
    """Parse CLI args for staging/pushing a RetroTAC ensemble repo."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True, type=Path,
                     help="ensemble weights JSON, e.g. ensemble_weights_caruana.json")
    ap.add_argument("--cv-dir", required=True, type=Path,
                     help="outputs/cv/ -- one {backend}_<run> subdirectory per backend")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                     help="models_config.yaml the runs were trained with (default: %(default)s)")
    ap.add_argument("--repo-id", default=None,
                     help="Hugging Face Hub repo id to push to, e.g. ribesstefano/retrotac")
    ap.add_argument("--local-dir", type=Path, default=None,
                     help="also (or only, without --repo-id) stage the repo layout here")
    ap.add_argument("--private", action="store_true", help="create the Hub repo as private")
    ap.add_argument("--dry-run", action="store_true",
                     help="stage and validate but do not push to the Hub")
    ap.add_argument("--commit-message", default="Push RetroTAC ensemble",
                     help="Hub commit message (default: %(default)r)")
    ap.add_argument("--token", default=None,
                     help="Hugging Face token; defaults to HF_TOKEN from .env / the environment")
    args = ap.parse_args()
    if not args.repo_id and not args.local_dir:
        ap.error("at least one of --repo-id or --local-dir is required")
    return args


def main() -> None:
    """Parse CLI args and stage/push a RetroTAC ensemble repo end-to-end."""
    args = parse_args()

    load_dotenv(_PROJECT_ROOT / ".env")
    token = args.token or os.getenv("HF_TOKEN")
    will_push = bool(args.repo_id) and not args.dry_run
    if will_push and not token:
        raise SystemExit(
            "HF_TOKEN not set (checked --token and HF_TOKEN in .env/the environment) -- "
            "required to push to the Hub. Pass --dry-run or --local-dir to stage without pushing."
        )

    weights_doc = json.loads(args.weights.read_text())
    weights = {k: float(v) for k, v in weights_doc["weights"].items()}

    resolved = resolve_members(weights, args.cv_dir)
    manifest = build_manifest(weights_doc, resolved, args.weights)

    cfg = ModelsConfig.load(args.config)
    feature_config = build_feature_config(cfg, args.config)
    readme = build_readme(manifest, args.repo_id)

    def _finish(staging_dir: Path) -> None:
        stage_repo(staging_dir, manifest, feature_config, readme, resolved)
        _print_summary(manifest, staging_dir)
        if not args.repo_id:
            return
        if args.dry_run:
            print("  (--dry-run: not pushing)")
            return
        push(staging_dir, args.repo_id, args.private, token, args.commit_message)
        print(f"  pushed -> https://huggingface.co/{args.repo_id}")

    if args.local_dir:
        _finish(args.local_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="retrotac_push_") as tmp:
            _finish(Path(tmp))


if __name__ == "__main__":
    main()
