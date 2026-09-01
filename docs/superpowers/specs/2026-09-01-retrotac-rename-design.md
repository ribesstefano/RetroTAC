# Renaming `protac_synth` to `retrotac`

**Date:** 2026-09-01
**Status:** Approved, ready for implementation planning

## Problem

The main package is named `protac_synth`. We want `retrotac`. Two of the five saved
model-artifact types embed the package's import path, so a naive rename breaks them.

## What breaks, and what does not

Verified by reading the bytes of the artifacts in `outputs/` and by loading a skops
file across a simulated rename.

| Artifact | Embeds `protac_synth` | Survives rename |
|---|---|---|
| xgb `*.skops` | Yes: `__module__: "protac_synth.models.xgb.model"` | No |
| mlp `*.skops` | Yes: `__module__: "protac_synth.models.mlp.model"` | No |
| xgb `*.ubj` | No | Yes |
| mlp `*.pt` | No | Yes |
| gnn `*.ckpt` | No | Yes |
| `*_hparams.yaml`, `score_*.json`, `trials_*.csv`, `outputs/results/*.pkl` | No | Yes |

Fifty-two `.skops` files are at risk: 26 xgb and 26 mlp, each 25 cross-validation folds
plus one final model. They live in `outputs/cv/{xgb,mlp}_20260828_182305/` and
`outputs/models/{xgb,mlp}_20260828_182305/`.

The gnn backend needs no migration. The learned weights of every backend survive
untouched; only the sklearn-side wrapper of xgb and mlp — preprocessor, target
`QuantileTransformer`, constructor config — rides in the `.skops` file.

Loading an unmigrated file raises `ModuleNotFoundError` inside `skops.io.load`.

Neither backend's `load()` guard needs changing. Both test the class name as a
substring (`any(a in t for a in ("XGBoostRegressor", "numpy.dtype"))`), which the
module rename leaves intact.

## Naming

| Thing | From | To |
|---|---|---|
| Import package | `protac_synth` | `retrotac` |
| Distribution | `protac-synthesizability` | `retrotac` |
| Mamba env | `env-protac-synth` | `env-retrotac` |
| W&B project | `protac-synth` | `retrotac` |
| GitHub repo and local directory | `PROTAC-Synthesizability` | `RetroTAC` |

`retro_scores/` keeps its name. Only its docstring references to the main package change.

## Compatibility mechanism

Two independent layers. Either alone suffices; together they cover artifacts we cannot
reach.

### Layer 1: alias shim

`retrotac/_compat.py`, invoked from `retrotac/__init__.py`, aliases the top-level
package only:

```python
sys.modules["protac_synth"] = sys.modules["retrotac"]
```

and emits a `DeprecationWarning`. This rescues artifacts nobody migrated: copies
published to Hugging Face, a colleague's `outputs/`, an old backup. Confirmed working:
a dumped object whose defining package had been renamed away loaded correctly through
the alias, fitted sklearn state included.

**Alias the top-level package only.** Registering the leaf modules
(`protac_synth.models.mlp.model` and friends) eagerly would import torch and xgboost on
every `import retrotac`, destroying the lazy-backend-import property that `get_build_fn`
exists to provide. Verified: with the top-level alias alone, Python resolves
`importlib.import_module("protac_synth.models.xgb.model")` on demand through the
package's `__path__`, and the leaf module stays unimported until something asks for it.

That resolution produces a class object distinct from `retrotac.models.xgb.model.
XGBoostRegressor`, so `isinstance` across the two would fail. Harmless here: the
codebase contains no `isinstance` check against any backend class and dispatches purely
on the `load`/`predict` contract. The oracle test guards the property that matters,
which is prediction equality.

The shim is temporary. Delete it once no unmigrated artifact remains in circulation.

### Layer 2: artifact migration

`scripts/maintenance/migrate_skops_module_path.py` rewrites `schema.json` inside each
`.skops` zip, replacing `"protac_synth.` with `"retrotac.`, and copies every other
member byte for byte. Confirmed on a real artifact: skops afterwards reports
`retrotac.models.xgb.model.XGBoostRegressor`, and nothing checksums the archive.

Requirements: idempotent, `--dry-run`, `--revert`, and a `.bak` sidecar per file.

## Regression oracle

Before any edit, load all three final models, predict on a fixed SMILES set, and freeze
the output to JSON. Every phase replays that comparison. A rename that moves a single
prediction has failed.

Capture and replay need a GPU allocation. Any xgboost call fails on the Berzelius login
node, because the wheel probes for a GPU even under `device="cpu"` and the login node
sets `compute_mode=Prohibited`. Use a SLURM job or `apptainer --nv`.

## Phases

Phases 1 through 4 form one atomic unit. The repository does not work between them.

| Phase | Work |
|---|---|
| 0 | Branch `stefano/rename-retrotac`. Capture golden predictions. |
| 1 | `git mv protac_synth retrotac`. Rewrite 43 imports. Update `pyproject.toml` `name` and `packages.find`. Delete the stale `protac_synthesizability.egg-info`. |
| 2 | Add `retrotac/_compat.py` and a test that loads an unmigrated `.skops` and matches the oracle. |
| 3 | Run the migration over all 52 artifacts. Replay the oracle. |
| 4 | Update the `%files` line in the three `apptainer/*.def` files. Rebuild the images. Replay the oracle inside `inference.sif`. |
| 5 | Rename the W&B project and mamba env. Update docs and the four notebooks. |
| 6 | Rename the GitHub repository and the local directory. Fix the absolute `cd` in `scripts/slurm/run_protac_splitter.sh`. |

Phases 5 and 6 revert independently.

`apptainer/bind_live_repo.sh` needs no change. It enumerates top-level repository
entries at runtime rather than naming the package.

## Reference surface

Forty-three import statements across 35 files, all of the form
`from protac_synth.X import ...`. Beyond those:

- `pyproject.toml`: `name`, `[tool.setuptools.packages.find] include`
- `apptainer/{inference,scoring,training}.def`: one `%files` line each
- Four notebooks: `analyze_llm_scores`, `analyze_routes`, `figures`, `test_models`
- Docs: `CLAUDE.md`, `CONTRIBUTING.md`, `README.md`, `apptainer/README.md`,
  `retro_scores/README.md`
- Six SLURM submitters plus `setup_env.sh`: `mamba activate env-protac-synth`
- `config/models_config.yaml` and `config/models_config_routes.yaml`: `wandb_project`

`scripts/models/predict.py` takes the Hugging Face repository id as an argument, so no
hardcoded id needs updating.

## Out of scope

Two paths keep the old name deliberately:

- `config/models_config*.yaml`'s `chemeleon_weights` points at
  `/home/x_steri/storage/PROTAC-Synthesizability/`, a different directory from this
  checkout. Renaming that is a separate decision.
- `docs/PIPELINE.md`, `README_old.md`, and `docs/CHEM_SCORING.md` describe a layout that
  no longer exists. Leave them stale rather than half-correct.

## Incidental fixes

Folded in because the rename touches the same lines:

- `notebooks/analyze_routes.ipynb` and `notebooks/analyze_llm_scores.ipynb` import
  `protac_synth.route_parsing`, which has never existed. The module is
  `scripts/llm_scoring/route_parsing.py`. Point them at it.
- `train_5x5.sh` and `retro_scores/route_scores/aizynthfinder_utils.py` reference the
  removed `src/protac_synth/` layout. Rename the strings for consistency. Both stay
  broken and unsupported; repairing them is separate work.

## Risks accepted by the user

- The local directory rename breaks `.venv`, whose console scripts hardcode the absolute
  path. Accepted: the user works through the `.sif` images, not the venv.
- Renaming the W&B project orphans existing run history. Accepted: the user is not
  currently using W&B.
