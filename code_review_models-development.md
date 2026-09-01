# Code Review: `models-development` (vs `main`)

Reviewed diff: `git diff main...HEAD` on branch `models-development`.
Scope: new ML training subsystem under `src/protac_synth/models/` (`mlp/`, `xgb/`, `mol_utils.py`, `torch_common.py`, `train.py`, `models_config.yaml`).

9 findings, most severe first.

## Correctness

### 1. Tuple-assignment bug breaks on-the-fly fingerprinting in XGBoost
**File:** `src/protac_synth/models/xgb/model.py:55` · **Verdict:** CONFIRMED

```python
X_fp = compute_fingerprints(smiles_list), self.fp_size, self.fp_radius
```

Missing parentheses turn this into a 3-tuple assignment `(ndarray, fp_size, fp_radius)` instead of a call to `compute_fingerprints(smiles_list, self.fp_size, self.fp_radius)`.

**Failure scenario:** This path runs whenever `X_fp` is `None` — e.g. `XGBoostRegressor.predict()` on new/uncached SMILES after `load()`, the documented inference path. `sanitize_matrix(X_fp)` then calls `np.isinf()`/`np.clip()` on a tuple containing an ndarray plus two ints, raising an error or producing a malformed array. Even where it doesn't crash, `self.fp_size`/`self.fp_radius` are silently ignored, desyncing feature width from what `make_preprocessor`'s `ColumnTransformer` expects. The parallel line in `mlp/model.py:66` does this correctly, confirming it's a copy-paste typo.

### 2. `--model gnn` is a valid CLI choice but crashes with an unclear error
**File:** `src/protac_synth/models/train.py:276` · **Verdict:** CONFIRMED

`src/protac_synth/models/gnn/` contains only an empty `__init__.py`, no `hpo.py`.

**Failure scenario:** `python train.py --model gnn --input data.csv --seed 0 --fold 0` passes argparse validation, then crashes with `ModuleNotFoundError: No module named 'gnn.hpo'` deep inside `get_build_fn()` (train.py:59-61) instead of failing at the CLI-argument level with a clear message.

## Efficiency

### 3. Every molecule is RDKit-standardized twice per feature-cache pass
**File:** `src/protac_synth/models/train.py:96` · **Verdict:** CONFIRMED

`cache_features()` calls `compute_fingerprints(smiles, ...)` and `compute_descriptors(smiles)` on the same SMILES list. Both functions independently call `mol_utils.standardize(smi)` per molecule (`mol_utils.py:50` and `:122`).

**Failure scenario:** With the default config (`use_fingerprints: true`, `use_descriptors: true`), every molecule is parsed/standardized twice instead of once, doubling RDKit cost across the whole dataset during `--precompute`.

## Reuse

### 4. `_featurize`/`score` duplicated between XGB and MLP with no shared base
**File:** `src/protac_synth/models/mlp/model.py:120` (and `xgb/model.py`) · **Verdict:** PLAUSIBLE

**Failure scenario:** The two copies have already diverged: XGB's copy has the tuple bug above while MLP's copy is correct. Without a shared implementation in `mol_utils.py`, future changes to featurization logic must be applied twice and can silently drift apart, as already happened here.

## CLAUDE.md Convention Violations

### 5. Module-level side effects on import
**File:** `src/protac_synth/models/train.py:29` (also `mlp/hpo.py:24-26`, `xgb/hpo.py:864-867`) · **Verdict:** CONFIRMED

CLAUDE.md: *"Every script should end up: With a `main()` + `parse_args()` so it is invocable from the CLI without side effects on import."* `train.py` opens/parses `models_config.yaml` and creates `OPTUNA_DB`/`CV_DIR`/`MODELS_DIR` directories at module scope — a bare `import train` performs file I/O and creates directories on disk.

### 6. Missing `tqdm` on per-molecule loops
**File:** `src/protac_synth/models/mol_utils.py:49` (and `:121`) · **Verdict:** CONFIRMED

CLAUDE.md: *"Using `tqdm` for any slow loop (pandas `apply`, API calls, file iteration)."* `compute_fingerprints` and `compute_descriptors` both loop over `smiles_list` with plain `for` loops and no progress bar — silent on the dataset scales (hundreds of thousands of SMILES) this repo works with elsewhere.

### 7. `dict = None` instead of `Optional[Dict]`
**File:** `src/protac_synth/models/xgb/model.py:24` · **Verdict:** CONFIRMED

CLAUDE.md: *"Type hints ... Use `typing` module types: List, Dict, Optional, Tuple, Any — not PEP 585 built-in generics."* `xgb_params: dict = None` uses the bare `dict` builtin with a `None` default instead of `typing.Optional[Dict]`, inconsistent with `Optional[np.ndarray]` two lines below in the same file (and `typing.Optional` is already imported).

### 8. Undocumented public functions
**File:** `src/protac_synth/models/train.py:52` · **Verdict:** CONFIRMED

CLAUDE.md: *"Every public function gets a docstring with an Args: and Returns: section."* `get_build_fn`, `tune_inner_fold`, `run_single_fold`, and `aggregate_results` — the core of the new CV pipeline — have no docstrings at all, breaking the convention already followed in `mlp/hpo.py`'s `build_mlp` and `xgb/hpo.py`'s `build_xgb`.

## Simplification

### 9. Dead `descriptors_list` constructor parameter
**File:** `src/protac_synth/models/xgb/model.py:25` · **Verdict:** CONFIRMED

`descriptors_list` is stored on `self` but never read by `_featurize`, `fit`, `predict`, `save`, or `load`.

**Failure scenario:** A caller passing `XGBoostRegressor(descriptors_list=[...])` to restrict which RDKit descriptors are used gets no effect — descriptor computation always goes through the module-level `mol_utils.compute_descriptors`/`DESCRIPTOR_CALCULATOR` instead, which duplicates (and can silently diverge from) the descriptor list this parameter builds from `Descriptors._descList`.
