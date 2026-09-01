# `retrotac` Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the main Python package from `protac_synth` to `retrotac` without changing a single prediction from the already-trained models.

**Architecture:** A regression oracle is captured first and replayed after every phase. The package moves with `git mv`; imports are rewritten mechanically. Two independent compatibility layers protect the 52 `.skops` artifacts that embed the old import path: a lazy top-level `sys.modules` alias, and a one-time rewrite of `schema.json` inside each archive.

**Tech Stack:** Python 3.12, uv, skops, xgboost, PyTorch Lightning, chemprop, Apptainer.

**Spec:** `docs/superpowers/specs/2026-09-01-retrotac-rename-design.md`

## Global Constraints

- Old name: `protac_synth` (package), `protac-synthesizability` (distribution). New name: `retrotac` for both.
- Run everything through the container. There is no usable `.venv` for this work:
  `apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif python ...`
- `training.sif` is the image to use. It carries all three backends. Do not rebuild images until Task 5.
- There is no `tests/` directory and **pytest is not installed anywhere**. Verification is standalone scripts run with `python`, never `pytest`. Do not add a pytest dependency.
- `outputs/` is gitignored. Migrated artifacts and `.bak` files are never committed.
- Golden predictions for `["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]` against the `20260828_182305` final models: xgb `0.602399, 0.583112, 0.638612`; mlp `0.603565, 0.633335, 0.000000`; gnn `0.360836, 0.651754, 0.667062`. The mlp zero is real; preserve it, do not fix it.
- Tolerance for prediction equality: exact to 6 decimal places.
- Never edit `docs/PIPELINE.md`, `README_old.md`, or `docs/CHEM_SCORING.md`. They are knowingly stale.
- Do not touch `chemeleon_weights` paths in `config/models_config*.yaml`. They point outside this checkout.

---

### Task 1: Regression oracle

The safety net for everything that follows. Must land before any rename.

**Files:**
- Create: `scripts/maintenance/verify_model_artifacts.py`
- Create (generated, gitignored): `outputs/golden_predictions.json`

**Interfaces:**
- Consumes: nothing.
- Produces: `scripts/maintenance/verify_model_artifacts.py` with `--capture` and `--verify` flags and a `--package NAME` flag (default `retrotac`) so later tasks can point it at either package name. Exit code 0 on match, 1 on mismatch.

- [ ] **Step 1: Create the script**

```python
"""Capture and replay golden predictions from the saved final models.

Guards the protac_synth -> retrotac rename: the learned weights must not move.
Run --capture before renaming anything, --verify after every phase.

    apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \\
        python scripts/maintenance/verify_model_artifacts.py --capture
"""

import argparse
import json
import sys
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
GOLDEN_PATH = Path("outputs/golden_predictions.json")
RUN = "20260828_182305"
BACKENDS = {
    "xgb": ("models.xgb.model", "XGBoostRegressor"),
    "mlp": ("models.mlp.model", "TorchMLPRegressor"),
    "gnn": ("models.gnn.model", "CheMeleonRegressor"),
}


def collect(package: str) -> dict:
    """Load each final model through `package` and predict on SMILES."""
    chem = import_module(f"{package}.chem_utils")
    x_desc = chem.compute_descriptors(chem.standardize_all(SMILES))
    out = {}
    for name, (submodule, clsname) in BACKENDS.items():
        backend = getattr(import_module(f"{package}.{submodule}"), clsname)
        model = backend.load(f"outputs/models/{name}_{RUN}/{name}_{RUN}_final")
        # Only the gnn backend featurizes internally; the other two take descriptors.
        preds = (
            model.predict(SMILES)
            if name == "gnn"
            else model.predict(SMILES, X_desc=x_desc)
        )
        out[name] = [round(float(v), 6) for v in np.asarray(preds).ravel()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true", help="Write the golden file.")
    mode.add_argument("--verify", action="store_true", help="Compare against it.")
    ap.add_argument("--package", default="retrotac", help="Package to import from.")
    args = ap.parse_args()

    actual = collect(args.package)

    if args.capture:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        print(f"captured -> {GOLDEN_PATH}")
        for name, preds in sorted(actual.items()):
            print(f"  {name}: {preds}")
        return 0

    if not GOLDEN_PATH.exists():
        print(f"FAIL: no golden file at {GOLDEN_PATH}; run --capture first.")
        return 1

    expected = json.loads(GOLDEN_PATH.read_text())
    ok = True
    for name in sorted(BACKENDS):
        if actual.get(name) != expected.get(name):
            ok = False
            print(f"  MISMATCH {name}: expected {expected.get(name)}, got {actual.get(name)}")
        else:
            print(f"  ok {name}: {actual[name]}")
    print("PASS: all backends match." if ok else "FAIL: predictions moved.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Capture the baseline against the OLD package name**

The package is still called `protac_synth` at this point, so pass it explicitly.

Run:
```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/verify_model_artifacts.py --capture --package protac_synth
```
Expected: `captured -> outputs/golden_predictions.json`, printing the three arrays from Global Constraints.

- [ ] **Step 3: Confirm the values match the plan's recorded constants**

Run:
```bash
cat outputs/golden_predictions.json
```
Expected: xgb `[0.602399, 0.583112, 0.638612]`, mlp `[0.603565, 0.633335, 0.0]`, gnn `[0.360836, 0.651754, 0.667062]`.

If they differ, **stop and report**. It means the artifacts on disk are not the ones this plan was written against, and the rename must not proceed until that is understood.

- [ ] **Step 4: Confirm verify mode detects a real mismatch**

A test that never fails proves nothing. Temporarily corrupt the golden file and check the script notices.

Run:
```bash
cp outputs/golden_predictions.json /tmp/golden.bak
python3 -c "
import json,pathlib
p=pathlib.Path('outputs/golden_predictions.json'); d=json.loads(p.read_text())
d['xgb'][0]=0.123456; p.write_text(json.dumps(d,indent=2,sort_keys=True))"
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/verify_model_artifacts.py --verify --package protac_synth; echo "exit=$?"
cp /tmp/golden.bak outputs/golden_predictions.json
```
Expected: `MISMATCH xgb`, `FAIL: predictions moved.`, `exit=1`. Then the copy restores the real file.

- [ ] **Step 5: Confirm verify passes on the restored file**

Run:
```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/verify_model_artifacts.py --verify --package protac_synth; echo "exit=$?"
```
Expected: three `ok` lines, `PASS: all backends match.`, `exit=0`.

- [ ] **Step 6: Commit**

```bash
git add scripts/maintenance/verify_model_artifacts.py
git commit -m "Add regression oracle for the retrotac rename

Loads each final model and predicts on a fixed SMILES set, so the
rename can be proven not to move any prediction. outputs/ is
gitignored, so the golden file itself stays local."
```

---

### Task 2: Rename the package

**Files:**
- Rename: `protac_synth/` -> `retrotac/` (20 `.py` files)
- Modify: `pyproject.toml` (`name`, `[tool.setuptools.packages.find] include`)
- Modify: 19 `.py` files containing `from protac_synth`
- Delete: `protac_synthesizability.egg-info/`

**Interfaces:**
- Consumes: `scripts/maintenance/verify_model_artifacts.py` from Task 1.
- Produces: importable package `retrotac` with identical module layout. Every `from protac_synth.X import Y` becomes `from retrotac.X import Y`.

- [ ] **Step 1: Move the directory**

```bash
git mv protac_synth retrotac
```

- [ ] **Step 2: Rewrite every import**

All 43 import statements use the `from protac_synth.` form. Rewrite them across the whole tree, excluding `.git`, `.venv` and the notebooks (notebooks are Task 6).

```bash
grep -rl "protac_synth" --include=*.py . --exclude-dir=.git --exclude-dir=.venv \
  | xargs sed -i 's/\bprotac_synth\b/retrotac/g'
```

- [ ] **Step 3: Update pyproject.toml**

Change two lines:

```toml
name = "retrotac"
```
```toml
include = ["retrotac*"]
```

Leave every dependency, extra, and `[tool.uv.sources]` entry untouched.

- [ ] **Step 4: Remove the stale egg-info**

```bash
rm -rf protac_synthesizability.egg-info
```

- [ ] **Step 5: Verify no `protac_synth` remains in Python or packaging files**

Run:
```bash
grep -rn "protac_synth" --include=*.py --include=*.toml . --exclude-dir=.git --exclude-dir=.venv
```
Expected: **no output at all.** The substitution covers `retro_scores/`'s docstrings and
`aizynthfinder_utils.py`'s `src.protac_synth` string too, since they are `.py` files.

Then check the underscore-spelled distribution name, which `\bprotac_synth\b` does not
match and so survives:

```bash
grep -rn "protac_synthesizability" --include=*.py . --exclude-dir=.git --exclude-dir=.venv
```
Expected: one hit, `retrotac/stock_utils/sqlite_stock.py:14`, a docstring. Task 6 fixes it.

- [ ] **Step 6: Verify the package imports and predictions are unchanged**

The container has the old distribution installed, but `bind_live_repo.sh` mounts the live tree over `/opt/repo`, so `retrotac/` is importable from the working directory.

Run:
```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/verify_model_artifacts.py --verify --package retrotac; echo "exit=$?"
```
Expected: **`FAIL`** — specifically `ModuleNotFoundError: No module named 'protac_synth'` raised from inside `skops.io.load` for xgb and mlp. This is the expected, documented breakage, and confirms the problem this plan exists to solve is real.

If gnn appears in the output as `ok`, that is correct: the gnn checkpoint carries no module path.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Rename package protac_synth -> retrotac

Moves the package directory, rewrites all 43 imports, and updates the
distribution name and setuptools package glob. xgb and mlp .skops
artifacts do not load at this commit; Tasks 3 and 4 restore them."
```

---

### Task 3: Compatibility shim

**Files:**
- Create: `retrotac/_compat.py`
- Modify: `retrotac/__init__.py` (currently empty)

**Interfaces:**
- Consumes: package `retrotac` from Task 2.
- Produces: `retrotac._compat.install_legacy_aliases()`, called at package import. After it runs, `importlib.import_module("protac_synth.models.xgb.model")` resolves.

- [ ] **Step 1: Create the shim**

Alias the **top-level package only**. Registering leaf modules eagerly would import torch and xgboost on every `import retrotac`, destroying the lazy-backend-import property that `get_build_fn` exists to provide.

```python
"""Backward compatibility for artifacts saved under the old package name.

Models trained before the rename embed `protac_synth.models.*.model` in their
.skops archives (see docs/superpowers/specs/2026-09-01-retrotac-rename-design.md).
scripts/maintenance/migrate_skops_module_path.py rewrites the archives we can
reach; this shim covers the ones we cannot, such as copies published to the
Hugging Face Hub.

Temporary. Delete once no unmigrated artifact remains in circulation.
"""

import sys

LEGACY_NAME = "protac_synth"


def install_legacy_aliases() -> None:
    """Make `protac_synth[...]` imports resolve to `retrotac[...]`.

    Only the top-level package is aliased. Python then resolves submodules on
    demand through the package's __path__, which keeps the backend imports
    lazy: aliasing `retrotac.models.mlp.model` here would pull in torch on
    every import of this package.
    """
    if LEGACY_NAME in sys.modules:
        return
    sys.modules[LEGACY_NAME] = sys.modules[__name__.rsplit(".", 1)[0]]
```

The alias is deliberately silent. Warning at install time would fire on every
`import retrotac`, including the overwhelming majority of runs that touch no legacy
artifact at all, and warning only on genuine legacy use would need a `MetaPathFinder`
that buys nothing: `migrate_skops_module_path.py --dry-run` already reports exactly
which archives still carry the old path.

- [ ] **Step 2: Call it from the package init**

`retrotac/__init__.py` is currently empty. Replace it with:

```python
"""RetroTAC: PROTAC synthesizability prediction."""

from retrotac._compat import install_legacy_aliases

install_legacy_aliases()
```

- [ ] **Step 3: Verify the shim restores loading**

Run:
```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/verify_model_artifacts.py --verify --package retrotac; echo "exit=$?"
```
Expected: three `ok` lines, `PASS: all backends match.`, `exit=0`. The unmigrated `.skops` files now load through the alias.

- [ ] **Step 4: Verify the shim did NOT break lazy backend imports**

This is the regression the shim design most easily causes. An xgb-only run must not import torch.

Run:
```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif python -c "
import sys, retrotac, retrotac.models.xgb.model
print('torch imported:', 'torch' in sys.modules)
print('protac_synth aliased:', sys.modules.get('protac_synth') is retrotac)
"
```
Expected:
```
torch imported: False
protac_synth aliased: True
```
If `torch imported: True`, the shim is aliasing leaf modules. Fix it before continuing.

- [ ] **Step 5: Commit**

```bash
git add retrotac/_compat.py retrotac/__init__.py
git commit -m "Add legacy protac_synth import alias

Aliases the top-level package only, so submodules still resolve lazily
and an xgb run does not import torch. Restores loading of .skops
artifacts that embed the pre-rename module path."
```

---

### Task 4: Migrate the saved artifacts

**Files:**
- Create: `scripts/maintenance/migrate_skops_module_path.py`

**Interfaces:**
- Consumes: package `retrotac` from Task 2.
- Produces: a CLI over `outputs/`, with `--dry-run`, `--revert`, and `--root PATH` (default `outputs`). Rewrites `schema.json` inside each `.skops` archive, leaving every other member byte-identical, and writes a `.bak` sidecar per file.

- [ ] **Step 1: Create the migration script**

```python
"""Rewrite the embedded package path inside saved .skops archives.

A .skops file is a zip whose schema.json records the fully qualified module
of the pickled class. Models saved before the rename record
`protac_synth.models.{xgb,mlp}.model`, which no longer imports. This rewrites
that string and copies every other member unchanged. Nothing checksums the
archive, so the edit is safe.

Idempotent: already-migrated files are skipped.

    python scripts/maintenance/migrate_skops_module_path.py --dry-run
    python scripts/maintenance/migrate_skops_module_path.py
    python scripts/maintenance/migrate_skops_module_path.py --revert
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

OLD = '"protac_synth.'
NEW = '"retrotac.'
SCHEMA = "schema.json"


def needs_migration(path: Path) -> bool:
    """True when the archive's schema still names the old package."""
    with zipfile.ZipFile(path) as zf:
        return OLD in zf.read(SCHEMA).decode()


def migrate(path: Path, dry_run: bool) -> bool:
    """Rewrite one archive in place, keeping a .bak sidecar. True if changed."""
    if not needs_migration(path):
        return False
    if dry_run:
        return True

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == SCHEMA:
                data = data.decode().replace(OLD, NEW).encode()
            zout.writestr(item, data)
    tmp.replace(path)
    return True


def revert(path: Path) -> bool:
    """Restore one archive from its .bak sidecar. True if restored."""
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        return False
    shutil.copy2(backup, path)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs", help="Directory to scan.")
    ap.add_argument("--dry-run", action="store_true", help="Report, change nothing.")
    ap.add_argument("--revert", action="store_true", help="Restore from .bak files.")
    args = ap.parse_args()

    files = sorted(Path(args.root).rglob("*.skops"))
    if not files:
        print(f"No .skops files under {args.root}/")
        return 1

    changed = 0
    for path in files:
        if args.revert:
            if revert(path):
                changed += 1
                print(f"  reverted {path}")
            continue
        if migrate(path, args.dry_run):
            changed += 1
            print(f"  {'would migrate' if args.dry_run else 'migrated'} {path}")

    verb = "reverted" if args.revert else ("would migrate" if args.dry_run else "migrated")
    print(f"{verb} {changed} of {len(files)} .skops files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Dry run**

Run:
```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/migrate_skops_module_path.py --dry-run
```
Expected: `would migrate 52 of 52 .skops files`.

If the count is not 52, **stop and report** before writing anything.

- [ ] **Step 3: Migrate**

Run:
```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/migrate_skops_module_path.py
```
Expected: `migrated 52 of 52 .skops files`.

- [ ] **Step 4: Verify predictions still match**

Run:
```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/verify_model_artifacts.py --verify --package retrotac; echo "exit=$?"
```
Expected: `PASS: all backends match.`, `exit=0`.

- [ ] **Step 5: Verify the migration is idempotent**

Run:
```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/migrate_skops_module_path.py --dry-run
```
Expected: `would migrate 0 of 52 .skops files`.

- [ ] **Step 6: Verify migrated artifacts load WITHOUT the shim**

This proves the migration genuinely worked rather than the shim masking it.

Run:
```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif python -c "
import sys, warnings; warnings.filterwarnings('ignore')
import retrotac
del sys.modules['protac_synth']          # disable the shim
import skops.io as sio
f='outputs/models/xgb_20260828_182305/xgb_20260828_182305_final.skops'
print('untrusted types:', sio.get_untrusted_types(file=f))
sio.load(f, trusted=sio.get_untrusted_types(file=f)); print('loaded without shim: OK')
"
```
Expected: `['numpy.dtype', 'retrotac.models.xgb.model.XGBoostRegressor']` then `loaded without shim: OK`.

- [ ] **Step 7: Verify revert works**

The escape hatch must be proven before it is needed.

Run:
```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/migrate_skops_module_path.py --revert | tail -1
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/migrate_skops_module_path.py --dry-run | tail -1
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/migrate_skops_module_path.py | tail -1
```
Expected, in order: `reverted 52 of 52`, `would migrate 52 of 52`, `migrated 52 of 52`. The tree ends migrated again.

- [ ] **Step 8: Commit**

Only the script is committed; `outputs/` is gitignored.

```bash
git add scripts/maintenance/migrate_skops_module_path.py
git commit -m "Add .skops module-path migration script

Rewrites the embedded package path inside saved skops archives so
artifacts trained before the rename load without the compatibility
shim. Idempotent, with --dry-run and --revert."
```

---

### Task 5: Containers

**Files:**
- Modify: `apptainer/inference.def:13`, `apptainer/scoring.def:22`, `apptainer/training.def:15`

**Interfaces:**
- Consumes: package `retrotac` from Task 2.
- Produces: three `.def` files whose `%files` section copies `retrotac`. `apptainer/bind_live_repo.sh` needs **no change**; it enumerates top-level entries at runtime.

- [ ] **Step 1: Update all three definition files**

Each contains exactly one line to change, in its `%files` section:

```bash
sed -i 's|^    protac_synth /opt/repo/protac_synth$|    retrotac /opt/repo/retrotac|' \
    apptainer/inference.def apptainer/scoring.def apptainer/training.def
```

Also update `apptainer/inference.def:5`, a comment reading `surrogate model (protac_synth core deps ...)`, to say `retrotac`.

- [ ] **Step 2: Verify no definition file still names the old package**

Run:
```bash
grep -n "protac_synth" apptainer/*.def apptainer/*.sh; echo "exit=$?"
```
Expected: no output, `exit=1` (grep found nothing).

- [ ] **Step 3: Commit the definition changes before rebuilding**

Rebuilds are slow and the images are gitignored, so commit the source change first.

```bash
git add apptainer/
git commit -m "Point container definitions at the renamed package"
```

- [ ] **Step 4: Rebuild the three images**

Each takes several minutes and the existing `.sif` files are large (3.4 GB, 3.5 GB, 6.9 GB). Confirm free disk before starting.

Run:
```bash
df -h . | tail -1
apptainer build --fakeroot apptainer/inference.sif apptainer/inference.def
apptainer build --fakeroot apptainer/training.sif apptainer/training.def
apptainer build --fakeroot apptainer/scoring.sif apptainer/scoring.def
```
Expected: three `INFO: Build complete` lines.

If a build fails, **stop and report**. Do not delete the existing images to make room without asking; they are the only working copies and are not in git.

- [ ] **Step 5: Verify predictions inside the rebuilt image**

Run:
```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/verify_model_artifacts.py --verify --package retrotac; echo "exit=$?"
```
Expected: `PASS: all backends match.`, `exit=0`.

- [ ] **Step 6: Verify the image works without the live bind mount**

The baked-in copy must be correct too, otherwise the image is only usable from this checkout.

Run:
```bash
apptainer exec apptainer/inference.sif python -c "
import retrotac, retrotac.chem_utils as c
print('baked-in import OK:', c.canon_smiles('CCO'))"
```
Expected: `baked-in import OK: CCO`.

---

### Task 6: Ancillary names, docs, and notebooks

**Files:**
- Modify: `config/models_config.yaml:29`, `config/models_config_routes.yaml:17` (`wandb_project`)
- Modify: `setup_env.sh:12` and five SLURM submitters (`mamba activate env-protac-synth`)
- Modify: `CLAUDE.md`, `CONTRIBUTING.md`, `README.md`, `apptainer/README.md`, `retro_scores/README.md`
- Modify: `notebooks/analyze_llm_scores.ipynb`, `notebooks/analyze_routes.ipynb`, `notebooks/figures.ipynb`, `notebooks/test_models.ipynb`
- Modify: `train_5x5.sh`, `retro_scores/route_scores/aizynthfinder_utils.py`

**Interfaces:**
- Consumes: everything above. Produces no importable interface.

- [ ] **Step 1: Rename the W&B project and mamba environment**

```bash
sed -i 's/wandb_project: protac-synth/wandb_project: retrotac/' \
    config/models_config.yaml config/models_config_routes.yaml
grep -rl "env-protac-synth" --include=*.sh --include=*.py . --exclude-dir=.git --exclude-dir=.venv \
  | xargs sed -i 's/env-protac-synth/env-retrotac/g'
```

- [ ] **Step 2: Rewrite the package name in notebooks**

Notebooks are JSON; a plain textual substitution is safe here because `protac_synth` appears only inside source strings.

```bash
sed -i 's/\bprotac_synth\b/retrotac/g' notebooks/*.ipynb
```

- [ ] **Step 3: Fix the two notebooks importing a module that never existed**

`analyze_routes.ipynb` and `analyze_llm_scores.ipynb` import `retrotac.route_parsing` (was `protac_synth.route_parsing`), which has never existed in either package. The real module is `scripts/llm_scoring/route_parsing.py`. Replace the import with a path-based one:

```python
import sys; sys.path.insert(0, "scripts/llm_scoring")
from route_parsing import parse_route_row, render_route_for_llm
```

In `analyze_llm_scores.ipynb` the imported names are `describe_molecule, parse_route_row`. Use those instead. Also fix the markdown cell in that notebook claiming the module lives at `src/protac_synth/route_parsing.py`.

- [ ] **Step 4: Update the remaining prose references**

Rewrite `protac_synth` -> `retrotac` and both spellings of the distribution name
(`protac-synthesizability`, `protac_synthesizability`) -> `retrotac` throughout
`CLAUDE.md`, `CONTRIBUTING.md`, `README.md`, `apptainer/README.md`,
`retro_scores/README.md`, and `scripts/analysis/README.md`.

Three further spots the Task 2 substitution could not reach:

- `train_5x5.sh` — a shell script, excluded by that task's `--include=*.py`.
- `retrotac/stock_utils/sqlite_stock.py:14` — the docstring's `protac_synthesizability/stock/`.
- `scripts/analysis/README.md:21,24` — `src/protac_synthesizability/route_analysis/`.

`train_5x5.sh` and `retro_scores/route_scores/aizynthfinder_utils.py` reference the long-removed `src/` layout. They stay broken and unsupported. Do not attempt to repair them.

**Do not edit** `docs/PIPELINE.md`, `README_old.md`, or `docs/CHEM_SCORING.md`.

- [ ] **Step 5: Correct CLAUDE.md's login-node claim**

CLAUDE.md states that any xgboost call fails on the Berzelius login node because the wheel probes for a GPU even under `device='cpu'`. This plan measured otherwise: all three backends load and predict there. Narrow the claim to `.fit()`, in both the "Model training cannot run on the Berzelius login node" section and the "Containers" section.

- [ ] **Step 6: Document the compatibility shim in CLAUDE.md**

Add a short paragraph to the `retrotac/` architecture section: models trained before the rename embed the old module path; `retrotac/_compat.py` aliases it lazily; `scripts/maintenance/migrate_skops_module_path.py` migrates archives permanently; the shim is temporary.

- [ ] **Step 7: Verify only intentional references remain**

Run:
```bash
grep -rn "protac_synth\|protac-synthesizability\|protac_synthesizability\|env-protac-synth" \
    --exclude-dir=.git --exclude-dir=.venv --exclude-dir=outputs . \
  | grep -v "docs/PIPELINE.md\|README_old.md\|docs/CHEM_SCORING.md\|docs/superpowers/"
```
Expected: only `retrotac/_compat.py` (which must name the legacy package) and `scripts/maintenance/migrate_skops_module_path.py` (whose docstring and `OLD` constant must too).

- [ ] **Step 8: Verify the notebooks are still valid JSON**

`sed` on a notebook can corrupt it. Check before committing.

Run:
```bash
python3 -c "
import json,glob
for f in glob.glob('notebooks/*.ipynb'):
    json.load(open(f)); print('ok', f)"
```
Expected: one `ok` line per notebook, no traceback.

- [ ] **Step 9: Final prediction check**

Run:
```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/verify_model_artifacts.py --verify --package retrotac; echo "exit=$?"
```
Expected: `PASS: all backends match.`, `exit=0`.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "Rename remaining protac_synth references in configs, docs, notebooks

Renames the W&B project and mamba environment, updates the five
maintained docs and four notebooks, and documents the compatibility
shim. Also narrows CLAUDE.md's claim that xgboost cannot run on the
login node: measured, that applies to .fit() but not to inference.
Fixes two notebook imports of a route_parsing module that has never
existed in the package."
```

---

### Task 7: Rename the repository and directory

Do this last and separately. Everything above is complete and verified without it.

**Files:**
- Modify: `scripts/slurm/run_protac_splitter.sh:23` (absolute `cd`)

**Interfaces:**
- Consumes: a fully renamed, verified working tree.
- Produces: nothing importable.

- [ ] **Step 1: Confirm with the user before touching anything outside the checkout**

Renaming the GitHub repository and moving the working directory affect things this plan cannot see: the user's shell history, any other clone, and scheduled SLURM scripts holding the absolute path. **Ask before proceeding.** Report that the local `.venv` will break (its console scripts hardcode the absolute path) and that this was already accepted, since the user works through the `.sif` images.

- [ ] **Step 2: Update the one absolute path inside the repo**

```bash
sed -i 's|/proj/berzelius-2026-62/users/x_steri/PROTAC-Synthesizability|/proj/berzelius-2026-62/users/x_steri/RetroTAC|' \
    scripts/slurm/run_protac_splitter.sh
```

Leave `config/models_config*.yaml`'s `chemeleon_weights` alone. It points at `/home/x_steri/storage/PROTAC-Synthesizability/`, a different directory that this rename does not move.

- [ ] **Step 3: Commit and push while the path is still valid**

```bash
git add -A && git commit -m "Point the SLURM splitter script at the renamed repository directory"
git push -u origin stefano/rename-retrotac
```

- [ ] **Step 4: Rename the GitHub repository**

```bash
gh repo rename RetroTAC --repo ribesstefano/PROTAC-Synthesizability
git remote set-url origin https://github.com/ribesstefano/RetroTAC.git
```
GitHub redirects the old URL, so other clones keep working.

- [ ] **Step 5: Move the working directory**

Must run from the parent directory, not from inside the tree.

```bash
cd /proj/berzelius-2026-62/users/x_steri
mv PROTAC-Synthesizability RetroTAC
cd RetroTAC
```

- [ ] **Step 6: Final verification from the new location**

Run:
```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/verify_model_artifacts.py --verify --package retrotac; echo "exit=$?"
```
Expected: `PASS: all backends match.`, `exit=0`.

- [ ] **Step 7: Commit any straggler**

```bash
git status --short
```
Expected: clean. If anything appears, inspect before committing.

---

## Rollback

Each task commits separately, so `git revert` handles the code. The artifacts need their own step:

```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/maintenance/migrate_skops_module_path.py --revert
```

The `.bak` sidecars survive until deleted by hand. Remove them only after the rename has been in use long enough to trust:

```bash
find outputs -name "*.skops.bak" -delete
```
