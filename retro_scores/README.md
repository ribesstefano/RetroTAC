# Synthesizability Scoring — Environment Setup

This document describes how to rebuild the environment for the synthesizability
scoring pipeline (`retro_scores/`, a standalone package installed separately
from the main `src/protac_synth/` codebase — see "Why separate?" below). The
pipeline computes six scores per molecule — SA score, SCScore, RAscore, SYBA,
GASA, and FSscore — each with a raw and a scaled value.

### Why separate?

`src/protac_synth/` (installed via `uv sync`, Python 3.12) trains and runs
inference for the surrogate model. This scoring pipeline needs a completely
different, fragile stack (TF 2.8 + torch 2.0 + dgl 2.1 + old xgboost, Python
3.10) that has nothing to do with that. Keeping it in its own package with
its own `pyproject.toml` means installing or breaking `scoring_env` never
touches the main environment's dependency constraints, and vice versa.

The environment is **not** fully pip-installable: three scorers need models or
repos fetched manually, several packages must be installed with `--no-deps`, and
the version pins are mutually fragile. Follow the steps in order.

> Built and tested on Python 3.10, Linux, with a CUDA-capable GPU (required by
> FSscore). Any machine with `conda`/`mamba` and ~5 GB of free space works —
> HPC-specific notes are called out separately where they apply.

## 1. Create the environment

```bash
mamba create -n scoring_env python=3.10 -y
mamba activate scoring_env
```

(`conda` works the same way if you don't have `mamba`; it's just slower to
solve. If your machine has neither, install Miniforge first.)

> **If you're on an HPC cluster with a quota-limited `$HOME`:** this stack is
> several GB once built, and pip's cache adds more on top. If you have a
> larger project/scratch storage area, point conda and pip at it *before*
> creating the environment:
> ```yaml
> # ~/.condarc
> envs_dirs: [/path/to/your/storage/conda/envs]
> pkgs_dirs: [/path/to/your/storage/conda/pkgs]
> ```
> ```bash
> export PIP_CACHE_DIR=/path/to/your/storage/pip_cache
> ```
> Load `conda`/`mamba` however your site provides it (a module system, a
> local install, etc.) — check your site's docs for whether `mamba activate`
> belongs in `.bashrc` or should be run per-session; some clusters advise
> against the former.

## 2. Install the Python stack (staged)

A flat `pip install -r` does not solve reliably for this combination
(TF 2.8 + torch 2.0 + dgl 2.1 + old xgboost). Install in this order:

```bash
pip install numpy==1.24.3 protobuf==3.20.0
pip install tensorflow==2.8.0 keras==2.8.0
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install dgl==2.1.0 dgllife==0.3.2
pip install xgboost==1.0.2 scikit-learn==1.7.2 rdkit==2026.3.3
pip install pytorch-lightning==2.0.2 lightning==2.0.7
pip install pandas==2.3.3 gdown==6.1.0

# pkg_resources fix: pytorch-lightning imports it at runtime; setuptools>=81
# removed it. Pin below 81.
pip install "setuptools<81"

# torchdata: dgl 2.1's graphbolt imports torchdata.datapipes, removed in newer
# torchdata. 0.6.0 keeps it. If a scorer pulls a newer one, re-pin this.
pip install "torchdata==0.6.0"

# fsscore.models.ranknet imports this directly; fsscore's own pyproject.toml
# pins this exact version too. Easy to miss since fsscore itself installs
# with --no-deps below and *appears* to succeed -- the gap only shows up the
# first time something actually calls fs_score.compute().
pip install torch-geometric==2.3.0
```

### git-based scorers (install with `--no-deps`)

Each pins ancient dependencies in its own `setup.py` that conflict with the
stack above; `--no-deps` installs the code without disturbing the env.

```bash
pip install --no-deps --no-build-isolation git+https://github.com/lich-uct/syba.git
pip install --no-deps --no-build-isolation git+https://github.com/reymond-group/RAscore.git
pip install --no-deps --no-build-isolation git+https://github.com/schwallergroup/fsscore.git
```

**syba's model files are Git-LFS-tracked** (`resources/syba.csv.gz`,
`resources/syba4.csv.gz`, ~115 MB). `pip install git+...` (the syba line
above) checks out the ~130-byte LFS pointer stub instead of the real file,
even with `git-lfs` installed and registered first -- confirmed that
`git lfs install --skip-repo` beforehand does NOT make pip's own internal
git clone smudge correctly, only an actual `git clone` command does.
`SybaClassifier().fitDefaultScore()` then fails with a gzip error (`Not a
gzipped file (b've')`, i.e. it read the pointer text's `"version ..."` line
instead of gzip magic bytes). Install `git-lfs`, register the filter
globally, then clone syba yourself and install from that local copy instead
of the remote URL (replaces the plain `pip install git+...` line above for
syba specifically):
```bash
sudo apt-get install git-lfs   # or your package manager's equivalent
git lfs install --skip-repo

SYBA_TMP=$(mktemp -d)
git clone https://github.com/lich-uct/syba.git "$SYBA_TMP"
pip install --no-deps --no-build-isolation "$SYBA_TMP"
rm -rf "$SYBA_TMP"
```
Verify after installing: `python -c "import syba, os; print(os.path.getsize(os.path.join(os.path.dirname(syba.__file__), 'resources', 'syba4.csv.gz')))"`
should print ~115000000-ish, not ~134.

**fsscore's own `pyproject.toml` under-declares its packages** (`packages =
["fsscore"]` under a `src/` layout, missing the `models`/`data`/`utils`
subpackages it actually ships), so `pip install --no-deps` above only
installs `fsscore/__init__.py` -- `from fsscore.models.ranknet import
LitRankNet` then fails with `ModuleNotFoundError: No module named
'fsscore.models'`. Fill in the missing subpackages from a full clone (see
"FSscore checkpoint" below, which folds this into one step since both need
the same clone).

## 3. Fetch models / repos that are NOT pip-installed

### RAscore model (manual copy)

`--no-deps` skips RAscore's bundled model. Clone the repo and copy the XGB model
into the installed package (`SP` below resolves to wherever `scoring_env`
actually put it, so this works regardless of machine or Python version).
Clone into a private `mktemp -d` directory, not a fixed path like
`/tmp/RAscore` -- on a shared HPC login node, `/tmp` is shared across users,
and a predictable name there can collide with another user who followed
these same instructions (hit this for real on Berzelius: another user's own
leftover `/tmp/RAscore` broke a from-scratch build using this exact path):

```bash
RASCORE_TMP=$(mktemp -d)
git clone https://github.com/reymond-group/RAscore.git "$RASCORE_TMP"
SP=$(python -c "import os, RAscore; print(os.path.dirname(RAscore.__file__))")
mkdir -p $SP/models
cp -r "$RASCORE_TMP/RAscore/models/"* $SP/models/
rm -rf "$RASCORE_TMP"
# verify model.pkl is a real ~9.7 MB file, not an LFS stub:
ls -la $SP/models/XGB_chembl_ecfp_counts/model.pkl
```

Test: `python -c "from RAscore import RAscore_XGB; print(RAscore_XGB.RAScorerXGB().predict('CCO'))"`
(should print ~0.99).

### SCScore & GASA (run from cloned repos)

These are not pip packages; the code is imported by filesystem path. Clone into
`external/` at the project root:

```bash
cd <project-root>/external
git clone https://github.com/CatSci/SCScore.git   # CatSci fork ships the .json.gz weights the code loads
git clone https://github.com/cadd-synthetic/GASA.git   # ships gasa.pth in GASA/model/
```

**Upstream bug in `cadd-synthetic/GASA`:** `model/data.py`'s `mkdir_p()`
catches `OSError` and checks `exc.errno == errno.EEXIST` without ever
`import errno` -- raises `NameError: name 'errno' is not defined` instead
of the intended "directory already exists, continue" handling, and it hits
this on every run (the target directory always already exists). One-line
patch after cloning:
```bash
sed -i '1i import errno' external/GASA/model/data.py
```

Paths are resolved in `retro_scores/mol_scores/__init__.py` relative to the
project root — no edits needed if the layout matches.

### FSscore: missing subpackages + checkpoint (one clone fixes both)

Despite the `figshare` link this section used to point at, the pretrained
checkpoint is checked directly into the `schwallergroup/fsscore` repo
(`models/*.ckpt`, ~1.4 MB, a real file, not a Git-LFS pointer) -- no
figshare account or manual download needed. The same clone also fixes the
missing-subpackages issue noted above (`pip install --no-deps` only gets
`fsscore/__init__.py`). Use a private `mktemp -d`, not a fixed path, for the
same shared-`/tmp` reason as RAscore above:

```bash
FSSCORE_TMP=$(mktemp -d)
git clone https://github.com/schwallergroup/fsscore.git "$FSSCORE_TMP"
FSP=$(python -c "import os, fsscore; print(os.path.dirname(fsscore.__file__))")
cp -r "$FSSCORE_TMP/src/fsscore/." "$FSP/"          # fills in models/, data/, utils/
mkdir -p external/fsscore/models
cp "$FSSCORE_TMP/models/pretrain_graph_GGLGGL_ep242_best_valloss.ckpt" external/fsscore/models/
rm -rf "$FSSCORE_TMP"
```

The checkpoint must still end up at:

```
external/fsscore/models/pretrain_graph_GGLGGL_ep242_best_valloss.ckpt
```

(matching `FSSCORE_MODEL` in `__init__.py`).

## 4. Install this package (editable)

`retro_scores/` is a standalone project with its own `pyproject.toml` — it
does not depend on `protac_synth` or the main repo's `pyproject.toml` at all,
so installing it never touches the main Python-3.12 environment. It provides
two importable packages: `mol_scores` (the six scorers above, what this doc
is about) and `route_scores` (AiZynthFinder-route-based scoring, e.g. the
HAC-weighted score — needs `aizynthfinder`, so it's meant for the main
`protac_synth` environment rather than `scoring_env`; see its own module
docstring).

```bash
cd <project-root>/retro_scores
pip install -e . --no-build-isolation
```

Verify:

```bash
python -c "from mol_scores import compute_scores, SCORE_COLUMNS; print('ok')"
python -c "from mol_scores import PROJECT_ROOT; print(PROJECT_ROOT)"
```

## 5. Run

FSscore needs a GPU; everything else runs on CPU. Input is a CSV with a SMILES
column named `molecule` by default (override with `--smiles-col`); output gets
the raw + scaled score columns appended.

```bash
python scripts/retrosynthesis/synthesizability_scores.py \
    data/raw/<input>.csv \
    data/synth_scores/<output>.csv \
    --smiles-col molecule
```

On a SLURM cluster, `slurm/submit_protac_scores.sh` is a ready-to-adapt job
script — it derives all paths from its own location, so only the
`#SBATCH --account`/`--partition` lines and the input/output paths need
editing.

Notes:
- GASA runs CPU-only (hardcoded `map_location='cpu'`) — it will be the slow step.
- FSscore's `--cpus-per-task` (or however your scheduler limits CPUs) should
  match `num_workers` (default 4) passed to the scorer, so the dataloader
  doesn't oversubscribe.
- Invalid / non-string SMILES are scored as NaN (row alignment is preserved).

## Version pins that matter (why they're there)

| Package             | Pin        | Reason                                             |
|---------------------|------------|----------------------------------------------------|
| xgboost             | 1.0.2      | RAscore's model.pkl was pickled with it            |
| torchdata           | 0.6.0      | dgl 2.1 graphbolt imports `torchdata.datapipes`    |
| setuptools          | <81        | pytorch-lightning imports `pkg_resources`          |
| numpy               | 1.24.3     | stack-wide compatibility                           |
| protobuf            | 3.20.0     | tensorflow 2.8 compatibility                       |
| tensorflow / keras  | 2.8.0      | GASA / RAscore NN backends                          |
| torch               | 2.0.0      | fsscore / dgl compatibility                         |
| dgl / dgllife       | 2.1.0/0.3.2| GASA                                               |

If you need to debug a version conflict, `pip list` inside `scoring_env` shows
what's actually installed — this table is the curated subset that matters;
avoid committing a full `pip freeze` dump (a past one leaked a GitHub token
via an editable self-install — see `git log` for `requirements_scoring.txt`
if you need the history).