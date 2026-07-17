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
```

### git-based scorers (install with `--no-deps`)

Each pins ancient dependencies in its own `setup.py` that conflict with the
stack above; `--no-deps` installs the code without disturbing the env.

```bash
pip install --no-deps --no-build-isolation git+https://github.com/lich-uct/syba.git
pip install --no-deps --no-build-isolation git+https://github.com/reymond-group/RAscore.git
pip install --no-deps --no-build-isolation git+https://github.com/schwallergroup/fsscore.git
```

## 3. Fetch models / repos that are NOT pip-installed

### RAscore model (manual copy)

`--no-deps` skips RAscore's bundled model. Clone the repo and copy the XGB model
into the installed package (`SP` below resolves to wherever `scoring_env`
actually put it, so this works regardless of machine or Python version):

```bash
git clone https://github.com/reymond-group/RAscore.git /tmp/RAscore
SP=$(python -c "import os, RAscore; print(os.path.dirname(RAscore.__file__))")
mkdir -p $SP/models
cp -r /tmp/RAscore/RAscore/models/* $SP/models/
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

Paths are resolved in `retro_scores/mol_scores/__init__.py` relative to the
project root — no edits needed if the layout matches.

### FSscore checkpoint (figshare)

The pretrained checkpoint is on figshare (not in the pip package, not gdown-able):
https://figshare.com/s/2db88a98f73e22af6868

Download the `models` folder and place the checkpoint at:

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