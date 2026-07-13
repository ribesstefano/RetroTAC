# Synthesizability Scoring — Environment Setup

This document describes how to rebuild the environment for the synthesizability
scoring pipeline (`src/protac_synth/retro_scores/`). The pipeline computes six
scores per molecule — SA score, SCScore, RAscore, SYBA, GASA, and FSscore — each
with a raw and a scaled value.

The environment is **not** fully pip-installable: three scorers need models or
repos fetched manually, several packages must be installed with `--no-deps`, and
the version pins are mutually fragile. Follow the steps in order.

> Built and tested on the Berzelius cluster (NSC), Python 3.10, project storage
> under `/proj/berzelius-2026-62/users/x_jzhuz/`.

---

## 1. Create the environment

Environments and caches must live on project storage, not `$HOME` (home quota is
small and the stack is several GB). Configure this once in `~/.condarc`:

```yaml
envs_dirs:
  - /proj/berzelius-2026-62/users/x_jzhuz/conda/envs
pkgs_dirs:
  - /proj/berzelius-2026-62/users/x_jzhuz/conda/pkgs
```

and point pip's cache off home (e.g. in your shell profile):

```bash
export PIP_CACHE_DIR=/proj/berzelius-2026-62/users/x_jzhuz/pip_cache
```

Then, each session:

```bash
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba create -n scoring_env python=3.10 -y
mamba activate scoring_env
```

> NSC advises against putting `mamba activate` / `conda init` in `.bashrc`.
> Load the module and activate manually (or in the SLURM script) instead.

---

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

---

## 3. Fetch models / repos that are NOT pip-installed

### RAscore model (manual copy)

`--no-deps` skips RAscore's bundled model. Clone the repo and copy the XGB model
into the installed package (adjust the site-packages path to your env):

```bash
git clone https://github.com/reymond-group/RAscore.git /tmp/RAscore
SP=/proj/berzelius-2026-62/users/x_jzhuz/conda/envs/scoring_env/lib/python3.10/site-packages/RAscore
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

Paths are resolved in `src/protac_synth/retro_scores/__init__.py` relative to the
project root — no edits needed if the layout matches.

### FSscore checkpoint (figshare)

The pretrained checkpoint is on figshare (not in the pip package, not gdown-able):
https://figshare.com/s/2db88a98f73e22af6868

Download the `models` folder and place the checkpoint at:

```
external/fsscore/models/pretrain_graph_GGLGGL_ep242_best_valloss.ckpt
```

(matching `FSSCORE_MODEL` in `__init__.py`).

---

## 4. Install this package (editable, no deps)

The shared `pyproject.toml` targets Python 3.12 and lists deps that don't install
under 3.10 (e.g. `protac-splitter`). Install the package code only:

```bash
cd <project-root>
pip install -e . --no-deps --no-build-isolation
```

Verify:

```bash
python -c "from protac_synth.retro_scores import compute_scores, SCORE_COLUMNS; print('ok')"
python -c "from protac_synth.retro_scores import PROJECT_ROOT; print(PROJECT_ROOT)"
```

---

## 5. Run

FSscore uses a GPU; run via SLURM on a GPU node (see
`slurm/submit_protac_scores.sh`). Input is a CSV with a SMILES column named
`molecule`; output gets the raw + scaled score columns appended.

```bash
python scripts/retrosynthesis/synthesizability_scores.py \
    data/raw/<input>.csv \
    data/synth_scores/<output>.csv \
    --smiles-col molecule
```

Notes:
- GASA runs CPU-only (hardcoded `map_location='cpu'`) — it will be the slow step.
- FSscore `num_workers` defaults to 4 to match `--cpus-per-task=4`; keep them in sync.
- Invalid / non-string SMILES are scored as NaN (row alignment is preserved).

---

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

To capture the exact working environment after a successful build:

```bash
pip freeze > requirements_scoring.txt
```