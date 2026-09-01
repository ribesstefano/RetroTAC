# Apptainer containers

Three container images, one per `pyproject.toml` optional-dependency group
(the mapping is stated at the top of `pyproject.toml`): `inference.def`
(default profile, no extra — what most users need), `training.def` (`--extra
training`), `scoring.def` (`--extra scoring` + the separate `scoring_env`
stack from `retro_scores/README.md`). All three also install `--extra
notebooks` (jupyter/jupyterlab/ipywidgets), so any of them can run a Jupyter
server for interactive development — see "Developing against a container"
below. Each builds to a single `.sif` file regardless of how many
packages/files it holds — that's the actual point: `uv sync`/`mamba create`
on Berzelius can produce tens of thousands of small files per environment,
which is what blows a `$HOME` file-count quota. A squashfs-packed `.sif` is
one file no matter what's inside it.

No `.sif` is checked into the repo (they're multi-GB binaries, and
`.gitignore` excludes `apptainer/*.sif`) — build them yourself with the
commands below; each machine/cluster should build its own anyway, since a
`.sif` embeds absolute paths and (for `training.sif`) CUDA wheels tied to
this project's `uv.lock`.

This doc has two parts: the general recipe (skip to "Adding a new profile"
if you just want to add a fourth container later) and the concrete
build/run instructions for the three that already exist.

## The general recipe

Every `.def` file here follows the same five decisions. Understanding *why*
matters more than the syntax — the "different TOML configurations" the repo
docs ask about are just different values plugged into decision 3.

**1. Base image.** Plain `ubuntu:22.04`, not an `nvidia/cuda:*` base. The
PyPI wheels this project depends on (`torch`, `xgboost`, `tensorflow`)
already vendor the CUDA runtime libraries they need; the only thing a GPU
job needs from the *host* is the driver itself, which Apptainer's `--nv`
flag bind-mounts in at run time. Baking a CUDA toolkit into the image would
be dead weight.

**1b. A few headless X11 libs (`libxrender1 libxext6 libsm6`).** Not obvious
from the Python dependency list: `rdkit.Chem.Draw` (pulled in transitively
by `protac-splitter`, and something any chemistry notebook will want for
rendering molecules) needs `libXrender`/`libXext`/`libSM` even with no
display attached — a bare `ubuntu:22.04` doesn't ship them, and the failure
mode is an opaque `ImportError: libXrender.so.1: cannot open shared object
file` the first time something touches `Draw`, not at `import rdkit` time.
Found by actually running each image after building it, not by inspection —
build and smoke-test yours the same way before trusting it.

**1c. Treat every `%post` step as if it might re-run against leftover state.**
On this system, `--fakeroot` builds share the host's real `/tmp` rather than
starting from an empty one each time (observed while building
`scoring.def`: an interrupted build's `git clone ... /tmp/RAscore` left a
non-empty directory that made the next build's identical clone fail
non-idempotently). `rm -rf` a scratch path immediately before writing to it
rather than assuming a fresh filesystem.

**1d. `pip install git+URL` doesn't reliably do what a plain `git clone`
does.** Two failure modes hit while building `scoring.def`, both invisible
until something actually *ran*, not at install time: `uv pip install
git+https://github.com/lich-uct/syba.git` silently installed a ~130-byte
Git-LFS pointer stub instead of syba's real ~115 MB model file (`git lfs
install --skip-repo` beforehand made no difference to pip's *internal*
clone; an explicit `git clone` of the same URL right next to it *did*
smudge correctly) -- the fix was to `git clone` first and `pip install
--no-deps` from that local directory instead of the remote URL. Separately,
`fsscore`'s own `pyproject.toml` under-declares its packages (`packages =
["fsscore"]` under a `src/` layout, missing `models`/`data`/`utils`), so
`--no-deps` alone left `fsscore.models` unimportable regardless of pip vs.
git — fixed by copying the missing subpackages in from a full clone after
the pip install (same shape as the already-documented RAscore
model-file gap). When a git-based dependency behaves oddly, clone it
yourself and compare, rather than assuming the packaged install is
complete.

**1e. A baked-in vendored repo that writes next to itself needs
`--writable-tmpfs` at run time, not a build fix.** `cadd-synthetic/GASA`'s
own code always writes a `results/` directory relative to its own
location — harmless when that location is a normal writable clone, fatal
(`OSError: [Errno 30] Read-only file system`) when it's baked into a `.sif`,
which is read-only by design. Since the write is disposable (prediction
output we don't need on disk), the fix is a run-time flag
(`apptainer run --writable-tmpfs ...`, an in-memory overlay for the
container's lifetime) rather than patching the vendored code or trying to
carve out one writable subdirectory at build time.

**2. `uv` installed fresh in `%post`, not pip.** `curl -LsSf
https://astral.sh/uv/install.sh | sh` gets a working `uv` without needing a
conda/mamba layer at all — one fewer moving part than the bare-metal
`mamba create` + `pip install uv` dance in `CLAUDE.md`. `uv python install
X.Y` then fetches a standalone Python build, so the image doesn't depend on
whatever Python happens to ship with the base OS.

**3. `uv sync --frozen --extra <name>` — this is the "different TOML
configuration" knob.** `--frozen` means "install exactly what `uv.lock`
already resolved, don't re-resolve" — the image is reproducible from the
lockfile alone, and the build never silently drifts from what a developer's
own `uv sync` would install. Swapping which `--extra` (or extras) you pass
here is the entire difference between `inference.def`, `training.def`, and
the main-env half of `scoring.def`. `scoring.def` additionally builds a
*second*, Python-3.10 environment by hand (`uv venv` + staged `uv pip
install`), because that stack (`retro_scores/mol_scores/`) isn't expressible
as a uv extra at all — see its own comment block in that file.

**4. The venv lives outside the code it serves — this is the one
non-obvious trick that makes everything else work.** `%files` copies a
build-time snapshot of the repo into `/opt/repo` inside the image (needed so
the editable `retrotac` install has *something* to point at), but
`UV_PROJECT_ENVIRONMENT=/opt/venv` puts the actual virtualenv at a path
**outside** `/opt/repo`. At run time, bind-mounting your live checkout over
`/opt/repo` (see "Why bind-mount instead of rebuild" below) replaces the
stale build-time snapshot with current code — pulls, edits, new scripts, all
of it — without touching `/opt/venv`, because bind-mounting a directory only
replaces *that* directory's contents, not anything installed elsewhere. If
the venv lived inside `/opt/repo` (e.g. a plain `.venv` there), the bind
mount would bury it along with the stale code.

**5. `%environment` wires up `PATH`/`VIRTUAL_ENV` so nothing needs
activating.** `apptainer exec image.sif python ...` (or `apptainer run`) Just
Works — no `source .venv/bin/activate` inside the container.

### Why bind-mount instead of rebuild

Because the point is a file-count-cheap *environment*, not a frozen
snapshot of the code. Data files, config edits, and script changes are cheap
and frequent; rebuilding a multi-GB image for each one would defeat the
purpose. `apptainer/bind_live_repo.sh` prints one `--bind` flag per
top-level repo entry (skipping `external/`, `.git/`, `.venv/`,
`__pycache__/`) so `/opt/repo` always reflects what's on disk right now:

```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/inference.sif python --version
```

`external/` is excluded on purpose: `scoring.def` bakes the SCScore/GASA
vendor code into the image at `/opt/vendor` (outside `/opt/repo` entirely,
precisely so this bind-mount pattern can't shadow it — see the comment
block in that file and `PROTAC_EXTERNAL_DIR` in
`retro_scores/mol_scores/__init__.py`).

### Adding a new profile

Two ways to parametrize a new `.def` by "TOML configuration," depending on
how many you expect:

**(a) One `.def` per profile — what this repo does.** Copy the closest
existing `.def`, change the `%files` list and the `uv sync --extra ...`
line, done. Best when you have a small, fixed, long-lived set of profiles
that get referenced by name in SLURM scripts and docs (`training.def` reads
better in a `sbatch` script than a `--build-arg` incantation would).

**(b) One templated `.def` + `--build-arg`**, for many ad hoc profiles you
don't want a file per combination for. Apptainer substitutes `{{ VARNAME
}}` anywhere in the `.def` at build time:

```
# extras.def
Bootstrap: docker
From: ubuntu:22.04
%post
    ...
    uv sync --frozen --extra {{ EXTRA }}
```

```bash
apptainer build --build-arg EXTRA=training --fakeroot custom.sif extras.def
```

Not used here since three named, stable files are easier to find and to
reference from `slurm/*.sh` than a build-arg someone has to remember.

## Building and running the three containers

Run every command below **from the repo root** — `%files` paths in a `.def`
are resolved relative to the build's working directory, not the `.def`
file's location.

```bash
apptainer build --fakeroot apptainer/inference.sif apptainer/inference.def
apptainer build --fakeroot apptainer/training.sif  apptainer/training.def
apptainer build --fakeroot apptainer/scoring.sif   apptainer/scoring.def   # slow: several GB, two git clones
```

If `--fakeroot` isn't permitted for your account (`apptainer build
--fakeroot ...` fails with a permissions error), build on a machine where
you do have root or `--fakeroot` (a laptop, a VM) and copy the resulting
`.sif` over, or use a remote build service (`apptainer build --remote`,
needs `apptainer remote login` against a build endpoint first). Berzelius'
login node has outbound internet, which is what these builds need (PyPI/git
downloads); compute nodes reached via SLURM do not, so builds must happen on
the login node or off-cluster, never inside a SLURM job.

Each `.def`'s `%help` (`apptainer run-help apptainer/<name>.sif`) has
copy-pasteable examples; the short version:

```bash
# Inference (default profile) -- also see scripts/models/predict.py --help
apptainer run --app predict $(bash apptainer/bind_live_repo.sh) \
    apptainer/inference.sif --model xgb --model-path outputs/models/xgb_default \
    --input data/new_molecules.csv --output data/scored.csv

# Training -- needs a SLURM GPU allocation + --nv (see CLAUDE.md, "Model
# training cannot run on the Berzelius login node")
srun -A berzelius-2026-62 -p berzelius --gpus=1 --cpus-per-task=16 \
    apptainer run --nv --app train $(bash apptainer/bind_live_repo.sh) \
    apptainer/training.sif --model xgb --input data.csv --seed 42 --fold 0 --prefix v1 --device cuda

# Scoring -- lightweight scorers (python 3.12)
apptainer run --app score-route $(bash apptainer/bind_live_repo.sh) \
    apptainer/scoring.sif data/routes/routes.csv data/processed/chem_route_scoring/scored.csv

# Scoring -- heavy SA/SC/RA/SYBA/GASA/FSscore stack (python 3.10; --nv for
# FSscore, --writable-tmpfs for GASA -- see "The general recipe" 1e)
apptainer run --nv --writable-tmpfs --app score-mol-heavy $(bash apptainer/bind_live_repo.sh) \
    apptainer/scoring.sif data/raw/input.csv data/synth_scores/output.csv --smiles-col molecule
```

Without `--app`, `apptainer run` uses `%runscript`, which just runs
`python "$@"` inside `/opt/repo` — handy for anything not covered by a named
app, e.g. `apptainer run $(bash apptainer/bind_live_repo.sh)
apptainer/inference.sif -c "import retrotac; print('ok')"`.

### GPU caveat that applies even to plain inference

`xgboost`'s PyPI wheel is CUDA-enabled and probes for a usable GPU even for
CPU-only work. On a host with a *present but unusable* GPU
(`compute_mode=Prohibited`, as on the Berzelius login node), this raises
`XGBoostError` — confirmed here for `.fit()` while smoke-testing
`scripts/models/train.py` on the login node, `device='cpu'` included;
`.predict()` on an already-loaded model goes through the same DMatrix
construction path, so treat plain inference as suspect on the same host
until you've checked. If you hit this, request a SLURM GPU job and add
`--nv`, the same as for training; it's a property of the host and the
wheel, not of the container.

## Developing against a container

**Code changes need no rebuild.** `bind_live_repo.sh` mounts your live
checkout over `/opt/repo`, so editing a `.py` file on the host (in VS Code,
as usual) and re-running a script or restarting a notebook kernel picks it
up immediately — the same as working outside a container. Rebuilding is
only for *dependency* changes (a new package, a bumped pin in
`pyproject.toml`/`uv.lock`).

### Launching Jupyter

All three images install `jupyter`/`jupyterlab`/`ipywidgets` (the
`notebooks` extra). Each has a `notebook` app that starts JupyterLab bound
to every interface, listening on `$JUPYTER_PORT` (default 8888) — pick
whichever image matches the code you're touching (`training.sif` for
model-development work, `scoring.sif` for the scoring pipeline, etc.):

```bash
# on the login node, or inside an salloc/srun GPU shell for GNN/torch work
JUPYTER_PORT=8899 apptainer run --app notebook $(bash apptainer/bind_live_repo.sh) \
    [--nv] apptainer/training.sif
```

Pick a **non-default port** (not 8888) since the login node is shared —
Jupyter prints the URL with a token, e.g. `http://127.0.0.1:8899/lab?token=...`.
`--notebook-dir=/opt/repo` (baked into the app) means saves land in your
repo's `notebooks/` (or wherever you navigate) exactly as if run locally —
it's the live bind mount again, not a copy inside the container.

If the job landed on a SLURM compute node rather than the login node (i.e.
you used `srun`/`salloc` for `--nv`), note the node's hostname
(`$SLURMD_NODENAME` inside the job, or `squeue -u $USER`) — you'll tunnel to
*that* node, not the login node.

### Connecting to it

**From a local browser (SSH tunnel):** on your laptop, using whatever host
you normally SSH to Berzelius through (`<login-host>` below):
```bash
ssh -L 8899:localhost:8899 x_steri@<login-host>                                   # login node
ssh -J x_steri@<login-host> -L 8899:localhost:8899 x_steri@<compute-node>          # via SLURM
```
then open the `http://127.0.0.1:8899/lab?token=...` URL Jupyter printed.

**From VS Code (recommended here, since you're already remote-attached):**
Command Palette → "Jupyter: Specify Jupyter Server for Connections" →
"Existing" → paste that same URL. Any `.ipynb` you open then runs its cells
against the container's Python (rdkit/torch/xgboost and all), while you
keep editing in VS Code exactly as now — no separate browser tab needed.
The kernel is the container's own `ipykernel` (installed as part of the
`jupyter` metapackage), so no manual kernel registration is required.

### A faster inner loop for non-notebook work

For quick one-off checks you don't need a whole Jupyter server for:
```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python -c "from retrotac.chem_utils import canon_smiles; print(canon_smiles('CCO'))"
```
Same live-mounted code, no server to manage, exits when the command does.
