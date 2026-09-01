#!/bin/bash
# Activate the conda env that provides Python 3.12 and uv, then install
# the project dependencies into a .venv managed by uv.
#
# First-time setup:
#   bash setup_env.sh
#
# Subsequent activations (already installed):
#   source setup_env.sh

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate env-retrotac

export UV_CACHE_DIR="/proj/berzelius-2026-62/users/x_steri/.cache"

# Install core (inference) + training + scoring dependencies from uv.lock
# into .venv (idempotent). protac-splitter's own pinned deps used to conflict
# with aizynthfinder on rdkit/networkx/pillow/urllib3, requiring a separate
# --no-deps install; aizynthfinder is dropped project-wide now, so a plain
# `uv sync --extra training` resolves cleanly on its own.
uv sync --extra training --extra scoring

# Activate the project venv so subsequent `python` calls use it.
source .venv/bin/activate
