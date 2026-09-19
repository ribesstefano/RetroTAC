#!/bin/bash
# Builds apptainer/deeppsa.sif in the background ON THE LOGIN NODE -- this is
# deliberately NOT an sbatch script, despite living under slurm/ for
# discoverability next to the other build/submit scripts here. Compute nodes
# reached via SLURM have no outbound internet (see CLAUDE.md's "Setup on
# Berzelius" and apptainer/README.md's build instructions), and this build
# needs several GB of conda-forge/PyPI/GitHub downloads (mamba replaying
# DeepPSA's pinned environment.yaml, plus two git clones -- see
# apptainer/deeppsa.def). `sbatch`-ing it would just fail with no network
# once it lands on a compute node.
#
# Run this ON THE LOGIN NODE with nohup so it survives your SSH session
# disconnecting overnight -- functionally the same "kick it off, check in
# the morning" workflow an sbatch job would give you, minus the part that
# would break.
#
# Usage (from the repo root):
#   nohup bash slurm/build_deeppsa_sif.sh > logs/apptainer/deeppsa_build.log 2>&1 &
#   disown
#   # check progress any time with:
#   tail -f logs/apptainer/deeppsa_build.log
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs/apptainer

echo "[$(date)] starting: apptainer build --fakeroot apptainer/deeppsa.sif apptainer/deeppsa.def"
apptainer build --fakeroot apptainer/deeppsa.sif apptainer/deeppsa.def
echo "[$(date)] build finished: $(ls -lh apptainer/deeppsa.sif)"
