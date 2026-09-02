#!/bin/bash
#SBATCH --account=Berzelius-2026-62
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH -J predict_synthetic_data
#SBATCH --output=logs/negative_data/predict_synthetic_data.out
#SBATCH --error=logs/negative_data/predict_synthetic_data.err

set -euo pipefail

# Run from the repo root regardless of where sbatch was invoked from.
cd $SLURM_SUBMIT_DIR

apptainer exec --nv $(bash apptainer/bind_live_repo.sh) apptainer/inference.sif \
  python scripts/negative_data/predict_synthetic_data.py \
  --n-jobs $SLURM_CPUS_PER_TASK