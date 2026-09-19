#!/bin/bash
#SBATCH --account=Berzelius-2026-62
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH -J "protac_scores"
#SBATCH --output=logs/synth_scores/%x_%j.out
#SBATCH --error=logs/synth_scores/%x_%j.err

# --- environment ---------------------------------------------------------
module purge
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate scoring_env

# Keep FSscore's dataloader workers in line with the allocation
# (fs_score.compute defaults num_workers=4, matching --cpus-per-task above).

# --- paths ---------------------------------------------------------------
# Derived from this script's own location so it works from any checkout,
# not just the one it was authored on.
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT=$PROJ/data/raw/routes_smiles_only.csv
OUTPUT=$PROJ/data/synth_scores/routes_scores.csv

# --- run -----------------------------------------------------------------
# NOTE: SLURM opens --output/--error above before this script body runs, so
# logs/synth_scores/ must already exist — create it once before first use:
#   mkdir -p "$PROJ"/logs/synth_scores
cd $PROJ
python retro_scores/synthesizability_scores.py \
    "$INPUT" \
    "$OUTPUT" \
    --smiles-col molecule