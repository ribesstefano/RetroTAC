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
PROJ=/proj/berzelius-2026-62/users/x_jzhuz/PROTAC-Synthesizability
INPUT=$PROJ/data/raw/routes_smiles_only.csv
OUTPUT=$PROJ/data/synth_scores/routes_scores.csv

# --- run -----------------------------------------------------------------
cd $PROJ
python scripts/retrosynthesis/synthesizability_scores.py \
    "$INPUT" \
    "$OUTPUT" \
    --smiles-col molecule