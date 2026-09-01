#!/bin/bash
# Run protac-splitter (adaptive model, transformer-assisted) on Berzelius CPUs.
#
# Usage:
#   sbatch scripts/slurm/run_protac_splitter.sh
#
#SBATCH -A berzelius-2026-62
#SBATCH -p berzelius-cpu
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -J protac_splitter
#SBATCH --output=logs/protac_splitter_%j.out
#SBATCH --error=logs/protac_splitter_%j.err

module load Mambaforge/23.3.1-1-hpc1-bdist
eval "$(conda shell.bash hook)"
mamba activate env-retrotac

export UV_CACHE_DIR="/proj/berzelius-2026-62/users/x_steri/.cache"

cd /proj/berzelius-2026-62/users/x_steri/PROTAC-Synthesizability

echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "CPUs        : $SLURM_CPUS_PER_TASK"

uv run --extra transformer protac-splitter \
    --input-csv tack_smiles.csv \
    --output-csv tack_smiles_split.csv \
    --smiles-col "SMILES" \
    --model adaptive \
    --adaptive-use-transformer \
    --num_proc="$SLURM_CPUS_PER_TASK"

echo "Job finished: $(date)"
