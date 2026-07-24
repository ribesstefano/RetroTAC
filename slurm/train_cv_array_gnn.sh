#!/bin/bash
#SBATCH --account=Berzelius-2026-62
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --array=0-24
#SBATCH -J gnn_cv
#SBATCH --output=logs/models/gnn/%x_%A_%a.out
#SBATCH --error=logs/models/gnn/%x_%A_%a.err
#
# 5x5 nested scaffold CV for the gnn surrogate: one array task per (seed, fold).
# Array index 0-24 = 5 seeds x 5 folds (task/5 -> seed, task%5 -> fold).
# The GNN builds graphs on the fly, so no --precompute step is needed.
#
# 1. Submit this array:
#      sbatch slurm/train_cv_array_gnn.sh
# 2. When all 25 folds are done, aggregate + fit the final model:
#      .venv/bin/python scripts/models/train.py --model gnn \
#          --input data/llm_scoring/routes_llm_scores.csv --aggregate --prefix v1

set -euo pipefail

# Run from the repo root regardless of where sbatch was invoked from.
cd $SLURM_SBMIT_DIR

MODEL=gnn
INPUT=data/llm_scoring/routes_llm_scores.csv
CONFIG=config/models_config.yaml
OUTPUT_ROOT=outputs/
PREFIX=v1
N_TRIALS=20

# Seeds MUST match cross_validation.seeds in $CONFIG; folds are 0..N_FOLDS-1.
SEEDS=(42 123 456 789 1011)
N_FOLDS=5
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / N_FOLDS))]}
FOLD=$((SLURM_ARRAY_TASK_ID % N_FOLDS))

echo "task ${SLURM_ARRAY_TASK_ID}: model=${MODEL} seed=${SEED} fold=${FOLD}"

.venv/bin/python scripts/models/train.py \
    --model "$MODEL" \
    --input "$INPUT" \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --prefix "$PREFIX" \
    --n_trials "$N_TRIALS" \
    --seed "$SEED" \
    --fold "$FOLD" \
    --device gpu
