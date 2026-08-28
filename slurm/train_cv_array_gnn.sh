#!/bin/bash
#SBATCH --account=Berzelius-2026-62
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
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
cd $SLURM_SUBMIT_DIR

MODEL=gnn
INPUT=${INPUT:-data/sets/routes_train_val.csv}
CONFIG=${CONFIG:-config/models_config_routes.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/}
PREFIX=${PREFIX:-routes}
N_TRIALS=${N_TRIALS:-15}
CACHE_DIR=${CACHE_DIR:-outputs/feature_cache_routes}
DEVICE=${DEVICE:-cuda}   # torch.device("gpu") is invalid; xgboost/lightning both take cuda

# Seeds MUST match cross_validation.seeds in $CONFIG; folds are 0..N_FOLDS-1.
SEEDS=(42 123 456 789 1011)
N_FOLDS=5
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / N_FOLDS))]}
FOLD=$((SLURM_ARRAY_TASK_ID % N_FOLDS))

echo "task ${SLURM_ARRAY_TASK_ID}: model=${MODEL} seed=${SEED} fold=${FOLD}"

apptainer exec --nv $(bash apptainer/bind_live_repo.sh) \
    apptainer/training.sif \
    python scripts/models/train.py \
    --model "$MODEL" \
    --input "$INPUT" \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --cache-dir "$CACHE_DIR" \
    --prefix "$PREFIX" \
    --n_trials "$N_TRIALS" \
    --seed "$SEED" \
    --fold "$FOLD" \
    --device "$DEVICE" \
    --scaffold-col wh_smiles
