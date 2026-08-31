#!/bin/bash
#SBATCH --account=Berzelius-2026-62
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH -J aggregate
#SBATCH --output=logs/models/aggregate/%x_%j.out
#SBATCH --error=logs/models/aggregate/%x_%j.err

cd $SLURM_SUBMIT_DIR

MODEL=${MODEL:-xgb}
INPUT=${INPUT:-data/sets/routes_train_val.csv}
CONFIG=${CONFIG:-config/models_config_routes.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/}
PREFIX=${PREFIX:-routes}
CACHE_DIR=${CACHE_DIR:-outputs/feature_cache_routes}
DEVICE=${DEVICE:-cuda}


apptainer exec --nv $(bash apptainer/bind_live_repo.sh) \
    apptainer/training.sif \
    python scripts/models/train.py \
    --model "$MODEL" \
    --input "$INPUT" \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --cache-dir "$CACHE_DIR" \
    --prefix "$PREFIX" \
    --device "$DEVICE" \
    --aggregate