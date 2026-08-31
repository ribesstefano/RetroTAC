#!/bin/bash
#SBATCH --account=Berzelius-2026-62
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH -J evaluate
#SBATCH --output=logs/models/evaluate/%x_%j.out
#SBATCH --error=logs/models/evaluate/%x_%j.err

cd $SLURM_SUBMIT_DIR

MODELS=${MODELS:-"xgb_20260828_182305 mlp_20260828_182305 gnn_20260828_182305"}
CONFIG=${CONFIG:-config/models_config_routes.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
OUT=${OUT:-routes_20260828_182305}
TEST_CSV=${TEST_CSV:-data/sets/routes_test.csv}
RANK_METRIC=${RANK_METRIC:-r2}

apptainer exec --nv $(bash apptainer/bind_live_repo.sh) \
    apptainer/training.sif \
    python scripts/models/evaluation.py \
    --models $MODELS \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --out "$OUT" \
    --test-csv "$TEST_CSV" \
    --rank-metric "$RANK_METRIC"
