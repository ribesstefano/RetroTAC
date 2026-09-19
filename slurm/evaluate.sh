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
OUT=${OUT:-results_20260828_182305}
TEST_CSV=${TEST_CSV:-data/sets/routes_test.csv}
RANK_METRIC=${RANK_METRIC:-r2}

# Set ENSEMBLE=1 to also load every 5x5-CV fold model, predict on TEST_CSV, and
# score best_single/uniform/best_backend/Caruana ensemble strategies (see
# evaluation.py's --ensemble). Fold predictions are cached under
# {OUTPUT_ROOT}/results/{OUT}/cv_fold_test_predictions.pkl and reused on rerun
# unless FORCE_PREDICT=1.
ENSEMBLE=${ENSEMBLE:-0}
ENSEMBLE_ARGS=""
if [ "$ENSEMBLE" = "1" ]; then
    ENSEMBLE_ARGS="--ensemble"
    [ -n "$ENSEMBLE_CARUANA_FRAC" ] && ENSEMBLE_ARGS="$ENSEMBLE_ARGS --ensemble-caruana-frac $ENSEMBLE_CARUANA_FRAC"
    [ -n "$ENSEMBLE_ITERATIONS" ] && ENSEMBLE_ARGS="$ENSEMBLE_ARGS --ensemble-iterations $ENSEMBLE_ITERATIONS"
    [ "$FORCE_PREDICT" = "1" ] && ENSEMBLE_ARGS="$ENSEMBLE_ARGS --force-predict"
fi

apptainer exec --nv $(bash apptainer/bind_live_repo.sh) \
    apptainer/training.sif \
    python scripts/models/evaluation.py \
    --models $MODELS \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --out "$OUT" \
    --test-csv "$TEST_CSV" \
    --rank-metric "$RANK_METRIC" \
    $ENSEMBLE_ARGS
