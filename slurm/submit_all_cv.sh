#!/usr/bin/env bash
#
# slurm/submit_all_cv.sh
# ======================
# Submit the FULL 5x5 nested scaffold CV for every surrogate in one shot:
# 3 models x 5 seeds x 5 folds = 75 independent SLURM array tasks, one job per
# combination, all eligible to run in parallel.
#
# Wiring handled here (the reason not to just run three sbatch calls by hand):
#   * one shared --precompute job, with the xgb/mlp arrays held behind it via
#     --dependency=afterok so the 50 tabular tasks can't race to write the
#     fp/desc .npz caches; the gnn array builds graphs on the fly and starts
#     immediately.
#   * per-model trial budgets (the search spaces differ in size).
#   * one aggregate job per model, each held behind its own array.
#
# Usage:
#   bash slurm/submit_all_cv.sh                       # all 3 models, defaults
#   N_TRIALS_XGB=40 bash slurm/submit_all_cv.sh       # override one budget
#   MODELS="xgb mlp" bash slurm/submit_all_cv.sh      # subset of models
#   MAX_CONCURRENT=10 bash slurm/submit_all_cv.sh     # throttle each array
#   DRY_RUN=1 bash slurm/submit_all_cv.sh             # print, don't submit
set -euo pipefail

# ── shared run config (every value overridable from the environment) ────────
INPUT=${INPUT:-data/sets/routes_train_val.csv}
CONFIG=${CONFIG:-config/models_config_routes.yaml}
PREFIX=${PREFIX:-$(date +%Y%m%d_%H%M%S)}   # computed once, shared by every job below
CACHE_DIR=${CACHE_DIR:-outputs/feature_cache_routes}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/}
DEVICE=${DEVICE:-cuda}
MODELS=${MODELS:-"xgb mlp gnn"}

# ── per-model Optuna budgets, sized to each build_* search space ────────────
# xgb: 8 tuned dims | mlp: 6 (+ MedianPruner) | gnn: 4 (frozen backbone, no pruning)
# Floor is ~15: TPESampler's default n_startup_trials=10 samples randomly first.
N_TRIALS_XGB=${N_TRIALS_XGB:-30}
N_TRIALS_MLP=${N_TRIALS_MLP:-25}
N_TRIALS_GNN=${N_TRIALS_GNN:-15}

# Optional per-array concurrency cap, e.g. MAX_CONCURRENT=10 -> --array=0-24%10
THROTTLE=""
[ -n "${MAX_CONCURRENT:-}" ] && THROTTLE="%${MAX_CONCURRENT}"

DRY_RUN=${DRY_RUN:-0}
submit() {                       # echo the command, then run it unless DRY_RUN
    echo "+ sbatch $*" >&2
    if [ "$DRY_RUN" = "1" ]; then echo "DRYRUN_$RANDOM"; else sbatch "$@"; fi
}

mkdir -p logs/models/xgb logs/models/mlp logs/models/gnn logs/models/aggregate

COMMON="INPUT=$INPUT,CONFIG=$CONFIG,PREFIX=$PREFIX,CACHE_DIR=$CACHE_DIR"
COMMON="$COMMON,OUTPUT_ROOT=$OUTPUT_ROOT,DEVICE=$DEVICE"

echo "input=$INPUT  config=$CONFIG  prefix=$PREFIX  models=[$MODELS]"

# ── 0. shared feature cache, once, only if a tabular model is in play ───────
# The cache is keyed by fp params alone (never by dataset), so a dedicated
# CACHE_DIR per dataset is what keeps a stale-row-count cache from colliding.
DEP_TABULAR=""
if [[ " $MODELS " == *" xgb "* || " $MODELS " == *" mlp "* ]]; then
    PRE_ID=$(submit --parsable \
        --account=Berzelius-2026-62 --partition=berzelius --gpus=1 \
        --time=2:00:00 --cpus-per-task=8 -J "precompute_${PREFIX}" \
        --output=logs/models/%x_%j.out --error=logs/models/%x_%j.err \
        --wrap="cd \$SLURM_SUBMIT_DIR && apptainer exec \
                  \$(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
                  python scripts/models/train.py \
                  --model xgb --input $INPUT --config $CONFIG \
                  --cache-dir $CACHE_DIR --precompute")
    echo "precompute -> job $PRE_ID"
    DEP_TABULAR="--dependency=afterok:$PRE_ID"
fi

# ── 1. one 25-task array per model (idx/5 -> seed, idx%5 -> fold) ───────────
for MODEL in $MODELS; do
    case $MODEL in
        xgb) N_TRIALS=$N_TRIALS_XGB; DEP=$DEP_TABULAR ;;
        mlp) N_TRIALS=$N_TRIALS_MLP; DEP=$DEP_TABULAR ;;
        gnn) N_TRIALS=$N_TRIALS_GNN; DEP="" ;;          # no cache -> no dependency
        *)   echo "unknown model: $MODEL" >&2; exit 1 ;;
    esac

    ARRAY_ID=$(submit --parsable ${DEP:+$DEP} \
        --array="0-24${THROTTLE}" \
        --export="ALL,${COMMON},N_TRIALS=${N_TRIALS}" \
        "slurm/train_cv_array_${MODEL}.sh")
    echo "${MODEL}: 25 folds x ${N_TRIALS} trials -> array job $ARRAY_ID"

    # ── 2. aggregate + final refit, held behind that model's whole array ────
    AGG_ID=$(submit --parsable --dependency="afterok:$ARRAY_ID" \
        --export="ALL,${COMMON},MODEL=${MODEL}" \
        -J "aggregate_${MODEL}_${PREFIX}" \
        slurm/aggregate.sh)
    echo "${MODEL}: aggregate -> job $AGG_ID"
done
