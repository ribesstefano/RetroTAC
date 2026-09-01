#!/usr/bin/env bash
#
# usage:
#   bash train_5x5.sh xgb data/processed/surrogate_model/full_dataset_fp_r5_512.csv myprefix
#
# smoke-test ONE fold interactively first (catches path/column/env errors fast):
#   module load Miniforge3/24.7.1-2-hpc1-bdist && mamba activate protac_synth
#   python src/protac_synth/models/train.py --model xgb \
#       --input src/protac_synth/protac_synth_data.csv \
#       --seed 0 --fold 0 --prefix testrun --n_trials 3
 
set -euo pipefail
module --force purge
# load the env so the YAML-reading python one-liner below can import yaml
module load Miniforge3/24.7.1-2-hpc1-bdist
set +u                       
mamba activate protac_synth
set -u
 
MODEL=$1
INPUT_PATH=$2
PREFIX=${3:-"default"}
N_TRIALS=${4:-25}
 
CONFIG=PROTAC-Synthesizability/src/protac_synth/models/models_config.yaml

LOG_DIR=logs/${MODEL}/${PREFIX}
 
# single source of truth: read seeds + n_folds from the YAML
SEEDS=$(python -c "import yaml; print(' '.join(map(str, yaml.safe_load(open('$CONFIG'))['cross_validation']['seeds'])))")
N_FOLDS=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['cross_validation']['n_folds'])")
 
echo "seeds: $SEEDS | n_folds: $N_FOLDS"
mkdir -p logs_test
 
# ── 0. Precompute features ONCE (tabular models only; GNN builds graphs) ────
DEP_FLAG=""
if [ "$MODEL" != "gnn" ]; then
    PRECOMPUTE_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --account=Berzelius-2026-62
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH -J "${MODEL}_${PREFIX}_precompute"
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
 
module --force purge
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate protac_synth

python PROTAC-Synthesizability/src/protac_synth/models/train.py --model $MODEL --input $INPUT_PATH --precompute
EOF
)
    echo "submitted precompute job $PRECOMPUTE_ID"
    DEP_FLAG="--dependency=afterok:$PRECOMPUTE_ID"
fi
 
# ── 1. Submit the per-fold jobs (tabular folds wait for precompute) ─────────
FOLD_JOB_IDS=()
for SEED_IDX in $SEEDS; do
    for FOLD_IDX in $(seq 0 $((N_FOLDS - 1))); do
        JID=$(sbatch --parsable $DEP_FLAG <<EOF
#!/bin/bash
#SBATCH --account=Berzelius-2026-62
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH -J "${MODEL}_s${SEED_IDX}_f${FOLD_IDX}"
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --mail-user=zhuji@chalmers.se --mail-type=fail
 
module --force purge
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate protac_synth
 
python PROTAC-Synthesizability/src/protac_synth/models/train.py --model $MODEL --input $INPUT_PATH --seed $SEED_IDX --fold $FOLD_IDX --prefix $PREFIX --n_trials $N_TRIALS
EOF
)
        FOLD_JOB_IDS+=("$JID")
        echo "submitted fold job $JID  (seed $SEED_IDX, fold $FOLD_IDX)"
    done
done
 
# ── 2. Submit aggregation, dependent on ALL 25 finishing successfully ───────
DEP=$(IFS=:; echo "${FOLD_JOB_IDS[*]}")   # colon-joined job IDs
sbatch --dependency=afterok:$DEP <<EOF
#!/bin/bash
#SBATCH --account=Berzelius-2026-62
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH -J "${MODEL}_${PREFIX}_aggregate"
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --mail-user=zhuji@chalmers.se --mail-type=end,fail
 
module --force purge
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate protac_synth
 
python PROTAC-Synthesizability/src/protac_synth/models/train.py --model $MODEL --input $INPUT_PATH --aggregate --prefix $PREFIX
EOF