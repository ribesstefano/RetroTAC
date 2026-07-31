#!/bin/bash
#SBATCH --account=Berzelius-2026-62
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH -J aggregate
#SBATCH --output=logs/models/aggregate/%x_%A_%a.out
#SBATCH --error=logs/models/aggregate/%x_%A_%a.err

cd $SLURM_SUBMIT_DIR

MODEL=gnn
INPUT=data/protac_synth_data.csv
CONFIG=config/models_config.yaml
OUTPUT_ROOT=outputs/
PREFIX=v1


uv run scripts/models/train.py \
    --model "$MODEL" \
    --input "$INPUT" \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --prefix "$PREFIX" \
    --device cpu \
    --aggregate