# Contributing Guidelines

## Project Architecture & Configuration

* **Configuration File Placement:** Store YAML configuration files in a top-level `config/` directory rather than within the source codebase.
* **Immutable Source Code:** Applications must accept configuration paths as input arguments. Users should never be required to modify source files (which may reside in virtual environments) to change configurations.
* **Model Weights and Checkpoint File Placement:** Store checkpoint files in a top-level dedicated directory rather than within the source codebase. The directory shall be ignored by Git.
* **Dependency Management:** Keep the `.toml` file updated when introducing new packages. Categorize dependencies appropriately (e.g., separating core inference requirements from development and testing tools) to allow for environment-specific installations.
* **Cache Management:** Output cache files to a dedicated, user-configurable directory. Avoid generating cache files in the same directories as input data (e.g., CSV folders) to prevent clutter.
* **Hardware Agnosticism:** Do not hardcode hardware-specific parameters (such as CUDA device IDs in XGBoost or similar libraries). These must always be exposed as user-configurable arguments.

## Code Quality & Formatting

* **Type Hinting:** Define explicit type signatures for all function arguments and return types.
* **Linting:** Utilize `pylint` continuously to enforce quality standards and maintain a high code quality score.
* **Standard Formatting:** Adhere to standard spacing rules (e.g., PEP 8 for Python). Avoid custom or non-standard visual alignments, such as aligning `=` assignment operators across consecutive lines.

## Function Design & State Management

* **Variable Scope:** Minimize the use of global variables, including global configuration objects. Pass necessary state and parameters directly via function arguments.
* **Sensible Defaults:** Provide default values for function arguments where appropriate to streamline API usage and allow for concise function calls.

## Documentation & Reproducibility

* **Reproducibility:** Design all code and workflows to be fully reproducible. Maintain comprehensive guidelines and tutorials to assist users in replicating results.
* **Inline Documentation:** Supplement standard function docstrings with fine-grained, inline comments. Provide explanatory context for substantial logical blocks or complex operations within the code.

## Development

## Environment Setup

On Berzelius, to create the environment:

``bash
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba create -n env-protac-synth python=3.12 -y
mamba activate env-protac-synth
pip install uv
```

Export the `uv` to a proper location and activate the environment:

```bash
export UV_CACHE_DIR="/proj/berzelius-2026-62/users/x_steri/.cache"
mamba activate env-protac-synth
uv sync --extra dev --extra retrosynth # For Stefano and Andrea, training and retrosynthesis stuff
uv sync --all-extras # For Lukas, training only
```

### Training

Example:

```bash
uv run python scripts/models/train.py \
  --model xgb \
  --input data/llm_scoring/routes_llm_scores.csv \
  --config config/models_config.yaml \
  --output-root outputs/ \
  --prefix PREFIX \
  --n_trials 5 \
  --seed 42 \
  --fold 1
```

To specify the SMILES and label columns, update the config file. GNN pretrained files can be obtained via:

```bash
wget https://zenodo.org/records/15460715/files/chemeleon_mp.pt
```

To test SLURM jobs on Berzelius:

```bash
srun --account=Berzelius-2026-62 --partition=berzelius --gpus=1 --cpus-per-task=16  python scripts/models/train.py   --model xgb   --input data/llm_scoring/routes_llm_scores.csv   --config config/models_config.yaml   --output-root outputs/   --prefix PREFIX   --n_trials 2   --seed 42   --fold 1 --device gpu

srun --account=Berzelius-2026-62 --partition=berzelius --gpus=1 --cpus-per-task=16  python scripts/models/train.py   --model mlp   --input data/llm_scoring/routes_llm_scores.csv   --config config/models_config.yaml   --output-root outputs/   --prefix PREFIX   --n_trials 2   --seed 42   --fold 1 --device gpu

srun --account=Berzelius-2026-62 --partition=berzelius --gpus=1 --cpus-per-task=16  python scripts/models/train.py   --model gnn   --input data/llm_scoring/routes_llm_scores.csv   --config config/models_config.yaml   --output-root outputs/   --prefix PREFIX   --n_trials 2   --seed 42   --fold 2 --device gpu
```

Change the PREFIX to anything useful to keep track of the runs.

To run a job array on SLURM, run:

```bash
sbatch slurm/train_cv_array_xgb.sh
sbatch slurm/train_cv_array_mlp.sh
sbatch slurm/train_cv_array_gnn.sh
```

Logs will saved under: `logs/models/<xgb|gnn|mlp>/`

## Tree route scoring 
Tree routes' scores can be computed as follows:
```bash
python retro_scores/route_scores/route_tree_score.py \\
      --input data/raw/routes.csv \\
      --config config/route_scoring.yaml \\
      --output data/outputs/routes_scored.csv \\
      --sep '\\t'
```
It takes SMILES, resolved and route as input columns. The output file contains the original columns with the scoring metrics, weighted synthesizability score and scoring notes. 
The input columns and scoring weights can be modified in config/route_scoring.yaml.
