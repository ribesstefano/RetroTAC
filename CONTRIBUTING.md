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
uv sync --extra training # For Stefano and Andrea, training stuff
uv sync --all-extras # For Lukas, training only
```

If `uv sync`/`mamba create` blow your `$HOME` file-count quota (each env is tens of
thousands of small files), use the prebuilt Apptainer containers instead —
see `apptainer/README.md`. There's one per profile above (`inference`,
`training`, `scoring`), each a single `.sif` file regardless of how many
packages it holds.

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

### Cross-validation scheme

Training uses **5×5 nested, scaffold-grouped cross-validation**: 5 outer seeds
(`cross_validation.seeds` in `config/models_config.yaml`) × 5 outer folds
(`cross_validation.n_folds`) = 25 independent `(seed, fold_idx)` fits. Each pair is
one CLI invocation of `scripts/models/train.py` (see "Training" above), typically
one SLURM array task (`slurm/train_cv_array_*.sh`). Grouping by Bemis–Murcko
scaffold guarantees no scaffold spans a train/val boundary, at either nesting level.

#### Pipeline

```mermaid
flowchart TD
    IN["Input CSV<br/>(SMILES + target column)"] --> PRE["--precompute<br/>standardize SMILES once,<br/>cache Morgan FP + RDKit descriptors<br/>outputs/feature_cache/*.npz"]
    PRE --> LOOP["25 x (seed, fold_idx) runs<br/>5 seeds x 5 folds, one SLURM<br/>array task each"]

    subgraph ONEFOLD["run_single_fold(seed, fold_idx)"]
        direction TB
        S1["get_fold_indices()<br/>scaffold-grouped outer split"] --> S2["tune_inner_fold()<br/>Optuna TPE search,<br/>objective = R2 on inner-val"]
        S2 --> S3["Refit best params<br/>on full outer-train"]
        S3 --> S4["Score refit model<br/>on outer-val -> R2"]
        S4 --> S5["Persist score_seed(S)_fold(F).json,<br/>model_seed(S)_fold(F),<br/>trials_seed(S)_fold(F).csv,<br/>best_params_seed(S)_fold(F).json"]
    end

    LOOP --> ONEFOLD
    ONEFOLD -->|repeat for all 25 seed/fold pairs| AGG["--aggregate<br/>once all 25 score JSONs exist"]
    AGG --> MEAN["mean +/- std R2<br/>over 25 outer-val scores"]
    AGG --> PICK["pick hyperparams with best<br/>best_r2_inner (never outer)"]
    PICK --> REFIT["Refit final model<br/>on the FULL dataset"]
    REFIT --> OUT["outputs/models/model_prefix/<br/>model_prefix_final<br/>+ model_prefix_hparams.yaml"]
```

#### Fold schematic — outer rotation

For one outer seed, scaffolds are shuffled with a seeded RNG then greedily
balanced into 5 groups `G0..G4`; each `fold_idx` in turn holds out a different
group as outer-val. All 5 seeds repeat this with an independent shuffle, giving
25 total outer-val evaluations.

```mermaid
flowchart TB
    classDef val fill:#f28e8e,stroke:#333,color:#000;
    classDef train fill:#8ecae6,stroke:#333,color:#000;

    subgraph SEED["one outer seed: shuffle scaffolds, balance into G0..G4"]
        direction LR
        subgraph FI0["fold_idx = 0"]
            direction TB
            A0[G0]:::val
            A1[G1]:::train
            A2[G2]:::train
            A3[G3]:::train
            A4[G4]:::train
        end
        subgraph FI1["fold_idx = 1"]
            direction TB
            B0[G0]:::train
            B1[G1]:::val
            B2[G2]:::train
            B3[G3]:::train
            B4[G4]:::train
        end
        subgraph FI2["fold_idx = 2"]
            direction TB
            C0[G0]:::train
            C1[G1]:::train
            C2[G2]:::val
            C3[G3]:::train
            C4[G4]:::train
        end
        subgraph FI3["fold_idx = 3"]
            direction TB
            D0[G0]:::train
            D1[G1]:::train
            D2[G2]:::train
            D3[G3]:::val
            D4[G4]:::train
        end
        subgraph FI4["fold_idx = 4"]
            direction TB
            E0[G0]:::train
            E1[G1]:::train
            E2[G2]:::train
            E3[G3]:::train
            E4[G4]:::val
        end
    end
```

#### Fold schematic — inner split (per outer fold)

The outer-train from the rotation above (4 of the 5 groups) is shuffled *again*
with an independent seed (`seed + fold_idx + 1000`) and split ~80/20 by scaffold
into inner-train/inner-val, used only by Optuna. The held-out outer-val group is
never touched during tuning — it only re-enters at the final scoring step.

```mermaid
flowchart LR
    OT["Outer-train<br/>4 of 5 scaffold groups"] --> SH2["Shuffle scaffolds again<br/>seed + fold_idx + 1000"]
    SH2 --> SPLIT["Greedily balance ~80/20 by scaffold"]
    SPLIT --> ITR["Inner-train (~80%)"]
    SPLIT --> IVAL["Inner-val (~20%)"]
    ITR --> OPT["Optuna study<br/>TPE sampler + MedianPruner<br/>n_trials trials"]
    IVAL --> OPT
    OPT --> BEST["best_params<br/>+ best_r2_inner"]
    BEST --> REFIT2["Refit on full outer-train"]
    OV["Outer-val<br/>held-out scaffold group"] --> SCORE["Evaluate refit model"]
    REFIT2 --> SCORE
    SCORE --> R2OUT["R2 -> score_seed(S)_fold(F).json"]
```

Once all 25 `score_seed{S}_fold{F}.json` files exist, `--aggregate`:

1. reports mean ± std R² across the 25 outer-val scores;
2. picks the hyperparameters with the best **inner** R² (never outer — avoids
   leakage from picking on a test-like split);
3. refits one final model on the **full** dataset with those hyperparameters;
4. saves the final model plus a `_hparams.yaml` CV summary under
   `outputs/models/<model>_<prefix>/`.

#### Output layout

Example for `--model xgb --prefix v1 --output-root outputs` (`outputs/` is
gitignored). File extensions on `model_seed*_fold*` / `<run_id>_final` depend on
the backend: xgb saves `.skops` + `.ubj`, mlp saves `.skops` + `.pt`, gnn saves a
single `.ckpt`. Per-fold models are omitted entirely with `--no-save-fold-models`.

**After one fold** (`--seed 42 --fold 0`, feature cache from a prior `--precompute`):

```
outputs/
├── feature_cache/                          # model-agnostic, built once via --precompute
│   ├── fp_r3_512.npz                       # cached Morgan fingerprints (named by fp_radius/fp_size)
│   └── desc.npz                            # cached RDKit descriptors
├── cv/
│   └── xgb_v1/                             # run_id = <model>_<prefix>
│       ├── xgb_v1_seed42_fold0.db          # Optuna SQLite study (resumable snapshot)
│       ├── trials_seed42_fold0.csv         # Optuna trials dataframe
│       ├── best_params_seed42_fold0.json   # best inner-val hyperparameters + best_r2_inner
│       ├── model_seed42_fold0.skops        # refit fold model - sklearn wrapper
│       ├── model_seed42_fold0.ubj          # refit fold model - XGBoost native (xgb only)
│       └── score_seed42_fold0.json         # {"seed": 42, "fold_idx": 0, "r2": ...}
└── models/                                 # empty until --aggregate
```

**After all 25 folds + `--aggregate`:**

```
outputs/
├── feature_cache/
│   ├── fp_r3_512.npz
│   └── desc.npz
├── cv/
│   └── xgb_v1/
│       ├── xgb_v1_seed{0..4}_fold{0..4}.db          # 25 Optuna studies
│       ├── trials_seed{0..4}_fold{0..4}.csv         # 25 trials dataframes
│       ├── best_params_seed{0..4}_fold{0..4}.json   # 25 best-param files
│       ├── model_seed{0..4}_fold{0..4}.{skops,ubj}  # 25 refit fold models
│       └── score_seed{0..4}_fold{0..4}.json         # 25 outer-val R2 scores
└── models/
    └── xgb_v1/
        ├── xgb_v1_final.skops    # final model, refit on the FULL dataset
        ├── xgb_v1_final.ubj      # (xgb only; mlp: .skops+.pt, gnn: single .ckpt)
        └── xgb_v1_hparams.yaml   # chosen hyperparams + cv_mean_r2/cv_std_r2/best_inner_r2/n_folds
```

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
The input columns and scoring weights can be modified in `config/route_scoring.yaml`.
