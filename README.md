# PROTAC-Synthesizability Surrogate

Work in progress repository to train a surrogate model for predicting PROTAC synthesizability.

## Notes on Implementation

We do not enforce the output of the model between 0 and 1, so that if the model would predict a score outside of that range, it would mean that the input is very different from the training data.

## Reproducing

### Create Containers

**TODO**

### Add Component Splits

Join SMILES CSV with scored routes with the output from the PROTAC-Splitter, which includes the warhead, linker, and E3 components for each PROTAC. This allows us to calculate $\eta^2$ for each component and determine which one is leaking the most label information.

```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/scoring.sif \
  python scripts/dataset/add_component_splits.py data/routes/routes_scored.csv \
  data/tack/tack_smiles_split.csv \
  --output data/routes/routes_scored_with_components.csv
```

### Isolate Held-Out Set

```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/scoring.sif \
  python scripts/dataset/isolate_heldout.py data/routes/routes_scored_with_components.csv \
  --method adaptive \
  --output-dir data/sets/ \
  --make-figures --n-figure-mols 20
```

### Check Scaffold Leakage

If we use a standard random split or a whole-molecule scaffold split (which often degenerates into random splits for large PROTACs), identical warhead scaffolds will appear in both the training and test sets. The model will not learn the underlying physical or topological rules of synthesizability; it will simply memorize a lookup table (e.g., "If I detect this warhead scaffold, predict 3.8").

By calculating $\eta^2$ for the warhead, linker, and E3 separately, we identify which sub-space is leaking the most label information. We must then use that highest-$\eta^2$ sub-component to define our cross-validation folds, forcing the model to demonstrate true zero-shot generalization (scaffold hopping) on that specific substructure rather than exploiting statistical correlations in the training data.

- If $\eta^2 = 0$: The group means are identical to the grand mean. Knowing the scaffold tells us absolutely nothing about the synthesizability score. The variance is driven entirely by something else (e.g., functional group decorations that were stripped away by the scaffold generation).
- If $\eta^2 = 1$: The variance within any given group is zero. Every single PROTAC that shares a specific warhead scaffold has the exact same synthesizability score. The scaffold deterministically drives the label.

$\eta^2$ is chosen because it is the mathematically correct metric for quantifying the association between an unordered categorical independent variable (a scaffold string like "c1ccccc1") and a continuous dependent variable (a numerical synthesizability score).

In code:

```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/scoring.sif \
  python scripts/dataset/scaffold_analysis.py \
  data/sets/routes_train_val.csv \
  --combos all --combo-keys warhead linker e3 --combo-mode both \
  --targets synthesizability sa_score struct_n_steps \
  --output-dir outputs/analysis/scaffold_choice
```

Combinations are now measured rather than assumed, and neither helps — product keys leak more, union keys percolate. Warhead grouping remains the best feasible point: 0% warhead leakage, 3,341 groups, largest 2.5% of data, and it closes the highest-η² channel (0.58–0.86). The residual 78% linker / 92% E3 leakage is not fixable on this dataset without discarding most of it, so the honest move is to report it as a stated limitation.

One thing worth flagging for the thesis: warhead&e3's near-zero η² is itself informative — it says that once you remove warhead and E3 identity, almost none of the label variance survives. That's a strong statement about what your targets are actually measuring.

### Training

Precompute the feature cache for all models (XGB, MLP) before training. This is a one-time step that can be done on a login node (no GPU needed). The feature cache will be stored in `outputs/feature_cache_routes`.

```bash
# 0. one-time feature cache (login node is fine — no GPU needed)
.venv/bin/python scripts/models/train.py --model xgb \
    --input data/sets/routes_train_val.csv \
    --config config/models_config_routes.yaml \
    --cache-dir outputs/feature_cache_routes --precompute
```

```bash
# 1. all 25 (seed, fold) pairs per model — array idx/5 -> seed, idx%5 -> fold
sbatch slurm/train_cv_array_xgb.sh          # 5 seeds x 5 folds, 30 trials each
sbatch slurm/train_cv_array_mlp.sh          # 25 trials
sbatch slurm/train_cv_array_gnn.sh          # 15 trials

# 2. after all 25 folds of a model land
MODEL=xgb sbatch slurm/aggregate.sh
MODEL=mlp sbatch slurm/aggregate.sh
MODEL=gnn sbatch slurm/aggregate.sh
```



```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/scoring.sif \
  python scripts/models/train.py \
  data/sets/routes_train_val.csv \
  --config configs/models_config_.yaml \
  --scaffold-col wh_smiles \
  --prefix `date +%Y%m%d_%H%M%S` \
  --device cuda \
  --molecule-col smiles \
  --target synthesizability \

```

Each fold's val predictions get scored with:

- regression: r2, rmse, mae, medae, max_error, explained_variance, bias, pearson_r, spearman_rho, kendall_tau
- binary @ 0.7: roc_auc, pr_auc, mcc, f1, precision, recall, specificity, balanced_accuracy, accuracy, cohen_kappa, pos_rate (the PR-AUC no-skill baseline), pred_pos_rate, tp/fp/tn/fn