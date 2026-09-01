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

### Evaluation

```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
  python scripts/models/evaluation.py \
    --models xgb_20260828_182305 mlp_20260828_182305 gnn_20260828_182305 \
    --config config/models_config_routes.yaml \
    --output-root outputs \
    --out results_20260828_182305
```

With test set:

```bash
sbatch slurm/evaluate.sh

# OR:

srun -A berzelius-2026-62 -p berzelius --gpus=1 --time=00:30:00 --pty \
  apptainer exec --nv $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
  python scripts/models/evaluation.py \
    --models xgb_20260828_182305 mlp_20260828_182305 gnn_20260828_182305 \
    --config config/models_config_routes.yaml \
    --output-root outputs \
    --out results_20260828_182305 \
    --test-csv data/sets/routes_test.csv
```

#### Plotting

```bash
apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
  python scripts/models/plotting.py --results outputs/results/results_20260828_182305
```

#### Ensemble

```bash
ENSEMBLE=1 sbatch slurm/evaluate.sh

# OR:

apptainer exec --nv $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \
    python scripts/models/evaluation.py \
    --models xgb_20260828_182305 mlp_20260828_182305 gnn_20260828_182305 \
    --config config/models_config_routes.yaml \
    --output-root outputs \
    --out routes_20260828_182305 \
    --test-csv data/sets/routes_test.csv \
    --ensemble

# Rerunning cheaply once fold predictions are cached (e.g. to retune Caruana
# without reloading 75 models):
ENSEMBLE=1 ENSEMBLE_ITERATIONS=200 sbatch slurm/evaluate.sh   # skips load/predict, reuses cv_fold_test_predictions.pkl
```

## Results

```
[1/3] Loading CV fold metrics...

Loaded 3 models over 25 paired folds.
  XGB    mean r2 = 0.2374 ± 0.1210
  MLP    mean r2 = 0.2902 ± 0.1141
  GNN    mean r2 = 0.5094 ± 0.0559

── AutoRank ─────────────────────────────────────────────────────────
  Levene's test: p=0.0877 | max/min fold-variance ratio=4.6849 -> variances homogeneous
Tests for normality and homoscedacity are ignored for test selection, forcing parametric tests
     meanrank      mean       std  ...   magnitude effect_size_above magnitude_above
XGB      2.68  0.237423  0.121024  ...  negligible               0.0      negligible
MLP      2.32  0.290191  0.114066  ...       small         -0.448728           small
GNN      1.00  0.509414  0.055914  ...       large         -2.440518           large

[3 rows x 9 columns]
The statistical analysis was conducted for 3 populations with 25 paired samples.
The family-wise significance level of the tests is alpha=0.050.
We rejected the null hypothesis that the population is normal for the population MLP (p=0.001). Therefore, we assume that not all populations are normal.
Because we have more than two populations and the populations and one of them is not normal, we should use the non-parametric Friedman test as omnibus test to determine if there are any significant differences between the median values of the populations and report the median (MD) and the median absolute deviation (MAD). However, the user decided to force the use of repeated measures ANOVA as omnibus test which assume homoscedascity to determine if there are any significant difference between the mean values of the populations. If the results of the ANOVA test are significant, we use the post-hoc Tukey HSD test to infer which differences are significant. We report the mean value (M) and the standard deviation (SD) for each population. Populations are significantly different if their confidence intervals are not overlapping.
We reject the null hypothesis (p=0.000) of the repeated measures ANOVA that there is a difference between the mean values of the populations XGB (M=0.237+-0.034, SD=0.121), MLP (M=0.290+-0.034, SD=0.114), and GNN (M=0.509+-0.034, SD=0.056). Therefore, we assume that there is a statistically significant difference between the mean values of the populations.
Based on post-hoc Tukey HSD test, we assume that all differences between the populations are significant.

[2/3] Evaluating final models on held-out test set (smiles_col='smiles', target_col='synthesizability')...
  XGB    r2=0.441  rmse=0.165  mae=0.091  spearman_rho=0.661  clf_roc_auc=0.811
  MLP    r2=0.467  rmse=0.161  mae=0.087  spearman_rho=0.665  clf_roc_auc=0.809
  GNN    r2=0.536  rmse=0.150  mae=0.095  spearman_rho=0.666  clf_roc_auc=0.824

[2b/3] Predicting test set with every 5x5-CV fold model...
  Found cached fold predictions at outputs/results/results_20260828_182305/cv_fold_test_predictions.pkl -- skipping load/predict.

[2c/3] Scoring ensemble strategies...

  Caruana selection split: 463 selection / 1854 evaluation samples (every strategy is scored on the 1854-sample evaluation split)
  Backend RMSE (uniform avg of all its fold models): GNN=0.1332, MLP=0.1437, XGB=0.1567 -> best_backend picks GNN

── Ensemble strategies (scored on the shared evaluation split) ─────────
  best_single  n_models=1   rmse=0.1378  r2=0.6160  composition={'GNN': 1}
  uniform      n_models=75  rmse=0.1346  r2=0.6333  composition={'XGB': 25, 'MLP': 25, 'GNN': 25}
  best_backend n_models=25  rmse=0.1332  r2=0.6413  composition={'GNN': 25}
  caruana      n_models=27  rmse=0.1320  r2=0.6475  composition={'GNN': 13, 'MLP': 7, 'XGB': 7}

              n_models           composition    rmse     r2  mean_std  unc_err_rho  cov_1sig_%  cov_2sig_%  cov_3sig_%  ece_%  mce_%
strategy                                                                                                                            
best_single          1                 GNN:1  0.1378  0.616       NaN          NaN         NaN         NaN         NaN   38.7   52.3
uniform             75  XGB:25,MLP:25,GNN:25  0.1346  0.633    0.0693        0.332        46.9        70.3        83.7   37.3   52.2
best_backend        25                GNN:25  0.1332  0.641    0.0481        0.369        38.2        66.6        82.0   37.5   52.5
caruana             27    GNN:13,MLP:7,XGB:7  0.1320  0.647    0.0675        0.337        47.0        72.9        84.4   36.5   52.6
  (expected coverage: 68.3% / 95.5% / 99.7% at 1σ/2σ/3σ; NaN = undefined for a 1-model ensemble; ECE/MCE treat the ensemble's raw prediction as P(synthesizable))
```