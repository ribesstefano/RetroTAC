# PROTAC-Synthesizability Surrogate

Work in progress repository to train a surrogate model for predicting PROTAC synthesizability.

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