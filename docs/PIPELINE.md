# Pipeline reference

A complete, per-script reference for the PROTAC synthesizability pipeline. The
[README](../README.md) gives the quick-start and the high-level stage tables;
**this document is the deep dive** — every module under [`src/`](../src), its
role, inputs, outputs, and the order in which it runs within its stage.

The pipeline has three thesis stages, realised as five `src/` areas:

| Thesis stage | `src/` area | Orchestrator |
|--------------|-------------|--------------|
| 1. Data preparation | [`data_preprocessing/`](../src/data_preprocessing) | `run_data_pipeline.py` |
| 1–2. Component retrosynthesis | [`component_synthesizability/`](../src/component_synthesizability) | `run_data_pipeline.py` |
| 2. Whole-PROTAC retrosynthesis | [`protac_synthesizability/`](../src/protac_synthesizability) | `run_data_pipeline.py` |
| 3. Surrogate — classification | [`classification_model/`](../src/classification_model) | `run_models.py` |
| 3. Surrogate — regression | [`regression_model/`](../src/regression_model) | `run_models.py` |

## How to read these tables

- **#** — execution order *within the stage*.
- **Kind** —
  - `entrypoint` : a CLI script with `argparse`, invoked by the orchestrator.
  - `script` : runs its work at import time (module-level constants for I/O);
    invoked as `python <file>` with no arguments.
  - `library` : imported by other modules; never run directly.
- **Key** — the orchestrator stage key (`--only <key>`), where one exists.

All paths are project-relative and centralized in
[`src/config/config.py`](../src/config/config.py); the orchestrator mirrors them
in [`scripts/pipeline/paths.py`](../scripts/pipeline/paths.py).

---

## Stage 1 — Data preparation

> *Methods §"Data Preparation"* — curate, standardize, decompose, characterize
> attachment environments, and cap the components. Directory:
> [`src/data_preprocessing/`](../src/data_preprocessing).

| # | Key | Script | Kind | Inputs → Outputs | Purpose |
|---|-----|--------|------|------------------|---------|
| 1 | `split` | [tpddb_split.py](../src/data_preprocessing/tpddb_split.py) | script | `raw/tpddb_protacs(2).csv`, `raw/dataset-curated-held-out.csv` → `raw/tpddb_split_raw_complete.csv` | Remove held-out overlap, run **PROTAC-Splitter** to mark the two attachment points with dummy atoms. *Slow — run once.* |
| 2 | `master` | [build_protac_master_table.py](../src/data_preprocessing/build_protac_master_table.py) | script | split + held-out → `processed/protac_smiles_master_std.csv` | RDKit standardize/canonicalize; assemble the master table with per-component SMILES + IDs (warhead/linker/E3). |
| 3 | `attachment` | [attachment_analysis.py](../src/data_preprocessing/attachment_analysis.py) | entrypoint¹ | master → `processed/analysis/attachment_side_classified.csv`, `figures/attachment_env.png` | Classify each attachment environment via SMARTS (≤2 bonds); justifies the chosen cap set. **Needs PROTAC-Splitter.** |
| 4 | `cap` | [capping_component.py](../src/data_preprocessing/capping_component.py) | entrypoint¹ | master → `processed/component_capped.csv` | Replace each dummy atom with **H, OH, CH₃, NH₂, =O, COOH**; RDKit-validate. Linker gets all 36 cap-pair combinations. |
| 5 | `vendor_components` | [comp_cid_vendor_check.py](../src/data_preprocessing/comp_cid_vendor_check.py) | entrypoint¹ | capped → `processed/component_smiles_to_cid.csv`, `processed/component_check_cid_vendor.csv` | PubChem CID lookup + vendor availability for capped components (PUG REST, rate-limited). |

¹ Has an `if __name__ == "__main__"` guard but takes no CLI arguments — I/O
paths come from `config.py`.

---

## Stage 2 — Component-level synthesizability

> *Methods §"Component-level Synthesizability"* — score each capped component;
> a PROTAC is *component-level solved* if at least one cap combination makes all
> three components individually solved. Directory:
> [`src/component_synthesizability/`](../src/component_synthesizability).

| # | Key | Script | Kind | Inputs → Outputs | Purpose |
|---|-----|--------|------|------------------|---------|
| 1 | `component_scores` | [component_routes_scores.py](../src/component_synthesizability/component_routes_scores.py) | entrypoint | `processed/component_capped.csv`, `aizynthfinder_stock.db` → `processed/scores/components/<target>_cap_type_<cap>_results_{summary,precursors,steps}.csv` | Run **AiZynthFinder** per `(component, cap-type)`. Filter flags: `--target {warhead,e3,linker}`, `--filter_val <cap>`. The orchestrator expands all 48 combos automatically. Resumable. |
| 2 | `component_master` | [build_comp_master.py](../src/component_synthesizability/build_comp_master.py) | script | capped + component scores dir + vendor CSV → `processed/component_master.csv` | Merge AiZynth scores + HAC score + PubChem vendor hits; compute `aizynth_solved_tag`, `pubchem_vendor_tag`, `final_solved_tag`. |

> The attachment-environment analysis lives only in **Stage 1**
> (`data_preprocessing/attachment_analysis.py`); a stale hardcoded-path copy
> previously duplicated here was removed.

---

## Stage 3 — Whole-PROTAC synthesizability

> *Methods §"Whole-PROTAC Synthesizability"* — build the augmented stock,
> select a diverse tuning subset, score whole molecules, and run the three
> stock-configuration experiments. Directory:
> [`src/protac_synthesizability/`](../src/protac_synthesizability), now grouped
> into `stock/` (DB builders) and `route_analysis/` (diagnostics).

### 3a. Build the AiZynthFinder stock database — [`stock/`](../src/stock)

These three builders populate `data/external/aizynthfinder_stock.db`, whose
tables are read by `get_scores.py`. They are independent and can run in any
order; load the building-block sources once, then the component/vendor tables.

| # | Key | Script | Kind | Inputs → Outputs | Purpose |
|---|-----|--------|------|------------------|---------|
| 1 | — | [stock/cxsmiles_to_sqlite.py](../src/stock/cxsmiles_to_sqlite.py) | entrypoint | `external/*.cxsmiles`, `<table>` → DB table | Load ZINC / Enamine REAL building-block subsets (one table per subset). |
| 2 | `load_components` | [stock/components_to_sqlite.py](../src/stock/components_to_sqlite.py) | entrypoint | `processed/component_master.csv` → `warhead`/`e3_ligase`/`linker` tables | Load **solved components** (`final_solved_tag` ∈ {1,2,3}) as supplementary stock. |
| 3 | `load_vendor` | [stock/csv_to_sqlite.py](../src/stock/csv_to_sqlite.py) | entrypoint | `processed/component_master.csv` → `vendor_stock`, `solved_stock` tables | Load the PubChem-vendor stock and the AiZynth-solved stock. |

### 3b. Tuning subset, scoring, and stock augmentation (top level)

| # | Key | Script | Kind | Inputs → Outputs | Purpose |
|---|-----|--------|------|------------------|---------|
| 4 | `select_diverse` | [select_diverse_protacs.py](../src/protac_synthesizability/select_diverse_protacs.py) | entrypoint | `processed/protac_smiles_master_std.csv` → `processed/selected_diverse_protacs.csv` | Pick *k* diverse PROTACs for Optuna tuning. Score = 0.5·FP + 0.3·functional-group + 0.2·component diversity (Morgan r3/2048, greedy MaxMin, τ=0.85). |
| 5 | `protac_scores` | [get_scores.py](../src/protac_synthesizability/get_scores.py) | entrypoint | `protac_smiles_master_std.csv`, stock DB → `processed/scores/protacs/exp_<cfg>{.json,_summary.csv,_precursors.csv,_steps.csv}` | Run **AiZynthFinder** on whole PROTACs → HAC-weighted score. Resumable JSONL checkpoint; SLURM array via `--total_tasks/--task_id`. |
| 6 | `merge` | [merge_results.py](../src/protac_synthesizability/merge_results.py) | entrypoint | `scores/protacs/*_task*` → merged per-prefix files | Consolidate per-array-task score outputs. |
| 7 | — | [filter_unsolved_precursors.py](../src/protac_synthesizability/filter_unsolved_precursors.py) | entrypoint | scores JSON/JSONL → CSV of out-of-stock precursor SMILES | Extract precursors still missing after a run, for PubChem cross-check (feeds the *component-augmented* experiment). |
| 8 | `vendor_precursors` | [smiles_vendor_check.py](../src/protac_synthesizability/smiles_vendor_check.py) | entrypoint | precursor SMILES CSV → CID CSV + vendor CSV + summary | PubChem CID + vendor status for the unsolved precursors; add confirmed ones back to stock and re-run. |

**Stock-configuration experiments** (run `protac_scores` with the matching DB
contents; `--experiment` names the output prefix):
`baseline` (ZINC + Enamine) · `component_augmented` (+ solved-component & vendor
stock) · `component_only` (component & vendor stock alone).

### 3c. Route diagnostics — [`route_analysis/`](../src/protac_synthesizability/route_analysis) (optional)

| Script | Kind | Inputs → Outputs | Purpose |
|--------|------|------------------|---------|
| [route_analysis/analyze_routes.py](../src/protac_synthesizability/route_analysis/analyze_routes.py) | entrypoint | scores JSON/JSONL → printed statistics | Split solved/unsolved; report mean/median/percentiles per metric. |
| [route_analysis/plot_distributions.py](../src/protac_synthesizability/route_analysis/plot_distributions.py) | entrypoint | scores JSON/JSONL/CSV → score & search-time histograms | Compare AiZynth vs HAC score and search time for solved vs unsolved. |

---

## Stage 4 — Classification surrogate

> *Methods §"Surrogate Model Development"* — binary "synthesizable vs not"
> (`solved_tag`), optimized for PR-AUC. Directory:
> [`src/classification_model/`](../src/classification_model).

| Key | Script | Kind | Inputs → Outputs | Purpose |
|-----|--------|------|------------------|---------|
| — | [classification_model.py](../src/classification_model/classification_model.py) | library | — | sklearn-compatible RF / XGB / MLP classifiers that take raw SMILES and build Morgan FP + RDKit descriptors internally. |
| `train_cls` | [train_classifier.py](../src/classification_model/train_classifier.py) | entrypoint | `exp2_protacs_solved_tag.csv` → saved model + `optuna_trials_*.csv` + test predictions | Single-model training: scaffold `GroupKFold` + Optuna. Flags: `--target`, `--n-trials`, `--downsample {random,scaffold_stratified}` (seeds 42/123/456), `--generic-scaffold`. |
| — | [cl_hyp_tuning_5x5.py](../src/classification_model/cl_hyp_tuning_5x5.py) | entrypoint | features CSV → per-fold PR-AUC | 5×5 repeated nested scaffold CV (parallel via `--seed/--fold`), the classification analogue of the regression `train.py`. |
| — | [evaluation_plot.py](../src/classification_model/evaluation_plot.py) | script | classifier outputs → plots | Threshold analysis & evaluation plots (default 0.5 vs F1-optimal). |

---

## Stage 5 — Regression surrogate

> *Methods §"Surrogate Model Development"* — predict the continuous
> `hac_weighted_score`; primary metric R² (plus MAE/RMSE, AutoRank). Directory:
> [`src/regression_model/`](../src/regression_model). 5×5 nested scaffold CV with
> inner Optuna tuning.

**Entrypoints (run in this order):**

| # | Key | Script | Inputs → Outputs | Purpose |
|---|-----|--------|------------------|---------|
| 0 | `filter` | [filter_similar_protacs.py](../src/regression_model/filter_similar_protacs.py) | `exp2_protacs_solved_tag.csv` → `exp2_protacs_filtered.csv`, `tanimoto_similarities.csv` | *(filtered experiment only)* drop unsolved molecules with Tanimoto > 0.7 to a conflicting solved one. |
| 1 | `features` | [compute_fingerprints_descriptors.py](../src/regression_model/compute_fingerprints_descriptors.py) | scored CSV (`molecule` col) → `surrogate_model/<stem>_fp_r<R>_<N>.csv` | Append Morgan FP (`fp_*`) + RDKit descriptors (`desc_*`). |
| 2 | `train_reg` | [train.py](../src/regression_model/train.py) | features CSV → per-fold JSON + `experiments/regression/<prefix>.yml` + saved model | Drive 5×5 CV. Sequential (all 25 folds + aggregate) or one fold via `--seed/--fold`; `--aggregate` retrains the final model and writes the experiment YAML. |
| 3 | `evaluate_reg` | [evaluation.py](../src/regression_model/evaluation.py) | YAML + saved models + split → `results/regression/<prefix>/{test_metrics.csv, autorank_*.png, *.pkl}` | Held-out test metrics + **AutoRank** critical-difference comparison across configs. |
| — | — | [fragment_enrichment.py](../src/regression_model/fragment_enrichment.py) | filter outputs → enrichment table | Fisher's-exact functional-group enrichment of filtered-out vs kept molecules. |

**Library modules** (imported by the entrypoints; never run directly):

| Module | Responsibility |
|--------|----------------|
| [config_model.py](../src/regression_model/config_model.py) | feature flags, FP radius/size, `TARGET`, `CV_SEEDS`, W&B project, study prefix. |
| [data_utils.py](../src/regression_model/data_utils.py) | load CSV, derive `solved_tag`/SA score, compute scaffolds, build/load the fixed scaffold split. |
| [scaffold_utils.py](../src/regression_model/scaffold_utils.py) | Bemis–Murcko scaffolds + test-first scaffold train/test split. |
| [cross_validation.py](../src/regression_model/cross_validation.py) | deterministic 5×5 fold indices (`GroupKFold` by scaffold). |
| [model_builders.py](../src/regression_model/model_builders.py) | RF / XGB / MLP builders that map an Optuna trial → a fitted model. |
| [regressor_model.py](../src/regression_model/regressor_model.py) | sklearn-compatible regressors taking raw SMILES (FP + descriptors + optional SVD). |
| [tuning.py](../src/regression_model/tuning.py) | inner Optuna tuning + outer fold eval; aggregation + YAML export. |

---

## Shared infrastructure — [`src/config/`](../src/config)

| File | Kind | Purpose |
|------|------|---------|
| [config.py](../src/config/config.py) | library | All project paths (`RAW`, `PROCESSED`, `SCORES`, `EXTERNAL`, `SURROGATE_*`, `OPTUNA_DB`, …); creates missing dirs on import. |
| [config.yml](../src/config/config.yml) | data | AiZynthFinder MCTS config: USPTO expansion/filter models, ZINC stock, search limits. |
| [sqlite_stock.py](../src/stock/sqlite_stock.py) | library | `SQLiteStock` adapter exposing a SQLite table as an AiZynthFinder stock. |

## Figures — [`src/figures/`](../src/figures)

Thesis-figure scripts (component bottleneck analysis, solved bar plots,
component-vs-PROTAC overlap Venn). These still carry hardcoded Alvis paths and
are run ad hoc, outside the orchestrators.

## Orchestration package — [`scripts/pipeline/`](../scripts/pipeline)

| Module | Responsibility |
|--------|----------------|
| `paths.py` | every filesystem location, mirroring `config.py`. |
| `core.py` | `Stage` / `Pipeline` data model (pure, no I/O). |
| `cli.py` | shared flags + the subprocess runner. |
| `data_stages.py` | `data_pipeline()` — Stage 1–3 stages. |
| `model_stages.py` | `model_pipeline()` — Stage 4–5 stages. |

Add a step by appending one `Stage(...)` to the relevant `*_stages.py` module.
