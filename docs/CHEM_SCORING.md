# Chemistry scoring pipeline

How to score [`data/routes/routes.csv`](../data/routes/routes.csv) with the
standard cheminformatics scores (SA, NP, QED, RDKit descriptors, SCScore,
RAscore), merge those with an existing LLM-judge run, and where to take the
result for analysis. The per-script details live in
[`scripts/chem_mol_scoring/README.md`](../scripts/chem_mol_scoring/README.md)
and
[`scripts/chem_route_scoring/README.md`](../scripts/chem_route_scoring/README.md);
this page is the connective walkthrough across all three scoring stages.

## Pipeline shape

```
data/routes/routes.csv (4000 rows: SMILES, score, resolved, route, BBs, ...)
  │
  ├─→ scripts/chem_mol_scoring/molecule_scores.py   → per-target molecule scores
  │       (sa_score, np_score, qed, desc_*, scscore, rascore -- one row per route)
  │
  ├─→ scripts/chem_route_scoring/route_scores.py    → route-level scores
  │       (route_* structural metrics, route_hac_weighted_score,
  │        all_<score>_<mean|median|min|max|std> rollups across every
  │        molecule in the route)
  │
  └─→ scripts/llm_scoring/llm_scoring.py             → llm_* columns
          (already run separately -- see scripts/llm_scoring/README.md;
           an existing output is e.g. data/llm_scoring/routes_with_score_*.csv)

  all three outputs are row-aligned 1:1 with routes.csv (same rows, same
  order) → merge with:

  scripts/analysis/aggregate_route_scores.py → data/processed/analysis/routes_aggregated.csv
       → notebooks/analyze_routes.ipynb (correlations, distributions, AutoRank, ...)
```

## Step by step

```bash
# 0. One-time: convert RAscore's pretrained model (see "Score reference" below
#    for why this is a conversion, not a plain download). Skip if already done.
python scripts/chem_mol_scoring/convert_rascore_model.py

# 1. Per-target molecule scores (SA/NP/QED/descriptors/SCScore/RAscore for the SMILES column)
python scripts/chem_mol_scoring/molecule_scores.py \
    data/routes/routes.csv \
    data/processed/chem_mol_scoring/routes_molecule_scores.csv

# 2. Route-level scores: structural metrics, HAC-weighted availability, and the
#    same molecule scores rolled up across every molecule in each route
python scripts/chem_route_scoring/route_scores.py \
    data/routes/routes.csv \
    data/processed/chem_route_scoring/routes_route_scores.csv

# 3. Merge everything -- base routes.csv + both outputs above + an existing
#    LLM-scoring run -- into one wide CSV, joined by row position (routes.csv
#    has duplicate SMILES, so a SMILES-keyed join would multiply those rows)
python scripts/analysis/aggregate_route_scores.py \
    --molecule-scores data/processed/chem_mol_scoring/routes_molecule_scores.csv \
    --route-scores data/processed/chem_route_scoring/routes_route_scores.csv \
    --llm-scores data/llm_scoring/routes_with_score_20260711142743.csv \
    --output data/processed/analysis/routes_aggregated.csv
```

Step 3 only requires that every input was produced by scoring the *same*
`routes.csv` with no `--limit`/reordering -- it checks each file's `SMILES`
column against the base file's before merging and raises rather than
silently misaligning columns if they don't match. Any of `--molecule-scores`
/ `--route-scores` / `--llm-scores` can be omitted if you only want a subset.

## Compute requirements

Nothing here needs a GPU, and nothing needs SLURM:

| Stage | Compute | Measured | Needs SLURM? |
|-------|---------|----------|--------------|
| `convert_rascore_model.py` (one-time) | downloads ~52 MB, CPU-only conversion | a few seconds after the download | No |
| `molecule_scores.py` (4000 rows) | CPU only (RDKit + a small numpy MLP for SCScore + a batched XGBoost call for RAscore) | ~1m40s wall-clock on the login node | No |
| `route_scores.py` (4000 routes, ~21.8k unique molecules across target+intermediates+BBs+unresolved) | CPU only | ~5-6 min (RDKit-bound; the RAscore XGBoost call itself is <10s once batched) | No |
| `aggregate_route_scores.py` | pandas concat | sub-second | No |
| `llm_scoring.py` (already run) | I/O-bound on the judge API (Gemini/OpenRouter), not local compute | N/A (external latency, not CPU/GPU) | No, but can be slow -- see its own README for `--threads`/`--k` |

Contrast with the stage that actually *produced* `routes.csv` in the first
place (AiZynthFinder's MCTS route search, `scripts/retrosynthesis/`) -- that
one is CPU-heavy over thousands of molecules and is the pipeline stage that
genuinely benefits from a SLURM array (`scripts/slurm/submit_protac_scores.py`).
It's not part of this walkthrough since `routes.csv` already exists.

**If you touch `rascore.py`:** score XGBoost-backed models in batches
(`RAScorer.score_mols`), never one `DMatrix`/`predict()` call per molecule --
the fixed per-call overhead measured ~400x slower one at a time (165 ms vs.
0.4 ms/molecule) for this model's depth/tree count. `score_smiles_batch` in
`protac_synth.chem_scoring.scorer` already does this; it's only a trap if you
bypass that entry point.

## Score reference

| Column(s) | Section | Kind | Range | Direction |
|-----------|---------|------|-------|-----------|
| `sa_score` | `[rule_based.sa_score]` | fragment contribution (Ertl & Schuffenhauer, 2009) | ~[1, 10] | LOW = easier |
| `np_score` | `[rule_based.np_score]` | fragment contribution (Ertl et al., 2008) | ~[-5, 5] | HIGH = more natural-product-like |
| `qed` | `[rule_based.qed]` | RDKit `QED.qed` (Bickerton et al., 2012) | [0, 1] | HIGH = more drug-like |
| `desc_*` | `[rule_based.descriptors]` | plain RDKit descriptors (MolWt, TPSA, BertzCT, ...) | varies | see RDKit docs |
| `scscore` | `[ai_based.scscore]` | learned MLP (Coley et al., 2018) | (1, 5) | HIGH = more synthetically complex |
| `rascore` | `[ai_based.rascore]` | learned XGBoost classifier (Thakkar et al., 2021) | [0, 1] | HIGH = more retrosynthetically accessible |
| `route_hac_weighted_score` | `[route.hac_weighted]` | reuses `protac_synth.retro_scores.hac_score` | [0, 1] | HIGH = more heavy-atom-weighted stock coverage |
| `route_n_steps`, `route_max_depth`, `route_n_building_blocks`, `route_n_unresolved_leaves`, `route_convergence` | `[route.structural]` | read off the parsed route graph | varies | see `scripts/chem_route_scoring/README.md` |
| `all_<score>_<method>` | `[route.aggregation]` | mean/median/min/max/std of the scores above, pooled across a route's molecules | varies | same direction as the underlying score |
| `llm_score`, `llm_category`, `llm_confidence`, ... | `scripts/llm_scoring/` (separate pipeline) | LLM-judge route/molecule grading | [0, 100] | HIGH = easier/more feasible |

`config/scoring_config.toml` is the single source of truth for which scores
run and their hyperparameters (toggle a whole section off by setting
`enabled = false`).

`rascore` is trained directly on AiZynthFinder-style solved/unsolved outcomes
(Thakkar et al., Chem. Sci. 12:3339, 2021;
https://github.com/reymond-group/RAscore, MIT), which makes it the most
directly comparable *external* score to this repo's own
`route_hac_weighted_score`. Its published model predates this project's
pinned `xgboost` and no longer unpickles directly (`XGBoostLabelEncoder` was
removed upstream) -- `scripts/chem_mol_scoring/convert_rascore_model.py`
shims past that once and re-exports to XGBoost's native format, which is what
actually gets loaded at scoring time. See that script's docstring for the
full explanation and its validation against the paper's own worked examples.

## What's in the notebook

[`notebooks/analyze_routes.ipynb`](../notebooks/analyze_routes.ipynb) loads
`routes_aggregated.csv` and works through:

1. **Score distributions** -- histogram + KDE per score. Worth actually
   looking at: `scscore` is saturated at its ceiling (~5.0) for nearly this
   entire PROTAC dataset and barely discriminates, while `rascore` spreads
   fairly evenly across [0, 1] -- visible immediately in the histograms, and
   consistent with `rascore` correlating with `llm_score` (r=+0.28) far
   better than `scscore` does (r=-0.06).
2. **Cheap-vs-expensive agreement** -- `llm_score` (expensive, LLM-judged)
   vs. `sa_score`/`scscore`/`rascore`/`route_hac_weighted_score` (cheap,
   local), as a correlation table and scatter plots with fit lines.
3. **Resolved vs. unresolved** -- do the cheap scores separate
   `resolved == True/False` routes too, or does AiZynthFinder fail for
   reasons those scores don't see?
4. **Disagreement outliers** -- molecule grids for the routes where a cheap
   score and the LLM judge disagree most, in both directions.
5. **`llm_category` vs. continuous scores** -- boxplots, kept split by
   `llm_mode` (`route` vs. `molecule_only` use different category
   vocabularies entirely; merging them would be misleading).
6. **Route structure vs. score** -- step count and convergence vs. score.
7. **AutoRank** -- `sa_score`/`scscore`/`rascore`/`route_hac_weighted_score`/`llm_score`
   treated as five paired "methods" on the same resolved routes (after
   rescaling each to a common percentile so direction/units don't matter),
   testing whether their *rank distributions* differ significantly
   (Friedman + Nemenyi). Note this repo does not otherwise use `autorank`
   yet -- `src/protac_synth/models/evaluation.py` is currently a stub, despite
   being pinned as a dependency; this notebook is the first real usage. With
   four scorers the omnibus test was significant (p=0.012); adding `rascore`
   as a fifth pushed it to a borderline non-significant p=0.057 -- reported
   honestly rather than forced, since `autorank.plot_stats` refuses to draw a
   CD diagram for a non-significant result unless told to anyway
   (`allow_insignificant=True`).

Finally, a preserved/cleaned-up PROTAC vs. non-PROTAC classification QA
section (`scripts/llm_scoring/classify_protac.py`'s output), unrelated to the
synthesizability scores above but kept since a classification miss would
quietly bias any of it if used as a filter.
