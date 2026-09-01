# Refactoring

## Questions

- How was the `aizynthfinder_stock.db` generated?
- For components, what's the difference between the CSV results: `_step`, `_precursors`, `_summary`?

## Prompt

```
Refactor the file to make it more elegant, concise, and maintainable.
Please read the file, plan your changes, and update the code directly applying these constraints:
1. Architecture:
- Wrap the primary execution logic in a main() function.
- Include a standard if __name__ == '__main__': block.
2. Dynamic Inputs:
- Eliminate all hardcoded configurations, paths, and magic numbers.
- Implement argparse to handle these as command-line arguments, including sensible defaults and clear help= descriptions.
3. Maintainability:
- Add strict Type Hints (PEP 484) to all function parameters and return types.
- Write clear, concise docstrings for all functions.
- Use idiomatic Python (e.g., use pathlib for file operations, use context managers).
4. Execution:
- Briefly summarize the structural changes you plan to make.
- Apply the refactor to the file.
5. Verification (Crucial):
- After applying your edits, run a linter (like ruff check or flake8) and a type checker (like mypy) on the file using the terminal.
- Once saved, run python -m py_compile [filename.py] to verify there are no syntax errors.
- Fix any errors or critical warnings you find before telling me the task is complete. Briefly summarize the structural changes you made.
```

Under Maintainability: `Extract any long or complex blocks of logic into smaller, single-responsibility helper functions.`

## Data Preprocessing

The file `src/data_preprocessing/tpddb_split.py` filter the SMILES for duplicates and calls PROTAC-Splitter. It can be replaced by calling the PROTAC-Splitter in the command line:

```bash
protac-splitter \
    --input_csv data/raw/tack_smiles.csv \
    --model "heuristic->xgboost" \
    --num_proc=8 \
    --output_csv data/processed/tack_split.csv
```

Build a single table with all PROTACs and their components

```bash
python src/data_preprocessing/build_protac_master_table.py \
    --held-out data/raw/dataset-curated-held-out.csv \
    --tack data/processed/tack_split.csv \
    --output data/processed/protac_smiles_master_std.csv
```

**TODO**: The column `protac_smiles_index` could likely be dropped (it's always zero, why?)

The following is more of an analysis, not required by the pipeline:

```bash
python src/data_preprocessing/protac_smiles_master_std.py \
    --input attachment_side_classified \
    --output data/processed/component_capped.csv
```

The following caps the components for running them on AiZynthFinder:

```bash
python src/data_preprocessing/capping_component.py \
    --input data/processed/protac_smiles_master_std.csv \
    --output data/processed/component_capped.csv
```

Resulting columns and examples in `component_capped.csv`:

- component_id: `WH_4c795e13`, ...
- component_smiles_with_dummy: `COc1cc2c(OC[C@@H]3CCC(=O)N3)ncc([*:1])c2cc1C(N)=O`
- cap_type: `H`, `OH`, `CH3`, `NH2`, `COOH`, ...
- cap_smiles: `COc1cc2c(OC[C@@H]3CCC(=O)N3)nccc2cc1C(N)=O`

Example:

```
component_id,component_smiles_with_dummy,cap_type,cap_smiles
WH_4c795e13,COc1cc2c(OC[C@@H]3CCC(=O)N3)ncc([*:1])c2cc1C(N)=O,H,COc1cc2c(OC[C@@H]3CCC(=O)N3)nccc2cc1C(N)=O
WH_4c795e13,COc1cc2c(OC[C@@H]3CCC(=O)N3)ncc([*:1])c2cc1C(N)=O,OH,COc1cc2c(OC[C@@H]3CCC(=O)N3)ncc(O)c2cc1C(N)=O
WH_4c795e13,COc1cc2c(OC[C@@H]3CCC(=O)N3)ncc([*:1])c2cc1C(N)=O,CH3,COc1cc2c(OC[C@@H]3CCC(=O)N3)ncc(C)c2cc1C(N)=O
WH_4c795e13,COc1cc2c(OC[C@@H]3CCC(=O)N3)ncc([*:1])c2cc1C(N)=O,NH2,COc1cc2c(OC[C@@H]3CCC(=O)N3)ncc(N)c2cc1C(N)=O
WH_4c795e13,COc1cc2c(OC[C@@H]3CCC(=O)N3)ncc([*:1])c2cc1C(N)=O,COOH,COc1cc2c(OC[C@@H]3CCC(=O)N3)ncc(C(=O)O)c2cc1C(N)=O
```

To get the PubChem CID:

```bash
python scripts/submit_vendor_check.py \
    --account Berzelius-2026-62 \
    --partition=berzelius-cpu \
    --mail ribes@chalmers.se \
    --input "/proj/berzelius-2026-62/users/x_steri/master_thesis_2026_PROTAC_Synthesizability/data/preprocessed/component_capped.csv" \
    --output "/proj/berzelius-2026-62/users/x_steri/master_thesis_2026_PROTAC_Synthesizability/data/preprocessed/component_cid_vendor_actual.csv" \
    --cid-cache "/proj/berzelius-2026-62/users/x_steri/master_thesis_2026_PROTAC_Synthesizability/data/preprocessed/cid_cache.csv" \
    --vendor-cache "/proj/berzelius-2026-62/users/x_steri/master_thesis_2026_PROTAC_Synthesizability/data/preprocessed/vendor_cache.csv" \
    --time "72:00:00" \
    --retries 5
```

## Running AiZynthFinder

- Created non-source `config` directory for AiZynthFinder configuration
- Moved template files (ONNX and CSV) under `data/templates`
- We need three stocks and so three different `db` files:
    - ZINC (default)
    - ZINC + Enamine Real
    - ZINC + Enamine Real + Components

### On Components

```bash
python src/component_synthesizability/get_component_routes_and_scores.py \
    --config config/aizynthfinder_component_config.yaml \
    --stock_db data/external/aizynthfinder_stock.db \
    --csv data/preprocessed/component_capped.csv \
    --component_type warhead \
    --outdir results/components \
    --smiles_col cap_smiles \
    --cap H \
    --chunk_size 4 \
    --limit 8

python src/component_synthesizability/get_component_routes_and_scores.py \
    --config config/aizynthfinder_component_config.yaml \
    --stock_db data/external/aizynthfinder_stock.db \
    --csv data/preprocessed/component_capped.csv \
    --component_type e3 \
    --outdir results/components \
    --smiles_col cap_smiles \
    --cap H \
    --chunk_size 4 \
    --limit 8

python src/component_synthesizability/get_component_routes_and_scores.py \
    --config config/aizynthfinder_component_config.yaml \
    --stock_db data/external/aizynthfinder_stock.db \
    --csv data/preprocessed/component_capped.csv \
    --component_type linker \
    --outdir results/components \
    --smiles_col cap_smiles \
    --cap H \
    --chunk_size 4 \
    --limit 8
```

Running via SLURM in one go:

```bash
python scripts/submit_component_scores.py \
    --csv data/preprocessed/component_capped.csv \
    --config config/aizynthfinder_component_config.yaml \
    --stock_db data/external/aizynthfinder_stock.db \
    --outdir results/components \
    --account berzelius-2026-62 \
    --chunk_size 2000 \
    --time 72:00:00 \
    --mail ribes@chalmers.se
```

### On PROTACs

```bash
python scripts/submit_protac_scores.py \
    --csv data/preprocessed/protac_master.csv \
    --smiles_col protac_smiles \
    --config config/aizynthfinder_protac_config.yaml \
    --stock_db data/external/aizynthfinder_stock.db \
    --outdir results/protacs \
    --account berzelius-2026-62 \
    --chunk_size 2000 \
    --time 72:00:00 \
    --mail ribes@chalmers.se
```



The files @scripts/retrosynthesis/get_component_routes_and_scores.py  and @scripts/retrosynthesis/get_protac_routes_and_scores.py  share similar functions to handle AiZynthFinder. Please write shared functions in a single file @src/aizynthfinder_utils.py , then  update the scripts in order to leverage them. Finally, double check the scripts under @scripts/slurm/ shall be updated or could be simplified.


* insalata
* pomodori
* Melanzane x4
* peperoni
* banane
* melone / pesche / kiwi
* vegofars
* tofu x2
* pane
* pomodori barattolo
* tonno
* dado carne rossa
* uova
* assorbenti con ali 3/5
* anticalcare spray dirk (il piu cheap)
* dentifricio