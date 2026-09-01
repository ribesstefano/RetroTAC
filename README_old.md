# PROTAC Synthesizability Scoring Pipeline
> Master Thesis 2026 — Predicting and evaluating the synthesizability of PROTACs using retrosynthetic analysis and machine learning.

This pipeline decomposes PROTACs into their components (warhead, linker, E3 ligase), caps them with functional groups, scores retrosynthetic accessibility via AiZynthFinder, and trains classification/regression models to predict synthesizability.

---

## ⚠️ Large Files Notice

Large database and fragment files (`.db`, `.hdf5`, `.cxsmiles`) are **not pushed to GitHub** due to size limits.

You can find them on Alvis at:
```
/mimer/NOBACKUP/groups/naiss2023-6-290/tingtingmo/matser_thesis_2026_PROTAC_Synthesizability/data/external
```

---

## Pipeline Overview

| Step | Folder | Description |
|------|--------|-------------|
| 0 | `config` | Load stock into SQLite for faster searching |
| 1 | `data_preprocessing` | Step 1,Decompose TPDDB PROTACs into warhead / linker / E3 ligase components;Step 2,build PROTAC master table merging TPDDB and held-out set; Step3, Attachment analysis and determine the capping function groups;Step4, Cap components with functional groups; Step5, look up PubChem CIDs and check commercial availability|
| 2 | `component_synthesizability` | Step1, Run AiZynthFinder retrosynthetic routes on components; Step2, Build component master table merging vendor check and score results |
| 3 | `protac_synthesizability` | Load vendor/component stock into SQLite; run AiZynthFinder on full PROTACs; compute synthesizability scores and routes |
| 4 | `classification_model` | Build and train classifier with hyperparameter tuning |
| 5 | `regression_model` | Build and train regression model with hyperparameter tuning |
| 6 | `surrogate_model` | Train classification and regression models; evaluate and save results |

---

## Environment Setup

This project has two key non-standard dependencies that must be installed first.

### 1. AiZynthFinder

AiZynthFinder requires its own conda environment. Follow the official installation instructions:

```
https://molecularai.github.io/aizynthfinder/installation.html
```

### 2. PROTAC-Splitter

Install PROTAC-Splitter following its documentation. See `protac_splitter_requirements.txt` for the exact dependencies used in this project.

### 3. General Dependencies

For all other dependencies, refer to `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
├── data/
│   ├── external/        # Large files (not on GitHub, see Alvis path above)
│   ├── processed/       # Processed datasets, scores, model summaries
│   └── raw/             # Raw TPDDB datasets
├── figures/             # Generated plots and figures
├── notebooks/           # Jupyter notebooks for figure generation
├── src/
│   ├── config/          # Config files and SQLite stock loading
│   ├── data_preprocessing/
│   ├── component_synthesizability/
│   ├── protac_synthesizability/
│   ├── classification_model/
│   ├── regression_model/
│   ├── surrogate_model/ (WIP)
│   └── figures/
├── requirements.txt
├── protac_splitter_requirements.txt
└── test_pipeline.sh
```

---

## Notes

- Models were trained and evaluated on the **Alvis HPC cluster** (NAISS 2023/6/290)
- Hyperparameter tuning uses **Optuna**
- Experiment tracking uses **Weights & Biases (wandb)**
