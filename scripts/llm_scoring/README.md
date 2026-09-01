# Scoring routes with LLMs


```bash
uv run --extra scoring scripts/llm_scoring/llm_scoring.py \
    data/routes/routes.csv \
    data/llm_scoring/routes_with_score_`date +%Y%m%d%H%M%S`.csv \
    --k 3 \
    --cot \
    --threads 8 \
    --limit 200 \
    --model openrouter/google/gemini-2.5-flash-lite
```

Signatures, output schemas, and the DSPy scoring modules live in `models.py`.

## Classifying PROTAC / non-PROTAC

`classify_protac.py` runs a separate, opt-in DSPy judge (`models.ProtacClassifier`)
on a plain SMILES CSV -- no route/AiZynthFinder columns required:

```bash
uv run --extra scoring scripts/llm_scoring/classify_protac.py \
    data/some_molecules.csv \
    data/llm_scoring/molecules_classified_`date +%Y%m%d%H%M%S`.csv \
    --smiles-column SMILES \
    --rationale \
    --threads 8 \
    --model openrouter/google/gemini-2.5-flash-lite
```


```
python3.12 scripts/slurm/submit_classify_protac.py \
    --in-csv data/routes/routes.csv \
    --out-csv data/llm_scoring/routes_with_classification_`date +%Y%m%d%H%M%S`.csv \
    --account berzelius-2026-62 \
    --cpus 16 \
    --threads 16 \
    --limit 100 \
    --model openrouter/google/gemini-2.5-flash-lite

python3.12 scripts/slurm/submit_llm_scoring.py \
    --in-csv data/routes/routes.csv \
    --out-csv data/llm_scoring/routes_with_score_`date +%Y%m%d%H%M%S`.csv \
    --account berzelius-2026-62 \
    --cpus 16 \
    --threads 16 \
    --k 3 \
    --model openrouter/google/gemini-2.5-flash-lite
```