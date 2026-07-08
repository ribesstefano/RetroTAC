# Scoring routes with LLMs


```bash
uv run --extra llm_scoring scripts/llm_scoring/llm_scoring.py \
    data/routes/routes.csv \
    data/llm_scoring/routes_with_score_`date +%Y%m%d%H%M%S`.csv \
    --k 3 \
    --cot \
    --threads 4 \
    --limit 20 \
    --model openrouter/google/gemini-2.5-flash-lite
```