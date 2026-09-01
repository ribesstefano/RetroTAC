# Route analysis (optional diagnostics)

Post-hoc tooling for inspecting the AiZynthFinder output produced by
`get_scores.py`. These scripts are **not part of the core pipeline** — they help
understand and visualize the scored routes (e.g. for the thesis results). They
are not wired into the orchestrator.

A molecule is treated as **solved** if every leaf precursor of its best route is
in stock, and **unsolved** otherwise.

## Scripts

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| [analyze_routes.py](analyze_routes.py) | scores `.json` / `.jsonl` | printed statistics | Per-group (solved/unsolved) summary stats — mean, median, std, min/max, p90, p95 — for each metric. |
| [plot_distributions.py](plot_distributions.py) | scores `.json` / `.jsonl` / `.csv` | histogram figures | Compare AiZynth vs HAC score and MCTS search time across solved/unsolved groups. |

## Examples

```bash
python src/protac_synthesizability/route_analysis/analyze_routes.py \
    data/processed/scores/protacs/exp_baseline.json

python src/protac_synthesizability/route_analysis/plot_distributions.py \
    data/processed/scores/protacs/exp_baseline.json
```
