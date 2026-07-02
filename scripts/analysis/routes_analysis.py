"""
analyze_routes.py
=================
Reads a newline-delimited JSON (JSONL) or standard JSON file produced by
the AiZynthFinder scoring pipeline, splits molecules into *solved* and
*unsolved* groups, and computes descriptive statistics for each group.

A molecule is considered **solved** if every one of its leaf precursors is
flagged as in-stock (``in_stock: true``). Otherwise it is **unsolved**.

Statistics computed (per group, per metric)
-------------------------------------------
mean, median, standard deviation, min, max, 90th percentile, 95th percentile

Metrics analysed
----------------
- ``old_score``    : original AiZynthFinder route score (``route_score``).
- ``new_score``    : HAC-weighted availability score (``hac_weighted_score``).
- ``search_time``  : wall-clock time of the MCTS search in seconds.
- ``route_length`` : number of reaction steps in the best route.

Input JSON format
-----------------
Each record must be a JSON object with at least the following keys::

    {
        "target_smiles":      "<SMILES string>",
        "route_score":        <float>,
        "hac_weighted_score": <float>,
        "search_time_seconds": <float>,
        "route_length":       <int>,          # optional; falls back to len(synthesis_steps)
        "precursors": [
            {"precursor_smiles": "<SMILES>", "in_stock": <bool>, ...},
            ...
        ],
        "synthesis_steps": [...]
    }

The file may be either a JSON array ``[{...}, {...}]`` or newline-delimited
JSON (one record per line), as produced by ``get_scores_new_v2.py``.

Output files  (written to the directory defined by ``config.ANALYSIS``)
-----------------------------------------------------------------------
``<stem>_analysis.csv``
    Machine-readable statistics table. One row per (group, metric) pair.
    Columns: group, metric, mean, median, std, min, max, p90, p95.

``<stem>_analysis.txt``
    Human-readable report with pretty-printed tables, mirrored to stdout
    during execution.

Usage
-----
    python analyze_routes.py path/to/routes.json

Arguments
---------
    input   Path to the input JSON or JSONL file (required, positional).
"""

import argparse
import csv
import json
import sys
import numpy as np
from tabulate import tabulate
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# I/O HELPERS
# ============================================================

def load_routes(json_path: str) -> list:
    """
    Load route records from a JSON or JSONL file.

    Tries to parse the file as a standard JSON document first. If that fails
    (e.g. the file is newline-delimited JSON), falls back to parsing each
    non-empty line independently. Malformed lines raise an exception.

    Parameters
    ----------
    json_path : str
        Path to the input file.

    Returns
    -------
    list[dict]
        List of route record dictionaries.
    """
    with open(json_path, "r") as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


# ============================================================
# STATISTICS HELPER
# ============================================================

def stats(values: list) -> dict:
    """
    Compute descriptive statistics for a list of numeric values.

    Parameters
    ----------
    values : list[float]
        Input values. May be empty.

    Returns
    -------
    dict
        Keys: ``mean``, ``median``, ``std``, ``min``, ``max``, ``p90``, ``p95``.
        All values are ``None`` if the input list is empty.
    """
    values = list(values)

    if not values:
        return {k: None for k in ("mean", "median", "std", "min", "max", "p90", "p95")}

    return {
        "mean":   float(np.mean(values)),
        "median": float(np.median(values)),
        "std":    float(np.std(values)),
        "min":    float(np.min(values)),
        "max":    float(np.max(values)),
        "p90":    float(np.percentile(values, 90)),
        "p95":    float(np.percentile(values, 95)),
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_routes(json_path: str) -> dict:
    """
    Split routes into solved/unsolved groups and compute statistics for each.

    A route is **solved** if every precursor has ``in_stock == True``.

    Parameters
    ----------
    json_path : str
        Path to the JSON or JSONL input file.

    Returns
    -------
    dict
        Summary dictionary with the following keys:

        - ``n_molecules``        : total number of routes.
        - ``solved_molecules``   : count of fully solved routes.
        - ``unsolved_molecules`` : count of unsolved routes.
        - ``solved_ratio``       : fraction of solved routes.
        - ``unsolved_ratio``     : fraction of unsolved routes.
        - ``old_score_solved``   : stats dict for ``route_score`` (solved).
        - ``new_score_solved``   : stats dict for ``hac_weighted_score`` (solved).
        - ``search_time_solved`` : stats dict for ``search_time_seconds`` (solved).
        - ``route_length_solved``: stats dict for route length (solved).
        - *(same four keys with ``_unsolved`` suffix)*
    """
    routes = load_routes(json_path)

    solved, unsolved = [], []

    for route in routes:
        entry = {
            "molecule":     route["target_smiles"],
            "old_score":    route["route_score"],
            "new_score":    route["hac_weighted_score"],
            "search_time":  route["search_time_seconds"],
            "route_length": route.get("route_length", len(route.get("synthesis_steps", []))),
        }
        if all(p["in_stock"] for p in route["precursors"]):
            solved.append(entry)
        else:
            unsolved.append(entry)

    total = len(routes)

    summary = {
        "n_molecules":        total,
        "solved_molecules":   len(solved),
        "unsolved_molecules": len(unsolved),
        "solved_ratio":       len(solved)   / total if total else 0,
        "unsolved_ratio":     len(unsolved) / total if total else 0,
    }

    for group, entries in [("solved", solved), ("unsolved", unsolved)]:
        summary[f"old_score_{group}"]    = stats([e["old_score"]    for e in entries])
        summary[f"new_score_{group}"]    = stats([e["new_score"]    for e in entries])
        summary[f"search_time_{group}"]  = stats([e["search_time"]  for e in entries])
        summary[f"route_length_{group}"] = stats([e["route_length"] for e in entries])

    return summary


# ============================================================
# OUTPUT HELPERS
# ============================================================

def write_csv(summary: dict, csv_path: Path) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Summary counts at the top
        writer.writerow(["total_molecules", summary["n_molecules"]])
        writer.writerow(["solved_molecules", summary["solved_molecules"]])
        writer.writerow(["unsolved_molecules", summary["unsolved_molecules"]])
        writer.writerow([])  # blank separator row
        # Stats table
        writer.writerow(["group", "metric", "mean", "median", "std", "min", "max", "p90", "p95"])
        for group in ("solved", "unsolved"):
            for metric in ("old_score", "new_score", "search_time", "route_length"):
                s = summary[f"{metric}_{group}"]
                writer.writerow([
                    group, metric,
                    s["mean"], s["median"], s["std"],
                    s["min"], s["max"], s["p90"], s["p95"],
                ])


def print_block(title: str, count: int, ratio: float, block: dict) -> None:
    """
    Pretty-print a statistics block for one group (solved or unsolved).

    Parameters
    ----------
    title : str
        Section header (e.g. ``"Solved"``).
    count : int
        Number of molecules in the group.
    ratio : float
        Fraction of total molecules in the group.
    block : dict
        Dict with keys ``old_score``, ``new_score``, ``search_time``,
        ``route_length``; each value is a stats dict from :func:`stats`.
    """
    print(f"\n--- {title} ---")
    print(f"Count: {count}    Ratio: {ratio:.2%}")

    rows = [
        ["", "mean", "median", "std", "min", "max", "p90", "p95"],
        ["Old score",
         f"{block['old_score']['mean']:.4f}",   f"{block['old_score']['median']:.4f}",
         f"{block['old_score']['std']:.4f}",    f"{block['old_score']['min']:.4f}",
         f"{block['old_score']['max']:.4f}",    f"{block['old_score']['p90']:.4f}",
         f"{block['old_score']['p95']:.4f}"],
        ["New score",
         f"{block['new_score']['mean']:.4f}",   f"{block['new_score']['median']:.4f}",
         f"{block['new_score']['std']:.4f}",    f"{block['new_score']['min']:.4f}",
         f"{block['new_score']['max']:.4f}",    f"{block['new_score']['p90']:.4f}",
         f"{block['new_score']['p95']:.4f}"],
        ["Search time (s)",
         f"{block['search_time']['mean']:.2f}",  f"{block['search_time']['median']:.2f}",
         f"{block['search_time']['std']:.2f}",   f"{block['search_time']['min']:.2f}",
         f"{block['search_time']['max']:.2f}",   f"{block['search_time']['p90']:.2f}",
         f"{block['search_time']['p95']:.2f}"],
        ["Route length",
         f"{block['route_length']['mean']:.2f}", f"{block['route_length']['median']:.2f}",
         f"{block['route_length']['std']:.2f}",  f"{block['route_length']['min']}",
         f"{block['route_length']['max']}",      f"{block['route_length']['p90']:.2f}",
         f"{block['route_length']['p95']:.2f}"],
    ]

    print(tabulate(rows[1:], headers=rows[0], tablefmt="pretty"))


class Tee:
    """Write simultaneously to stdout (captured by SLURM) and a log file."""

    def __init__(self, path: Path):
        self._stdout = sys.stdout  # save original stdout before reassignment
        self.file = open(path, "w")

    def write(self, data: str):
        self._stdout.write(data)   # use saved reference, not sys.stdout
        self.file.write(data)

    def flush(self):
        self._stdout.flush()       # same here
        self.file.flush()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read JSON/JSONL routes and generate descriptive statistics "
            "(mean, median, std, min, max, p90, p95) for scores, search time, "
            "and route length, split by solved vs unsolved molecules."
        )
    )
    parser.add_argument(
        "input", type=str,
        help="Path to the input JSON or JSONL file produced by get_scores_new_v2.py."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_PROJECT_ROOT / "data/processed/analysis",
        help="Directory for CSV and TXT analysis outputs.",
    )
    args = parser.parse_args()

    json_path  = args.input
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run analysis first so `summary` is available for both outputs
    summary = analyze_routes(json_path)

    # 2. Write CSV (no stdout redirection needed)
    csv_path = output_dir / (Path(json_path).stem + "_analysis.csv")
    write_csv(summary, csv_path)
    print(f"✓ CSV saved to: {csv_path}")

    # 3. Write TXT report (mirror to stdout via Tee)
    report_path = output_dir / (Path(json_path).stem + "_analysis.txt")
    sys.stdout = Tee(report_path)

    try:
        print("\n=== Route Analysis Summary ===")
        print(f"Total molecules      : {summary['n_molecules']}")
        print(f"Solved molecules     : {summary['solved_molecules']}")
        print(f"Unsolved molecules   : {summary['unsolved_molecules']}")

        print_block(
            "Solved",
            summary["solved_molecules"],
            summary["solved_ratio"],
            {k: summary[f"{k}_solved"] for k in ("old_score", "new_score", "search_time", "route_length")},
        )
        print_block(
            "Unsolved",
            summary["unsolved_molecules"],
            summary["unsolved_ratio"],
            {k: summary[f"{k}_unsolved"] for k in ("old_score", "new_score", "search_time", "route_length")},
        )
    finally:
        tee.file.close()
        sys.stdout = sys.__stdout__

    print(f"✓ Report saved to: {report_path}")



if __name__ == "__main__":
    main()