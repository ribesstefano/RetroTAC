"""
filter_unsolved_precursors.py
─────────────────────────────
Filters synthesis routes from a JSON file, keeping only "unsolved" routes —
those where at least one precursor is not in stock — and writes the unique
out-of-stock precursor SMILES to a CSV file.

A route is considered **unsolved** if at least one of its precursors has
``in_stock`` set to False.

Input JSON format
-----------------
The file must contain either:
  - A JSON array of route objects, or
  - A newline-delimited JSON file (one route object per line).

Each route object is expected to have the following fields::

  {
    "target_smiles":       str,    # SMILES of the target molecule
    "hac_weighted_score":  float,  # HAC-weighted synthetic accessibility score
    "route_score":         float,  # Overall route score
    "route_length":        int,    # Number of steps in the route
    "precursors": [
      {
        "precursor_smiles": str,   # SMILES of the precursor
        "in_stock":         bool,  # Whether the precursor is commercially available
        "stock_source":     str    # Comma-separated list of stock sources
      },
      ...
    ]
  }

Output
------
- Stdout: summary of matching routes with their out-of-stock precursors.
- CSV:    saved to data/processed, named
          <input_basename>_unsolved_precursors.csv
          Columns: out_of_stock_precursor
          One row per unique out-of-stock precursor SMILES.

Usage
-----
    python filter_unsolved_precursors.py <json_path>

Arguments
---------
    input   Path to the input JSON file (required).

Examples
--------
    python filter_unsolved_precursors.py routes.json
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_routes(json_path: str) -> List[Dict[str, Any]]:
    """Load synthesis routes from a JSON or NDJSON file.

    Args:
        json_path: Path to the JSON or NDJSON file.

    Returns:
        List of route dictionaries.

    Raises:
        json.JSONDecodeError: If the file cannot be parsed as either format.
    """
    with open(json_path, "r") as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def filter_unsolved(json_path: str, output_dir: Path) -> None:
    """Extract unique out-of-stock precursors from unsolved routes.

    A route is unsolved if at least one precursor has ``in_stock`` set to
    False. Fully solved routes are skipped. Results are printed to stdout
    and saved as a CSV of unique precursor SMILES.

    Args:
        json_path: Path to the input JSON file containing synthesis routes.
        output_dir: Directory the output CSV is written to.
    """
    routes = load_routes(json_path)

    results = []
    for route in routes:
        precursors = route["precursors"]

        # Skip fully solved routes
        if all(p["in_stock"] for p in precursors):
            continue

        out_of_stock = [p for p in precursors if not p["in_stock"]]
        results.append({
            "molecule": route["target_smiles"],
            "out_of_stock_precursors": [p["precursor_smiles"] for p in out_of_stock],
        })

    # ── Stdout report ─────────────────────────────────────────────────────────
    print(f"Unsolved molecules: {len(results)}\n")
    for r in results:
        print(f"Molecule : {r['molecule']}")
        print(f"Out-of-stock precursors ({len(r['out_of_stock_precursors'])}):")
        for smi in r["out_of_stock_precursors"]:
            print(f"  {smi}")
        print()

    # ── CSV export ────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(json_path).stem + "_unsolved_precursors.csv"
    output_csv = output_dir / filename

    seen = set()
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["out_of_stock_precursor"])
        for r in results:
            for smi in r["out_of_stock_precursors"]:
                if smi not in seen:
                    seen.add(smi)
                    writer.writerow([smi])

    print(f"CSV saved to: {output_csv} ({len(seen)} unique precursors)")


def main() -> None:
    """Parse CLI arguments and run ``filter_unsolved`` on the given JSON file."""
    parser = argparse.ArgumentParser(description="Extract unique out-of-stock precursors from unsolved routes.")
    parser.add_argument("input", type=str, help="Path to input JSON file.")
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / "data/processed",
                        help="Directory for the output CSV (default: data/processed/).")
    args = parser.parse_args()
    filter_unsolved(args.input, args.output_dir)


if __name__ == "__main__":
    main()