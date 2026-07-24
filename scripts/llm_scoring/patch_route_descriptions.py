"""Backfill `llm_route_description` onto a CSV already scored by llm_scoring.py.

Older scoring runs predate the `llm_route_description` output column, so
`llm_rationale`'s [I2]/[U1]-style tags are undecodable without it. The compound
legend + steps text is rebuilt purely from each row's `route`/`BBs`/`resolved`
columns -- the same deterministic step `RouteScorer` takes before calling the
LLM -- so this never re-queries the judge; it only recovers what the original
run threw away.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_PROJECT_ROOT / "src"))

from scripts.llm_scoring.route_parsing import parse_route_row, render_route_for_llm  # noqa: E402


def _route_description(row: Dict[str, Any]) -> str:
    """Mirrors SynthesisScorer's route/molecule-only/in-stock branching, minus the LLM call."""
    parsed = parse_route_row(row)
    if parsed.in_stock or not parsed.reactions:
        return ""
    return render_route_for_llm(parsed).route_text


def patch_route_descriptions(in_csv: Path, out_csv: Path, log_errors: bool = False) -> None:
    """Add or overwrite `llm_route_description` on every row of a scored CSV.

    Args:
        in_csv: CSV produced by llm_scoring.py (needs the `route`/`BBs`/`resolved`
            input columns; the llm_* columns are untouched other than the new one).
        out_csv: Destination path. When equal to `in_csv`, the original is copied to
            `<in_csv>.bak` first and the write goes through a temp file + rename so
            a crash mid-write can't corrupt the only copy of an expensive scoring run.
        log_errors: Print the row index and exception for rows that fail to parse
            (score is left blank either way). Off by default.

    Returns:
        None. Writes `out_csv` and prints a one-line summary.
    """
    df = pd.read_csv(in_csv)
    rows = df.to_dict("records")

    descriptions = []
    n_errors = 0
    for i, row in enumerate(tqdm(rows, desc="rendering route trees")):
        try:
            descriptions.append(_route_description(row))
        except Exception as e:
            descriptions.append("")
            n_errors += 1
            if log_errors:
                print(f"row {i}: {e!r}")

    df = df.drop(columns=["llm_route_description"], errors="ignore")
    # Match llm_scoring.py's column order so freshly-scored and patched CSVs read the same.
    insert_at = (df.columns.get_loc("llm_rationale") + 1
                 if "llm_rationale" in df.columns else len(df.columns))
    df.insert(insert_at, "llm_route_description", descriptions)

    if out_csv == in_csv:
        backup = in_csv.with_suffix(in_csv.suffix + ".bak")
        shutil.copy2(in_csv, backup)
        print(f"backed up original -> {backup}")

    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_csv)  # atomic rename on the same filesystem

    n_filled = sum(1 for d in descriptions if d)
    print(f"llm_route_description: {n_filled}/{len(df)} rows filled "
          f"({n_errors} parse errors left blank) -> {out_csv}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in-csv", type=Path,
                    default=_PROJECT_ROOT / "data/llm_scoring/routes_llm_scores.csv",
                    help="CSV produced by llm_scoring.py to patch (default: %(default)s)")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="output path; default overwrites --in-csv (a .bak backup is made first)")
    ap.add_argument("--log-errors", action="store_true",
                    help="print row index + exception for rows that fail to parse")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_csv = args.out_csv or args.in_csv
    patch_route_descriptions(args.in_csv, out_csv, log_errors=args.log_errors)


if __name__ == "__main__":
    main()
