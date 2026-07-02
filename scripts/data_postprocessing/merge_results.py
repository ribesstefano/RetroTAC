"""
merge_results.py
================
Merges the per-task output files produced by a SLURM array run of
``aizynthfinder_run.py`` into single consolidated files per prefix.

When ``aizynthfinder_run.py`` runs as a SLURM array job, each task writes its
own set of output files with names like::

    <prefix>_task0_summary.csv
    <prefix>_task0_precursors.csv
    <prefix>_task0_steps.csv
    <prefix>_task0.json
    <prefix>_task1_summary.csv
    ...

This script discovers all task files inside ``input_dir``, infers all unique
prefixes automatically, and consolidates them into one file per type per prefix:

    <prefix>_summary.csv
    <prefix>_precursors.csv
    <prefix>_steps.csv
    <prefix>.json

Input
-----
Directory containing per-task output files produced by ``aizynthfinder_run.py``.
Files must follow the naming convention ``<prefix>_task<N>_<type>.csv`` or
``<prefix>_task<N>.json``.

Output  (written to ``input_dir`` unless ``--output_dir`` is specified)
-----------------------------------------------------------------------
<prefix>_summary.csv
    One row per molecule: scores, step count, stock stats.
<prefix>_precursors.csv
    One row per precursor: SMILES, stock source.
<prefix>_steps.csv
    One row per synthesis step: reactants, product, template.
<prefix>.json
    All route records as a standard JSON array.

Usage
-----
    python merge_results.py /path/to/outputs
    python merge_results.py /path/to/outputs --output_dir /path/to/merged

Arguments
---------
    input_dir     Directory containing the per-task output files (required).
    --output_dir  Directory where merged files will be saved. Defaults to
                  the same directory as the input files.
"""
import argparse
import re
import json
import pandas as pd
from pathlib import Path


def merge_csvs(input_dir: Path, output_dir: Path, prefix: str) -> None:
    """Merge per-task CSV files into a single consolidated file.

    Args:
        input_dir  (Path): Directory containing per-task CSV files.
        output_dir (Path): Directory where merged files will be saved.
        prefix     (str):  Shared filename prefix inferred from task files.
    """
    for suffix, label in [
        ("_summary.csv",    "summary"),
        ("_precursors.csv", "precursors"),
        ("_steps.csv",      "steps"),
    ]:
        files = sorted(input_dir.glob(f"{prefix}_task*{suffix}"))

        if not files:
            print(f"No files found for pattern: {prefix}_task*{suffix}")
            continue

        print(f"Found {len(files)} files for {label}")
        merged   = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        out_path = output_dir / f"{prefix}{suffix}"
        merged.to_csv(out_path, index=False)
        print(f"✓ Saved {label}: {out_path} ({len(merged)} rows)")


def merge_json(input_dir: Path, output_dir: Path, prefix: str) -> None:
    """Merge per-task JSON/JSONL files into a single JSON array.

    Args:
        input_dir  (Path): Directory containing per-task JSON files.
        output_dir (Path): Directory where merged file will be saved.
        prefix     (str):  Shared filename prefix inferred from task files.
    """
    json_files = sorted(input_dir.glob(f"{prefix}_task*.json"))

    if not json_files:
        print(f"No JSON files found in: {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files")
    merged_json = []

    for f in json_files:
        content = f.read_text().strip()
        try:
            data = json.loads(content)
            merged_json.extend(data if isinstance(data, list) else [data])
        except json.JSONDecodeError:
            merged_json.extend(
                json.loads(line) for line in content.splitlines() if line.strip()
            )

    out_path = output_dir / f"{prefix}.json"
    out_path.write_text(json.dumps(merged_json, indent=2))
    print(f"✓ Saved merged JSON: {out_path} ({len(merged_json)} routes)")


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-task SLURM output files into consolidated files."
    )
    parser.add_argument(
        "input_dir", type=str,
        help="Directory containing the per-task output files."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory where merged files will be saved (default: same as input_dir)."
    )
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Infer shared prefix from first matched task file
    all_task_files = sorted(input_dir.glob("*_task*"))
    if not all_task_files:
        print(f"No task files found in: {input_dir}")
        raise SystemExit(1)

    first_stem = all_task_files[0].stem
    prefix     = re.sub(r"_task\d+.*", "", first_stem)
    print(f"Inferred prefix: '{prefix}'")

    merge_csvs(input_dir, output_dir, prefix)
    merge_json(input_dir, output_dir, prefix)


if __name__ == "__main__":
    main()