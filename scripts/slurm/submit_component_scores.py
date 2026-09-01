"""
submit_component_scores.py
==========================
Build and submit a SLURM job array that calls get_component_routes_and_scores.py
on every unique (component_type, cap) slice in the capped-components CSV.

Each SLURM array task maps to exactly one (component_type, cap) pair, or — when
``--chunk_size`` is set — to one chunk within a large slice. A TSV task-list is
written to ``--log-dir`` so that each task can look up its own parameters by
1-based array index. The SLURM script is also saved there for reproducibility.

Usage
-----
    python scripts/submit_component_scores.py \\
        --csv data/processed/component_capped.csv \\
        --config config/config.yml \\
        --stock_db data/external/aizynthfinder_stock.db \\
        --outdir data/processed/scores/components \\
        --account berzelius-2026-62 \\
        [--chunk_size 200] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORER = REPO_ROOT / "scripts" / "retrosynthesis" / "get_component_routes_and_scores.py"


def _enumerate_tasks(
    csv: Path,
    smiles_col: str,
    error_col: str,
    chunk_size: int,
) -> List[Tuple[str, str, int, int]]:
    """Read the capped CSV and enumerate all (component_type, cap, chunk_idx, n_chunks) tasks.

    Applies the same error-row filtering and component-label normalisation as the
    scoring script so that the task count and slicing logic stay in sync.

    Args:
        csv: Path to the capped-components CSV.
        smiles_col: Column holding canonical capped SMILES (used to count unique molecules).
        error_col: Column holding capping error strings; rows with non-empty values are skipped.
        chunk_size: If > 0, large slices are split into chunks of this size.

    Returns:
        List of (component_type, cap, chunk_idx, n_chunks) tuples, one per SLURM task.
        When chunk_size == 0, every task has chunk_idx=0 and n_chunks=1.
    """
    df = pd.read_csv(csv)

    # Mirror the preprocessing in get_component_routes_and_scores.py exactly so
    # the task list reflects the molecules that will actually be processed.
    if error_col in df.columns:
        df = df[
            df[error_col].isna() | (df[error_col].astype(str).str.strip() == "")
        ].copy()

    df["component"] = df["component_id"].str.split("_").str[0]
    df["component"] = df["component"].replace({"E3": "e3", "LK": "linker", "WH": "warhead"})
    df["cap_type"] = df["cap_type"].astype(str).str.strip()

    tasks: List[Tuple[str, str, int, int]] = []
    for (component_type, cap), group in df.groupby(["component", "cap_type"]):
        n_unique = group[smiles_col].dropna().nunique()
        if chunk_size > 0 and n_unique > chunk_size:
            n_chunks = math.ceil(n_unique / chunk_size)
            for chunk_idx in range(n_chunks):
                tasks.append((str(component_type), str(cap), chunk_idx, n_chunks))
        else:
            tasks.append((str(component_type), str(cap), 0, 1))

    return tasks


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SLURM submission script.

    Returns:
        Populated ``argparse.Namespace``.
    """
    ap = argparse.ArgumentParser(
        description="Submit a SLURM job array for AiZynthFinder component scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    slurm = ap.add_argument_group("SLURM")
    slurm.add_argument("--account", required=True,
                       help="SLURM account (e.g. berzelius-2026-62).")
    slurm.add_argument("--partition", default="berzelius-cpu",
                       help="SLURM partition.")
    slurm.add_argument("--time", default="08:00:00",
                       help="Wall-clock time limit per task.")
    slurm.add_argument("--cpus", type=int, default=16,
                       help="CPUs per task (--cpus-per-task). CPU nodes have 128 cores (2×AMD EPYC 9534); "
                            "16 fits 8 concurrent tasks per node and saturates TF intra-op parallelism.")
    slurm.add_argument("--mem", default="32G",
                       help="Memory per task.")
    slurm.add_argument("--job-name", default="comp_scores",
                       help="SLURM job name.")
    slurm.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs" / "component_scores",
                       help="Directory for SLURM stdout/stderr logs and the task list/script.")
    slurm.add_argument("--mail", default=None,
                       help="Email address for END/FAIL notifications.")

    scorer = ap.add_argument_group("scorer (forwarded to get_component_routes_and_scores.py)")
    scorer.add_argument("--csv", required=True, type=Path, help="Capped components CSV.")
    scorer.add_argument("--config", required=True, type=Path, help="AiZynthFinder config YAML.")
    scorer.add_argument("--stock_db", required=True, type=Path, help="SQLite stock database.")
    scorer.add_argument("--outdir", required=True, type=Path,
                        help="Output directory for score CSVs.")
    scorer.add_argument("--chunk_size", type=int, default=0,
                        help="Split slices with more unique SMILES than this into chunks (0 = no split).")
    scorer.add_argument("--smiles_col", default="cap_smiles",
                        help="SMILES column in the capped CSV.")
    scorer.add_argument("--error_col", default="error",
                        help="Error column in the capped CSV.")

    ap.add_argument("--dry-run", action="store_true",
                    help="Write the sbatch script and print it without submitting.")
    return ap.parse_args()


def main() -> None:
    """Build the task list and submit (or dry-run) the SLURM job array."""
    args = parse_args()

    # Resolve user-supplied paths to absolute so the sbatch script is location-independent.
    csv = args.csv.resolve()
    config = args.config.resolve()
    stock_db = args.stock_db.resolve()
    outdir = args.outdir.resolve()
    log_dir = args.log_dir.resolve()

    for p, label in [(csv, "--csv"), (config, "--config"), (stock_db, "--stock_db")]:
        if not p.exists():
            print(f"Error: {label} path not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Enumerating tasks from {csv} ...")
    tasks = _enumerate_tasks(csv, args.smiles_col, args.error_col, args.chunk_size)
    if not tasks:
        print("No tasks found — is the CSV empty or missing cap_type / component_id columns?",
              file=sys.stderr)
        sys.exit(1)

    print(f"Total SLURM tasks: {len(tasks)}")
    for t in tasks[:8]:
        print(f"  component_type={t[0]}  cap={t[1]}  chunk={t[2]}/{t[3]-1}")
    if len(tasks) > 8:
        print(f"  ... ({len(tasks) - 8} more)")

    # Write the task list; each array task reads its own line by 1-based SLURM_ARRAY_TASK_ID.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    task_list = log_dir / f"tasks_{stamp}.tsv"
    with open(task_list, "w") as fh:
        fh.write("component_type\tcap\tchunk_idx\tn_chunks\n")
        for component_type, cap, chunk_idx, n_chunks in tasks:
            fh.write(f"{component_type}\t{cap}\t{chunk_idx}\t{n_chunks}\n")
    print(f"Task list: {task_list}")

    # Build the python command; chunk args are only appended when chunking is active.
    cmd_parts = [
        f"python {SCORER}",
        f"    --config {config}",
        f"    --stock_db {stock_db}",
        f"    --csv {csv}",
        '    --component_type "$COMPONENT_TYPE"',
        '    --cap "$CAP"',
        f"    --outdir {outdir}",
    ]
    if args.chunk_size > 0:
        cmd_parts.append(f"    --chunk_size {args.chunk_size}")
        cmd_parts.append('    --chunk_idx "$CHUNK_IDX"')
    python_cmd = " \\\n".join(cmd_parts)

    mail_line = (
        f"#SBATCH --mail-user={args.mail} --mail-type=END,FAIL"
        if args.mail else ""
    )

    batch_script = f"""\
#!/bin/bash
#SBATCH -A {args.account}
#SBATCH -p {args.partition}
#SBATCH -t {args.time}
#SBATCH --array=1-{len(tasks)}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.mem}
#SBATCH -J {args.job_name}
#SBATCH --output={log_dir}/{args.job_name}_%A_%a.out
#SBATCH --error={log_dir}/{args.job_name}_%A_%a.err
{mail_line}
module load Mambaforge/23.3.1-1-hpc1-bdist
eval "$(conda shell.bash hook)"
mamba activate env-retrotac

cd {REPO_ROOT}
source .venv/bin/activate

TASK_FILE="{task_list}"

# Resolve this task's parameters by reading line SLURM_ARRAY_TASK_ID+1
# (line 1 is the TSV header, so task 1 maps to line 2, task N to line N+1).
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
COMPONENT_TYPE=$(echo "$LINE" | cut -f1)
CAP=$(echo "$LINE"           | cut -f2)
CHUNK_IDX=$(echo "$LINE"     | cut -f3)

echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "Task        : $SLURM_ARRAY_TASK_ID  component_type=$COMPONENT_TYPE  cap=$CAP  chunk_idx=$CHUNK_IDX"

{python_cmd}

echo "Job finished: $(date)"
"""

    sbatch_path = log_dir / f"{args.job_name}_{stamp}.sh"
    sbatch_path.write_text(batch_script)
    print(f"SLURM script: {sbatch_path}")

    if args.dry_run:
        print("\n--- sbatch script (dry run, not submitted) ---")
        print(batch_script)
        return

    result = subprocess.run(["sbatch", str(sbatch_path)], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(f"sbatch failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
