"""
submit_protac_scores.py
=======================
Build and submit a SLURM job array that calls get_protac_routes_and_scores.py
on interleaved chunks of a SMILES CSV.

Each array task covers molecules at positions [task_id, task_id + N,
task_id + 2N, ...] (round-robin), so the work is balanced regardless of any
ordering bias in the input CSV. The number of tasks is set directly with
``--total_tasks`` or inferred from ``--chunk_size`` as
``ceil(n_molecules / chunk_size)``.

Usage
-----
    python scripts/submit_protac_scores.py \\
        --csv data/processed/protac_smiles.csv \\
        --config config/aizynthfinder_protac_config.yaml \\
        --stock_db data/external/aizynthfinder_stock.db \\
        --outdir data/processed/scores/protacs \\
        --account berzelius-2026-62 \\
        --total_tasks 8 \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORER = REPO_ROOT / "scripts" / "retrosynthesis" / "get_protac_routes_and_scores.py"


def _count_molecules(csv: Path, smiles_col: str) -> int:
    """Count unique, non-null SMILES in the input CSV.

    Mirrors the column-detection logic in get_protac_routes_and_scores.py so
    that the task count computed here matches the number actually processed.

    Args:
        csv: Path to the SMILES CSV.
        smiles_col: Name of the SMILES column; if empty, common names are tried.

    Returns:
        Number of unique non-null SMILES strings found.

    Raises:
        ValueError: If no SMILES column can be found.
    """
    df = pd.read_csv(csv)

    col = smiles_col
    if not col:
        for candidate in ("smiles", "SMILES", "Smiles", "smi", "SMI", "canonical_smiles"):
            if candidate in df.columns:
                col = candidate
                break
    if not col or col not in df.columns:
        raise ValueError(
            f"Cannot find SMILES column in {csv}. Available: {list(df.columns)}"
        )

    return int(df[col].dropna().nunique())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SLURM submission script.

    Returns:
        Populated ``argparse.Namespace``.
    """
    ap = argparse.ArgumentParser(
        description="Submit a SLURM job array for AiZynthFinder PROTAC scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    slurm = ap.add_argument_group("SLURM")
    slurm.add_argument("--account", required=True,
                       help="SLURM account (e.g. berzelius-2026-62).")
    slurm.add_argument("--partition", default="berzelius-cpu",
                       help="SLURM partition.")
    slurm.add_argument("--time", default="12:00:00",
                       help="Wall-clock time limit per task.")
    slurm.add_argument("--cpus", type=int, default=16,
                       help="CPUs per task (--cpus-per-task). CPU nodes have 128 cores (2×AMD EPYC 9534); "
                            "16 fits 8 concurrent tasks per node and saturates TF intra-op parallelism.")
    slurm.add_argument("--mem", default="32G",
                       help="Memory per task.")
    slurm.add_argument("--job-name", default="protac_scores",
                       help="SLURM job name.")
    slurm.add_argument("--log-dir", type=Path,
                       default=REPO_ROOT / "logs" / "protac_scores",
                       help="Directory for SLURM stdout/stderr logs and the sbatch script.")
    slurm.add_argument("--mail", default=None,
                       help="Email address for END/FAIL notifications.")

    scorer = ap.add_argument_group("scorer (forwarded to get_protac_routes_and_scores.py)")
    scorer.add_argument("--csv", required=True, type=Path,
                        help="Input CSV containing SMILES to score.")
    scorer.add_argument("--config", required=True, type=Path,
                        help="AiZynthFinder config YAML.")
    scorer.add_argument("--stock_db", required=True, type=Path,
                        help="SQLite stock database.")
    scorer.add_argument("--outdir", required=True, type=Path,
                        help="Output directory for score CSVs and JSONL.")
    scorer.add_argument("--prefix", default="protac",
                        help="Filename stem for output files.")
    scorer.add_argument("--smiles_col", default="",
                        help="SMILES column in the input CSV (auto-detected if empty).")

    sizing = ap.add_mutually_exclusive_group(required=True)
    sizing.add_argument(
        "--total_tasks", type=int,
        help="Fixed number of SLURM array tasks.",
    )
    sizing.add_argument(
        "--chunk_size", type=int,
        help="Molecules per task; total tasks = ceil(n_molecules / chunk_size).",
    )

    ap.add_argument("--dry-run", action="store_true",
                    help="Write the sbatch script and print it without submitting.")
    return ap.parse_args()


def main() -> None:
    """Compute total_tasks, build the sbatch script, and submit (or dry-run)."""
    args = parse_args()

    csv = args.csv.resolve()
    config = args.config.resolve()
    stock_db = args.stock_db.resolve()
    outdir = args.outdir.resolve()
    log_dir = args.log_dir.resolve()

    for p, label in [(csv, "--csv"), (config, "--config"), (stock_db, "--stock_db")]:
        if not p.exists():
            print(f"Error: {label} path not found: {p}", file=sys.stderr)
            sys.exit(1)

    if args.total_tasks is not None:
        total_tasks = args.total_tasks
    else:
        n_molecules = _count_molecules(csv, args.smiles_col)
        total_tasks = math.ceil(n_molecules / args.chunk_size)
        print(f"Found {n_molecules} unique SMILES → {total_tasks} tasks "
              f"(chunk_size={args.chunk_size})")

    if total_tasks < 1:
        print("Error: total_tasks must be >= 1.", file=sys.stderr)
        sys.exit(1)

    print(f"Total SLURM array tasks: {total_tasks}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    mail_line = (
        f"#SBATCH --mail-user={args.mail} --mail-type=END,FAIL"
        if args.mail else ""
    )

    smiles_col_flag = f"--smiles_col {args.smiles_col}" if args.smiles_col else ""

    # SLURM_ARRAY_TASK_ID is 1-based; the scorer converts it internally to 0-based.
    batch_script = f"""\
#!/bin/bash
#SBATCH -A {args.account}
#SBATCH -p {args.partition}
#SBATCH -t {args.time}
#SBATCH --array=1-{total_tasks}
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

echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "Task        : $SLURM_ARRAY_TASK_ID / {total_tasks}"

python {SCORER} \\
    --config {config} \\
    --stock_db {stock_db} \\
    --input_csv {csv} \\
    --outdir {outdir} \\
    --prefix {args.prefix} \\
    --total_tasks {total_tasks} \\
    {smiles_col_flag}

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
