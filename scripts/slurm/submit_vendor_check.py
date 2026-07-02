#!/usr/bin/env python3
"""
Submit comp_cid_vendor_check.py as a single SLURM batch job.

A job array is counterproductive here: all tasks share the cluster IP and
would collectively exceed PubChem's rate limit. Instead, the batched POST
approach (--batch-size 100) reduces ~500k individual calls to ~5k requests,
which a single node handles in a few hours well within PubChem's limits.

Usage:
    python scripts/submit_vendor_check.py --account berzelius-2026-62
    python scripts/submit_vendor_check.py --account berzelius-2026-62 --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT     = Path(__file__).resolve().parent.parent
VENDOR_SCRIPT = REPO_ROOT / "src" / "data_preprocessing" / "comp_cid_vendor_check.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit the PubChem vendor check as a SLURM batch job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    slurm = p.add_argument_group("SLURM")
    slurm.add_argument("--account",   required=True,
                       help="SLURM account (e.g. berzelius-2026-62).")
    slurm.add_argument("--partition", default="berzelius",
                       help="SLURM partition.")
    slurm.add_argument("--time",      default="48:00:00",
                       help="Wall-clock time limit.")
    slurm.add_argument("--mem",       default="8G",
                       help="Memory per node.")
    slurm.add_argument("--cpus",      type=int, default=1,
                       help="CPUs per task.")
    slurm.add_argument("--job-name",  default="vendor_check",
                       help="SLURM job name.")
    slurm.add_argument("--log-dir",   type=Path, default=REPO_ROOT / "logs",
                       help="Directory for .out / .err logs.")
    slurm.add_argument("--mail",      default=None,
                       help="Email address for END/FAIL notifications.")

    script = p.add_argument_group("vendor check (forwarded to comp_cid_vendor_check.py)")
    script.add_argument("--input",        type=Path,  default=None,
                        help="Input CSV (uses script default when omitted).")
    script.add_argument("--cid-cache",    type=Path,  default=None,
                        help="Resumable SMILES→CID cache CSV.")
    script.add_argument("--vendor-cache", type=Path,  default=None,
                        help="Resumable CID→vendor cache CSV.")
    script.add_argument("--output",       type=Path,  default=None,
                        help="Final output CSV.")
    script.add_argument("--sleep",        type=float, default=0.35,
                        help="Delay between PubChem API calls (seconds).")
    script.add_argument("--retries",      type=int,   default=1,
                        help="Retry attempts on transient server errors.")
    script.add_argument("--batch-size",   type=int,   default=100,
                        help="SMILES per batched POST request.")

    p.add_argument("--dry-run", action="store_true",
                   help="Print the generated batch script without submitting.")
    return p.parse_args()


def _script_args(args: argparse.Namespace) -> str:
    parts = [
        f"--sleep {args.sleep}",
        f"--retries {args.retries}",
        f"--batch-size {args.batch_size}",
    ]
    if args.input:        parts.append(f"--input {args.input}")
    if args.cid_cache:    parts.append(f"--cid-cache {args.cid_cache}")
    if args.vendor_cache: parts.append(f"--vendor-cache {args.vendor_cache}")
    if args.output:       parts.append(f"--output {args.output}")
    return " \\\n        ".join(parts)


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    mail_line = (
        f"#SBATCH --mail-user={args.mail} --mail-type=END,FAIL"
        if args.mail else ""
    )

    batch_script = dedent(f"""\
        #!/bin/bash
        #SBATCH -A {args.account}
        #SBATCH -p {args.partition}
        #SBATCH -t {args.time}
        #SBATCH -N 1
        #SBATCH --cpus-per-task={args.cpus}
        #SBATCH --mem={args.mem}
        #SBATCH -J {args.job_name}
        #SBATCH --output={args.log_dir}/{args.job_name}_%j.out
        #SBATCH --error={args.log_dir}/{args.job_name}_%j.err
        {mail_line}

        module load Mambaforge/23.3.1-1-hpc1-bdist
        eval "$(conda shell.bash hook)"
        mamba activate env-protac-synth

        cd {REPO_ROOT}
        source .venv/bin/activate

        echo "Job started : $(date)"
        echo "Node        : $SLURMD_NODENAME"

        python {VENDOR_SCRIPT} \\
            {_script_args(args)}

        echo "Job finished: $(date)"
    """)

    if args.dry_run:
        print(batch_script)
        return

    result = subprocess.run(["sbatch"], input=batch_script, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(f"sbatch failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
