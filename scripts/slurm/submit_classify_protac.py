#!/usr/bin/env python3
"""
submit_classify_protac.py
==========================
Build and submit a PROTAC/non-PROTAC LLM classification SLURM job
(scripts/llm_scoring/classify_protac.py).

Single job, not an array: classify_protac.py already parallelizes LLM calls
internally via a thread pool (--threads), so one job with enough threads
covers the whole CSV. LLM scoring's dependencies (dspy, litellm, ...) live in
the `scoring` optional extra, not installed by plain `uv sync`, so the job
runs via `uv run --extra scoring` rather than an already-activated .venv.

Usage
-----
    python scripts/slurm/submit_classify_protac.py \\
        --in-csv data/some_molecules.csv \\
        --out-csv data/llm_scoring/molecules_classified.csv \\
        --account berzelius-2026-62 \\
        --smiles-column SMILES --rationale --threads 8 \\
        --model openrouter/google/gemini-2.5-flash-lite \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER = _PROJECT_ROOT / "scripts" / "llm_scoring" / "classify_protac.py"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SLURM submission script.

    Returns:
        Populated ``argparse.Namespace``.
    """
    ap = argparse.ArgumentParser(
        description="Submit PROTAC/non-PROTAC LLM classification as a SLURM job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    slurm = ap.add_argument_group("SLURM")
    slurm.add_argument("--account", required=True,
                       help="SLURM account (e.g. berzelius-2026-62).")
    slurm.add_argument("--partition", default="berzelius-cpu",
                       help="SLURM partition.")
    slurm.add_argument("--time", default="12:00:00",
                       help="Wall-clock time limit.")
    slurm.add_argument("--cpus", type=int, default=8,
                       help="CPUs per task. The job is API-bound (waiting on the LLM "
                            "provider), not compute-bound; kept roughly in line with "
                            "--threads rather than sized for local computation.")
    slurm.add_argument("--mem", default="8G",
                       help="Memory per task.")
    slurm.add_argument("--job-name", default="classify_protac",
                       help="SLURM job name.")
    slurm.add_argument("--log-dir", type=Path,
                       default=_PROJECT_ROOT / "logs" / "classify_protac",
                       help="Directory for SLURM stdout/stderr logs and the sbatch script.")
    slurm.add_argument("--mail", default=None,
                       help="Email address for END/FAIL notifications.")

    clf = ap.add_argument_group("classification (forwarded to classify_protac.py)")
    clf.add_argument("--in-csv", required=True, type=Path,
                     help="Input CSV containing a SMILES column.")
    clf.add_argument("--out-csv", required=True, type=Path,
                     help="Output CSV (input columns + appended llm_protac_* columns).")
    clf.add_argument("--smiles-column", default="SMILES",
                     help="Name of the SMILES column in --in-csv.")
    clf.add_argument("--rationale", action="store_true",
                     help="Also elicit a concise rationale string (extra tokens); default off.")
    clf.add_argument("--threads", type=int, default=8,
                     help="Concurrent LLM calls.")
    clf.add_argument("--limit", type=int, default=None,
                     help="Classify only the first N rows (for testing).")
    clf.add_argument("--model", default="gemini/gemini-3.1-pro-preview",
                     help="litellm-style model string, '<provider>/<model>'. "
                          "See classify_protac.py --model for the Gemini/OpenRouter convention.")
    clf.add_argument("--temperature", type=float, default=None,
                     help="Sampling temperature. Default: unset (model's own default).")

    ap.add_argument("--dry-run", action="store_true",
                    help="Write the sbatch script and print it without submitting.")
    return ap.parse_args()


def _classifier_args(args: argparse.Namespace) -> str:
    """Build the classify_protac.py flag string from the forwarded classification args."""
    parts = [
        f"--smiles-column {args.smiles_column}",
        f"--threads {args.threads}",
        f"--model {args.model}",
    ]
    if args.rationale:
        parts.append("--rationale")
    if args.limit is not None:
        parts.append(f"--limit {args.limit}")
    if args.temperature is not None:
        parts.append(f"--temperature {args.temperature}")
    return " \\\n        ".join(parts)


def main() -> None:
    """Validate paths, build the sbatch script, and submit (or dry-run)."""
    args = parse_args()

    in_csv = args.in_csv.resolve()
    if not in_csv.exists():
        print(f"Error: --in-csv path not found: {in_csv}", file=sys.stderr)
        sys.exit(1)
    out_csv = args.out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    mail_line = (
        f"#SBATCH --mail-user={args.mail} --mail-type=END,FAIL"
        if args.mail else ""
    )

    batch_script = f"""\
#!/bin/bash
#SBATCH -A {args.account}
#SBATCH -p {args.partition}
#SBATCH -t {args.time}
#SBATCH -N 1
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.mem}
#SBATCH -J {args.job_name}
#SBATCH --output={log_dir}/{args.job_name}_%j.out
#SBATCH --error={log_dir}/{args.job_name}_%j.err
{mail_line}

module load Mambaforge/23.3.1-1-hpc1-bdist
eval "$(conda shell.bash hook)"
mamba activate env-retrotac

cd {_PROJECT_ROOT}

echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"

uv run --extra scoring {CLASSIFIER} \\
    {in_csv} \\
    {out_csv} \\
    {_classifier_args(args)}

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
