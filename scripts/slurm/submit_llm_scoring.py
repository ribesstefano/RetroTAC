#!/usr/bin/env python3
"""
submit_llm_scoring.py
======================
Build and submit an LLM-as-judge route-scoring SLURM job
(scripts/llm_scoring/llm_scoring.py).

Single job, not an array: llm_scoring.py already parallelizes LLM calls
internally via a thread pool (--threads), so one job with enough threads
covers the whole CSV. `llm_scoring`'s dependencies (dspy, litellm, ...) are an
optional extra, not installed by plain `uv sync`, so the job runs via
`uv run --extra llm_scoring` rather than an already-activated .venv.

Usage
-----
    python scripts/slurm/submit_llm_scoring.py \\
        --in-csv data/routes/routes.csv \\
        --out-csv data/llm_scoring/routes_with_score.csv \\
        --account berzelius-2026-62 \\
        --k 3 --threads 8 --model openrouter/google/gemini-2.5-flash-lite \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCORER = _PROJECT_ROOT / "scripts" / "llm_scoring" / "llm_scoring.py"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SLURM submission script.

    Returns:
        Populated ``argparse.Namespace``.
    """
    ap = argparse.ArgumentParser(
        description="Submit LLM-as-judge retrosynthesis route scoring as a SLURM job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    slurm = ap.add_argument_group("SLURM")
    slurm.add_argument("--account", required=True,
                       help="SLURM account (e.g. berzelius-2026-62).")
    slurm.add_argument("--partition", default="berzelius-cpu",
                       help="SLURM partition.")
    slurm.add_argument("--time", default="24:00:00",
                       help="Wall-clock time limit.")
    slurm.add_argument("--cpus", type=int, default=8,
                       help="CPUs per task. The job is API-bound (waiting on the LLM "
                            "provider), not compute-bound; kept roughly in line with "
                            "--threads rather than sized for local computation.")
    slurm.add_argument("--mem", default="8G",
                       help="Memory per task.")
    slurm.add_argument("--job-name", default="llm_scoring",
                       help="SLURM job name.")
    slurm.add_argument("--log-dir", type=Path,
                       default=_PROJECT_ROOT / "logs" / "llm_scoring",
                       help="Directory for SLURM stdout/stderr logs and the sbatch script.")
    slurm.add_argument("--mail", default=None,
                       help="Email address for END/FAIL notifications.")

    scorer = ap.add_argument_group("scoring (forwarded to llm_scoring.py)")
    scorer.add_argument("--in-csv", required=True, type=Path,
                        help="Input routes CSV.")
    scorer.add_argument("--out-csv", required=True, type=Path,
                        help="Output CSV (input columns + appended llm_* columns).")
    scorer.add_argument("--k", type=int, default=3,
                        help="LLM samples per row (denoising).")
    scorer.add_argument("--threads", type=int, default=8,
                        help="Concurrent LLM calls.")
    scorer.add_argument("--limit", type=int, default=None,
                        help="Score only the first N rows (for testing).")
    scorer.add_argument("--cot", action="store_true",
                        help="Use ChainOfThought (zero-shot reasoning field); default off.")
    scorer.add_argument("--score-steps", action="store_true",
                        help="Also elicit per-step evaluations alongside the route score "
                             "(costs more tokens); default is route-only.")
    scorer.add_argument("--model", default="gemini/gemini-3.1-pro-preview",
                        help="litellm-style model string, '<provider>/<model>'. "
                             "See llm_scoring.py --model for the Gemini/OpenRouter convention.")
    scorer.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature. Default: unset (model's own default).")
    scorer.add_argument("--raw-jsonl", default=None,
                        help="Path for raw per-sample predictions; pass 'none' to disable. "
                             "Default (unset): llm_scoring.py derives it from --out-csv.")

    ap.add_argument("--dry-run", action="store_true",
                    help="Write the sbatch script and print it without submitting.")
    return ap.parse_args()


def _scorer_args(args: argparse.Namespace) -> str:
    """Build the llm_scoring.py flag string from the forwarded scoring args."""
    parts = [
        f"--k {args.k}",
        f"--threads {args.threads}",
        f"--model {args.model}",
    ]
    if args.limit is not None:
        parts.append(f"--limit {args.limit}")
    if args.cot:
        parts.append("--cot")
    if args.score_steps:
        parts.append("--score-steps")
    if args.temperature is not None:
        parts.append(f"--temperature {args.temperature}")
    if args.raw_jsonl is not None:
        parts.append(f"--raw-jsonl {args.raw_jsonl}")
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
mamba activate env-protac-synth

cd {_PROJECT_ROOT}

echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"

uv run --extra llm_scoring {SCORER} \\
    {in_csv} \\
    {out_csv} \\
    {_scorer_args(args)}

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
