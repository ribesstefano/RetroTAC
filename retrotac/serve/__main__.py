"""Entry point for the RetroTAC scoring server: `python -m retrotac.serve`.

Two things happen here that cannot happen anywhere later:

* `OMP_NUM_THREADS=1` is set *before* torch or xgboost is imported. The
  ensemble holds both, each with its own bundled libomp, and the pair
  segfaults when both run multi-threaded in one process. After the import the
  variable has no effect, which is why this module imports nothing from
  `retrotac` at module level.
* `TQDM_DISABLE=1` silences the per-request progress bars
  (`RetroTAC.predict` draws one across its 27 members, `compute_features`
  three more). A generative run makes thousands of requests; `--progress`
  puts them back for an interactive debugging session.

Argument parsing precedes the fastapi import, so `--help` works in an
environment without the `serve` extra installed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

HF_REPO = "ailab-bio/RetroTAC"


def parse_args() -> argparse.Namespace:
    """Parse the server's command-line arguments.

    Returns:
        argparse.Namespace with the parsed arguments.
    """
    ap = argparse.ArgumentParser(
        prog="python -m retrotac.serve",
        description="Serve the RetroTAC PROTAC-synthesizability ensemble over HTTP.",
    )
    ap.add_argument("--model", default=HF_REPO,
                    help=f"Hub repo id or local ensemble directory (default: {HF_REPO})")
    ap.add_argument("--host", default="127.0.0.1",
                    help="interface to bind; loopback by default, as there is no auth "
                         "(default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000,
                    help="port to bind, or 0 to let the kernel pick one (default: 8000)")
    ap.add_argument("--port-file", type=Path, default=None,
                    help="write the bound port here (removed on shutdown); pair with "
                         "--port 0 to run several servers on one node")
    ap.add_argument("--device", default="cpu", help="device for the gnn members (default: cpu)")
    ap.add_argument("--load-strategy", choices=["eager", "lazy", "stream"], default="eager",
                    help="member loading strategy; eager keeps all 27 resident (default: eager)")
    ap.add_argument("--max-resident", type=int, default=None,
                    help="with --load-strategy lazy: cap on members held in memory")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="featurization workers per request; 1 avoids per-request pool "
                         "spawn-up, which a warm server does not need (default: 1)")
    ap.add_argument("--gnn-batch-size", type=int, default=64,
                    help="molecules per gnn forward pass (default: 64)")
    ap.add_argument("--log-level", default="info",
                    choices=["critical", "error", "warning", "info", "debug", "trace"],
                    help="uvicorn log level (default: info)")
    ap.add_argument("--progress", action="store_true",
                    help="keep the per-request tqdm bars (debugging; noisy under load)")
    return ap.parse_args()


def main() -> None:
    """Load the ensemble, validate it, and serve it until interrupted.

    Raises:
        SystemExit: If the `serve` extra is not installed, or if the ensemble
            fails its start-up validation -- both reported as a message rather
            than a traceback, since both are configuration problems.
    """
    args = parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if not args.progress:
        os.environ.setdefault("TQDM_DISABLE", "1")

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"The RetroTAC server needs the optional 'serve' extra ({exc.name} is missing).\n"
            "Install it with:  uv sync --extra serve"
        ) from exc

    import logging

    from retrotac.models.ensemble import RetroTAC
    from retrotac.serve.app import ScoringService, run_server

    logging.basicConfig(level=args.log_level.upper() if args.log_level != "trace" else "DEBUG")

    ensemble = RetroTAC.from_pretrained(
        args.model,
        device=args.device,
        load_strategy=args.load_strategy,
        max_resident=args.max_resident,
    )
    try:
        service = ScoringService(
            ensemble,
            source=args.model,
            n_jobs=args.n_jobs,
            gnn_batch_size=args.gnn_batch_size,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    run_server(
        service,
        host=args.host,
        port=args.port,
        port_file=args.port_file,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
