"""FastAPI application serving the RetroTAC ensemble over HTTP.

Why a server: REINVENT4 (and any other generative loop) pins its own torch
and rdkit, and a 27-member ensemble costs seconds to deserialize. A separate
process keeps the dependency sets from meeting -- torch and xgboost each
bundle a libomp and segfault when both run multi-threaded in one process --
and keeps the model warm across the thousands of scoring calls a run makes.

Three design points worth not undoing:

* **One worker, one lock.** Each uvicorn worker would hold its own copy of
  the 1.4 GiB ensemble, so this serves from a single worker and scales with
  batch size rather than concurrency. Endpoints are plain `def`, so starlette
  runs them in a threadpool, and `ScoringService` takes a lock around predict
  so concurrent batches cannot interleave inside the members.
* **Everything knowable at load time is checked at load time.** A feature-width
  or artifact mismatch otherwise surfaces as a 500 on *every* request for the
  life of the server. `ScoringService` scores one reference molecule during
  construction, so a broken ensemble fails before the socket is listening.
* **An unparseable SMILES is data, not an error.** It comes back as `null` in
  place, keeping the response aligned with the request; a 500 would kill a
  generative run at step 1400 over one malformed string.

Note: this module deliberately does *not* use `from __future__ import
annotations`. FastAPI resolves an endpoint's parameter annotations at
decoration time, and the schema classes are imported inside `create_app` to
keep this module importable without the `serve` extra -- stringized
annotations would leave FastAPI unable to find them in that local scope.
"""
import logging
import math
import socket
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# One real PROTAC (the README's example molecule), scored once at startup to
# prove the whole featurize -> member -> weighted-mean path works.
VALIDATION_SMILES = (
    "COc1ccc(-c2ocnc2C(=O)NCCCc2cn(CCOCCOCCNc3cccc4c3C(=O)N(C3CCC(=O)NC3=O)C4=O)nn2)cc1I"
)

# Per-member raw predictions: 27 arrays per request, useful for debugging a
# single molecule but not something to put on the wire by default.
DETAIL_KEY_MEMBERS = "members"


def _package_version() -> str:
    """Installed retrotac version, or "unknown" outside an install."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("retrotac")
    except PackageNotFoundError:
        return "unknown"


class ScoringService:
    """A loaded RetroTAC ensemble, serialized behind a lock.

    Holds the model, the connection-independent prediction knobs, and the
    SMILES-validity bookkeeping that keeps a response aligned with its
    request. Construction is where load-time validation happens, so building
    one is the expensive, failure-prone step and `predict` is neither.
    """

    def __init__(
        self,
        ensemble: Any,
        source: str,
        *,
        n_jobs: int = 1,
        gnn_batch_size: int = 64,
    ):
        """Wrap a loaded ensemble and validate it against one reference molecule.

        Args:
            ensemble: A `retrotac.models.ensemble.RetroTAC` instance.
            source: Where it came from (Hub repo id or local directory), for `/info`.
            n_jobs: Featurization parallelism forwarded to `RetroTAC.predict`.
                Left at 1 by default: a server already has its own process
                and worker pools per request cost more than they save.
            gnn_batch_size: Molecules per GNN forward pass.

        Raises:
            RuntimeError: If scoring the reference molecule fails or returns
                anything other than one finite value per output key -- the
                symptom of a feature-width or artifact mismatch, which would
                otherwise 500 on every request instead.
        """
        self._ensemble = ensemble
        self.source = source
        self._n_jobs = n_jobs
        self._gnn_batch_size = gnn_batch_size
        self._lock = threading.Lock()
        self.output_keys = self._validate()

    def _validate(self) -> List[str]:
        """Score one reference molecule and return the ensemble's output keys."""
        try:
            scored = self._score([VALIDATION_SMILES])
        except Exception as exc:  # noqa: BLE001 -- re-raised with a diagnosis
            raise RuntimeError(
                f"Start-up validation failed: the ensemble at {self.source!r} could not "
                f"score a reference molecule ({exc}). A 'Feature shape mismatch' here "
                "means config.json's fp_size/fp_radius/use_* block disagrees with what "
                "the members were fitted on; a missing-file error means the artifact "
                "directory is incomplete."
            ) from exc

        broken = {key: array.shape for key, array in scored.items() if array.shape != (1,)}
        if broken:
            raise RuntimeError(
                f"Start-up validation failed: the ensemble at {self.source!r} returned "
                f"{broken} for one molecule, expected one value per output key."
            )
        return list(scored)

    def _score(self, smiles: List[str]) -> Dict[str, np.ndarray]:
        """Run the ensemble on parseable SMILES only, dropping the member block."""
        details = self._ensemble.predict(
            smiles,
            return_details=True,
            n_jobs=self._n_jobs,
            gnn_batch_size=self._gnn_batch_size,
        )
        return {
            key: np.asarray(value, dtype=float).reshape(-1)
            for key, value in details.items()
            if key != DETAIL_KEY_MEMBERS
        }

    def predict(self, smiles: List[str]) -> Dict[str, List[Optional[float]]]:
        """Score a batch, returning one value (or `null`) per input SMILES.

        Unparseable SMILES are filtered out before the model sees them --
        `compute_fingerprints` would otherwise turn them into an all-zero row
        and the ensemble would happily score that -- and come back as `null`.
        Non-finite outputs become `null` too, `NaN` not being valid JSON.

        Args:
            smiles: SMILES strings, in the caller's order.

        Returns:
            One list per ensemble output key, each the same length as `smiles`.

        Raises:
            RuntimeError: If the ensemble returns a different number of values
                than molecules it was given. One value per molecule is the
                invariant a caller cannot recover from, since it cannot tell
                which molecule lost its score.
        """
        from retrotac.chem_utils import std_smiles

        valid = [i for i, smi in enumerate(smiles) if std_smiles(smi) is not None]
        if len(valid) < len(smiles):
            logger.info("%d/%d SMILES unparseable, scored as null", len(smiles) - len(valid), len(smiles))

        if not valid:
            return {key: [None] * len(smiles) for key in self.output_keys}

        with self._lock:
            scored = self._score([smiles[i] for i in valid])

        response: Dict[str, List[Optional[float]]] = {}
        for key, array in scored.items():
            if array.shape != (len(valid),):
                raise RuntimeError(
                    f"Ensemble returned {array.shape[0]} values for key {key!r} on "
                    f"{len(valid)} molecules; predictions cannot be realigned."
                )
            column: List[Optional[float]] = [None] * len(smiles)
            for position, value in zip(valid, array.tolist()):
                column[position] = value if math.isfinite(value) else None
            response[key] = column
        return response

    def info(self) -> Dict[str, Any]:
        """Report what is loaded, in enough detail to diagnose a mismatch."""
        members = self._ensemble.members
        return {
            "source": self.source,
            "retrotac_version": _package_version(),
            "n_members": len(members),
            "backends": dict(Counter(spec["backend"] for spec in members.values())),
            "feature_config": dict(self._ensemble.feature_config),
            "output_keys": self.output_keys,
            "device": self._ensemble.device,
            "load_strategy": self._ensemble.load_strategy,
        }


def create_app(service: ScoringService, on_shutdown: Optional[Callable[[], None]] = None) -> Any:
    """Build the FastAPI app around an already-validated service.

    Endpoints are declared `def` rather than `async def` on purpose: starlette
    then runs each in its own threadpool thread, so a long batch does not
    block the event loop (and `/health` keeps answering), while the service's
    own lock still serializes the model.

    Args:
        service: A constructed `ScoringService` -- the model is loaded and
            validated before the app exists, so the app itself cannot 500 on
            a start-up problem.
        on_shutdown: Cleanup to run during the app's lifespan shutdown. This
            is the *only* place cleanup can live: uvicorn re-raises the
            signal it captured once shutdown completes, which terminates the
            process without unwinding `run_server`'s stack, so a `finally`
            there never runs on a SIGTERM/SIGINT.

    Returns:
        The FastAPI application.
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from retrotac.serve.schemas import InfoResponse, PredictRequest, PredictResponse

    @asynccontextmanager
    async def lifespan(app: Any):
        """Serve, then run `on_shutdown` on the way out."""
        yield
        if on_shutdown is not None:
            on_shutdown()

    app = FastAPI(
        lifespan=lifespan,
        title="RetroTAC scoring server",
        version=_package_version(),
        description=(
            "PROTAC synthesizability from SMILES, as the Caruana-weighted ensemble mean "
            "plus the spread across its members. One value per submitted SMILES; `null` "
            "where a molecule could not be scored."
        ),
    )

    @app.get("/health", summary="Liveness check, no model work")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/info", response_model=InfoResponse, summary="What this process has loaded")
    def info() -> Dict[str, Any]:
        return service.info()

    @app.post("/predict", response_model=PredictResponse, summary="Score a batch of SMILES")
    def predict(request: PredictRequest) -> Dict[str, List[Optional[float]]]:
        return service.predict(request.smiles)

    return app


def run_server(
    service: ScoringService,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    port_file: Optional[Path] = None,
    log_level: str = "info",
) -> None:
    """Serve `service` with uvicorn on a socket bound here, not by uvicorn.

    Binding first is what makes `--port 0` usable: the kernel picks a free
    port, this function reads it back and writes it to `port_file`, and only
    then hands the listening socket over. Several scoring jobs can then share
    a node without coordinating port numbers, each pointing its client at its
    own port file.

    Args:
        service: The validated scoring service.
        host: Interface to bind. Defaults to loopback -- there is no
            authentication here, so `0.0.0.0` on a shared node would expose
            inference to the whole network.
        port: Port to bind, or 0 to let the kernel choose.
        port_file: Path to write the bound port to. Removed during the app's
            lifespan shutdown, so a client never reads a port that is gone.
        log_level: uvicorn log level.
    """
    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    bound_port = sock.getsockname()[1]

    if port_file is not None:
        port_file.parent.mkdir(parents=True, exist_ok=True)
        port_file.write_text(f"{bound_port}\n")
    logger.info("Serving RetroTAC on http://%s:%d (docs at /docs)", host, bound_port)

    def cleanup() -> None:
        """Drop the port file and the listening socket on the way out."""
        if port_file is not None:
            port_file.unlink(missing_ok=True)
        sock.close()

    config = uvicorn.Config(
        create_app(service, on_shutdown=cleanup), log_level=log_level, workers=1
    )
    uvicorn.Server(config).run(sockets=[sock])
