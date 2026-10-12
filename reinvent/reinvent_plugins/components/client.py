"""Standard-library HTTP client for the RetroTAC scoring server.

This file lives in REINVENT's plugin tree but is deliberately *not* named
`comp_*.py`: REINVENT only imports those, so keeping the client under this
name stops it being registered as a scoring component. `comp_retrotac.py`
imports it.

Standard library only, on purpose -- the whole point of the server is to add
nothing to REINVENT's environment, and one JSON POST is `urllib`-sized work.
It is also runnable on its own::

    python client.py --smiles "CCO" "c1ccccc1" --key mean
    python client.py --input molecules.smi --port-file /tmp/retrotac.port

which is what you debug with before involving REINVENT at all.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_BATCH = 512


class RetroTACClient:
    """Posts SMILES to a RetroTAC server and returns its own response shape.

    Split deliberately into `predict` (one request, the full response) and
    `select` (one series out of a response), so a caller scoring several
    endpoints -- mean and std, say -- pays for one request and one
    featurization rather than one per endpoint.
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        port_file: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_batch: int = DEFAULT_MAX_BATCH,
        retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """Configure where and how to reach the server.

        Args:
            url: Base URL, e.g. "http://127.0.0.1:8000". Its port is replaced
                by `port_file`'s contents when that is given.
            port_file: File the server wrote its bound port to (`--port-file`),
                for when the port was chosen by the kernel (`--port 0`).
            timeout: Per-request timeout in seconds. A large batch of PROTACs
                through 27 members is slow; this is not a latency budget.
            max_batch: Molecules per request. Larger batches are chunked.
            retries: Attempts per chunk on a *connection* failure (server
                restarting). HTTP error statuses are never retried -- they
                will fail identically, and their body carries FastAPI's
                validation detail.
            retry_delay: Seconds between those attempts.
        """
        self.url = url.rstrip("/")
        self.port_file = port_file
        self.timeout = timeout
        self.max_batch = max_batch
        self.retries = retries
        self.retry_delay = retry_delay

    def endpoint(self, path: str) -> str:
        """Resolve a full URL for `path`, honouring `port_file` if set.

        Args:
            path: Server path, e.g. "/predict".

        Returns:
            The absolute URL to request.

        Raises:
            FileNotFoundError: If `port_file` is set but absent -- usually it
                means the server has not started yet, or was started without
                `--port-file`.
        """
        base = self.url
        if self.port_file:
            port = Path(self.port_file).read_text().strip()
            scheme, _, rest = base.partition("://")
            host = rest.split(":")[0].split("/")[0]
            base = f"{scheme}://{host}:{port}"
        return f"{base}{path}"

    def _request(self, path: str, payload: Optional[dict] = None) -> dict:
        """POST `payload` (or GET when None) and decode the JSON response."""
        url = self.endpoint(path)
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"} if data else {}

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            request = urllib.request.Request(url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                raise RuntimeError(
                    f"RetroTAC server returned HTTP {exc.code} for {url}: {body}"
                ) from exc
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.retry_delay)
        raise RuntimeError(
            f"Could not reach the RetroTAC server at {url} after {self.retries} attempts "
            f"({last_error}). Start it with: python -m retrotac.serve"
        )

    def info(self) -> dict:
        """Return the server's `/info` payload (what model it has loaded)."""
        return self._request("/info")

    def wait_until_ready(self, timeout: float = 60.0, poll: float = 1.0) -> None:
        """Block until `/health` answers, so a bad config fails before sampling.

        Called at component construction: a run pointed at a server that was
        never started must stop now, not thousands of steps in. The timeout is
        short because the usual cause is a wrong URL, not a slow load -- raise
        it if you routinely start the server and REINVENT together, since the
        ensemble takes a while to deserialize.

        Args:
            timeout: Seconds to keep polling before giving up.
            poll: Seconds between polls.

        Raises:
            RuntimeError: If the server never answers within `timeout`.
        """
        deadline = time.time() + timeout
        attempt_client = RetroTACClient(
            self.url, self.port_file, timeout=min(self.timeout, 10.0), retries=1
        )
        while True:
            try:
                attempt_client._request("/health")
                return
            except (RuntimeError, FileNotFoundError) as exc:
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"RetroTAC server not ready within {timeout:g}s: {exc}"
                    ) from exc
                time.sleep(poll)

    def predict(self, smiles: Sequence[str]) -> Dict[str, List[Optional[float]]]:
        """Score `smiles` in one request per `max_batch` chunk.

        Args:
            smiles: SMILES strings, in order.

        Returns:
            The server's own response: one list per model output key ("mean",
            "std", ...), each aligned with `smiles`, `None` where a molecule
            could not be scored.

        Raises:
            RuntimeError: If the response is not aligned with the request.
                One value per molecule is the invariant a generative loop
                cannot recover from -- it cannot tell which molecule lost its
                score -- so this is an error rather than a padded guess.

        Note:
            Chunking is not bit-reproducible: xgboost blocks prediction by row
            count, so a molecule scored in a chunk of 512 can differ from the
            same molecule scored in a chunk of 64 in the last ULP (~1e-16).
            Harmless for ranking; worth knowing when comparing logs.
        """
        smiles = list(smiles)
        if not smiles:
            return {}

        merged: Dict[str, List[Optional[float]]] = {}
        for start in range(0, len(smiles), self.max_batch):
            chunk = smiles[start:start + self.max_batch]
            response = self._request("/predict", {"smiles": chunk})
            for key, values in response.items():
                if len(values) != len(chunk):
                    raise RuntimeError(
                        f"RetroTAC server returned {len(values)} values for key {key!r} "
                        f"on {len(chunk)} molecules; scores cannot be realigned."
                    )
                merged.setdefault(key, []).extend(values)

        misaligned = {k: len(v) for k, v in merged.items() if len(v) != len(smiles)}
        if misaligned:
            raise RuntimeError(
                f"RetroTAC response lengths {misaligned} do not match {len(smiles)} molecules."
            )
        return merged

    @staticmethod
    def select(
        response: Dict[str, List[Optional[float]]], key: str
    ) -> List[float]:
        """Pull one series out of a `predict` response, `None` becoming `nan`.

        Args:
            response: A `predict` return value.
            key: Which model output to take, e.g. "mean" or "std".

        Returns:
            One float per molecule; `nan` for unscorable molecules, which is
            REINVENT's own convention for "do not score this".

        Raises:
            KeyError: If the server's response has no such key -- listing the
                keys it does have, since that is the whole diagnosis.
        """
        if key not in response:
            raise KeyError(
                f"RetroTAC server has no output key {key!r}; it returned {sorted(response)}"
            )
        return [math.nan if value is None else float(value) for value in response[key]]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Score SMILES from the command line, for debugging without REINVENT.

    Args:
        argv: Argument vector; `sys.argv[1:]` when omitted.

    Returns:
        Process exit status: 0 on success, 1 if the server could not be used.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--smiles", nargs="+", help="SMILES to score")
    source.add_argument("--input", type=Path, help="file with one SMILES per line")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"server base URL (default: {DEFAULT_URL})")
    ap.add_argument("--port-file", default=None, help="file holding the server's bound port")
    ap.add_argument("--key", default=None,
                    help="print just this output key (default: the whole response)")
    ap.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH,
                    help=f"molecules per request (default: {DEFAULT_MAX_BATCH})")
    ap.add_argument("--info", action="store_true", help="print /info and exit")
    args = ap.parse_args(argv)
    if not (args.info or args.smiles or args.input):
        ap.error("one of --smiles, --input or --info is required")

    client = RetroTACClient(args.url, port_file=args.port_file, max_batch=args.max_batch)
    try:
        if args.info:
            print(json.dumps(client.info(), indent=2))
            return 0
        smiles = (
            args.smiles
            if args.smiles
            else [line.strip() for line in args.input.read_text().splitlines() if line.strip()]
        )
        response = client.predict(smiles)
        if args.key:
            for smi, value in zip(smiles, client.select(response, args.key)):
                print(f"{value:.6f}\t{smi}")
        else:
            print(json.dumps(response, indent=2))
    except (RuntimeError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
