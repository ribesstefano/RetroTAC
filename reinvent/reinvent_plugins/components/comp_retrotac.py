"""REINVENT4 scoring component: PROTAC synthesizability from a RetroTAC server.

Mechanics that REINVENT's plugin loader imposes, each of which makes the
component *silently absent* when broken rather than raising something that
names the cause:

* The file must be named `comp_*.py` -- only those are imported.
* There must be no `__init__.py` in `reinvent_plugins/` or `components/`:
  they are native namespace packages, which is exactly what merges this
  directory with REINVENT's own. An init file shadows REINVENT's package and
  breaks its built-in components instead.
* `PYTHONPATH` must point at the directory *containing* `reinvent_plugins`
  (here: `reinvent/`), not at `components/`. Nothing is copied into
  REINVENT's install.

One request per population, not per endpoint: `Parameters` fields are
parallel lists with one entry per endpoint, so `key` varies per endpoint
while the connection settings (`url`, `port_file`, `timeout`, `max_batch`)
describe the one server this component talks to. Conflicting connection
values are refused rather than quietly honoured -- honouring them would mean
one request each, re-featurizing the same molecules per endpoint, which is
the whole thing this design avoids. A second server needs a second component
block.
"""
from __future__ import annotations

__all__ = ["RetroTAC"]

from dataclasses import dataclass, field, fields
from typing import Any, List, Optional

import numpy as np

from .add_tag import add_tag
from .client_retrotac import DEFAULT_MAX_BATCH, DEFAULT_TIMEOUT, DEFAULT_URL, RetroTACClient
from .component_results import ComponentResults

# Scoring-component defaults that are not the client's own: which model output
# an endpoint reads, and whether a dead server stops the run.
DEFAULT_KEY = "mean"
DEFAULT_READY_TIMEOUT = 60.0


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parallel per-endpoint lists, as REINVENT assembles them from the TOML.

    A key omitted from the TOML arrives as a short list or a list of `None`,
    so every field is read through `_per_endpoint`/`_shared`, never indexed
    directly.

    Attributes:
        key: Model output this endpoint scores -- "mean" (the synthesizability
            prediction) or "std" (the spread across ensemble members). Per
            endpoint.
        url: Base URL of the server. Shared across endpoints.
        port_file: File the server wrote its bound port to, for a server
            started with `--port 0`. Shared across endpoints.
        timeout: Per-request timeout in seconds. Shared across endpoints.
        max_batch: Molecules per request; larger populations are chunked.
            Shared across endpoints.
        ready_timeout: Seconds to wait for `/health` at construction. Shared.
        fail_on_error: If True (default), a server failure raises and stops
            the run. False scores the batch `nan` and carries on, which
            survives a server restart but silently wastes every remaining
            step if the server stays down. Shared across endpoints.
    """

    key: List[Optional[str]] = field(default_factory=list)
    url: List[Optional[str]] = field(default_factory=list)
    port_file: List[Optional[str]] = field(default_factory=list)
    timeout: List[Optional[float]] = field(default_factory=list)
    max_batch: List[Optional[int]] = field(default_factory=list)
    ready_timeout: List[Optional[float]] = field(default_factory=list)
    fail_on_error: List[Optional[bool]] = field(default_factory=list)


def _endpoint_count(parameters: Parameters) -> int:
    """Number of endpoints: the longest per-endpoint list, or 1 if all empty."""
    lengths = [len(getattr(parameters, f.name) or []) for f in fields(parameters)]
    return max(lengths) or 1


def _per_endpoint(values: Optional[List[Any]], index: int, default: Any) -> Any:
    """Read one endpoint's value from a possibly short or `None`-filled list."""
    if not values or index >= len(values) or values[index] is None:
        return default
    return values[index]


def _shared(values: Optional[List[Any]], name: str, default: Any) -> Any:
    """Collapse a per-endpoint list that must hold one single shared value.

    Args:
        values: The raw per-endpoint list from the TOML.
        name: Parameter name, for the error message.
        default: Value to use when no endpoint sets it.

    Returns:
        The one value every endpoint that set it agreed on, else `default`.

    Raises:
        ValueError: If two endpoints set it differently. Quietly picking one
            would mean a request per endpoint instead of one per population;
            point the second endpoint at a second component block instead.
    """
    distinct = {v for v in (values or []) if v is not None}
    if len(distinct) > 1:
        raise ValueError(
            f"RetroTAC component: conflicting {name!r} values {sorted(map(str, distinct))}. "
            f"{name} describes the one server this component talks to, since all its "
            "endpoints are served by a single request. Use a second RetroTAC component "
            "block to score against a second server."
        )
    return distinct.pop() if distinct else default


@add_tag("__component")
class RetroTAC:
    """Scores PROTAC synthesizability by calling a RetroTAC server once per call.

    Construction checks the server is reachable, so a run configured against
    a server that was never started stops before it samples anything.
    """

    def __init__(self, parameters: Parameters):
        """Resolve per-endpoint keys and shared connection settings.

        Args:
            parameters: The component's parsed TOML parameters.

        Raises:
            ValueError: If endpoints disagree on a connection setting.
            RuntimeError: If the server does not answer `/health` in time.
        """
        n_endpoints = _endpoint_count(parameters)
        self.keys = [
            _per_endpoint(parameters.key, i, DEFAULT_KEY) for i in range(n_endpoints)
        ]
        self.fail_on_error = bool(_shared(parameters.fail_on_error, "fail_on_error", True))

        self.client = RetroTACClient(
            url=_shared(parameters.url, "url", DEFAULT_URL),
            port_file=_shared(parameters.port_file, "port_file", None),
            timeout=float(_shared(parameters.timeout, "timeout", DEFAULT_TIMEOUT)),
            max_batch=int(_shared(parameters.max_batch, "max_batch", DEFAULT_MAX_BATCH)),
        )
        self.client.wait_until_ready(
            timeout=float(_shared(parameters.ready_timeout, "ready_timeout", DEFAULT_READY_TIMEOUT))
        )

    def __call__(self, smilies: List[str]) -> ComponentResults:
        """Score one population: a single request, then one slice per endpoint.

        Args:
            smilies: SMILES of the population to score (REINVENT's spelling).

        Returns:
            ComponentResults holding one float array per endpoint, `nan` for
            molecules the server could not score.

        Raises:
            RuntimeError: On a server failure, unless `fail_on_error` is False.
        """
        try:
            response = self.client.predict(smilies)
        except (RuntimeError, KeyError, FileNotFoundError):
            if self.fail_on_error:
                raise
            nan_scores = np.full(len(smilies), np.nan, dtype=float)
            return ComponentResults([nan_scores.copy() for _ in self.keys])

        return ComponentResults(
            [np.array(self.client.select(response, key), dtype=float) for key in self.keys]
        )
