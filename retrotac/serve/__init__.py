"""HTTP serving layer for the RetroTAC ensemble (optional `serve` extra).

Importing this package pulls in nothing heavier than the standard library:
`app` and `schemas` import fastapi/pydantic at module level, so they are
resolved lazily on first attribute access. That keeps `python -m
retrotac.serve --help` working in an environment without the `serve` extra
installed, and keeps `import retrotac.serve` free for callers that only want
`ScoringService`'s docstring.
"""
from typing import Any

__all__ = ["ScoringService", "create_app", "run_server"]


def __getattr__(name: str) -> Any:
    """Resolve the fastapi-dependent symbols on first access."""
    if name in __all__:
        from retrotac.serve import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
