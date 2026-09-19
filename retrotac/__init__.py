"""RetroTAC: PROTAC synthesizability prediction (surrogate-model package).

Root package for the surrogate-model half of the RetroTAC pipeline (see the
repo's CLAUDE.md / CONTRIBUTING.md for the full pipeline). Synthesizability
*scoring* -- SA/SC/RA/SYBA/GASA/FS, route-tree, HAC-weighted, LLM-judge --
lives in the separate `retro_scores` package, not here.

Most users only need one top-level symbol, the Caruana-weighted xgb/mlp/gnn
ensemble (see `retrotac.models.ensemble.RetroTAC`)::

    from retrotac import RetroTAC

    model = RetroTAC.from_pretrained("ribesstefano/retrotac")
    scores = model.predict(["CCO", "c1ccccc1"])

`RetroTAC` is resolved lazily via module `__getattr__` (PEP 562) rather than
imported eagerly here, so that `import retrotac` -- and imports of unrelated
submodules such as `retrotac.chem_utils` -- stay cheap. Accessing `RetroTAC`
still pays the full cost of importing `retrotac.models` (which eagerly loads
every backend -- xgb, mlp, gnn -- and therefore torch/xgboost/lightning,
regardless of which one you actually use; see `retrotac.models.__init__`'s
docstring note), it just defers that cost until the symbol is actually used
instead of paying it on every `import retrotac`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from retrotac._compat import install_legacy_aliases

install_legacy_aliases()

if TYPE_CHECKING:
    from retrotac.models.ensemble import RetroTAC  # noqa: F401

__all__ = ["RetroTAC"]


def __getattr__(name: str):
    """Lazily resolve top-level symbols on first access (PEP 562).

    Only `RetroTAC` is exposed this way today. Keeping it out of the eager
    module-level imports above is what lets `import retrotac` (and imports
    of sibling submodules like `retrotac.chem_utils`) avoid pulling in
    torch/xgboost/lightning until `RetroTAC` is actually used.
    """
    if name == "RetroTAC":
        from retrotac.models.ensemble import RetroTAC

        return RetroTAC
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + __all__)
