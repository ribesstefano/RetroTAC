"""Backward compatibility for artifacts saved under the old package name.

Models trained before the rename embed `protac_synth.models.*.model` in their
.skops archives (see docs/superpowers/specs/2026-09-01-retrotac-rename-design.md).
scripts/maintenance/migrate_skops_module_path.py rewrites the archives we can
reach; this shim covers the ones we cannot, such as copies published to the
Hugging Face Hub.

Temporary. Delete once no unmigrated artifact remains in circulation.
"""

import sys

LEGACY_NAME = "protac_synth"


def install_legacy_aliases() -> None:
    """Make `protac_synth[...]` imports resolve to `retrotac[...]`.

    Only the top-level package is aliased. Python then resolves submodules on
    demand through the package's __path__, which keeps the backend imports
    lazy: aliasing `retrotac.models.mlp.model` here would pull in torch on
    every import of this package.
    """
    if LEGACY_NAME in sys.modules:
        return
    sys.modules[LEGACY_NAME] = sys.modules[__name__.rsplit(".", 1)[0]]
