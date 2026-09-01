"""Rewrite the embedded package path inside saved .skops archives.

A .skops file is a zip whose schema.json records the fully qualified module
of the pickled class. Models saved before the rename record
`protac_synth.models.{xgb,mlp}.model`, which no longer imports. This rewrites
that string and copies every other member unchanged. Nothing checksums the
archive, so the edit is safe.

Idempotent: already-migrated files are skipped.

    python scripts/maintenance/migrate_skops_module_path.py --dry-run
    python scripts/maintenance/migrate_skops_module_path.py
    python scripts/maintenance/migrate_skops_module_path.py --revert
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

OLD = '"protac_synth.'
NEW = '"retrotac.'
SCHEMA = "schema.json"


def needs_migration(path: Path) -> bool:
    """True when the archive's schema still names the old package."""
    with zipfile.ZipFile(path) as zf:
        return OLD in zf.read(SCHEMA).decode()


def migrate(path: Path, dry_run: bool) -> bool:
    """Rewrite one archive in place, keeping a .bak sidecar. True if changed."""
    if not needs_migration(path):
        return False
    if dry_run:
        return True

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == SCHEMA:
                data = data.decode().replace(OLD, NEW).encode()
            zout.writestr(item, data)
    tmp.replace(path)
    return True


def revert(path: Path) -> bool:
    """Restore one archive from its .bak sidecar. True if restored."""
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        return False
    shutil.copy2(backup, path)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs", help="Directory to scan.")
    ap.add_argument("--dry-run", action="store_true", help="Report, change nothing.")
    ap.add_argument("--revert", action="store_true", help="Restore from .bak files.")
    args = ap.parse_args()

    files = sorted(Path(args.root).rglob("*.skops"))
    if not files:
        print(f"No .skops files under {args.root}/")
        return 1

    changed = 0
    for path in files:
        if args.revert:
            if revert(path):
                changed += 1
                print(f"  reverted {path}")
            continue
        if migrate(path, args.dry_run):
            changed += 1
            print(f"  {'would migrate' if args.dry_run else 'migrated'} {path}")

    verb = "reverted" if args.revert else ("would migrate" if args.dry_run else "migrated")
    print(f"{verb} {changed} of {len(files)} .skops files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
