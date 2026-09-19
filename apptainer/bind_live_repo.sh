#!/bin/bash
# Prints one --bind flag per top-level repo entry, for mounting the live
# checkout over the image's baked-in snapshot at /opt/repo (see
# apptainer/README.md, "Why bind-mount instead of rebuild"). external/ is
# skipped so apptainer/scoring.def's baked vendor scorer weights (SCScore,
# GASA -- see retro_scores/README.md) aren't shadowed by an absent or
# incomplete host copy; .git/.venv/__pycache__ are skipped as irrelevant
# inside the container.
#
# Usage:
#   apptainer exec $(bash apptainer/bind_live_repo.sh) [--nv] IMAGE.sif CMD...
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXCLUDE=(external .git .venv __pycache__)

binds=()
for entry in "$REPO_ROOT"/*; do
    name="$(basename "$entry")"
    skip=false
    for x in "${EXCLUDE[@]}"; do
        [[ "$name" == "$x" ]] && skip=true && break
    done
    $skip && continue
    binds+=("--bind" "$entry:/opt/repo/$name")
done

printf '%s\n' "${binds[*]}"
