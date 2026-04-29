#!/usr/bin/env bash
# Regenerate every hero figure into docs/figures/.
#
# Usage:  scripts/regenerate_figures.sh  [--N <int>]
#
# Defaults to N=10 to keep wall-clock time reasonable on a laptop. Pass
# --N 16 to match the paper's experimental ring exactly (the Lindblad
# decay figure auto-clamps to N=10 for memory).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
EXTRA_ARGS=("$@")

for figure in decay gamma resonance bubbles imperfection; do
    echo "==> $figure"
    "$PYTHON" -m rydberg_trampoline.cli figures "$figure" -- "${EXTRA_ARGS[@]}"
done

echo
echo "Figures written to docs/figures/. Update README.md if a new PNG was added."
