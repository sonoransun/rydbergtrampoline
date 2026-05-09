"""Smoke tests for the static-schematic generators in ``scripts/``.

Each ``scripts/draw_*.py`` is a one-shot matplotlib generator that
writes a PNG to ``docs/figures/``. These tests run each generator
end-to-end to catch import / API drift; the resulting PNG bytes are
also pinned by ``tests/test_figure_regression.py`` against
``tests/figure_hashes.json`` (perceptual hash, ≤ 6 Hamming distance).

We invoke the scripts via ``subprocess`` rather than importing their
``main`` so they exercise the same code path as
``scripts/regenerate_figures.sh`` and the manual regen workflow.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"


# Only the three new generators added in this round; the original five are
# already covered indirectly by `tests/test_figure_regression.py` and don't
# need duplicate import-smoke coverage.
NEW_SCHEMATIC_SCRIPTS = [
    ("draw_bubble_pedagogy.py", "bubble_pedagogy.png"),
    ("draw_itebd_unit_cell.py", "itebd_unit_cell.png"),
    ("draw_neel_prep_ramp.py", "neel_prep_ramp.png"),
]


@pytest.mark.parametrize("script_name,output_png", NEW_SCHEMATIC_SCRIPTS)
def test_draw_schematic_runs(script_name: str, output_png: str) -> None:
    """Each new draw_*.py runs to completion and produces its PNG."""
    script = SCRIPTS_DIR / script_name
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        env={"MPLBACKEND": "Agg", **__import__("os").environ},
    )
    assert proc.returncode == 0, (
        f"scripts/{script_name} exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # The script should announce its output path.
    assert output_png in proc.stdout, (
        f"expected '{output_png}' to appear in stdout, got:\n{proc.stdout}"
    )
    # And the PNG should now exist on disk.
    assert (FIGURES_DIR / output_png).is_file()
