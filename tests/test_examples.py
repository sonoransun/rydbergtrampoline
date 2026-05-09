"""Smoke tests for the standalone scripts under ``examples/``.

Each script is invoked via ``subprocess.run`` with
``RYDBERG_TRAMPOLINE_TEST_MODE=1`` so it clamps grids and finishes in
~1 s. Backends required by individual scripts are gated via
``available_backends()``; missing extras cause clean skips, not
failures.

Also pins that every executable script in ``examples/`` is documented
in ``examples/README.md``, which catches drift if a new example is
added without listing.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def _example_files() -> list[Path]:
    return sorted(
        p for p in EXAMPLES_DIR.glob("*.py")
        if p.name not in ("__init__.py",)
    )


def test_examples_listed_in_readme() -> None:
    """Every example script must appear in examples/README.md."""
    readme_text = (EXAMPLES_DIR / "README.md").read_text(encoding="utf-8")
    for path in _example_files():
        assert path.name in readme_text, (
            f"{path.name} is in examples/ but not mentioned in examples/README.md"
        )


@pytest.mark.parametrize(
    "script_name", [p.name for p in _example_files()]
)
def test_example_runs(script_name: str, tmp_path: Path) -> None:
    """Run each example end-to-end at the test-mode config."""
    script = EXAMPLES_DIR / script_name
    env = os.environ.copy()
    env["RYDBERG_TRAMPOLINE_TEST_MODE"] = "1"
    # Force a non-interactive matplotlib backend in case any example imports
    # pyplot transitively (these scripts don't open windows but stay safe).
    env.setdefault("MPLBACKEND", "Agg")
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"examples/{script_name} exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
