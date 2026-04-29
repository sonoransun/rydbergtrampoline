"""Hero-figure generators reproducing key plots from Chao et al. (2026).

Each ``fig_<name>.py`` is a runnable module: ``python -m
rydberg_trampoline.figures.fig_<name>`` writes a PNG to the
``--out`` directory (default ``docs/figures/``) and a ``.json`` sidecar
with the run parameters.

Use the CLI ``rydberg-trampoline figures all`` to regenerate every figure.
"""
from pathlib import Path

DEFAULT_FIG_DIR = Path(__file__).resolve().parents[2] / "docs" / "figures"
