"""Shared utilities for the figure scripts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from rydberg_trampoline.figures import DEFAULT_FIG_DIR


def common_argparser(description: str) -> argparse.ArgumentParser:
    """Return an ``ArgumentParser`` with ``--out`` and ``--no-show`` flags."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help=f"directory for the rendered PNG (default: {DEFAULT_FIG_DIR})",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=("numpy", "qutip", "quspin"),
        help="closed-system backend for the simulation (default: numpy)",
    )
    p.add_argument(
        "--N",
        type=int,
        default=10,
        help="atoms in the ring (default: 10 — keeps figures fast)",
    )
    return p


def save_figure_with_sidecar(fig: plt.Figure, out: Path, stem: str, params: dict) -> Path:
    """Write ``out/stem.png`` plus a ``out/stem.json`` sidecar with ``params``."""
    out.mkdir(parents=True, exist_ok=True)
    png_path = out / f"{stem}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    sidecar = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stem": stem,
        "params": params,
    }
    (out / f"{stem}.json").write_text(json.dumps(sidecar, indent=2))
    return png_path


def style_axes(ax: plt.Axes) -> None:
    """Apply consistent styling across hero figures."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, linestyle=":")


def overlay_experimental(ax: plt.Axes, dataset_name: str, *, x: str, y: str, **kwargs):
    """Overlay digitised experimental points onto ``ax`` if the dataset exists.

    No-ops with a one-line stderr message when the CSV is missing — this
    keeps the figure pipeline working before any data has been digitised.
    Returns ``True`` iff an overlay was drawn.
    """
    from rydberg_trampoline.data.loader import load_experimental_csv

    ds = load_experimental_csv(dataset_name)
    if ds is None:
        return False
    xs = ds.column(x)
    ys = ds.column(y)
    style = dict(marker="x", linestyle="", color="#c0392b", markersize=6,
                 label=f"experiment (digitised, Fig. {dataset_name})")
    style.update(kwargs)
    ax.plot(xs, ys, **style)
    return True
