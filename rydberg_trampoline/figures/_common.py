"""Shared utilities for the figure scripts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from rydberg_trampoline.figures import DEFAULT_FIG_DIR


PALETTE = {
    "ink":     "#2c3e50",
    "ink_alt": "#34495e",
    "accent1": "#c0392b",
    "accent2": "#16a085",
    "accent3": "#e67e22",
    "accent4": "#2980b9",
    "muted":   "#7f8c8d",
}


def apply_paper_rcparams() -> None:
    """Install hero-figure rcParams. Idempotent and PNG-byte-stable for the
    scripts shipped with this package — the values match what fig_*.py scripts
    already establish per-axis via :func:`style_axes` and explicit kwargs."""
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
    })


def common_argparser(
    description: str, *, add_n_values: bool = False
) -> argparse.ArgumentParser:
    """Return an ``ArgumentParser`` with ``--out`` and ``--no-show`` flags.

    Pass ``add_n_values=True`` to additionally accept ``--N-values N1 N2 ...``
    for figure scripts that overlay multiple ring sizes (e.g. the finite-size
    scaling figures). The singleton ``--N`` is always present.
    """
    apply_paper_rcparams()
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
    if add_n_values:
        p.add_argument(
            "--N-values",
            type=int,
            nargs="+",
            default=None,
            help="ring sizes for N-sweep figures (overrides --N if given)",
        )
    return p


def pick_unitary_backend(N: int) -> tuple[str, dict]:
    """Pick the cheapest closed-system unitary backend for ring size ``N``.

    Returns ``(backend_name, kwargs)`` ready to splat into
    :func:`run_unitary` (e.g. ``run_unitary(params, t, backend=name, **kwargs)``).

    Selection rule:

    * ``N ≤ 12`` → ``("numpy", {})``. Full ED is faster than the QuSpin
      sector setup at small N.
    * ``13 ≤ N ≤ 22`` → ``("quspin", {"kblock": 0})``. The Néel false-vacuum
      lives in the ``(kblock=0, pblock=+1)`` sector under translation-by-2
      (see ``backends/quspin_backend.py``); restricting to that sector cuts
      the dimension by ``N // 2``. ``pblock`` is left ``None`` because the
      bond-inversion / translation-by-2 algebra emits a documented
      ``GeneralBasisWarning`` (CLAUDE.md ``QuSpin`` note).
    * ``N > 22`` raises — closed-system unitary ED on a single ring of that
      size is out of laptop budget. Use :func:`run_itebd` for the
      thermodynamic limit or the bloqade cloud path for true finite N.

    Falls back to NumPy when QuSpin is not importable and ``N ≤ 18``
    (NumPy ED still fits in laptop RAM at those sizes).
    """
    if N > 22:
        raise ValueError(
            f"N={N} exceeds the closed-system finite-N ED budget; "
            "use run_itebd for the thermodynamic limit, or the bloqade "
            "cloud path (psi0_protocol='neel_via_ramp') for true finite N."
        )
    if N <= 12:
        return "numpy", {}
    from rydberg_trampoline.backends import available_backends

    if "quspin" in available_backends():
        return "quspin", {"kblock": 0}
    if N <= 18:
        return "numpy", {}
    raise ModuleNotFoundError(
        f"N={N} requires the quspin backend (kblock=0 sector). "
        "Install with: pip install 'rydberg-trampoline[quspin]'"
    )


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
    """Apply consistent styling across hero figures.

    Spines and grid styling are also installed globally by
    :func:`apply_paper_rcparams`; this per-axis call additionally enables the
    grid (which rcParams cannot do without affecting `axis('off')` schematics).
    """
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
    style = dict(marker="x", linestyle="", color=PALETTE["accent1"], markersize=6,
                 label=f"experiment (digitised, Fig. {dataset_name})")
    style.update(kwargs)
    ax.plot(xs, ys, **style)
    return True
