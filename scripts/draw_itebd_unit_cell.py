"""Render the iTEBD 2-site unit cell used by the TeNPy backend.

Drives `docs/figures/itebd_unit_cell.png`. The TeNPy backend
(``backends/tenpy_backend.py``) builds an iMPS with a 2-site unit cell
``(A, B)``: site A carries the ``+Δ_l`` staggered offset and site B
carries ``−Δ_l``. Translation-by-2 is built in (the unit cell tiles
infinitely), and the NN-only vdW bond between A and B (and across the
periodic copy boundary) is the only two-body coupling.

This figure is purely schematic — it is not generated from package
state — and runs in <1 s.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# Matches the project palette across other draw_*.py scripts.
EVEN_BLUE = "#3498db"
ODD_ORANGE = "#e67e22"
INK = "#34495e"
GREEN = "#16a085"
MUTED = "#7f8c8d"
RED = "#c0392b"


def _site(ax, x: float, y: float, *, color: str, label: str, sub: str,
          sub_y: float = -0.65):
    ax.add_patch(plt.Circle((x, y), 0.30, color=color,
                            ec=INK, lw=1.4, zorder=3))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=12, fontweight="bold", color="white", zorder=4)
    ax.text(x, y + sub_y, sub, ha="center", va="center",
            fontsize=9, color=INK)


def _bond(ax, x1: float, x2: float, y: float, *, label: str, ls: str = "-",
          color: str = GREEN, lw: float = 2.0, label_dy: float = 0.3,
          zorder: int = 2):
    ax.plot([x1, x2], [y, y], color=color, lw=lw, ls=ls, zorder=zorder)
    ax.text((x1 + x2) / 2, y + label_dy, label, ha="center", va="center",
            fontsize=9, color=color)


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 4.0))

    # Unit cell at the centre, two ghost copies left and right with reduced
    # opacity so the reader sees the tiling pattern without drowning in detail.
    centre_x = 0.0
    pair_pitch = 1.4
    site_pitch = 0.7
    y0 = 0.0

    # Centre unit cell (A, B) — solid. Stagger the subtitles vertically so
    # they don't run into each other on the narrow site_pitch.
    A_x = centre_x - site_pitch / 2
    B_x = centre_x + site_pitch / 2
    _site(ax, A_x, y0, color=EVEN_BLUE,
          label="A", sub=r"even site, $+\Delta_\ell$", sub_y=-0.62)
    _site(ax, B_x, y0, color=ODD_ORANGE,
          label="B", sub=r"odd site, $-\Delta_\ell$", sub_y=-0.92)
    _bond(ax, A_x, B_x, y0,
          label=r"$V_{NN}$", color=GREEN, lw=2.4)

    # Left ghost cell (A', B') — translucent.
    AL_x = centre_x - pair_pitch - site_pitch / 2
    BL_x = centre_x - pair_pitch + site_pitch / 2
    for x, c in [(AL_x, EVEN_BLUE), (BL_x, ODD_ORANGE)]:
        ax.add_patch(plt.Circle((x, y0), 0.30, color=c,
                                ec=INK, lw=1.4, alpha=0.35, zorder=3))
    _bond(ax, AL_x, BL_x, y0, label="", color=GREEN, lw=1.6, label_dy=0)
    _bond(ax, BL_x, A_x, y0, label="", color=GREEN, lw=1.6, label_dy=0)

    # Right ghost cell.
    AR_x = centre_x + pair_pitch - site_pitch / 2
    BR_x = centre_x + pair_pitch + site_pitch / 2
    for x, c in [(AR_x, EVEN_BLUE), (BR_x, ODD_ORANGE)]:
        ax.add_patch(plt.Circle((x, y0), 0.30, color=c,
                                ec=INK, lw=1.4, alpha=0.35, zorder=3))
    _bond(ax, AR_x, BR_x, y0, label="", color=GREEN, lw=1.6, label_dy=0)
    _bond(ax, B_x, AR_x, y0, label="", color=GREEN, lw=1.6, label_dy=0)

    # MPS bond legs (vertical lines below sites for the chain bond dimensions).
    leg_top = y0 - 0.30
    leg_bot = y0 - 1.55
    for x in (AL_x, BL_x, A_x, B_x, AR_x, BR_x):
        ax.plot([x, x], [leg_top, leg_bot], color=MUTED, lw=1.0,
                ls="-", zorder=1)
        ax.text(x, leg_bot - 0.10, r"$\chi$", ha="center", va="top",
                fontsize=8, color=MUTED)

    # Translation-by-2 indicator: a curved arrow from the centre cell to the
    # right ghost, showing the unit-cell repeat.
    ax.annotate(
        "", xy=(AR_x + 0.05, y0 + 1.0), xytext=(A_x - 0.05, y0 + 1.0),
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.4,
                        connectionstyle="arc3,rad=-0.2"),
    )
    ax.text((A_x + AR_x) / 2, y0 + 1.35,
            "translation by 2 sites  $\\Rightarrow$  unit cell tiles infinitely",
            ha="center", va="center", fontsize=10, color=INK)

    # NN-only annotation, anchored well below the χ legs so it doesn't crowd.
    ax.annotate(
        r"only nearest-neighbour $V_{ij}$ kept (TEBD locality)",
        xy=(centre_x, y0 - 0.05),
        xytext=(centre_x, y0 - 2.40),
        ha="center",
        fontsize=9.5,
        color=RED,
        arrowprops=dict(arrowstyle="-", color=RED, lw=0.8),
    )

    # Frame the central unit cell with a soft box that surrounds only the
    # circles (not the subtitles), keeping the diagram airy.
    ax.add_patch(mpatches.FancyBboxPatch(
        (A_x - 0.55, y0 - 0.45), site_pitch + 1.10, 0.9,
        boxstyle="round,pad=0.02,rounding_size=0.20",
        edgecolor=INK, facecolor="none", lw=1.2, ls="--", zorder=0,
    ))
    ax.text(centre_x, y0 + 0.62, "2-site unit cell", ha="center",
            fontsize=10, color=INK, fontweight="bold")

    # Far-left "..." and far-right "..." to suggest the chain extends.
    ax.text(AL_x - 0.65, y0, "$\\cdots$", fontsize=14, color=MUTED,
            ha="right", va="center")
    ax.text(BR_x + 0.65, y0, "$\\cdots$", fontsize=14, color=MUTED,
            ha="left", va="center")

    ax.set_xlim(AL_x - 1.20, BR_x + 1.20)
    ax.set_ylim(y0 - 2.85, y0 + 1.85)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "iTEBD on the 2-site iMPS unit cell  "
        "($A=$ even, $B=$ odd; staggered $\\pm \\Delta_\\ell$ baked in)",
        fontsize=11.5,
    )

    fig.tight_layout()
    out_path = out_dir / "itebd_unit_cell.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
