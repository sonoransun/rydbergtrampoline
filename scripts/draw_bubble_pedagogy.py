"""Render a pedagogical chart of length-L bubbles for L = 1, 2, 3, 4.

Drives `docs/figures/bubble_pedagogy.png`. The hero `bubble_cartoon.png`
shows L=1 and L=3 only and leaves the reader to extrapolate; this figure
shows the family side-by-side so a novice can read off "what is a
length-L bubble?" directly from the diagram.

A length-L bubble is a contiguous run of L sites whose occupations are
flipped relative to the false-vacuum Néel, with the sites on either side
both back in the false-vacuum pattern. Equivalently the operator

    Σ_L = Σ_j  n̄_{j-1}  · X_{j} · X_{j+1} · … · X_{j+L-1}  · n̄_{j+L}

with X_k = n_k if k is in an FV-occupied site (so a flip ⇒ X_k = 1) and
n̄_k = 1 - n_k for the ordinary "FV here" complement. Each panel labels
the bubble interior, the two domain walls (kinks where neighbours
agree), and the surrounding FV.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from rydberg_trampoline.conventions import neel_occupation


# Match the existing draw_*.py palette (also in figures/_common.PALETTE).
RYDBERG = "#c0392b"
INK = "#34495e"
DOMAIN_WALL = "#16a085"
INDEX = "#7f8c8d"
BUBBLE_BAND = "#fdf2e9"
BUBBLE_EDGE = "#e67e22"


def _flip_run(occ: list[int], start: int, length: int) -> list[int]:
    """Flip ``length`` consecutive sites starting at index ``start``."""
    out = list(occ)
    for k in range(length):
        j = start + k
        out[j] = 1 - out[j]
    return out


def _draw_panel(ax, occ: list[int], *, bubble_span: tuple[int, int], L: int) -> None:
    N = len(occ)
    # Site circles.
    for j, n in enumerate(occ):
        if n == 1:
            ax.add_patch(plt.Circle((j, 0), 0.34, color=RYDBERG, zorder=3))
        else:
            ax.add_patch(plt.Circle(
                (j, 0), 0.34, fill=False, edgecolor=INK, lw=1.6, zorder=3
            ))
    # Bubble-interior band.
    a, b = bubble_span
    ax.add_patch(mpatches.FancyBboxPatch(
        (a - 0.55, -0.55), b - a + 1.1, 1.1,
        boxstyle="round,pad=0.02,rounding_size=0.3",
        edgecolor=BUBBLE_EDGE, facecolor=BUBBLE_BAND,
        lw=1.5, zorder=1,
    ))
    # Domain-wall hashes on the bonds bracketing the bubble interior.
    for x_dw in (a - 0.5, b + 0.5):
        ax.plot([x_dw, x_dw], [-0.5, 0.5], color=DOMAIN_WALL, lw=2.4, zorder=4)
        for ydir in (-1.0, 1.0):
            ax.plot([x_dw - 0.10, x_dw + 0.10],
                    [ydir * 0.5, ydir * 0.5],
                    color=DOMAIN_WALL, lw=1.7, zorder=4)
    # Site indices.
    for j in range(N):
        ax.text(j, -0.85, str(j), fontsize=7, ha="center", va="center",
                color=INDEX)
    # Panel title.
    ax.set_title(f"L = {L}", fontsize=11, color=INK, pad=4)
    ax.set_xlim(-1.2, N - 0.2)
    ax.set_ylim(-1.4, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    N = 12
    fv = list(neel_occupation(N, phase=0))   # (1, 0, 1, 0, ...)
    # Place each bubble centred-ish in the ring, leaving room on both sides.
    panels = [
        (1, 5),
        (2, 5),
        (3, 4),
        (4, 4),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.0))
    for ax, (L, start) in zip(axes, panels):
        occ = _flip_run(fv, start, L)
        _draw_panel(ax, occ, bubble_span=(start, start + L - 1), L=L)

    # Shared legend at the top.
    rg_handle = mpatches.Patch(facecolor=RYDBERG, edgecolor=RYDBERG,
                               label="Rydberg (n = 1)")
    gd_handle = mpatches.Patch(facecolor="white", edgecolor=INK,
                               label="ground (n = 0)")
    dw_handle = plt.Line2D([0], [0], color=DOMAIN_WALL, lw=2.4,
                           label="domain wall (kink)")
    bb_handle = mpatches.Patch(facecolor=BUBBLE_BAND, edgecolor=BUBBLE_EDGE,
                               label="bubble interior")
    fig.legend(
        handles=[rg_handle, gd_handle, dw_handle, bb_handle],
        loc="lower center",
        ncols=4,
        fontsize=9,
        framealpha=0.95,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(
        r"Length-$L$ bubbles inside the false-vacuum Néel  "
        r"($N=12$):  one contiguous flipped run, two domain walls",
        fontsize=11.5, y=1.02,
    )

    fig.tight_layout()
    out_path = out_dir / "bubble_pedagogy.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
