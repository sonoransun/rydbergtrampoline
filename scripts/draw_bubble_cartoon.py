"""Render a bubble-nucleation cartoon for the docs.

Three rows of N=12 sites each:
  Row 1 — clean Néel false vacuum (alternating)
  Row 2 — single-flip bubble (length-1 inside FV)
  Row 3 — length-3 bubble bordered by FV on both sides

Sites are circles coloured by occupation; the bubble interior is
highlighted with a soft band underneath.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from rydberg_trampoline.conventions import neel_occupation


def _draw_row(ax, y: float, occ, *, bubble_span=None, label: str = ""):
    N = len(occ)
    for j, n in enumerate(occ):
        # Filled = Rydberg (n=1), open = ground (n=0).
        if n == 1:
            ax.add_patch(plt.Circle((j, y), 0.34, color="#c0392b", zorder=3))
        else:
            ax.add_patch(plt.Circle((j, y), 0.34, fill=False,
                                    edgecolor="#34495e", lw=1.6, zorder=3))
    # Highlight the bubble span as a translucent band.
    if bubble_span is not None:
        a, b = bubble_span
        ax.add_patch(mpatches.FancyBboxPatch(
            (a - 0.55, y - 0.55), b - a + 1.1, 1.1,
            boxstyle="round,pad=0.02,rounding_size=0.3",
            edgecolor="#e67e22", facecolor="#fdf2e9",
            lw=1.5, zorder=1,
        ))
    # Site indices.
    for j in range(N):
        ax.text(j, y - 0.78, str(j), fontsize=8, ha="center", va="center",
                color="#7f8c8d")
    # Row label on the left.
    ax.text(-1.2, y, label, fontsize=10, ha="right", va="center")


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    N = 12
    fv = list(neel_occupation(N, phase=0))     # (1, 0, 1, 0, ...)

    # Row 1: clean false vacuum.
    row1 = list(fv)
    # Row 2: single-flip bubble at site 5 (TV at site 5 means n_5=1 since FV's odd
    # site has n=0). Flipping gives n_5 = 1.
    row2 = list(fv)
    row2[5] = 1
    # Row 3: length-3 bubble at sites 4, 5, 6. TV occupations are
    # opposite of FV, so n_4 = 0, n_5 = 1, n_6 = 0 — wait that's just
    # FV with phase swapped. Use flips: site 4 (FV n=1) → 0, site 5 (FV n=0) → 1,
    # site 6 (FV n=1) → 0.
    row3 = list(fv)
    row3[4] = 0  # was 1
    row3[5] = 1  # was 0
    row3[6] = 0  # was 1

    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    _draw_row(ax, y=4.0, occ=row1, label="false\nvacuum")
    _draw_row(ax, y=2.0, occ=row2, bubble_span=(5, 5),
              label="single\nflip")
    _draw_row(ax, y=0.0, occ=row3, bubble_span=(4, 6),
              label="length-3\nbubble")

    # Domain-wall arrows on row 3.
    ax.annotate("domain wall", xy=(3.5, 0.0), xytext=(2.5, -1.4),
                fontsize=9, color="#16a085", ha="center",
                arrowprops=dict(arrowstyle="->", color="#16a085", lw=0.8))
    ax.annotate("domain wall", xy=(6.5, 0.0), xytext=(7.5, -1.4),
                fontsize=9, color="#16a085", ha="center",
                arrowprops=dict(arrowstyle="->", color="#16a085", lw=0.8))

    # Legend.
    rg = plt.Circle((0, 0), 0.34, color="#c0392b")
    gd = plt.Circle((0, 0), 0.34, fill=False, edgecolor="#34495e", lw=1.6)
    ax.legend(
        [rg, gd],
        ["Rydberg (n = 1, σᶻ = +1)", "ground (n = 0, σᶻ = −1)"],
        loc="upper right", fontsize=9, framealpha=0.85,
        handler_map={
            plt.Circle: type("HC", (), {
                "legend_artist": staticmethod(
                    lambda legend, orig_handle, fontsize, handlebox: (
                        handlebox.add_artist(plt.Circle(
                            (handlebox.xdescent + 8, handlebox.ydescent + 5),
                            5,
                            color=orig_handle.get_facecolor(),
                            fill=orig_handle.get_fill(),
                            edgecolor=orig_handle.get_edgecolor(),
                            lw=orig_handle.get_linewidth(),
                        )),
                        plt.Circle((0, 0), 0),
                    )[1]
                ),
            })()
        },
    )

    ax.set_xlim(-2.5, N + 0.5)
    ax.set_ylim(-2.0, 5.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Néel false vacuum, a single-flip bubble, "
        "and a length-3 bubble bordered by domain walls",
        fontsize=11,
    )

    fig.tight_layout()
    out_path = out_dir / "bubble_cartoon.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
