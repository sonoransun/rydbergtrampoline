"""Render a true-geometry ring of N atoms with staggered detuning.

Drives `docs/figures/ring_geometry.png`. The mermaid `graph LR` block
in README.md communicates the bond structure but draws a linear chain;
this PNG complements it by showing the actual circular geometry that
the package's `Geometry.RING` (PBC) corresponds to.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    N = 16
    radius = 1.0
    angles = np.linspace(0.5 * np.pi, 0.5 * np.pi - 2 * np.pi, N, endpoint=False)
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    # Bonds (NN) drawn as straight segments — circular ring topology is
    # already conveyed by the node positions.
    for i in range(N):
        j = (i + 1) % N
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                color="#bdc3c7", lw=1.2, zorder=1)

    # Sites colour-coded by parity. Even (j%2==0) carries +Δ_l, odd carries −Δ_l
    # (following the staggered detuning Δ_j = −Δ_g + (−1)^j Δ_l with j=0 even).
    even_color = "#3498db"
    odd_color = "#e67e22"
    for j, (x, y) in enumerate(zip(xs, ys)):
        c = even_color if j % 2 == 0 else odd_color
        ax.plot(x, y, "o", ms=24, color=c, markeredgecolor="#2c3e50",
                markeredgewidth=1.4, zorder=3)
        # Site index outside the circle.
        lx = 1.18 * x
        ly = 1.18 * y
        ax.text(lx, ly, str(j), fontsize=10, ha="center", va="center",
                color="#2c3e50")

    # Centre annotation: what the colours mean.
    ax.text(0, 0.10,
            r"$\Delta_j = -\Delta_g + (-1)^{j}\,\Delta_l$",
            ha="center", va="center", fontsize=12, color="#2c3e50")
    ax.text(0, -0.05,
            "ring of $N=16$ atoms,  PBC",
            ha="center", va="center", fontsize=10, color="#7f8c8d",
            style="italic")
    # Even / odd sub-legend in the centre.
    ax.plot(-0.30, -0.22, "o", ms=14, color=even_color,
            markeredgecolor="#2c3e50", markeredgewidth=1.0)
    ax.text(-0.20, -0.22, r"even site:  $-\Delta_g + \Delta_l$",
            fontsize=10, color="#2c3e50", va="center")
    ax.plot(-0.30, -0.36, "o", ms=14, color=odd_color,
            markeredgecolor="#2c3e50", markeredgewidth=1.0)
    ax.text(-0.20, -0.36, r"odd site:   $-\Delta_g - \Delta_l$",
            fontsize=10, color="#2c3e50", va="center")

    # Mark the NN-bond by an annotation arrow + label.
    bond_x = (xs[0] + xs[1]) / 2
    bond_y = (ys[0] + ys[1]) / 2
    ax.annotate(
        r"NN bond  $V_{\mathrm{NN}}$",
        xy=(bond_x, bond_y),
        xytext=(bond_x + 0.45, bond_y + 0.30),
        fontsize=10, color="#16a085",
        arrowprops=dict(arrowstyle="->", color="#16a085", lw=0.9),
    )
    # And one further-neighbour bond example.
    diag_x = (xs[0] + xs[3]) / 2
    diag_y = (ys[0] + ys[3]) / 2
    ax.plot([xs[0], xs[3]], [ys[0], ys[3]], color="#16a085",
            lw=0.7, ls=":", alpha=0.5)
    ax.annotate(
        r"longer-range  $V_{ij} \propto |i-j|^{-6}$",
        xy=(diag_x, diag_y),
        xytext=(diag_x + 0.50, diag_y - 0.02),
        fontsize=9.5, color="#16a085",
        arrowprops=dict(arrowstyle="->", color="#16a085", lw=0.9),
    )

    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Rydberg ring geometry with staggered detuning",
        fontsize=12,
    )

    fig.tight_layout()
    out_path = out_dir / "ring_geometry.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
