"""Render a static level-scheme PNG for the README.

Not part of the package — just a one-shot drawing script. Re-run it if
you change the README layout.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # Two side-by-side panels in axes coords:
    #   left half (x < 0.55) — single-atom level ladder
    #   right half (x > 0.55) — two-atom vdW illustration
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.axis("off")

    # ----- left panel: single-atom three-level ladder -----
    levels = [
        (0.18, r"$|g\rangle$  5S$_{1/2}$",       "#2c3e50"),
        (0.50, r"$|e\rangle$  intermediate",     "#7f8c8d"),
        (0.86, r"$|r\rangle$  Rydberg, 70S$_{1/2}$", "#2c3e50"),
    ]
    x_left, x_right = 0.06, 0.28
    for y, label, color in levels:
        ax.hlines(y, x_left, x_right, color=color, lw=2.4)
        ax.text(x_right + 0.015, y, label, va="center", fontsize=10.5, color=color)

    # Two-photon drive: tall arrow centred on the level-line midpoint.
    arrow_x = (x_left + x_right) / 2
    ax.annotate(
        "",
        xy=(arrow_x, levels[2][0] - 0.015),
        xytext=(arrow_x, levels[0][0] + 0.015),
        arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.8, mutation_scale=18),
    )

    # Rabi label: to the LEFT of the arrow, vertically between |g> and |e>,
    # so it cannot collide with any level line or label.
    ax.text(
        arrow_x - 0.025, 0.33,
        r"$\Omega \sim 1.8$ MHz" "\n" r"(two-photon)",
        ha="right", va="center",
        fontsize=10, color="#c0392b",
    )

    # Detunings: leaders point at |r>, text on the right side of the panel
    # but well above the |r> line so leaders don't tangle.
    ax.annotate(
        r"site-staggered $(-1)^{j}\Delta_l$",
        xy=(x_right, levels[2][0] + 0.018),
        xytext=(x_right + 0.02, 0.99),
        fontsize=9.5, color="#34495e",
        arrowprops=dict(arrowstyle="-", color="#7f8c8d", lw=0.7),
    )
    ax.annotate(
        r"global $-\Delta_g$",
        xy=(x_right, levels[2][0] - 0.012),
        xytext=(x_right + 0.10, 0.71),
        fontsize=9.5, color="#34495e",
        arrowprops=dict(arrowstyle="-", color="#7f8c8d", lw=0.7),
    )

    ax.text(
        (x_left + x_right) / 2, 0.03,
        "single-atom drive",
        fontsize=9, ha="center", color="#7f8c8d", style="italic",
    )

    # ----- divider between panels -----
    ax.vlines(0.66, 0.05, 0.95, color="#bdc3c7", lw=0.8, ls=":")

    # ----- right panel: two-atom vdW illustration -----
    panel_x = 0.83
    atom_y = 0.50
    sep = 0.18
    for sign in (-1, 1):
        ax.plot(panel_x + sign * sep / 2, atom_y, "o", ms=22,
                color="#c0392b", zorder=3)
        ax.text(panel_x + sign * sep / 2, atom_y - 0.13,
                r"$|r\rangle$", ha="center", fontsize=9, color="#2c3e50")
    # Distance bar
    ax.annotate(
        "",
        xy=(panel_x + sep / 2 - 0.02, atom_y),
        xytext=(panel_x - sep / 2 + 0.02, atom_y),
        arrowprops=dict(arrowstyle="<->", color="#16a085", lw=1.4),
    )
    ax.text(panel_x, atom_y + 0.10,
            r"$V_{ij}\propto |i-j|^{-6}$",
            ha="center", fontsize=11, color="#16a085")
    ax.text(panel_x, atom_y - 0.22,
            "two-Rydberg vdW", ha="center", fontsize=9.5, color="#16a085",
            style="italic")
    # Site labels
    ax.text(panel_x - sep / 2, atom_y + 0.18, "site $i$", ha="center",
            fontsize=8.5, color="#7f8c8d")
    ax.text(panel_x + sep / 2, atom_y + 0.18, "site $j$", ha="center",
            fontsize=8.5, color="#7f8c8d")

    fig.suptitle(
        r"Effective two-level Rydberg atom + pair vdW: $\Omega$, $\Delta_g$, $\Delta_l$, $V_{ij}$",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "level_scheme.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out_dir / 'level_scheme.png'}")


if __name__ == "__main__":
    main()
