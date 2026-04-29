"""Render a static three-level Rydberg scheme PNG for the README.

Not part of the package — just a one-shot drawing script. Re-run it if
you change the README layout.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    levels = [
        (0.05, 0.15, r"$|g\rangle$  (5S$_{1/2}$)"),
        (0.05, 0.55, r"$|e\rangle$  (intermediate)"),
        (0.05, 0.92, r"$|r\rangle$  Rydberg  (≈ 70S$_{1/2}$)"),
    ]
    line_w = 0.25
    for x, y, label in levels:
        ax.hlines(y, x, x + line_w, color="black", lw=2.0)
        ax.text(x + line_w + 0.02, y, label, va="center", fontsize=11)

    # Two-photon transition from g → e → r drawn as curved arrow.
    ax.annotate(
        "",
        xy=(0.17, 0.92),
        xytext=(0.17, 0.15),
        arrowprops=dict(
            arrowstyle="->",
            color="#c0392b",
            lw=1.6,
            connectionstyle="arc3,rad=0.05",
        ),
    )
    ax.text(0.20, 0.55, r"effective Rabi $\Omega \sim 1.8$ MHz", fontsize=10, color="#c0392b")

    # Detunings annotated to the side.
    ax.annotate(
        r"global $\Delta_g$",
        xy=(0.31, 0.94),
        xytext=(0.55, 0.83),
        fontsize=10,
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.7),
    )
    ax.annotate(
        r"staggered $(-1)^j \Delta_l$",
        xy=(0.31, 0.92),
        xytext=(0.55, 0.72),
        fontsize=10,
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.7),
    )

    # vdW interaction note.
    ax.text(
        0.5,
        0.05,
        r"two-Rydberg interaction $V_{ij} \propto |i-j|^{-6}$",
        fontsize=10,
        ha="center",
        color="#16a085",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.axis("off")
    ax.set_title(r"Effective two-level Rydberg atom: drive $\Omega$, detunings, vdW")

    fig.tight_layout()
    fig.savefig(out_dir / "level_scheme.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'level_scheme.png'}")


if __name__ == "__main__":
    main()
