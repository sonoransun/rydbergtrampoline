"""Render a static plot of V_ij(d)/V_NN with vdW-cutoff overlays.

Drives `docs/figures/vij_curve.png`. One-shot script — re-run after
changing the figure layout. Reuses `ModelParams.vdw_coupling` so this
script breaks loudly if the V convention drifts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rydberg_trampoline.model import ModelParams


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    distances = np.arange(1, 17)
    cutoffs = (1, 2, 4, 8)
    colors = ("#c0392b", "#e67e22", "#16a085", "#2c3e50")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    # The full 1/d^6 ratio (no cutoff) plotted as the underlying truth.
    full_params = ModelParams(N=32, vdW_cutoff=32)
    full_ratio = np.array([
        full_params.vdw_coupling(0, int(d)) / full_params.V_NN if d > 0 else np.nan
        for d in distances
    ])
    ax.plot(distances, full_ratio, color="black", lw=1.6, ls="-",
            label=r"true $1 / d^6$", zorder=2)

    # Each cutoff drops to zero at d > R. To visualise the dropped energy on
    # a log scale we plot a tiny floor (1e-8) so the step is still visible.
    floor = 1.0e-8
    for k, R in enumerate(cutoffs):
        ratios = np.array([
            (1.0 / d**6) if 0 < d <= R else floor
            for d in distances
        ])
        ax.plot(distances, ratios, "o-", color=colors[k], ms=4, lw=1.0,
                alpha=0.85,
                label=fr"vdW_cutoff $R={R}$")

    ax.set_xlabel("lattice distance $d$ (in units of nearest-neighbour spacing)")
    ax.set_ylabel(r"$V_{ij} / V_{\mathrm{NN}}$")
    ax.set_yscale("log")
    ax.set_xlim(0.5, distances[-1] + 0.5)
    ax.set_ylim(1.0e-8, 2.0)
    ax.grid(alpha=0.3, linestyle=":", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85, ncols=1)

    # Annotate the dominant NN-vs-NNN ratio of 64×.
    ax.annotate(
        r"NN dominates NNN by $2^6 = 64\times$",
        xy=(2, 1.0 / 64),
        xytext=(4.5, 0.5),
        fontsize=9.5,
        color="#34495e",
        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=0.8),
    )

    ax.set_title(
        r"Van-der-Waals tail $V_{ij} \propto 1/d^6$ and the vdW_cutoff truncation"
    )

    fig.tight_layout()
    out_path = out_dir / "vij_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
