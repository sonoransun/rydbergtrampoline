"""Figure 4: bubble-length histogram on/off resonance (paper Fig. 4).

Compares the time-averaged bubble densities ⟨Σ_L⟩ for L = 1, 2, 3 between
two values of Δ_l: one in a regime where the QFT-style law dominates
(small Σ_2/Σ_3) and one where a discrete-spectrum resonance amplifies
larger bubbles.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from rydberg_trampoline.dynamics import run_unitary
from rydberg_trampoline.figures._common import (
    common_argparser,
    save_figure_with_sidecar,
    style_axes,
)
from rydberg_trampoline.model import ModelParams


def main(argv: list[str] | None = None) -> int:
    parser = common_argparser(__doc__)
    parser.add_argument(
        "--off-resonance",
        type=float,
        default=0.6,
        help="Δ_l in MHz at which the smooth tunneling law dominates",
    )
    parser.add_argument(
        "--on-resonance",
        type=float,
        default=2.5,
        help="Δ_l in MHz at which the discrete-spectrum channel amplifies bubbles",
    )
    parser.add_argument("--t-max", type=float, default=3.0)
    parser.add_argument("--n-times", type=int, default=121)
    args = parser.parse_args(argv)

    times = np.linspace(0.0, args.t_max, args.n_times)
    bubble_means: dict[str, dict[int, float]] = {}
    for label, dl in (("off-resonance", args.off_resonance), ("on-resonance", args.on_resonance)):
        params = ModelParams(N=args.N, Delta_l=float(dl))
        res = run_unitary(params, times, backend=args.backend, bubble_lengths=[1, 2, 3])
        bubble_means[label] = {L: float(np.mean(res.bubble_densities[L])) for L in (1, 2, 3)}

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    Ls = np.array([1, 2, 3])
    width = 0.38
    off = np.array([bubble_means["off-resonance"][L] for L in Ls])
    on = np.array([bubble_means["on-resonance"][L] for L in Ls])
    ax.bar(Ls - width / 2, off, width=width, color="#34495e",
           label=fr"off-resonance  $\Delta_l = {args.off_resonance:g}$ MHz")
    ax.bar(Ls + width / 2, on, width=width, color="#e67e22",
           label=fr"on-resonance   $\Delta_l = {args.on_resonance:g}$ MHz")
    # Numeric values atop each bar so the L=2,3 bars don't get lost by eye.
    for x, h in zip(Ls - width / 2, off):
        ax.text(x, h + 0.01, f"{h:.2f}", ha="center", fontsize=8.5, color="#34495e")
    for x, h in zip(Ls + width / 2, on):
        ax.text(x, h + 0.01, f"{h:.2f}", ha="center", fontsize=8.5, color="#e67e22")
    ax.set_xticks(Ls)
    ax.set_xticklabels([f"L = {L}" for L in Ls])
    ax.set_xlabel("bubble length")
    ax.set_ylabel(r"time-averaged density  $\langle \Sigma_L \rangle_t$")
    ax.set_title(
        f"Bubble-length distribution: off- vs on-resonance  (N = {args.N})",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85)
    ax.set_ylim(0, max(off.max(), on.max()) * 1.18)
    style_axes(ax)
    fig.tight_layout()

    save_figure_with_sidecar(
        fig,
        args.out,
        "fig_bubble_histogram",
        {
            "N": args.N,
            "backend": args.backend,
            "off_resonance_delta_l": args.off_resonance,
            "on_resonance_delta_l": args.on_resonance,
            "off_means": bubble_means["off-resonance"],
            "on_means": bubble_means["on-resonance"],
            "t_max": args.t_max,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
