"""Figure 1: M_AFM(t) decay traces for several Δ_l (paper Fig. 2).

Reproduces the rescaled antiferromagnetic order parameter

    M^res(t) = (M_AFM(t) + M_AFM(0)) / (2 M_AFM(0))

decaying from 1 toward 1/2 (maximally mixed) for a sweep of staggered fields
Δ_l. We use the experimental decoherence times T₁ = 28 μs, T₂* = 3.8 μs and
QuTiP's ``mesolve`` (or ``mcsolve`` if the ring is large enough). Generated
PNG lives in ``docs/figures/``.
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from rydberg_trampoline.dynamics import run_lindblad
from rydberg_trampoline.figures._common import (
    common_argparser,
    overlay_experimental,
    save_figure_with_sidecar,
    style_axes,
)
from rydberg_trampoline.model import ModelParams
from rydberg_trampoline.observables import m_afm_rescaled


def main(argv: list[str] | None = None) -> int:
    parser = common_argparser(__doc__)
    parser.add_argument(
        "--delta-l",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 3.0],
        help="Δ_l values to sweep (MHz)",
    )
    parser.add_argument("--t-max", type=float, default=4.0, help="evolution time (μs)")
    parser.add_argument("--n-times", type=int, default=41)
    args = parser.parse_args(argv)

    # Restrict ring size so dense Lindblad fits in memory.
    N = min(args.N, 10)
    times = np.linspace(0.0, args.t_max, args.n_times)

    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for k, dl in enumerate(args.delta_l):
        params = ModelParams(N=N, Delta_l=dl, T1=28.0, T2_star=3.8)
        res = run_lindblad(params, times, backend="qutip", method="mesolve")
        m_res = m_afm_rescaled(res.m_afm)
        color = cmap(k / max(1, len(args.delta_l) - 1))
        ax.plot(times, m_res, color=color, label=f"Δ$_l$ = {dl:g} MHz")

    overlay_experimental(ax, "fig2_decay", x="time_us", y="m_afm_res")
    ax.set_xlabel("time (μs)")
    ax.set_ylabel(r"$M_{\mathrm{AFM}}^{\mathrm{res}}(t)$")
    ax.axhline(0.5, color="k", lw=0.7, ls=":", alpha=0.6, label="maximally mixed")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.0, args.t_max)
    ax.legend(loc="upper right", ncols=2, fontsize=8, framealpha=0.85)
    ax.set_title(
        f"Rescaled M$_{{AFM}}$ decay from the false-vacuum Néel (N = {N}, T$_1$ = 28 μs, T$_2^*$ = 3.8 μs)"
    )
    style_axes(ax)
    fig.tight_layout()

    save_figure_with_sidecar(
        fig,
        args.out,
        "fig_decay_traces",
        {
            "N": N,
            "delta_l": list(args.delta_l),
            "t_max": args.t_max,
            "n_times": args.n_times,
            "backend": "qutip mesolve",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
