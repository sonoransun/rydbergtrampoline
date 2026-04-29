"""Figure 6: finite-N ED vs iTEBD (N → ∞) at the same Δ_l.

Shows ``M_AFM(t)`` from three computations sharing identical NN-only vdW
parameters:

  - N=8 ED (closed-system unitary)
  - N=12 ED (closed-system unitary)
  - iTEBD (thermodynamic limit) on the 2-site iMPS unit cell

The three curves coincide at short times, before the light cone has wrapped
around the finite ring; deviations at later times encode the finite-size
effects that the paper explicitly avoids by reporting iTEBD curves alongside
finite-N data. Useful both as a backend showcase and as a sanity check that
the iTEBD ↔ ED short-time agreement holds at the figure level (not only
in the regression test).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from rydberg_trampoline.dynamics import run_itebd, run_unitary
from rydberg_trampoline.figures._common import (
    common_argparser,
    save_figure_with_sidecar,
    style_axes,
)
from rydberg_trampoline.model import ModelParams


def main(argv: list[str] | None = None) -> int:
    parser = common_argparser(__doc__)
    parser.add_argument("--delta-l", type=float, default=2.0)
    parser.add_argument("--t-max", type=float, default=2.5)
    parser.add_argument("--n-times", type=int, default=51)
    parser.add_argument("--chi", type=int, default=80)
    args = parser.parse_args(argv)

    times = np.linspace(0.0, args.t_max, args.n_times)
    # vdW_cutoff = 1 because TEBD is NN-only; we want apples-to-apples.
    base = ModelParams(N=8, Delta_l=args.delta_l, vdW_cutoff=1)

    res_n8 = run_unitary(base.with_(N=8), times, backend=args.backend)
    res_n12 = run_unitary(base.with_(N=12), times, backend=args.backend)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(times, res_n8.m_afm, color="#34495e", lw=1.6, label="ED, N = 8")
    ax.plot(times, res_n12.m_afm, color="#16a085", lw=1.6, label="ED, N = 12")
    try:
        res_itebd = run_itebd(base, times, chi=args.chi)
        ax.plot(times, res_itebd.m_afm, color="#c0392b", lw=2.0, ls="--",
                label=f"iTEBD, N → ∞ (χ = {args.chi})")
        itebd_note = f"iTEBD chi={args.chi}"
    except ModuleNotFoundError:
        itebd_note = "iTEBD skipped (TeNPy not installed)"
        ax.text(0.5, 0.05, itebd_note, transform=ax.transAxes,
                ha="center", color="#c0392b")

    ax.set_xlabel("time (μs)")
    ax.set_ylabel(r"$\langle M_{\mathrm{AFM}}\rangle(t)$")
    ax.set_title(
        rf"Finite-N ED vs iTEBD on the staggered Rydberg ring "
        rf"($\Delta_l$ = {args.delta_l:g} MHz, NN-only)"
    )
    ax.set_ylim(-0.15, 1.05)
    ax.legend(loc="best", fontsize=10)
    style_axes(ax)
    fig.tight_layout()

    save_figure_with_sidecar(
        fig,
        args.out,
        "fig_thermodynamic_limit",
        {
            "delta_l": args.delta_l,
            "t_max": args.t_max,
            "n_times": args.n_times,
            "chi": args.chi,
            "backend_finite_N": args.backend,
            "itebd_status": itebd_note,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
