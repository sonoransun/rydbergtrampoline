"""Figure 3: Γ and bubble density vs Δ_l on a linear scale (paper Fig. 3 inset / Fig. 4).

Highlights the *resonance peaks* — values of Δ_l for which Γ exceeds the
smooth ``A exp(−B/Δ_l)`` law, indicating a discrete-spectrum nucleation
channel becomes accessible. We also overlay the time-averaged bubble
density Σ_1+Σ_2+Σ_3 to corroborate the resonance interpretation.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from rydberg_trampoline.analysis import (
    fit_decay_rate,
    fit_tunneling_action,
)
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
        "--delta-l",
        type=float,
        nargs="+",
        default=list(np.linspace(0.4, 4.0, 30)),
    )
    parser.add_argument("--t-max", type=float, default=3.0)
    parser.add_argument("--n-times", type=int, default=151)
    args = parser.parse_args(argv)

    times = np.linspace(0.0, args.t_max, args.n_times)
    deltas = np.asarray(args.delta_l, dtype=np.float64)
    gammas = np.empty_like(deltas)
    bubble_avg = np.empty_like(deltas)

    for k, dl in enumerate(deltas):
        params = ModelParams(N=args.N, Delta_l=float(dl))
        res = run_unitary(params, times, backend=args.backend, bubble_lengths=[1, 2, 3])
        fit = fit_decay_rate(times, res.m_afm, t_max=args.t_max * 0.6)
        gammas[k] = fit.Gamma if fit.success else np.nan
        # Time-average bubble densities over the full window.
        total = sum(res.bubble_densities[L] for L in (1, 2, 3))
        bubble_avg[k] = float(np.mean(total))

    valid = np.isfinite(gammas) & (gammas > 0)
    smooth_fit = fit_tunneling_action(deltas[valid], gammas[valid])

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)

    ax = axes[0]
    ax.plot(deltas[valid], gammas[valid], "o-", color="#2c3e50", lw=1.2, label=r"$\Gamma$")
    grid = np.linspace(deltas.min(), deltas.max(), 200)
    if smooth_fit.success:
        ax.plot(
            grid,
            smooth_fit.A * np.exp(-smooth_fit.B / grid),
            color="#c0392b",
            lw=1.5,
            ls="--",
            label=fr"$A e^{{-B/\Delta_l}}$ (B = {smooth_fit.B:.2f})",
        )
    ax.set_ylabel(r"decay rate $\Gamma$  (1/μs)")
    ax.legend(loc="best", fontsize=10)
    style_axes(ax)

    ax = axes[1]
    ax.plot(deltas, bubble_avg, "o-", color="#16a085", lw=1.2)
    ax.set_xlabel(r"$\Delta_l$  (MHz)")
    ax.set_ylabel(r"$\langle \Sigma_1 + \Sigma_2 + \Sigma_3 \rangle_t$")
    style_axes(ax)

    fig.suptitle(
        rf"Resonance scan: deviations from $\Gamma = A e^{{-B/\Delta_l}}$ "
        rf"signal discrete-spectrum bubble channels (N = {args.N})"
    )
    fig.tight_layout()

    save_figure_with_sidecar(
        fig,
        args.out,
        "fig_resonance_scan",
        {
            "N": args.N,
            "backend": args.backend,
            "delta_l": deltas.tolist(),
            "gammas": gammas.tolist(),
            "bubble_avg": bubble_avg.tolist(),
            "fit_A": float(smooth_fit.A),
            "fit_B": float(smooth_fit.B),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
