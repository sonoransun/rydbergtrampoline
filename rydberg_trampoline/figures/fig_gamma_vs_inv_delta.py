"""Figure 2: log Γ vs 1/Δ_l with the QFT-style Γ ∝ exp(−B/Δ_l) law (paper Fig. 3).

This is the *headline* scientific plot of the paper. We sweep Δ_l, run a
unitary evolution from the false-vacuum Néel state, fit a single-exponential
to the rescaled magnetisation trace to extract Γ(Δ_l), then linear-fit
``log Γ`` vs ``1/Δ_l`` to obtain the tunneling action B.

Closed-system unitary dynamics is used here so the suppression law is
unambiguous; the experimental decoherence-limited counterpart is in
:mod:`fig_decay_traces` and the open-system overlay version is left for the
data-overlay variant.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from rydberg_trampoline.analysis import fit_decay_rate, fit_tunneling_action
from rydberg_trampoline.dynamics import run_unitary
from rydberg_trampoline.figures._common import (
    common_argparser,
    overlay_experimental,
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
        default=list(np.linspace(0.4, 3.0, 14)),
        help="Δ_l values to sweep (MHz)",
    )
    parser.add_argument("--t-max", type=float, default=3.0, help="evolution time (μs)")
    parser.add_argument("--n-times", type=int, default=121)
    args = parser.parse_args(argv)

    times = np.linspace(0.0, args.t_max, args.n_times)
    deltas = np.asarray(args.delta_l, dtype=np.float64)
    gammas = np.empty_like(deltas)

    for k, dl in enumerate(deltas):
        params = ModelParams(N=args.N, Delta_l=float(dl))
        res = run_unitary(params, times, backend=args.backend)
        # Fit on the early window where the exp law applies (before recurrences).
        fit = fit_decay_rate(times, res.m_afm, t_max=args.t_max * 0.6)
        gammas[k] = fit.Gamma if fit.success else np.nan

    valid = np.isfinite(gammas) & (gammas > 0)
    fit = fit_tunneling_action(deltas[valid], gammas[valid])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(1.0 / deltas[valid], gammas[valid], s=40, color="#2c3e50", zorder=3)
    grid = np.linspace(deltas.min() * 0.95, deltas.max() * 1.05, 200)
    if fit.success:
        ax.plot(
            1.0 / grid,
            fit.A * np.exp(-fit.B / grid),
            color="#c0392b",
            lw=2,
            label=fr"fit: $\Gamma = {fit.A:.3f} \, e^{{-{fit.B:.2f} / \Delta_l}}$",
        )
    overlay_experimental(ax, "fig3_gamma", x="inv_delta_l", y="gamma_per_us")
    ax.set_yscale("log")
    ax.set_xlabel(r"$1 / \Delta_l$  (1/MHz)")
    ax.set_ylabel(r"decay rate $\Gamma$  (1/μs)")
    ax.set_title(
        rf"False-vacuum decay rate vs $1/\Delta_l$ "
        rf"(N = {args.N}, closed system, {args.backend})"
    )
    ax.legend(loc="lower left", fontsize=10)
    style_axes(ax)
    fig.tight_layout()

    save_figure_with_sidecar(
        fig,
        args.out,
        "fig_gamma_vs_inv_delta",
        {
            "N": args.N,
            "backend": args.backend,
            "delta_l": deltas.tolist(),
            "gammas": gammas.tolist(),
            "fit_A": float(fit.A),
            "fit_B": float(fit.B),
            "t_max": args.t_max,
            "n_times": args.n_times,
        },
    )
    print(f"fitted Γ(Δ_l) = {fit.A:.3g} · exp(-{fit.B:.3g} / Δ_l)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
