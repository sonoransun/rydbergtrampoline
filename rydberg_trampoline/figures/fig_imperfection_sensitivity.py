"""Figure 5: clean vs perturbed Néel initial state (paper Fig. 5 / SM).

Shows the paper's striking observation that small imperfections in the
metastable preparation collapse the exponential decay law: the rescaled
M_AFM trace from a slightly perturbed initial state deviates substantially
from the clean trace at intermediate times.
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
from rydberg_trampoline.observables import m_afm_rescaled
from rydberg_trampoline.states import (
    neel_state,
    perturbed_neel_state,
    single_flip_admixed_neel,
)


def main(argv: list[str] | None = None) -> int:
    parser = common_argparser(__doc__)
    parser.add_argument("--delta-l", type=float, default=2.0)
    parser.add_argument("--t-max", type=float, default=3.0)
    parser.add_argument("--n-times", type=int, default=121)
    parser.add_argument(
        "--fidelities",
        type=float,
        nargs="+",
        default=[1.0, 0.99, 0.95, 0.85],
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--noise-model",
        choices=("single-flip", "haar"),
        default="single-flip",
        help=(
            "single-flip: coherent admixture of single-flip states (paper-realistic); "
            "haar: Haar-random orthogonal admixture (averages over all errors)."
        ),
    )
    args = parser.parse_args(argv)

    times = np.linspace(0.0, args.t_max, args.n_times)
    params = ModelParams(N=args.N, Delta_l=args.delta_l)
    rng = np.random.default_rng(args.seed)

    cmap = plt.get_cmap("plasma")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for k, fidelity in enumerate(sorted(args.fidelities, reverse=True)):
        if fidelity == 1.0:
            psi0 = neel_state(args.N, phase=0)
            label = "clean Néel"
        elif args.noise_model == "single-flip":
            psi0 = single_flip_admixed_neel(args.N, phase=0, fidelity=fidelity)
            label = f"single-flip, fidelity {fidelity:.2f}"
        else:
            psi0 = perturbed_neel_state(args.N, phase=0, fidelity=fidelity, rng=rng)
            label = f"Haar, fidelity {fidelity:.2f}"
        res = run_unitary(params, times, psi0=psi0, backend=args.backend)
        m_res = m_afm_rescaled(res.m_afm)
        color = cmap(k / max(1, len(args.fidelities) - 1))
        ax.plot(times, m_res, color=color, lw=1.8, label=label)

    ax.set_xlabel("time  t  (μs)")
    ax.set_ylabel(r"$M_{\mathrm{AFM}}^{\mathrm{res}}(t)$")
    ax.axhline(0.5, color="k", lw=0.6, ls=":", alpha=0.6,
               label="maximally mixed (1/2)")
    ax.set_title(
        rf"Sensitivity to initial-state imperfection  "
        rf"(N = {args.N}, $\Delta_l$ = {args.delta_l:g} MHz, {args.noise_model})",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85,
              title="initial state", title_fontsize=9.5)
    style_axes(ax)
    fig.tight_layout()

    save_figure_with_sidecar(
        fig,
        args.out,
        "fig_imperfection_sensitivity",
        {
            "N": args.N,
            "backend": args.backend,
            "delta_l": args.delta_l,
            "fidelities": args.fidelities,
            "seed": args.seed,
            "noise_model": args.noise_model,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
