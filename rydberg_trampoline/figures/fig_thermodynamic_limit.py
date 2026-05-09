"""Finite-N ED vs iTEBD (N → ∞) at the same Δ_l.

Shows ``M_AFM(t)`` from several closed-system unitary computations sharing
identical NN-only vdW parameters: a list of finite ring sizes ``N`` plus an
iTEBD curve in the thermodynamic limit. The finite-N curves coincide with
each other and with the iTEBD curve at short times, before the light cone
has wrapped around the ring; deviations at later times encode the
finite-size effects.

By default we plot ``N ∈ {8, 12, 16, 18}`` plus the iTEBD asymptote — a
proxy for the {32, 64} ring sizes that are out of reach for finite-N ED.
Override with ``--N-values 8 12 20 22`` for a denser finite-N sweep.

Backend selection per N is automatic via
:func:`figures._common.pick_unitary_backend`: NumPy below N=12, QuSpin
``kblock=0`` sector at N=13–22.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from rydberg_trampoline.dynamics import run_itebd, run_unitary
from rydberg_trampoline.figures._common import (
    PALETTE,
    common_argparser,
    pick_unitary_backend,
    save_figure_with_sidecar,
    style_axes,
)
from rydberg_trampoline.model import ModelParams


_N_COLORS = [
    "#34495e",
    PALETTE["accent2"],
    PALETTE["accent4"],
    PALETTE["accent3"],
    "#8e44ad",
    "#27ae60",
]


def _color_for_index(k: int, total: int) -> str:
    if k < len(_N_COLORS):
        return _N_COLORS[k]
    return plt.cm.viridis(k / max(total - 1, 1))


def main(argv: list[str] | None = None) -> int:
    parser = common_argparser(__doc__, add_n_values=True)
    parser.set_defaults(N_values=[8, 12, 16, 18])
    parser.add_argument("--delta-l", type=float, default=2.0)
    parser.add_argument("--t-max", type=float, default=5.0)
    parser.add_argument("--n-times", type=int, default=101)
    parser.add_argument("--chi", type=int, default=80)
    args = parser.parse_args(argv)

    times = np.linspace(0.0, args.t_max, args.n_times)
    # vdW_cutoff = 1 because TEBD is NN-only; we want apples-to-apples.
    base = ModelParams(N=2, Delta_l=args.delta_l, vdW_cutoff=1)

    n_values: list[int] = list(args.N_values) if args.N_values else [args.N]

    traces_by_N: dict[int, np.ndarray] = {}
    backend_per_N: dict[int, str] = {}
    for N in n_values:
        backend_name, backend_kwargs = pick_unitary_backend(N)
        backend_per_N[N] = backend_name + (
            f" (kblock={backend_kwargs['kblock']})" if "kblock" in backend_kwargs else ""
        )
        res = run_unitary(
            base.with_(N=N), times, backend=backend_name, **backend_kwargs
        )
        traces_by_N[N] = np.asarray(res.m_afm, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    for k, N in enumerate(n_values):
        c = _color_for_index(k, len(n_values) + 1)
        ax.plot(
            times, traces_by_N[N], color=c, lw=1.6,
            label=f"ED, N = {N}  ({backend_per_N[N]})",
        )

    itebd_status: str
    itebd_trace: list[float] | None = None
    try:
        res_itebd = run_itebd(base, times, chi=args.chi)
        ax.plot(
            times, res_itebd.m_afm, color=PALETTE["accent1"], lw=2.0, ls="--",
            label=fr"iTEBD, $N \to \infty$ ($\chi$ = {args.chi})",
        )
        itebd_status = f"iTEBD chi={args.chi}"
        itebd_trace = list(map(float, res_itebd.m_afm))
    except ModuleNotFoundError:
        itebd_status = "iTEBD skipped (TeNPy not installed)"
        ax.text(
            0.5, 0.05, itebd_status, transform=ax.transAxes,
            ha="center", color=PALETTE["accent1"],
        )

    # Light-cone annotation: locate where the smallest finite-N trace first
    # diverges from the next-larger one by > 0.02. Skip if only one N.
    if len(n_values) >= 2:
        N_small, N_next = sorted(n_values)[:2]
        diff = np.abs(traces_by_N[N_small] - traces_by_N[N_next])
        onset_mask = diff > 0.02
        if onset_mask.any():
            t_onset = times[np.argmax(onset_mask)]
            ax.axvline(t_onset, color=PALETTE["muted"], lw=0.8, ls=":")
            ax.text(
                t_onset + 0.05, 0.92,
                f"N={N_small} light cone\nreaches boundary  (t ≈ {t_onset:.1f} μs)",
                fontsize=9, color="#34495e", va="top",
            )

    ax.set_xlabel("time  t  (μs)")
    ax.set_ylabel(r"$\langle M_{\mathrm{AFM}}\rangle(t)$")
    ax.set_title(
        r"Finite-N ED vs iTEBD ($N \to \infty$) on the staggered Rydberg ring  "
        rf"($\Delta_l$ = {args.delta_l:g} MHz, NN-only vdW)"
    )
    ax.set_ylim(-0.55, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    style_axes(ax)
    fig.tight_layout()

    save_figure_with_sidecar(
        fig,
        args.out,
        "fig_thermodynamic_limit",
        {
            "N_values": n_values,
            "delta_l": args.delta_l,
            "t_max": args.t_max,
            "n_times": args.n_times,
            "chi": args.chi,
            "backend_per_N": {int(N): backend_per_N[N] for N in n_values},
            "traces_by_N": {int(N): traces_by_N[N].tolist() for N in n_values},
            "itebd_trace": itebd_trace,
            "itebd_status": itebd_status,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
