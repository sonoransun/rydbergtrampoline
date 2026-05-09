"""Finite-size scaling: Γ vs 1/Δ_l overlaid across ring sizes N.

Sweeps Δ_l for each N in ``--N-values`` and fits the tunneling action
``B(N)`` in ``Γ(Δ_l) ≈ A · exp(-B / Δ_l)``. Three "modes" share the same
pipeline:

* **Mode A** (proxy)   — finite-N ED at small N + iTEBD curve as the
  ``N→∞`` (NN-only) anchor. Pass ``--include-itebd`` and a small
  ``--N-values`` list (e.g. ``8 12 16 18``).
* **Mode B** (hybrid)  — denser finite-N ED (e.g. ``16 18 20 22``) plus
  the iTEBD overlay. Same flags, larger ring sizes.
* **Mode C** (cloud)   — finite-N data at ``--cloud-N 32 64`` via the
  bloqade Aquila path with ``psi0_protocol='neel_via_ramp'``. Requires
  AWS credentials and the explicit
  ``--i-understand-this-costs-money`` opt-in. Local backends still
  handle ``--N-values``; cloud only handles ``--cloud-N``.

Backend selection for finite-N ED is automatic per-N via
:func:`figures._common.pick_unitary_backend`: NumPy below N=12, QuSpin
``kblock=0`` sector at N=13–22.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from rydberg_trampoline.analysis import fit_decay_rate, fit_tunneling_action
from rydberg_trampoline.dynamics import run_itebd, run_unitary
from rydberg_trampoline.figures._common import (
    PALETTE,
    common_argparser,
    pick_unitary_backend,
    save_figure_with_sidecar,
    style_axes,
)
from rydberg_trampoline.model import ModelParams


# Color cycle for up to ~8 N-values; falls back to viridis past that.
_N_COLORS = [
    PALETTE["ink"],
    PALETTE["accent4"],
    PALETTE["accent2"],
    PALETTE["accent3"],
    PALETTE["accent1"],
    "#8e44ad",
    "#27ae60",
    "#d35400",
]


def _color_for_index(k: int, total: int) -> str:
    if k < len(_N_COLORS):
        return _N_COLORS[k]
    return plt.cm.viridis(k / max(total - 1, 1))


def _gammas_for_N(
    N: int,
    deltas: np.ndarray,
    times: np.ndarray,
    *,
    fit_window_frac: float,
    backend_name: str,
    backend_kwargs: dict,
    vdw_cutoff: int,
) -> np.ndarray:
    """Run the Δ_l sweep at fixed N and return Γ(Δ_l)."""
    gammas = np.empty_like(deltas)
    for k, dl in enumerate(deltas):
        params = ModelParams(N=N, Delta_l=float(dl), vdW_cutoff=vdw_cutoff)
        res = run_unitary(params, times, backend=backend_name, **backend_kwargs)
        fit = fit_decay_rate(times, res.m_afm, t_max=times[-1] * fit_window_frac)
        gammas[k] = fit.Gamma if fit.success else np.nan
    return gammas


def _gammas_for_itebd(
    deltas: np.ndarray,
    times: np.ndarray,
    *,
    fit_window_frac: float,
    chi: int,
) -> np.ndarray:
    gammas = np.empty_like(deltas)
    for k, dl in enumerate(deltas):
        params = ModelParams(N=2, Delta_l=float(dl), vdW_cutoff=1)
        res = run_itebd(params, times, chi=chi)
        fit = fit_decay_rate(times, res.m_afm, t_max=times[-1] * fit_window_frac)
        gammas[k] = fit.Gamma if fit.success else np.nan
    return gammas


def _gammas_for_cloud(
    N: int,
    deltas: np.ndarray,
    times: np.ndarray,
    *,
    fit_window_frac: float,
    n_shots: int,
    seed: int | None,
    i_understand_this_costs_money: bool,
) -> np.ndarray:
    """Mode C: bloqade cloud path with Néel state-prep ramp.

    One Aquila task per (Δ_l, t) sample. Shot-noisy; the resulting Γ fits
    are coarser than the local-ED curves and benefit from larger n_shots.
    """
    from rydberg_trampoline.states import neel_state

    psi0 = neel_state(N, phase=0)
    gammas = np.empty_like(deltas)
    for k, dl in enumerate(deltas):
        params = ModelParams(N=N, Delta_l=float(dl))
        res = run_unitary(
            params,
            times,
            backend="bloqade",
            psi0=psi0,
            psi0_protocol="neel_via_ramp",
            device="cloud",
            i_understand_this_costs_money=i_understand_this_costs_money,
            n_shots=n_shots,
            seed=None if seed is None else seed + k,
        )
        fit = fit_decay_rate(times, res.m_afm, t_max=times[-1] * fit_window_frac)
        gammas[k] = fit.Gamma if fit.success else np.nan
    return gammas


def main(argv: list[str] | None = None) -> int:
    parser = common_argparser(__doc__, add_n_values=True)
    parser.set_defaults(N_values=[8, 12, 16, 18])
    parser.add_argument(
        "--delta-l",
        type=float,
        nargs="+",
        default=list(np.linspace(0.4, 3.0, 14)),
        help="Δ_l values to sweep (MHz)",
    )
    parser.add_argument("--t-max", type=float, default=3.0, help="evolution time (μs)")
    parser.add_argument("--n-times", type=int, default=121)
    parser.add_argument(
        "--fit-window-frac",
        type=float,
        default=0.6,
        help="fraction of the time window used for the Γ fit (default 0.6)",
    )
    parser.add_argument(
        "--vdw-cutoff",
        type=int,
        default=8,
        help="vdW truncation distance for finite-N ED runs (default 8 = paper)",
    )
    parser.add_argument(
        "--include-itebd",
        action="store_true",
        help="overlay an iTEBD N→∞ (NN-only) curve",
    )
    parser.add_argument(
        "--chi",
        type=int,
        default=80,
        help="iTEBD bond dimension (only used with --include-itebd)",
    )
    parser.add_argument(
        "--cloud-N",
        type=int,
        nargs="*",
        default=None,
        help="ring sizes to dispatch via the bloqade Aquila cloud path (Mode C)",
    )
    parser.add_argument(
        "--cloud-n-shots",
        type=int,
        default=1000,
        help="shots per (Δ_l, t) point for the cloud path (default 1000)",
    )
    parser.add_argument(
        "--i-understand-this-costs-money",
        action="store_true",
        help="required for --cloud-N submission",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    times = np.linspace(0.0, args.t_max, args.n_times)
    deltas = np.asarray(args.delta_l, dtype=np.float64)

    n_values: list[int] = list(args.N_values) if args.N_values else [args.N]
    cloud_n: list[int] = list(args.cloud_N) if args.cloud_N else []

    gammas_by_N: dict[int, np.ndarray] = {}
    fit_B_by_N: dict[int, float] = {}
    fit_A_by_N: dict[int, float] = {}
    backend_per_N: dict[int, str] = {}

    # Local finite-N ED dispatched per-N via pick_unitary_backend.
    for N in n_values:
        backend_name, backend_kwargs = pick_unitary_backend(N)
        backend_per_N[N] = backend_name + (
            f" (kblock={backend_kwargs['kblock']})" if "kblock" in backend_kwargs else ""
        )
        gammas = _gammas_for_N(
            N,
            deltas,
            times,
            fit_window_frac=args.fit_window_frac,
            backend_name=backend_name,
            backend_kwargs=backend_kwargs,
            vdw_cutoff=args.vdw_cutoff,
        )
        gammas_by_N[N] = gammas
        valid = np.isfinite(gammas) & (gammas > 0)
        if valid.sum() >= 2:
            tfit = fit_tunneling_action(deltas[valid], gammas[valid])
            fit_A_by_N[N] = float(tfit.A) if tfit.success else float("nan")
            fit_B_by_N[N] = float(tfit.B) if tfit.success else float("nan")
        else:
            fit_A_by_N[N] = float("nan")
            fit_B_by_N[N] = float("nan")

    # Mode C: cloud finite-N. Lives in a separate label space (not in
    # backend_per_N) because the dispatch is "bloqade-cloud" regardless of N.
    cloud_gammas: dict[int, np.ndarray] = {}
    cloud_fit_B: dict[int, float] = {}
    cloud_fit_A: dict[int, float] = {}
    for N in cloud_n:
        gammas = _gammas_for_cloud(
            N,
            deltas,
            times,
            fit_window_frac=args.fit_window_frac,
            n_shots=args.cloud_n_shots,
            seed=args.seed,
            i_understand_this_costs_money=args.i_understand_this_costs_money,
        )
        cloud_gammas[N] = gammas
        valid = np.isfinite(gammas) & (gammas > 0)
        if valid.sum() >= 2:
            tfit = fit_tunneling_action(deltas[valid], gammas[valid])
            cloud_fit_A[N] = float(tfit.A) if tfit.success else float("nan")
            cloud_fit_B[N] = float(tfit.B) if tfit.success else float("nan")
        else:
            cloud_fit_A[N] = float("nan")
            cloud_fit_B[N] = float("nan")

    # iTEBD asymptote.
    itebd_gammas: np.ndarray | None = None
    itebd_fit_B: float | None = None
    itebd_fit_A: float | None = None
    if args.include_itebd:
        try:
            itebd_gammas = _gammas_for_itebd(
                deltas, times, fit_window_frac=args.fit_window_frac, chi=args.chi
            )
            valid = np.isfinite(itebd_gammas) & (itebd_gammas > 0)
            if valid.sum() >= 2:
                tfit = fit_tunneling_action(deltas[valid], itebd_gammas[valid])
                itebd_fit_A = float(tfit.A) if tfit.success else float("nan")
                itebd_fit_B = float(tfit.B) if tfit.success else float("nan")
        except ModuleNotFoundError:
            print("iTEBD overlay skipped: TeNPy not installed.")
            itebd_gammas = None

    # ----- plot -----
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    grid = np.linspace(deltas.min() * 0.95, deltas.max() * 1.05, 200)
    n_total = len(n_values) + len(cloud_n) + (1 if itebd_gammas is not None else 0)

    color_idx = 0
    for N in n_values:
        gammas = gammas_by_N[N]
        valid = np.isfinite(gammas) & (gammas > 0)
        c = _color_for_index(color_idx, n_total)
        color_idx += 1
        ax.scatter(
            1.0 / deltas[valid], gammas[valid], s=34, color=c,
            zorder=3, label=f"N = {N}  ({backend_per_N[N]})",
        )
        if np.isfinite(fit_B_by_N[N]):
            ax.plot(
                1.0 / grid,
                fit_A_by_N[N] * np.exp(-fit_B_by_N[N] / grid),
                color=c, lw=1.4, alpha=0.7,
            )
    for N in cloud_n:
        gammas = cloud_gammas[N]
        valid = np.isfinite(gammas) & (gammas > 0)
        c = _color_for_index(color_idx, n_total)
        color_idx += 1
        ax.scatter(
            1.0 / deltas[valid], gammas[valid], s=34, color=c, marker="D",
            zorder=3, label=f"N = {N}  (Aquila cloud)",
        )
        if np.isfinite(cloud_fit_B[N]):
            ax.plot(
                1.0 / grid,
                cloud_fit_A[N] * np.exp(-cloud_fit_B[N] / grid),
                color=c, lw=1.4, ls="--", alpha=0.7,
            )
    if itebd_gammas is not None:
        valid = np.isfinite(itebd_gammas) & (itebd_gammas > 0)
        c = PALETTE["accent1"]
        ax.scatter(
            1.0 / deltas[valid], itebd_gammas[valid], s=42, color=c,
            marker="*", zorder=4, label=r"$N\to\infty$ (iTEBD, NN-only)",
        )
        if itebd_fit_B is not None and np.isfinite(itebd_fit_B):
            ax.plot(
                1.0 / grid,
                itebd_fit_A * np.exp(-itebd_fit_B / grid),
                color=c, lw=2.0, ls="--",
            )

    ax.set_yscale("log")
    ax.set_xlabel(r"$1 / \Delta_l$  (1/MHz)   $\to$  smaller $\Delta_l$, deeper false vacuum")
    ax.set_ylabel(r"decay rate  $\Gamma$  (1/μs)")
    ax.set_title(
        "Finite-size scaling of false-vacuum decay  "
        f"(N ∈ {{{', '.join(str(N) for N in n_values)}}}"
        + (f" + cloud {{{', '.join(str(N) for N in cloud_n)}}}" if cloud_n else "")
        + (r" + $N\to\infty$" if itebd_gammas is not None else "")
        + ")",
        fontsize=10.5,
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)

    # Annotate B(N) trend in the corner.
    if fit_B_by_N:
        Bs = ", ".join(
            f"B(N={N})={fit_B_by_N[N]:.2f}"
            for N in n_values
            if np.isfinite(fit_B_by_N[N])
        )
        if itebd_fit_B is not None and np.isfinite(itebd_fit_B):
            Bs += f"  |  B(∞)={itebd_fit_B:.2f}"
        ax.text(
            0.98, 0.98, Bs,
            transform=ax.transAxes, fontsize=8.5, color=PALETTE["muted"],
            ha="right", va="top", style="italic",
        )

    style_axes(ax)
    fig.tight_layout()

    sidecar_params = {
        "N_values": n_values,
        "cloud_N": cloud_n,
        "delta_l": deltas.tolist(),
        "gammas_by_N": {int(N): gammas_by_N[N].tolist() for N in n_values},
        "gammas_by_cloud_N": {int(N): cloud_gammas[N].tolist() for N in cloud_n},
        "fit_A_by_N": {int(N): fit_A_by_N[N] for N in n_values},
        "fit_B_by_N": {int(N): fit_B_by_N[N] for N in n_values},
        "fit_A_by_cloud_N": {int(N): cloud_fit_A[N] for N in cloud_n},
        "fit_B_by_cloud_N": {int(N): cloud_fit_B[N] for N in cloud_n},
        "backend_per_N": {int(N): backend_per_N[N] for N in n_values},
        "itebd_gammas": (itebd_gammas.tolist() if itebd_gammas is not None else None),
        "itebd_fit_A": itebd_fit_A,
        "itebd_fit_B": itebd_fit_B,
        "t_max": args.t_max,
        "n_times": args.n_times,
        "fit_window_frac": args.fit_window_frac,
        "vdw_cutoff": args.vdw_cutoff,
        "chi": args.chi if args.include_itebd else None,
    }
    save_figure_with_sidecar(fig, args.out, "fig_gamma_N_dependence", sidecar_params)
    for N in n_values:
        if np.isfinite(fit_B_by_N[N]):
            print(
                f"N={N:>3d}  ({backend_per_N[N]:>22s})  "
                f"B={fit_B_by_N[N]:.3f}, A={fit_A_by_N[N]:.3g}"
            )
    if itebd_fit_B is not None and np.isfinite(itebd_fit_B):
        print(f"N→∞  (iTEBD, NN-only)            B={itebd_fit_B:.3f}, A={itebd_fit_A:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
