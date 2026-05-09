"""Render the bloqade Néel-prep ramp profile.

Drives `docs/figures/neel_prep_ramp.png`. Two stacked time-axis panels
plot Ω(t) and Δ_uniform(t) under the :class:`NeelPrepRamp` defaults
from ``backends/bloqade_backend.py``: ``t_up = 1.0``, ``t_sweep = 5.0``,
``t_down = 1.0`` μs (so ``prep_time = 7 μs``), with
``omega_max_factor = 4.0`` and ``delta_max_factor = 4.0``. A third
strip below shows the per-site staggered detuning offset jumping
piecewise-constant from prep magnitude (``±delta_l_prep``) to physics
magnitude (``∓Δ_l``) at the prep→evolution boundary.

Why we sign-flip the per-site offsets at the boundary: the prep field
favours even-occupied (the false vacuum) while the evolution field
under the package's physics convention then makes the prepared state
the metastable Néel. See the docstring of ``_build_program`` for the
full derivation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rydberg_trampoline.backends.bloqade_backend import NeelPrepRamp


# Project palette (matches PALETTE in figures/_common.py).
INK = "#34495e"
ACCENT_GREEN = "#16a085"
ACCENT_BLUE = "#2980b9"
ACCENT_RED = "#c0392b"
ACCENT_ORANGE = "#e67e22"
MUTED = "#7f8c8d"
PHASE_BANDS = ("#fdf2e9", "#ecf0f1", "#fdf2e9", "#d5f5e3")
PHASE_LABELS = ("up", "sweep", "down", "evolve")


def _shade_phases(ax, ramp: NeelPrepRamp, T_evol: float, *, label_y: float):
    """Add coloured vertical bands behind the curves to label phases."""
    bounds = [
        0.0,
        ramp.t_up,
        ramp.t_up + ramp.t_sweep,
        ramp.t_up + ramp.t_sweep + ramp.t_down,
        ramp.t_up + ramp.t_sweep + ramp.t_down + T_evol,
    ]
    for k, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        ax.axvspan(lo, hi, color=PHASE_BANDS[k], alpha=0.65, zorder=0)
        ax.text(0.5 * (lo + hi), label_y, PHASE_LABELS[k],
                fontsize=8.5, color=INK, ha="center", va="bottom")


def _omega_profile(ramp: NeelPrepRamp, omega: float, T_evol: float):
    """Knot points (t, Ω) defining the piecewise-linear Rabi profile."""
    omega_max = ramp.omega_max_factor * omega
    t = [0.0]
    y = [0.0]
    t.append(ramp.t_up);                            y.append(omega_max)
    t.append(ramp.t_up + ramp.t_sweep);             y.append(omega_max)
    t.append(ramp.t_up + ramp.t_sweep + ramp.t_down); y.append(omega)
    t.append(t[-1] + T_evol);                        y.append(omega)
    return np.asarray(t), np.asarray(y)


def _delta_profile(ramp: NeelPrepRamp, delta_g: float, T_evol: float):
    """Knot points for Δ_uniform(t)."""
    delta_max = ramp.delta_max_factor * delta_g
    t = [0.0]; y = [-delta_max]
    t.append(ramp.t_up);                             y.append(-delta_max)
    t.append(ramp.t_up + ramp.t_sweep);              y.append(+delta_max)
    t.append(ramp.t_up + ramp.t_sweep + ramp.t_down); y.append(delta_g)
    t.append(t[-1] + T_evol);                         y.append(delta_g)
    return np.asarray(t), np.asarray(y)


def _delta_l_profile(ramp: NeelPrepRamp, delta_l: float, T_evol: float):
    """Knot points for the per-site detuning offset (even / odd).

    Returns three arrays: t_steps, even_values, odd_values.
    The waveform is piecewise-constant with the jump at t = prep_time.
    """
    delta_l_prep = ramp.delta_l_prep_factor * delta_l
    prep_total = ramp.prep_time
    t = [0.0, prep_total, prep_total, prep_total + T_evol]
    even = [-delta_l_prep, -delta_l_prep, -delta_l, -delta_l]
    odd = [+delta_l_prep, +delta_l_prep, +delta_l, +delta_l]
    return np.asarray(t), np.asarray(even), np.asarray(odd)


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    ramp = NeelPrepRamp()
    omega = 1.8
    delta_g = 4.8
    delta_l = 0.5
    T_evol = 2.0

    fig, (ax_omega, ax_delta, ax_local) = plt.subplots(
        3, 1, figsize=(8.0, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.7], "hspace": 0.18},
    )

    # ----- Ω(t) panel -----
    t_O, y_O = _omega_profile(ramp, omega, T_evol)
    _shade_phases(ax_omega, ramp, T_evol, label_y=ramp.omega_max_factor * omega * 1.05)
    ax_omega.plot(t_O, y_O, color=ACCENT_GREEN, lw=2.2)
    ax_omega.axhline(omega, color=MUTED, lw=0.8, ls=":", alpha=0.7)
    ax_omega.text(t_O[-1] + 0.2, omega, r"$\Omega$ (evolve)",
                  fontsize=9, color=MUTED, va="center")
    ax_omega.text(t_O[-1] + 0.2, ramp.omega_max_factor * omega,
                  r"$\Omega_{\max}$ (prep)",
                  fontsize=9, color=ACCENT_GREEN, va="center")
    ax_omega.set_ylabel(r"$\Omega(t)$  (rad/μs)", fontsize=10, color=INK)
    ax_omega.spines["top"].set_visible(False)
    ax_omega.spines["right"].set_visible(False)
    ax_omega.set_ylim(-0.6, ramp.omega_max_factor * omega * 1.20)

    # ----- Δ_uniform(t) panel -----
    t_D, y_D = _delta_profile(ramp, delta_g, T_evol)
    _shade_phases(ax_delta, ramp, T_evol,
                  label_y=ramp.delta_max_factor * delta_g * 1.05)
    ax_delta.plot(t_D, y_D, color=ACCENT_BLUE, lw=2.2)
    ax_delta.axhline(0.0, color=MUTED, lw=0.6, ls="-", alpha=0.5)
    ax_delta.axhline(delta_g, color=MUTED, lw=0.8, ls=":", alpha=0.7)
    ax_delta.text(t_D[-1] + 0.2, delta_g, r"$\Delta_g$ (evolve)",
                  fontsize=9, color=MUTED, va="center")
    ax_delta.text(t_D[-1] + 0.2, ramp.delta_max_factor * delta_g,
                  r"$+\Delta_{\max}$",
                  fontsize=9, color=ACCENT_BLUE, va="center")
    ax_delta.text(t_D[-1] + 0.2, -ramp.delta_max_factor * delta_g,
                  r"$-\Delta_{\max}$",
                  fontsize=9, color=ACCENT_BLUE, va="center")
    ax_delta.set_ylabel(r"$\Delta_{\mathrm{uniform}}(t)$  (rad/μs)",
                        fontsize=10, color=INK)
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["right"].set_visible(False)
    ax_delta.set_ylim(-ramp.delta_max_factor * delta_g * 1.20,
                      ramp.delta_max_factor * delta_g * 1.20)

    # ----- Per-site offset panel: piecewise-constant, sign jump at prep boundary -----
    t_L, even, odd = _delta_l_profile(ramp, delta_l, T_evol)
    delta_l_prep = ramp.delta_l_prep_factor * delta_l
    _shade_phases(ax_local, ramp, T_evol,
                  label_y=delta_l_prep * 1.10)

    # Drawn as step functions.
    ax_local.step(t_L, even, where="post", color=ACCENT_RED, lw=2.0,
                  label=r"even-site offset")
    ax_local.step(t_L, odd, where="post", color=ACCENT_ORANGE, lw=2.0,
                  label=r"odd-site offset")
    ax_local.axhline(0.0, color=MUTED, lw=0.6, ls="-", alpha=0.5)

    # Annotate the magnitude jump at the prep→evolution boundary.
    boundary = ramp.prep_time
    ax_local.axvline(boundary, color=INK, lw=0.7, ls="--", alpha=0.5)
    ax_local.annotate(
        r"piecewise-constant jump:  $\pm\Delta_{\ell,\,\mathrm{prep}}\to\mp\Delta_\ell$",
        xy=(boundary, 0.0),
        xytext=(boundary - 1.2, delta_l_prep * 0.55),
        fontsize=8.5, color=INK, ha="right",
        arrowprops=dict(arrowstyle="->", color=INK, lw=0.7),
    )
    ax_local.set_ylabel(r"per-site offset  (rad/μs)", fontsize=10, color=INK)
    ax_local.set_xlabel("time  t  (μs)", fontsize=11, color=INK)
    ax_local.spines["top"].set_visible(False)
    ax_local.spines["right"].set_visible(False)
    ax_local.legend(loc="lower right", fontsize=8.5, framealpha=0.85, ncols=2)
    ax_local.set_ylim(-delta_l_prep * 1.30, delta_l_prep * 1.30)

    ax_omega.set_title(
        r"Bloqade Néel-prep ramp + evolution  "
        rf"(prep = {ramp.prep_time:.0f} μs, then evolve at $\Omega, \Delta_g$)",
        fontsize=11.5, color=INK,
    )

    fig.tight_layout()
    out_path = out_dir / "neel_prep_ramp.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
