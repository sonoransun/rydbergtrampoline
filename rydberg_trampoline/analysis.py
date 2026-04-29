"""Post-processing of dynamics traces.

Two headline operations:

* :func:`fit_decay_rate` — extract Γ from a single ``M_AFM(t)`` trace by
  fitting the rescaled observable ``M^res(t) = (M(t) + M(0)) / (2 M(0))``
  to ``a · exp(-Γ t) + (1 - a) · 1/2`` (a one-sided exponential approach
  to the maximally mixed value 1/2).

* :func:`fit_tunneling_action` — given a list of (Δ_l, Γ) pairs, fit
  ``Γ(Δ_l) = A · exp(-B / Δ_l)`` and return ``(A, B)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(slots=True, frozen=True)
class DecayFit:
    """Result of fitting an exponential decay to a single trace."""

    Gamma: float
    """Decay rate, in inverse-time units of the input grid."""
    amplitude: float
    """The fitted ``a`` (initial weight on the +1 component)."""
    offset: float
    """The asymptotic value (held at 0.5 when ``free_offset=False``)."""
    success: bool
    """``True`` if the optimiser converged."""


def _decay_model(t: np.ndarray, Gamma: float, amplitude: float, offset: float) -> np.ndarray:
    return amplitude * np.exp(-Gamma * t) + offset


def fit_decay_rate(
    times: np.ndarray,
    m_afm: np.ndarray,
    *,
    rescale: bool = True,
    free_offset: bool = False,
    t_min: float | None = None,
    t_max: float | None = None,
) -> DecayFit:
    """Fit a single-exponential decay to an M_AFM trace.

    Parameters
    ----------
    times
        Time grid (μs).
    m_afm
        Either the raw ``⟨M_AFM⟩(t)`` or the already-rescaled trace.
    rescale
        If true (default), apply the paper's rescaling
        ``M^res(t) = (M(t) + M(0)) / (2 M(0))`` before fitting.
    free_offset
        If true, fit the asymptote as a free parameter instead of holding it
        at 1/2 (the maximally-mixed value).
    t_min, t_max
        Restrict the fit window. Default uses the full grid.
    """
    times = np.asarray(times, dtype=np.float64)
    m = np.asarray(m_afm, dtype=np.float64)
    if rescale:
        m0 = m[0]
        if abs(m0) < 1e-12:
            raise ValueError("M_AFM(0) ~ 0; cannot rescale (pass rescale=False if intentional)")
        m = (m + m0) / (2.0 * m0)

    mask = np.ones_like(times, dtype=bool)
    if t_min is not None:
        mask &= times >= t_min
    if t_max is not None:
        mask &= times <= t_max
    if mask.sum() < 3:
        raise ValueError("fit window contains fewer than 3 points")

    t_fit = times[mask]
    m_fit = m[mask]

    if free_offset:
        p0 = (1.0 / max(np.ptp(t_fit), 1e-9), m_fit[0] - 0.5, 0.5)
        try:
            popt, _ = curve_fit(_decay_model, t_fit, m_fit, p0=p0, maxfev=10000)
            return DecayFit(Gamma=popt[0], amplitude=popt[1], offset=popt[2], success=True)
        except RuntimeError:
            return DecayFit(Gamma=np.nan, amplitude=np.nan, offset=np.nan, success=False)
    else:
        def model_fixed(t, Gamma, amplitude):
            return _decay_model(t, Gamma, amplitude, 0.5)
        p0 = (1.0 / max(np.ptp(t_fit), 1e-9), m_fit[0] - 0.5)
        try:
            popt, _ = curve_fit(model_fixed, t_fit, m_fit, p0=p0, maxfev=10000)
            return DecayFit(Gamma=popt[0], amplitude=popt[1], offset=0.5, success=True)
        except RuntimeError:
            return DecayFit(Gamma=np.nan, amplitude=np.nan, offset=0.5, success=False)


@dataclass(slots=True, frozen=True)
class TunnelingFit:
    """Result of fitting Γ(Δ_l) = A · exp(-B / Δ_l)."""

    A: float
    B: float
    success: bool


def fit_tunneling_action(
    delta_l_values: np.ndarray, gamma_values: np.ndarray
) -> TunnelingFit:
    """Fit ``Γ(Δ_l) = A · exp(-B / Δ_l)`` and return the prefactor and action.

    Performs a linear regression on ``log Γ`` vs ``1/Δ_l`` for stability,
    which is exactly the form of the paper's Fig. 3.
    """
    x = 1.0 / np.asarray(delta_l_values, dtype=np.float64)
    y = np.log(np.asarray(gamma_values, dtype=np.float64))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return TunnelingFit(A=np.nan, B=np.nan, success=False)
    # log Γ = log A − B · (1/Δ_l)
    coef = np.polyfit(x, y, 1)
    slope, intercept = coef
    return TunnelingFit(A=float(np.exp(intercept)), B=float(-slope), success=True)


def find_resonances(
    delta_l_values: np.ndarray,
    gamma_values: np.ndarray,
    *,
    smooth_B: float | None = None,
    threshold: float = 2.0,
) -> np.ndarray:
    """Identify Δ_l locations where Γ exceeds the smooth ``A exp(-B/Δ_l)`` law.

    A Δ_l value is flagged as a resonance if its Γ is more than ``threshold``
    times the value predicted by the global exponential fit. ``smooth_B``
    overrides the fit to use a known theoretical action instead.
    """
    x = np.asarray(delta_l_values, dtype=np.float64)
    g = np.asarray(gamma_values, dtype=np.float64)
    fit = fit_tunneling_action(x, g)
    A = fit.A
    B = smooth_B if smooth_B is not None else fit.B
    smooth = A * np.exp(-B / x)
    return x[g > threshold * smooth]
