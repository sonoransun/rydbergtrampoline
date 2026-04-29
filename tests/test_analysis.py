"""Decay-rate and tunneling-action fits."""
from __future__ import annotations

import numpy as np

from rydberg_trampoline.analysis import (
    fit_decay_rate,
    fit_tunneling_action,
)


def test_fit_decay_rate_recovers_synthetic_gamma() -> None:
    times = np.linspace(0.0, 5.0, 200)
    Gamma = 0.42
    m_res = 0.5 + 0.5 * np.exp(-Gamma * times)
    # Convert to the unrescaled form expected by fit_decay_rate (rescale=True).
    m_afm = 2 * m_res - 1
    fit = fit_decay_rate(times, m_afm)
    assert fit.success
    assert abs(fit.Gamma - Gamma) < 1e-3


def test_fit_tunneling_action_recovers_B() -> None:
    delta = np.linspace(0.5, 4.0, 12)
    A_true, B_true = 1.7, 0.8
    gamma = A_true * np.exp(-B_true / delta)
    fit = fit_tunneling_action(delta, gamma)
    assert fit.success
    assert abs(fit.B - B_true) < 1e-6
    assert abs(fit.A - A_true) / A_true < 1e-6
