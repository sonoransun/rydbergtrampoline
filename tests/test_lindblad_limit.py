"""Lindblad with T1, T2* → ∞ should reduce to the unitary trajectory."""
from __future__ import annotations

import numpy as np

from rydberg_trampoline.dynamics import run_lindblad, run_unitary
from rydberg_trampoline.model import ModelParams
from tests.conftest import qutip_required


@qutip_required
def test_qutip_lindblad_unitary_limit() -> None:
    params_closed = ModelParams(N=4, Delta_l=1.5)
    params_open = params_closed.with_(T1=1.0e6, T2_star=1.0e6)
    times = np.linspace(0.0, 1.0, 11)
    u = run_unitary(params_closed, times, backend="qutip")
    l = run_lindblad(params_open, times, backend="qutip", method="mesolve")
    diff = float(np.max(np.abs(u.m_afm - l.m_afm)))
    assert diff < 1e-4, f"unitary/Lindblad-no-decoherence disagree by {diff:.3g}"


def test_numpy_lindblad_returns_unit_initial() -> None:
    """Sanity: M_AFM(0) for the false-vacuum Néel must be +1."""
    params = ModelParams(N=4, Delta_l=1.0, T1=1.0, T2_star=0.5)
    times = np.linspace(0.0, 0.5, 6)
    res = run_lindblad(params, times, backend="numpy")
    assert abs(res.m_afm[0] - 1.0) < 1e-12
