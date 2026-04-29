"""Cross-backend regression on N=8.

Pins the closed-system unitary M_AFM(t) trajectories from every available
backend to agree at the level of QuTiP's RK tolerance (≈ 1e-6). NumPy and
QuSpin both use Krylov methods and agree to machine precision.
"""
from __future__ import annotations

import numpy as np
import pytest

from rydberg_trampoline.dynamics import run_unitary
from rydberg_trampoline.model import ModelParams
from tests.conftest import qutip_required, quspin_required


N_REGRESSION = 8


def _run(backend: str):
    params = ModelParams(N=N_REGRESSION, Omega=1.8, Delta_g=4.8, Delta_l=2.0, V_NN=6.0, vdW_cutoff=4)
    times = np.linspace(0.0, 1.0, 21)
    return run_unitary(params, times, backend=backend)


@pytest.mark.parametrize("dl", [0.0, 0.5, 2.0])
def test_numpy_self_consistent(dl: float) -> None:
    params = ModelParams(N=N_REGRESSION, Delta_l=dl)
    times = np.linspace(0.0, 0.5, 11)
    a = run_unitary(params, times, backend="numpy")
    b = run_unitary(params, times, backend="numpy")
    np.testing.assert_allclose(a.m_afm, b.m_afm, atol=1e-14)


@quspin_required
def test_numpy_quspin_agree() -> None:
    a = _run("numpy")
    b = _run("quspin")
    diff = float(np.max(np.abs(a.m_afm - b.m_afm)))
    assert diff < 1e-10, f"NumPy/QuSpin disagree by {diff:.3g}"


@qutip_required
def test_numpy_qutip_agree() -> None:
    a = _run("numpy")
    b = _run("qutip")
    diff = float(np.max(np.abs(a.m_afm - b.m_afm)))
    # QuTiP uses adaptive RK, so tolerance is looser than the Krylov-vs-Krylov check.
    assert diff < 1e-5, f"NumPy/QuTiP disagree by {diff:.3g}"


# The iTEBD-vs-ED short-time agreement test now lives in tests/test_itebd_vs_ed.py.
