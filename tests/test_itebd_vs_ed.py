"""iTEBD (N → ∞) vs ED (finite N): agreement at short times.

The TEBD-only iTEBD backend works on a 2-site iMPS unit cell with
nearest-neighbour vdW. With ``vdW_cutoff = 1`` the same model is exactly
representable in ED, so the two should agree to numerical precision until
the light cone reaches the boundary of the finite ring (roughly t ~ 2/V_NN).

This test pins that agreement and is a primary regression for the iTEBD
backend; if the TeNPy version changes the W-MPO convention or the TEBD
order, this test will catch it.
"""
from __future__ import annotations

import numpy as np

from rydberg_trampoline.dynamics import run_unitary
from rydberg_trampoline.model import ModelParams
from tests.conftest import tenpy_required


@tenpy_required
def test_itebd_matches_ed_short_time_nn_only() -> None:
    from rydberg_trampoline.dynamics import run_itebd

    params = ModelParams(N=12, Delta_l=2.0, vdW_cutoff=1)
    times = np.linspace(0.0, 0.4, 9)
    ed = run_unitary(params, times, backend="numpy")
    tn = run_itebd(params, times, chi=80)
    short_t_diff = float(np.max(np.abs(ed.m_afm[:4] - tn.m_afm[:4])))
    assert short_t_diff < 1e-6, f"iTEBD/ED short-time disagreement: {short_t_diff:.3g}"


@tenpy_required
def test_itebd_neel_state_starts_at_unit_m_afm() -> None:
    """Sanity: the false-vacuum Néel iMPS reads M_AFM = +1 before any evolution."""
    from rydberg_trampoline.dynamics import run_itebd

    params = ModelParams(N=12, Delta_l=2.0, vdW_cutoff=1)
    res = run_itebd(params, np.linspace(0.0, 0.05, 3), chi=20)
    assert abs(res.m_afm[0] - 1.0) < 1e-10
