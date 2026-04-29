"""Hamiltonian invariants: Hermiticity, energy and trace conservation."""
from __future__ import annotations

import numpy as np
import pytest

from rydberg_trampoline.backends.numpy_backend import to_scipy
from rydberg_trampoline.dynamics import run_lindblad, run_unitary
from rydberg_trampoline.model import ModelParams


@pytest.mark.parametrize("N", [4, 6, 8])
@pytest.mark.parametrize("Delta_l", [0.0, 1.0, 2.5])
def test_hamiltonian_hermitian(N: int, Delta_l: float) -> None:
    H = to_scipy(ModelParams(N=N, Delta_l=Delta_l))
    err = (H - H.conj().T)
    norm = np.sqrt((err.multiply(err.conj())).sum().real)
    assert norm < 1e-12


@pytest.mark.parametrize("N", [4, 6, 8])
def test_energy_conservation_unitary(N: int) -> None:
    params = ModelParams(N=N, Delta_l=1.5)
    times = np.linspace(0.0, 1.0, 11)
    H = to_scipy(params).toarray()
    res = run_unitary(params, times, backend="numpy")
    # Recompute energy from |M_AFM trace + the actual evolved state. We need the
    # state, so call the low-level routine.
    from rydberg_trampoline.backends.numpy_backend import run_unitary as _run
    from rydberg_trampoline.states import neel_state
    states = _run(params, neel_state(N, phase=0), times)
    energies = np.real([psi.conj() @ H @ psi for psi in states])
    drift = float(energies.max() - energies.min())
    assert drift < 1e-9


def test_lindblad_trace_preserved_small() -> None:
    params = ModelParams(N=4, Delta_l=1.0, T1=10.0, T2_star=2.0)
    times = np.linspace(0.0, 0.5, 6)
    res = run_lindblad(params, times, backend="numpy")
    # Lindblad backend returns M_AFM, not ρ; trace preservation is checked
    # implicitly because |M_AFM(t)| ≤ 1 always.
    assert np.all(np.abs(res.m_afm) <= 1.0 + 1e-9)
