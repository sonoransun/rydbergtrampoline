"""Tests for the bloqade / QuEra Aquila cloud backend.

These exercise the **emulator** path only — no real cloud submission, no
AWS credentials, no network. The two cost-gating tests verify that
``device='cloud'`` cannot accidentally submit a paid task (no flag, or no
auth, both raise before any program is built).
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from rydberg_trampoline.dynamics import run_unitary, run_unitary_async
from rydberg_trampoline.model import ModelParams
from rydberg_trampoline.states import computational_basis_vector, neel_state
from tests.conftest import bloqade_required


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


@bloqade_required
def test_emulator_smoke_runs_without_aws_env() -> None:
    """The emulator path runs to completion with no AWS env-vars set."""
    params = ModelParams(N=4, Omega=1.8, Delta_g=4.8, Delta_l=0.5, V_NN=6.0)
    times = np.array([0.0, 0.05, 0.1])
    res = run_unitary(params, times, backend="bloqade", n_shots=200, seed=1)
    assert res.m_afm.shape == times.shape
    assert res.backend == "bloqade-emulator"
    # M_AFM(0) on |gg...g⟩ is identically 0 for even N (no shot noise).
    assert res.m_afm[0] == 0.0


@bloqade_required
def test_run_unitary_async_is_a_coroutine_function() -> None:
    """The async entry point is a real coroutine (so `await` works)."""
    assert asyncio.iscoroutinefunction(run_unitary_async)


# ---------------------------------------------------------------------------
# Cost-gating
# ---------------------------------------------------------------------------


@bloqade_required
def test_cloud_without_safety_flag_raises() -> None:
    params = ModelParams(N=4, Omega=1.8, Delta_g=4.8, Delta_l=0.5, V_NN=6.0)
    with pytest.raises(RuntimeError, match="i_understand_this_costs_money"):
        run_unitary(
            params,
            np.array([0.1]),
            backend="bloqade",
            device="cloud",
            n_shots=10,
        )


@bloqade_required
def test_cloud_with_flag_but_no_creds_fails_at_auth_probe(monkeypatch) -> None:
    """The auth probe must fire BEFORE any program is built."""
    params = ModelParams(N=4, Omega=1.8, Delta_g=4.8, Delta_l=0.5, V_NN=6.0)
    # Strip every AWS env var so boto3 cannot find credentials.
    for var in (
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE", "AWS_DEFAULT_REGION", "AWS_REGION",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", "/nonexistent-test-home")
    with pytest.raises(RuntimeError, match="auth probe failed|cloud"):
        run_unitary(
            params,
            np.array([0.1]),
            backend="bloqade",
            device="cloud",
            n_shots=10,
            i_understand_this_costs_money=True,
        )


# ---------------------------------------------------------------------------
# Initial-state guard
# ---------------------------------------------------------------------------


@bloqade_required
def test_psi0_must_be_all_ground() -> None:
    params = ModelParams(N=4, Omega=1.8, Delta_g=4.8, Delta_l=0.5, V_NN=6.0)
    with pytest.raises(ValueError, match="all-ground"):
        run_unitary(
            params,
            np.array([0.1]),
            backend="bloqade",
            psi0=neel_state(4, phase=0),
            n_shots=10,
        )


@bloqade_required
def test_explicit_gg_state_is_accepted() -> None:
    params = ModelParams(N=4, Omega=1.8, Delta_g=4.8, Delta_l=0.5, V_NN=6.0)
    psi0 = computational_basis_vector(4, 0)  # |gg...g⟩
    res = run_unitary(params, np.array([0.0, 0.05]), backend="bloqade",
                      psi0=psi0, n_shots=100, seed=2)
    assert res.m_afm[0] == 0.0


# ---------------------------------------------------------------------------
# Cross-backend agreement (the substantial test)
# ---------------------------------------------------------------------------


@bloqade_required
def test_emulator_matches_numpy_from_ground_state() -> None:
    """Emulator and NumPy ED agree on M_AFM(t) within shot noise.

    Both backends start from |gg...g⟩ — the bloqade-allowed default — and
    evolve under identical Hamiltonian parameters. With n_shots=4000 the
    1-σ shot noise floor is ~1/√4000 ≈ 0.016; we tolerate 0.05 to stay
    comfortably above the 99 % statistical band.
    """
    params = ModelParams(N=4, Omega=1.8, Delta_g=4.8, Delta_l=0.5, V_NN=6.0)
    times = np.linspace(0.0, 0.5, 6)

    psi_gg = computational_basis_vector(4, 0)
    res_np = run_unitary(params, times, backend="numpy", psi0=psi_gg)
    res_bq = run_unitary(params, times, backend="bloqade", n_shots=4000, seed=42)

    # Both agree on the (shot-noise-free) initial value of 0.
    assert res_np.m_afm[0] == 0.0
    assert res_bq.m_afm[0] == 0.0

    diff = np.abs(res_np.m_afm - res_bq.m_afm)
    assert diff.max() < 0.05, (
        f"bloqade emulator deviates from NumPy by {diff.max():.4f} > 0.05; "
        "either the units conversion is wrong or shot statistics are too thin."
    )


# ---------------------------------------------------------------------------
# Async usage
# ---------------------------------------------------------------------------


@bloqade_required
def test_async_path_returns_dynamics_result() -> None:
    params = ModelParams(N=4, Omega=1.8, Delta_g=4.8, Delta_l=0.5, V_NN=6.0)

    async def go():
        return await run_unitary_async(
            params, np.array([0.0, 0.05]), n_shots=200, seed=7
        )

    res = asyncio.run(go())
    assert res.backend == "bloqade-emulator"
    assert len(res.m_afm) == 2
