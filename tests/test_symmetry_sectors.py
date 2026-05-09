"""QuSpin: spectrum decomposed by translation-by-2 sectors equals full spectrum."""
from __future__ import annotations

import numpy as np
import pytest

from rydberg_trampoline.model import ModelParams
from tests.conftest import quspin_required


@quspin_required
@pytest.mark.parametrize("N", [4, 6, 8, 10, 12])
def test_kblock_decomposition_recovers_full_spectrum(N: int) -> None:
    from rydberg_trampoline.backends.quspin_backend import (
        to_quspin,
        translation_block_count,
    )

    params = ModelParams(N=N, Omega=1.8, Delta_g=4.8, Delta_l=2.0, V_NN=6.0, vdW_cutoff=8)

    H_full, b_full = to_quspin(params)
    full_eigs = np.sort(np.linalg.eigvalsh(H_full.toarray()))

    sector_eigs = []
    total_dim = 0
    for k in range(translation_block_count(N)):
        H_k, b_k = to_quspin(params, kblock=k)
        total_dim += b_k.Ns
        eigs = np.linalg.eigvalsh(H_k.toarray())
        sector_eigs.append(eigs)
    combined = np.sort(np.concatenate(sector_eigs))

    assert total_dim == 1 << N, f"sector dims sum to {total_dim} != {1 << N}"
    np.testing.assert_allclose(combined, full_eigs, atol=1e-10)


@quspin_required
def test_translation_block_count_value() -> None:
    from rydberg_trampoline.backends.quspin_backend import translation_block_count

    assert translation_block_count(4) == 2
    assert translation_block_count(8) == 4
    assert translation_block_count(16) == 8
    with pytest.raises(ValueError):
        translation_block_count(7)


@quspin_required
@pytest.mark.parametrize("N", [8, 12])
def test_sector_dynamics_match_full(N: int) -> None:
    """Sector-resolved M_AFM(t) from the Néel matches full-Hilbert evolution.

    Parametrising up to N=12 pins the kblock=0 ↔ full agreement at the
    largest size where the full path is still fast enough for CI; the
    figure pipeline relies on this equivalence to dispatch quspin/kblock=0
    automatically for N ≥ 13 via :func:`figures._common.pick_unitary_backend`.
    """
    from rydberg_trampoline.dynamics import run_unitary

    params = ModelParams(N=N, Delta_l=2.0, vdW_cutoff=4)
    times = np.linspace(0.0, 1.0, 21)
    res_full = run_unitary(params, times, backend="quspin")
    res_k0 = run_unitary(params, times, backend="quspin", kblock=0)
    diff = float(np.max(np.abs(res_full.m_afm - res_k0.m_afm)))
    assert diff < 1e-10, f"sector vs full M_AFM disagree by {diff:.3g}"
    # Confirm the sector path actually reduced dimension (the speedup point).
    assert "dim=" in res_k0.notes
    assert "kblock=0" in res_k0.notes


@quspin_required
def test_sector_rejects_state_outside_sector() -> None:
    """A Néel + perturbed state may not live in (kblock=0); check the guard."""
    import numpy as np
    from rydberg_trampoline.conventions import neel_bitstring
    from rydberg_trampoline.dynamics import run_unitary
    from rydberg_trampoline.states import computational_basis_vector

    params = ModelParams(N=8, Delta_l=2.0, vdW_cutoff=4)
    # A single computational basis state that is *not* T_2-invariant.
    # |10000000> (only site 0 occupied) is NOT a Néel-like state and
    # has overlap with multiple kblocks.
    psi0 = computational_basis_vector(8, 1)  # site 0 = 1, others = 0
    times = np.linspace(0.0, 0.1, 3)
    with pytest.raises(ValueError, match="does not lie purely"):
        run_unitary(params, times, psi0=psi0, backend="quspin", kblock=1)
