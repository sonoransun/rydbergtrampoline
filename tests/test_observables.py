"""Dedicated observable tests."""
from __future__ import annotations

import numpy as np
import pytest

from rydberg_trampoline.conventions import neel_bitstring, neel_occupation
from rydberg_trampoline.observables import (
    bubble_correlator_expectation,
    m_afm_diagonal,
    m_afm_expectation,
    m_afm_rescaled,
    sigma_L_diagonal,
    site_occupations,
)
from rydberg_trampoline.states import (
    computational_basis_vector,
    equal_superposition_state,
    neel_state,
    perturbed_neel_state,
)


# ---------------------------------------------------------------------------
# m_afm_diagonal & m_afm_expectation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [4, 6, 8, 10])
def test_m_afm_diagonal_extrema_match_neel(N: int) -> None:
    diag = m_afm_diagonal(N)
    assert pytest.approx(diag[neel_bitstring(N, phase=0)]) == 1.0
    assert pytest.approx(diag[neel_bitstring(N, phase=1)]) == -1.0


@pytest.mark.parametrize("N", [4, 6, 8])
def test_m_afm_diagonal_bounded(N: int) -> None:
    diag = m_afm_diagonal(N)
    assert diag.shape == (1 << N,)
    assert np.all(np.abs(diag) <= 1.0 + 1e-12)


def test_m_afm_expectation_on_uniform_superposition_is_zero() -> None:
    psi = equal_superposition_state(8)
    # Equal weights on every basis state averages M_AFM to zero by symmetry.
    assert abs(m_afm_expectation(psi)) < 1e-12


def test_m_afm_accepts_density_matrix_diagonal() -> None:
    N = 4
    diag_rho = np.zeros(1 << N)
    diag_rho[neel_bitstring(N, phase=0)] = 0.7
    diag_rho[neel_bitstring(N, phase=1)] = 0.3
    expected = 0.7 * 1.0 + 0.3 * (-1.0)
    got = m_afm_expectation(diag_rho, N)
    assert pytest.approx(got) == expected


# ---------------------------------------------------------------------------
# Σ_L bubble correlators
# ---------------------------------------------------------------------------


def _state_from_occupation(N: int, occ: tuple[int, ...]) -> np.ndarray:
    bits = 0
    for j, b in enumerate(occ):
        if b:
            bits |= 1 << j
    return computational_basis_vector(N, bits)


def test_sigma_L_neel_state_has_no_bubbles() -> None:
    psi = neel_state(8, phase=0)
    for L in (1, 2, 3):
        assert pytest.approx(bubble_correlator_expectation(psi, L)) == 0.0


def test_sigma_L_counts_two_separate_length_one_bubbles() -> None:
    # N=8 ring, FV = (1,0,1,0,1,0,1,0); flip site 1 (→TV) and flip site 5 (→TV).
    # Each is a length-1 bubble bordered by FV on both sides.
    occ = (1, 1, 1, 0, 1, 0, 1, 0)  # this is wrong because flipping site 1 to TV (TV at site 1 = n=1) → site 1=1, site 0=1 stays in FV. Site 0 already 1 in FV... let me think.
    # The false vacuum on phase=0 has n_j = 1 if j even else 0. So site 0=1, 1=0, 2=1, 3=0, etc.
    # True vacuum is the opposite: site 0=0, 1=1, 2=0, etc.
    # A length-1 bubble centred at site j is when site j is in TV and j-1, j+1 are in FV.
    # Site 1 in TV means n_1 = 1 (since FV has n_1 = 0); neighbours site 0 in FV (n_0=1) and site 2 in FV (n_2=1). All ✓.
    # Same for site 5: n_5 = 1 (TV), site 4 = 1 (FV), site 6 = 1 (FV). All ✓.
    occ = (1, 1, 1, 0, 1, 1, 1, 0)
    psi = _state_from_occupation(8, occ)
    assert pytest.approx(bubble_correlator_expectation(psi, 1)) == 2.0
    assert pytest.approx(bubble_correlator_expectation(psi, 2)) == 0.0


def test_sigma_L_correctly_handles_ring_wraparound() -> None:
    # N=6, FV=(1,0,1,0,1,0). A length-2 bubble at sites 5,0 wraps the ring:
    # site 5 should be TV (n=1, since FV says 0); site 0 should be TV (n=0, since FV says 1).
    # Bordering sites: site 4 (FV, n=1), site 1 (FV, n=0).
    occ = (0, 0, 1, 0, 1, 1)
    psi = _state_from_occupation(6, occ)
    assert pytest.approx(bubble_correlator_expectation(psi, 2)) == 1.0


def test_sigma_L_invalid_length_raises() -> None:
    with pytest.raises(ValueError):
        sigma_L_diagonal(6, L=0)
    with pytest.raises(ValueError):
        sigma_L_diagonal(6, L=5)  # L = N - 1 not allowed


# ---------------------------------------------------------------------------
# m_afm_rescaled
# ---------------------------------------------------------------------------


def test_m_afm_rescaled_maps_endpoints() -> None:
    # An ideal trace: M(0) = 1, M(t) = -1 → M^res = (−1+1)/2 = 0
    traces = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    out = m_afm_rescaled(traces)
    assert pytest.approx(out[0]) == 1.0
    assert pytest.approx(out[-1]) == 0.0
    assert pytest.approx(out[2]) == 0.5  # mixed → 0.5


def test_m_afm_rescaled_zero_initial_raises() -> None:
    traces = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError):
        m_afm_rescaled(traces)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_site_occupations_recovers_neel_from_bitstring() -> None:
    N = 6
    bits = neel_bitstring(N, phase=0)
    occ = site_occupations(N)[:, bits]
    expected = neel_occupation(N, phase=0)
    np.testing.assert_array_equal(occ, expected)


def test_perturbed_neel_normalised() -> None:
    rng = np.random.default_rng(0)
    psi = perturbed_neel_state(6, fidelity=0.9, rng=rng)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-12
    overlap_sq = abs(np.vdot(neel_state(6, phase=0), psi)) ** 2
    assert pytest.approx(overlap_sq, abs=1e-12) == 0.9
