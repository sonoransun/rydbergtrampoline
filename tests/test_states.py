"""Tests for the state factories in :mod:`rydberg_trampoline.states`."""
from __future__ import annotations

import numpy as np
import pytest

from rydberg_trampoline.conventions import neel_bitstring, neel_occupation
from rydberg_trampoline.observables import m_afm_expectation
from rydberg_trampoline.states import (
    computational_basis_vector,
    equal_superposition_state,
    neel_state,
    perturbed_neel_state,
    single_flip_admixed_neel,
)


# ---------------------------------------------------------------------------
# computational_basis_vector
# ---------------------------------------------------------------------------


def test_computational_basis_vector_unit_norm() -> None:
    psi = computational_basis_vector(6, 17)
    assert pytest.approx(np.linalg.norm(psi)) == 1.0
    # |17⟩ = bit pattern 010001 → only one nonzero entry at index 17.
    assert psi[17] == 1.0
    psi[17] = 0
    assert np.allclose(psi, 0.0)


@pytest.mark.parametrize("bad_index", [-1, 64])
def test_computational_basis_vector_rejects_out_of_range(bad_index: int) -> None:
    with pytest.raises(ValueError):
        computational_basis_vector(6, bad_index)


# ---------------------------------------------------------------------------
# neel_state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [4, 6, 8, 10])
@pytest.mark.parametrize("phase", [0, 1])
def test_neel_state_concentrates_on_correct_index(N: int, phase: int) -> None:
    psi = neel_state(N, phase=phase)
    target = neel_bitstring(N, phase=phase)
    assert pytest.approx(abs(psi[target])) == 1.0
    other_norm_sq = float(np.sum(np.abs(psi) ** 2)) - abs(psi[target]) ** 2
    assert abs(other_norm_sq) < 1e-12


def test_neel_state_phases_are_orthogonal() -> None:
    a = neel_state(8, phase=0)
    b = neel_state(8, phase=1)
    assert abs(np.vdot(a, b)) < 1e-12


# ---------------------------------------------------------------------------
# perturbed_neel_state (Haar-random admixture)
# ---------------------------------------------------------------------------


def test_perturbed_neel_normalised_and_correct_overlap() -> None:
    rng = np.random.default_rng(42)
    psi = perturbed_neel_state(8, fidelity=0.7, rng=rng)
    assert pytest.approx(np.linalg.norm(psi), abs=1e-12) == 1.0
    overlap_sq = abs(np.vdot(neel_state(8, phase=0), psi)) ** 2
    assert pytest.approx(overlap_sq, abs=1e-12) == 0.7


def test_perturbed_neel_full_fidelity_returns_neel() -> None:
    psi_clean = neel_state(6, phase=0)
    psi_pert = perturbed_neel_state(6, fidelity=1.0)
    np.testing.assert_allclose(psi_pert, psi_clean)


def test_perturbed_neel_seed_is_reproducible() -> None:
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    a = perturbed_neel_state(6, fidelity=0.8, rng=rng_a)
    b = perturbed_neel_state(6, fidelity=0.8, rng=rng_b)
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_perturbed_neel_rejects_bad_fidelity(bad: float) -> None:
    with pytest.raises(ValueError):
        perturbed_neel_state(4, fidelity=bad)


# ---------------------------------------------------------------------------
# single_flip_admixed_neel (paper-style preparation noise)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [4, 6, 8])
def test_single_flip_admixed_norm_and_overlap(N: int) -> None:
    psi = single_flip_admixed_neel(N, fidelity=0.9)
    assert pytest.approx(np.linalg.norm(psi), abs=1e-12) == 1.0
    overlap_sq = abs(np.vdot(neel_state(N, phase=0), psi)) ** 2
    assert pytest.approx(overlap_sq, abs=1e-12) == 0.9


def test_single_flip_admixed_lowers_m_afm_below_one() -> None:
    """Single flips reduce the AFM order parameter symmetrically.

    With Néel(phase=0) at M_AFM=+1 and admixed states (single flips) all at
    M_AFM = +1 - 4/N (each flip changes one (-1)^j σ^z_j by 2, divided by N),
    the average is positive but strictly below 1.
    """
    psi = single_flip_admixed_neel(8, fidelity=0.95)
    m = m_afm_expectation(psi, 8)
    assert m < 1.0
    assert m > 0.5  # still mostly Néel


def test_single_flip_admixed_at_unit_fidelity_is_neel() -> None:
    psi = single_flip_admixed_neel(6, fidelity=1.0)
    np.testing.assert_allclose(psi, neel_state(6, phase=0))


def test_single_flip_admixture_lives_at_hamming_distance_one() -> None:
    """The non-Néel weight sits *only* on bitstrings one site away from Néel."""
    N = 6
    psi = single_flip_admixed_neel(N, fidelity=0.75)
    target = neel_bitstring(N, phase=0)
    nonzero = np.where(np.abs(psi) > 1e-12)[0]
    for idx in nonzero:
        if idx == target:
            continue
        # XOR gives the differing bits; popcount must be 1.
        assert bin(int(idx ^ target)).count("1") == 1


# ---------------------------------------------------------------------------
# equal_superposition_state
# ---------------------------------------------------------------------------


def test_equal_superposition_normalised_and_uniform() -> None:
    psi = equal_superposition_state(5)
    assert pytest.approx(np.linalg.norm(psi), abs=1e-12) == 1.0
    assert np.allclose(np.abs(psi), 1.0 / np.sqrt(32))
