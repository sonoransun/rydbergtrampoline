"""Conventions and observable sanity checks."""
from __future__ import annotations

import numpy as np
import pytest

from rydberg_trampoline.conventions import (
    Geometry,
    n_to_sigmaz,
    neel_bitstring,
    neel_occupation,
    sigmaz_to_n,
    site_distance,
)
from rydberg_trampoline.observables import (
    bubble_correlator_expectation,
    m_afm_expectation,
    sigma_L_diagonal,
)
from rydberg_trampoline.states import neel_state


def test_n_sigmaz_round_trip():
    for n in (0, 1):
        assert sigmaz_to_n(n_to_sigmaz(n)) == n


def test_invalid_n_raises():
    with pytest.raises(ValueError):
        n_to_sigmaz(2)
    with pytest.raises(ValueError):
        sigmaz_to_n(0)


def test_neel_bitstring_phase_zero():
    # phase=0 → even sites occupied: n = (1, 0, 1, 0)
    assert neel_occupation(4, phase=0) == (1, 0, 1, 0)
    # bit i set iff site i occupied → bits 0 and 2 set → integer 5
    assert neel_bitstring(4, phase=0) == 0b0101


def test_neel_bitstring_phase_one():
    # phase=1 → odd sites occupied: n = (0, 1, 0, 1)
    assert neel_occupation(4, phase=1) == (0, 1, 0, 1)
    assert neel_bitstring(4, phase=1) == 0b1010


def test_site_distance_ring_vs_chain():
    assert site_distance(0, 5, 8, Geometry.RING) == 3  # short way around
    assert site_distance(0, 5, 8, Geometry.CHAIN) == 5
    assert site_distance(0, 4, 8, Geometry.RING) == 4  # exactly opposite
    assert site_distance(2, 2, 8, Geometry.RING) == 0


@pytest.mark.parametrize("N", [4, 6, 8])
def test_m_afm_on_neel_states(N: int):
    psi0 = neel_state(N, phase=0)
    psi1 = neel_state(N, phase=1)
    assert pytest.approx(m_afm_expectation(psi0)) == 1.0
    assert pytest.approx(m_afm_expectation(psi1)) == -1.0


def test_bubble_correlator_counts_one_bubble():
    # N=6 ring, false vacuum n=(1,0,1,0,1,0). Flip sites 1,2 → n=(1,1,0,0,1,0).
    # That is one length-2 bubble bordered by FV on both ends.
    N = 6
    target_bits = 0
    target_occupation = (1, 1, 0, 0, 1, 0)
    for j, b in enumerate(target_occupation):
        if b:
            target_bits |= 1 << j
    psi = np.zeros(1 << N, dtype=complex)
    psi[target_bits] = 1.0
    assert pytest.approx(bubble_correlator_expectation(psi, 1)) == 0.0
    assert pytest.approx(bubble_correlator_expectation(psi, 2)) == 1.0
    assert pytest.approx(bubble_correlator_expectation(psi, 3)) == 0.0


def test_sigma_L_diagonal_is_nonnegative():
    diag = sigma_L_diagonal(8, 2)
    assert np.all(diag >= 0)
    # Maximum number of length-2 bubbles on N=8 ring is N//4=2 (alternating).
    assert diag.max() <= 8
