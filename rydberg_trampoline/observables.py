"""Diagonal observables in the computational basis.

Because every observable in this project is diagonal in the σ^z basis (the
Hamiltonian is *not* diagonal — but every quantity we measure is) we represent
each observable as a length-2^N real array of eigenvalues. Expectation values
are then a single numpy reduction, which keeps things fast and lets us share
the same observable arrays between every backend.

Observables defined here:

``M_AFM``
    Antiferromagnetic order parameter (1/N) Σ_j (-1)^j ⟨σ^z_j⟩. Equals +1 on
    the false-vacuum Néel state, −1 on the true-vacuum Néel.

``Sigma_L``
    Bubble-density operator: number of length-L contiguous "true-vacuum"
    runs that begin at each site, summed over the ring. ``Sigma_1, Sigma_2,
    Sigma_3`` are the three histogram bins reported in the paper.

``M_AFM_rescaled(t)``
    The figure-of-merit M_AFM^res(t) = (M_AFM(t) + M_AFM(0)) / (2 M_AFM(0))
    used for visualising decay traces. Defined as a helper, not as an
    observable.
"""
from __future__ import annotations

import numpy as np

from rydberg_trampoline.conventions import neel_occupation


def _bit(idx: np.ndarray, j: int) -> np.ndarray:
    """Return bit ``j`` of every index in ``idx`` as a 0/1 int array."""
    return ((idx >> j) & 1).astype(np.int8)


def site_occupations(N: int) -> np.ndarray:
    """Return an (N, 2^N) int array of n_j eigenvalues for every basis state."""
    idx = np.arange(1 << N, dtype=np.int64)
    return np.stack([_bit(idx, j) for j in range(N)], axis=0)


def m_afm_diagonal(N: int) -> np.ndarray:
    """Diagonal of the M_AFM observable in the computational basis.

    M_AFM = (1/N) Σ_j (-1)^j σ^z_j with σ^z = 2n − 1.
    """
    n = site_occupations(N).astype(np.float64)  # (N, 2^N)
    sigma_z = 2.0 * n - 1.0
    signs = np.array([(-1) ** j for j in range(N)], dtype=np.float64)
    return signs @ sigma_z / N


def sigma_L_diagonal(N: int, L: int, *, fv_phase: int = 0) -> np.ndarray:
    """Diagonal of the Σ_L bubble-density operator.

    A "bubble of length L" centred at site j is a configuration in which sites
    ``j, j+1, …, j+L-1`` are flipped relative to the false vacuum (i.e. they
    match the *true* vacuum) while sites ``j-1`` and ``j+L`` retain the
    false-vacuum occupation (so this run is genuinely a bubble bounded by
    domain walls, not part of a longer flipped region).

    Operator definition::

        Σ_L = Σ_j P^FV_{j-1} · P^TV_j · P^TV_{j+1} · … · P^TV_{j+L-1} · P^FV_{j+L}

    where ``P^TV_k = |true-vacuum-on-site-k><…|`` and similarly for FV. With
    Néel false vacuum (n = 1, 0, 1, 0, …), site ``k`` is in TV iff
    ``n_k = 1 − ((k + fv_phase) mod 2 == 0)`` etc. — handled below.

    Parameters
    ----------
    N
        Ring size.
    L
        Bubble length to count, ``1 ≤ L ≤ N − 2``.
    fv_phase
        The Néel phase that defines the false vacuum (matches
        :func:`~rydberg_trampoline.conventions.neel_bitstring`'s ``phase``).
    """
    if not 1 <= L <= N - 2:
        raise ValueError(f"L must satisfy 1 <= L <= N-2 for N={N}, got L={L}")

    fv = np.array(neel_occupation(N, phase=fv_phase), dtype=np.int8)
    n = site_occupations(N)  # (N, 2^N) of int8

    fv_match = (n == fv[:, None]).astype(np.int8)        # 1 where site is in FV
    tv_match = (n == (1 - fv[:, None])).astype(np.int8)  # 1 where site is in TV

    out = np.zeros(1 << N, dtype=np.float64)
    # Slide a window of length L+2 around the ring: left-edge FV, L middle TV's, right-edge FV.
    for j in range(N):
        left = (j - 1) % N
        right = (j + L) % N
        block = fv_match[left]
        for k in range(L):
            block = block * tv_match[(j + k) % N]
        block = block * fv_match[right]
        out += block
    return out.astype(np.float64)


def m_afm_expectation(psi_or_diag: np.ndarray, N: int | None = None) -> float | np.ndarray:
    """Expectation value of M_AFM.

    Two calling modes:

    * ``psi_or_diag`` is a state vector of length 2^N (or a (T, 2^N) batch
      of vectors): we use the cached diagonal and contract.
    * ``psi_or_diag`` is a density-matrix diagonal of length 2^N: we just dot.

    ``N`` is required only when it cannot be inferred from the array length.
    """
    arr = np.asarray(psi_or_diag)
    if N is None:
        N = int(np.log2(arr.shape[-1]))
    diag = m_afm_diagonal(N)
    if np.iscomplexobj(arr):
        # treat as state vector(s)
        prob = np.abs(arr) ** 2
    else:
        # treat as already-real diagonal of ρ
        prob = arr
    return prob @ diag


def bubble_correlator_expectation(
    psi: np.ndarray, L: int, *, fv_phase: int = 0, N: int | None = None
) -> float | np.ndarray:
    """Expectation value of the bubble-density operator Σ_L."""
    arr = np.asarray(psi)
    if N is None:
        N = int(np.log2(arr.shape[-1]))
    diag = sigma_L_diagonal(N, L, fv_phase=fv_phase)
    if np.iscomplexobj(arr):
        prob = np.abs(arr) ** 2
    else:
        prob = arr
    return prob @ diag


def m_afm_rescaled(traces: np.ndarray, *, m0: float | None = None) -> np.ndarray:
    """Compute the rescaled figure-of-merit M_AFM^res(t).

    M_AFM^res(t) = (M_AFM(t) + M_AFM(0)) / (2 M_AFM(0))

    Maps a perfect Néel (M=+1) onto +1 and the opposite Néel (M=−1) onto 0,
    so a decaying trace falls from 1 toward 1/2 (mixed) and onward.
    """
    traces = np.asarray(traces, dtype=np.float64)
    if m0 is None:
        m0 = float(traces.flat[0])
    if abs(m0) < 1e-15:
        raise ValueError("M_AFM(0) is essentially zero; cannot rescale")
    return (traces + m0) / (2.0 * m0)
