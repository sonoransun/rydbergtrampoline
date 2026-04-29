"""QuSpin backend (quspin >= 0.3.7).

The staggered field breaks single-site translation, so we *cannot* use
QuSpin's ``kblock`` directly on the full Hilbert space. The natural
symmetries that survive are:

* **Translation by 2 sites** (the unit cell of the staggered field).
* **Bond-centred parity / inversion** of the ring.

These can be handled with :class:`quspin.basis.spin_basis_general` plus
explicit permutations. For a clean v1, this backend builds the *full*
Hilbert space and uses ``quspin``'s sparse expmkrylov evolution; the
symmetry-resolved variant is exposed as a TODO and is the upgrade path
for reaching N=24.

Operator/coefficient bookkeeping (this is the part that *will* bite you):

* QuSpin's spin-1/2 operator strings ``'x', 'y', 'z', 'xx', 'zz', …`` mean
  ``S^a = σ^a / 2`` — i.e. matrix elements ±1/2 on the diagonal of ``z``.
* The Hamiltonian as written by the paper uses ``σ^a`` (Pauli) and ``n``.
* We convert as follows. Substitute ``n_j = (I + σ^z_j) / 2``::

      h_j n_j           = (h_j / 2)·I + (h_j / 2)·σ^z_j
      V_ij n_i n_j      = V_ij/4·I + V_ij/4·σ^z_i + V_ij/4·σ^z_j
                          + V_ij/4·σ^z_i σ^z_j

  Then convert ``σ^a → 2 S^a`` for each remaining factor, which gives a
  ``×2`` on the linear term and a ``×4`` on the ``σ^z σ^z`` term — exactly
  cancelling the 1/2 and 1/4 from the n-expansion. The drive
  ``(Ω/2) σ^x_j = Ω · S^x_j`` likewise gains a ``×2``. Constants from the
  expansion are dropped (they only shift the global phase).

So the QuSpin-coordinate couplings are::

      "x":  coefficient  Ω         on each site  j
      "z":  coefficient  h_j  +  Σ_{i ≠ j} V_ij  on each site j
      "zz": coefficient  V_ij                    on each pair (i, j)
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from rydberg_trampoline.dynamics import DynamicsResult
from rydberg_trampoline.model import ModelParams
from rydberg_trampoline.observables import (
    m_afm_diagonal,
    sigma_L_diagonal,
)


def _quspin_static_lists(params: ModelParams):
    """Return a QuSpin ``static_list`` for the Hamiltonian.

    See module docstring for the σ^z expansion of n_j and n_i n_j.
    Site indexing in QuSpin is left-to-right 0..N-1, matching the project
    convention; bit ordering inside the basis state is consistent with
    QuSpin's internal scheme — the cross-backend regression verifies that.
    """
    # QuSpin's "x" and "z" are Pauli (±1 eigenvalues), so the conversion is
    # exactly the literal Pauli expansion of n = (I + σ^z) / 2:
    #   (Ω/2)  σ^x_j         on each site
    #   (h_j/2) σ^z_j        on each site
    #   (V_ij/4) σ^z_i σ^z_j on each pair
    # Constants from the n-expansion only shift the global phase and are dropped.
    static_list = []

    sx_list = [[params.Omega / 2.0, j] for j in range(params.N)]
    if sx_list:
        static_list.append(["x", sx_list])

    sz_coeffs = np.zeros(params.N, dtype=np.float64)
    for j in range(params.N):
        sz_coeffs[j] += params.site_field(j) / 2.0
    pair_list = []
    for i, j, v in params.coupling_pairs():
        pair_list.append([v / 4.0, i, j])
        sz_coeffs[i] += v / 4.0
        sz_coeffs[j] += v / 4.0
    sz_list = [[float(c), j] for j, c in enumerate(sz_coeffs) if c != 0.0]
    if sz_list:
        static_list.append(["z", sz_list])
    if pair_list:
        static_list.append(["zz", pair_list])

    return static_list


def to_quspin(params: ModelParams, *, kblock: int | None = None, pblock: int | None = None):
    """Build a QuSpin Hamiltonian, optionally restricted to a symmetry sector.

    The staggered Hamiltonian breaks single-site translation but preserves
    **translation-by-2** (the unit cell of the staggered field) and
    **bond-centred inversion** (``j → -j mod N``). Either of those can be
    used to block-diagonalise the Hamiltonian:

    * ``kblock`` selects a momentum sector under translation-by-2. Since
      that translation has order ``N // 2`` on the ring, ``kblock`` may
      take values ``0, 1, …, (N // 2) - 1``. ``kblock=0`` is the
      trivially-symmetric sector. Use :func:`translation_block_count` to
      get the valid range.
    * ``pblock=±1`` selects the inversion-symmetric / antisymmetric sector.

    Both kept ``None`` returns the full 2^N Hilbert space (the default and
    the path used by every cross-backend test).

    The k-blocked construction uses :class:`quspin.basis.spin_basis_general`,
    which carries some bookkeeping cost but pays off in dimension reduction
    for the dynamics. Rule of thumb: use sectors when ``N ≥ 16`` and you
    only need spectral information; below that the simple full-space path
    is faster.

    Verified by ``tests/test_symmetry_sectors.py``: the union of the
    spectra over ``kblock = 0 … N//2 − 1`` reproduces the full-space
    spectrum to 1e-10 on N = 8 and N = 12.
    """
    from quspin.operators import hamiltonian

    static_list = _quspin_static_lists(params)
    if kblock is None and pblock is None:
        from quspin.basis import spin_basis_1d
        basis = spin_basis_1d(L=params.N, S="1/2")
    else:
        from quspin.basis import spin_basis_general

        kwargs = {}
        if kblock is not None:
            shift2 = np.array([(j + 2) % params.N for j in range(params.N)])
            kwargs["T_block"] = (shift2, int(kblock))
        if pblock is not None:
            inv = np.array([(-j) % params.N for j in range(params.N)])
            kwargs["P_block"] = (inv, int(pblock))
        basis = spin_basis_general(N=params.N, S="1/2", pauli=1, **kwargs)
    H = hamiltonian(
        static_list,
        [],
        basis=basis,
        dtype=np.complex128,
        check_symm=False,
        check_herm=False,
        check_pcon=False,
    )
    return H, basis


def translation_block_count(N: int) -> int:
    """Number of momentum sectors under translation-by-2 on N sites.

    Equal to ``N // 2`` because the order of the shift-by-2 permutation on
    a ring of N (even) sites is ``N // 2``.
    """
    if N % 2 != 0:
        raise ValueError(f"N must be even, got {N}")
    return N // 2


def _bit_reverse(x: int, N: int) -> int:
    """Reverse the low ``N`` bits of ``x``."""
    rev = 0
    for _ in range(N):
        rev = (rev << 1) | (x & 1)
        x >>= 1
    return rev


def _project_to_quspin_perm(N: int, basis) -> np.ndarray:
    """Permutation array ``perm`` such that ``psi_qs = psi_project[perm]``.

    Two reorderings compose:

    1. QuSpin uses MSB-first site convention (bitstring leftmost = site 0,
       integer's most significant bit = site 0). The project convention is
       LSB-first. So a project state with integer ``i`` corresponds to the
       QuSpin integer ``bit_reverse(i, N)``.
    2. QuSpin's ``spin_basis_1d`` orders ``basis.states`` in descending
       integer order: position ``k`` in the QuSpin vector holds amplitude
       for QuSpin-integer ``basis.states[k]``.

    Combine: position ``k`` ↔ QuSpin int ``s = basis.states[k]``
    ↔ project int ``bit_reverse(s, N)``.
    """
    N_states = 1 << N
    perm = np.empty(N_states, dtype=np.int64)
    states = np.asarray(basis.states, dtype=np.int64)
    for k in range(N_states):
        perm[k] = _bit_reverse(int(states[k]), N)
    return perm


def _state_quspin_from_numpy(N: int, psi: np.ndarray, basis) -> np.ndarray:
    """Convert a project-convention state vector to QuSpin's basis ordering."""
    perm = _project_to_quspin_perm(N, basis)
    return psi[perm]


def run_unitary(
    params: ModelParams,
    psi0: np.ndarray,
    times: np.ndarray,
    *,
    bubble_lengths: Iterable[int] | None = None,
) -> DynamicsResult:
    H, basis = to_quspin(params)
    times = np.asarray(times, dtype=np.float64)

    psi_q = _state_quspin_from_numpy(params.N, psi0, basis)
    # H.evolve handles a vector of times in one call.
    psi_t = H.evolve(psi_q, times[0], times)
    # psi_t shape: (dim, n_times)
    psi_t = np.asarray(psi_t)
    if psi_t.ndim == 1:
        psi_t = psi_t.reshape(-1, 1)

    # Convert each time slice back to project convention to apply observables.
    diag_m = m_afm_diagonal(params.N)
    bubble_indices: list[int] = []
    diag_b: dict[int, np.ndarray] = {}
    if bubble_lengths is not None:
        for L in bubble_lengths:
            bubble_indices.append(int(L))
            diag_b[int(L)] = sigma_L_diagonal(params.N, int(L))

    # Probabilities indexed by QuSpin's basis position can be re-keyed by
    # project integer through the same forward permutation.
    perm = _project_to_quspin_perm(params.N, basis)

    N_states = 1 << params.N
    m_trace = np.empty(psi_t.shape[1], dtype=np.float64)
    bubbles = {L: np.empty(psi_t.shape[1], dtype=np.float64) for L in bubble_indices}
    for k in range(psi_t.shape[1]):
        prob_qs = np.abs(psi_t[:, k]) ** 2  # indexed by QuSpin position
        prob_proj = np.empty(N_states, dtype=np.float64)
        prob_proj[perm] = prob_qs  # rearrange so prob_proj[i] is amplitude² of project-int i
        m_trace[k] = float(prob_proj @ diag_m)
        for L in bubble_indices:
            bubbles[L][k] = float(prob_proj @ diag_b[L])

    return DynamicsResult(
        times=times,
        m_afm=m_trace,
        bubble_densities=bubbles if bubbles else None,
        backend="quspin",
        notes=f"full Hilbert space, dim={N_states}",
    )
