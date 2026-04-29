"""Single source of truth for spin / occupation conventions.

The paper's Hamiltonian uses occupation operators

    n_j = |r><r|_j  ∈ {0, 1}

with σ^z = 2 n − 1 ∈ {−1, +1}. The "ground" basis state has n = 0, σ^z = −1;
the Rydberg state has n = +1, σ^z = +1.

Every backend must agree on:

* Site ordering: ``j ∈ {0, …, N−1}`` arranged left-to-right around the ring.
* Computational basis bit ordering: state |b_{N-1} … b_1 b_0> indexed by
  ``int(b_0 + 2 b_1 + 4 b_2 + … )`` — i.e. site 0 is the *least*-significant bit.
  This matches ``numpy.kron(I, ..., I, A_j, I, ..., I)`` where ``A_j`` sits at
  position j counting from the right.
* Geometry: a ring (periodic boundary conditions) is the physical system.
  A linear chain (open boundary) is supported only as a testing convenience.
* Néel reference: the staggered "false vacuum" used in this project is the
  state with the **even sites occupied** (n_{2k} = 1, n_{2k+1} = 0); its
  spin pattern is +−+−…+−. The "true vacuum" Néel for ``Δ_l > 0`` is the
  opposite phase, with odd sites occupied.

Functions in this module are deliberately tiny so they can be reused
unchanged by every backend. Anything that needs the convention should
import from here.
"""
from __future__ import annotations

from enum import Enum
from typing import Final


SIGMA_Z_GROUND: Final[int] = -1
SIGMA_Z_RYDBERG: Final[int] = +1


class Geometry(str, Enum):
    """Boundary conditions for the 1D lattice."""
    RING = "ring"
    CHAIN = "chain"


def n_to_sigmaz(n: int) -> int:
    """Map occupation ``n ∈ {0, 1}`` to ``σ^z ∈ {-1, +1}``."""
    if n not in (0, 1):
        raise ValueError(f"occupation must be 0 or 1, got {n}")
    return 2 * n - 1


def sigmaz_to_n(s: int) -> int:
    """Map ``σ^z ∈ {-1, +1}`` to occupation ``n ∈ {0, 1}``."""
    if s not in (-1, +1):
        raise ValueError(f"σ^z must be ±1, got {s}")
    return (s + 1) // 2


def neel_bitstring(N: int, phase: int = 0) -> int:
    """Return the integer index of the Néel product state.

    ``phase = 0`` gives the *false vacuum* with even sites occupied:
    n = (1, 0, 1, 0, …). ``phase = 1`` gives the opposite phase
    (the *true vacuum* for Δ_l > 0).

    Site 0 sits in the least-significant bit (see module docstring).
    """
    if phase not in (0, 1):
        raise ValueError("phase must be 0 (false vacuum) or 1 (true vacuum)")
    bits = 0
    for j in range(N):
        if (j + phase) % 2 == 0:
            bits |= 1 << j
    return bits


def neel_occupation(N: int, phase: int = 0) -> tuple[int, ...]:
    """Return the per-site occupation tuple for the Néel state.

    ``phase = 0`` → (1, 0, 1, 0, …) (false vacuum).
    ``phase = 1`` → (0, 1, 0, 1, …) (true vacuum for Δ_l > 0).
    """
    bits = neel_bitstring(N, phase)
    return tuple((bits >> j) & 1 for j in range(N))


def site_distance(i: int, j: int, N: int, geometry: Geometry) -> int:
    """Lattice distance between sites ``i`` and ``j``.

    For a ring this is the shorter way around: ``min(|i-j|, N-|i-j|)``.
    For an open chain it is simply ``|i-j|``.
    """
    d = abs(i - j)
    if geometry is Geometry.RING:
        return min(d, N - d)
    return d
