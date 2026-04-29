"""Model parameters and the canonical Hamiltonian specification.

The Hamiltonian (in units where ℏ = 1, energies in MHz × 2π is folded into Ω/Δ):

    H = (Ω / 2) Σ_j σ^x_j  +  Σ_j [-Δ_g + (-1)^j Δ_l] n_j  +  Σ_{i<j} V_{ij} n_i n_j

with ``V_{ij} = V_NN / d(i, j)^6`` and ``d`` the lattice distance under the
chosen :class:`~rydberg_trampoline.conventions.Geometry`.

This module is *backend-agnostic* — it defines the physics. Each backend
provides its own emitter that consumes a :class:`ModelParams` instance.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

from rydberg_trampoline.conventions import Geometry, site_distance


@dataclass(frozen=True, slots=True)
class ModelParams:
    """Specification of the Rydberg-array Hamiltonian.

    Energies are in angular-frequency units (MHz). The paper uses
    Ω ≈ 1.8, Δ_g ≈ 4.8, V_NN ≈ 6.0 (all in MHz).

    Attributes
    ----------
    N
        Number of atoms on the ring.
    Omega
        Rabi frequency Ω.
    Delta_g
        Global detuning Δ_g (note the minus sign in front in H).
    Delta_l
        Staggered detuning Δ_l. Setting this to zero recovers the homogeneous
        Rydberg-blockade Hamiltonian; nonzero Δ_l explicitly breaks the Z2
        between the two Néel phases and is the symmetry-breaking parameter
        whose 1/Δ_l-suppressed bubble-nucleation rate is the headline
        observable.
    V_NN
        Nearest-neighbour van-der-Waals coupling. Long-range couplings fall
        off as 1/d^6.
    vdW_cutoff
        Truncation distance for the 1/d^6 tail (in units of the lattice
        spacing). ``vdW_cutoff = 1`` keeps only nearest-neighbour terms;
        ``vdW_cutoff = N`` keeps all couplings on the ring.
    geometry
        :class:`~rydberg_trampoline.conventions.Geometry.RING` (paper) or
        :class:`~rydberg_trampoline.conventions.Geometry.CHAIN` (testing).
    T1
        Single-atom relaxation time (μs). Used by Lindblad backends. ``None``
        means closed-system evolution.
    T2_star
        Dephasing time (μs). Used by Lindblad backends. ``None`` means no
        pure dephasing.
    """

    N: int
    Omega: float = 1.8
    Delta_g: float = 4.8
    Delta_l: float = 0.0
    V_NN: float = 6.0
    vdW_cutoff: int = 8
    geometry: Geometry = Geometry.RING
    T1: float | None = None
    T2_star: float | None = None

    def __post_init__(self) -> None:
        if self.N < 2:
            raise ValueError(f"N must be at least 2, got {self.N}")
        if self.N % 2 != 0:
            raise ValueError(
                f"N must be even so the staggered field is consistent on a ring, got N={self.N}"
            )
        if self.vdW_cutoff < 1:
            raise ValueError(f"vdW_cutoff must be >= 1, got {self.vdW_cutoff}")
        if self.T1 is not None and self.T1 <= 0:
            raise ValueError(f"T1 must be positive or None, got {self.T1}")
        if self.T2_star is not None and self.T2_star <= 0:
            raise ValueError(f"T2_star must be positive or None, got {self.T2_star}")

    def with_(self, **changes) -> "ModelParams":
        """Return a copy of this :class:`ModelParams` with the given fields replaced."""
        return replace(self, **changes)

    def is_open(self) -> bool:
        """Whether decoherence channels are active."""
        return self.T1 is not None or self.T2_star is not None

    def site_distance(self, i: int, j: int) -> int:
        """Lattice distance between sites ``i`` and ``j`` under the chosen geometry."""
        return site_distance(i, j, self.N, self.geometry)

    def vdw_coupling(self, i: int, j: int) -> float:
        """Van-der-Waals coupling V_ij used in the Hamiltonian.

        Returns 0 when the lattice distance exceeds :attr:`vdW_cutoff` or
        when ``i == j``.
        """
        if i == j:
            return 0.0
        d = self.site_distance(i, j)
        if d == 0 or d > self.vdW_cutoff:
            return 0.0
        return self.V_NN / d**6

    def coupling_pairs(self) -> Iterable[tuple[int, int, float]]:
        """Iterate over ``(i, j, V_ij)`` for all unordered pairs with V_ij ≠ 0."""
        for i in range(self.N):
            for j in range(i + 1, self.N):
                v = self.vdw_coupling(i, j)
                if v != 0.0:
                    yield i, j, v

    def site_field(self, j: int) -> float:
        """Local detuning coefficient on site ``j`` (the prefactor of n_j)."""
        return -self.Delta_g + ((-1) ** j) * self.Delta_l


# Lightweight benchmark presets used by figure scripts and tests.
PAPER_DEFAULT = ModelParams(
    N=16,
    Omega=1.8,
    Delta_g=4.8,
    Delta_l=0.0,
    V_NN=6.0,
    vdW_cutoff=8,
    geometry=Geometry.RING,
)
"""Closed-system parameters at the operating point of Chao et al. (2026)."""


PAPER_OPEN = PAPER_DEFAULT.with_(T1=28.0, T2_star=3.8)
"""Open-system parameters with the experimental decoherence times."""
