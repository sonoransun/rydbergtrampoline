"""Initial-state factories.

The headline initial state is the Néel product, which under the staggered
field plays the role of a metastable false vacuum. We also expose perturbed
variants for the imperfection-sensitivity figure.
"""
from __future__ import annotations

import numpy as np

from rydberg_trampoline.conventions import neel_bitstring


def computational_basis_vector(N: int, index: int) -> np.ndarray:
    """Return the computational-basis state of dimension 2^N at the given index."""
    if not 0 <= index < (1 << N):
        raise ValueError(f"index {index} out of range for N={N}")
    psi = np.zeros(1 << N, dtype=np.complex128)
    psi[index] = 1.0
    return psi


def neel_state(N: int, phase: int = 0) -> np.ndarray:
    """Return the Néel product state in the computational basis.

    See :func:`rydberg_trampoline.conventions.neel_bitstring` for the phase
    convention. ``phase = 0`` is the false vacuum (even sites occupied).
    """
    return computational_basis_vector(N, neel_bitstring(N, phase))


def perturbed_neel_state(
    N: int, phase: int = 0, *, fidelity: float = 0.95, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Return a Néel state mixed with a small uniform admixture of all states.

    ``|ψ⟩ = √fidelity |Néel⟩ + √(1−fidelity) |uniform⟩``

    where ``|uniform⟩`` is a Haar-random unit vector in the (2^N − 1)-dimensional
    subspace orthogonal to the chosen Néel state. The fidelity argument is
    therefore the squared overlap with the perfect Néel.

    Used to model the experimentally imperfect state preparation that the
    paper shows can collapse the exponential decay law (Fig. 5 / SM).
    """
    if not 0.0 < fidelity <= 1.0:
        raise ValueError(f"fidelity must be in (0, 1], got {fidelity}")
    base = neel_state(N, phase)
    if fidelity == 1.0:
        return base
    rng = np.random.default_rng() if rng is None else rng
    raw = rng.standard_normal(1 << N) + 1j * rng.standard_normal(1 << N)
    raw -= np.vdot(base, raw) * base
    raw /= np.linalg.norm(raw)
    return np.sqrt(fidelity) * base + np.sqrt(1.0 - fidelity) * raw


def equal_superposition_state(N: int) -> np.ndarray:
    """``|+>^⊗N`` — useful as a sanity-check initial state."""
    return np.full(1 << N, 1.0 / np.sqrt(1 << N), dtype=np.complex128)


def single_flip_admixed_neel(
    N: int, phase: int = 0, *, fidelity: float = 0.95
) -> np.ndarray:
    """Return a Néel coherently mixed with all single-flip states.

    Models the experimental preparation infidelity from an imperfect Rabi
    pulse: the dominant error is a small leak into states one Hamming
    distance from the target Néel. Concretely::

        |ψ⟩ = √fidelity |Néel⟩ + √(1 − fidelity) (1/√N) Σ_j |Néel ⊕ e_j⟩

    where ``e_j`` denotes flipping the occupation of site ``j``. The
    admixture is normalised by ``1/√N`` because there are ``N`` orthogonal
    single-flip states, giving an overall unit-norm state.

    This is the perturbation the paper's Fig. 5 effectively probes — it
    seeds the resonant bubble-nucleation channels — and it is much more
    physically informative than a Haar-random admixture, which averages
    over all preparation errors uniformly.
    """
    if not 0.0 < fidelity <= 1.0:
        raise ValueError(f"fidelity must be in (0, 1], got {fidelity}")
    base_index = neel_bitstring(N, phase)
    psi = np.zeros(1 << N, dtype=np.complex128)
    psi[base_index] = np.sqrt(fidelity)
    if fidelity < 1.0:
        amp = np.sqrt((1.0 - fidelity) / N)
        for j in range(N):
            psi[base_index ^ (1 << j)] = amp
    return psi
