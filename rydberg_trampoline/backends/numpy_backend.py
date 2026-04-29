"""Pure NumPy / SciPy backend.

* Hamiltonian: built as a ``scipy.sparse.csr_matrix`` of dimension 2^N.
* Unitary evolution: Krylov-subspace ``scipy.sparse.linalg.expm_multiply``
  applied iteratively along the time grid (constant H).
* Lindblad evolution: vectorised dense Liouvillian + ``solve_ivp``. Hard-capped
  at N ≤ 10 because ρ already costs O(4^N) memory.

The backend exposes its primitives as free functions so that the dispatcher in
:mod:`rydberg_trampoline.dynamics` can compose them.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from rydberg_trampoline.model import ModelParams
from rydberg_trampoline.observables import m_afm_diagonal


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------

# Single-site Pauli matrices in the (|0>, |1>) basis where |1> = Rydberg.
_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_N = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
_I = np.eye(2, dtype=np.complex128)


def _kron_op_at(N: int, j: int, op: np.ndarray) -> sp.csr_matrix:
    """Embed a single-site operator at site ``j`` in the full 2^N space.

    Site 0 is the *least*-significant qubit, matching
    :mod:`rydberg_trampoline.conventions`. Ordering convention:
    ``H = I_{N-1} ⊗ … ⊗ I_{j+1} ⊗ op_j ⊗ I_{j-1} ⊗ … ⊗ I_0``.
    """
    pieces: list[sp.csr_matrix] = []
    for k in range(N - 1, -1, -1):
        pieces.append(sp.csr_matrix(op if k == j else _I))
    out = pieces[0]
    for piece in pieces[1:]:
        out = sp.kron(out, piece, format="csr")
    return out


def _diag_n_at(N: int, j: int) -> sp.csr_matrix:
    """Diagonal n_j operator on the full 2^N space, as a sparse diagonal."""
    idx = np.arange(1 << N)
    diag = ((idx >> j) & 1).astype(np.float64)
    return sp.diags(diag, format="csr", dtype=np.complex128)


def to_scipy(params: ModelParams) -> sp.csr_matrix:
    """Build the Hamiltonian as a sparse matrix.

    The returned matrix is Hermitian (modulo floating-point noise) and uses
    complex128 dtype — numerical operators in :mod:`scipy.sparse.linalg`
    require it.
    """
    dim = 1 << params.N
    H = sp.csr_matrix((dim, dim), dtype=np.complex128)

    # Single-site σ^x drive: (Ω/2) Σ_j σ^x_j
    sx_total = sp.csr_matrix((dim, dim), dtype=np.complex128)
    for j in range(params.N):
        sx_total = sx_total + _kron_op_at(params.N, j, _SX)
    H = H + (params.Omega / 2.0) * sx_total

    # Local detuning: Σ_j [-Δ_g + (-1)^j Δ_l] n_j
    diag_field = np.zeros(dim, dtype=np.float64)
    idx = np.arange(dim)
    for j in range(params.N):
        bits = ((idx >> j) & 1).astype(np.float64)
        diag_field += params.site_field(j) * bits
    H = H + sp.diags(diag_field.astype(np.complex128), format="csr")

    # Pair interactions: Σ_{i<j} V_ij n_i n_j (diagonal in computational basis)
    diag_int = np.zeros(dim, dtype=np.float64)
    for i, j, v in params.coupling_pairs():
        ni = ((idx >> i) & 1).astype(np.float64)
        nj = ((idx >> j) & 1).astype(np.float64)
        diag_int += v * ni * nj
    H = H + sp.diags(diag_int.astype(np.complex128), format="csr")

    return H.tocsr()


# ---------------------------------------------------------------------------
# Unitary evolution
# ---------------------------------------------------------------------------

# Hard caps protect users from silent OOM. Numbers from the plan.
NUMPY_ED_MAX_N = 18
NUMPY_LINDBLAD_MAX_N = 10


def run_unitary(
    params: ModelParams,
    psi0: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Time-evolve ``psi0`` under the Hamiltonian on the given time grid.

    Uses :func:`scipy.sparse.linalg.expm_multiply` segment-by-segment, which
    is Krylov-based and stable for the modestly-sized matrices we generate
    (dim ≤ 2^18). Returns an array of shape ``(len(times), 2^N)`` with the
    state at each grid point. ``times[0]`` need not be zero — the initial
    state is placed at index 0 and propagated forward from there.
    """
    if params.N > NUMPY_ED_MAX_N:
        raise ValueError(
            f"NumPy backend ED hard cap is N={NUMPY_ED_MAX_N}; "
            f"got N={params.N}. Use the QuSpin backend for larger systems."
        )
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 1:
        raise ValueError("times must be a 1D array with at least one entry")

    H = to_scipy(params)
    out = np.empty((len(times), 1 << params.N), dtype=np.complex128)
    out[0] = psi0
    psi = psi0.astype(np.complex128, copy=True)
    for k in range(1, len(times)):
        dt = float(times[k] - times[k - 1])
        # ψ(t+dt) = exp(-i H dt) ψ(t)
        psi = expm_multiply(-1j * dt * H, psi)
        out[k] = psi
    return out


# ---------------------------------------------------------------------------
# Lindblad evolution (small-N reference)
# ---------------------------------------------------------------------------


def _build_collapse_ops(params: ModelParams) -> list[sp.csr_matrix]:
    """Construct the collapse operators implied by T1 and T2*."""
    ops: list[sp.csr_matrix] = []
    if params.T1 is not None:
        rate = 1.0 / params.T1
        sigma_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)  # |0><1|
        for j in range(params.N):
            ops.append(np.sqrt(rate) * _kron_op_at(params.N, j, sigma_minus))
    if params.T2_star is not None:
        # The *pure* dephasing rate is 1/T2* − 1/(2 T1), clamped at 0.
        gamma_phi = 1.0 / params.T2_star
        if params.T1 is not None:
            gamma_phi = max(0.0, gamma_phi - 0.5 / params.T1)
        if gamma_phi > 0:
            sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
            for j in range(params.N):
                ops.append(np.sqrt(gamma_phi / 2.0) * _kron_op_at(params.N, j, sigma_z))
    return ops


def _liouvillian(H: sp.csr_matrix, c_ops: list[sp.csr_matrix]) -> sp.csr_matrix:
    """Build the dense Liouvillian super-operator acting on vec(ρ).

    Convention: vec(ρ) stacks columns, so vec(A ρ B) = (B^T ⊗ A) vec(ρ).
    """
    dim = H.shape[0]
    Id = sp.eye(dim, format="csr", dtype=np.complex128)
    L = -1j * (sp.kron(Id, H, format="csr") - sp.kron(H.T, Id, format="csr"))
    for c in c_ops:
        cdc = c.conj().T @ c
        L = L + sp.kron(c.conj(), c, format="csr")
        L = L - 0.5 * sp.kron(Id, cdc, format="csr")
        L = L - 0.5 * sp.kron(cdc.T, Id, format="csr")
    return L.tocsr()


def run_lindblad(
    params: ModelParams,
    psi0: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Solve the Lindblad equation; return ``M_AFM(t)`` rather than ρ(t).

    Returning the trace observable directly keeps memory tractable and avoids
    a 4^N output for the figures that consume this. Hard-capped at
    N ≤ NUMPY_LINDBLAD_MAX_N because the dense Liouvillian is 4^N × 4^N.
    """
    if params.N > NUMPY_LINDBLAD_MAX_N:
        raise ValueError(
            f"NumPy Lindblad hard cap is N={NUMPY_LINDBLAD_MAX_N}; "
            f"got N={params.N}. Use the QuTiP mcsolve trajectories for larger systems."
        )
    times = np.asarray(times, dtype=np.float64)
    H = to_scipy(params)
    c_ops = _build_collapse_ops(params)
    L = _liouvillian(H, c_ops)

    dim = 1 << params.N
    # Initial density matrix (pure) → vectorise (column-major).
    rho0 = np.outer(psi0, psi0.conj())
    vec = rho0.reshape(dim * dim, order="F")

    diag = m_afm_diagonal(params.N)
    out = np.empty(len(times), dtype=np.float64)
    out[0] = float((np.abs(psi0) ** 2) @ diag)

    for k in range(1, len(times)):
        dt = float(times[k] - times[k - 1])
        vec = expm_multiply(L * dt, vec)
        rho = vec.reshape(dim, dim, order="F")
        # M_AFM is diagonal in the computational basis, so Tr(ρ M) = Σ_n ρ_nn diag_n.
        out[k] = float(np.real(np.diag(rho) @ diag))
    return out
