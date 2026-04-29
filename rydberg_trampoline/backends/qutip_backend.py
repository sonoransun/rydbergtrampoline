"""QuTiP backend (qutip >= 5.0).

Provides:

* :func:`to_qutip` — emit the Hamiltonian as a ``qutip.Qobj``.
* :func:`run_unitary` — ``qutip.sesolve`` driver.
* :func:`run_lindblad` — ``qutip.mesolve`` for small N, ``qutip.mcsolve``
  (Monte Carlo trajectories) for larger N. The cutover is N=10.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from rydberg_trampoline.dynamics import DynamicsResult
from rydberg_trampoline.model import ModelParams
from rydberg_trampoline.observables import (
    bubble_correlator_expectation,
    m_afm_diagonal,
    sigma_L_diagonal,
)


# QuTiP's `tensor()` orders site 0 *first* (most-significant), so to match the
# project-wide convention (site 0 in the least-significant position) we build
# operators with reversed site ordering and document it once here.
def _site_op(N: int, j: int, op: "qutip.Qobj") -> "qutip.Qobj":  # noqa: F821
    import qutip as qt
    factors = [qt.qeye(2)] * N
    factors[N - 1 - j] = op  # reverse so QuTiP's left-most factor is site N-1
    return qt.tensor(*factors)


def to_qutip(params: ModelParams) -> "qutip.Qobj":  # noqa: F821
    import qutip as qt
    sx = qt.sigmax()
    n = qt.num(2)
    H = 0
    for j in range(params.N):
        H = H + (params.Omega / 2.0) * _site_op(params.N, j, sx)
        H = H + params.site_field(j) * _site_op(params.N, j, n)
    for i, j, v in params.coupling_pairs():
        ni = _site_op(params.N, i, n)
        nj = _site_op(params.N, j, n)
        H = H + v * ni * nj
    return H


def _state_qutip_from_numpy(N: int, psi: np.ndarray) -> "qutip.Qobj":  # noqa: F821
    """Convert a numpy state-vector (project convention) to a QuTiP ket.

    Project convention places site 0 in the least-significant bit. QuTiP's
    tensor product places its first factor in the most-significant slot. With
    :func:`_site_op` reversing the factor list, the *integer index* of a basis
    state is the same in both conventions, so a flat copy suffices.
    """
    import qutip as qt
    return qt.Qobj(psi.reshape(-1, 1), dims=[[2] * N, [1] * N])


def _make_collapse_ops(params: ModelParams) -> list["qutip.Qobj"]:  # noqa: F821
    import qutip as qt
    ops: list[qt.Qobj] = []
    if params.T1 is not None:
        rate = 1.0 / params.T1
        sm = qt.sigmam()  # |0><1|
        for j in range(params.N):
            ops.append(np.sqrt(rate) * _site_op(params.N, j, sm))
    if params.T2_star is not None:
        gamma_phi = 1.0 / params.T2_star
        if params.T1 is not None:
            gamma_phi = max(0.0, gamma_phi - 0.5 / params.T1)
        if gamma_phi > 0:
            sz = qt.sigmaz()
            for j in range(params.N):
                ops.append(np.sqrt(gamma_phi / 2.0) * _site_op(params.N, j, sz))
    return ops


def _diag_to_qobj(N: int, diag: np.ndarray) -> "qutip.Qobj":  # noqa: F821
    import qutip as qt
    return qt.Qobj(np.diag(diag.astype(np.float64)), dims=[[2] * N, [2] * N])


def run_unitary(
    params: ModelParams,
    psi0: np.ndarray,
    times: np.ndarray,
    *,
    bubble_lengths: Iterable[int] | None = None,
) -> DynamicsResult:
    import qutip as qt
    H = to_qutip(params)
    psi = _state_qutip_from_numpy(params.N, psi0)
    e_ops = [_diag_to_qobj(params.N, m_afm_diagonal(params.N))]
    bubble_indices: list[int] = []
    if bubble_lengths is not None:
        for L in bubble_lengths:
            e_ops.append(_diag_to_qobj(params.N, sigma_L_diagonal(params.N, int(L))))
            bubble_indices.append(int(L))
    res = qt.sesolve(H, psi, times, e_ops=e_ops)
    expect = np.asarray(res.expect)
    bubbles = None
    if bubble_indices:
        bubbles = {L: np.asarray(expect[1 + k]) for k, L in enumerate(bubble_indices)}
    return DynamicsResult(
        times=np.asarray(times),
        m_afm=np.asarray(expect[0]),
        bubble_densities=bubbles,
        backend="qutip-sesolve",
    )


def run_lindblad(
    params: ModelParams,
    psi0: np.ndarray,
    times: np.ndarray,
    *,
    method: str = "auto",
    n_traj: int = 500,
    bubble_lengths: Iterable[int] | None = None,
    seed: int | None = None,
) -> DynamicsResult:
    import qutip as qt

    if method == "auto":
        method = "mesolve" if params.N <= 10 else "mcsolve"
    if method not in ("mesolve", "mcsolve"):
        raise ValueError(f"method must be 'auto', 'mesolve', or 'mcsolve'; got {method!r}")
    if method == "mesolve" and params.N > 10:
        raise ValueError(
            f"mesolve dense Lindblad capped at N=10 (got N={params.N}); "
            "use method='mcsolve' for trajectories."
        )

    H = to_qutip(params)
    psi = _state_qutip_from_numpy(params.N, psi0)
    c_ops = _make_collapse_ops(params)
    e_ops = [_diag_to_qobj(params.N, m_afm_diagonal(params.N))]
    bubble_indices: list[int] = []
    if bubble_lengths is not None:
        for L in bubble_lengths:
            e_ops.append(_diag_to_qobj(params.N, sigma_L_diagonal(params.N, int(L))))
            bubble_indices.append(int(L))

    if method == "mesolve":
        res = qt.mesolve(H, psi, times, c_ops=c_ops, e_ops=e_ops)
        expect = np.asarray(res.expect)
        notes = "qutip mesolve (dense ρ)"
    else:
        options = {"map": "serial"}
        res = qt.mcsolve(
            H,
            psi,
            times,
            c_ops=c_ops,
            e_ops=e_ops,
            ntraj=n_traj,
            seeds=seed,
            options=options,
        )
        expect = np.asarray(res.expect)
        notes = f"qutip mcsolve (n_traj={n_traj})"

    bubbles = None
    if bubble_indices:
        bubbles = {L: np.asarray(expect[1 + k]) for k, L in enumerate(bubble_indices)}

    return DynamicsResult(
        times=np.asarray(times),
        m_afm=np.asarray(expect[0]),
        bubble_densities=bubbles,
        backend=f"qutip-{method}",
        notes=notes,
    )
