"""High-level dynamics dispatcher.

Wraps backend-specific evolution routines behind a single ``run_*`` API.
Each function returns either an ``M_AFM(t)`` trace (cheap, the common case)
or, when the user asks for it, a richer record including bubble correlators.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from rydberg_trampoline.backends import BackendName, require_backend
from rydberg_trampoline.model import ModelParams
from rydberg_trampoline.observables import (
    bubble_correlator_expectation,
    m_afm_expectation,
)
from rydberg_trampoline.states import neel_state


@dataclass(slots=True)
class DynamicsResult:
    """Container returned by every ``run_*`` entry point."""

    times: np.ndarray
    """The evaluation grid (μs)."""

    m_afm: np.ndarray
    """⟨M_AFM⟩(t)."""

    bubble_densities: dict[int, np.ndarray] | None = None
    """Optional ⟨Σ_L⟩(t) per requested L."""

    backend: str = ""
    """Name of the backend that produced this trace."""

    notes: str = ""
    """Free-text annotations (e.g. trajectory count, bond dimension)."""


def _resolve_psi0(params: ModelParams, psi0: np.ndarray | None) -> np.ndarray:
    if psi0 is not None:
        if psi0.shape != (1 << params.N,):
            raise ValueError(
                f"psi0 has shape {psi0.shape}; expected ({1 << params.N},)"
            )
        return psi0.astype(np.complex128, copy=False)
    return neel_state(params.N, phase=0)


def run_unitary(
    params: ModelParams,
    times: np.ndarray | Sequence[float],
    *,
    psi0: np.ndarray | None = None,
    backend: BackendName = "numpy",
    bubble_lengths: Iterable[int] | None = None,
    kblock: int | None = None,
    pblock: int | None = None,
    n_shots: int = 1000,
    device: str = "emulator",
    i_understand_this_costs_money: bool = False,
    seed: int | None = None,
) -> DynamicsResult:
    """Closed-system unitary evolution under :class:`ModelParams`.

    Parameters
    ----------
    params
        Model specification. Decoherence times, if set, are ignored here —
        use :func:`run_lindblad` for open-system dynamics.
    times
        1D array of evaluation times in μs.
    psi0
        Initial state vector of length 2^N. Defaults to the false-vacuum
        Néel state.
    backend
        ``"numpy"`` (default), ``"qutip"``, or ``"quspin"``.
    bubble_lengths
        Optional iterable of bubble lengths L for which to additionally
        return ⟨Σ_L⟩(t).
    kblock, pblock
        Symmetry-sector selectors honoured only by the ``quspin`` backend.
        ``kblock`` selects a translation-by-2 momentum sector (``0 ≤ kblock <
        N // 2``); ``pblock`` selects a bond-inversion parity sector (±1).
        The initial state must lie inside the requested sector. The Néel
        false-vacuum sits in (kblock=0, pblock=+1).
    n_shots, device, i_understand_this_costs_money, seed
        Honoured only by the ``bloqade`` backend. ``device='emulator'``
        (default) runs the in-process emulator; ``device='cloud'`` submits
        a paid task to QuEra Aquila on AWS Braket and requires
        ``i_understand_this_costs_money=True``. The bloqade path can only
        start from the all-ground ``|gg…g⟩`` state — see
        :func:`rydberg_trampoline.backends.bloqade_backend.run_unitary_async`.
    """
    require_backend(backend)
    if (kblock is not None or pblock is not None) and backend != "quspin":
        raise ValueError(
            f"kblock/pblock are honoured only by the quspin backend; got backend={backend!r}"
        )
    times = np.asarray(times, dtype=np.float64)
    psi_init = _resolve_psi0(params, psi0)

    if backend == "numpy":
        from rydberg_trampoline.backends.numpy_backend import run_unitary as _run
        states = _run(params, psi_init, times)
        m = np.array([m_afm_expectation(psi, params.N) for psi in states], dtype=np.float64)
        bubbles = None
        if bubble_lengths is not None:
            bubbles = {
                int(L): np.array(
                    [bubble_correlator_expectation(psi, int(L), N=params.N) for psi in states],
                    dtype=np.float64,
                )
                for L in bubble_lengths
            }
        return DynamicsResult(times=times, m_afm=m, bubble_densities=bubbles, backend="numpy")

    if backend == "qutip":
        from rydberg_trampoline.backends.qutip_backend import run_unitary as _run
        return _run(params, psi_init, times, bubble_lengths=bubble_lengths)

    if backend == "quspin":
        from rydberg_trampoline.backends.quspin_backend import run_unitary as _run
        return _run(
            params,
            psi_init,
            times,
            bubble_lengths=bubble_lengths,
            kblock=kblock,
            pblock=pblock,
        )

    if backend == "tenpy":
        raise ValueError("tenpy backend is for run_itebd, not run_unitary on finite N")

    if backend == "bloqade":
        # Bloqade can only start from |gg...g⟩, so we forward psi0 (which may
        # be None — the bloqade backend treats that as the Aquila default)
        # rather than the Néel-defaulted psi_init the other backends use.
        from rydberg_trampoline.backends.bloqade_backend import run_unitary as _run
        return _run(
            params,
            psi0,                       # forward None unchanged; bloqade validates
            times,
            n_shots=n_shots,
            device=device,
            i_understand_this_costs_money=i_understand_this_costs_money,
            seed=seed,
            bubble_lengths=bubble_lengths,
        )

    raise ValueError(f"unknown backend: {backend}")


async def run_unitary_async(
    params: ModelParams,
    times: np.ndarray | Sequence[float],
    *,
    psi0: np.ndarray | None = None,
    n_shots: int = 1000,
    device: str = "emulator",
    i_understand_this_costs_money: bool = False,
    seed: int | None = None,
) -> DynamicsResult:
    """Awaitable counterpart of :func:`run_unitary` for the ``bloqade`` backend.

    Use this from inside an already-running asyncio event loop (e.g. a
    Jupyter cell or a larger async pipeline). It is shaped exactly like
    the synchronous bloqade path but yields control back to the caller's
    loop while shots are being taken.

    Local backends (``numpy``, ``qutip``, ``quspin``) are inherently
    synchronous; if you need them inside an async context just call the
    blocking :func:`run_unitary` from within a thread executor.
    """
    require_backend("bloqade")
    times = np.asarray(times, dtype=np.float64)
    from rydberg_trampoline.backends.bloqade_backend import run_unitary_async as _run
    return await _run(
        params,
        times,
        psi0=psi0,                      # forward as-is; bloqade validates None / |gg...g⟩
        n_shots=n_shots,
        device=device,
        i_understand_this_costs_money=i_understand_this_costs_money,
        seed=seed,
    )


def run_lindblad(
    params: ModelParams,
    times: np.ndarray | Sequence[float],
    *,
    psi0: np.ndarray | None = None,
    backend: BackendName = "qutip",
    method: str = "auto",
    n_traj: int = 500,
    bubble_lengths: Iterable[int] | None = None,
    seed: int | None = None,
) -> DynamicsResult:
    """Open-system Lindblad evolution.

    The ``method`` argument is honoured by the QuTiP backend:

    * ``"auto"`` — ``mesolve`` for N ≤ 10, ``mcsolve`` (trajectories) above.
    * ``"mesolve"`` — dense density matrix; raises if N > 10.
    * ``"mcsolve"`` — Monte Carlo trajectories; ``n_traj`` controls statistics.

    The NumPy backend always uses dense ``mesolve`` (small N reference).
    """
    require_backend(backend)
    if not params.is_open():
        raise ValueError(
            "run_lindblad called on closed-system params (T1 and T2_star both None). "
            "Use run_unitary for closed-system dynamics."
        )
    times = np.asarray(times, dtype=np.float64)
    psi_init = _resolve_psi0(params, psi0)

    if backend == "numpy":
        from rydberg_trampoline.backends.numpy_backend import run_lindblad as _run
        m = _run(params, psi_init, times)
        return DynamicsResult(times=times, m_afm=m, backend="numpy")

    if backend == "qutip":
        from rydberg_trampoline.backends.qutip_backend import run_lindblad as _run
        return _run(
            params,
            psi_init,
            times,
            method=method,
            n_traj=n_traj,
            bubble_lengths=bubble_lengths,
            seed=seed,
        )

    raise ValueError(f"backend {backend!r} does not support Lindblad evolution")


def run_itebd(
    params: ModelParams,
    times: np.ndarray | Sequence[float],
    *,
    chi: int = 100,
    bubble_lengths: Iterable[int] | None = None,
) -> DynamicsResult:
    """Infinite-chain TEBD using TeNPy.

    Returns the M_AFM trace evaluated on the two-site unit cell of the
    iMPS. ``chi`` is the maximum bond dimension (paper uses 100).
    """
    require_backend("tenpy")
    from rydberg_trampoline.backends.tenpy_backend import run_itebd as _run
    return _run(params, times, chi=chi, bubble_lengths=bubble_lengths)
