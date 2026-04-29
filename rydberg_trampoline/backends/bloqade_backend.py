"""Bloqade / QuEra Aquila cloud backend (analog Rydberg).

Submits the staggered-detuning Rydberg Hamiltonian to either an in-process
local emulator (default, free, no network) or — opt-in — to QuEra Aquila
on AWS Braket. Both paths run the *same* analog Hamiltonian we use
elsewhere in the package; no Trotterization is performed.

Two contracts that are different from the ED / iTEBD backends:

1. **Initial state.** Aquila and the emulator both start in ``|gg…g⟩``
   (all atoms in the ground state). We refuse to silently start from
   a different ``psi0``; the caller must either pass ``psi0=None`` (or
   the explicit all-ground basis vector) or do their own state-prep
   ramp using bloqade's builder. The ``M_AFM(t)`` returned is therefore
   *not* the same trace as ``run_unitary(..., backend="numpy")`` from
   the Néel state — apples-to-apples comparison requires matching
   initial conditions on both sides.

2. **Shot-statistical observable.** ``M_AFM(t)`` is an empirical
   estimate from ``n_shots`` projective measurements per timepoint;
   the ``1/sqrt(n_shots)`` shot-noise floor is real. The default of
   1000 shots gives ~3% noise on each timepoint.

Bloqade unit conventions (verified against bloqade-analog 0.33):

* Bloqade's analog Hamiltonian is ``H/ℏ = (Ω/2) Σ σ^x_j − Δ Σ n_j + …``
  with Ω, Δ in **rad/μs**. The rest of this package builds the same
  H/ℏ via ``exp(-i H t)`` with ``Omega``, ``Delta_g``, ``Delta_l`` taken
  directly as the coefficient of ``(1/2) σ^x`` and ``n_j`` respectively
  in the time-evolution generator — so they are *already* in rad/μs.
  We pass them through to bloqade unchanged. (If you want a linear MHz
  number, multiply by ``2π`` before constructing :class:`ModelParams`.)
* Positions are in **μm**. The lattice spacing ``a`` is chosen so that
  the natural Aquila van-der-Waals coupling ``C₆ / a^6`` matches our
  ``V_NN``: ``a = (C₆ / V_NN)^(1/6)``.
* Time durations are in **μs**, matching the rest of this package.

Sign conventions:

* The package-wide Hamiltonian uses ``[-Δ_g + (-1)^j Δ_l] n_j``.
* Bloqade's analog program uses ``-Δ(t) n_j`` (note the sign).
* So bloqade's per-site detuning is ``Δ_bloqade(j) = Δ_g - (-1)^j Δ_l``
  → even sites get ``Δ_g - Δ_l``, odd sites get ``Δ_g + Δ_l``.
"""
from __future__ import annotations

import asyncio
import os
import sys
import warnings
from typing import Iterable, Literal

import numpy as np

from rydberg_trampoline.dynamics import DynamicsResult
from rydberg_trampoline.model import ModelParams


def _ensure_bloqade_ground_state(N: int, psi0: np.ndarray | None) -> None:
    """Refuse to silently start from anything other than ``|gg…g⟩``."""
    if psi0 is None:
        return
    expected = np.zeros(1 << N, dtype=np.complex128)
    expected[0] = 1.0  # |gg…g⟩ is the all-zero bitstring → integer 0
    if not np.allclose(psi0, expected, atol=1e-10):
        raise ValueError(
            "bloqade backend can only start from the all-ground state "
            "|gg...g⟩ (Aquila hardware constraint). Pass psi0=None or the "
            "explicit |gg...g⟩ basis vector. To compare with the Néel-"
            "initial trace from other backends, run them with the same "
            "matching initial state."
        )


def _check_cost_gate(device: str, i_understand: bool) -> None:
    """Guard real-cloud submissions so they cost zero by accident."""
    if device != "cloud":
        return
    if not i_understand:
        raise RuntimeError(
            "device='cloud' submits a paid AWS Braket task to QuEra Aquila "
            "(currently $0.30/task + per-shot surcharge as of 2026). Pass "
            "i_understand_this_costs_money=True to acknowledge the charge."
        )
    # Probe AWS auth without submitting any work.
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "device='cloud' requires the [cloud] extra. Run "
            "`pip install rydberg-trampoline[cloud]`."
        ) from exc
    try:
        client = boto3.Session().client("braket")
        client.search_devices(filters=[{"name": "deviceName", "values": ["Aquila"]}])
    except Exception as exc:
        raise RuntimeError(
            f"AWS Braket auth probe failed before any program was submitted: {exc}. "
            "Configure AWS credentials (e.g. via `aws configure` or environment "
            "variables) and try again. No charges have been incurred."
        ) from exc
    print(
        f"[rydberg_trampoline] cloud submission acknowledged — proceeding to QuEra Aquila",
        file=sys.stderr,
    )


def _lattice_spacing_um(V_NN_MHz: float) -> float:
    """Convert the desired V_NN (MHz) to the Aquila lattice spacing (μm)."""
    from bloqade.analog import RB_C6
    if V_NN_MHz <= 0:
        raise ValueError(f"V_NN must be positive, got {V_NN_MHz}")
    return float((RB_C6 / V_NN_MHz) ** (1.0 / 6.0))


def _m_afm_from_bitstrings(bitstrings: np.ndarray) -> float:
    """Empirical M_AFM = (1/N) Σ_j (-1)^j ⟨σ^z_j⟩ from shot bitstrings.

    ``bitstrings`` is a (n_shots, N) int array of 0 / 1 occupations.
    """
    if bitstrings.size == 0:
        return float("nan")
    bits = np.asarray(bitstrings, dtype=np.int8)
    sigma_z = 2 * bits - 1                                 # ±1
    signs = np.array([(-1) ** j for j in range(bits.shape[1])])
    per_shot_m_afm = (sigma_z * signs).mean(axis=1)        # (n_shots,)
    return float(per_shot_m_afm.mean())


def _build_program(params: ModelParams, T_us: float):
    """Construct a bloqade analog program for evolution of duration ``T_us``."""
    import bloqade.analog as ba

    a = _lattice_spacing_um(params.V_NN)
    positions = [(j * a, 0.0) for j in range(params.N)]
    even_sites = list(range(0, params.N, 2))
    odd_sites = list(range(1, params.N, 2))

    # Pass coefficients through unchanged; project convention matches bloqade's.
    omega_rad = float(params.Omega)
    detuning_uniform_rad = float(params.Delta_g)
    detuning_local_rad = float(params.Delta_l)

    # Aquila has a minimum runtime (~0.05 μs); guard against zero-duration tasks
    # since they're usually a sign of a bug rather than physically meaningful.
    if T_us <= 0:
        raise ValueError(f"duration must be positive, got {T_us}")

    program = (
        ba.start
        .add_position(positions)
        .rydberg.rabi.amplitude.uniform.constant(value=omega_rad, duration=T_us)
        .rydberg.detuning.uniform.constant(value=detuning_uniform_rad, duration=T_us)
    )
    if params.Delta_l != 0.0 and even_sites:
        program = (
            program.rydberg.detuning.location(even_sites)
            .constant(value=-detuning_local_rad, duration=T_us)
        )
    if params.Delta_l != 0.0 and odd_sites:
        program = (
            program.rydberg.detuning.location(odd_sites)
            .constant(value=+detuning_local_rad, duration=T_us)
        )
    return program


async def run_unitary_async(
    params: ModelParams,
    times: np.ndarray,
    *,
    psi0: np.ndarray | None = None,
    n_shots: int = 1000,
    device: Literal["emulator", "cloud"] = "emulator",
    i_understand_this_costs_money: bool = False,
    seed: int | None = None,
    bubble_lengths: Iterable[int] | None = None,  # accepted for API parity; ignored
) -> DynamicsResult:
    """Evolve under the Rydberg Hamiltonian using the bloqade analog backend.

    The local emulator runs in-process, requires no AWS auth, and is free.
    Setting ``device='cloud'`` submits to QuEra Aquila on AWS Braket (paid).

    Parameters
    ----------
    params
        Model specification. The bloqade emitter computes the lattice
        spacing internally so that the natural Aquila vdW reproduces
        ``params.V_NN``.
    times
        1D array of evaluation times in μs. ``times[0]`` should be ≥ 0;
        ``times[0] == 0`` is treated specially (M_AFM(0) on |gg…g⟩ is 0
        for even N, no submission needed).
    psi0
        Must be ``None`` or the explicit ``|gg…g⟩`` basis vector;
        anything else is rejected because Aquila cannot prepare it.
    n_shots
        Shots per timepoint.
    device
        ``'emulator'`` (default) or ``'cloud'``.
    i_understand_this_costs_money
        Required when ``device='cloud'``.
    seed
        Optional seed for the local emulator's RNG.
    bubble_lengths
        Accepted for API parity with the other backends; ignored — bubble
        correlators on shot data require many more samples than the
        ``M_AFM`` expectation and are deferred to a future round.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 1:
        raise ValueError("times must be a 1-D array with at least one entry")
    if np.any(times < 0):
        raise ValueError("times must be non-negative")
    if bubble_lengths is not None:
        warnings.warn(
            "bloqade backend ignores bubble_lengths in v1; M_AFM(t) only.",
            stacklevel=2,
        )

    _ensure_bloqade_ground_state(params.N, psi0)
    _check_cost_gate(device, i_understand_this_costs_money)

    n_times = len(times)
    m_trace = np.empty(n_times, dtype=np.float64)

    # M_AFM(0) on |gg...g⟩ for even N is exactly zero — short-circuit and
    # save a (potentially paid) zero-duration program.
    starts_at_zero = float(times[0]) == 0.0
    if starts_at_zero:
        m_trace[0] = 0.0

    sub_times = times[1:] if starts_at_zero else times

    for idx, t in enumerate(sub_times):
        program = _build_program(params, float(t))
        bitstrings_per_task = await _run_program(
            program,
            n_shots=n_shots,
            device=device,
            seed=None if seed is None else seed + idx,
        )
        m_trace[idx + (1 if starts_at_zero else 0)] = _m_afm_from_bitstrings(
            bitstrings_per_task
        )

    notes = (
        f"bloqade {'emulator (in-process)' if device == 'emulator' else 'QuEra Aquila (cloud)'}, "
        f"n_shots={n_shots} per timepoint"
    )
    return DynamicsResult(
        times=times, m_afm=m_trace, backend=f"bloqade-{device}", notes=notes
    )


async def _run_program(program, *, n_shots: int, device: str, seed: int | None) -> np.ndarray:
    """Execute one bloqade program and return a (n_shots, N) bitstring array."""
    if device == "emulator":
        return await asyncio.to_thread(_run_emulator, program, n_shots, seed)
    if device == "cloud":
        return await asyncio.to_thread(_run_cloud, program, n_shots)
    raise ValueError(f"unknown device {device!r}; expected 'emulator' or 'cloud'")


def _run_emulator(program, n_shots: int, seed: int | None) -> np.ndarray:
    """In-process Schrödinger emulator. Synchronous; called via asyncio.to_thread."""
    runtime = program.bloqade.python()
    kwargs = {"shots": n_shots}
    if seed is not None:
        # Bloqade's run() does not accept a seed directly in 0.33; the emulator
        # uses numpy's global state. Snapshot+restore around the call.
        rng_state = np.random.get_state()
        np.random.seed(int(seed) & 0xFFFFFFFF)
        try:
            batch = runtime.run(**kwargs)
        finally:
            np.random.set_state(rng_state)
    else:
        batch = runtime.run(**kwargs)
    bitstrings = batch.report().bitstrings()
    if not bitstrings:
        raise RuntimeError("bloqade emulator returned no bitstrings")
    arr = np.asarray(bitstrings[0])  # one task per program
    if arr.ndim != 2 or arr.shape[0] != n_shots:
        raise RuntimeError(
            f"unexpected emulator output shape {arr.shape}; expected ({n_shots}, N)"
        )
    return arr.astype(np.int8)


def _run_cloud(program, n_shots: int) -> np.ndarray:
    """QuEra Aquila on AWS Braket. Synchronous; called via asyncio.to_thread."""
    runtime = program.braket.aquila()
    batch = runtime.run(shots=n_shots)
    # Aquila returns post-selected bitstrings (filtered for missing atoms).
    bitstrings = batch.report().bitstrings()
    if not bitstrings:
        raise RuntimeError("QuEra Aquila returned no bitstrings (post-selection?)")
    return np.asarray(bitstrings[0]).astype(np.int8)


def run_unitary(
    params: ModelParams,
    psi0: np.ndarray,
    times: np.ndarray,
    *,
    n_shots: int = 1000,
    device: Literal["emulator", "cloud"] = "emulator",
    i_understand_this_costs_money: bool = False,
    seed: int | None = None,
    bubble_lengths: Iterable[int] | None = None,
) -> DynamicsResult:
    """Synchronous wrapper around :func:`run_unitary_async`.

    Calls ``asyncio.run`` internally; raises a helpful error if invoked from
    inside an already-running event loop (e.g. a Jupyter cell). In that
    setting, ``await run_unitary_async(...)`` directly.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to use asyncio.run.
        return asyncio.run(
            run_unitary_async(
                params,
                times,
                psi0=psi0,
                n_shots=n_shots,
                device=device,
                i_understand_this_costs_money=i_understand_this_costs_money,
                seed=seed,
                bubble_lengths=bubble_lengths,
            )
        )
    raise RuntimeError(
        "run_unitary called from inside a running asyncio loop (e.g. a Jupyter "
        "cell). Use `await run_unitary_async(...)` directly instead."
    )
