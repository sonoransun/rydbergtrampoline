"""04_bloqade_emulator.py — bloqade in-process emulator with the Néel-prep ramp.

Run from the repository root:

    python examples/04_bloqade_emulator.py

Demonstrates the bloqade backend's two psi₀ protocols:

* ``psi0_protocol='ground'`` (the historical Aquila contract) starts
  from ``|gg…g⟩`` and produces a shot-noisy M_AFM trace.
* ``psi0_protocol='neel_via_ramp'`` (added with Mode C of the
  finite-size figure pipeline) prepends an adiabatic Bernien-style Z2
  state-prep ramp (see ``docs/figures/neel_prep_ramp.png``) so the
  program lands in the false-vacuum Néel before the staggered
  evolution begins.

Both runs use the in-process emulator — free, no AWS credentials. The
cloud submission path (``device='cloud'``) is gated behind
``i_understand_this_costs_money=True`` and is documented in
``docs/cloud_quickstart.md``.

Skips cleanly if bloqade is not installed.
"""
from __future__ import annotations

import os

import numpy as np

from rydberg_trampoline import ModelParams, run_unitary
from rydberg_trampoline.backends import available_backends
from rydberg_trampoline.states import neel_state


def main() -> int:
    if "bloqade" not in available_backends():
        print("bloqade not installed — install via 'pip install rydberg-trampoline[cloud]'.")
        return 0

    test_mode = os.environ.get("RYDBERG_TRAMPOLINE_TEST_MODE") == "1"
    N = 4
    n_shots = 200 if test_mode else 1000
    times = np.array([0.0, 0.05]) if test_mode else np.linspace(0.0, 0.4, 5)

    params = ModelParams(N=N, Omega=1.8, Delta_g=4.8, Delta_l=0.5, V_NN=6.0)

    print(f"params: N={N}, Δ_l={params.Delta_l}, n_shots={n_shots}, "
          f"len(times)={len(times)}")
    print()

    # 1) Ground-state protocol (historical default).
    res_ground = run_unitary(
        params, times, backend="bloqade",
        n_shots=n_shots, seed=0,
    )
    print("psi0_protocol='ground'  (starts from |gg…g⟩):")
    for t, m in zip(times, res_ground.m_afm):
        print(f"  t={t:5.2f} μs   M_AFM = {m:+.4f}")
    print()

    # 2) Néel-prep ramp protocol (Mode C entry point).
    res_neel = run_unitary(
        params, times, backend="bloqade",
        psi0=neel_state(N, phase=0),
        psi0_protocol="neel_via_ramp",
        n_shots=n_shots, seed=42,
    )
    print("psi0_protocol='neel_via_ramp'  (Bernien-style Z2 prep):")
    for t, m in zip(times, res_neel.m_afm):
        print(f"  t={t:5.2f} μs   M_AFM = {m:+.4f}")
    print()
    print(f"Notes: {res_neel.notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
