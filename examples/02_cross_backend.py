"""02_cross_backend.py — verify M_AFM(t) agrees across installed backends.

Run from the repository root:

    python examples/02_cross_backend.py

Runs the same closed-system evolution under every locally-importable
backend (``numpy``, optionally ``qutip``, optionally ``quspin``) and
prints the maximum disagreement on M_AFM(t). NumPy is the reference;
backends not installed are skipped cleanly.

This script is the running-from-the-shell counterpart of the cross-
backend regression test ``tests/test_cross_backend.py`` and a useful
sanity check after an environment change.
"""
from __future__ import annotations

import os

import numpy as np

from rydberg_trampoline import ModelParams, run_unitary
from rydberg_trampoline.backends import available_backends


def main() -> int:
    test_mode = os.environ.get("RYDBERG_TRAMPOLINE_TEST_MODE") == "1"
    N = 6 if test_mode else 8
    n_times = 11 if test_mode else 21

    params = ModelParams(
        N=N,
        Omega=1.8, Delta_g=4.8, Delta_l=2.0, V_NN=6.0,
        # vdW_cutoff defaults to 8; full vdW for the headline ED comparison.
    )
    times = np.linspace(0.0, 1.0, n_times)

    res_ref = run_unitary(params, times, backend="numpy")
    print(f"reference   : numpy   ({res_ref.backend})")
    print(f"  M_AFM(0)  = {res_ref.m_afm[0]:+.6f}")

    # Compare every other importable backend that supports run_unitary.
    candidates = [b for b in available_backends() if b in ("qutip", "quspin")]
    if not candidates:
        print("  (no comparison backends installed; install [qutip] or [quspin])")
        return 0

    for backend in candidates:
        res = run_unitary(params, times, backend=backend)
        diff = float(np.max(np.abs(res.m_afm - res_ref.m_afm)))
        ok = diff < 1.0e-4
        print(
            f"vs {backend:<7s}: max |ΔM_AFM| = {diff:.2e}   "
            f"{'OK' if ok else 'OUTSIDE TOLERANCE'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
