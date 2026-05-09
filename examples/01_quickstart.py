"""01_quickstart.py — minimal closed-system unitary run from the Néel false vacuum.

Run from the repository root:

    python examples/01_quickstart.py

Builds the staggered Rydberg Hamiltonian for a small ring, evolves the
false-vacuum Néel under closed-system dynamics, and prints the
``M_AFM(t)`` decay trace. This is the smallest end-to-end example in the
package and uses only the ``[base]`` install (no extras needed).

For a backend-comparison story, see ``02_cross_backend.py``.
For the physics story (B(N) finite-size scaling), see
``03_finite_size_scaling.py``.
"""
from __future__ import annotations

import os

import numpy as np

from rydberg_trampoline import ModelParams, run_unitary
from rydberg_trampoline.analysis import fit_decay_rate


def main() -> int:
    # Test mode: shrink the grid so CI runs in <1 s.
    test_mode = os.environ.get("RYDBERG_TRAMPOLINE_TEST_MODE") == "1"
    N = 8 if test_mode else 10
    n_times = 41 if test_mode else 121

    params = ModelParams(
        N=N,
        Omega=1.8, Delta_g=4.8, Delta_l=2.0, V_NN=6.0,
    )
    times = np.linspace(0.0, 2.0, n_times)
    res = run_unitary(params, times, backend="numpy")

    # The trace decays from M_AFM(0) = 1 (false vacuum) toward 0 (mixed).
    # Fit Γ on the first ~60 % of the window before recurrences kick in.
    fit = fit_decay_rate(times, res.m_afm, t_max=times[-1] * 0.6)

    print(f"backend     : {res.backend}")
    print(f"N           : {params.N}")
    print(f"Δ_l         : {params.Delta_l} MHz")
    print(f"M_AFM(0)    : {res.m_afm[0]:+.4f}")
    print(f"M_AFM(t_max): {res.m_afm[-1]:+.4f}")
    print(f"fit Γ       : {fit.Gamma:.4f} 1/μs")
    print(f"fit success : {fit.success}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
