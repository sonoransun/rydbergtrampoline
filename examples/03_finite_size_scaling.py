"""03_finite_size_scaling.py — fit B(N) as a function of ring size.

Run from the repository root:

    python examples/03_finite_size_scaling.py

The headline scientific claim of Chao et al. is that
``Γ(Δ_l) ≈ A·exp(−B/Δ_l)``. The action ``B`` is meant to be N-independent
in the thermodynamic limit, but at the finite ring sizes tractable on
ED there are visible finite-size effects. This script sweeps ``Δ_l``
at two ring sizes ``N ∈ {8, 12}`` (or ``{6, 8}`` in test mode), extracts
``Γ(Δ_l, N)`` from a single-exponential fit on each trace, then fits
``B(N)`` from ``log Γ`` vs ``1/Δ_l``. The pipeline composes
:func:`rydberg_trampoline.analysis.fit_decay_rate` with
:func:`fit_tunneling_action`.

For the full Γ-vs-1/Δ_l overlay (and the ``--include-itebd`` and
``--cloud-N`` flags), use the production figure script
``rydberg_trampoline.figures.fig_gamma_N_dependence``.
"""
from __future__ import annotations

import os

import numpy as np

from rydberg_trampoline import ModelParams, run_unitary
from rydberg_trampoline.analysis import fit_decay_rate, fit_tunneling_action
from rydberg_trampoline.figures._common import pick_unitary_backend


def main() -> int:
    test_mode = os.environ.get("RYDBERG_TRAMPOLINE_TEST_MODE") == "1"
    if test_mode:
        N_values = [6, 8]
        deltas = np.array([1.5, 2.0, 2.5, 3.0])
        n_times = 21
    else:
        N_values = [8, 12]
        deltas = np.linspace(0.5, 3.0, 8)
        n_times = 81

    times = np.linspace(0.0, 2.0, n_times)

    print(f"{'N':>3s}  {'backend':<22s}  {'B':>7s}  {'A':>7s}")
    print("-" * 46)
    for N in N_values:
        backend, kwargs = pick_unitary_backend(N)
        gammas = []
        for dl in deltas:
            params = ModelParams(N=N, Delta_l=float(dl))
            res = run_unitary(params, times, backend=backend, **kwargs)
            fit = fit_decay_rate(times, res.m_afm, t_max=times[-1] * 0.6)
            gammas.append(fit.Gamma if fit.success else np.nan)
        gammas = np.asarray(gammas)
        valid = np.isfinite(gammas) & (gammas > 0)
        if valid.sum() >= 2:
            tfit = fit_tunneling_action(deltas[valid], gammas[valid])
            tag = backend + (
                f" (kblock={kwargs['kblock']})" if "kblock" in kwargs else ""
            )
            print(
                f"{N:>3d}  {tag:<22s}  {tfit.B:>7.3f}  {tfit.A:>7.3f}"
            )
        else:
            print(f"{N:>3d}  {backend:<22s}  (fit failed; insufficient valid Γ)")
    print("\nB is the package's tunneling action;  Γ ≈ A·exp(−B / Δ_l).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
