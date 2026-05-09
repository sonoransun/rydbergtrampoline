"""Rydberg Trampoline: false-vacuum decay and bubble nucleation in Rydberg arrays.

Implements the model and methods of Chao et al., PRL 136, 120407 (2026),
arXiv:2512.04637 — a 1D ring of Rydberg atoms with staggered detuning whose
Néel state is a metastable false vacuum decaying via bubble nucleation.

Quickstart::

    >>> import numpy as np
    >>> from rydberg_trampoline import ModelParams, run_unitary
    >>> params = ModelParams(N=8, Omega=1.8, Delta_g=4.8, Delta_l=2.0, V_NN=6.0)
    >>> times = np.linspace(0.0, 1.0, 11)
    >>> res = run_unitary(params, times)
    >>> res.m_afm[0]
    1.0

See ``examples/01_quickstart.py`` and ``docs/background.md`` for richer
walkthroughs.
"""

from rydberg_trampoline.model import ModelParams
from rydberg_trampoline.dynamics import (
    run_unitary,
    run_unitary_async,
    run_lindblad,
    run_itebd,
)
from rydberg_trampoline.observables import (
    m_afm_expectation,
    bubble_correlator_expectation,
)
from rydberg_trampoline.analysis import fit_decay_rate, fit_tunneling_action

__all__ = [
    "ModelParams",
    "run_unitary",
    "run_unitary_async",
    "run_lindblad",
    "run_itebd",
    "m_afm_expectation",
    "bubble_correlator_expectation",
    "fit_decay_rate",
    "fit_tunneling_action",
]

__version__ = "0.1.0"
