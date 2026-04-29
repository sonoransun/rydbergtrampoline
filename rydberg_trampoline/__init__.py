"""Rydberg Trampoline: false-vacuum decay and bubble nucleation in Rydberg arrays.

Implements the model and methods of Chao et al., PRL 136, 120407 (2026),
arXiv:2512.04637 — a 1D ring of Rydberg atoms with staggered detuning whose
Néel state is a metastable false vacuum decaying via bubble nucleation.
"""

from rydberg_trampoline.model import ModelParams
from rydberg_trampoline.dynamics import run_unitary, run_lindblad, run_itebd
from rydberg_trampoline.observables import (
    m_afm_expectation,
    bubble_correlator_expectation,
)
from rydberg_trampoline.analysis import fit_decay_rate, fit_tunneling_action

__all__ = [
    "ModelParams",
    "run_unitary",
    "run_lindblad",
    "run_itebd",
    "m_afm_expectation",
    "bubble_correlator_expectation",
    "fit_decay_rate",
    "fit_tunneling_action",
]

__version__ = "0.1.0"
