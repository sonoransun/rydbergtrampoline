"""Smoke tests for the finite-N scaling figures and helper.

These exercise the new pipeline end-to-end at small N so they stay fast
in CI: the production runs at N ∈ {16, 18, 20, 22} are operator-driven.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_pick_unitary_backend_small_N_is_numpy() -> None:
    from rydberg_trampoline.figures._common import pick_unitary_backend

    name, kwargs = pick_unitary_backend(8)
    assert name == "numpy"
    assert kwargs == {}


def test_pick_unitary_backend_above_22_raises() -> None:
    from rydberg_trampoline.figures._common import pick_unitary_backend

    with pytest.raises(ValueError, match="closed-system finite-N ED budget"):
        pick_unitary_backend(32)


def test_pick_unitary_backend_midrange_prefers_quspin_or_falls_back() -> None:
    """At N=14 the helper either selects quspin/kblock=0 or falls back to numpy."""
    from rydberg_trampoline.backends import available_backends
    from rydberg_trampoline.figures._common import pick_unitary_backend

    name, kwargs = pick_unitary_backend(14)
    if "quspin" in available_backends():
        assert name == "quspin"
        assert kwargs == {"kblock": 0}
    else:
        assert name == "numpy"


def test_fig_gamma_N_dependence_smoke(tmp_path: Path) -> None:
    """Smoke: tiny N sweep + 2 Δ_l points produces PNG and JSON sidecar."""
    from rydberg_trampoline.figures.fig_gamma_N_dependence import main

    rc = main(
        [
            "--N-values", "6", "8", "10",
            "--delta-l", "1.5", "2.5",
            "--t-max", "1.0",
            "--n-times", "21",
            "--out", str(tmp_path),
        ]
    )
    assert rc == 0
    png = tmp_path / "fig_gamma_N_dependence.png"
    sidecar = tmp_path / "fig_gamma_N_dependence.json"
    assert png.exists()
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())["params"]
    assert sorted(data["N_values"]) == [6, 8, 10]
    # All three N runs should produce numeric Γ entries (same length as Δ_l).
    for N in (6, 8, 10):
        assert len(data["gammas_by_N"][str(N)]) == 2
    # Backend tags should be one of the known dispatch labels.
    for label in data["backend_per_N"].values():
        assert label.startswith("numpy") or label.startswith("quspin")


def test_fig_thermo_N_sweep_smoke(tmp_path: Path) -> None:
    """Smoke: extended fig_thermodynamic_limit takes --N-values and emits a
    sidecar listing the per-N traces and backend dispatch."""
    from rydberg_trampoline.figures.fig_thermodynamic_limit import main

    rc = main(
        [
            "--N-values", "6", "8",
            "--delta-l", "2.0",
            "--t-max", "0.5",
            "--n-times", "11",
            "--out", str(tmp_path),
        ]
    )
    assert rc == 0
    png = tmp_path / "fig_thermodynamic_limit.png"
    sidecar = tmp_path / "fig_thermodynamic_limit.json"
    assert png.exists()
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())["params"]
    assert sorted(data["N_values"]) == [6, 8]
    for N in (6, 8):
        assert len(data["traces_by_N"][str(N)]) == 11


def test_fig_gamma_N_dependence_include_itebd_smoke(tmp_path: Path) -> None:
    """If TeNPy is available, --include-itebd must add the asymptote to the
    sidecar; if not, the script should print a notice and still succeed."""
    from rydberg_trampoline.figures.fig_gamma_N_dependence import main

    rc = main(
        [
            "--N-values", "6", "8",
            "--delta-l", "2.0", "2.5",
            "--t-max", "1.0",
            "--n-times", "21",
            "--include-itebd",
            "--chi", "32",
            "--out", str(tmp_path),
        ]
    )
    assert rc == 0
    sidecar = json.loads(
        (tmp_path / "fig_gamma_N_dependence.json").read_text()
    )["params"]
    # Either iTEBD ran (key non-null) or it was gracefully skipped (key null).
    assert "itebd_gammas" in sidecar
