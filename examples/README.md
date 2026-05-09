# Examples

Standalone, runnable scripts that exercise the public API. Each is
≤80 lines and runs from the repository root:

```bash
python examples/01_quickstart.py
```

Each example also honours the env var `RYDBERG_TRAMPOLINE_TEST_MODE=1`
to clamp grids for fast smoke-testing under `pytest`.

| File | Audience | What it shows |
|------|----------|---------------|
| [`01_quickstart.py`](01_quickstart.py) | mixed | `ModelParams` → `run_unitary` from the false-vacuum Néel; print `M_AFM(t)` and a single-exponential Γ fit. Smallest end-to-end example. |
| [`02_cross_backend.py`](02_cross_backend.py) | API | Run the same evolution through every locally-importable backend (`numpy`, `qutip`, `quspin`) and report `max\|ΔM_AFM\|`. Companion to `tests/test_cross_backend.py`. |
| [`03_finite_size_scaling.py`](03_finite_size_scaling.py) | physics | Sweep `Δ_l` at two ring sizes; extract `Γ(Δ_l, N)` and fit `B(N)` from `log Γ` vs `1/Δ_l`. Distilled from `figures/fig_gamma_N_dependence.py`. |
| [`04_bloqade_emulator.py`](04_bloqade_emulator.py) | API | bloqade in-process emulator under both `psi0_protocol='ground'` and the new `psi0_protocol='neel_via_ramp'`. Skips cleanly if bloqade is not installed. |

## Prerequisites

| Example | Required extra |
|---------|----------------|
| `01_quickstart.py` | none — base install only |
| `02_cross_backend.py` | base install; `[qutip]` and/or `[quspin]` enable additional comparison backends. The script skips missing backends cleanly. |
| `03_finite_size_scaling.py` | base install for `N ≤ 12`; `[quspin]` for `N ∈ [13, 22]` (sector dispatch via `pick_unitary_backend`). |
| `04_bloqade_emulator.py` | `[cloud]` for the bloqade emulator. The script no-ops if missing. |

Install with e.g.

```bash
pip install -e '.[all,dev]'
```

## See also

- The hero figures under [`rydberg_trampoline/figures/`](../rydberg_trampoline/figures/)
  are also runnable as `python -m rydberg_trampoline.figures.fig_<name>`
  and are what the production paper-figure pipeline uses.
- [`docs/background.md`](../docs/background.md) — physics walkthrough.
- [`docs/architecture.md`](../docs/architecture.md) — module map and
  dispatcher contract.
- [`docs/cloud_quickstart.md`](../docs/cloud_quickstart.md) —
  bloqade Aquila on AWS Braket.
