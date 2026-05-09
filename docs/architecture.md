# Architecture

This document is the technical companion to [`README.md`](../README.md).
It describes how the package is organised, what each module is
responsible for, the contract between the dispatcher and individual
backends, and the small set of conventions that every backend must agree
on. If you only want to *use* the package, the README is enough; this
document is for contributors who want to *change* it.

## Module map

```
rydberg_trampoline/
├── conventions.py          single source of truth for n vs σ^z, site
│                           order, ring vs chain
├── model.py                ModelParams (frozen dataclass) — the only
│                           physics specification consumed by every
│                           backend
├── states.py               product-state factories: Néel,
│                           perturbed-Néel, single-flip-admixed,
│                           equal-superposition
├── observables.py          M_AFM and Σ_L diagonals (real arrays of
│                           length 2^N, evaluated as a single dot
│                           product on a state vector or density-
│                           matrix diagonal)
├── dynamics.py             top-level dispatcher: run_unitary,
│                           run_unitary_async, run_lindblad, run_itebd
│                           and the DynamicsResult dataclass
├── analysis.py             curve fits: fit_decay_rate (Γ from a
│                           single trace) and fit_tunneling_action
│                           (B from Γ vs 1/Δ_l)
├── backends/
│   ├── __init__.py         BackendName Literal + available_backends()
│   │                       discovery helper + require_backend() guard
│   ├── numpy_backend.py    scipy.sparse + expm_multiply + dense
│   │                       Liouvillian Lindblad
│   ├── qutip_backend.py    sesolve / mesolve / mcsolve (auto-switch
│   │                       at N=10)
│   ├── quspin_backend.py   full-Hilbert and translation-by-2 sector-
│   │                       resolved Krylov
│   ├── tenpy_backend.py    TEBD on a 2-site iMPS (NN-only vdW)
│   └── bloqade_backend.py  in-process emulator and QuEra Aquila on
│                           AWS Braket (cost-gated)
├── figures/
│   ├── _common.py          common argparse + sidecar-writer helpers
│   └── fig_*.py            one runnable script per hero figure
├── data/
│   ├── loader.py           CSV loader for digitised paper data
│   └── experimental/       digitised CSVs + provenance YAML sidecars
└── cli.py                  thin argparse wrapper for figures
```

## Figure-generation scripts

The package ships two complementary kinds of figure scripts:

* **Hero figures** — `rydberg_trampoline/figures/fig_*.py`. Each runs a
  full simulation (parametrized by CLI flags) and writes a PNG plus a
  JSON sidecar to `docs/figures/`. They are also runnable as
  `python -m rydberg_trampoline.figures.fig_<name>` and double as
  worked examples of the public API.
* **Static schematics** — `scripts/draw_*.py`. One-shot matplotlib
  generators that *don't* depend on the simulation pipeline; they
  visualise concepts (geometry, vdW tail, level scheme, bubbles,
  the iTEBD unit cell, the Néel-prep ramp, the Hilbert-space
  scaling) and write to `docs/figures/`. They run in seconds with
  no extras beyond the base install.

| Script | Concept visualised | Output |
|--------|-------------------|--------|
| [`scripts/draw_level_scheme.py`](../scripts/draw_level_scheme.py) | Two-photon dressed `g↔r` Rabi drive Ω, detuning Δ | `level_scheme.png` |
| [`scripts/draw_ring_geometry.py`](../scripts/draw_ring_geometry.py) | 16-site PBC ring with even/odd staggered colouring | `ring_geometry.png` |
| [`scripts/draw_vij_curve.py`](../scripts/draw_vij_curve.py) | `1/r⁶` pair coupling with `vdW_cutoff` truncations | `vij_curve.png` |
| [`scripts/draw_hilbert_dim.py`](../scripts/draw_hilbert_dim.py) | `2^N`, `4^N`, sector, iTEBD memory cost vs N | `hilbert_dim_vs_N.png` |
| [`scripts/draw_bubble_cartoon.py`](../scripts/draw_bubble_cartoon.py) | Three-row cartoon: FV, single-flip bubble, length-3 bubble | `bubble_cartoon.png` |
| [`scripts/draw_bubble_pedagogy.py`](../scripts/draw_bubble_pedagogy.py) | Length-1, 2, 3, 4 bubbles side-by-side | `bubble_pedagogy.png` |
| [`scripts/draw_itebd_unit_cell.py`](../scripts/draw_itebd_unit_cell.py) | 2-site iMPS unit cell, NN-only vdW, translation-by-2 | `itebd_unit_cell.png` |
| [`scripts/draw_neel_prep_ramp.py`](../scripts/draw_neel_prep_ramp.py) | Bernien-style bloqade Néel-prep ramp profile | `neel_prep_ramp.png` |

When you commit a new schematic PNG, regenerate
`tests/figure_hashes.json` per the one-liner in
[`tests/test_figure_regression.py`](../tests/test_figure_regression.py)
to keep the perceptual-hash regression honest.

## The `run_unitary` call flow

```mermaid
sequenceDiagram
    autonumber
    participant U as user
    participant D as dynamics.run_unitary
    participant R as backends.require_backend
    participant E as backend.to_H_emitter
    participant S as solver
    participant O as observables.m_afm
    U->>D: run_unitary(params, times, backend=...)
    D->>R: require_backend(backend)
    R-->>D: (raises ModuleNotFoundError if extra missing)
    D->>E: emit Hamiltonian for this backend
    E-->>D: H (sparse / Qobj / quspin op / MPO / bloqade program)
    D->>S: evolve (psi0, times) under H
    S-->>D: state(s) at each time
    D->>O: project ⟨M_AFM⟩(t)
    O-->>D: m_afm trace (and bubble correlators if requested)
    D-->>U: DynamicsResult(times, m_afm, ...)
```

*`run_unitary` call flow. Backend imports are checked first, the per-backend Hamiltonian emitter is invoked, the appropriate solver evolves the state, and observables are projected centrally so every backend returns the same `DynamicsResult` shape.*

The bloqade path is the only one that is shot-statistical: each timepoint is
its own program submitted to the emulator (or to QuEra Aquila), and the
"state(s) at each time" arrow above is replaced by a `(n_shots, N)` array
of bitstrings that gets fed to `_m_afm_from_bitstrings` rather than to the
diagonal-observable evaluator.

## From traces to action B (the fit pipeline)

`analysis.py` composes two fits to convert per-Δ_l traces into the
headline tunneling action:

```mermaid
flowchart LR
    classDef trace fill:#fadbd8,stroke:#c0392b,color:#922b21;
    classDef fit fill:#d5f5e3,stroke:#16a085,color:#0e6655;
    classDef out fill:#ecf0f1,stroke:#2c3e50,color:#2c3e50;

    T["DynamicsResult.m_afm(t)<br/>at fixed Δ_l"]:::trace
        --> F1["fit_decay_rate<br/>(single-exp on M^res)"]:::fit
    F1 --> G["DecayFit.Gamma(Δ_l)"]:::out
    G --> F2["fit_tunneling_action<br/>(linreg on log Γ vs 1/Δ_l)"]:::fit
    F2 --> B["TunnelingFit.A, B"]:::out
```

*Both stages live in [`analysis.py`](../rydberg_trampoline/analysis.py): `fit_decay_rate` consumes one trace and returns `Γ`; `fit_tunneling_action` consumes the family `(Δ_l, Γ)` and returns the prefactor `A` and action `B` in `Γ ≈ A·exp(−B/Δ_l)`. The figure scripts `fig_gamma_vs_inv_delta` and `fig_gamma_N_dependence` chain both stages.*

## Type and data-class structure

```mermaid
classDiagram
    class ModelParams {
        +int N
        +float Omega
        +float Delta_g
        +float Delta_l
        +float V_NN
        +int vdW_cutoff
        +Geometry geometry
        +float|None T1
        +float|None T2_star
        +with_(**changes) ModelParams
        +is_open() bool
        +site_distance(i, j) int
        +vdw_coupling(i, j) float
        +coupling_pairs() Iterator
        +site_field(j) float
    }
    class DynamicsResult {
        +ndarray times
        +ndarray m_afm
        +dict|None bubble_densities
        +str backend
        +str notes
    }
    class BackendName {
        <<Literal>>
        numpy
        qutip
        quspin
        tenpy
        bloqade
    }
    class numpy_backend {
        +to_scipy(params) csr_matrix
        +run_unitary(params, psi0, times)
        +run_lindblad(params, psi0, times)
    }
    class qutip_backend {
        +to_qutip(params) Qobj
        +run_unitary(...)
        +run_lindblad(... method=auto/mesolve/mcsolve)
    }
    class quspin_backend {
        +to_quspin(params, kblock?, pblock?)
        +run_unitary(... kblock?, pblock?)
        +translation_block_count(N) int
    }
    class tenpy_backend {
        +run_itebd(params, times, chi)
    }
    class bloqade_backend {
        +run_unitary_async(...) Coroutine
        +run_unitary(...)
    }

    ModelParams --> numpy_backend : consumed by
    ModelParams --> qutip_backend
    ModelParams --> quspin_backend
    ModelParams --> tenpy_backend
    ModelParams --> bloqade_backend
    numpy_backend ..> DynamicsResult : returns
    qutip_backend ..> DynamicsResult
    quspin_backend ..> DynamicsResult
    tenpy_backend ..> DynamicsResult
    bloqade_backend ..> DynamicsResult
```

*`ModelParams` is the single physics-spec input consumed by all five backends; `DynamicsResult` is the lowest-common-denominator output. Adding a backend means extending the `BackendName` Literal in one place.*

`ModelParams` is the *only* physics specification — all five backends
consume it unchanged. `DynamicsResult` is the lowest common denominator
return type: every entry point produces one. The `BackendName` literal
is deliberately tight; adding a backend means editing it in one place
and registering its detector in `backends/__init__.py::available_backends`.

## Hidden coupling: conventions

Every backend has its own basis-ordering convention. To keep the rest of
the package free of this complexity we reconcile each backend's
convention to the project's at the *boundary* of that backend, and the
project convention is the single source of truth:

* **Occupation vs spin.** Internally we use `n_j ∈ {0, 1}`; the conversion
  to `σ^z = 2n − 1 ∈ {±1}` only happens at emit time. `ground` is `n=0`,
  `Rydberg` is `n=1`.
* **Site ↔ bit ordering.** Site 0 is the *least*-significant bit of the
  integer that indexes the computational basis state. `|b_{N-1} … b_1
  b_0⟩` has integer index `Σ_j b_j · 2^j`.
* **Geometry.** A ring (PBC) is the physics; an open chain is supported
  only for testing.
* **Néel phase.** `phase=0` is the false vacuum: even sites occupied,
  `n = (1, 0, 1, 0, …)`. `phase=1` is the true vacuum: odd sites
  occupied.

The reconciliations live in:

| Backend | Where | What it reconciles |
|---|---|---|
| `numpy` | `_kron_op_at` and `_diag_n_at` (numpy_backend.py) | Builds `tensor(I, …, op_j, …, I)` with `op_j` in the *last* slot, so the integer index has site 0 = LSB. |
| `qutip` | `_site_op` (qutip_backend.py) | Reverses QuTiP's left-to-right tensor order with `factors[N-1-j] = op` so QuTiP's leftmost factor is site `N-1`. |
| `quspin` | `_project_to_quspin_perm` (quspin_backend.py) | Composes a bit-reverse (QuSpin uses MSB-first site encoding) with a `basis.states` reorder (QuSpin sorts integers descending in the basis). |
| `tenpy` | `_initial_neel_imps` (tenpy_backend.py) | A 2-site iMPS with `up`/`down` strings; SpinHalfSite's `Sz` returns `σ^z / 2`, so M_AFM = `S^z_0 − S^z_1`. |
| `bloqade` | `_m_afm_from_bitstrings` (bloqade_backend.py) | Aquila's measurement basis is exactly `n ∈ {0, 1}` with site 0 leftmost in the bitstring — matches the project convention directly. |

Touching `conventions.py`, the M_AFM diagonal, or the basis-ordering
helpers above is a cross-cutting change: the cross-backend regression
test (`tests/test_cross_backend.py`) is what proves you got it right.

## Physics on the bloqade backend

The bloqade backend has two additional contracts that the others don't:

1. **Initial state is forced to `|gg…g⟩`.** Aquila prepares atoms in the
   ground state and cannot instantaneously load an arbitrary product
   state. `_ensure_bloqade_ground_state` raises `ValueError` if the
   caller passes any other `psi0`. To compare the bloqade trace against
   another backend, set `psi0 = computational_basis_vector(N, 0)` on the
   reference backend.
2. **Observables are shot-statistical.** `M_AFM(t)` is the mean of `n_shots`
   per-shot AFM values, with `1/√n_shots` shot noise. The
   `bloqade-emulator vs numpy` regression test
   (`tests/test_bloqade_backend.py`) sets `n_shots=4000` so the noise
   floor is well below its tolerance.

The cost gate (`_check_cost_gate`) fires *before* any program is built:
without `i_understand_this_costs_money=True`, `device='cloud'` raises
immediately; with the flag but missing AWS credentials, the auth probe
raises before incurring any charge.

## Where to add a new backend

To add a backend `foo`:

1. **`backends/foo_backend.py`** — implement `run_unitary(params, psi0,
   times, **opts) -> DynamicsResult` (and `run_lindblad` / `run_itebd` if
   relevant). Reconcile basis ordering at this boundary.
2. **`backends/__init__.py`** — extend `BackendName`, add a lazy import
   probe to `available_backends()`, and add an entry to
   `_EXTRA_FOR_BACKEND` so `require_backend` can suggest the right
   `pip install` extra.
3. **`dynamics.py`** — add a `if backend == "foo":` branch in
   `run_unitary` (or other dispatcher functions) that delegates to
   `backends.foo_backend.run_unitary`.
4. **`pyproject.toml`** — add a `[project.optional-dependencies] foo =
   ["foo-sdk>=…"]` extra and pull it into `all`.
5. **`tests/test_cross_backend.py`** — add `foo_required` skip marker
   import and a regression test that pins agreement with the NumPy
   reference at small N.
6. **`README.md`** + **`docs/architecture.md`** — extend the backend
   reference table; document any extra contract (e.g. shot noise,
   sector-only states, async only).

The cross-backend regression test is the gate. If the new backend
doesn't agree with NumPy on N=8 closed-system M_AFM(t) to within its
documented tolerance, something is wrong.

## See also

* [`docs/numerical_methods.md`](numerical_methods.md) — what each
  backend's solver is actually doing under the hood.
* [`docs/background.md`](background.md) — the physics this package
  computes.
* [`docs/cloud_quickstart.md`](cloud_quickstart.md) — running the
  bloqade backend on QuEra Aquila.
* [`examples/`](../examples/) — runnable scripts that exercise the
  public API; in particular
  [`02_cross_backend.py`](../examples/02_cross_backend.py) is the
  shell counterpart of `tests/test_cross_backend.py`, and
  [`04_bloqade_emulator.py`](../examples/04_bloqade_emulator.py)
  demonstrates both `psi0_protocol` settings.
* [`CLAUDE.md`](../CLAUDE.md) — the same architecture summary in the
  format consumed by Claude Code agents.
