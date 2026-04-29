# Rydberg Trampoline

> Resonant bubble nucleation and false-vacuum decay in arbitrary Rydberg atom arrays.

`rydberg_trampoline` is a Python package that reproduces the model and methods of

> Y.-X. Chao, P. Ge, Z.-X. Hua, C. Jia, X. Wang, X. Liang, Z. Yue, R. Lu, M. K. Tey, L. You,
> **Probing False Vacuum Decay and Bubble Nucleation in a Rydberg Atom Array**,
> *Phys. Rev. Lett.* **136**, 120407 (2026); preprint
> [arXiv:2512.04637](https://arxiv.org/abs/2512.04637).

It builds the same staggered-detuning Rydberg ring, time-evolves the Néel
"false vacuum" through three independent solvers (pure NumPy/SciPy, QuTiP,
QuSpin), and adds an iTEBD path for the thermodynamic limit. The hero
figures from the paper are regenerable from runnable scripts and committed
to `docs/figures/`.

---

## Why this is interesting

Coleman's theory of *false-vacuum decay* says that a metastable phase of
the universe can tunnel to a more-stable phase by nucleating a bubble of
true vacuum that then expands at the speed of light. The same equations
govern symmetry-breaking phase transitions in cosmology, condensed
matter, and quantum field theory — but the rates involved are
astronomical, so direct experimental tests are scarce.

Rydberg atom arrays are nearly ideal **analog quantum simulators** of
this physics: the atoms are individually controllable, the interactions
are well-characterised van-der-Waals couplings, and a staggered detuning
turns the alternating-occupation Néel state into a metastable false
vacuum. Chao et al. (2026) used this platform to (a) confirm the
QFT-style suppression law

```math
\Gamma(\Delta_l) \propto \exp\!\bigl(-B/\Delta_l\bigr)
```

over four-plus orders of magnitude, and (b) discover **resonant**
nucleation — discrete-spectrum channels at specific detunings that
exponentially enhance bubble production beyond the continuum prediction.
This package reproduces the closed-system simulation backbone of that
study and packages the analyses for reuse.

---

## The model

Each atom is treated as an effective two-level system
(`|g⟩`, `|r⟩`) driven on the two-photon transition. The full
many-body Hamiltonian is

```math
\hat H/\hbar
= \frac{\Omega}{2}\sum_{j=1}^{N} \hat\sigma^{x}_{j}
  + \sum_{j=1}^{N}\Bigl[-\Delta_g + (-1)^{j}\Delta_l\Bigr]\hat n_j
  + \sum_{i<j} V_{ij}\,\hat n_i\hat n_j,\qquad V_{ij}\propto \frac{1}{|i-j|^6}.
```

| Symbol | Meaning | Paper value |
|--------|---------|-------------|
| `Ω` | Rabi frequency on the dressed ground–Rydberg transition | 1.8 MHz |
| `Δ_g` | Global detuning | 4.8 MHz |
| `Δ_l` | Staggered detuning (the symmetry-breaking parameter) | 0.4–3.0 MHz scan |
| `V_NN` | Nearest-neighbour van-der-Waals coupling | 6 MHz |
| `T₁ / T₂*` | Atomic relaxation / dephasing | 28 / 3.8 μs |

![Two-level Rydberg scheme](docs/figures/level_scheme.png)

---

## Vacuum mapping

The staggered detuning makes the two Néel configurations
(*even-sites occupied* vs *odd-sites occupied*) energetically inequivalent.
The shallow Néel phase plays the role of a metastable "false vacuum",
the deeper one is the "true vacuum", and small contiguous flipped runs
("bubbles") mediate the transition.

```mermaid
stateDiagram-v2
    direction LR
    FV: False vacuum<br/>Néel n=(1,0,1,0,...)
    B1: Length-1 bubble<br/>seed nucleation
    B2: Length-2 bubble<br/>resonant channel
    TV: True vacuum<br/>opposite Néel
    FV --> B1: tunnel under barrier
    B1 --> B2: bubble grows
    B2 --> TV: domain wall propagation
    FV --> TV: direct (suppressed)
```

The `M_AFM` order parameter

```math
M_{\mathrm{AFM}} = \frac{1}{N}\sum_{j} (-1)^{j}\,\langle\hat\sigma^z_j\rangle
```

equals `+1` on the false vacuum, `-1` on the true vacuum, and decays
through `0` (mixed) as the system tunnels.

---

## Lattice and computation diagrams

A 16-site ring with the staggered detuning highlighted by alternate
colours:

```mermaid
graph LR
    classDef even fill:#3498db,stroke:#2c3e50,color:#fff;
    classDef odd fill:#e67e22,stroke:#2c3e50,color:#fff;
    s0((0)):::even --- s1((1)):::odd --- s2((2)):::even --- s3((3)):::odd
    s3 --- s4((4)):::even --- s5((5)):::odd --- s6((6)):::even --- s7((7)):::odd
    s7 --- s8((8)):::even --- s9((9)):::odd --- s10((10)):::even --- s11((11)):::odd
    s11 --- s12((12)):::even --- s13((13)):::odd --- s14((14)):::even --- s15((15)):::odd
    s15 -. periodic .- s0
```

End-to-end pipeline from `ModelParams` to a hero figure:

```mermaid
flowchart LR
    P[ModelParams<br/>N, Ω, Δ_g, Δ_l, V_NN, T1, T2*] --> S{Backend?}
    S -->|numpy| H1[scipy.sparse H]
    S -->|qutip| H2[qutip.Qobj H]
    S -->|quspin| H3[QuSpin hamiltonian]
    S -->|tenpy| H4[2-site iMPO + iMPS]
    H1 --> E[Krylov exp_multiply]
    H2 --> M[mesolve / mcsolve]
    H3 --> E
    H4 --> T[TEBDEngine]
    E --> O[M_AFM, Σ_L]
    M --> O
    T --> O
    O --> A[fit_decay_rate → Γ]
    A --> F[fig_*.py → PNG]
```

Backend selection at a glance:

```mermaid
flowchart TD
    classDef box fill:#ecf0f1,stroke:#2c3e50,color:#2c3e50;
    A[numpy<br/>ED ≤ N=18, Lindblad ≤ N=10]:::box
    B[qutip<br/>sesolve, mesolve, mcsolve<br/>auto-switch at N=10]:::box
    C[quspin<br/>full Hilbert space ≤ N=22 with symmetries]:::box
    D[tenpy iTEBD<br/>thermodynamic limit, χ=100, NN-only]:::box
    A -. cross-check .-> B
    A -. cross-check .-> C
    A -. cross-check .-> D
```

---

## Hero figures

Each panel is regenerated by a runnable Python module under
`rydberg_trampoline/figures/`. PNGs in `docs/figures/` were rendered with
the parameters in the corresponding `.json` sidecars.

### M_AFM(t) decay traces (paper Fig. 2)

![Decay traces](docs/figures/fig_decay_traces.png)

Rescaled antiferromagnetic order under Lindblad evolution with the
experimental T₁ and T₂* decoherence times. Larger `Δ_l` makes the
metastable Néel decay faster, but the rate is not monotone in `Δ_l`
because of the underlying resonance structure.

### Γ vs 1/Δ_l (paper Fig. 3)

![Gamma vs 1/Delta_l](docs/figures/fig_gamma_vs_inv_delta.png)

Closed-system unitary decay rate from the same false-vacuum Néel,
plotted against `1/Δ_l` on a log-y scale. The red line is a global
`Γ = A · exp(−B/Δ_l)` fit; deviations (humps and dips) above the line
signal discrete-spectrum bubble channels.

### Resonance scan (paper Fig. 3 inset / Fig. 4)

![Resonance scan](docs/figures/fig_resonance_scan.png)

Linear-scale `Γ(Δ_l)` and time-averaged total bubble density. The
deviations from the smooth tunneling law track the time-averaged bubble
content, supporting the resonant-nucleation interpretation.

### Bubble-length distribution (paper Fig. 4)

![Bubble histogram](docs/figures/fig_bubble_histogram.png)

Time-averaged `⟨Σ_L⟩` for `L = 1, 2, 3` at an "off-resonance" `Δ_l`
where the smooth law dominates and an "on-resonance" `Δ_l` where larger
bubbles are amplified.

### Imperfection sensitivity (paper Fig. 5 / SM)

![Imperfection sensitivity](docs/figures/fig_imperfection_sensitivity.png)

Replacing the perfect Néel by a slightly perturbed initial state shifts
the trajectory. The default noise model is a coherent admixture of
single-flip states (`single_flip_admixed_neel`), which mimics finite
Rabi-pulse preparation infidelity; lower fidelity decays faster, exactly
the qualitative effect the paper highlights. A Haar-random admixture is
available via `--noise-model haar`.

### Finite-N ED vs iTEBD (this package)

![Finite-N vs iTEBD](docs/figures/fig_thermodynamic_limit.png)

Closed-system M_AFM(t) at the same Δ_l for N=8 ED, N=12 ED, and iTEBD
(N → ∞) with NN-only vdW. The three curves overlap at short times and
the finite-N=8 trace separates from N=12 and iTEBD around t ≈ 2 μs as
the boundary makes itself felt. Useful both as a backend showcase and
as a figure-level cross-check for the iTEBD ↔ ED short-time agreement.

---

## Installation

```bash
# Base install — pure NumPy/SciPy backend, all figures except Lindblad.
pip install -e .

# Add backends as you need them:
pip install -e .[qutip]     # closed/open-system via QuTiP (recommended)
pip install -e .[quspin]    # symmetry-resolved ED via QuSpin
pip install -e .[itebd]     # infinite-chain iTEBD via TeNPy
pip install -e .[all]       # everything
pip install -e .[dev]       # plus pytest etc.
```

Python 3.11 or 3.12 are supported. QuSpin and TeNPy require a working
C++/Cython toolchain.

---

## Quickstart

```python
import numpy as np
from rydberg_trampoline import ModelParams, run_unitary

params = ModelParams(N=8, Omega=1.8, Delta_g=4.8, Delta_l=2.0, V_NN=6.0)
times = np.linspace(0.0, 2.0, 41)
res = run_unitary(params, times)         # default backend: numpy
print(res.m_afm[:5])                      # M_AFM(t) at the first five timesteps
```

Open-system evolution with the experimental decoherence:

```python
from rydberg_trampoline import run_lindblad

params = params.with_(T1=28.0, T2_star=3.8)
res = run_lindblad(params, times, backend="qutip", method="auto")
```

Switch backends transparently:

```python
res_numpy  = run_unitary(params.with_(T1=None, T2_star=None), times, backend="numpy")
res_qutip  = run_unitary(params.with_(T1=None, T2_star=None), times, backend="qutip")
res_quspin = run_unitary(params.with_(T1=None, T2_star=None), times, backend="quspin")
```

iTEBD on the infinite chain (`vdW_cutoff` truncated to NN — see the
backend docstring):

```python
from rydberg_trampoline import run_itebd
res = run_itebd(params.with_(T1=None, T2_star=None, vdW_cutoff=1), times, chi=80)
```

---

## Backend reference

| Backend | Methods | Hard cap on N | Extra dependency |
|---------|---------|---------------|-------------------|
| `numpy` | `expm_multiply` Krylov ED, dense Liouvillian Lindblad | 18 / 10 | none |
| `qutip` | `sesolve`, `mesolve` (≤ 10), `mcsolve` (> 10) | 18 (mcsolve) | `qutip>=5.0` |
| `quspin`| Krylov ED on full Hilbert space *or* a `kblock` / `pblock` symmetry sector | 22 with `kblock` | `quspin>=0.3.7` |
| `tenpy` | TEBD on a 2-site iMPS (NN-only vdW) | thermodynamic limit | `physics-tenpy>=0.11` |
| `bloqade` | Analog Rydberg via in-process emulator *or* QuEra Aquila on AWS Braket (paid, opt-in) — shot-statistical, M_AFM only, starts from \|gg…g⟩ | 256 (Aquila) | `bloqade>=0.30`, `amazon-braket-sdk` |

The QuSpin sector path:

```python
res = run_unitary(params, times, backend="quspin", kblock=0)
# The Néel false-vacuum lives in (kblock=0, pblock=+1). Larger N benefits
# most: at N=16 the k=0 sector is ≈ 4× smaller than the full Hilbert space.
```

The bloqade cloud-and-emulator path:

```python
# In-process emulator (free, no AWS):
res = run_unitary(params, times, backend="bloqade", n_shots=2000, seed=0)

# Async variant for notebooks / larger pipelines:
from rydberg_trampoline import run_unitary_async
res = await run_unitary_async(params, times, n_shots=2000)
```

Real-cloud submission to QuEra Aquila and AWS auth setup are documented
in [`docs/cloud_quickstart.md`](docs/cloud_quickstart.md). Submission is
gated behind an explicit `i_understand_this_costs_money=True` flag so
nothing is billed by accident.

Cross-backend regression on N = 8 confirms the closed-system
M_AFM(t) trajectories agree to ~10⁻⁶ (QuTiP RK tolerance) or 10⁻¹⁴
(NumPy ↔ QuSpin Krylov).

---

## Reproducing the paper

```bash
# Rebuild every PNG in docs/figures/
scripts/regenerate_figures.sh

# Or one at a time, with full per-figure flags:
python -m rydberg_trampoline.figures.fig_gamma_vs_inv_delta --N 12

# CLI shortcuts:
python -m rydberg_trampoline.cli backends            # list installed backends
python -m rydberg_trampoline.cli figures all         # rerun every figure
```

Experimental data overlay (digitised from paper figures via WebPlotDigitizer):
see [`rydberg_trampoline/data/experimental/PROVENANCE.md`](rydberg_trampoline/data/experimental/PROVENANCE.md).
Until digitisation is in, the figure scripts simply omit the overlay.

---

## Status and known limitations

- iTEBD long-range support beyond nearest-neighbour was prototyped via
  TeNPy's `ExpMPOEvolution` (W^II MPO) and found to be numerically
  unstable for our parameters in TeNPy 1.1; it is disabled until a TDVP
  iMPS path is wired up.
- The QuSpin backend supports **translation-by-2 momentum sectors** at
  the dynamics level: pass `kblock=k` to `run_unitary` and the initial
  state is projected into the sector basis before evolution. Bond
  inversion (`pblock`) is exposed but emits a `GeneralBasisWarning`
  because it does not strictly commute with translation-by-2 — use one
  or the other, not both, in production.
- The imperfection figure now defaults to a coherent **single-flip
  admixture** (`single_flip_admixed_neel`) which models the dominant
  preparation infidelity from a finite Rabi pulse; the Haar-random model
  remains available via `--noise-model haar`.
- Experimental data overlay infrastructure is wired into the figure
  scripts but the digitised CSVs are not yet shipped (see
  [`PROVENANCE.md`](rydberg_trampoline/data/experimental/PROVENANCE.md)).
  Overlays no-op gracefully when CSVs are absent.

## Reproducible Docker image

```bash
docker build -t rydberg-trampoline:dev .
docker run --rm -it -v "$PWD":/work -w /work rydberg-trampoline:dev pytest -q
```

Avoids the QuSpin / TeNPy install dance on contributor machines.

---

## References

- Chao et al., *Probing False Vacuum Decay and Bubble Nucleation in a
  Rydberg Atom Array*, PRL **136**, 120407 (2026) —
  [arXiv:2512.04637](https://arxiv.org/abs/2512.04637).
- S. Coleman, *Fate of the False Vacuum*, Phys. Rev. D **15**, 2929 (1977).
- Bernien et al., *Probing many-body dynamics on a 51-atom quantum simulator*,
  Nature **551**, 579 (2017).

## License

MIT.
