# Changelog

All notable changes to this project will be documented in this file. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project adheres to [Semantic Versioning](https://semver.org/) once it
reaches 1.0.

## [Unreleased]

### Added
- **bloqade / QuEra Aquila cloud backend.** New `backend="bloqade"` runs
  the staggered Rydberg Hamiltonian on bloqade-analog's in-process
  emulator (default, free) or — opt-in via
  `i_understand_this_costs_money=True` — on QuEra Aquila on AWS Braket.
  Async API exposed as `run_unitary_async`; sync `run_unitary` wraps it.
  The bloqade path is shot-statistical (`M_AFM(t)` from bitstrings) and
  starts from `|gg…g⟩` only (Aquila hardware constraint).
- New `[cloud]` extra in `pyproject.toml` (`bloqade>=0.30`,
  `amazon-braket-sdk>=1.112`, `boto3>=1.34`).
- New `tests/test_bloqade_backend.py` (8 tests): emulator smoke,
  cost-gate without flag, AWS-auth probe with flag and no creds, ψ₀
  must be all-ground, explicit `|gg…g⟩` accepted, NumPy↔bloqade-emulator
  agreement on `M_AFM(t)` within shot noise (4000 shots, tolerance
  0.05), `run_unitary_async` is a coroutine.
- New `docs/cloud_quickstart.md` covering install, emulator usage,
  AWS setup, cost-control tips, and out-of-scope items.
- iTEBD-visible figure `fig_thermodynamic_limit.py`: overlays `M_AFM(t)` for
  N=8 ED, N=12 ED, and iTEBD (N → ∞) with NN-only vdW, exposing the
  finite-size light-cone and the iTEBD backend visually.
- `tests/test_figure_regression.py`: perceptual-hash (`imagehash.phash`)
  regression for every committed PNG against pinned hashes in
  `tests/figure_hashes.json`. Skips gracefully if `imagehash` is not
  installed.
- `tests/test_states.py`: dedicated coverage for state factories.
- `tests/test_itebd_vs_ed.py`: split out from the umbrella cross-backend file
  per the original layout plan.
- Symmetry-resolved dynamics in QuSpin: `run_unitary` accepts `kblock` /
  `pblock`, projects the initial state into the sector, evolves there, and
  reads observables back. The headline Néel false vacuum lives in
  (k=0, p=+1) and is the intended driver of the speedup.
- `LICENSE`, `CHANGELOG.md`, `MANIFEST.in`, `.pre-commit-config.yaml`, and a
  `[tool.ruff]` block in `pyproject.toml`.
- `imagehash>=4.3` added to the `[dev]` extras.

### Changed
- `backends/tenpy_backend.py::_initial_neel_imps` passes `unit_cell_width`
  through `MPS.from_product_state` directly, silencing the TeNPy 1.1
  `UserWarning` at the source. The fallback `warnings.filterwarnings`
  workaround in `tests/conftest.py` is removed.
- README now embeds the new iTEBD figure and documents the sector-dynamics
  path in the backend table.

## [0.1.0] – 2026-04-28

### Added
- Initial implementation of the model and methods of Chao et al., PRL **136**,
  120407 (2026); arXiv:2512.04637.
- Four backends: pure NumPy/SciPy, QuTiP, QuSpin, and TeNPy iTEBD (NN-only).
- Five hero figures: decay traces, Γ vs 1/Δ_l, resonance scan, bubble
  histogram, imperfection sensitivity. All regenerable from runnable scripts
  with JSON sidecars and committed PNGs in `docs/figures/`.
- 55-test pytest suite: conventions, observables, cross-backend regression,
  invariants (Hermiticity / energy / trace), Lindblad → unitary limit,
  iTEBD ↔ ED short-time agreement, QuSpin translation-by-2 sector
  decomposition.
- Expanded README with physics background, Mermaid topology / pipeline
  diagrams, and embedded figures.
- `docs/background.md` long-form physics page.
- Reproducible `Dockerfile`, `.github/workflows/ci.yml` matrix, and a
  project-tailored `.gitignore`.
- Single-flip imperfection model (`single_flip_admixed_neel`) that mirrors
  the paper's preparation-error physics; Haar-random alternative remains
  available via `--noise-model haar`.
- QuSpin translation-by-2 (`kblock`) and bond-inversion (`pblock`) sectors,
  with `translation_block_count` helper.
- Experimental data overlay infrastructure (`data/loader.py`,
  `data/experimental/PROVENANCE.md`); CSVs not yet shipped — overlays
  no-op gracefully.

[Unreleased]: https://github.com/example/rydberg-trampoline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/example/rydberg-trampoline/releases/tag/v0.1.0
