"""TeNPy iTEBD backend (physics-tenpy >= 1.0).

Implements the staggered Rydberg model on an *infinite* chain with a 2-site
unit cell (matching the staggered detuning) and time-evolves the false-vacuum
Néel iMPS via TEBD.

**Long-range note.** TEBD requires a nearest-neighbour Hamiltonian on the MPS
chain. The Rydberg model's vdW interaction has a 1/r^6 tail that is dominated
by the nearest neighbour (V_NN ≫ V_NNN by a factor 64). For this backend we
therefore truncate the vdW tail at ``vdW_cutoff = 1`` regardless of what the
user requests in ``ModelParams.vdW_cutoff`` and emit a warning if the original
cutoff was higher. Beyond-NN long-range support via ``ExpMPOEvolution`` /
``W^II`` was prototyped and found to be numerically unstable in TeNPy 1.1.0
for our parameters; that path is left as a TODO.

The cross-backend test in ``tests/test_itebd_vs_ed.py`` verifies that this
truncation reproduces ED on a finite ring at small N and short times, before
the light cone reaches the boundary.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np

from rydberg_trampoline.dynamics import DynamicsResult
from rydberg_trampoline.model import ModelParams


class _RydbergStaggeredModel:
    """Build a TeNPy ``CouplingMPOModel`` for the staggered Rydberg chain.

    Wrapped in a class so we can defer the TeNPy import until call time.
    """

    def __init__(self, params: ModelParams):
        from tenpy.models.lattice import Lattice
        from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
        from tenpy.networks.site import SpinHalfSite
        from tenpy.tools.params import asConfig

        if params.vdW_cutoff > 1:
            warnings.warn(
                "TeNPy backend currently truncates vdW interactions to nearest-neighbour "
                f"only (got vdW_cutoff={params.vdW_cutoff}). The MPS chain must be "
                "nearest-neighbour for TEBD; long-range support via ExpMPOEvolution is "
                "experimental and disabled in v1.",
                stacklevel=2,
            )

        class _Inner(CouplingMPOModel, NearestNeighborModel):
            def __init__(inner, model_params):
                super().__init__(model_params)

            def init_lattice(inner, model_params):
                # 2-site unit cell so the staggered field is properly translation-invariant.
                site_a = SpinHalfSite(conserve=None)
                site_b = SpinHalfSite(conserve=None)
                L = model_params.get("L", 1)
                bc_MPS = model_params.get("bc_MPS", "infinite")
                return Lattice(
                    Ls=[L],
                    unit_cell=[site_a, site_b],
                    bc_MPS=bc_MPS,
                    bc="periodic",
                )

            def init_sites(inner, model_params):
                return SpinHalfSite(conserve=None)

            def init_terms(inner, model_params):
                Omega = model_params.get("Omega", 1.8)
                Delta_g = model_params.get("Delta_g", 4.8)
                Delta_l = model_params.get("Delta_l", 0.0)
                V_NN = model_params.get("V_NN", 6.0)
                # NB: TEBD requires nearest-neighbour Hamiltonian; we force R=1.
                R = 1
                # Drive: (Ω/2) σ^x_j on every site of the (2-site) unit cell.
                # SpinHalfSite has 'Sx' = σ^x / 2, so coefficient in S^x units is Ω.
                for u in range(len(inner.lat.unit_cell)):
                    inner.add_onsite(Omega, u, "Sx")
                # Local detuning: site_field(j) n_j = site_field(j) (I + σ^z)/2
                # In S-operator units: site_field(j) (I + 2 S^z) / 2
                #   constant: site_field(j) / 2  (irrelevant)
                #   Sz coeff: site_field(j)
                # For a 2-site unit cell, site_field alternates: u=0 → −Δ_g + Δ_l, u=1 → −Δ_g − Δ_l.
                inner.add_onsite(-Delta_g + Delta_l, 0, "Sz")
                inner.add_onsite(-Delta_g - Delta_l, 1, "Sz")
                # Pair interactions: V_ij n_i n_j → V_ij (I + 2 S^z_i)(I + 2 S^z_j)/4
                # Drop constant and S^z⊗I, I⊗S^z onsite contributions which we add explicitly:
                #   V_ij/4  · I               (drop)
                #   V_ij/2  · S^z_i           → onsite to site i
                #   V_ij/2  · S^z_j           → onsite to site j
                #   V_ij    · S^z_i S^z_j     → coupling
                # We accumulate the linear contributions per unit-cell site below.
                #
                # On a 2-site unit cell, each site couples to neighbours at offsets
                # 1, 2, …, R. The coupling to a neighbour at offset d connects
                # u → (u + d) mod 2 with shift floor((u + d) / 2). We use TeNPy's
                # add_coupling which already handles the lattice offset.
                from tenpy.networks.site import SpinHalfSite  # noqa: F401 (used implicitly)
                # accumulate onsite Sz contributions from neighbour pairs
                onsite_sz = np.zeros(2, dtype=np.float64)
                for d in range(1, R + 1):
                    v = V_NN / d**6
                    if v == 0.0:
                        continue
                    for u_from in range(2):
                        u_to = (u_from + d) % 2
                        dx = (u_from + d) // 2
                        # σ^z_i σ^z_j coupling (in S units, coeff = v · 4 / 4 · 1 = v)
                        inner.add_coupling(v, u_from, "Sz", u_to, "Sz", dx)
                        onsite_sz[u_from] += 0.5 * v
                        onsite_sz[u_to] += 0.5 * v
                inner.add_onsite(onsite_sz[0], 0, "Sz")
                inner.add_onsite(onsite_sz[1], 1, "Sz")

        cfg = asConfig(
            {
                # Ls = [1] together with the 2-site unit cell gives 2 MPS sites total,
                # which is the iTEBD unit cell of the staggered Hamiltonian.
                "L": 1,
                "bc_MPS": "infinite",
                "Omega": params.Omega,
                "Delta_g": params.Delta_g,
                "Delta_l": params.Delta_l,
                "V_NN": params.V_NN,
                # vdW_cutoff is intentionally not propagated: TEBD is NN-only,
                # so the inner model hard-codes R=1 in init_terms. Surfacing
                # the cutoff would only cause TeNPy's "unused option" warning.
            },
            "RydbergStaggered",
        )
        self.tenpy_model = _Inner(cfg)


def _initial_neel_imps(model: "_RydbergStaggeredModel", phase: int = 0):
    """Construct the 2-site Néel iMPS as the initial state.

    ``phase=0`` gives ⟨n⟩ = (1, 0) on the unit cell — the false vacuum.
    """
    from tenpy.networks.mps import MPS
    sites = model.tenpy_model.lat.mps_sites()
    if phase not in (0, 1):
        raise ValueError("phase must be 0 or 1")
    # SpinHalfSite states are 'up' (Sz = +1/2 → σ^z = +1 → n = 1) and 'down' (n=0).
    # phase 0 (false vacuum) → site 0 = up, site 1 = down (alternating in unit cell).
    pattern = ["up", "down"] if phase == 0 else ["down", "up"]
    # Pass unit_cell_width explicitly to silence TeNPy 1.1's UserWarning. Our
    # lattice is a 2-site unit cell on a Chain (one-dimensional), so the
    # default-value-equivalent (= len(sites) = 2) is correct here.
    return MPS.from_product_state(
        sites, pattern, bc="infinite", unit_cell_width=len(sites)
    )


def run_itebd(
    params: ModelParams,
    times: np.ndarray,
    *,
    chi: int = 100,
    bubble_lengths: Iterable[int] | None = None,
) -> DynamicsResult:
    """Time-evolve the false-vacuum Néel iMPS under the staggered Rydberg H.

    Returns ⟨M_AFM⟩(t) computed on the 2-site unit cell:

        ⟨M_AFM⟩ = (⟨σ^z_0⟩ − ⟨σ^z_1⟩) / 2 = ⟨S^z_0⟩ − ⟨S^z_1⟩

    Bubble correlators in the infinite-chain limit could be added by
    constructing the corresponding MPO; this is left as a TODO since the
    paper's bubble histogram is reported on finite N where ED suffices.
    """
    from tenpy.algorithms.tebd import TEBDEngine

    times = np.asarray(times, dtype=np.float64)
    model = _RydbergStaggeredModel(params)
    psi = _initial_neel_imps(model, phase=0)

    n_out = len(times)
    if n_out < 2:
        raise ValueError("times must have at least two entries")

    eng_params = {
        "dt": 0.05,
        "N_steps": 1,
        "order": 4,
        "trunc_params": {"chi_max": chi, "svd_min": 1.0e-10},
    }
    engine = TEBDEngine(psi, model.tenpy_model, eng_params)

    def _measure() -> float:
        sz0 = psi.expectation_value("Sz", sites=[0])[0]
        sz1 = psi.expectation_value("Sz", sites=[1])[0]
        # M_AFM = (1/N) Σ_j (-1)^j σ^z_j; averaged on the 2-site unit cell:
        #         M_AFM = (σ^z_0 − σ^z_1) / 2 = S^z_0 − S^z_1
        # since TeNPy's SpinHalfSite Sz is σ^z / 2 (eigenvalues ±1/2).
        return float(sz0 - sz1)

    m_trace = np.empty(n_out, dtype=np.float64)
    m_trace[0] = _measure()

    t_curr = float(times[0])
    for k in range(1, n_out):
        t_next = float(times[k])
        # Sub-divide each output interval into ~50 ns steps for accuracy.
        n_sub = max(1, int(np.ceil((t_next - t_curr) / 0.05)))
        engine.options["dt"] = (t_next - t_curr) / n_sub
        engine.options["N_steps"] = n_sub
        engine.run()
        m_trace[k] = _measure()
        t_curr = t_next

    return DynamicsResult(
        times=times,
        m_afm=m_trace,
        backend="tenpy-iTEBD",
        notes=f"ExpMPOEvolution chi_max={chi}, vdW_cutoff={params.vdW_cutoff}",
    )
