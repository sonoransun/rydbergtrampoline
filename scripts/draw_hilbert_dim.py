"""Render Hilbert-space cost vs N for each backend.

Drives `docs/figures/hilbert_dim_vs_N.png`. Shows on a log-y axis:

* full Hilbert space: 2^N
* Lindblad density-matrix space: 4^N
* QuSpin k=0 sector dim (sample exact values up to a safe N)
* iTEBD memory ~ 2 · χ² · d (constant in N)

Vertical dashed lines mark the package's hard caps and the paper's
N=24 target so the reader sees where each backend stops.

One-shot script — re-run after the layout changes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rydberg_trampoline.backends.numpy_backend import (
    NUMPY_ED_MAX_N,
    NUMPY_LINDBLAD_MAX_N,
)


def _exact_kblock_dim_via_quspin(N: int) -> int | None:
    """Return the dim of the (kblock=0) sector for ring of N atoms via QuSpin.

    Returns None if the QuSpin import fails or the sector build OOMs.
    """
    try:
        from rydberg_trampoline.backends.quspin_backend import to_quspin
        from rydberg_trampoline.model import ModelParams
    except Exception:
        return None
    try:
        _, basis = to_quspin(ModelParams(N=N, Delta_l=1.0), kblock=0)
    except Exception:
        return None
    return int(basis.Ns)


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    N_full = np.arange(2, 27, dtype=np.int64)
    N_quspin = np.arange(2, 19, 2, dtype=np.int64)  # only enumerate exactly up to 18
    chi = 100  # tenpy default

    full = 2.0 ** N_full
    rho = 4.0 ** N_full
    # iTEBD bond cost: store 2 site tensors of shape (chi, d, chi); memory ~ 2·χ²·d.
    itebd = np.full_like(full, 2.0 * chi * chi * 2)

    # QuSpin sector dim — exact where we can compute it, asymptote 2^N / (N/2)
    # beyond that.
    sector_exact = []
    for N in N_quspin:
        d = _exact_kblock_dim_via_quspin(int(N))
        sector_exact.append(d if d is not None else 2.0 ** int(N) / max(1, int(N) // 2))
    sector_estimate_full = np.where(N_full >= 2, 2.0 ** N_full / np.maximum(1, N_full // 2), np.nan)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(N_full, full, "-", color="#34495e", lw=1.8,
            label=r"full Hilbert  $2^N$  (numpy / qutip / quspin)")
    ax.plot(N_quspin, sector_exact, "o", color="#16a085", ms=6,
            label=r"QuSpin $k\!=\!0$ sector (exact)")
    ax.plot(N_full, sector_estimate_full, ":", color="#16a085", lw=1.0,
            label=r"      $\sim 2^N / (N/2)$  asymptote")
    ax.plot(N_full, rho, "-", color="#c0392b", lw=1.8,
            label=r"Lindblad $\rho$  $4^N$  (qutip mesolve / numpy)")
    ax.plot(N_full, itebd, "-", color="#2980b9", lw=1.8,
            label=fr"iTEBD memory $\sim 2 \chi^2 d$  ($\chi={chi}$)")

    ax.set_yscale("log")
    ax.set_xlabel("ring size  N")
    ax.set_ylabel("Hilbert-space / memory dimension")
    ax.set_xlim(N_full.min(), N_full.max())
    ax.grid(alpha=0.3, linestyle=":", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Hard-cap markers — a vertical line annotated at a y-position chosen to sit
    # in clear space (above iTEBD's flat blue line, below the Lindblad curve).
    cap_specs = [
        (NUMPY_LINDBLAD_MAX_N, f"Lindblad cap  (N={NUMPY_LINDBLAD_MAX_N})", "#c0392b", 1e9),
        (NUMPY_ED_MAX_N,       f"numpy ED cap  (N={NUMPY_ED_MAX_N})",       "#34495e", 1e9),
        (24,                   "paper target  (N=24)",                     "black",   1e9),
    ]
    for x, label, color, y in cap_specs:
        ax.axvline(x, color=color, lw=0.9, ls="--", alpha=0.55)
        ax.text(
            x - 0.18, y, label,
            color=color, fontsize=9, ha="right", va="center", rotation=90,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        )

    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.set_title(
        "Memory cost vs ring size N for each backend  (log scale; lower = laptop-friendly)",
        fontsize=11,
    )

    fig.tight_layout()
    out_path = out_dir / "hilbert_dim_vs_N.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
