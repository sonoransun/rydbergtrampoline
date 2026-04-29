"""QuSpin: spectrum decomposed by translation-by-2 sectors equals full spectrum."""
from __future__ import annotations

import numpy as np
import pytest

from rydberg_trampoline.model import ModelParams
from tests.conftest import quspin_required


@quspin_required
@pytest.mark.parametrize("N", [4, 6, 8, 10])
def test_kblock_decomposition_recovers_full_spectrum(N: int) -> None:
    from rydberg_trampoline.backends.quspin_backend import (
        to_quspin,
        translation_block_count,
    )

    params = ModelParams(N=N, Omega=1.8, Delta_g=4.8, Delta_l=2.0, V_NN=6.0, vdW_cutoff=8)

    H_full, b_full = to_quspin(params)
    full_eigs = np.sort(np.linalg.eigvalsh(H_full.toarray()))

    sector_eigs = []
    total_dim = 0
    for k in range(translation_block_count(N)):
        H_k, b_k = to_quspin(params, kblock=k)
        total_dim += b_k.Ns
        eigs = np.linalg.eigvalsh(H_k.toarray())
        sector_eigs.append(eigs)
    combined = np.sort(np.concatenate(sector_eigs))

    assert total_dim == 1 << N, f"sector dims sum to {total_dim} != {1 << N}"
    np.testing.assert_allclose(combined, full_eigs, atol=1e-10)


@quspin_required
def test_translation_block_count_value() -> None:
    from rydberg_trampoline.backends.quspin_backend import translation_block_count

    assert translation_block_count(4) == 2
    assert translation_block_count(8) == 4
    assert translation_block_count(16) == 8
    with pytest.raises(ValueError):
        translation_block_count(7)
