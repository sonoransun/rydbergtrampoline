"""Shared pytest fixtures and skip markers."""
from __future__ import annotations

import importlib
import warnings

import pytest

# Silence the TeNPy 1.1 warning we cannot suppress at import time.
warnings.filterwarnings(
    "ignore",
    message="unit_cell_width is a new argument",
    category=UserWarning,
)


def _has(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


HAS_QUTIP = _has("qutip")
HAS_QUSPIN = _has("quspin")
HAS_TENPY = _has("tenpy")


qutip_required = pytest.mark.skipif(not HAS_QUTIP, reason="qutip not installed")
quspin_required = pytest.mark.skipif(not HAS_QUSPIN, reason="quspin not installed")
tenpy_required = pytest.mark.skipif(not HAS_TENPY, reason="tenpy not installed")
