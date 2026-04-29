"""Shared pytest fixtures and skip markers."""
from __future__ import annotations

import importlib

import pytest


def _has(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


HAS_QUTIP = _has("qutip")
HAS_QUSPIN = _has("quspin")
HAS_TENPY = _has("tenpy")
HAS_BLOQADE = _has("bloqade.analog")


qutip_required = pytest.mark.skipif(not HAS_QUTIP, reason="qutip not installed")
quspin_required = pytest.mark.skipif(not HAS_QUSPIN, reason="quspin not installed")
tenpy_required = pytest.mark.skipif(not HAS_TENPY, reason="tenpy not installed")
bloqade_required = pytest.mark.skipif(not HAS_BLOQADE, reason="bloqade not installed")
