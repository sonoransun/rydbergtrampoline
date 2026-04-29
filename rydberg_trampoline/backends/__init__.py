"""Backend registry.

Importing a backend submodule will fail with a clean ``ModuleNotFoundError`` if
its optional dependency is missing. The registry below records what's
available without forcing the import.
"""
from __future__ import annotations

from typing import Literal

BackendName = Literal["numpy", "qutip", "quspin", "tenpy"]


def available_backends() -> list[BackendName]:
    """Return the list of backends importable in this environment."""
    out: list[BackendName] = ["numpy"]  # numpy backend has only required deps
    try:
        import qutip  # noqa: F401
        out.append("qutip")
    except Exception:
        pass
    try:
        import quspin  # noqa: F401
        out.append("quspin")
    except Exception:
        pass
    try:
        import tenpy  # noqa: F401
        out.append("tenpy")
    except Exception:
        pass
    return out


def require_backend(name: BackendName) -> None:
    """Raise an informative error if backend ``name`` cannot be imported."""
    if name == "numpy":
        return
    if name not in available_backends():
        raise ModuleNotFoundError(
            f"backend '{name}' requires the optional dependency. "
            f"Install with: pip install 'rydberg-trampoline[{ {'qutip': 'qutip', 'quspin': 'quspin', 'tenpy': 'itebd'}[name] }]'"
        )
