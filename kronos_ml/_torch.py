"""Lazy torch import so `import kronos_ml` works in a pure numpy/scipy env.
Mirrors kronos_toolkit.report.plotting._mpl — torch is only touched on use."""
from __future__ import annotations

_torch = None


def torch():
    global _torch
    if _torch is None:
        try:
            import torch as _t
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "torch is required for this surrogate; install the extra: "
                "pip install 'kronos-ml[nn]'") from e
        _torch = _t
    return _torch


def available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False
