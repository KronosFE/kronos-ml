"""KODEX — the Kronos Family of Codes.

A benchmarked AI/ML surrogate suite for fusion.  Every code obeys one contract,
`predict(x) -> Prediction(y, uncertainty, in_domain)`, carries honest provenance
(`[T]` tagged, retired_by the real code it stands in for), and reports the same
calibration metrics.  Imports cleanly without torch/sklearn — heavy deps load
only when a surrogate actually runs.
"""
from __future__ import annotations

__version__ = "0.3.1"

from .base import Surrogate, Prediction  # noqa: E402

#: brand-name -> Surrogate subclass
SURROGATES: dict[str, type] = {}


def register(cls):
    """Class decorator: add a Surrogate subclass to the fleet registry."""
    SURROGATES[cls.name] = cls
    return cls


def get(name: str) -> Surrogate:
    """Instantiate a surrogate by brand name (e.g. get('KYRO'))."""
    return SURROGATES[name]()


def run(name: str, x):
    """Convenience: predict with a named surrogate. -> Prediction."""
    return get(name).predict(x)


def list_surrogates():
    return sorted(SURROGATES)


def fleet(phase: int | None = None, status: str | None = None):
    """Return the fleet as a list of cards, optionally filtered."""
    cards = [get(n).card() for n in list_surrogates()]
    if phase is not None:
        cards = [c for c in cards if c["phase"] == phase]
    if status is not None:
        cards = [c for c in cards if c["status"] == status]
    return sorted(cards, key=lambda c: (c["phase"], c["name"]))


# populate the registry (members import numpy + base only at module load;
# torch/sklearn are lazy, so this stays import-clean in a bare env)
from . import members  # noqa: E402,F401
