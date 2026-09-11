"""The one contract every KODEX code obeys.

    predict(x) -> Prediction(y, uncertainty, in_domain)

A KODEX surrogate is a fast, calibrated stand-in for an expensive fusion
computation.  Three promises come with every prediction:

    y            the point estimate
    uncertainty  a calibrated 1-sigma (or class-probability spread)
    in_domain    whether the input is inside the region the surrogate is
                 trusted on; when False the caller must defer to full physics.

Provenance is carried in the type: `.as_tagged()` returns a
`kronos_toolkit.verify.tags.Tagged`, and every surrogate that stands in for a
real code is a `[T]` naming the code that retires it (`retired_by`).  Nothing
here is ever called "live"; a code is BUILT, PARTIAL, or ROADMAP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:                                    # kronos-ml depends on kronos-toolkit
    from kronos_toolkit.verify.tags import Tagged, surrogate as _surrogate_tag
except Exception:                       # pragma: no cover - toolkit always present in the env
    Tagged = None
    _surrogate_tag = None

STATUSES = ("BUILT", "PARTIAL", "ROADMAP")
PROVENANCE = ("SIM", "TWIN", "CGYRO-urep", "ANALYTIC", "REAL-ANCHOR", "n/a")


@dataclass
class Prediction:
    """A single surrogate prediction with its calibrated uncertainty and gate."""
    y: Any
    uncertainty: Any = None
    in_domain: bool = True
    note: str = ""

    def as_tuple(self):
        return (self.y, self.uncertainty, self.in_domain)

    def as_tagged(self, retired_by: str, note: str = ""):
        """Wrap as a [T] Tagged value naming the real code that retires it."""
        if Tagged is None:
            raise RuntimeError("kronos_toolkit not importable; cannot tag")
        payload = {"y": self.y, "uncertainty": self.uncertainty,
                   "in_domain": self.in_domain}
        return Tagged(payload, "T", note=note or self.note, retired_by=retired_by)


class Surrogate:
    """Base class for every KODEX code.  Subclasses set the class attributes and
    implement `_predict`; roadmap codes leave `_predict` raising and set
    status='ROADMAP'."""

    #: brand name, e.g. "KYRO"
    name: str = "SURROGATE"
    #: technical descriptor, e.g. "TRANSPORT"
    function: str = ""
    #: the real code(s) this stands in for
    real_codes: tuple = ()
    #: the real code that retires this [T] surrogate (required for BUILT/PARTIAL)
    retired_by: str = ""
    #: SIM / TWIN / CGYRO-urep / ANALYTIC / REAL-ANCHOR
    provenance: str = "SIM"
    #: BUILT / PARTIAL / ROADMAP
    status: str = "ROADMAP"
    #: rollout phase 1/2/3
    phase: int = 3
    #: de-risking register gate(s) this ties to
    gates: tuple = ()
    #: one-line description
    note: str = ""
    #: frozen input->expected-output pin for the drift regression (or None)
    ANCHOR = None

    # -- prediction -------------------------------------------------------
    def _predict(self, x) -> Prediction:
        raise NotImplementedError(
            f"{self.name} is ROADMAP — not built yet")

    def predict(self, x) -> Prediction:
        p = self._predict(x)
        if not isinstance(p, Prediction):
            # tolerate a (y, unc, in_domain) tuple
            p = Prediction(*p) if isinstance(p, tuple) else Prediction(p)
        return p

    def evaluator(self, **params) -> dict:
        """Adapter for kronos_toolkit.uq / report.Study: returns a flat dict of
        scalars {y, uncertainty, in_domain} so the existing UQ/Study machinery
        consumes a KODEX surrogate unchanged."""
        p = self.predict(params if not params else self._x_from_params(**params))
        out = {"y": _scalar(p.y), "in_domain": float(bool(p.in_domain))}
        if p.uncertainty is not None:
            out["uncertainty"] = _scalar(p.uncertainty)
        return out

    def _x_from_params(self, **params):
        """Default: pass the params dict straight through. Members override when
        they need an array/graph input."""
        return params

    # -- availability / metadata -----------------------------------------
    def available(self) -> bool:
        """True when this code can actually run here (deps + data/checkpoint on
        disk).  ROADMAP codes are never available."""
        return self.status in ("BUILT", "PARTIAL")

    def benchmark(self) -> dict:
        """Return the honest on-disk benchmark: speed / accuracy / uq calibration.
        Members override; roadmap codes return {}."""
        return {}

    def card(self) -> dict:
        return {
            "name": self.name, "function": self.function,
            "status": self.status, "phase": self.phase,
            "provenance": self.provenance, "retired_by": self.retired_by,
            "gates": list(self.gates), "note": self.note,
            "available": self.available(),
        }


def _scalar(v):
    try:
        import numpy as _np
        if isinstance(v, _np.ndarray):
            return float(v.reshape(-1)[0]) if v.size == 1 else float(_np.mean(v))
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return v
