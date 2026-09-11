"""Neutronics surrogates over the kronos-toolkit neutronics engine.

KBREED — tritium breeding ratio surrogate (Hyperion blanket).
KFLUX  — neutron shielding / coil-fluence surrogate.

Both train a fast GP over the toolkit's neutronics (analytic now; the toolkit
dispatches to OpenMC Monte-Carlo transport automatically when it is installed, so
the OpenMC high-fidelity upgrade needs no code change here — just the compute).
"""
from __future__ import annotations

import numpy as np

from .. import register
from ..base import Surrogate, Prediction
from .. import uq, metrics


def _api():
    from kronos_toolkit.neutronics import api
    return api


def _val(tagged, *keys):
    v = tagged.value if hasattr(tagged, "value") else tagged
    if isinstance(v, dict):
        for k in keys:
            if k in v:
                return float(v[k])
        # fall back to the first finite float
        for x in v.values():
            try:
                return float(x)
            except Exception:
                pass
    return float(v)


@register
class KBREED(Surrogate):
    name = "KBREED"; function = "breeder"; phase = 2; status = "BUILT"
    provenance = "SIM"
    retired_by = "OpenMC (Monte-Carlo neutron transport)"
    real_codes = ("openmc",)
    gates = ("BR-neut",)
    note = ("tritium breeding-ratio surrogate (Hyperion blanket) — fast GP over the "
            "toolkit neutronics; OpenMC is the auto-dispatch high-fidelity upgrade")

    _B = [(0.3, 0.95), (0.5, 1.2), (0.85, 0.95)]   # li6_enrichment, thickness_m, coverage

    def __init__(self):
        self._gp = None

    def _sweep(self, n=120, seed=uq.SEED):
        api = _api(); rng = np.random.default_rng(seed)
        P = np.c_[rng.uniform(*self._B[0], n), rng.uniform(*self._B[1], n), rng.uniform(*self._B[2], n)]
        y = np.array([_val(api.tbr(li6_enrichment=p[0], blanket_thickness_m=p[1],
                                   coverage=p[2], force_analytic=True), "net_tbr")
                      for p in P])
        return P, y

    def _fit(self):
        if self._gp is None:
            P, y = self._sweep()
            self._gp = uq.GPHead().fit(P, y); self._P, self._y = P, y
        return self._gp

    def in_domain(self, p):
        return all(self._B[i][0] <= p[i] <= self._B[i][1] for i in range(3))

    def _predict(self, x):
        gp = self._fit(); x = np.asarray(x, float).reshape(1, -1)
        m, s = gp.predict(x)
        return Prediction(float(m[0]), float(s[0]), self.in_domain(x.ravel()),
                          note="net TBR (>1 = self-sufficient)")

    def _x_from_params(self, li6_enrichment=0.9, blanket_thickness_m=0.8, coverage=0.92, **_):
        return [li6_enrichment, blanket_thickness_m, coverage]

    def benchmark(self):
        gp = self._fit()
        Pte, yte = self._sweep(n=60, seed=uq.SEED + 5)
        m, s = gp.predict(Pte)
        api = _api()
        nom = _val(api.tbr(force_analytic=True), "net_tbr")
        return {"member": self.name,
                "live": {"r2_vs_engine": round(metrics.r2(yte, m), 3),
                         "coverage_90": round(metrics.coverage_at_level(yte, m, np.clip(s, 1e-6, None), 0.90), 3),
                         "nominal_net_tbr": round(nom, 3)},
                "note": "surrogate over toolkit neutronics (analytic); OpenMC upgrade = compute route [A]"}


@register
class KFLUX(Surrogate):
    name = "KFLUX"; function = "neutronics"; phase = 2; status = "BUILT"
    provenance = "SIM"
    retired_by = "OpenMC (Monte-Carlo neutron transport)"
    real_codes = ("openmc",)
    note = ("neutron shielding / coil-fluence surrogate — fast GP over the toolkit "
            "neutronics; OpenMC is the auto-dispatch high-fidelity upgrade")

    _B = [(10.0, 100.0), (0.3, 1.5)]   # p_neutron_MW, shield_thickness_m

    def __init__(self):
        self._gp = None

    def _sweep(self, n=120, seed=uq.SEED):
        api = _api(); rng = np.random.default_rng(seed)
        P = np.c_[rng.uniform(*self._B[0], n), rng.uniform(*self._B[1], n)]
        y = np.array([np.log10(_val(api.coil_fluence(p_neutron_MW=p[0], shield_thickness_m=p[1],
                                                     force_analytic=True)) + 1.0)
                      for p in P])
        return P, y

    def _fit(self):
        if self._gp is None:
            P, y = self._sweep()
            self._gp = uq.GPHead().fit(P, y)
        return self._gp

    def in_domain(self, p):
        return all(self._B[i][0] <= p[i] <= self._B[i][1] for i in range(2))

    def _predict(self, x):
        gp = self._fit(); x = np.asarray(x, float).reshape(1, -1)
        m, s = gp.predict(x)
        return Prediction(float(10 ** m[0] - 1.0), float(s[0]), self.in_domain(x.ravel()),
                          note="coil neutron fluence (log10 model)")

    def _x_from_params(self, p_neutron_MW=50.0, shield_thickness_m=0.8, **_):
        return [p_neutron_MW, shield_thickness_m]

    def benchmark(self):
        gp = self._fit()
        Pte, yte = self._sweep(n=60, seed=uq.SEED + 5)
        m, _ = gp.predict(Pte)
        return {"member": self.name,
                "live": {"r2_log_fluence_vs_engine": round(metrics.r2(yte, m), 3),
                         "n_test": len(Pte)},
                "note": "surrogate over toolkit shielding/fluence (analytic); OpenMC upgrade = compute route [A]"}
