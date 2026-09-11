"""KFUSE — multi-fidelity transport surrogate.

Fuses a cheap analytic low-fidelity transport estimate (evaluable everywhere)
with the sparse, expensive high-fidelity CGYRO points (mu=400) via a residual
GP: prediction = low_fidelity + GP(high - low).  This is the classic multi-
fidelity pattern — the cheap model gives the shape, the few expensive points
correct it.  The 3rd fidelity (real-mass mu=3672 "gold") slots in as another
correction when it lands (deferred).
"""
from __future__ import annotations

import numpy as np

from .. import register
from ..base import Surrogate, Prediction
from .. import uq, metrics, data


def _lf_log(a_LT, shear):
    """Cheap analytic low-fidelity: a critical-gradient / shear-damped proxy for
    log10(total heat flux).  Captures the trend, not the magnitude."""
    return np.clip(3.0 * (a_LT - 2.2) - 1.2 * (shear - 0.4) - 1.0, -4.0, 4.0)


@register
class KFUSE(Surrogate):
    name = "KFUSE"; function = "MULTIFID"; phase = 2; status = "BUILT"
    provenance = "CGYRO-urep"
    retired_by = "CGYRO real-mass converged (mu=3672)"
    real_codes = ("multi-fidelity GP", "CGYRO")
    gates = ("BR-L2-A1e", "BR-L2-A1c-MS")
    note = ("multi-fidelity surrogate: cheap analytic low-fidelity + a residual GP "
            "correction from the mu=400 CGYRO points; real-mass gold is the 3rd fidelity")

    def __init__(self):
        self._gp = None

    def _map(self):
        df = data.cgyro_flux_map_final()
        X = df[["a_LT", "shear"]].to_numpy(float)
        y = np.log10(np.clip(df.Q_tot.to_numpy(float), 0, None) + 1e-4)
        return X, y

    def _fit(self):
        if self._gp is None:
            X, y = self._map()
            resid = y - _lf_log(X[:, 0], X[:, 1])          # high - low fidelity
            self._gp = uq.GPHead().fit(X, resid)
        return self._gp

    def _predict(self, x):
        gp = self._fit(); x = np.asarray(x, float).reshape(1, -1)
        lf = _lf_log(x[0, 0], x[0, 1])
        dm, ds = gp.predict(x)
        log_q = lf + dm[0]
        q = max(float(10 ** log_q - 1e-4), 0.0)
        unc = float(q * np.log(10) * ds[0] * 1.25)         # +25% for pending real-mass
        return Prediction(q, unc, bool(2.0 <= x[0, 0] <= 3.5 and 0.4 <= x[0, 1] <= 1.6),
                          note="multi-fidelity Q_tot (mu=400); real-mass gold pending -> +25% unc")

    def _x_from_params(self, a_LT=3.0, shear=1.0, **_):
        return [float(a_LT), float(shear)]

    def benchmark(self):
        X, y = self._map(); n = len(y)
        lf = _lf_log(X[:, 0], X[:, 1])
        # leave-one-out: low-fidelity alone vs multi-fidelity (LF + residual GP)
        mf = np.zeros(n)
        for i in range(n):
            tr = np.arange(n) != i
            g = uq.GPHead().fit(X[tr], (y - lf)[tr])
            mf[i] = lf[i] + g.predict(X[i:i + 1])[0][0]
        return {"member": self.name,
                "live_multifidelity": {
                    "rmse_low_fidelity_only": round(metrics.rmse(y, lf), 3),
                    "rmse_multifidelity": round(metrics.rmse(y, mf), 3),
                    "r2_multifidelity": round(metrics.r2(y, mf), 3),
                    "note": "residual GP correction from 16 mu=400 CGYRO points reduces LF error"},
                "fidelities": "low = analytic proxy; high = mu=400 CGYRO; 3rd = real-mass gold (deferred)"}
