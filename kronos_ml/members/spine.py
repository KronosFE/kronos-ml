"""The KODEX spine — shared layers every surrogate wraps.

KHALO — calibrated uncertainty (GP + ensemble + the canonical reliability report).
KGATE — the trust-boundary / abstention gate (a model-free envelope clamp that
        decides in_domain and fail-closes on bad input).
"""
from __future__ import annotations

import time
import numpy as np

from .. import register
from ..base import Surrogate, Prediction
from .. import uq, metrics, data


def _kv(df):
    return {str(r["item"]): r["value"] for _, r in df.iterrows()}


@register
class KHALO(Surrogate):
    name = "KHALO"; function = "UQ"; phase = 1; status = "BUILT"
    provenance = "SIM"
    retired_by = "full ensemble UQ / conformal prediction"
    real_codes = ("deep-ensemble", "conformal")
    gates = ("KX-L1-A14", "KX-L1-A5")
    note = ("shared calibrated-uncertainty layer (GP posterior + deep-ensemble) "
            "+ the canonical reliability report")

    def head(self, baseline=None):
        return uq.GPHead(baseline=baseline)

    def ensemble(self):
        return uq.EnsembleHead()

    def calibrate(self, y_true, mean, std):
        return uq.calibrate(y_true, mean, std)

    def _predict(self, x):
        """x = (mean, std): stamp a calibrated Prediction (the layer's job is to
        attach uncertainty to another surrogate's point estimate)."""
        mean, std = x
        return Prediction(float(np.asarray(mean).ravel()[0]),
                          float(np.asarray(std).ravel()[0]), True,
                          note="calibrated via KHALO")

    def benchmark(self):
        out = {"member": self.name}
        # (1) sourced: the real on-disk UQ calibration
        try:
            kv = _kv(data.read_csv("track6_surrogate_uq", "uq_calibration_summary.csv"))
            out["sourced"] = {"file": "track6_surrogate_uq/uq_calibration_summary.csv",
                              "ece": kv.get("ECE_mean_std"),
                              "coverage_90": kv.get("coverage90_mean_std"),
                              "result": kv.get("AC28_RESULT")}
        except Exception as e:
            out["sourced"] = {"error": str(e)}
        # (2) live: reliability of a GP on the CGYRO clean subset (leave-one-out)
        try:
            df = data.cgyro_flux_map()
            cl = df[(df.saturated == 1) & np.isfinite(df.Q_i)].reset_index(drop=True)
            X = cl[["a_LT", "shear"]].to_numpy(float)
            y = np.log(np.clip(cl.Q_i.to_numpy(float), 1e-6, None))
            mu = np.zeros(len(y)); sd = np.zeros(len(y))
            for i in range(len(y)):
                tr = np.arange(len(y)) != i
                if tr.sum() < 3:
                    continue
                h = uq.GPHead().fit(X[tr], y[tr])
                m, s = h.predict(X[i:i + 1]); mu[i] = m[0]; sd[i] = s[0]
            rep = metrics.calibration_report(y, mu, np.clip(sd, 1e-6, None))
            out["live_gp_on_cgyro"] = {"ece": round(rep["ece"], 4),
                                       "coverage_90": round(rep["coverage_90"], 4),
                                       "n": int(len(y))}
        except Exception as e:
            out["live_gp_on_cgyro"] = {"error": str(e)}
        return out


# --- KGATE: model-free envelope clamp / abstention gate -------------------

# frozen physical envelope (cited limits; canon-safe — not the forbidden set)
_LO = np.array([1.5, -0.6, 0.0, 0.0, 0.0, 10.0])   # kappa, delta, beta_N, n_e, eps%, Ip
_HI = np.array([3.0, 0.6, 4.2, 3.0, 0.4, 47.21])
_IDX = ["kappa", "delta", "beta_N", "n_e", "eps_pct", "Ip_MA"]


@register
class KGATE(Surrogate):
    name = "KGATE"; function = "SAFE"; phase = 1; status = "BUILT"
    provenance = "TWIN"
    retired_by = "formal CBF verification + hardware failsafe"
    real_codes = ("CBF certificate", "hardware failsafe")
    note = ("trust-boundary / abstention gate — a model-free envelope clamp that "
            "decides in_domain and fail-closes on bad input")

    def clamp(self, u):
        """Project a proposed 6-command onto the frozen envelope; fail-closed to
        the low-corner HOLD on non-finite / wrong-shape input."""
        u = np.asarray(u, float).ravel()
        if u.shape != (6,) or not np.all(np.isfinite(u)):
            return _LO.copy(), False           # fail-closed HOLD
        safe = np.clip(u, _LO, _HI)
        # Greenwald density limit n_e <= Ip/pi (Ip in MA, n_e in 1e20 m^-3)
        safe[3] = min(safe[3], safe[5] / np.pi)
        return safe, True

    def in_domain(self, u):
        u = np.asarray(u, float).ravel()
        if u.shape != (6,) or not np.all(np.isfinite(u)):
            return False
        safe, _ = self.clamp(u)
        return bool(np.allclose(safe, u, atol=1e-9))

    def _predict(self, x):
        safe, ok = self.clamp(x)
        return Prediction(safe, None, self.in_domain(x),
                          note="clamped to envelope" if not self.in_domain(x) else "in envelope")

    def benchmark(self):
        """Reproduce the AC-25 headline: inject deliberate violations and confirm
        the clamp lets zero escape (a real, model-free numpy check)."""
        rng = np.random.default_rng(uq.SEED)
        n = 150_000
        # commands drawn wide (deliberately outside the envelope ~half the time)
        span = _HI - _LO
        u = _LO + rng.uniform(-0.5, 1.5, size=(n, 6)) * span
        # inject NaN garbage in a slice
        u[::7, rng.integers(0, 6)] = np.nan
        escapes = 0
        for i in range(n):
            safe, _ = self.clamp(u[i])
            if not (np.all(safe >= _LO - 1e-9) and np.all(safe <= _HI + 1e-9)
                    and safe[3] <= safe[5] / np.pi + 1e-9):
                escapes += 1
        return {"member": self.name, "steps": n, "escapes": escapes,
                "catch_rate": 1.0 - escapes / n,
                "note": "model-free clamp; safety independent of any model"}
