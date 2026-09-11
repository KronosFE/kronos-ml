"""Plant / operational surrogates trained on the reduced-twin control traces.

KBURN  — burner dispatch: fuel mix -> captured power (GP).
KPATH  — operational controller: (state, setpoint) -> command (imitation of the MPC).
KISO   — isotope servo: (measured mix, target) -> valve puffs (imitation of the PI loop).

All learned from the on-disk twin traces (SIM/TWIN); honest twin scope.
"""
from __future__ import annotations

import numpy as np

from .. import register
from ..base import Surrogate, Prediction
from .. import uq, metrics, data


def _rf(seed=uq.SEED, n=200):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=n, random_state=seed, n_jobs=-1)


def _rf_predict_std(rf, X):
    """Point prediction + per-output std across the forest (model uncertainty)."""
    per_tree = np.stack([t.predict(X) for t in rf.estimators_])   # (n_trees, n, out)
    return per_tree.mean(0), per_tree.std(0)


@register
class KBURN(Surrogate):
    name = "KBURN"; function = "burner"; phase = 2; status = "BUILT"
    provenance = "TWIN"
    retired_by = "full plant systems model"
    real_codes = ("plant systems model",)
    note = ("burner dispatch surrogate — fuel mix -> captured power (Aegis/MetroVolt); "
            "GP over the DEC dispatch traces")

    def __init__(self):
        self._gp = None; self._span = None

    def _fit(self):
        if self._gp is None:
            df = data.read_csv("track5_control", "dec_dispatch_traces.csv")
            X = df[["mix_D", "mix_He3", "mix_p", "mix_Li6"]].to_numpy(float)
            y = df["P_captured_MW"].to_numpy(float)
            rng = np.random.default_rng(uq.SEED)
            idx = rng.permutation(len(y))[:250]           # GP subsample
            self._gp = uq.GPHead().fit(X[idx], y[idx])
            self._span = (X.min(0), X.max(0)); self._Xte, self._yte = X, y
        return self._gp

    def _predict(self, x):
        gp = self._fit(); x = np.asarray(x, float).reshape(1, -1)
        m, s = gp.predict(x)
        lo, hi = self._span
        return Prediction(float(m[0]), float(s[0]),
                          bool(np.all(x.ravel() >= lo - 1e-6) and np.all(x.ravel() <= hi + 1e-6)),
                          note="captured power P (MW), gyro-twin dispatch")

    def _x_from_params(self, mix_D=0.5, mix_He3=0.3, mix_p=0.1, mix_Li6=0.1, **_):
        return [mix_D, mix_He3, mix_p, mix_Li6]

    def benchmark(self):
        gp = self._fit()
        rng = np.random.default_rng(uq.SEED + 2)
        te = rng.permutation(len(self._yte))[:400]
        m, s = gp.predict(self._Xte[te])
        return {"member": self.name,
                "live": {"r2": round(metrics.r2(self._yte[te], m), 3),
                         "rmse_MW": round(metrics.rmse(self._yte[te], m), 3),
                         "coverage_90": round(metrics.coverage_at_level(self._yte[te], m, np.clip(s, 1e-6, None), 0.90), 3),
                         "n_test": int(len(te))},
                "sourced": "track5_control/dec_dispatch_summary.csv (dynamic reweight +22% vs static)"}


@register
class KPATH(Surrogate):
    name = "KPATH"; function = "operational"; phase = 2; status = "BUILT"
    provenance = "TWIN"
    retired_by = "operational scenario optimizer (device-validated)"
    real_codes = ("MPC scenario optimizer",)
    note = ("operational controller surrogate — (state, setpoint) -> command; "
            "imitation of the MPC over the closed-loop control traces")

    _XCOLS = [f"x_{q}" for q in ("kappa", "delta", "beta_N", "n_e", "eps", "Ip")]
    _RCOLS = [f"ref_{q}" for q in ("kappa", "delta", "beta_N", "n_e", "eps", "Ip")]
    _UCOLS = [f"u_prop_{q}" for q in ("kappa", "delta", "beta_N", "n_e", "eps", "Ip")]

    def __init__(self):
        self._rf = None

    def _fit(self):
        if self._rf is None:
            df = data.read_csv("track5_control", "control_traces.csv")
            df = df[df.is_inject == 0]                    # clean control law only
            X = df[self._XCOLS + self._RCOLS].to_numpy(float)
            Y = df[self._UCOLS].to_numpy(float)
            rng = np.random.default_rng(uq.SEED); idx = rng.permutation(len(Y))
            X, Y = X[idx], Y[idx]                         # random split (not temporal)
            n = len(Y); ntr = int(0.8 * n)
            self._rf = _rf().fit(X[:ntr], Y[:ntr])
            self._Xte, self._Yte = X[ntr:], Y[ntr:]
        return self._rf

    def _predict(self, x):
        """x = (state6, ref6) -> next command (6) + per-axis model std."""
        rf = self._fit()
        state, ref = x
        xin = np.r_[np.asarray(state, float).ravel(), np.asarray(ref, float).ravel()].reshape(1, -1)
        m, s = _rf_predict_std(rf, xin)
        return Prediction(m[0], s[0], True, note="predicted control command (6-axis)")

    def benchmark(self):
        rf = self._fit()
        m, _ = _rf_predict_std(rf, self._Xte)
        r2 = np.mean([metrics.r2(self._Yte[:, i], m[:, i]) for i in range(6)])
        return {"member": self.name,
                "live": {"r2_mean_over_6_axes": round(float(r2), 3),
                         "n_test": int(len(self._Yte)),
                         "note": "imitation of the MPC command law"},
                "sourced": "track5_control/tracking_error.csv (MPC RMS <5% on all 6)"}


@register
class KISO(Surrogate):
    name = "KISO"; function = "isotopes"; phase = 2; status = "BUILT"
    provenance = "SIM"
    retired_by = "activation-transport + separation model (full spectrum)"
    real_codes = ("isotope PI servo", "FENDL-3.2 activation")
    note = ("isotope code — (a) servo surrogate imitating the PI fuel-mix loop, and "
            "(b) medical-isotope YIELD from real FENDL-3.2 (n,x) cross-sections "
            "convolved with the neutron spectrum (Mo-99/Tc-99m flagship)")

    _THERMAL_MeV = 2.53e-8      # 0.0253 eV thermal point
    _FAST_MeV = 14.1           # D-T fusion peak
    _FCOLS = ["f_D", "f_He3", "f_p", "f_Li6"]
    _TCOLS = ["tgt_D", "tgt_He3", "tgt_p", "tgt_Li6"]
    _PCOLS = ["puff_D", "puff_He3", "puff_p", "puff_Li6"]

    def __init__(self):
        self._rf = None

    def _fit(self):
        if self._rf is None:
            df = data.read_csv("track5_control", "isotope_control_traces.csv")
            df = df[df.closed_loop == 1] if "closed_loop" in df else df
            X = df[self._FCOLS + self._TCOLS].to_numpy(float)
            Y = df[self._PCOLS].to_numpy(float)
            rng = np.random.default_rng(uq.SEED); idx = rng.permutation(len(Y))
            X, Y = X[idx], Y[idx]                         # random split (not temporal)
            n = len(Y); ntr = int(0.8 * n)
            self._rf = _rf().fit(X[:ntr], Y[:ntr])
            self._Xte, self._Yte = X[ntr:], Y[ntr:]
        return self._rf

    def _predict(self, x):
        """x = (measured4, target4) -> valve puffs (4)."""
        rf = self._fit()
        meas, tgt = x
        xin = np.r_[np.asarray(meas, float).ravel(), np.asarray(tgt, float).ravel()].reshape(1, -1)
        m, s = _rf_predict_std(rf, xin)
        return Prediction(m[0], s[0], True, note="predicted valve puffs (4 species)")

    def medical_yield(self, phi_thermal=1e14, phi_fast=1e14):
        """Per-isotope production reaction rate (per target atom per second) under a
        2-group fusion neutron spectrum: thermal (0.0253 eV) + 14.1 MeV D-T peak.
        R = (σ_th·φ_th + σ_14·φ_fast)·1e-24 cm²/barn.  Real FENDL-3.2 σ(E)."""
        xs = data.medical_xs()
        b_to_cm2 = 1e-24
        out = {}
        for prod, (E, sig, rxn) in xs.items():
            s_th = float(np.interp(self._THERMAL_MeV, E, sig))
            s_f = float(np.interp(self._FAST_MeV, E, sig))
            out[prod] = {"reaction": rxn, "sigma_thermal_b": round(s_th, 4),
                         "sigma_14MeV_b": round(s_f, 5),
                         "reaction_rate_per_atom_s": (s_th * phi_thermal + s_f * phi_fast) * b_to_cm2}
        return out

    def benchmark(self):
        rf = self._fit()
        m, _ = _rf_predict_std(rf, self._Xte)
        r2 = np.mean([metrics.r2(self._Yte[:, i], m[:, i]) for i in range(4)])
        out = {"member": self.name,
               "live_servo": {"r2_mean_over_4_species": round(float(r2), 3),
                              "n_test": int(len(self._Yte))},
               "sourced": "track5_control/isotope_control_summary.csv (closed-loop 13.6x vs open)"}
        try:
            y = self.medical_yield()
            ranked = sorted(y.items(), key=lambda kv: -kv[1]["reaction_rate_per_atom_s"])
            out["live_medical_yield"] = {
                "n_isotopes": len(y),
                "flagship_Mo-99": {k: y["Mo-99"][k] for k in ("sigma_thermal_b", "sigma_14MeV_b")}
                if "Mo-99" in y else None,
                "ranked_by_thermal_yield": [p for p, _ in ranked],
                "source": "FENDL-3.2 (n,x) cross-sections, 2-group fusion spectrum",
                "note": "full-spectrum convolution + separation chemistry = the fidelity upgrade"}
        except Exception as e:
            out["live_medical_yield"] = {"error": str(e), "note": "cross-sections not staged yet"}
        return out
