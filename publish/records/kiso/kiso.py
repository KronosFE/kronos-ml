"""KODEX KISO — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

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
