"""KODEX KBURN — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

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
