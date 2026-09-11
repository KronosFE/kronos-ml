"""KODEX KPATH — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

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
