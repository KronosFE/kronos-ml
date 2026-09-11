"""KODEX KHEAT — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KHEAT(Surrogate):
    name = "KHEAT"; function = "heating&CD"; phase = 2; status = "BUILT"
    provenance = "SIM"
    retired_by = "full RF/NBI ray-tracing (GENRAY/TORAY/NUBEAM)"
    real_codes = ("current-drive design scan",)
    gates = ("H10",)
    note = ("heating & current-drive actuator-response surrogate — fast RandomForest over a "
            "3888-point current-drive design scan; predicts driven current I_cd from the RF/NBI "
            "drive parameters + plasma state. Reduced CD model; ray-tracing = fidelity upgrade")
    _FEATS = ["gamma_cd", "P_cd", "ne", "R0", "B0", "Ti0", "beta_N"]

    def _fit(self):
        if getattr(self, "_m", None) is not None:
            return self._m
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        from .. import uq
        d = pd.read_csv(_CD_SCAN)[self._FEATS + ["I_cd"]].dropna()
        X = d[self._FEATS].to_numpy(float); y = d["I_cd"].to_numpy(float)
        rng = np.random.default_rng(uq.SEED); p = rng.permutation(len(y)); X, y = X[p], y[p]
        ntr = int(0.7 * len(y))
        m = RandomForestRegressor(n_estimators=200, random_state=0).fit(X[:ntr], y[:ntr])
        self._r2 = float(r2_score(y[ntr:], m.predict(X[ntr:]))); self._n = len(y)
        self._m = m
        return m

    def _predict(self, x):
        m = self._fit()
        return Prediction(float(m.predict(np.atleast_2d(np.asarray(x, float)))[0]), None, True,
                          note="driven current I_cd (MA) from [gamma_cd,P_cd,ne,R0,B0,Ti0,beta_N]")

    def benchmark(self):
        self._fit()
        return {"member": self.name,
                "live_cd_surrogate": {
                    "source": "d1_cd_search.csv (3888-point current-drive design scan)",
                    "features": self._FEATS, "target": "I_cd (driven current, MA)",
                    "r2_vs_scan": round(self._r2, 3), "n_samples": self._n,
                    "verdict": (f"Fast surrogate of the current-drive scan: predicts driven current from "
                                f"RF/NBI drive + plasma params, R2={self._r2:.3f} ({self._n} configs). Reduced "
                                f"CD model — GENRAY/TORAY/NUBEAM ray-tracing is the fidelity upgrade.")},
                "caveat": "engineering CD scan (not full ray-tracing); feeds KAIROS heating control"}
