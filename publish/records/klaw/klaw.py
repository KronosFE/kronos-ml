"""KODEX KLAW — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KLAW(Surrogate):
    name = "KLAW"; function = "eqn-discovery"; phase = 3; status = "BUILT"
    provenance = "CGYRO-urep"
    retired_by = "first-principles reduced-transport theory"
    real_codes = ("sparse regression / SINDy",)
    note = ("equation discovery — sparse regression finds a compact closure for "
            "log10(Q_tot) over the CGYRO map (mu=400)")

    def __init__(self):
        self._m = None; self._sc = None; self._terms = None

    def _fit(self):
        if self._m is None:
            from sklearn.linear_model import LassoCV
            from sklearn.preprocessing import StandardScaler
            df = data.cgyro_flux_map_final()
            X = df[["a_LT", "shear"]].to_numpy(float)
            y = np.log10(np.clip(df.Q_tot.to_numpy(float), 0, None) + 1e-4)
            F, names = _library(X)
            self._sc = StandardScaler().fit(F)
            self._m = LassoCV(cv=5, random_state=uq.SEED, max_iter=100000).fit(self._sc.transform(F), y)
            self._terms = [(names[i], round(float(c), 3))
                           for i, c in enumerate(self._m.coef_) if abs(c) > 1e-6]
        return self._m

    def _predict(self, x):
        self._fit(); x = np.asarray(x, float).reshape(1, -1)
        F, _ = _library(x)
        log_q = float(self._m.predict(self._sc.transform(F))[0])
        q = max(10 ** log_q - 1e-4, 0.0)
        return Prediction(q, None, bool(2.0 <= x[0, 0] <= 3.5 and 0.4 <= x[0, 1] <= 1.6),
                          note="Q_tot from the discovered closure")

    def _x_from_params(self, a_LT=3.0, shear=1.0, **_):
        return [float(a_LT), float(shear)]

    def benchmark(self):
        m = self._fit()
        df = data.cgyro_flux_map_final()
        X = df[["a_LT", "shear"]].to_numpy(float)
        y = np.log10(np.clip(df.Q_tot.to_numpy(float), 0, None) + 1e-4)
        F, _ = _library(X)
        pred = m.predict(self._sc.transform(F))
        return {"member": self.name,
                "live": {"r2_fit": round(metrics.r2(y, pred), 3),
                         "n_terms_selected": len(self._terms),
                         "discovered_terms": self._terms},
                "note": "compact closure discovered from the 16/16 CGYRO map"}
