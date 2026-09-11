"""KODEX KSEEK — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KSEEK(Surrogate):
    name = "KSEEK"; function = "ACTIVE"; phase = 3; status = "BUILT"
    provenance = "SIM"
    retired_by = "the full CGYRO parameter scan (once every point is simulated)"
    real_codes = ("Gaussian-process active learning", "max-variance acquisition")
    gates = ("BR-L2-A1e",)
    note = ("active-learning acquisition — proposes the NEXT most-informative CGYRO run from "
            "KYRO's GP posterior (max-variance / uncertainty sampling), so expensive GPU-hours "
            "go where the surrogate is least sure. A real experimental-design tool")

    _BOX = ((2.0, 3.5), (0.4, 1.6))          # a_LT, shear in-domain box (matches KYRO)

    def _grid(self, n=41):
        a = np.linspace(*self._BOX[0], n); s = np.linspace(*self._BOX[1], n)
        A, S = np.meshgrid(a, s)
        return np.c_[A.ravel(), S.ravel()]

    def _model(self):
        import kronos_ml.data as data
        gp = get("KYRO")._fit()               # reuse the exact GP the fleet queries
        df = data.cgyro_flux_map_final()
        return gp, df[["a_LT", "shear"]].to_numpy(float), \
            np.log10(np.clip(df.Q_tot.to_numpy(float), 0, None) + 1e-4)

    def _acquire(self, k=3, min_sep=0.16):
        gp, Xtr, _ = self._model()
        G = self._grid()
        _, std = gp.predict(G)
        std = np.asarray(std).ravel()
        span = np.array([self._BOX[0][1] - self._BOX[0][0], self._BOX[1][1] - self._BOX[1][0]])
        picks, chosen = [], []
        for idx in np.argsort(-std):
            g = G[idx]
            d_tr = np.linalg.norm((g - Xtr) / span, axis=1).min()
            d_ch = min([np.linalg.norm((g - c) / span) for c in chosen], default=9.0)
            if min(d_tr, d_ch) < min_sep:
                continue
            chosen.append(g); picks.append({"a_LT": round(float(g[0]), 3),
                                            "shear": round(float(g[1]), 3),
                                            "gp_std": round(float(std[idx]), 4)})
            if len(picks) >= k:
                break
        return picks

    def _loo(self):
        """Leave-one-out: does GP posterior std actually predict where the model is wrong?"""
        from .. import uq
        _, X, y = self._model()
        errs, stds = [], []
        for i in range(len(y)):
            m = np.ones(len(y), bool); m[i] = False
            g = uq.GPHead().fit(X[m], y[m])
            mu, sd = g.predict(X[i:i + 1])
            errs.append(abs(float(np.ravel(mu)[0]) - y[i])); stds.append(float(np.ravel(sd)[0]))
        errs, stds = np.array(errs), np.array(stds)
        if errs.std() < 1e-9 or stds.std() < 1e-9:
            return None
        return float(np.corrcoef(stds, errs)[0, 1])

    def _predict(self, x):
        p = self._acquire(k=1)[0]
        return Prediction(p, None, True, note="next most-informative CGYRO operating point (max GP variance)")

    def benchmark(self):
        picks = self._acquire(k=3)
        corr = self._loo()
        return {"member": self.name,
                "live_active_learning": {
                    "source": "KYRO GP over the completed 16-pt CGYRO A1e map",
                    "acquisition": "max GP posterior std (uncertainty sampling), min-separation filtered",
                    "next_runs": picks,
                    "loo_std_vs_error_corr": None if corr is None else round(corr, 3),
                    "cgyro_gpu_h_per_point": 2.89,
                    "verdict": (f"Proposes the next CGYRO run at a/L_T={picks[0]['a_LT']}, "
                                f"shear={picks[0]['shear']} (highest GP uncertainty). Leave-one-out: GP "
                                f"posterior-std vs actual error correlation = "
                                f"{'n/a' if corr is None else round(corr, 3)} — "
                                f"{'the acquisition targets real model error' if (corr or 0) > 0.2 else 'weak on this small map'}. "
                                f"A real experimental-design tool: spend ~2.9 GPU-h/point where it matters.")},
                "caveat": "16 training points is small — acquisition is directional guidance, not a guarantee"}
