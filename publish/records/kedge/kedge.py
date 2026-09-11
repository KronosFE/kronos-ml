"""KODEX KEDGE — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KEDGE(Surrogate):
    name = "KEDGE"; function = "edge-transport"; phase = 3; status = "BUILT"
    provenance = "SIM"
    retired_by = "SOLPS-ITER / EIRENE edge campaign"
    real_codes = ("divertor target-thermal scan",)
    gates = ("H9",)
    note = ("divertor / edge heat-flux -> target-thermal surrogate — maps divertor heat flux to "
            "target surface temperature + material limits (W / CuCrZr) from the H9 exhaust scan. "
            "Reduced 0-D thermal model; SOLPS-ITER/EIRENE = fidelity upgrade")

    def _fit(self):
        if getattr(self, "_coef", None) is not None:
            return
        import pandas as pd
        from sklearn.metrics import r2_score
        df = pd.read_csv(_DIV_THERMAL).dropna(subset=["q_MWm2", "T_w_surf_C"])
        q = df["q_MWm2"].to_numpy(float); T = df["T_w_surf_C"].to_numpy(float)
        self._coef = np.polyfit(q, T, 2)
        self._r2 = float(r2_score(T, np.polyval(self._coef, q))); self._n = len(q)
        bad = df[df["ok_CuCrZr"] == False] if "ok_CuCrZr" in df else df.iloc[0:0]
        self._q_limit = float(bad["q_MWm2"].min()) if len(bad) else float(q.max())

    def _predict(self, x):
        self._fit()
        q = float(np.ravel(np.asarray(x, float))[0])
        return Prediction(float(np.polyval(self._coef, q)), None, q <= self._q_limit,
                          note=f"target surface temp (C) at q={q} MW/m^2; in_domain = under CuCrZr limit")

    def benchmark(self):
        self._fit()
        return {"member": self.name,
                "live_divertor_thermal": {
                    "source": "h9_target_thermal.csv (H9 exhaust/divertor engineering scan)",
                    "map": "divertor heat flux q [MW/m^2] -> target surface temperature [C]",
                    "r2_fit": round(self._r2, 3), "n_samples": self._n,
                    "max_safe_q_MWm2_CuCrZr": self._q_limit,
                    "verdict": (f"Divertor target-thermal surrogate: q->T_surf fit R2={self._r2:.3f} over "
                                f"{self._n} points; CuCrZr material limit at q~{self._q_limit} MW/m2. Reduced "
                                f"0-D thermal model — SOLPS-ITER/EIRENE edge campaign is the fidelity upgrade.")},
                "caveat": "0-D target-thermal scan, not a full 2-D edge-transport solve"}
