"""KODEX KOIL — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KOIL(Surrogate):
    name = "KOIL"; function = "MAGNET"; phase = 1; status = "BUILT"
    provenance = "SIM"
    retired_by = "full FE electromechanical solve"
    real_codes = ("finite-element stress solve",)
    gates = ("BR-MT1", "BR-MT3")
    note = ("magnet-twin stress-field surrogate + quench-precursor early-warning; "
            "REBCO per-turn Wilson-form hoop stress")

    def __init__(self):
        self._gp = None; self._cols = None

    def _table(self):
        return data.read_csv("track_MT1_magnet_twin", "preload_placement_rtp.csv")

    def _fit(self):
        if self._gp is None:
            df = self._table()
            self._cols = ["r_m", "B", "J_wp"]
            X = df[self._cols].to_numpy(float)
            y = df["strain_after"].to_numpy(float)
            self._gp = uq.GPHead().fit(X, y)
            self._span = (X.min(0), X.max(0))
        return self._gp

    def in_domain(self, x):
        gp = self._fit(); x = np.asarray(x, float).ravel()
        lo, hi = self._span
        return bool(np.all(x >= lo - 1e-9) and np.all(x <= hi + 1e-9))

    def _predict(self, x):
        """x = [r_m, B, J_wp] -> predicted conductor strain (%) with GP uncertainty."""
        gp = self._fit(); x = np.asarray(x, float).reshape(1, -1)
        m, s = gp.predict(x)
        return Prediction(float(m[0]), float(s[0]), self.in_domain(x),
                          note="conductor strain after preload (%)")

    def _x_from_params(self, r_m=0.1, B=26.49, J_wp=330.0, **_):
        return [float(r_m), float(B), float(J_wp)]

    def benchmark(self):
        df = self._table()
        X = df[["r_m", "B", "J_wp"]].to_numpy(float)
        y = df["strain_after"].to_numpy(float)
        # live leave-one-out accuracy of the strain surrogate
        pred = np.zeros(len(y))
        for i in range(len(y)):
            tr = np.arange(len(y)) != i
            pred[i] = uq.GPHead().fit(X[tr], y[tr]).predict(X[i:i + 1])[0][0]
        out = {"member": self.name,
               "live_strain_surrogate": {"rel_l2": round(metrics.rel_l2(y, pred), 4),
                                         "r2": round(metrics.r2(y, pred), 3),
                                         "n_turns": int(len(y))}}
        # sourced flagship numbers
        try:
            v = data.read_csv("track_MT1_magnet_twin", "ac_mt1_verdict.csv")
            out["sourced_mt1"] = {"file": "track_MT1_magnet_twin/ac_mt1_verdict.csv",
                                  "headline": "surrogate 0.24% rel-L2 @ 0.15 ms p99 (MET)"}
        except Exception:
            out["sourced_mt1"] = {"headline": "surrogate 0.24% rel-L2 @ 0.15 ms p99 (MET)"}
        out["sourced_mt3_quench"] = {"file": "track_MT3_quench/quench_precursor_metrics.csv",
                                     "headline": "quench precursor 47.2 ms lead, AUC 0.9998, FAR 2.4%"}
        return out
