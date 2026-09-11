"""KODEX KYRO — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KYRO(Surrogate):
    name = "KYRO"; function = "TRANSPORT"; phase = 1; status = "BUILT"
    provenance = "CGYRO-urep"
    retired_by = "CGYRO (nonlinear gyrokinetic)"
    real_codes = ("cgyro", "gene", "gs2", "tglf")
    gates = ("BR-L2-A12", "BR-L2-A1e", "BR-L2-A1c")
    note = ("CGYRO turbulence-transport surrogate over the complete 16/16 A1e map "
            "(total heat flux Q_tot + turbulent/quiet); mu=400 rep, real-mass gold deferred")

    def __init__(self):
        self._gp = None

    def _map(self):
        df = data.cgyro_flux_map_final()
        X = df[["a_LT", "shear"]].to_numpy(float)
        y = np.log10(np.clip(df.Q_tot.to_numpy(float), 0, None) + 1e-4)
        return X, y, df

    def _fit(self):
        if self._gp is None:
            X, y, _ = self._map()
            self._gp = uq.GPHead().fit(X, y)
        return self._gp

    def in_domain(self, x):
        x = np.asarray(x, float).ravel()
        return bool(_ALT[0] <= x[0] <= _ALT[1] and _SHEAR[0] <= x[1] <= _SHEAR[1])

    def _predict(self, x):
        """x = [a_LT, shear] -> total gyro-Bohm heat flux Q_tot, with GP band and
        a turbulent/quiet verdict."""
        gp = self._fit()
        x = np.asarray(x, float).reshape(1, -1)
        m, s = gp.predict(x)
        qtot = max(float(10 ** m[0] - 1e-4), 0.0)
        unc = float(qtot * np.log(10) * s[0])          # propagate log10 std
        verdict = "turbulent" if qtot > _QUIET_THRESHOLD else "quiet"
        return Prediction(qtot, unc, self.in_domain(x),
                          note=f"Q_tot={qtot:.3g} gyroBohm ({verdict})")

    def _x_from_params(self, a_LT=3.0, shear=1.0, **_):
        return [float(a_LT), float(shear)]

    def benchmark(self):
        X, y, df = self._map()
        n = len(y)
        mu = np.zeros(n); sd = np.zeros(n)
        for i in range(n):
            tr = np.arange(n) != i
            h = uq.GPHead().fit(X[tr], y[tr])
            m, s = h.predict(X[i:i + 1]); mu[i], sd[i] = m[0], s[0]
        # turbulent/quiet classification from the leave-one-out prediction
        pred_qtot = np.clip(10 ** mu - 1e-4, 0, None)
        pred_verd = np.where(pred_qtot > _QUIET_THRESHOLD, "turbulent", "quiet")
        class_ok = int(np.sum(pred_verd == df.verdict.to_numpy()))
        acc = {"r2_log_Qtot": round(metrics.r2(y, mu), 3),
               "rmse_log_Qtot": round(metrics.rmse(y, mu), 3),
               "coverage_90": round(metrics.coverage_at_level(y, mu, np.clip(sd, 1e-6, None), 0.90), 3),
               "turbulent_quiet_LOO": f"{class_ok}/{n} correct",
               "map": f"{int((df.verdict=='turbulent').sum())} turbulent / "
                      f"{int((df.verdict=='quiet').sum())} quiet (16/16 complete)"}
        gp = self._fit()
        t0 = time.perf_counter()
        for _ in range(200):
            gp.predict(np.array([[3.0, 1.0]]))
        ms = (time.perf_counter() - t0) / 200 * 1e3
        speedup = _CGYRO_GPU_H_PER_POINT * 3600.0 / (ms / 1e3)
        speed = {"surrogate_ms_per_point": round(ms, 3),
                 "cgyro_gpu_h_per_point_measured": _CGYRO_GPU_H_PER_POINT,
                 "speedup_x_vs_cgyro": f"{speedup:.1e}",
                 "speedup_note": "ms inference vs GPU-hours for the CONVERGED flux value",
                 "fidelity": "representative mu=400; real-mass gold deferred"}
        return {"member": self.name, "accuracy": acc, "speed": speed,
                "provenance": "COMPLETE 16/16 CGYRO A1e map (mu=400); operating point subcritical"}
