"""KODEX KHALO — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KHALO(Surrogate):
    name = "KHALO"; function = "UQ"; phase = 1; status = "BUILT"
    provenance = "SIM"
    retired_by = "full ensemble UQ / conformal prediction"
    real_codes = ("deep-ensemble", "conformal")
    gates = ("KX-L1-A14", "KX-L1-A5")
    note = ("shared calibrated-uncertainty layer (GP posterior + deep-ensemble) "
            "+ the canonical reliability report")

    def head(self, baseline=None):
        return uq.GPHead(baseline=baseline)

    def ensemble(self):
        return uq.EnsembleHead()

    def calibrate(self, y_true, mean, std):
        return uq.calibrate(y_true, mean, std)

    def _predict(self, x):
        """x = (mean, std): stamp a calibrated Prediction (the layer's job is to
        attach uncertainty to another surrogate's point estimate)."""
        mean, std = x
        return Prediction(float(np.asarray(mean).ravel()[0]),
                          float(np.asarray(std).ravel()[0]), True,
                          note="calibrated via KHALO")

    def benchmark(self):
        out = {"member": self.name}
        # (1) sourced: the real on-disk UQ calibration
        try:
            kv = _kv(data.read_csv("track6_surrogate_uq", "uq_calibration_summary.csv"))
            out["sourced"] = {"file": "track6_surrogate_uq/uq_calibration_summary.csv",
                              "ece": kv.get("ECE_mean_std"),
                              "coverage_90": kv.get("coverage90_mean_std"),
                              "result": kv.get("AC28_RESULT")}
        except Exception as e:
            out["sourced"] = {"error": str(e)}
        # (2) live: reliability of a GP on the CGYRO clean subset (leave-one-out)
        try:
            df = data.cgyro_flux_map()
            cl = df[(df.saturated == 1) & np.isfinite(df.Q_i)].reset_index(drop=True)
            X = cl[["a_LT", "shear"]].to_numpy(float)
            y = np.log(np.clip(cl.Q_i.to_numpy(float), 1e-6, None))
            mu = np.zeros(len(y)); sd = np.zeros(len(y))
            for i in range(len(y)):
                tr = np.arange(len(y)) != i
                if tr.sum() < 3:
                    continue
                h = uq.GPHead().fit(X[tr], y[tr])
                m, s = h.predict(X[i:i + 1]); mu[i] = m[0]; sd[i] = s[0]
            rep = metrics.calibration_report(y, mu, np.clip(sd, 1e-6, None))
            out["live_gp_on_cgyro"] = {"ece": round(rep["ece"], 4),
                                       "coverage_90": round(rep["coverage_90"], 4),
                                       "n": int(len(y))}
        except Exception as e:
            out["live_gp_on_cgyro"] = {"error": str(e)}
        return out
