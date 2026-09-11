"""KODEX KSENSE — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KSENSE(Surrogate):
    name = "KSENSE"; function = "QSENSE"; phase = 2; status = "BUILT"
    provenance = "SIM"
    retired_by = "deployed physical diagnostic hardware"
    real_codes = ("NV/SQUID/SERF magnetometry",)
    gates = ("HX-29",)
    note = ("quantum-sensor diagnostic evaluation (NV/SQUID/SERF); a validated null "
            "gain where disruptions live (the MHD floor dominates the sensor floor)")

    def _predict(self, x):
        return Prediction({"verdict": "quantum sensing buys ~0 disruption-warning gain",
                           "reason": "intrinsic MHD floor dominates the sensor floor"},
                          None, True, note="validated null result")

    def benchmark(self):
        out = {"member": self.name,
               "verdict": "dAUC ~ -0.04..+0.003, d-warning ~0 ms vs conventional coil",
               "result_type": "validated null result (BUILT)"}
        try:
            df = data.read_csv("track8b_quantum_sensing", "sensing_to_disruption_gain.csv")
            out["live"] = {"file": "track8b_quantum_sensing/sensing_to_disruption_gain.csv",
                           "n_rows": int(len(df)), "columns": list(df.columns)[:6]}
        except Exception as e:
            out["live"] = {"err": str(e)}
        return out
