"""KODEX KFUEL — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KFUEL(Surrogate):
    name = "KFUEL"; function = "fuel-cycle"; phase = 3; status = "BUILT"
    provenance = "SIM"
    retired_by = "full fuel-cycle systems code (e.g. MIRC / TRANSAT)"
    real_codes = ("fuel-cycle systems model",)
    note = ("tritium fuel-cycle inventory — a reduced systems model "
            "(breeding, burn, decay, reserve) giving self-sufficiency + doubling time")

    T_HALFLIFE_YR = 12.32

    def _predict(self, x):
        """x = dict{TBR, burn_rate_kg_yr, reserve_kg, startup_kg} -> inventory outcome."""
        if not isinstance(x, dict):
            x = {}
        TBR = float(x.get("TBR", 1.05)); burn = float(x.get("burn_rate_kg_yr", 55.6))
        reserve = float(x.get("reserve_kg", 5.0)); start = float(x.get("startup_kg", 10.0))
        decay = np.log(2) / self.T_HALFLIFE_YR
        net_breed = burn * (TBR - 1.0) - decay * (start + reserve)   # kg/yr surplus
        self_sufficient = net_breed > 0
        doubling_yr = (start / net_breed) if net_breed > 1e-6 else float("inf")
        return Prediction({"net_breeding_kg_yr": round(net_breed, 3),
                           "self_sufficient": bool(self_sufficient),
                           "doubling_time_yr": round(doubling_yr, 2) if np.isfinite(doubling_yr) else None},
                          None, TBR >= 1.0, note="reduced tritium fuel-cycle balance")

    def benchmark(self):
        p = self.predict({"TBR": 1.05, "burn_rate_kg_yr": 55.6, "reserve_kg": 5.0, "startup_kg": 10.0})
        return {"member": self.name, "live_nominal": p.y,
                "note": "reduced systems model seeded from tritium physics (12.32 yr half-life); "
                        "full fuel-cycle code is the roadmap upgrade"}
