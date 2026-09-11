"""KODEX KECON — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KECON(Surrogate):
    """Generic, open fusion techno-economics — FINANCIAL FIREWALL.

    A community LCOE / cost-of-electricity calculator that runs ONLY on the
    caller's inputs.  It ships with ZERO Kronos cost/price/per-unit/valuation/
    funding data, and refuses to run without user-supplied inputs.  Kronos's own
    economics stay confidential and never enter this code."""
    name = "KECON"; function = "techno-economics"; phase = 3; status = "BUILT"
    provenance = "n/a"
    retired_by = "detailed engineering cost model (never public)"
    real_codes = ("open LCOE methodology",)
    note = ("GENERIC open techno-economics (LCOE) on the USER's inputs — "
            "FINANCIAL FIREWALL: no Kronos numbers, ever")

    #: hard firewall — no Kronos-specific defaults may ever be added here
    KRONOS_NUMBERS = None

    def _predict(self, x):
        """x = dict with the USER's own inputs:
            capex, opex_per_yr, annual_generation_MWh, discount_rate, lifetime_yr
        Returns generic LCOE ($/MWh).  Refuses to run without user inputs."""
        if not isinstance(x, dict) or not x:
            raise ValueError(
                "KECON requires the caller's own inputs (capex, opex_per_yr, "
                "annual_generation_MWh, discount_rate, lifetime_yr). "
                "It carries NO Kronos financial data.")
        capex = float(x["capex"]); opex = float(x["opex_per_yr"])
        gen = float(x["annual_generation_MWh"]); r = float(x["discount_rate"])
        n = int(x["lifetime_yr"])
        crf = r * (1 + r) ** n / ((1 + r) ** n - 1) if r > 0 else 1.0 / n
        cap_component = capex * crf / gen
        opex_component = opex / gen
        lcoe = cap_component + opex_component
        return Prediction({"lcoe_per_MWh": round(lcoe, 2),
                           "capital_component_per_MWh": round(cap_component, 2),
                           "opex_component_per_MWh": round(opex_component, 2),
                           "capital_recovery_factor": round(crf, 4)},
                          None, True,
                          note="generic LCOE $/MWh + breakdown from USER inputs (no Kronos data)")

    def benchmark(self):
        # a worked generic example (illustrative inputs — NOT Kronos numbers)
        demo = self.predict({"capex": 5e9, "opex_per_yr": 1e8,
                             "annual_generation_MWh": 3e6, "discount_rate": 0.07, "lifetime_yr": 30})
        return {"member": self.name,
                "live_worked_example": demo.y,
                "firewall": "generic community calculator; NO Kronos cost/price/valuation/funding, ever",
                "note": "founder-approved public, generic-only; inputs above are illustrative placeholders"}
