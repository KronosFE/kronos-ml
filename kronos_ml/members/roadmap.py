"""Fleet extensions (Phase 2, partial data) and roadmap codes (Phase 3).

Phase-2 codes have on-disk data (traces / physics engine) but no trained ML
surrogate yet — status PARTIAL, predict() raises with a clear pointer.
Phase-3 codes are ROADMAP placeholders.  KECON is the one exception: a real,
generic, firewalled techno-economics calculator (no Kronos numbers, ever).
"""
from __future__ import annotations

from .. import register
from ..base import Surrogate, Prediction


def _roadmap(cls_name, brand, func, phase, status, note, retired_by, gates=(), source=""):
    def _pred(self, x):
        raise NotImplementedError(
            f"{brand} is {status} — {note}. "
            f"{'Data/engine on disk: ' + source if source else 'No ML data on disk yet.'}")
    def _bench(self):
        return {"member": brand, "status": status,
                "sourced": source or "n/a", "note": note}
    return register(type(cls_name, (Surrogate,), {
        "name": brand, "function": func, "phase": phase, "status": status,
        "note": note, "retired_by": retired_by, "gates": tuple(gates),
        "provenance": "SIM" if status == "PARTIAL" else "n/a",
        "_predict": _pred, "benchmark": _bench,
    }))


# ---- Phase 2 (partial data; ML surrogate next) ---------------------------
# KBURN, KISO, KPATH are real surrogates now — see members/plant.py
# KBREED, KFLUX are real surrogates now — see members/neutronics.py
_roadmap("KHeat", "KHEAT", "heating&CD", 2, "PARTIAL",
         "RF / NBI heating & current-drive actuator-response surrogate",
         "RF/NBI physics codes", source="DEC-dispatch adjacency")

# ---- Phase 3 (roadmap; no ML data yet) -----------------------------------
# KFORGE, KPILOT are real capstones now — see members/capstone.py
# KDRIVE, KLAW, KGEN, KFUEL are real surrogates now — see members/advanced.py
# Only genuinely-gated codes remain ROADMAP: KEDGE + KRAD (need the SOLPS edge campaign)
_roadmap("KEdge", "KEDGE", "edge-transport", 3, "ROADMAP",
         "divertor / SOL / edge heat-flux transport surrogate",
         "SOLPS-ITER / EIRENE", source="needs [C]->[A] SOLPS-ITER/EIRENE edge campaign")
_roadmap("KRad", "KRAD", "radiation-control", 3, "ROADMAP",
         "impurity seeding / radiation / divertor-detachment control",
         "radiation transport + control", source="fold into KEDGE edge campaign")


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
