"""The capstones — they only work because the rest of the fleet exists.

KFORGE — inverse-design optimizer: searches a design space by calling the fleet
         surrogates as fast evaluators (differential evolution over KYRO + KORE).
KPILOT — fleet orchestration layer: routes a plain-language query to the right
         surrogate(s) and runs them (the tool-use scaffolding an LLM agent needs;
         the LLM reasoning layer is the roadmap upgrade).
"""
from __future__ import annotations

import numpy as np

from .. import register, get, list_surrogates
from ..base import Surrogate, Prediction
from .. import uq


@register
class KFORGE(Surrogate):
    name = "KFORGE"; function = "inverse-design"; phase = 3; status = "BUILT"
    provenance = "CGYRO-urep"
    retired_by = "full integrated design optimization"
    real_codes = ("differential evolution over the fleet",)
    note = ("inverse-design capstone — chains the fleet surrogates (KYRO transport "
            "+ KORE equilibrium feasibility) to search machine designs; the underlying "
            "KYRO transport is CGYRO mu=400 representative-mass (real-mass converged gold "
            "deferred), so the operating point is representative, not final")

    # design vector = [a_LT, shear, A, kappa, delta]
    BOUNDS = [(2.0, 3.5), (0.4, 1.6), (0.8, 1.2), (1.5, 2.5), (-0.5, 0.5)]

    def _cost(self, d):
        a_LT, shear, A, kappa, delta = d
        q = get("KYRO").predict([a_LT, shear]).y            # chain -> transport surrogate
        feasible = get("KORE").in_domain([A, kappa, delta]) # chain -> equilibrium gate
        penalty = 0.0 if feasible else 10.0
        return np.log10(q + 1e-4) + penalty                 # minimise transport, keep feasible

    def optimize(self, maxiter=15):
        from scipy.optimize import differential_evolution
        res = differential_evolution(self._cost, self.BOUNDS, seed=uq.SEED,
                                     maxiter=maxiter, tol=1e-3, polish=True)
        d = res.x
        return {"a_LT": round(d[0], 3), "shear": round(d[1], 3),
                "A": round(d[2], 3), "kappa": round(d[3], 3), "delta": round(d[4], 3),
                "predicted_Q_tot": round(float(get("KYRO").predict([d[0], d[1]]).y), 4),
                "surrogates_chained": ["KYRO", "KORE"], "objective": "min transport (feasible)"}

    def _predict(self, x):
        """x = optional dict of overrides -> the optimized design."""
        best = self.optimize()
        return Prediction(best, None, True,
                          note="inverse-designed operating point (quiet/low-transport)")

    def benchmark(self):
        best = self.optimize()
        return {"member": self.name, "live_inverse_design": best,
                "note": "differential evolution chained KYRO+KORE to find a low-transport, "
                        "feasible design — the capstone that requires the fleet"}


@register
class KPILOT(Surrogate):
    name = "KPILOT"; function = "agentic-AI"; phase = 3; status = "BUILT"
    provenance = "n/a"
    retired_by = "full LLM agent runtime over the fleet"
    real_codes = ("fleet router / tool dispatcher",)
    note = ("fleet orchestration layer — routes a plain-language query to the right "
            "KODEX code and runs it; the LLM reasoning layer is the roadmap upgrade")

    ROUTES = [
        (("transport", "heat flux", "turbulence", "gradient", "q_i", "cgyro"), "KYRO"),
        (("disruption", "stability", "warning", "quench-plasma"), "KWARD"),
        (("safe", "clamp", "envelope", "in domain", "abstain"), "KGATE"),
        (("equilibrium", "flux surface", "shape", "grad-shafranov"), "KORE"),
        (("magnet", "stress", "quench", "strain", "coil"), "KOIL"),
        (("material", "alloy", "hea"), "KMAT"),
        (("uncertainty", "calibration", "error bar"), "KHALO"),
        (("quantum",), "KQUBIT"),
        (("cost", "economics", "lcoe", "electricity"), "KECON"),
        (("design", "optimize", "inverse"), "KFORGE"),
        (("control", "setpoint", "command"), "KAIROS"),
    ]

    def route(self, query: str):
        q = str(query).lower()
        for keys, code in self.ROUTES:
            if any(k in q for k in keys):
                return code
        return None

    def _predict(self, x):
        """x = a plain-language query string -> which code answers it (+ a run when
        the query carries the numbers)."""
        code = self.route(x)
        if code is None:
            return Prediction({"answer": "no single code matches; fleet: " + ", ".join(list_surrogates())},
                              None, False, note="unrouted query")
        import re
        nums = [float(t) for t in re.findall(r"-?\d+\.?\d*", str(x))]
        result = None
        try:
            if code == "KYRO" and len(nums) >= 2:
                p = get("KYRO").predict(nums[:2]); result = {"Q_tot": round(p.y, 3), "in_domain": p.in_domain}
        except Exception:
            result = None
        return Prediction({"routed_to": code, "result": result}, None, True,
                          note=f"KPILOT routed the query to {code}")

    def benchmark(self):
        examples = {
            "what's the heat flux at gradient 3.5 shear 1.6?": None,
            "is this command inside the safe envelope?": None,
            "estimate the cost of electricity": None,
            "optimize a machine design": None,
            "which alloy should we use?": None,
        }
        routed = {q: self.route(q) for q in examples}
        return {"member": self.name, "live_routing": routed,
                "note": "keyword tool-router over the fleet; a full LLM agent is the roadmap upgrade"}


def _isnum(t):
    try:
        float(t); return True
    except Exception:
        return False
