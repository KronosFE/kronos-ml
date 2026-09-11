"""KODEX KPILOT — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

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
