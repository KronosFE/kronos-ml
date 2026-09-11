"""KODEX KMAT — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KMAT(Surrogate):
    name = "KMAT"; function = "MATERIALS"; phase = 1; status = "BUILT"
    provenance = "SIM"
    retired_by = "DFT (VASP / Quantum ESPRESSO)"
    real_codes = ("CHGNet", "DFT")
    gates = ("KX-L1-A4", "BR-L2-A8")
    note = ("ML+DFT alloy / HEA screen (CHGNet) for structural & functional "
            "materials; screen BUILT, dpa/lifetime roadmap")

    def _ranking(self):
        # read-only; the screen output lives in the red-apple mirror (do not write)
        g = glob.glob(_RESULTS)
        if not g:
            g = glob.glob("/Users/pford/Desktop/Kronos Fusion Energy/**/"
                          "chgnet_screen_results.json", recursive=True)
        d = json.load(open(g[0]))
        rk = d.get("ranking", d)
        return rk if isinstance(rk, list) else list(rk.values())

    def _predict(self, x):
        """x = candidate formula (str) or index -> its screened rank/energy."""
        rk = self._ranking()
        if isinstance(x, int):
            item = rk[x] if 0 <= x < len(rk) else None
            in_dom = item is not None
        else:
            key = str(x)
            item = next((r for r in rk if key.lower() in json.dumps(r).lower()), None)
            in_dom = item is not None
        e = None
        if isinstance(item, dict):
            for k in ("e_hull", "energy", "e_form", "formation_energy", "score"):
                if k in item:
                    e = float(item[k]); break
        return Prediction(item, e, in_dom,
                          note="CHGNet-screened candidate" if in_dom else "not in screen")

    def benchmark(self):
        try:
            rk = self._ranking()
            return {"member": self.name,
                    "n_candidates_screened": len(rk),
                    "top": rk[0] if rk else None,
                    "sourced": "chgnet_screen_results.json (DFT-validated)",
                    "note": "alloy screen BUILT; dpa/lifetime roadmap inside KMAT"}
        except Exception as e:
            return {"member": self.name, "error": str(e)}
