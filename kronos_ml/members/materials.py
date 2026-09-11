"""KMAT — ML+DFT alloy / high-entropy-alloy screen (CHGNet).

Reads the DFT-validated CHGNet screen ranking from disk and answers stability /
formation-energy queries.  One growing code: the screen is BUILT; component
dpa/lifetime is on the roadmap inside it.
"""
from __future__ import annotations

import glob
import json
import numpy as np

from .. import register
from ..base import Surrogate, Prediction

# prefer the LIVE tree (never the red-apple backup, which is read-only)
_RESULTS = ("/Users/pford/Desktop/Kronos Fusion Energy/01 - RESEARCH & DATA/"
            "01 - Research, Data & Codes/_LOCAL_DERISK_CAMPAIGN_2026-08-25/runs/"
            "KX-L1-A4__ml-alloy-screen-refresh/chgnet_screen_results.json")


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
