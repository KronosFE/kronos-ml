"""Fleet extensions (Phase 2, partial data) and roadmap codes (Phase 3).

Phase-2 codes have on-disk data (traces / physics engine) but no trained ML
surrogate yet — status PARTIAL, predict() raises with a clear pointer.
Phase-3 codes are ROADMAP placeholders.  KECON is the one exception: a real,
generic, firewalled techno-economics calculator (no Kronos numbers, ever).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .. import register
from ..base import Surrogate, Prediction

# real on-disk engineering data (reduced fidelity; full sims are the upgrade path)
_RESEARCH = Path(os.environ.get(
    "KODEX_RESEARCH_ROOT",
    "/Users/pford/Desktop/Kronos Fusion Energy/01 - RESEARCH & DATA/01 - Research, Data & Codes"))
_CD_SCAN = _RESEARCH / "KRONOS_PAPERS_2026-07-31" / "breeder" / "d1_cd_search.csv"
_DIV_THERMAL = _RESEARCH / "BREEDER - Phase 2" / "H9 - exhaust divertor engineering" / "h9_target_thermal.csv"
_SEEDING = _RESEARCH / "BREEDER - Phase 2" / "H9 - exhaust divertor engineering" / "h9_seeding.csv"


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


# KBURN/KISO/KPATH -> members/plant.py; KBREED/KFLUX -> members/neutronics.py;
# KFORGE/KPILOT -> members/capstone.py; KDRIVE/KLAW/KGEN/KFUEL/KSEEK/KBENCH -> members/advanced.py.
# KHEAT/KEDGE/KRAD below are real reduced-fidelity surrogates on on-disk engineering scans
# (full RF/NBI ray-tracing and SOLPS-ITER/EIRENE edge sims are the fidelity upgrades).


@register
class KHEAT(Surrogate):
    name = "KHEAT"; function = "heating&CD"; phase = 2; status = "BUILT"
    provenance = "SIM"
    retired_by = "full RF/NBI ray-tracing (GENRAY/TORAY/NUBEAM)"
    real_codes = ("current-drive design scan",)
    gates = ("H10",)
    note = ("heating & current-drive actuator-response surrogate — fast RandomForest over a "
            "3888-point current-drive design scan; predicts driven current I_cd from the RF/NBI "
            "drive parameters + plasma state. Reduced CD model; ray-tracing = fidelity upgrade")
    _FEATS = ["gamma_cd", "P_cd", "ne", "R0", "B0", "Ti0", "beta_N"]

    def _fit(self):
        if getattr(self, "_m", None) is not None:
            return self._m
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        from .. import uq
        d = pd.read_csv(_CD_SCAN)[self._FEATS + ["I_cd"]].dropna()
        X = d[self._FEATS].to_numpy(float); y = d["I_cd"].to_numpy(float)
        rng = np.random.default_rng(uq.SEED); p = rng.permutation(len(y)); X, y = X[p], y[p]
        ntr = int(0.7 * len(y))
        m = RandomForestRegressor(n_estimators=200, random_state=0).fit(X[:ntr], y[:ntr])
        self._r2 = float(r2_score(y[ntr:], m.predict(X[ntr:]))); self._n = len(y)
        self._m = m
        return m

    def _predict(self, x):
        m = self._fit()
        return Prediction(float(m.predict(np.atleast_2d(np.asarray(x, float)))[0]), None, True,
                          note="driven current I_cd (MA) from [gamma_cd,P_cd,ne,R0,B0,Ti0,beta_N]")

    def benchmark(self):
        self._fit()
        return {"member": self.name,
                "live_cd_surrogate": {
                    "source": "d1_cd_search.csv (3888-point current-drive design scan)",
                    "features": self._FEATS, "target": "I_cd (driven current, MA)",
                    "r2_vs_scan": round(self._r2, 3), "n_samples": self._n,
                    "verdict": (f"Fast surrogate of the current-drive scan: predicts driven current from "
                                f"RF/NBI drive + plasma params, R2={self._r2:.3f} ({self._n} configs). Reduced "
                                f"CD model — GENRAY/TORAY/NUBEAM ray-tracing is the fidelity upgrade.")},
                "caveat": "engineering CD scan (not full ray-tracing); feeds KAIROS heating control"}


@register
class KEDGE(Surrogate):
    name = "KEDGE"; function = "edge-transport"; phase = 3; status = "BUILT"
    provenance = "SIM"
    retired_by = "SOLPS-ITER / EIRENE edge campaign"
    real_codes = ("divertor target-thermal scan",)
    gates = ("H9",)
    note = ("divertor / edge heat-flux -> target-thermal surrogate — maps divertor heat flux to "
            "target surface temperature + material limits (W / CuCrZr) from the H9 exhaust scan. "
            "Reduced 0-D thermal model; SOLPS-ITER/EIRENE = fidelity upgrade")

    def _fit(self):
        if getattr(self, "_coef", None) is not None:
            return
        import pandas as pd
        from sklearn.metrics import r2_score
        df = pd.read_csv(_DIV_THERMAL).dropna(subset=["q_MWm2", "T_w_surf_C"])
        q = df["q_MWm2"].to_numpy(float); T = df["T_w_surf_C"].to_numpy(float)
        self._coef = np.polyfit(q, T, 2)
        self._r2 = float(r2_score(T, np.polyval(self._coef, q))); self._n = len(q)
        bad = df[df["ok_CuCrZr"] == False] if "ok_CuCrZr" in df else df.iloc[0:0]
        self._q_limit = float(bad["q_MWm2"].min()) if len(bad) else float(q.max())

    def _predict(self, x):
        self._fit()
        q = float(np.ravel(np.asarray(x, float))[0])
        return Prediction(float(np.polyval(self._coef, q)), None, q <= self._q_limit,
                          note=f"target surface temp (C) at q={q} MW/m^2; in_domain = under CuCrZr limit")

    def benchmark(self):
        self._fit()
        return {"member": self.name,
                "live_divertor_thermal": {
                    "source": "h9_target_thermal.csv (H9 exhaust/divertor engineering scan)",
                    "map": "divertor heat flux q [MW/m^2] -> target surface temperature [C]",
                    "r2_fit": round(self._r2, 3), "n_samples": self._n,
                    "max_safe_q_MWm2_CuCrZr": self._q_limit,
                    "verdict": (f"Divertor target-thermal surrogate: q->T_surf fit R2={self._r2:.3f} over "
                                f"{self._n} points; CuCrZr material limit at q~{self._q_limit} MW/m2. Reduced "
                                f"0-D thermal model — SOLPS-ITER/EIRENE edge campaign is the fidelity upgrade.")},
                "caveat": "0-D target-thermal scan, not a full 2-D edge-transport solve"}


@register
class KRAD(Surrogate):
    name = "KRAD"; function = "radiation-control"; phase = 3; status = "BUILT"
    provenance = "SIM"
    retired_by = "impurity transport (SOLPS + impurity) + radiation control"
    real_codes = ("impurity-seeding radiation scan",)
    gates = ("H9",)
    note = ("impurity-seeding radiation-control evaluator — maps a seeded impurity + radiated-power "
            "fraction to core Zeff penalty and P_rad from the H9 seeding scan. Small scan; full "
            "impurity-transport (SOLPS + impurity) = fidelity upgrade")

    def _table(self):
        import pandas as pd
        return pd.read_csv(_SEEDING)

    def _predict(self, x):
        df = self._table()
        imp = x.get("impurity", "N") if isinstance(x, dict) else "N"
        row = df[df["impurity"] == imp]
        r = (row.iloc[0] if len(row) else df.iloc[0])
        return Prediction({"impurity": str(r["impurity"]), "P_rad_MW": float(r["P_rad_MW"]),
                           "dZeff_core": [float(r["dZeff_core_lo"]), float(r["dZeff_core_hi"])],
                           "c_div_frac_pct": float(r["c_div_frac_pct"])},
                          None, True, note="radiated power + core-Zeff penalty for the seeded impurity")

    def benchmark(self):
        df = self._table()
        best = df.loc[df["dZeff_core_hi"].idxmin()]
        return {"member": self.name,
                "live_impurity_seeding": {
                    "source": "h9_seeding.csv (H9 impurity-seeding scan)",
                    "n_impurities": int(df["impurity"].nunique()),
                    "impurities": sorted(df["impurity"].unique().tolist()),
                    "radiated_fraction": float(df["f_rad"].iloc[0]),
                    "least_core_dilution_impurity": str(best["impurity"]),
                    "verdict": (f"Impurity-seeding radiation evaluator over {int(df['impurity'].nunique())} "
                                f"impurities at f_rad={df['f_rad'].iloc[0]}: '{best['impurity']}' gives the least "
                                f"core Zeff penalty (dZeff_hi={best['dZeff_core_hi']}). Small scan — full "
                                f"impurity-transport (SOLPS) is the fidelity upgrade.")},
                "caveat": "small seeding scan (few impurities); reduced 0-D radiation model"}


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
