"""KODEX KRAD — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

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
