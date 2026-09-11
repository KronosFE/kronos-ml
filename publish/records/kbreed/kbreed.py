"""KODEX KBREED — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KBREED(Surrogate):
    name = "KBREED"; function = "breeder"; phase = 2; status = "BUILT"
    provenance = "SIM"
    retired_by = "OpenMC (Monte-Carlo neutron transport)"
    real_codes = ("openmc",)
    gates = ("BR-neut",)
    note = ("tritium breeding-ratio surrogate (Hyperion blanket) — fast GP over the "
            "toolkit neutronics; OpenMC is the auto-dispatch high-fidelity upgrade")

    _B = [(0.3, 0.95), (0.5, 1.2), (0.85, 0.95)]   # li6_enrichment, thickness_m, coverage

    def __init__(self):
        self._gp = None

    def _sweep(self, n=120, seed=uq.SEED):
        api = _api(); rng = np.random.default_rng(seed)
        P = np.c_[rng.uniform(*self._B[0], n), rng.uniform(*self._B[1], n), rng.uniform(*self._B[2], n)]
        y = np.array([_val(api.tbr(li6_enrichment=p[0], blanket_thickness_m=p[1],
                                   coverage=p[2], force_analytic=True), "net_tbr")
                      for p in P])
        return P, y

    def _fit(self):
        if self._gp is None:
            P, y = self._sweep()
            self._gp = uq.GPHead().fit(P, y); self._P, self._y = P, y
        return self._gp

    def in_domain(self, p):
        return all(self._B[i][0] <= p[i] <= self._B[i][1] for i in range(3))

    def _predict(self, x):
        gp = self._fit(); x = np.asarray(x, float).reshape(1, -1)
        m, s = gp.predict(x)
        return Prediction(float(m[0]), float(s[0]), self.in_domain(x.ravel()),
                          note="net TBR (>1 = self-sufficient)")

    def _x_from_params(self, li6_enrichment=0.9, blanket_thickness_m=0.8, coverage=0.92, **_):
        return [li6_enrichment, blanket_thickness_m, coverage]

    def benchmark(self):
        gp = self._fit()
        Pte, yte = self._sweep(n=60, seed=uq.SEED + 5)
        m, s = gp.predict(Pte)
        api = _api()
        nom = _val(api.tbr(force_analytic=True), "net_tbr")
        return {"member": self.name,
                "live": {"r2_vs_engine": round(metrics.r2(yte, m), 3),
                         "coverage_90": round(metrics.coverage_at_level(yte, m, np.clip(s, 1e-6, None), 0.90), 3),
                         "nominal_net_tbr": round(nom, 3)},
                "note": "surrogate over toolkit neutronics (analytic); OpenMC upgrade = compute route [A]"}
