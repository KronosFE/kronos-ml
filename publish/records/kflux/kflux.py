"""KODEX KFLUX — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KFLUX(Surrogate):
    name = "KFLUX"; function = "neutronics"; phase = 2; status = "BUILT"
    provenance = "SIM"
    retired_by = "OpenMC (Monte-Carlo neutron transport)"
    real_codes = ("openmc",)
    note = ("neutron shielding / coil-fluence surrogate — fast GP over the toolkit "
            "neutronics; OpenMC is the auto-dispatch high-fidelity upgrade")

    _B = [(10.0, 100.0), (0.3, 1.5)]   # p_neutron_MW, shield_thickness_m

    def __init__(self):
        self._gp = None

    def _sweep(self, n=120, seed=uq.SEED):
        api = _api(); rng = np.random.default_rng(seed)
        P = np.c_[rng.uniform(*self._B[0], n), rng.uniform(*self._B[1], n)]
        y = np.array([np.log10(_val(api.coil_fluence(p_neutron_MW=p[0], shield_thickness_m=p[1],
                                                     force_analytic=True)) + 1.0)
                      for p in P])
        return P, y

    def _fit(self):
        if self._gp is None:
            P, y = self._sweep()
            self._gp = uq.GPHead().fit(P, y)
        return self._gp

    def in_domain(self, p):
        return all(self._B[i][0] <= p[i] <= self._B[i][1] for i in range(2))

    def _predict(self, x):
        gp = self._fit(); x = np.asarray(x, float).reshape(1, -1)
        m, s = gp.predict(x)
        return Prediction(float(10 ** m[0] - 1.0), float(s[0]), self.in_domain(x.ravel()),
                          note="coil neutron fluence (log10 model)")

    def _x_from_params(self, p_neutron_MW=50.0, shield_thickness_m=0.8, **_):
        return [p_neutron_MW, shield_thickness_m]

    def benchmark(self):
        gp = self._fit()
        Pte, yte = self._sweep(n=60, seed=uq.SEED + 5)
        m, _ = gp.predict(Pte)
        return {"member": self.name,
                "live": {"r2_log_fluence_vs_engine": round(metrics.r2(yte, m), 3),
                         "n_test": len(Pte)},
                "note": "surrogate over toolkit shielding/fluence (analytic); OpenMC upgrade = compute route [A]"}
