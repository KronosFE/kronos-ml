"""KODEX KORE — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KORE(Surrogate):
    name = "KORE"; function = "EQUIL"; phase = 1; status = "BUILT"
    provenance = "ANALYTIC"
    retired_by = "classical Grad-Shafranov solver"
    real_codes = ("Grad-Shafranov FD solver",)
    gates = ("AC-16", "AC-L1")
    note = ("fast learned MHD-equilibrium accelerator (neural surrogate, 3-seed "
            "ensemble); Grad-Shafranov, not turbulence")

    def __init__(self):
        self._ens = None

    def _fit(self):
        if self._ens is None:
            self._ens = [_Net(s).fit_or_load() for s in _SEEDS]
        return self._ens

    def in_domain(self, p):
        A, k, d = p
        return bool(_A[0] <= A <= _A[1] and _K[0] <= k <= _K[1] and _D[0] <= d <= _D[1])

    def _predict(self, x):
        """x = [A, kappa, delta] -> psi field (16x16) with ensemble uncertainty."""
        A, k, d = (list(x) + [1.0, 2.0, 0.3])[:3]
        ens = self._fit()
        preds = np.array([m.predict([A, k, d])[0] for m in ens])
        mean = preds.mean(0); std = preds.std(0)
        return Prediction(mean.reshape(_G, _G), std.reshape(_G, _G),
                          self.in_domain((A, k, d)),
                          note="learned psi field (16x16); uncertainty = ensemble std")

    def _x_from_params(self, A=1.0, kappa=2.0, delta=0.3, **_):
        return [float(A), float(kappa), float(delta)]

    def benchmark(self):
        ens = self._fit()
        Pte, Yte = _dataset(120, seed=999)          # held-out analytic members
        preds = np.array([np.array([m.predict(p)[0] for m in ens]).mean(0) for p in Pte])
        rl2 = float(np.mean([metrics.rel_l2(Yte[i], preds[i]) for i in range(len(Pte))]))
        # ensemble UQ coverage on held-out
        stds = np.array([np.array([m.predict(p)[0] for m in ens]).std(0) for p in Pte])
        cov = float(metrics.coverage_at_level(Yte.ravel(), preds.ravel(),
                                              np.clip(stds.ravel(), 1e-6, None), 0.90))
        t0 = time.perf_counter()
        for _ in range(200):
            self._predict([1.0, 2.0, 0.3])
        ms = (time.perf_counter() - t0) / 200 * 1e3
        return {"member": self.name,
                "live_learned_surrogate": {"rel_l2_vs_analytic": round(rl2, 4),
                                           "ensemble_cov90": round(cov, 3),
                                           "infer_ms": round(ms, 3), "grid": f"{_G}x{_G}",
                                           "n_heldout": len(Pte)},
                "sourced": {"file": "track_L1_neural_operator/operator_vs_pinn.csv",
                            "headline": "track FNO best 2.5% rel-L2 vs 1% bar (MISSED); "
                                        "~1 s classical solver -> sub-ms surrogate"}}
