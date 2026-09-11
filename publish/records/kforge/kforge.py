"""KODEX KFORGE — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

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
