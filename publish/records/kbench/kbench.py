"""KODEX KBENCH — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KBENCH(Surrogate):
    name = "KBENCH"; function = "BENCHMARK"; phase = 3; status = "BUILT"
    provenance = "n/a"
    retired_by = "community-standard fusion-ML benchmarks"
    real_codes = ("open ML benchmark suite",)
    gates = ()
    note = ("open, citable ML benchmark suite for fusion — real CGYRO turbulence + MAST "
            "disruption tasks with fixed splits, metrics and KODEX baselines to beat. Bring "
            "your own model; move the community forward")

    def tasks(self):
        return {
            "cgyro-turbulence-flux": {
                "inputs": ["a_LT", "shear"],
                "target": "log10 Q_tot (regression) + turbulent/quiet (classification)",
                "n_samples": 16, "metric": "R2 (regression) / leave-one-out accuracy (classification)",
                "loader": "kronos_ml.data.cgyro_flux_map_final()",
                "baseline_code": "KYRO", "baseline_score": "R2 ~0.86, turbulent/quiet 16/16",
                "note": "real CGYRO A1e saturated-flux (mu=400 representative)"},
            "mast-disruption": {
                "inputs": "physics features (Ip family + EFIT + n=1 Mirnov / P_rad)",
                "target": "disruptive (binary)",
                "n_samples": 591, "metric": "ROC-AUC (+ independent-precursor AUC)",
                "loader": "kronos_ml.data.kward_real()",
                "baseline_code": "KWARD", "baseline_score": "AUC ~0.98, independent-precursor 0.975",
                "note": "real MAST shots (FAIR-MAST); labels are heuristic Ip-quench"},
            "cgyro-rom-compressibility": {
                "inputs": "flux-database matrix (points x [inputs, fluxes])",
                "target": "rel-L2 reconstruction error vs retained bond dimension",
                "n_samples": 16, "metric": "rel-L2 vs rank",
                "loader": "kronos_ml.data.cgyro_flux_map_final()",
                "baseline_code": "KTENSOR", "baseline_score": "effective rank 2.93/4 (not strongly low-rank)",
                "note": "honest ROM characterization"},
        }

    def _predict(self, x):
        t = self.tasks()
        key = x if isinstance(x, str) and x in t else next(iter(t))
        return Prediction(t[key], None, True, note=f"fusion-ML benchmark task spec: {key}")

    def benchmark(self):
        t = self.tasks()
        return {"member": self.name,
                "live_benchmark_suite": {
                    "n_tasks": len(t), "tasks": t,
                    "how_to_use": ("load via each task's loader, use the fixed reproducible split, report "
                                   "the metric, and try to beat the named KODEX baseline_code"),
                    "verdict": ("3 open, citable fusion-ML tasks (CGYRO turbulence, MAST disruption, flux "
                                "ROM) with real on-disk data + reproducible KODEX baselines — a community "
                                "leaderboard starting point, not a private result")},
                "caveat": "small by mainstream-ML standards (CGYRO = 16 pts); honest pilot fusion-ML benchmarks"}
