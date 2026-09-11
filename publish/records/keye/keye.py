"""KODEX KEYE — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KEYE(Surrogate):
    name = "KEYE"; function = "DIAG"; phase = 2; status = "BUILT"
    provenance = "SIM"
    retired_by = "physical diagnostic suite + validated sensor fusion"
    real_codes = ("sensor fusion", "signal validation")
    gates = ("AC-36",)
    note = ("diagnostics / virtual-sensor / signal-validation observer — builds "
            "on KFLOW's graph imputation + quantum-sensing; flags faulty channels")

    def _predict(self, x):
        """x = observed 76-vector; flag channels whose residual vs same-class peers
        is anomalous (a virtual-sensor validation)."""
        obs = np.asarray(x, float).ravel()
        alive = np.isfinite(obs)
        z = data.gnn_snapshots(); cls = np.asarray(z["cls"])
        flags = np.zeros(76, bool)
        for c in np.unique(cls):
            m = alive & (cls == c)
            if m.sum() > 2:
                mu, sd = obs[m].mean(), obs[m].std() + 1e-9
                flags[m] = np.abs(obs[m] - mu) > 3 * sd
        return Prediction(obs, flags.astype(float), bool(alive.mean() >= 0.5),
                          note=f"{int(flags.sum())} channels flagged anomalous")

    def benchmark(self):
        """Live fault-detection: inject a large offset into one channel per snapshot
        and measure detection rate vs the false-alarm rate on clean channels."""
        z = data.gnn_snapshots()
        clean = np.asarray(z["Xn_clean"], float); cls = np.asarray(z["cls"])
        rng = np.random.default_rng(uq.SEED)
        rng_scale = clean.max(0) - clean.min(0) + 1e-9
        detected = 0; false_alarms = 0; clean_channels = 0; n = 300
        for _ in range(n):
            i = rng.integers(len(clean)); obs = clean[i].copy()
            j = rng.integers(76); obs[j] += 5 * rng_scale[j]      # inject a fault
            flags = self.predict(obs).uncertainty.astype(bool)
            detected += int(flags[j])
            fa = flags.copy(); fa[j] = False
            false_alarms += int(fa.sum()); clean_channels += 75
        return {"member": self.name,
                "live_fault_detection": {"detection_rate": round(detected / n, 3),
                                         "false_alarm_rate": round(false_alarms / clean_channels, 4),
                                         "n_trials": n, "method": "3-sigma same-class residual"},
                "sourced": {"file": "track8b_quantum_sensing/sensing_to_disruption_gain.csv",
                            "finding": "quantum sensing buys ~0 gain where disruptions live (model-limited)"}}
