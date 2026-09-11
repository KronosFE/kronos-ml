"""KODEX KFLOW — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KFLOW(Surrogate):
    name = "KFLOW"; function = "STATE"; phase = 1; status = "BUILT"
    provenance = "SIM"
    retired_by = "trained message-passing GNN (gnn_seed0..4.pt)"
    real_codes = ("message-passing GNN",)
    gates = ("AC-18", "AC-19")
    note = ("twin state-estimation / dropped-sensor imputation on the 76-node "
            "diagnostic sensor graph")

    def _load(self):
        z = data.gnn_snapshots()
        return (np.asarray(z["Xn_clean"], float), np.asarray(z["Xn_obs"], float),
                np.asarray(z["cls"]))          # class labels are strings, keep as-is

    def _predict(self, x):
        """x = observed 76-vector with NaN at failed channels -> imputed state."""
        obs = np.asarray(x, float).ravel()
        alive = np.isfinite(obs)
        _, _, cls = self._load()
        filled = _impute_same_class(np.where(alive, obs, 0.0), alive, cls)
        # per-channel uncertainty ~ same-class dispersion of alive channels
        unc = np.zeros(76)
        for j in np.where(~alive)[0]:
            same = alive & (cls == cls[j])
            unc[j] = obs[same].std() if same.sum() > 1 else 1.0
        in_dom = bool(alive.mean() >= 0.5)
        return Prediction(filled, unc, in_dom,
                          note=f"{(~alive).sum()} channels imputed")

    def benchmark(self):
        out = {"member": self.name}
        # sourced: the trained-GNN result curve on disk
        try:
            df = data.read_csv("track3_gnn", "gnn_imputation.csv")
            r8 = df.iloc[(df.k_failed - 8).abs().argmin()]
            out["sourced_gnn"] = {
                "file": "track3_gnn/gnn_imputation.csv",
                "k_failed": int(r8.k_failed),
                "gnn_rmse": round(float(r8.gnn_rmse_mean), 4),
                "naive_rmse": round(float(r8.naive_rmse_mean), 4),
                "gnn_beats_naive": bool(r8.gnn_beats_naive),
                "AC18_bar_0.05": "MISSED (kept honest)"}
        except Exception as e:
            out["sourced_gnn"] = {"error": str(e)}
        # live: run the naive imputation baseline at k=8
        try:
            clean, obsn, cls = self._load()
            rng = np.random.default_rng(uq.SEED)
            errs = []
            rng_range = clean.max(0) - clean.min(0) + 1e-9
            for _ in range(50):
                i = rng.integers(len(clean))
                mask = np.ones(76, bool)
                mask[rng.choice(76, 8, replace=False)] = False
                obs = np.where(mask, obsn[i], np.nan)
                filled = _impute_same_class(np.where(mask, obsn[i], 0.0), mask, cls)
                fj = ~mask
                errs.append(np.sqrt(np.mean(((filled[fj] - clean[i][fj]) / rng_range[fj]) ** 2)))
            out["live_classmean_baseline_rmse_k8"] = round(float(np.mean(errs)), 4)
            out["live_note"] = ("crude same-class-mean baseline (worse than the track's "
                                "inverse-distance naive 0.172 and the trained GNN 0.084); "
                                "shipped predict() uses the same-class imputation")
        except Exception as e:
            out["live_naive_rmse_k8"] = {"error": str(e)}
        return out
