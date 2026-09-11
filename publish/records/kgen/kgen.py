"""KODEX KGEN — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KGEN(Surrogate):
    name = "KGEN"; function = "generative"; phase = 3; status = "BUILT"
    provenance = "ANALYTIC"
    retired_by = "high-fidelity generative model (diffusion) on real fields"
    real_codes = ("generative model",)
    note = ("generative surrogate — a PCA/latent model that samples plausible "
            "equilibrium flux fields; diffusion is the roadmap upgrade")

    def __init__(self):
        self._pca = None; self._ev = None

    def _fit(self):
        if self._pca is None:
            from sklearn.decomposition import PCA
            rng = np.random.default_rng(uq.SEED)
            P = np.c_[rng.uniform(0.8, 1.2, 500), rng.uniform(1.5, 2.5, 500), rng.uniform(-0.5, 0.5, 500)]
            fields = np.array([_field(*p) for p in P])
            self._fields = fields
            self._pca = PCA(n_components=8, random_state=uq.SEED).fit(fields)
            self._ev = np.sqrt(self._pca.explained_variance_)
        return self._pca

    def _predict(self, x):
        """x = number of fields to sample -> generated flux fields (16x16 each)."""
        pca = self._fit()
        n = int(x) if np.isscalar(x) else 1
        rng = np.random.default_rng(uq.SEED + 3)
        z = rng.normal(0, 1, size=(n, 8)) * self._ev
        samples = pca.inverse_transform(z)
        return Prediction(samples.reshape(n, 16, 16), None, True,
                          note=f"{n} generated flux fields (PCA latent)")

    def benchmark(self):
        pca = self._fit()
        rng = np.random.default_rng(uq.SEED + 9)
        P = np.c_[rng.uniform(0.8, 1.2, 100), rng.uniform(1.5, 2.5, 100), rng.uniform(-0.5, 0.5, 100)]
        held = np.array([_field(*p) for p in P])
        recon = pca.inverse_transform(pca.transform(held))
        rl2 = np.mean([metrics.rel_l2(held[i], recon[i]) for i in range(len(held))])
        return {"member": self.name,
                "live": {"latent_dim": 8,
                         "reconstruction_rel_l2": float(f"{rl2:.2e}"),
                         "variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
                         "note": "reconstruction is near-exact BECAUSE the analytic field family is "
                                 "low-rank (3 shape params); the diffusion upgrade earns its keep on a "
                                 "richer real-equilibria corpus. KGEN's job is sampling new plausible fields."}}
