"""Cutting-edge learned codes trained on data already on disk.

KLAW   — equation discovery (sparse regression): find a compact closure for the
         CGYRO transport from the 16-point map.
KDRIVE — reinforcement-learning controller: a feedback policy learned by the
         cross-entropy method in the twin plant loop.
KGEN   — generative surrogate: a PCA/latent model that samples plausible
         equilibrium flux fields (diffusion is the roadmap upgrade).
KFUEL  — tritium fuel-cycle inventory (a reduced systems model).
"""
from __future__ import annotations

import numpy as np

from .. import register, get
from ..base import Surrogate, Prediction
from .. import uq, metrics, data
from .control import _ALPHA, _SIGMA, _plant_step
from .equil import solovev_psi, _field


# ------------------------------------------------------------------ KLAW
def _library(X):
    a, s = X[:, 0], X[:, 1]
    feats = np.c_[np.ones_like(a), a, s, a * a, s * s, a * s,
                  np.maximum(a - 2.1, 0), np.maximum(a - 2.1, 0) ** 2, a * s * s]
    names = ["1", "a_LT", "s", "a_LT^2", "s^2", "a_LT*s",
             "relu(a_LT-2.1)", "relu(a_LT-2.1)^2", "a_LT*s^2"]
    return feats, names


@register
class KLAW(Surrogate):
    name = "KLAW"; function = "eqn-discovery"; phase = 3; status = "BUILT"
    provenance = "CGYRO-urep"
    retired_by = "first-principles reduced-transport theory"
    real_codes = ("sparse regression / SINDy",)
    note = ("equation discovery — sparse regression finds a compact closure for "
            "log10(Q_tot) over the CGYRO map (mu=400)")

    def __init__(self):
        self._m = None; self._sc = None; self._terms = None

    def _fit(self):
        if self._m is None:
            from sklearn.linear_model import LassoCV
            from sklearn.preprocessing import StandardScaler
            df = data.cgyro_flux_map_final()
            X = df[["a_LT", "shear"]].to_numpy(float)
            y = np.log10(np.clip(df.Q_tot.to_numpy(float), 0, None) + 1e-4)
            F, names = _library(X)
            self._sc = StandardScaler().fit(F)
            self._m = LassoCV(cv=5, random_state=uq.SEED, max_iter=100000).fit(self._sc.transform(F), y)
            self._terms = [(names[i], round(float(c), 3))
                           for i, c in enumerate(self._m.coef_) if abs(c) > 1e-6]
        return self._m

    def _predict(self, x):
        self._fit(); x = np.asarray(x, float).reshape(1, -1)
        F, _ = _library(x)
        log_q = float(self._m.predict(self._sc.transform(F))[0])
        q = max(10 ** log_q - 1e-4, 0.0)
        return Prediction(q, None, bool(2.0 <= x[0, 0] <= 3.5 and 0.4 <= x[0, 1] <= 1.6),
                          note="Q_tot from the discovered closure")

    def _x_from_params(self, a_LT=3.0, shear=1.0, **_):
        return [float(a_LT), float(shear)]

    def benchmark(self):
        m = self._fit()
        df = data.cgyro_flux_map_final()
        X = df[["a_LT", "shear"]].to_numpy(float)
        y = np.log10(np.clip(df.Q_tot.to_numpy(float), 0, None) + 1e-4)
        F, _ = _library(X)
        pred = m.predict(self._sc.transform(F))
        return {"member": self.name,
                "live": {"r2_fit": round(metrics.r2(y, pred), 3),
                         "n_terms_selected": len(self._terms),
                         "discovered_terms": self._terms},
                "note": "compact closure discovered from the 16/16 CGYRO map"}


# ------------------------------------------------------------------ KDRIVE
def _rollout(K, ref, steps=150, seed=0):
    rng = np.random.default_rng(seed)
    x = np.array([2.0, 0.0, 2.0, 0.5, 0.1, 20.0])
    clamp = get("KGATE").clamp
    scale = np.array([1, 1, 4.2, 3, 0.4, 47.21])
    err = 0.0
    for _ in range(steps):
        u = x + K * (ref - x)
        safe, _ = clamp(u)
        x = _plant_step(x, safe, rng)
        err += np.mean(((x - ref) / scale) ** 2)
    return np.sqrt(err / steps)


@register
class KDRIVE(Surrogate):
    name = "KDRIVE"; function = "RL-control"; phase = 3; status = "BUILT"
    provenance = "TWIN"
    retired_by = "device-validated learned controller"
    real_codes = ("reinforcement learning",)
    note = ("reinforcement-learning controller — a feedback policy learned by the "
            "cross-entropy method in the twin plant loop; clamped by KGATE")

    def __init__(self):
        self._K = None

    def _fit(self):
        if self._K is None:
            ref = np.array([2.2, 0.3, 3.0, 1.0, 0.2, 30.0])
            rng = np.random.default_rng(uq.SEED)
            mu = np.full(6, 0.5); sd = np.full(6, 0.4)      # policy = per-axis gain
            for _ in range(20):                              # cross-entropy iterations
                pop = np.clip(rng.normal(mu, sd, size=(30, 6)), 0.0, 1.5)
                costs = np.array([_rollout(k, ref, seed=1) for k in pop])
                elite = pop[np.argsort(costs)[:6]]
                mu, sd = elite.mean(0), elite.std(0) + 1e-3
            self._K = mu
        return self._K

    def _predict(self, x):
        K = self._fit()
        state, ref = x
        state = np.asarray(state, float).ravel(); ref = np.asarray(ref, float).ravel()
        safe, _ = get("KGATE").clamp(state + K * (ref - state))
        return Prediction(safe, None, True, note="learned RL policy command (clamped)")

    def benchmark(self):
        K = self._fit()
        ref = np.array([2.2, 0.3, 3.0, 1.0, 0.2, 30.0])
        learned = _rollout(K, ref, seed=7)
        naive = _rollout(np.full(6, 0.2), ref, seed=7)
        return {"member": self.name,
                "live": {"learned_policy_tracking_rms": round(float(learned), 4),
                         "naive_gain_tracking_rms": round(float(naive), 4),
                         "learned_gains": [round(float(k), 3) for k in K],
                         "note": "cross-entropy-method policy search, twin-in-the-loop"}}


# ------------------------------------------------------------------ KGEN
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


# ------------------------------------------------------------------ KFUEL
@register
class KFUEL(Surrogate):
    name = "KFUEL"; function = "fuel-cycle"; phase = 3; status = "BUILT"
    provenance = "SIM"
    retired_by = "full fuel-cycle systems code (e.g. MIRC / TRANSAT)"
    real_codes = ("fuel-cycle systems model",)
    note = ("tritium fuel-cycle inventory — a reduced systems model "
            "(breeding, burn, decay, reserve) giving self-sufficiency + doubling time")

    T_HALFLIFE_YR = 12.32

    def _predict(self, x):
        """x = dict{TBR, burn_rate_kg_yr, reserve_kg, startup_kg} -> inventory outcome."""
        if not isinstance(x, dict):
            x = {}
        TBR = float(x.get("TBR", 1.05)); burn = float(x.get("burn_rate_kg_yr", 55.6))
        reserve = float(x.get("reserve_kg", 5.0)); start = float(x.get("startup_kg", 10.0))
        decay = np.log(2) / self.T_HALFLIFE_YR
        net_breed = burn * (TBR - 1.0) - decay * (start + reserve)   # kg/yr surplus
        self_sufficient = net_breed > 0
        doubling_yr = (start / net_breed) if net_breed > 1e-6 else float("inf")
        return Prediction({"net_breeding_kg_yr": round(net_breed, 3),
                           "self_sufficient": bool(self_sufficient),
                           "doubling_time_yr": round(doubling_yr, 2) if np.isfinite(doubling_yr) else None},
                          None, TBR >= 1.0, note="reduced tritium fuel-cycle balance")

    def benchmark(self):
        p = self.predict({"TBR": 1.05, "burn_rate_kg_yr": 55.6, "reserve_kg": 5.0, "startup_kg": 10.0})
        return {"member": self.name, "live_nominal": p.y,
                "note": "reduced systems model seeded from tritium physics (12.32 yr half-life); "
                        "full fuel-cycle code is the roadmap upgrade"}


@register
class KSEEK(Surrogate):
    name = "KSEEK"; function = "ACTIVE"; phase = 3; status = "BUILT"
    provenance = "SIM"
    retired_by = "the full CGYRO parameter scan (once every point is simulated)"
    real_codes = ("Gaussian-process active learning", "max-variance acquisition")
    gates = ("BR-L2-A1e",)
    note = ("active-learning acquisition — proposes the NEXT most-informative CGYRO run from "
            "KYRO's GP posterior (max-variance / uncertainty sampling), so expensive GPU-hours "
            "go where the surrogate is least sure. A real experimental-design tool")

    _BOX = ((2.0, 3.5), (0.4, 1.6))          # a_LT, shear in-domain box (matches KYRO)

    def _grid(self, n=41):
        a = np.linspace(*self._BOX[0], n); s = np.linspace(*self._BOX[1], n)
        A, S = np.meshgrid(a, s)
        return np.c_[A.ravel(), S.ravel()]

    def _model(self):
        import kronos_ml.data as data
        gp = get("KYRO")._fit()               # reuse the exact GP the fleet queries
        df = data.cgyro_flux_map_final()
        return gp, df[["a_LT", "shear"]].to_numpy(float), \
            np.log10(np.clip(df.Q_tot.to_numpy(float), 0, None) + 1e-4)

    def _acquire(self, k=3, min_sep=0.16):
        gp, Xtr, _ = self._model()
        G = self._grid()
        _, std = gp.predict(G)
        std = np.asarray(std).ravel()
        span = np.array([self._BOX[0][1] - self._BOX[0][0], self._BOX[1][1] - self._BOX[1][0]])
        picks, chosen = [], []
        for idx in np.argsort(-std):
            g = G[idx]
            d_tr = np.linalg.norm((g - Xtr) / span, axis=1).min()
            d_ch = min([np.linalg.norm((g - c) / span) for c in chosen], default=9.0)
            if min(d_tr, d_ch) < min_sep:
                continue
            chosen.append(g); picks.append({"a_LT": round(float(g[0]), 3),
                                            "shear": round(float(g[1]), 3),
                                            "gp_std": round(float(std[idx]), 4)})
            if len(picks) >= k:
                break
        return picks

    def _loo(self):
        """Leave-one-out: does GP posterior std actually predict where the model is wrong?"""
        from .. import uq
        _, X, y = self._model()
        errs, stds = [], []
        for i in range(len(y)):
            m = np.ones(len(y), bool); m[i] = False
            g = uq.GPHead().fit(X[m], y[m])
            mu, sd = g.predict(X[i:i + 1])
            errs.append(abs(float(np.ravel(mu)[0]) - y[i])); stds.append(float(np.ravel(sd)[0]))
        errs, stds = np.array(errs), np.array(stds)
        if errs.std() < 1e-9 or stds.std() < 1e-9:
            return None
        return float(np.corrcoef(stds, errs)[0, 1])

    def _predict(self, x):
        p = self._acquire(k=1)[0]
        return Prediction(p, None, True, note="next most-informative CGYRO operating point (max GP variance)")

    def benchmark(self):
        picks = self._acquire(k=3)
        corr = self._loo()
        return {"member": self.name,
                "live_active_learning": {
                    "source": "KYRO GP over the completed 16-pt CGYRO A1e map",
                    "acquisition": "max GP posterior std (uncertainty sampling), min-separation filtered",
                    "next_runs": picks,
                    "loo_std_vs_error_corr": None if corr is None else round(corr, 3),
                    "cgyro_gpu_h_per_point": 2.89,
                    "verdict": (f"Proposes the next CGYRO run at a/L_T={picks[0]['a_LT']}, "
                                f"shear={picks[0]['shear']} (highest GP uncertainty). Leave-one-out: GP "
                                f"posterior-std vs actual error correlation = "
                                f"{'n/a' if corr is None else round(corr, 3)} — "
                                f"{'the acquisition targets real model error' if (corr or 0) > 0.2 else 'weak on this small map'}. "
                                f"A real experimental-design tool: spend ~2.9 GPU-h/point where it matters.")},
                "caveat": "16 training points is small — acquisition is directional guidance, not a guarantee"}


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
