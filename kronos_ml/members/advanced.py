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


# ------------------------------------------------------------------ KSCOUT
@register
class KSCOUT(Surrogate):
    name = "KSCOUT"; function = "MF-CAMPAIGN"; phase = 3; status = "BUILT"
    provenance = "SIM"
    retired_by = "the full multi-fidelity CGYRO campaign (twin + mu=400 + real-mass gold)"
    real_codes = ("multi-fidelity Bayesian optimization", "cost-aware acquisition")
    gates = ("BR-L2-A1e",)
    note = ("multi-fidelity Bayesian-optimization campaign manager — extends KSEEK: chooses which "
            "operating point AND which fidelity (reduced twin / mu=400 CGYRO / real-mass gold) to "
            "run next, greedily maximizing truth-weighted information per GPU-hour under a budget")

    _BOX = ((2.0, 3.5), (0.4, 1.6))          # a_LT, shear (matches KYRO / KSEEK)
    _C0 = 0.10                                # fixed per-run setup overhead (GPU-h equiv.)
    _HOTSPOTS = 16                            # target uncertain locations for a campaign
    # honest fidelity model: measured mu=400 cost; twin ~free; gold ~10x mu=400 (estimate, not yet run).
    # rho = ASSUMED correlation-with-truth per fidelity (gold=1 by definition); labelled as assumptions.
    _FIDELITIES = {
        "twin":         {"gpu_h": 0.0,  "rho": 0.55, "desc": "reduced analytic twin (near-free triage)"},
        "cgyro_mu400":  {"gpu_h": 2.89, "rho": 0.90, "desc": "CGYRO representative mu=400 (measured GPU-h)"},
        "gold_mu3672":  {"gpu_h": 28.9, "rho": 1.00, "desc": "CGYRO real-mass gold (est. ~10x mu=400; not yet run)"},
    }

    def _grid(self, n=31):
        a = np.linspace(*self._BOX[0], n); s = np.linspace(*self._BOX[1], n)
        A, S = np.meshgrid(a, s)
        return np.c_[A.ravel(), S.ravel()]

    def _hotspots(self, n_hotspots, min_sep=0.16):
        """The n most-uncertain operating points (KYRO GP std), min-separated + away from data."""
        gp = get("KYRO")._fit()
        G = self._grid()
        _, std = gp.predict(G); std = np.asarray(std).ravel()
        Xtr = data.cgyro_flux_map_final()[["a_LT", "shear"]].to_numpy(float)
        span = np.array([self._BOX[0][1] - self._BOX[0][0], self._BOX[1][1] - self._BOX[1][0]])
        picks = []
        for idx in np.argsort(-std):
            g = G[idx]
            d_tr = np.linalg.norm((g - Xtr) / span, axis=1).min()
            d_ch = min([np.linalg.norm((g - c[0]) / span) for c in picks], default=9.0)
            if min(d_tr, d_ch) < min_sep:
                continue
            picks.append((g, float(std[idx])))
            if len(picks) >= n_hotspots:
                break
        return picks

    def _cost(self, fid):
        return self._FIDELITIES[fid]["gpu_h"] + self._C0

    def _plan(self, budget, n_hotspots=None):
        """Triage-and-escalate: twin-triage every hotspot, then escalate the most-uncertain to
        mu=400, then the single most-critical to real-mass gold — all within the GPU-h budget."""
        n_hotspots = n_hotspots or self._HOTSPOTS
        hs = self._hotspots(n_hotspots)              # sorted by std desc
        c_t, c_m, c_g = (self._cost("twin"), self._cost("cgyro_mu400"), self._cost("gold_mu3672"))
        tiers = [None] * len(hs); used = 0.0
        # tier 1 — cheap twin triage on as many hotspots as the budget allows (highest std first)
        for i in range(len(hs)):
            if used + c_t <= budget + 1e-9:
                tiers[i] = "twin"; used += c_t
        # tier 2 — escalate the single most-uncertain to real-mass gold if affordable
        if tiers and tiers[0] == "twin" and used + c_g <= budget + 1e-9:
            tiers[0] = "gold_mu3672"; used += c_g
        # tier 3 — escalate the next most-uncertain to mu=400 while budget remains
        for i in range(len(hs)):
            if tiers[i] == "twin" and used + c_m <= budget + 1e-9:
                tiers[i] = "cgyro_mu400"; used += c_m
        plan = []
        for (g, s), tier in zip(hs, tiers):
            if tier is None:
                continue
            plan.append({"a_LT": round(float(g[0]), 3), "shear": round(float(g[1]), 3),
                         "fidelity": tier, "gpu_h": round(self._cost(tier), 2),
                         "gp_std": round(float(s), 4),
                         "truth_confidence_rho": self._FIDELITIES[tier]["rho"]})
        return plan, round(used, 2), hs

    def _mean_confidence(self, plan, n_hotspots):
        """Mean truth-confidence (rho) achieved across ALL target hotspots (unvisited = 0)."""
        return round(sum(p["truth_confidence_rho"] for p in plan) / n_hotspots, 3)

    def _baseline_all_mu400(self, budget, n_hotspots):
        """Naive baseline: spend the whole budget at mu=400 on the highest-std hotspots."""
        c_m = self._cost("cgyro_mu400")
        n = min(n_hotspots, int((budget + 1e-9) // c_m))
        return {"n_runs": n, "gpu_h_used": round(n * c_m, 2),
                "mean_confidence_rho": round(n * self._FIDELITIES["cgyro_mu400"]["rho"] / n_hotspots, 3),
                "unvisited_hotspots": n_hotspots - n}

    def _predict(self, x):
        """x = GPU-hour budget (float) -> the recommended multi-fidelity campaign plan."""
        budget = float(np.ravel(np.asarray(x, float))[0]) if x is not None else 15.0
        plan, used, _ = self._plan(budget)
        by_fid = {f: sum(1 for p in plan if p["fidelity"] == f) for f in self._FIDELITIES}
        return Prediction(plan, None, True,
                          note=f"multi-fidelity campaign: {len(plan)} runs {by_fid}, {used} GPU-h of {budget}")

    def benchmark(self):
        budget = 15.0                                # tight budget: mu=400 can't cover every hotspot
        n = self._HOTSPOTS
        plan, used, hs = self._plan(budget, n)
        by_fid = {f: sum(1 for p in plan if p["fidelity"] == f) for f in self._FIDELITIES}
        mf_conf = self._mean_confidence(plan, n)
        base = self._baseline_all_mu400(budget, n)
        ratio = round(mf_conf / base["mean_confidence_rho"], 2) if base["mean_confidence_rho"] > 1e-9 else None
        return {"member": self.name,
                "live_mf_campaign": {
                    "source": "KYRO GP over the completed 16-pt CGYRO A1e map",
                    "budget_gpu_h": budget, "target_hotspots": n,
                    "policy": "twin-triage every hotspot, escalate most-uncertain to mu=400, reserve gold",
                    "fidelities": {k: {kk: v[kk] for kk in ("gpu_h", "rho", "desc")}
                                   for k, v in self._FIDELITIES.items()},
                    "multi_fidelity_plan": plan,
                    "mf_runs": len(plan), "mf_gpu_h_used": used, "mf_runs_by_fidelity": by_fid,
                    "mf_mean_confidence_rho": mf_conf,
                    "baseline_all_mu400": base,
                    "mf_confidence_gain_x_vs_all_mu400": ratio,
                    "verdict": (f"Under a tight {budget} GPU-h budget across {n} uncertain operating points, the "
                                f"multi-fidelity policy triages ALL {n} with cheap twin runs and escalates "
                                f"{by_fid['cgyro_mu400']} to mu=400 ({by_fid['gold_mu3672']} to real-mass gold), "
                                f"reaching mean truth-confidence rho {mf_conf} — vs {base['mean_confidence_rho']} "
                                f"for a naive all-mu=400 plan that exhausts the budget after {base['n_runs']} "
                                f"points and leaves {base['unvisited_hotspots']} hotspots unexamined ({ratio}x). "
                                f"A real cost-aware campaign manager: look everywhere cheaply, spend the "
                                f"expensive fidelity where KYRO is least sure.")},
                "caveats": ["the per-fidelity rho (correlation-with-truth) values are ASSUMPTIONS — the "
                            "real-mass gold has not been run — so this is a planning heuristic, not a "
                            "measured speedup; the advantage holds only while budget < hotspots x mu=400 cost",
                            "GP std comes from only 16 points (directional, like KSEEK)",
                            "GPU-hours only; no compute-cost figures (product-econ rule)"]}
