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
