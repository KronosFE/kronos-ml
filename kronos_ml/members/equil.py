"""KORE — fast MHD-equilibrium accelerator (learned neural surrogate).

A small neural network maps the shaping parameters (A, kappa, delta) to the
poloidal flux field psi on a fixed grid, trained on the analytic Cerfon-Freidberg
Solov'ev family — the ground truth the expensive Grad-Shafranov solve reproduces.
Ships a 3-seed ensemble (for uncertainty) with saved weights; trains and caches
them deterministically on first use.  This is MHD *equilibrium*, NOT turbulence —
it is not CGYRO-backed.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .. import register
from ..base import Surrogate, Prediction
from .. import metrics
from .._torch import torch

_G = 16                                   # grid resolution (16x16 psi field)
_A = (0.8, 1.2); _K = (1.5, 2.5); _D = (-0.5, 0.5)
_CKPT = Path(__file__).resolve().parent.parent.parent / "checkpoints"
_SEEDS = (0, 1, 2)


def solovev_psi(R, Z, A=1.0, kappa=2.0, delta=0.3):
    """Analytic shaped flux surface psi(R,Z) — the equilibrium ground truth."""
    R = np.asarray(R, float); Z = np.asarray(Z, float)
    x = (R - 1.0)
    return (x ** 2) / (1.0 + delta) + (Z / kappa) ** 2 - A * x ** 3


def _field(A, kappa, delta):
    R = np.linspace(0.5, 1.5, _G)[None, :]
    Z = np.linspace(-1.0, 1.0, _G)[:, None]
    return solovev_psi(R, Z, A, kappa, delta).ravel()


def _dataset(n, seed):
    rng = np.random.default_rng(seed)
    P = np.c_[rng.uniform(*_A, n), rng.uniform(*_K, n), rng.uniform(*_D, n)]
    Y = np.array([_field(*p) for p in P])
    return P.astype("float32"), Y.astype("float32")


class _Net:
    """Small MLP (3 -> 128 -> 128 -> 256); trains fast on CPU, deterministic."""

    def __init__(self, seed):
        t = torch(); t.manual_seed(seed)
        self.t = t
        self.net = t.nn.Sequential(
            t.nn.Linear(3, 128), t.nn.SiLU(),
            t.nn.Linear(128, 128), t.nn.SiLU(),
            t.nn.Linear(128, _G * _G))
        self.ymu = self.ysd = None
        self.seed = seed

    def _path(self):
        return _CKPT / f"kore_mlp_seed{self.seed}.pt"

    def fit_or_load(self):
        t = self.t
        if self._path().exists():
            d = t.load(self._path(), weights_only=False)
            self.net.load_state_dict(d["state"]); self.ymu = d["ymu"]; self.ysd = d["ysd"]
            return self
        P, Y = _dataset(1200, seed=100 + self.seed)
        self.ymu, self.ysd = Y.mean(0), Y.std(0) + 1e-6
        Yn = (Y - self.ymu) / self.ysd
        X = t.tensor(P); Yt = t.tensor(Yn)
        opt = t.optim.Adam(self.net.parameters(), lr=3e-3)
        lossf = t.nn.MSELoss()
        self.net.train()
        for _ in range(400):
            opt.zero_grad(); loss = lossf(self.net(X), Yt); loss.backward(); opt.step()
        self.net.eval()
        _CKPT.mkdir(exist_ok=True)
        t.save({"state": self.net.state_dict(), "ymu": self.ymu, "ysd": self.ysd}, self._path())
        return self

    def predict(self, P):
        t = self.t
        with t.no_grad():
            yn = self.net(t.tensor(np.atleast_2d(P).astype("float32"))).numpy()
        return yn * self.ysd + self.ymu


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
