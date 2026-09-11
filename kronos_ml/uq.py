"""KHALO's reusable uncertainty engine — the shared layer every surrogate wraps.

Two heads (GP posterior for regression, deep-ensemble for classification) plus
the canonical calibration report from `metrics`.  sklearn is imported lazily so
`import kronos_ml` stays clean in a bare env.
"""
from __future__ import annotations

import numpy as np

from . import metrics

SEED = 20260726  # the corpus-wide fixed seed


class GPHead:
    """Gaussian-process regressor (Matern-5/2 x Constant + White), optionally
    delta-learning on a cheap baseline.  Returns calibrated (mean, std)."""

    def __init__(self, baseline=None, length_scale=0.5, alpha=1e-8, n_restarts=2):
        self.baseline = baseline
        self.length_scale = length_scale
        self.alpha = alpha
        self.n_restarts = n_restarts
        self._gp = None
        self._xmu = self._xsd = None

    def _std_x(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if self._xmu is None:
            self._xmu = X.mean(0); self._xsd = X.std(0) + 1e-12
        return (X - self._xmu) / self._xsd

    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            Matern, ConstantKernel, WhiteKernel)
        X = np.atleast_2d(np.asarray(X, float)); y = np.asarray(y, float)
        resid = y - self._base(X) if self.baseline is not None else y
        k = (ConstantKernel(1.0, (1e-3, 1e3))
             * Matern(length_scale=self.length_scale, length_scale_bounds=(1e-2, 1e1), nu=2.5)
             + WhiteKernel())
        self._gp = GaussianProcessRegressor(
            kernel=k, alpha=self.alpha, normalize_y=True,
            n_restarts_optimizer=self.n_restarts, random_state=SEED)
        self._gp.fit(self._std_x(X), resid)
        return self

    def _base(self, X):
        return np.asarray([self.baseline(xi) for xi in np.atleast_2d(X)], float)

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        mean, std = self._gp.predict(self._std_x(X), return_std=True)
        if self.baseline is not None:
            mean = mean + self._base(X)
        return mean, std


class EnsembleHead:
    """Soft-vote deep ensemble (HistGB + RandomForest + MLP) for the disruption /
    classifier path.  Returns mean predict_proba."""

    def __init__(self):
        self.models = None
        self.scaler = None

    def fit(self, X, y):
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      RandomForestClassifier)
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        X = np.asarray(X, float); y = np.asarray(y, int)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.models = [
            HistGradientBoostingClassifier(max_iter=120, max_depth=6,
                                           learning_rate=0.12, random_state=SEED).fit(Xs, y),
            RandomForestClassifier(n_estimators=120, max_depth=10,
                                   random_state=SEED).fit(Xs, y),
            MLPClassifier(hidden_layer_sizes=(48, 24), max_iter=180,
                          early_stopping=True, random_state=SEED).fit(Xs, y),
        ]
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(np.asarray(X, float))
        return np.mean([m.predict_proba(Xs)[:, 1] for m in self.models], axis=0)


class CalibratedClassifier:
    """Deep ensemble + isotonic probability calibration (the net-new KODEX layer
    on the disruption path).  Fits the ensemble on a train split and the isotonic
    map on a held-out calibration split."""

    def __init__(self, cal_frac=0.3):
        self.cal_frac = cal_frac
        self.ens = None
        self.iso = None

    def fit(self, X, y):
        from sklearn.isotonic import IsotonicRegression
        X = np.asarray(X, float); y = np.asarray(y, int)
        n = len(y); ntr = int((1 - self.cal_frac) * n)
        self.ens = EnsembleHead().fit(X[:ntr], y[:ntr])
        p = self.ens.predict_proba(X[ntr:])
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(p, y[ntr:])
        self._cal_X, self._cal_y = X[ntr:], y[ntr:]
        return self

    def predict_proba(self, X):
        return self.iso.predict(self.ens.predict_proba(np.asarray(X, float)))


class MahalanobisGate:
    """Per-sample out-of-domain gate: Mahalanobis distance (LedoitWolf-shrunk
    precision) to the training distribution, thresholded to a target in-dist
    abstention rate.  in_domain(x) is False when x is anomalous vs training."""

    def __init__(self, abstain_rate=0.05):
        self.abstain_rate = abstain_rate
        self.mu = self.prec = self.thresh = None

    def fit(self, X):
        from sklearn.covariance import LedoitWolf
        X = np.asarray(X, float)
        self.mu = X.mean(0)
        self.prec = LedoitWolf().fit(X).precision_
        d = self._d(X)
        self.thresh = float(np.quantile(d, 1 - self.abstain_rate))
        return self

    def _d(self, X):
        D = np.atleast_2d(np.asarray(X, float)) - self.mu
        return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", D, self.prec, D), 0, None))

    def in_domain(self, x):
        return bool(self._d(x)[0] <= self.thresh)


def calibrate(y_true, mean, std, levels=metrics.DEFAULT_LEVELS):
    """The first-class calibration object KHALO stamps on any surrogate output."""
    return metrics.calibration_report(y_true, mean, std, levels)
