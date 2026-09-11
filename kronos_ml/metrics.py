"""Shared, canonical metrics for the whole fleet.

This is the one place calibration is defined, so every KODEX code reports the
same reliability numbers (the source tracks each rolled their own; KODEX
co-locates them here).  Regression accuracy + Gaussian-interval calibration +
classifier calibration.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# ---- regression accuracy -------------------------------------------------

def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def rel_l2(y_true, y_pred):
    """Relative L2 error ||pred-true|| / ||true|| — the tracks' equilibrium metric."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.linalg.norm(y_true.ravel())
    return float(np.linalg.norm((y_pred - y_true).ravel()) / denom) if denom > 0 else float("nan")


# ---- calibrated-uncertainty (Gaussian predictive intervals) --------------

DEFAULT_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def coverage_at_level(y_true, mean, std, level):
    """Empirical fraction of truths inside the central `level` Gaussian interval."""
    y_true, mean, std = map(lambda a: np.asarray(a, float), (y_true, mean, std))
    std = np.clip(std, 1e-12, None)
    z = norm.ppf(0.5 + level / 2.0)
    inside = np.abs(y_true - mean) <= z * std
    return float(np.mean(inside))


def reliability_curve(y_true, mean, std, levels=DEFAULT_LEVELS):
    """Nominal vs empirical coverage over a set of nominal levels."""
    nominal = np.asarray(levels, float)
    empirical = np.array([coverage_at_level(y_true, mean, std, l) for l in levels])
    return nominal, empirical


def ece(y_true, mean, std, levels=DEFAULT_LEVELS):
    """Expected calibration error = mean |nominal - empirical| over levels."""
    nominal, empirical = reliability_curve(y_true, mean, std, levels)
    return float(np.mean(np.abs(nominal - empirical)))


def sharpness(std):
    """Mean predictive std — how tight the intervals are (lower = sharper)."""
    return float(np.mean(np.asarray(std, float)))


def nll_gaussian(y_true, mean, std):
    """Mean negative log-likelihood under the predictive Gaussian."""
    y_true, mean, std = map(lambda a: np.asarray(a, float), (y_true, mean, std))
    std = np.clip(std, 1e-12, None)
    return float(np.mean(0.5 * np.log(2 * np.pi * std ** 2) +
                         (y_true - mean) ** 2 / (2 * std ** 2)))


def calibration_report(y_true, mean, std, levels=DEFAULT_LEVELS):
    """The full first-class UQ object every KODEX code can emit."""
    nominal, empirical = reliability_curve(y_true, mean, std, levels)
    return {
        "ece": ece(y_true, mean, std, levels),
        "coverage_90": coverage_at_level(y_true, mean, std, 0.90),
        "sharpness": sharpness(std),
        "nll": nll_gaussian(y_true, mean, std),
        "levels": nominal.tolist(),
        "empirical_coverage": empirical.tolist(),
    }


# ---- classifier calibration (KWARD / disruption) -------------------------

def brier(y_true, p):
    y_true, p = np.asarray(y_true, float), np.asarray(p, float)
    return float(np.mean((p - y_true) ** 2))


def ece_classification(y_true, p, n_bins=10):
    """Binned expected calibration error for probabilistic classifiers."""
    y_true, p = np.asarray(y_true, float), np.asarray(p, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        e += (np.sum(m) / len(p)) * abs(p[m].mean() - y_true[m].mean())
    return float(e)
