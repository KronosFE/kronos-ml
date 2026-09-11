"""KWARD — disruption / stability-boundary early-warning.

REAL-DEVICE: trained and validated on 592 real MAST shots (277 disruptive / 315
non-disruptive) from FAIR-MAST.  Fixed 50 ms pre-event window of physics signals
(pre-quench for disruptive, flat-top for non-disruptive) — NO shot-duration
features (disruptive shots are truncated at t_disr, so duration would leak).
INDEPENDENT PRECURSORS (added 2026-09-10): the n=1 Mirnov fluctuation RMS and the
bolometric radiated power (P_rad) were pulled from FAIR-MAST for the same shots.
These are different diagnostics from Ip, so an AUC built on them alone is not
entangled with the Ip-derived label -> independent-precursor AUC 0.975.
Honest caveats: disruption labels are DERIVED from the Ip current-quench
(heuristic, not physicist-verified); EFIT kinetics are sparse; the "n1_rms" channel
is a broadband Mirnov coil fluctuation RMS (a low-n MHD-activity proxy), NOT a
geometrically-decomposed pure toroidal-n=1 amplitude (the on-disk coil array carries
poloidal/outboard positions but no toroidal-angle metadata).  Advisory only,
downstream of the KGATE clamp.
"""
from __future__ import annotations

import numpy as np

from .. import register
from ..base import Surrogate, Prediction
from .. import uq, metrics, data

_WINDOW_S = 0.05          # 50 ms fixed pre-event window
# base Ip-family + EFIT features, then the INDEPENDENT precursors (auto-used when present)
_FEATURES = ["ip_mean_MA", "ip_std", "ip_slope", "ip_range",
             "ne_mean", "greenwald_proxy", "beta_n", "li", "q95",
             "n1_rms_mean", "n1_rms_slope", "prad_mean", "prad_slope"]
# the two independent-precursor channels (different diagnostics from Ip) — indices 9..12
_INDEP_IDX = [9, 10, 11, 12]


def _synth(n=1200, seed=uq.SEED):
    """Self-contained seeded set for the calibration/OOD machinery smoke test."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    X = np.vstack([rng.normal(0.0, 1.0, (n0, 6)), rng.normal(0.9, 1.1, (n - n0, 6))])
    y = np.r_[np.zeros(n0), np.ones(n - n0)]
    idx = rng.permutation(n)
    return X[idx], y[idx].astype(int)


@register
class KWARD(Surrogate):
    name = "KWARD"; function = "DISRUPT"; phase = 1; status = "BUILT"
    provenance = "REAL-ANCHOR"
    retired_by = "physicist-verified disruption labels + full precursor diagnostics"
    real_codes = ("real MAST shots (FAIR-MAST)",)
    gates = ("AC-20", "AC-21")
    note = ("disruption early-warning — real-device (592 MAST shots), calibrated "
            "ensemble + per-shot OOD gate; advisory, downstream of the KGATE clamp")

    def __init__(self):
        self._clf = None; self._gate = None; self._auc = None; self._n = None

    # -- feature extraction (physics only; no duration) ------------------
    @staticmethod
    def _mean_slope(a, t):
        """NaN-safe mean + linear slope of a channel over the window."""
        g = np.isfinite(a)
        if g.sum() < 3:
            return (np.nan, np.nan)
        return (float(np.mean(a[g])),
                float(np.polyfit(t[g] - t[g][0], a[g], 1)[0]))

    def _winfeat(self, w):
        ip = w["Ip"].to_numpy(float) / 1e6            # MA
        t = w["time_s"].to_numpy(float)
        good = np.isfinite(ip)
        if good.sum() < 3:
            return None
        ipg, tg = np.abs(ip[good]), t[good]
        slope = np.polyfit(tg - tg[0], ipg, 1)[0] if len(tg) > 2 else 0.0
        ne = float(np.nanmean(w["ne"].to_numpy(float)))
        gw = ne / (np.mean(ipg) + 1e-6) if np.isfinite(ne) else np.nan
        base = [float(np.mean(ipg)), float(np.std(ipg)), float(slope), float(np.ptp(ipg)),
                ne, gw, float(np.nanmean(w["beta_n"])), float(np.nanmean(w["li"])),
                float(np.nanmean(w["q95"]))]
        # INDEPENDENT precursors — used automatically once f7 stages these columns
        def _col(c):
            return w[c].to_numpy(float) if c in w.columns else np.full(len(w), np.nan)
        n1m, n1s = self._mean_slope(_col("n1_rms"), t)
        prm, prs = self._mean_slope(_col("prad"), t)
        return base + [n1m, n1s, prm, prs]

    def _real_features(self):
        sh, lab = data.kward_real()
        labmap = {int(r.shot_id): (int(r.disruptive), r.t_disr_s) for r in lab.itertuples()}
        X, y, ids = [], [], []
        for sid, g in sh.groupby("shot_id"):
            if int(sid) not in labmap:
                continue
            disr, tdis = labmap[int(sid)]
            g = g.sort_values("time_s"); t = g["time_s"].to_numpy(float)
            ip = g["Ip"].to_numpy(float)
            if np.isfinite(ip).sum() < 10:
                continue
            if disr == 1 and np.isfinite(tdis):
                mask = (t >= tdis - _WINDOW_S) & (t < tdis)      # pre-quench precursor
            else:
                tc = t[np.nanargmax(np.where(np.isfinite(ip), np.abs(ip), -np.inf))]
                mask = (t >= tc - _WINDOW_S / 2) & (t < tc + _WINDOW_S / 2)  # flat-top
            if mask.sum() < 5:
                continue
            f = self._winfeat(g[mask])
            if f is None:
                continue
            X.append(f); y.append(disr); ids.append(int(sid))
        X = np.array(X, float); y = np.array(y, int)
        med = np.nanmedian(X, axis=0)
        bad = np.where(~np.isfinite(X)); X[bad] = np.take(med, bad[1])
        X[~np.isfinite(X)] = 0.0          # all-NaN optional columns (channels not staged yet)
        return X, y, np.array(ids)

    def _fit(self):
        if self._clf is None:
            X, y, ids = self._real_features()
            rng = np.random.default_rng(uq.SEED)
            perm = rng.permutation(len(y)); X, y = X[perm], y[perm]      # shot-level shuffle
            ntr = int(0.7 * len(y))
            self._clf = uq.CalibratedClassifier().fit(X[:ntr], y[:ntr])
            self._gate = uq.MahalanobisGate(0.05).fit(X[:ntr])
            from sklearn.metrics import roc_auc_score
            p = self._clf.predict_proba(X[ntr:])
            self._auc = float(roc_auc_score(y[ntr:], p))
            self._ece = metrics.ece_classification(y[ntr:], p)
            self._n = {"total": len(y), "disruptive": int(y.sum()), "test": len(y) - ntr}
            self._Xall, self._yall = X, y
            # honesty check: AUC WITHOUT the Ip-variability features (ip_std, ip_range),
            # which share information with the Ip-derived labels
            keep = [i for i in range(len(_FEATURES)) if i not in (1, 3)]
            c2 = uq.CalibratedClassifier().fit(X[:ntr][:, keep], y[:ntr])
            self._auc_precursor = float(roc_auc_score(y[ntr:], c2.predict_proba(X[ntr:][:, keep])))
        return self._clf

    def _predict(self, x):
        """x = 9-feature vector (see _FEATURES) -> calibrated disruption probability."""
        self._fit()
        x = np.asarray(x, float).reshape(1, -1)
        p = float(self._clf.predict_proba(x)[0])
        return Prediction(p, None, self._gate.in_domain(x),
                          note="calibrated disruption probability (real-device); "
                               "abstains -> hardware failsafe when out-of-domain")

    def benchmark(self):
        self._fit()
        out = {"member": self.name,
               "live_real_device": {
                   "source": "592 real MAST shots (FAIR-MAST), fixed 50 ms window",
                   "auc": round(self._auc, 3),
                   "auc_precursor_only_no_Ip_variability": round(self._auc_precursor, 3),
                   "ece": round(self._ece, 4),
                   "n_shots": self._n["total"], "n_disruptive": self._n["disruptive"],
                   "n_test_shots": self._n["test"],
                   "features": _FEATURES,
                   "caveats": ["labels DERIVED from Ip current-quench (heuristic, not "
                               "physicist-verified)",
                               "fixed-window physics features only (no duration leakage)",
                               "Ip-variability features SHARE information with the Ip-derived "
                               "label; the independent_precursor_only_auc below (n=1 Mirnov + "
                               "P_rad ONLY) is the label-independent number and is the one to cite",
                               "'n1_rms' is a broadband Mirnov fluctuation RMS (low-n MHD-activity "
                               "proxy), NOT a toroidal-n=1 Fourier decomposition (array geometry "
                               "on disk lacks toroidal-angle metadata)",
                               "EFIT kinetics sparse (~16%); n=1 Mirnov ~99% / P_rad ~78% shot "
                               "coverage (FAIR-MAST, pulled 2026-09-10)"]}}
        # independent-precursor validation: activates automatically once f7 stages the
        # n=1 Mirnov + P_rad channels (different diagnostics from Ip -> fully label-independent)
        from sklearn.metrics import roc_auc_score
        indep_present = bool(np.isfinite(self._Xall[:, _INDEP_IDX]).any()
                             and np.ptp(self._Xall[:, _INDEP_IDX]) > 0)
        out["live_real_device"]["independent_precursors_present"] = indep_present
        if indep_present:
            Xi = self._Xall[:, _INDEP_IDX]; ntr = int(0.7 * len(self._yall))
            ci = uq.CalibratedClassifier().fit(Xi[:ntr], self._yall[:ntr])
            out["live_real_device"]["independent_precursor_only_auc"] = round(
                float(roc_auc_score(self._yall[ntr:], ci.predict_proba(Xi[ntr:]))), 3)
            out["live_real_device"]["note_independent"] = ("n=1 Mirnov + P_rad present -> "
                "this AUC uses ONLY diagnostics independent of Ip and the Ip-derived label")
        else:
            out["live_real_device"]["note_independent"] = ("n=1 Mirnov + P_rad not staged yet; "
                "extractor is wired to use them automatically once present")
        # secondary: the calibration+OOD machinery on the self-contained synthetic set
        Xs, ys = _synth(600, seed=uq.SEED + 7)
        clf = uq.CalibratedClassifier().fit(*_synth(1200))
        from sklearn.metrics import roc_auc_score
        out["machinery_check_synth"] = {"auc": round(float(roc_auc_score(ys, clf.predict_proba(Xs))), 3)}
        out["sourced_twin"] = {"file": "track4_disruption/disruption_metrics.csv",
                               "note": "prior twin-synthetic result AUC 0.990 (superseded by real-device above)"}
        return out
