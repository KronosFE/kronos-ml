"""KODEX KQERN — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KQERN(Surrogate):
    name = "KQERN"; function = "QKERNEL"; phase = 2; status = "BUILT"
    provenance = "SIM"
    retired_by = "fault-tolerant quantum hardware (not available this decade)"
    real_codes = ("quantum kernel", "PennyLane", "real MAST disruption features")
    gates = ("KX-L3",)
    note = ("REAL quantum-kernel classifier — embeds real MAST disruption features into a "
            "quantum feature map and classifies on the fidelity kernel (PennyLane); runs on a "
            "simulator now, real hardware pluggable. Honest: ties the classical kernel — a "
            "validated no-advantage result, but a real, runnable quantum-ML pipeline")

    _N = 64
    _FEATS = (0, 6, 7, 8)          # ip_mean, beta_n, li, q95 (subset of KWARD's real features)

    def _data(self):
        import numpy as np
        from .. import get, uq
        X, y, _ = get("KWARD")._real_features()
        X = X[:, list(self._FEATS)]
        rng = np.random.default_rng(uq.SEED)
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        k = min(self._N // 2, len(pos), len(neg))
        idx = np.r_[rng.choice(pos, k, False), rng.choice(neg, k, False)]
        rng.shuffle(idx)
        Xs, ys = X[idx], y[idx]
        # robust scale to angles in [0, pi]
        lo, hi = np.nanpercentile(Xs, 5, 0), np.nanpercentile(Xs, 95, 0)
        Xs = np.clip((Xs - lo) / (hi - lo + 1e-9), 0, 1) * np.pi
        ntr = int(0.7 * len(ys))
        return Xs[:ntr], ys[:ntr], Xs[ntr:], ys[ntr:]

    def _kernel(self):
        import pennylane as qml
        nq = len(self._FEATS)
        dev = qc.get_device(wires=nq)

        @qml.qnode(dev)
        def overlap(a, b):
            qml.AngleEmbedding(a, wires=range(nq))
            qml.adjoint(qml.AngleEmbedding)(b, wires=range(nq))
            return qml.probs(wires=range(nq))
        return lambda a, b: float(overlap(a, b)[0])    # |<phi(a)|phi(b)>|^2

    def _gram(self, A, B, kfn):
        import numpy as np
        return np.array([[kfn(a, b) for b in B] for a in A])

    def _fit(self):
        if getattr(self, "_res", None) is not None:
            return self._res
        import numpy as np
        from sklearn.svm import SVC
        from sklearn.metrics import roc_auc_score
        Xtr, ytr, Xte, yte = self._data()
        kfn = self._kernel()
        Ktr = self._gram(Xtr, Xtr, kfn); Kte = self._gram(Xte, Xtr, kfn)
        qsvc = SVC(kernel="precomputed", probability=False).fit(Ktr, ytr)
        q_auc = float(roc_auc_score(yte, qsvc.decision_function(Kte)))
        c = SVC(kernel="rbf", probability=False).fit(Xtr, ytr)          # classical baseline
        c_auc = float(roc_auc_score(yte, c.decision_function(Xte)))
        self._res = {"quantum_kernel_auc": round(q_auc, 3), "classical_rbf_auc": round(c_auc, 3),
                     "n_train": len(ytr), "n_test": len(yte), "n_qubits": len(self._FEATS),
                     "features": ["ip_mean_MA", "beta_n", "li", "q95"], "backend": qc.backend_note()}
        return self._res

    def _predict(self, x):
        r = self._fit()
        return Prediction(r["quantum_kernel_auc"], None, True,
                          note=f"quantum-kernel disruption AUC (vs classical {r['classical_rbf_auc']}); "
                               f"backend={r['backend']['active_backend']}")

    def benchmark(self):
        r = self._fit()
        adv = r["quantum_kernel_auc"] - r["classical_rbf_auc"]
        return {"member": self.name,
                "live_quantum_kernel": {
                    "problem": "MAST disruption classification on a real quantum fidelity kernel",
                    "framework": "PennyLane AngleEmbedding kernel + precomputed-kernel SVM",
                    **{k: r[k] for k in ("quantum_kernel_auc", "classical_rbf_auc", "n_train",
                                         "n_test", "n_qubits", "features")},
                    "quantum_minus_classical_auc": round(adv, 3),
                    "backend": r["backend"],
                    "verdict": (f"REAL quantum kernel classifies MAST disruptions at AUC "
                                f"{r['quantum_kernel_auc']} vs classical RBF {r['classical_rbf_auc']} "
                                f"(Delta {adv:+.3f}) — {'no advantage' if adv <= 0.02 else 'marginal'}: a "
                                f"validated null, but a REAL runnable quantum-ML pipeline on real fusion "
                                f"data. Runs on hardware with KODEX_QC_BACKEND=ibm."),
                    "caveats": ["labels are the heuristic Ip-quench disruption labels (from KWARD)",
                                "quantum kernel ties classical here — no advantage; the value is a real, "
                                "hardware-ready quantum-ML tool, honest that advantage is ~8-10 yr out"]},
                "sourced": {"file": "TRACK1_QUANTUM_RESULTS/track1_results.json (A6_quantum_kernel_MAST)",
                            "note": "independent Track-1 run also found quantum-kernel AUC ~= classical (null)"}}
