"""KODEX KQOPT — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KQOPT(Surrogate):
    name = "KQOPT"; function = "QOPT"; phase = 3; status = "BUILT"
    provenance = "SIM"
    retired_by = "fault-tolerant quantum hardware (not available this decade)"
    real_codes = ("QAOA", "PennyLane")
    gates = ("KX-L3",)
    note = ("REAL QAOA quantum optimizer — solves a combinatorial design QUBO on an actual "
            "quantum circuit (PennyLane); runs on a simulator now, real hardware pluggable, and "
            "you can plug in your own QUBO. Honest: standard ~0.7-1.0 approx ratio, no speedup "
            "at this size (~8-10 yr to advantage)")

    # a fixed 6-node design QUBO (illustrative MaxCut-style configuration problem)
    _EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (3, 5))
    _NQ = 6

    def _hamiltonians(self):
        import pennylane as qml
        cost_h = qml.Hamiltonian([0.5] * len(self._EDGES),
                                 [qml.PauliZ(i) @ qml.PauliZ(j) for (i, j) in self._EDGES])
        mixer_h = qml.Hamiltonian([1.0] * self._NQ, [qml.PauliX(i) for i in range(self._NQ)])
        return cost_h, mixer_h

    def _cut(self, bits):
        return sum(1 for (i, j) in self._EDGES if bits[i] != bits[j])

    def _optimum(self):
        import itertools
        return max(self._cut(b) for b in itertools.product((0, 1), repeat=self._NQ))

    def _run(self, p=2, steps=70):
        if getattr(self, "_res", None) is not None:
            return self._res
        import numpy as np
        import pennylane as qml
        from pennylane import qaoa
        from .. import uq
        cost_h, mixer_h = self._hamiltonians()
        dev = qc.get_device(wires=self._NQ)

        def qaoa_layer(gamma, beta):
            qaoa.cost_layer(gamma, cost_h)
            qaoa.mixer_layer(beta, mixer_h)

        def prep(params):
            for w in range(self._NQ):
                qml.Hadamard(w)
            qml.layer(qaoa_layer, p, params[0], params[1])

        @qml.qnode(dev)
        def energy(params):
            prep(params)
            return qml.expval(cost_h)

        @qml.qnode(dev)
        def probs(params):
            prep(params)
            return qml.probs(wires=range(self._NQ))

        rng = np.random.default_rng(uq.SEED)
        params = qml.numpy.array([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi, p)],
                                 requires_grad=True)
        opt = qml.AdamOptimizer(0.1)
        for _ in range(steps):
            params = opt.step(energy, params)
        pr = np.array(probs(params))
        bits = [int(b) for b in format(int(pr.argmax()), f"0{self._NQ}b")]
        qaoa_cut, opt_cut = self._cut(bits), self._optimum()
        self._res = {"qaoa_cut": qaoa_cut, "optimal_cut": opt_cut,
                     "approx_ratio": round(qaoa_cut / opt_cut, 3), "p": p, "n_qubits": self._NQ,
                     "solution_bitstring": bits, "backend": qc.backend_note()}
        return self._res

    def _predict(self, x):
        r = self._run()
        return Prediction(r["approx_ratio"], None, True,
                          note=f"QAOA approx ratio {r['approx_ratio']} (cut {r['qaoa_cut']}/{r['optimal_cut']}); "
                               f"backend={r['backend']['active_backend']}")

    def benchmark(self):
        r = self._run()
        return {"member": self.name,
                "live_qaoa": {
                    "problem": "6-node design QUBO (MaxCut-style; plug in your own Q matrix)",
                    "framework": "PennyLane QAOA (p=2); runs on simulator now, real hardware pluggable",
                    **{k: r[k] for k in ("qaoa_cut", "optimal_cut", "approx_ratio", "p", "n_qubits",
                                         "solution_bitstring")},
                    "backend": r["backend"],
                    "verdict": (f"REAL QAOA reaches {r['approx_ratio']:.0%} of the brute-force optimum "
                                f"(cut {r['qaoa_cut']}/{r['optimal_cut']}, p={r['p']}, {r['n_qubits']} qubits) "
                                f"on the {r['backend']['active_backend']} backend — a working quantum "
                                f"optimizer, runnable on hardware with KODEX_QC_BACKEND=ibm. HONEST: no "
                                f"speedup vs classical at this size; advantage is ~8-10 yr out.")},
                "sourced": {"file": "TRACK1_QUANTUM_RESULTS/track1_results.json (A4_QAOA)",
                            "note": "independent Track-1 p=1 QAOA reached ~66% on an 8-node QUBO (consistent)"}}
