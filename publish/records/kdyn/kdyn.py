"""KODEX KDYN — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KDYN(Surrogate):
    name = "KDYN"; function = "QDYN"; phase = 3; status = "BUILT"
    provenance = "SIM"
    retired_by = "fault-tolerant quantum hardware (not available this decade)"
    real_codes = ("Trotterized Hamiltonian simulation", "PennyLane")
    gates = ("KX-L3",)
    note = ("REAL quantum dynamics — Trotterized time-evolution of a transverse-field Ising "
            "'kinetic' Hamiltonian (PennyLane); the honest 1/n_steps cost curve for simulating "
            "plasma-like dynamics on a quantum computer. Runs on a simulator now, hardware pluggable")

    _NQ = 3; _J = 1.0; _HX = 0.8; _T = 1.0

    def _ham(self):
        import pennylane as qml
        coeffs = [-self._J] * (self._NQ - 1) + [-self._HX] * self._NQ
        ops = ([qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(self._NQ - 1)]
               + [qml.PauliX(i) for i in range(self._NQ)])
        return qml.Hamiltonian(coeffs, ops)

    def _curve(self):
        import numpy as np
        import pennylane as qml
        from scipy.linalg import expm
        H = self._ham()
        coeffs, ops = H.terms()
        mats = [(float(c), qml.matrix(o, wire_order=range(self._NQ))) for c, o in zip(coeffs, ops)]
        Hm = sum(c * M for c, M in mats)
        Uex = expm(-1j * self._T * Hm)
        out = {}
        for n in (1, 2, 4, 8, 16, 32):
            dt = self._T / n
            Ustep = np.eye(2 ** self._NQ, dtype=complex)
            for c, M in mats:
                Ustep = expm(-1j * c * dt * M) @ Ustep
            Utr = np.linalg.matrix_power(Ustep, n)
            out[n] = round(float(np.linalg.norm(Utr - Uex, 2)), 5)
        return out

    def _backend_run(self):
        import pennylane as qml
        H = self._ham(); dev = qc.get_device(wires=self._NQ)

        @qml.qnode(dev)
        def circ(n):
            qml.ApproxTimeEvolution(H, self._T, n)
            return qml.expval(qml.PauliZ(0))
        return float(circ(16))

    def _predict(self, x):
        import numpy as np
        c = self._curve()
        n = int(np.ravel(np.asarray(x, float))[0]) if x is not None else 16
        key = min(c, key=lambda k: abs(k - n))
        return Prediction(c[key], None, True, note=f"Trotter spectral-norm error at {key} steps")

    def benchmark(self):
        c = self._curve()
        z0 = self._backend_run()
        st = sorted(c)
        return {"member": self.name,
                "live_trotter": {
                    "system": f"{self._NQ}-qubit transverse-field Ising (J={self._J}, h={self._HX}), t={self._T}",
                    "framework": "PennyLane Trotter (ApproxTimeEvolution); sim now, real hardware pluggable",
                    "spectral_norm_error_vs_steps": c,
                    "converges_as": "~1/n_steps (1st-order Trotter, as expected)",
                    "backend_expval_Z0_at_16_steps": round(z0, 4),
                    "backend": qc.backend_note(),
                    "verdict": (f"REAL Trotterized quantum dynamics: 1st-order error falls {c[st[0]]} -> "
                                f"{c[st[-1]]} from {st[0]} to {st[-1]} steps (~1/n) — the honest cost curve for "
                                f"simulating plasma-like Hamiltonian dynamics on a quantum computer. Runs on "
                                f"real hardware with KODEX_QC_BACKEND=ibm. No advantage at this size; a real, "
                                f"testable pipeline.")},
                "sourced": {"file": "TRACK1_QUANTUM_RESULTS/track1_results.json (A8_Trotter)",
                            "note": "independent Track-1: 1st-order Trotter error 1.257->0.032 with steps (consistent)"}}
