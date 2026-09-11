"""KODEX KSENSE — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KSENSE(Surrogate):
    name = "KSENSE"; function = "QSENSE"; phase = 2; status = "BUILT"
    provenance = "SIM"
    retired_by = "deployed physical diagnostic hardware"
    real_codes = ("quantum Fisher information", "GHZ metrology", "PennyLane")
    gates = ("HX-29",)
    note = ("REAL quantum-metrology evaluation (PennyLane mixed-state) — GHZ interferometry: ideal "
            "gives Heisenberg (gain ~ N) scaling, but realistic dephasing (NV/SQUID/SERF reality) "
            "collapses it back to the standard quantum limit (~sqrt(N)). Honest null where fusion "
            "disruptions live")

    _P = 0.2                                  # per-qubit dephasing (realistic sensor decoherence)

    def _visibility(self, N, p):
        """GHZ interferometric visibility (X-parity contrast) under per-qubit dephasing."""
        import pennylane as qml
        dev = qml.device("default.mixed", wires=N)
        obs = qml.PauliX(0)
        for i in range(1, N):
            obs = obs @ qml.PauliX(i)

        @qml.qnode(dev)
        def sig():
            qml.Hadamard(0)
            for i in range(N - 1):
                qml.CNOT([0, i + 1])          # GHZ_N
            for i in range(N):
                qml.PhaseDamping(p, wires=i)  # dephasing channel
            return qml.expval(obs)
        return abs(float(sig()))

    def _scaling(self, Ns=(2, 3, 4, 5, 6, 7, 8)):
        import numpy as np
        rows, peak_N, peak_gain = [], None, -1.0
        for N in Ns:
            Vd = self._visibility(N, self._P)
            gain_ideal, gain_deph, gain_sql = float(N), N * Vd, float(np.sqrt(N))
            if gain_deph > peak_gain:
                peak_gain, peak_N = gain_deph, N
            rows.append({"N": N, "visibility_dephased": round(Vd, 3),
                         "GHZ_ideal_gain": round(gain_ideal, 2),
                         "GHZ_dephased_gain": round(gain_deph, 2),
                         "SQL_gain": round(gain_sql, 2)})
        return rows, peak_N

    def _predict(self, x):
        import numpy as np
        N = int(np.ravel(np.asarray(x, float))[0]) if x is not None else 4
        Vd = self._visibility(max(2, N), self._P)
        return Prediction(round(max(2, N) * Vd, 3), None, True,
                          note=f"GHZ dephased metrological gain at N={max(2,N)} (SQL={np.sqrt(N):.2f})")

    def benchmark(self):
        rows, peak_N = self._scaling()
        a, b = rows[0], rows[-1]
        g_ideal = round(b["GHZ_ideal_gain"] / a["GHZ_ideal_gain"], 2)     # ~ N growth (Heisenberg)
        g_deph = round(b["GHZ_dephased_gain"] / a["GHZ_dephased_gain"], 2)
        g_sql = round(b["SQL_gain"] / a["SQL_gain"], 2)                    # ~ sqrt(N)
        return {"member": self.name,
                "live_quantum_metrology": {
                    "method": "PennyLane mixed-state GHZ interferometry; Heisenberg vs SQL vs dephasing",
                    "dephasing_per_qubit": self._P,
                    "scaling": rows,
                    "gain_growth_N2_to_N8": {"GHZ_ideal": g_ideal, "GHZ_dephased": g_deph, "SQL": g_sql},
                    "analytic_turnover_N": peak_N,
                    "backend": qc.backend_note(),
                    "verdict": (f"REAL quantum-metrology computation: over N=2->8 the ideal GHZ gain grows "
                                f"{g_ideal}x (Heisenberg ~ N), but with per-qubit dephasing (p={self._P}) the "
                                f"dephased GHZ grows only {g_deph}x — essentially the SQL rate ({g_sql}x, "
                                f"~sqrt(N)). Dephasing ERASES the Heisenberg *scaling* to a bounded constant "
                                f"(gain turns over near N~{peak_N + 1}). HONEST NULL: no quantum-sensing "
                                f"advantage for fusion disruptions this decade — but a real, runnable "
                                f"metrology tool.")},
                "sourced": {"file": "TRACK1_QUANTUM_RESULTS/track1_results.json (A7_metrology)",
                            "note": "independent Track-1: correlated dephasing erases GHZ Heisenberg back to SQL"}}
