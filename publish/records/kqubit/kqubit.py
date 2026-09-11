"""KODEX KQUBIT — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KQUBIT(Surrogate):
    name = "KQUBIT"; function = "QML"; phase = 1; status = "BUILT"
    provenance = "SIM"
    retired_by = "fault-tolerant quantum hardware (not available this decade)"
    real_codes = ("VQE", "quantum chemistry", "PennyLane")
    gates = ("AC-44", "KX-L3-A4")
    note = ("REAL variational quantum eigensolver (VQE) for a molecular Hamiltonian — a "
            "runnable quantum program (PennyLane): executes on a simulator now and on real "
            "quantum hardware when you plug in a backend. Honest: no quantum advantage yet "
            "(~8-10 yr out); the tooling + testing are real TODAY")

    # H2 molecular Hamiltonian, 2-qubit parity-reduced (standard Qiskit/IBM textbook coeffs, Ha)
    _COEFFS = (-1.052373245772859, 0.39793742484318045, -0.39793742484318045,
               -0.01128010425623538, 0.18093119978423156)

    def _hamiltonian(self):
        import pennylane as qml
        ops = [qml.Identity(0), qml.PauliZ(0), qml.PauliZ(1),
               qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)]
        return qml.Hamiltonian(list(self._COEFFS), ops)

    @staticmethod
    def _ansatz(params):
        import pennylane as qml
        qml.PauliX(0)                        # Hartree-Fock reference |10>
        qml.RY(params[0], 0); qml.RY(params[1], 1)
        qml.CNOT([0, 1])
        qml.RY(params[2], 0); qml.RY(params[3], 1)

    def _run_vqe(self, steps=120):
        """Actually optimize the VQE on the active backend; cache the result."""
        if getattr(self, "_vqe", None) is not None:
            return self._vqe
        import numpy as np
        import pennylane as qml
        from .. import uq
        H = self._hamiltonian()
        exact = float(np.linalg.eigvalsh(qml.matrix(H))[0])       # ground truth (exact diag)
        dev = qc.get_device(wires=2)
        ansatz = self._ansatz

        @qml.qnode(dev)
        def cost(p):
            ansatz(p)
            return qml.expval(H)

        rng = np.random.default_rng(uq.SEED)
        p = qml.numpy.array(rng.normal(0, 0.1, 4), requires_grad=True)
        opt = qml.AdamOptimizer(0.1)
        for _ in range(steps):
            p = opt.step(cost, p)
        e = float(cost(p))
        self._vqe_params = p                      # cache for the noise sweep
        self._vqe = {"vqe_energy_Ha": round(e, 6), "exact_energy_Ha": round(exact, 6),
                     "recovery_mHa": round(abs(e - exact) * 1e3, 4),
                     "n_qubits": 2, "steps": steps, "backend": qc.backend_note()}
        return self._vqe

    def _noise_sweep(self, ps=(0.0, 0.001, 0.005, 0.01, 0.02, 0.05)):
        """Re-evaluate the optimized VQE under per-qubit depolarizing noise -> mHa error vs p."""
        import pennylane as qml
        self._run_vqe()
        params, H = self._vqe_params, self._hamiltonian()
        exact = self._vqe["exact_energy_Ha"]
        out = {}
        for p in ps:
            dev = qml.device("default.mixed", wires=2)

            @qml.qnode(dev)
            def noisy(par):
                self._ansatz(par)
                for w in (0, 1):
                    qml.DepolarizingChannel(p, wires=w)
                return qml.expval(H)
            out[p] = round(abs(float(noisy(params)) - exact) * 1e3, 3)   # mHa error
        return out

    def _predict(self, x):
        r = self._run_vqe()
        return Prediction(r["vqe_energy_Ha"], None, True,
                          note=f"VQE ground-state energy (recovery {r['recovery_mHa']} mHa vs exact); "
                               f"backend={r['backend']['active_backend']}")

    def benchmark(self):
        r = self._run_vqe()
        chem_acc = r["recovery_mHa"] <= 1.6
        return {"member": self.name,
                "live_vqe": {
                    "problem": "H2 molecular Hamiltonian (2-qubit parity-reduced; standard coeffs)",
                    "framework": "PennyLane (runnable on simulator now; real hardware pluggable)",
                    **{k: r[k] for k in ("vqe_energy_Ha", "exact_energy_Ha", "recovery_mHa",
                                         "n_qubits", "steps")},
                    "reaches_chemical_accuracy": bool(chem_acc),
                    "nisq_noise_sweep_mHa_error": self._noise_sweep(),
                    "noise_note": ("depolarizing-noise sweep (mHa error vs per-qubit p): NISQ noise breaks "
                                   "the 1.6 mHa chemical accuracy fast — this is WHY there is no advantage yet"),
                    "backend": r["backend"],
                    "verdict": (f"REAL VQE recovers the H2 ground state to {r['recovery_mHa']} mHa on "
                                f"the {r['backend']['active_backend']} backend "
                                f"({'within' if chem_acc else 'outside'} 1.6 mHa chemical accuracy). "
                                f"Runs on real quantum hardware when you set KODEX_QC_BACKEND=ibm. "
                                f"HONEST: no quantum advantage this decade — the value now is a real, "
                                f"testable quantum pipeline, not a speedup.")},
                "sourced": {"file": "track8_vqe_poc/vqe_convergence.csv",
                            "note": "prior noisy-VQE sweep: chemical accuracy breaks under NISQ noise "
                                    "(quantifies the no-advantage-this-decade caveat)"},
                "caveat": "MANDATORY: no quantum advantage this decade; hardware value is ~8-10 yr out"}
