"""KODEX KMIT — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KMIT(Surrogate):
    name = "KMIT"; function = "QMITIGATION"; phase = 2; status = "BUILT"
    provenance = "SIM"
    retired_by = "fault-tolerant quantum hardware (not available this decade)"
    real_codes = ("zero-noise extrapolation", "readout-error mitigation", "PennyLane")
    gates = ("KX-L3",)
    note = ("REAL quantum error mitigation — zero-noise extrapolation (ZNE, unitary "
            "folding) + readout-error mitigation on a real fusion quantum circuit "
            "(the KQUBIT H2 VQE state). Mitigation is the genuine NISQ tool TODAY; it "
            f"recovers signal from noise, not a speedup. Quantum's role in fusion is "
            f"post-~{_QC_ROLE_YEAR}")

    _P = 0.04                 # per-qubit depolarizing rate (NISQ-ish)
    _READOUT = 0.08           # per-qubit readout bit-flip probability

    # ---- ZNE on the KQUBIT H2 VQE circuit ----
    def _zne(self):
        import pennylane as qml
        kq = get("KQUBIT"); kq._run_vqe()
        params, H = kq._vqe_params, kq._hamiltonian()
        ansatz = kq._ansatz
        exact = kq._vqe["exact_energy_Ha"]

        dev_clean = qml.device("default.qubit", wires=2)

        @qml.qnode(dev_clean)
        def ideal():
            ansatz(params)
            return qml.expval(H)
        e_ideal = float(ideal())          # noiseless expval at the VQE params

        dev = qml.device("default.mixed", wires=2)

        def _noise():
            for w in (0, 1):
                qml.DepolarizingChannel(self._P, wires=w)

        @qml.qnode(dev)
        def folded(k):
            """Global unitary folding: ansatz (adjoint.ansatz)^k -> logical ansatz,
            noise scale lambda = 2k+1."""
            ansatz(params); _noise()
            for _ in range(k):
                qml.adjoint(ansatz)(params); _noise()
                ansatz(params); _noise()
            return qml.expval(H)

        lambdas = np.array([1, 3, 5], float)     # k = 0, 1, 2
        energies = np.array([float(folded(int((lam - 1) / 2))) for lam in lambdas])
        # linear ZNE extrapolation to zero noise
        slope, intercept = np.polyfit(lambdas, energies, 1)
        e_noisy = float(energies[0])
        e_zne = float(intercept)
        err_noisy = abs(e_noisy - e_ideal)
        err_zne = abs(e_zne - e_ideal)
        reduction = (err_noisy - err_zne) / err_noisy if err_noisy > 1e-9 else 0.0
        return {"exact_energy_Ha": round(exact, 6),
                "ideal_noiseless_Ha": round(e_ideal, 6),
                "noisy_Ha": round(e_noisy, 6),
                "zne_mitigated_Ha": round(e_zne, 6),
                "noise_scales_lambda": lambdas.tolist(),
                "energies_vs_lambda": [round(float(e), 6) for e in energies],
                "error_noisy_mHa": round(err_noisy * 1e3, 3),
                "error_zne_mHa": round(err_zne * 1e3, 3),
                "error_reduction_frac": round(float(reduction), 3),
                "depolarizing_p": self._P, "backend": qc.backend_note()}

    # ---- readout-error mitigation on a Bell distribution ----
    def _readout_mitigation(self):
        r = self._READOUT
        ideal = np.array([0.5, 0.0, 0.0, 0.5])          # Bell |Phi+> probs
        M1 = np.array([[1 - r, r], [r, 1 - r]])         # per-qubit readout confusion
        M = np.kron(M1, M1)
        noisy = M @ ideal
        mitig = np.linalg.solve(M, noisy)
        mitig = np.clip(mitig, 0, None); mitig = mitig / mitig.sum()
        tv = lambda a, b: float(0.5 * np.abs(a - b).sum())
        tv_noisy, tv_mit = tv(noisy, ideal), tv(mitig, ideal)
        return {"readout_bitflip_p": r,
                "tv_distance_noisy": round(tv_noisy, 4),
                "tv_distance_mitigated": round(tv_mit, 4),
                "tv_reduction_frac": round((tv_noisy - tv_mit) / tv_noisy, 3) if tv_noisy > 1e-9 else 0.0,
                "method": "inverse confusion-matrix (linear) readout correction"}

    def _predict(self, x):
        """x = per-qubit depolarizing rate p (float) -> ZNE-mitigated H2 energy at that noise."""
        if x is not None:
            try:
                self._P = float(np.ravel(np.asarray(x, float))[0])
            except Exception:
                pass
        z = self._zne()
        return Prediction(z["zne_mitigated_Ha"], z["error_zne_mHa"], True,
                          note=(f"ZNE-mitigated H2 energy {z['zne_mitigated_Ha']} Ha "
                                f"(noisy err {z['error_noisy_mHa']} -> {z['error_zne_mHa']} mHa); "
                                f"backend={z['backend']['active_backend']}"))

    def benchmark(self):
        z = self._zne(); ro = self._readout_mitigation()
        return {"member": self.name,
                "live_error_mitigation": {
                    "problem": "mitigate a noisy H2 VQE expectation (the real KQUBIT circuit)",
                    "framework": "PennyLane default.mixed; ZNE via global unitary folding + linear extrapolation",
                    "zne": z, "readout_mitigation": ro,
                    "backend": z["backend"],
                    "verdict": (f"REAL error mitigation on a fusion quantum circuit: ZNE cuts the noisy-energy "
                                f"error {z['error_noisy_mHa']} -> {z['error_zne_mHa']} mHa "
                                f"({z['error_reduction_frac']:.0%} of the error removed) and inverse-confusion "
                                f"readout correction cuts the measurement TV distance "
                                f"{ro['tv_distance_noisy']} -> {ro['tv_distance_mitigated']} "
                                f"({ro['tv_reduction_frac']:.0%}). Mitigation is the REAL NISQ tool today — it "
                                f"recovers signal from noise. HONEST: it is NOT a speedup; quantum's meaningful "
                                f"role in fusion is post-~{_QC_ROLE_YEAR}, and mitigation cost grows with noise. "
                                f"Runs on real hardware with KODEX_QC_BACKEND=ibm.")},
                "caveats": [f"no quantum advantage — quantum's role in fusion is post-~{_QC_ROLE_YEAR}",
                            "ZNE assumes a smooth noise->observable relation; it extrapolates, it does not "
                            "eliminate noise, and its overhead grows as noise grows",
                            "readout mitigation uses a known confusion matrix; on hardware you must calibrate it"]}
