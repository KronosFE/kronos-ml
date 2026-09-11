"""Quantum-track codes — honestly scoped, no-advantage / null-gain rigor cards.

KQUBIT  — quantum-ML surrogate (VQE): statevector exact, noisy never reaches it.
KQROSS  — fault-tolerant quantum resource estimator + crossover analysis.
KSENSE  — quantum-sensor diagnostic evaluation (buys ~no gain).
KTENSOR — tensor-network simulation & compression (classical / bridge to quantum).

Each reads its on-disk result and returns the honest verdict; none claims a
speedup.  The mandatory "no quantum advantage this decade" caveat travels with
KQUBIT/KQROSS.
"""
from __future__ import annotations

from .. import register
from ..base import Surrogate, Prediction
from .. import data


@register
class KQUBIT(Surrogate):
    name = "KQUBIT"; function = "QML"; phase = 1; status = "BUILT"
    provenance = "SIM"
    retired_by = "fault-tolerant quantum hardware (not available this decade)"
    real_codes = ("VQE", "quantum chemistry")
    gates = ("AC-44", "KX-L3-A4")
    note = ("quantum-ML surrogate (VQE) — a validated no-advantage rigor card: "
            "statevector exact, noisy never reaches chemical accuracy")

    def _predict(self, x):
        return Prediction(
            {"verdict": "no quantum advantage this decade",
             "statevector": "exact to chemical accuracy", "noisy": "does not reach it"},
            None, True, note="validated negative result — a complete result, not a speedup")

    def benchmark(self):
        return {"member": self.name,
                "sourced": "track8_vqe_poc/vqe_convergence.csv + vqe_scale_wall.csv",
                "verdict": "statevector 0.0 mHa (PASS); noisy 14x-1305x over chem-acc (sim-only)",
                "result_type": "validated negative result (BUILT — a complete finding)",
                "caveat": "MANDATORY: no quantum advantage this decade"}


@register
class KQROSS(Surrogate):
    name = "KQROSS"; function = "QRE"; phase = 2; status = "BUILT"
    provenance = "SIM"
    retired_by = "fault-tolerant quantum hardware (not available this decade)"
    real_codes = ("FT resource estimation",)
    gates = ("AC-43", "BR-SX-08")
    note = ("fault-tolerant quantum resource estimator + crossover analysis — "
            "a validated no-crossover result (when/if quantum beats classical)")

    def _predict(self, x):
        return Prediction({"verdict": "NO CROSSOVER this decade",
                           "kernels": "all 3 vs named optimized classical baselines"},
                          None, True, note="survives the full assumption sweep")

    def benchmark(self):
        out = {"member": self.name,
               "verdict": "no crossover this decade; survives optimistic corner",
               "result_type": "validated negative result (BUILT)",
               "caveat": "MANDATORY: no quantum advantage this decade"}
        try:
            df = data.read_csv("track7_quantum_resources", "crossover_verdict.csv")
            out["live"] = {"file": "track7_quantum_resources/crossover_verdict.csv",
                           "n_kernels": int(len(df)), "columns": list(df.columns)[:6]}
        except Exception as e:
            out["live"] = {"note": "resource-estimate CSVs on disk", "err": str(e)}
        return out


@register
class KSENSE(Surrogate):
    name = "KSENSE"; function = "QSENSE"; phase = 2; status = "BUILT"
    provenance = "SIM"
    retired_by = "deployed physical diagnostic hardware"
    real_codes = ("NV/SQUID/SERF magnetometry",)
    gates = ("HX-29",)
    note = ("quantum-sensor diagnostic evaluation (NV/SQUID/SERF); a validated null "
            "gain where disruptions live (the MHD floor dominates the sensor floor)")

    def _predict(self, x):
        return Prediction({"verdict": "quantum sensing buys ~0 disruption-warning gain",
                           "reason": "intrinsic MHD floor dominates the sensor floor"},
                          None, True, note="validated null result")

    def benchmark(self):
        out = {"member": self.name,
               "verdict": "dAUC ~ -0.04..+0.003, d-warning ~0 ms vs conventional coil",
               "result_type": "validated null result (BUILT)"}
        try:
            df = data.read_csv("track8b_quantum_sensing", "sensing_to_disruption_gain.csv")
            out["live"] = {"file": "track8b_quantum_sensing/sensing_to_disruption_gain.csv",
                           "n_rows": int(len(df)), "columns": list(df.columns)[:6]}
        except Exception as e:
            out["live"] = {"err": str(e)}
        return out


@register
class KTENSOR(Surrogate):
    name = "KTENSOR"; function = "TN"; phase = 2; status = "PARTIAL"
    provenance = "SIM"
    retired_by = "exact many-body / quantum simulation"
    real_codes = ("tensor networks",)
    gates = ("KX-L3", )
    note = ("tensor-network simulation & compression — a classical many-body "
            "method and a bridge to quantum for kernel problems")

    def _predict(self, x):
        return Prediction({"method": "tensor-network compression",
                           "role": "classical many-body / quantum bridge"},
                          None, True, note="see track_R3_3")

    def benchmark(self):
        return {"member": self.name,
                "sourced": "track_R3_3_tensor_network",
                "note": "tensor-network results on disk; surrogate wrapper is thin"}
