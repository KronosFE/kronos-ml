"""KAIROS — real-time closed-loop plasma control (MPC) + the safety clamp.

A finite-horizon receding-horizon controller (analytic per-axis QP — no external
solver) drives a reduced first-order plant to setpoint; every command is projected
onto the envelope by KGATE (the model-free clamp KAIROS supplies to the spine).
Includes the model-failure ("Exp-B") adversarial test: broken controllers emit
deliberately-unsafe commands and the clamp lets none escape.
"""
from __future__ import annotations

import time
import numpy as np

from .. import register, get
from ..base import Surrogate, Prediction
from .. import data, uq

# reduced first-order plant (from track5 plant_model): x_{k+1}=x+alpha*(u-x)+couple+noise
_ALPHA = np.array([0.20, 0.20, 0.10, 0.12, 0.05, 0.06])
_SIGMA = np.array([0.004, 0.004, 0.01, 0.01, 0.002, 0.02])   # per-step disturbance
_H = 8            # MPC horizon
_RHO = 0.02       # control-effort weight


def _mpc(x0, ref):
    """Analytic finite-horizon optimal constant-u command for the diagonal
    first-order plant (minimises tracking over H steps + rho*effort)."""
    x0 = np.asarray(x0, float); ref = np.asarray(ref, float)
    k = np.arange(1, _H + 1)
    u = np.zeros(6)
    for i in range(6):
        b = (1 - _ALPHA[i]) ** k          # decay of the initial-condition term
        c = 1 - b                          # coefficient on u
        d = x0[i] * b - ref[i]             # constant term
        u[i] = -np.sum(c * d) / (np.sum(c * c) + _RHO * _H)
    return u


def _plant_step(x, u_safe, rng):
    x = np.asarray(x, float)
    nxt = x + _ALPHA * (u_safe - x) + rng.normal(0, _SIGMA)
    nxt[2] -= 0.015 * (u_safe[5] - x[5])   # Ip -> beta_N coupling
    nxt[4] += 0.0012 * (u_safe[5] - x[5])  # Ip -> eps coupling
    return nxt


@register
class KAIROS(Surrogate):
    name = "KAIROS"; function = "CONTROL"; phase = 1; status = "BUILT"
    provenance = "TWIN"
    retired_by = "validated real-time plant controller (device-in-the-loop)"
    real_codes = ("receding-horizon MPC",)
    gates = ("AC-24", "AC-25")
    note = ("real-time closed-loop MPC control (analytic finite-horizon QP); "
            "supplies the model-free L4 clamp that KGATE enforces")

    def _predict(self, x):
        """x = (state6, ref6) -> next command from the MPC, clamped by KGATE."""
        state, ref = x
        u = _mpc(state, ref)
        safe, ok = get("KGATE").clamp(u)
        in_dom = bool(np.allclose(safe, u, atol=1e-6))
        return Prediction(safe, None, in_dom,
                          note="MPC step, clamped" if not in_dom else "MPC step, in envelope")

    def benchmark(self):
        rng = np.random.default_rng(uq.SEED)
        clamp = get("KGATE").clamp
        # (1) closed-loop tracking: drive from a start state to a setpoint
        ref = np.array([2.2, 0.3, 3.0, 1.0, 0.2, 30.0])
        x = np.array([2.0, 0.0, 2.0, 0.5, 0.1, 20.0])
        for _ in range(200):
            safe, _ = clamp(_mpc(x, ref)); x = _plant_step(x, safe, rng)
        rng_scale = np.array([1, 1, 4.2, 3, 0.4, 47.21])
        track_rms = float(np.sqrt(np.mean(((x - ref) / rng_scale) ** 2)))
        # (2) Exp-B: broken controllers emit unsafe commands; clamp must catch all
        from .spine import _LO, _HI
        adversaries = {
            "drift": lambda: _HI * (1 + 0.5 * rng.random(6)),
            "edge_seeker": lambda: _HI + 5,
            "nan_garbage": lambda: np.where(rng.random(6) < 0.3, np.nan, _HI * 3),
        }
        escapes = 0; steps = 0
        for _, gen in adversaries.items():
            for _ in range(50_000):
                safe, _ = clamp(gen()); steps += 1
                if not (np.all(safe >= _LO - 1e-9) and np.all(safe <= _HI + 1e-9)):
                    escapes += 1
        # (3) real latency of an MPC step + clamp
        t0 = time.perf_counter()
        for _ in range(2000):
            clamp(_mpc(x, ref))
        us = (time.perf_counter() - t0) / 2000 * 1e6
        return {"member": self.name,
                "live_tracking_rms_frac": round(track_rms, 4),
                "live_expB_model_failure": {"adversaries": list(adversaries),
                                            "steps": steps, "escapes": escapes,
                                            "catch_rate": 1.0 - escapes / steps},
                "live_mpc_step_us": round(us, 1),
                "sourced": {"file": "track5_control/clamp_activation_stats.csv",
                            "headline": "0 escapes / 150k steps / 3000 injected (AC-25); tracking <5%"}}
