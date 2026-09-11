"""KODEX KGATE — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KGATE(Surrogate):
    name = "KGATE"; function = "SAFE"; phase = 1; status = "BUILT"
    provenance = "TWIN"
    retired_by = "formal CBF verification + hardware failsafe"
    real_codes = ("CBF certificate", "hardware failsafe")
    note = ("trust-boundary / abstention gate — a model-free envelope clamp that "
            "decides in_domain and fail-closes on bad input")

    def clamp(self, u):
        """Project a proposed 6-command onto the frozen envelope; fail-closed to
        the low-corner HOLD on non-finite / wrong-shape input."""
        u = np.asarray(u, float).ravel()
        if u.shape != (6,) or not np.all(np.isfinite(u)):
            return _LO.copy(), False           # fail-closed HOLD
        safe = np.clip(u, _LO, _HI)
        # Greenwald density limit n_e <= Ip/pi (Ip in MA, n_e in 1e20 m^-3)
        safe[3] = min(safe[3], safe[5] / np.pi)
        return safe, True

    def in_domain(self, u):
        u = np.asarray(u, float).ravel()
        if u.shape != (6,) or not np.all(np.isfinite(u)):
            return False
        safe, _ = self.clamp(u)
        return bool(np.allclose(safe, u, atol=1e-9))

    def _predict(self, x):
        safe, ok = self.clamp(x)
        return Prediction(safe, None, self.in_domain(x),
                          note="clamped to envelope" if not self.in_domain(x) else "in envelope")

    def benchmark(self):
        """Reproduce the AC-25 headline: inject deliberate violations and confirm
        the clamp lets zero escape (a real, model-free numpy check)."""
        rng = np.random.default_rng(uq.SEED)
        n = 150_000
        # commands drawn wide (deliberately outside the envelope ~half the time)
        span = _HI - _LO
        u = _LO + rng.uniform(-0.5, 1.5, size=(n, 6)) * span
        # inject NaN garbage in a slice
        u[::7, rng.integers(0, 6)] = np.nan
        escapes = 0
        for i in range(n):
            safe, _ = self.clamp(u[i])
            if not (np.all(safe >= _LO - 1e-9) and np.all(safe <= _HI + 1e-9)
                    and safe[3] <= safe[5] / np.pi + 1e-9):
                escapes += 1
        return {"member": self.name, "steps": n, "escapes": escapes,
                "catch_rate": 1.0 - escapes / n,
                "note": "model-free clamp; safety independent of any model"}
