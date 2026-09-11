"""KODEX KDRIVE — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KDRIVE(Surrogate):
    name = "KDRIVE"; function = "RL-control"; phase = 3; status = "BUILT"
    provenance = "TWIN"
    retired_by = "device-validated learned controller"
    real_codes = ("reinforcement learning",)
    note = ("reinforcement-learning controller — a feedback policy learned by the "
            "cross-entropy method in the twin plant loop; clamped by KGATE")

    def __init__(self):
        self._K = None

    def _fit(self):
        if self._K is None:
            ref = np.array([2.2, 0.3, 3.0, 1.0, 0.2, 30.0])
            rng = np.random.default_rng(uq.SEED)
            mu = np.full(6, 0.5); sd = np.full(6, 0.4)      # policy = per-axis gain
            for _ in range(20):                              # cross-entropy iterations
                pop = np.clip(rng.normal(mu, sd, size=(30, 6)), 0.0, 1.5)
                costs = np.array([_rollout(k, ref, seed=1) for k in pop])
                elite = pop[np.argsort(costs)[:6]]
                mu, sd = elite.mean(0), elite.std(0) + 1e-3
            self._K = mu
        return self._K

    def _predict(self, x):
        K = self._fit()
        state, ref = x
        state = np.asarray(state, float).ravel(); ref = np.asarray(ref, float).ravel()
        safe, _ = get("KGATE").clamp(state + K * (ref - state))
        return Prediction(safe, None, True, note="learned RL policy command (clamped)")

    def benchmark(self):
        K = self._fit()
        ref = np.array([2.2, 0.3, 3.0, 1.0, 0.2, 30.0])
        learned = _rollout(K, ref, seed=7)
        naive = _rollout(np.full(6, 0.2), ref, seed=7)
        return {"member": self.name,
                "live": {"learned_policy_tracking_rms": round(float(learned), 4),
                         "naive_gain_tracking_rms": round(float(naive), 4),
                         "learned_gains": [round(float(k), 3) for k in K],
                         "note": "cross-entropy-method policy search, twin-in-the-loop"}}
