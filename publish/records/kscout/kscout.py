"""KODEX KSCOUT — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KSCOUT(Surrogate):
    name = "KSCOUT"; function = "MF-CAMPAIGN"; phase = 3; status = "BUILT"
    provenance = "SIM"
    retired_by = "the full multi-fidelity CGYRO campaign (twin + mu=400 + real-mass gold)"
    real_codes = ("multi-fidelity Bayesian optimization", "cost-aware acquisition")
    gates = ("BR-L2-A1e",)
    note = ("multi-fidelity Bayesian-optimization campaign manager — extends KSEEK: chooses which "
            "operating point AND which fidelity (reduced twin / mu=400 CGYRO / real-mass gold) to "
            "run next, greedily maximizing truth-weighted information per GPU-hour under a budget")

    _BOX = ((2.0, 3.5), (0.4, 1.6))          # a_LT, shear (matches KYRO / KSEEK)
    _C0 = 0.10                                # fixed per-run setup overhead (GPU-h equiv.)
    _HOTSPOTS = 16                            # target uncertain locations for a campaign
    # honest fidelity model: measured mu=400 cost; twin ~free; gold ~10x mu=400 (estimate, not yet run).
    # rho = ASSUMED correlation-with-truth per fidelity (gold=1 by definition); labelled as assumptions.
    _FIDELITIES = {
        "twin":         {"gpu_h": 0.0,  "rho": 0.55, "desc": "reduced analytic twin (near-free triage)"},
        "cgyro_mu400":  {"gpu_h": 2.89, "rho": 0.90, "desc": "CGYRO representative mu=400 (measured GPU-h)"},
        "gold_mu3672":  {"gpu_h": 28.9, "rho": 1.00, "desc": "CGYRO real-mass gold (est. ~10x mu=400; not yet run)"},
    }

    def _grid(self, n=31):
        a = np.linspace(*self._BOX[0], n); s = np.linspace(*self._BOX[1], n)
        A, S = np.meshgrid(a, s)
        return np.c_[A.ravel(), S.ravel()]

    def _hotspots(self, n_hotspots, min_sep=0.16):
        """The n most-uncertain operating points (KYRO GP std), min-separated + away from data."""
        gp = get("KYRO")._fit()
        G = self._grid()
        _, std = gp.predict(G); std = np.asarray(std).ravel()
        Xtr = data.cgyro_flux_map_final()[["a_LT", "shear"]].to_numpy(float)
        span = np.array([self._BOX[0][1] - self._BOX[0][0], self._BOX[1][1] - self._BOX[1][0]])
        picks = []
        for idx in np.argsort(-std):
            g = G[idx]
            d_tr = np.linalg.norm((g - Xtr) / span, axis=1).min()
            d_ch = min([np.linalg.norm((g - c[0]) / span) for c in picks], default=9.0)
            if min(d_tr, d_ch) < min_sep:
                continue
            picks.append((g, float(std[idx])))
            if len(picks) >= n_hotspots:
                break
        return picks

    def _cost(self, fid):
        return self._FIDELITIES[fid]["gpu_h"] + self._C0

    def _plan(self, budget, n_hotspots=None):
        """Triage-and-escalate: twin-triage every hotspot, then escalate the most-uncertain to
        mu=400, then the single most-critical to real-mass gold — all within the GPU-h budget."""
        n_hotspots = n_hotspots or self._HOTSPOTS
        hs = self._hotspots(n_hotspots)              # sorted by std desc
        c_t, c_m, c_g = (self._cost("twin"), self._cost("cgyro_mu400"), self._cost("gold_mu3672"))
        tiers = [None] * len(hs); used = 0.0
        # tier 1 — cheap twin triage on as many hotspots as the budget allows (highest std first)
        for i in range(len(hs)):
            if used + c_t <= budget + 1e-9:
                tiers[i] = "twin"; used += c_t
        # tier 2 — escalate the single most-uncertain to real-mass gold if affordable
        if tiers and tiers[0] == "twin" and used + c_g <= budget + 1e-9:
            tiers[0] = "gold_mu3672"; used += c_g
        # tier 3 — escalate the next most-uncertain to mu=400 while budget remains
        for i in range(len(hs)):
            if tiers[i] == "twin" and used + c_m <= budget + 1e-9:
                tiers[i] = "cgyro_mu400"; used += c_m
        plan = []
        for (g, s), tier in zip(hs, tiers):
            if tier is None:
                continue
            plan.append({"a_LT": round(float(g[0]), 3), "shear": round(float(g[1]), 3),
                         "fidelity": tier, "gpu_h": round(self._cost(tier), 2),
                         "gp_std": round(float(s), 4),
                         "truth_confidence_rho": self._FIDELITIES[tier]["rho"]})
        return plan, round(used, 2), hs

    def _mean_confidence(self, plan, n_hotspots):
        """Mean truth-confidence (rho) achieved across ALL target hotspots (unvisited = 0)."""
        return round(sum(p["truth_confidence_rho"] for p in plan) / n_hotspots, 3)

    def _baseline_all_mu400(self, budget, n_hotspots):
        """Naive baseline: spend the whole budget at mu=400 on the highest-std hotspots."""
        c_m = self._cost("cgyro_mu400")
        n = min(n_hotspots, int((budget + 1e-9) // c_m))
        return {"n_runs": n, "gpu_h_used": round(n * c_m, 2),
                "mean_confidence_rho": round(n * self._FIDELITIES["cgyro_mu400"]["rho"] / n_hotspots, 3),
                "unvisited_hotspots": n_hotspots - n}

    def _predict(self, x):
        """x = GPU-hour budget (float) -> the recommended multi-fidelity campaign plan."""
        budget = float(np.ravel(np.asarray(x, float))[0]) if x is not None else 15.0
        plan, used, _ = self._plan(budget)
        by_fid = {f: sum(1 for p in plan if p["fidelity"] == f) for f in self._FIDELITIES}
        return Prediction(plan, None, True,
                          note=f"multi-fidelity campaign: {len(plan)} runs {by_fid}, {used} GPU-h of {budget}")

    def benchmark(self):
        budget = 15.0                                # tight budget: mu=400 can't cover every hotspot
        n = self._HOTSPOTS
        plan, used, hs = self._plan(budget, n)
        by_fid = {f: sum(1 for p in plan if p["fidelity"] == f) for f in self._FIDELITIES}
        mf_conf = self._mean_confidence(plan, n)
        base = self._baseline_all_mu400(budget, n)
        ratio = round(mf_conf / base["mean_confidence_rho"], 2) if base["mean_confidence_rho"] > 1e-9 else None
        return {"member": self.name,
                "live_mf_campaign": {
                    "source": "KYRO GP over the completed 16-pt CGYRO A1e map",
                    "budget_gpu_h": budget, "target_hotspots": n,
                    "policy": "twin-triage every hotspot, escalate most-uncertain to mu=400, reserve gold",
                    "fidelities": {k: {kk: v[kk] for kk in ("gpu_h", "rho", "desc")}
                                   for k, v in self._FIDELITIES.items()},
                    "multi_fidelity_plan": plan,
                    "mf_runs": len(plan), "mf_gpu_h_used": used, "mf_runs_by_fidelity": by_fid,
                    "mf_mean_confidence_rho": mf_conf,
                    "baseline_all_mu400": base,
                    "mf_confidence_gain_x_vs_all_mu400": ratio,
                    "verdict": (f"Under a tight {budget} GPU-h budget across {n} uncertain operating points, the "
                                f"multi-fidelity policy triages ALL {n} with cheap twin runs and escalates "
                                f"{by_fid['cgyro_mu400']} to mu=400 ({by_fid['gold_mu3672']} to real-mass gold), "
                                f"reaching mean truth-confidence rho {mf_conf} — vs {base['mean_confidence_rho']} "
                                f"for a naive all-mu=400 plan that exhausts the budget after {base['n_runs']} "
                                f"points and leaves {base['unvisited_hotspots']} hotspots unexamined ({ratio}x). "
                                f"A real cost-aware campaign manager: look everywhere cheaply, spend the "
                                f"expensive fidelity where KYRO is least sure.")},
                "caveats": ["the per-fidelity rho (correlation-with-truth) values are ASSUMPTIONS — the "
                            "real-mass gold has not been run — so this is a planning heuristic, not a "
                            "measured speedup; the advantage holds only while budget < hotspots x mu=400 cost",
                            "GP std comes from only 16 points (directional, like KSEEK)",
                            "GPU-hours only; no compute-cost figures (product-econ rule)"]}
