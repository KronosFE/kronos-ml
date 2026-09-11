"""KODEX KTENSOR — source excerpt (class). Full runnable package: https://github.com/KronosFE/kronos-ml"""

@register
class KTENSOR(Surrogate):
    name = "KTENSOR"; function = "TN"; phase = 2; status = "BUILT"
    provenance = "SIM"
    retired_by = "exact many-body / quantum simulation"
    real_codes = ("tensor-network / low-rank SVD (MPS-style) ROM",)
    gates = ("KX-L3",)
    note = ("tensor-network / low-rank ROM compression of the real CGYRO A1e flux "
            "database (classical MPS-style SVD; a many-body method and bridge to "
            "quantum kernels — NO quantum-advantage claim)")

    _COLS = ("a_LT", "shear", "Q_i", "Q_e")   # operating grid + saturated fluxes

    def _load(self, turbulent_only=False):
        """Real CGYRO A1e flux database (the COMPLETE 16-pt map; NaN-free) as a
        (points x cols) matrix. The regen 9-pt DB is mostly NaN (a sparse linear scan),
        so we use the completed map, which is the honest, physically-complete dataset."""
        import numpy as np
        df = data.cgyro_flux_map_final()
        if turbulent_only:
            df = df[df["verdict"] == "turbulent"]
        A = df[list(self._COLS)].to_numpy(float)
        A = A[np.isfinite(A).all(1)]                       # drop any non-finite row (safety)
        src = "cgyro_flux_map_final" + (" (turbulent rows)" if turbulent_only else " (16 pts)")
        return A, list(self._COLS), src

    def _matrix(self, turbulent_only=False):
        """Per-column z-scored flux matrix so mixed-scale columns are comparable."""
        import numpy as np
        A, _, _ = self._load(turbulent_only)
        mu, sd = A.mean(0), A.std(0)
        sd = np.where(sd == 0, 1.0, sd)
        return (A - mu) / sd

    @staticmethod
    def _eff_rank(A):
        """Participation ratio of the singular-value spectrum (effective rank)."""
        import numpy as np
        s = np.linalg.svd(A, compute_uv=False)
        return float((s.sum() ** 2) / ((s ** 2).sum() + 1e-30))

    def _svd(self):
        import numpy as np
        A = self._matrix()
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        return A, U, s, Vt

    def _rel_error(self, rank):
        """rel-L2 (Frobenius) reconstruction error keeping `rank` singular values
        (= MPS bond dimension)."""
        import numpy as np
        A, U, s, Vt = self._svd()
        r = int(max(1, min(int(rank), len(s))))
        Ar = (U[:, :r] * s[:r]) @ Vt[:r]
        return float(np.linalg.norm(A - Ar) / (np.linalg.norm(A) + 1e-12))

    def _predict(self, x):
        """x = target bond dimension (retained rank) -> achievable reconstruction rel-error."""
        import numpy as np
        rank = int(np.ravel(np.asarray(x, dtype=float))[0])
        rmax = len(self._svd()[2])
        return Prediction(self._rel_error(rank), None, 1 <= rank <= rmax,
                          note=f"rel-L2 reconstruction of the CGYRO flux DB at bond-dim "
                               f"{rank} (of {rmax})")

    def benchmark(self):
        import numpy as np

        def curve_for(turb):
            A = self._matrix(turb)
            U, s, Vt = np.linalg.svd(A, full_matrices=False)

            def rel(r):
                r = int(max(1, min(r, len(s))))
                return float(np.linalg.norm(A - (U[:, :r] * s[:r]) @ Vt[:r]) / np.linalg.norm(A))
            var = (s ** 2) / (s ** 2).sum()
            return {"shape": list(A.shape),
                    "rel_error_vs_bond_dim": {r: round(rel(r), 4) for r in range(1, len(s) + 1)},
                    "cumulative_variance": {r: round(float(var[:r].sum()), 4)
                                            for r in range(1, len(s) + 1)},
                    "effective_rank": round(self._eff_rank(A), 2)}

        full = curve_for(False)
        turb = curve_for(True)
        v2 = full["cumulative_variance"][2]
        e2, e3 = full["rel_error_vs_bond_dim"][2], full["rel_error_vs_bond_dim"][3]
        return {"member": self.name,
                "live_low_rank_rom": {
                    "source": "real CGYRO A1e flux database (data.cgyro_flux_map_final)",
                    "columns": list(self._COLS),
                    "full_16pt_map": full, "turbulent_branch": turb,
                    "verdict": (f"MEASURED ROM characterization: the CGYRO flux database is only MODESTLY "
                                f"compressible — effective rank ~{full['effective_rank']} of 4; a bond-dim-2 "
                                f"MPS keeps {v2:.0%} of the variance but {e2:.0%} rel-L2 error (bond-dim 3 -> "
                                f"{e3:.0%}). The turbulent flux spans 3+ orders of magnitude with a sharp "
                                f"turbulent<->quiet transition, so it RESISTS dramatic low-rank compression. "
                                f"An honest counter to the naive 'flux is trivially low-rank' expectation. "
                                f"Classical SVD/MPS; NO quantum-advantage claim."),
                    "caveats": ["NOT a dramatic low-rank collapse: bond-dim-2 keeps ~91% variance but "
                                "~30% rel-L2 error",
                                "classical SVD/MPS ROM of real CGYRO data — a many-body / quantum-bridge "
                                "method, NOT a quantum-advantage claim",
                                "the 9-pt regen flux DB is mostly NaN (a sparse linear scan); the completed "
                                "16-pt map is the honest dataset used here"]},
                "method": "SVD low-rank / MPS bond-dimension compression (deterministic, CPU)",
                "sourced": {"file": "TRACK1_QUANTUM_RESULTS/track1_results.json (A5_MPS_flux)",
                            "note": "the Track-1 'rank-2 ~1%' figure was on the sparse 9x4 regen table; on "
                                    "the complete map the flux is not strongly low-rank (this card)"}}
