"""Read-only loaders for the on-disk research data KODEX consolidates.

The original track folders are the source of truth and are NEVER written here.
Paths resolve from $KODEX_DATA_ROOT (the AI_QUANTUM tree) and
$KODEX_PHASE3_ROOT (the HPC campaign results), falling back to the known
absolute locations on this machine.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_AIQ = ("/Users/pford/Desktop/Kronos Fusion Energy/01 - RESEARCH & DATA/"
                "01 - Research, Data & Codes/2026 Post Publication Research/AI_QUANTUM")
_DEFAULT_PHASE3 = ("/Users/pford/Desktop/Kronos Fusion Energy/01 - RESEARCH & DATA/"
                   "01 - Research, Data & Codes/PHASE 3 - Detailed Design Papers/"
                   "_HPC_CAMPAIGN_RESULTS_2026-08-25")

AIQ_ROOT = Path(os.environ.get("KODEX_DATA_ROOT", _DEFAULT_AIQ))
PHASE3_ROOT = Path(os.environ.get("KODEX_PHASE3_ROOT", _DEFAULT_PHASE3))


def track(name: str) -> Path:
    return AIQ_ROOT / name


def exists(p: Path) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False


# ---- specific loaders (added as members need them) -----------------------

def gnn_snapshots():
    """KFLOW/STATE training data: 5000 x 76 analytic-twin sensor snapshots (SIM)."""
    import numpy as np
    p = track("track3_gnn") / "data" / "twin_snapshots.npz"
    return np.load(p, allow_pickle=True)


def cgyro_flux_map():
    """KYRO/TRANSPORT target: the CGYRO A1e 2-D flux map (16 rows). μ=400 rep."""
    import pandas as pd
    p = PHASE3_ROOT / "cgyro_kinetic_flux_map_A1e" / "kin_flux_map" / "flux_db.csv"
    return pd.read_csv(p)


# the 6 near-threshold points re-run at fine resolution (N_RADIAL=128), 2026-09-10
_FINISH = PHASE3_ROOT / "_CGYRO_FINISH_2026-09-09"
_RERUN = {(2.5, 0.4), (2.5, 0.8), (2.5, 1.2), (2.5, 1.6), (3.0, 0.8), (3.5, 0.8)}
# 3.5/0.8 hi-res harvest has no result.json yet -> md-consolidated value (flagged)
_MD_ONLY = {(3.5, 0.8): {"Q_i": 0.0, "Q_e": 100.0}}


def cgyro_flux_map_final():
    """The COMPLETE 16/16 A1e map (2026-09-10): 10 original saturated points +
    6 re-run at N_RADIAL=128.  Adds Q_tot and the turbulent/quiet verdict.
    Reads the re-run values from `final5_harvest/<pt>/result.json`; boxes are all
    terminated (read-only)."""
    import json
    import numpy as np
    import pandas as pd
    base = cgyro_flux_map()
    rows = []
    for _, r in base.iterrows():
        key = (round(float(r.a_LT), 3), round(float(r.shear), 3))
        qi, qe, src = float(r.Q_i), float(r.Q_e), "flux_db.csv (original)"
        if key in _RERUN:
            j = _FINISH / "final5_harvest" / f"{float(r.a_LT):.1f}_{float(r.shear):.1f}" / "result.json"
            if j.exists():
                d = json.load(open(j)); qi, qe = float(d["Q_i"]), float(d["Q_e"])
                src = "final5_harvest N_RADIAL=128"
            elif key in _MD_ONLY:
                qi, qe = _MD_ONLY[key]["Q_i"], _MD_ONLY[key]["Q_e"]
                src = "CGYRO_16MAP_FINAL md (hi-res harvest pending json)"
        qtot = max(qi, 0.0) + max(qe, 0.0)
        rows.append({"a_LT": float(r.a_LT), "shear": float(r.shear),
                     "Q_i": qi, "Q_e": qe, "Q_tot": qtot,
                     "verdict": "turbulent" if qtot > 0.1 else "quiet",
                     "source": src})
    return pd.DataFrame(rows)


def read_csv(track_name: str, filename: str):
    import pandas as pd
    return pd.read_csv(track(track_name) / filename)


# KISO medical-isotope cross-sections (route-B data, FENDL-3.2), {reaction, energy_MeV, sigma_barns}
_DEFAULT_KISO = ("/Users/pford/Desktop/Kronos Fusion Energy/01 - RESEARCH & DATA/"
                 "01 - Research, Data & Codes/2026 Post Publication Research/AI_QUANTUM/"
                 "KRONOS_FAMILY_OF_CODES/_INCOMING_DATA/kiso")
KISO_XS_DIR = Path(os.environ.get("KODEX_KISO_XS", _DEFAULT_KISO))


_DEFAULT_KWARD = ("/Users/pford/Desktop/Kronos Fusion Energy/01 - RESEARCH & DATA/"
                  "01 - Research, Data & Codes/2026 Post Publication Research/AI_QUANTUM/"
                  "KRONOS_FAMILY_OF_CODES/_INCOMING_DATA/kward")
KWARD_DIR = Path(os.environ.get("KODEX_KWARD", _DEFAULT_KWARD))


def kward_real():
    """Real MAST disruption shots (route-B, FAIR-MAST) -> (shots_df, labels_df).
    Labels DERIVED from Ip current-quench (heuristic, not physicist-verified)."""
    import pandas as pd
    sh = pd.read_parquet(KWARD_DIR / "kward_shots.parquet")
    lab = pd.read_csv(KWARD_DIR / "labels.csv")
    return sh, lab


def kward_available():
    return (KWARD_DIR / "kward_shots.parquet").exists()


def medical_xs():
    """Load the medical-isotope production cross-sections -> {product: (E_MeV, sigma_b, reaction)}."""
    import pandas as pd
    out = {}
    for f in sorted(KISO_XS_DIR.glob("*.csv")):
        df = pd.read_csv(f).sort_values("energy_MeV")
        rxn = str(df.reaction.iloc[0])
        product = rxn.split(")")[-1] or f.stem       # e.g. "Mo-99"
        out[product] = (df.energy_MeV.to_numpy(float), df.sigma_barns.to_numpy(float), rxn)
    return out


def read_npz(track_name: str, filename: str):
    import numpy as np
    return np.load(track(track_name) / filename, allow_pickle=True)
