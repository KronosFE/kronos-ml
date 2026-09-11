"""Contract + honesty tests for the KODEX fleet.  Fast (no heavy training)."""
import warnings
import numpy as np
import pytest

warnings.filterwarnings("ignore")
import kronos_ml as K
from kronos_ml.base import Prediction


def test_registry():
    # roster grows as new codes are built (v0.1.0 was 30); guard consistency, not an exact count
    assert len(K.SURROGATES) >= 30
    phases = {1: 0, 2: 0, 3: 0}
    for c in K.fleet():
        phases[c["phase"]] += 1
    assert sum(phases.values()) == len(K.SURROGATES)   # every code lands in a phase
    assert all(v > 0 for v in phases.values())          # all three phases populated


def test_every_code_has_provenance_and_status():
    for name in K.list_surrogates():
        c = K.get(name).card()
        assert c["status"] in ("BUILT", "PARTIAL", "ROADMAP")
        # every BUILT/PARTIAL surrogate names the real code it stands in for
        if c["status"] in ("BUILT", "PARTIAL"):
            assert c["retired_by"], f"{name} missing retired_by"


def test_contract_returns_prediction():
    for name, x in [("KYRO", [3.0, 1.0]), ("KGATE", [2, 0.3, 3, 1, 0.2, 30]),
                    ("KOIL", [0.1, 26.49, 330.0])]:
        p = K.run(name, x)
        assert isinstance(p, Prediction)
        y, unc, indom = p.as_tuple()
        assert isinstance(indom, bool)


def test_kgate_clamps_and_fails_closed():
    g = K.get("KGATE")
    # over-limit command is projected inside the envelope
    safe, ok = g.clamp([2, 0.3, 9.0, 5.0, 0.9, 60.0])
    assert ok and safe[2] <= 4.2 + 1e-9 and safe[4] <= 0.4 + 1e-9
    # non-finite input fails closed to a HOLD (not in domain)
    safe, ok = g.clamp([np.nan, 0, 0, 0, 0, 0])
    assert not ok
    assert not g.in_domain([np.nan, 0, 0, 0, 0, 0])


def test_kyro_reproduces_training_point():
    # the GP should sit near the on-disk Q_i at a strongly-turbulent point
    p = K.run("KYRO", [3.5, 1.6])
    assert p.in_domain and p.y > 500  # Q_i ~ 1111 on disk
    assert p.uncertainty is not None and p.uncertainty > 0


def test_kecon_financial_firewall():
    # refuses to run without the caller's own inputs (no Kronos numbers baked in)
    with pytest.raises(ValueError):
        K.run("KECON", {})
    lcoe = K.run("KECON", {"capex": 5e9, "opex_per_yr": 1e8,
                           "annual_generation_MWh": 3e6,
                           "discount_rate": 0.07, "lifetime_yr": 30})
    assert lcoe.y["lcoe_per_MWh"] > 0
    assert K.get("KECON").KRONOS_NUMBERS is None  # firewall: never any Kronos data


def test_roadmap_codes_raise():
    # any ROADMAP-status code must raise cleanly on run(); as of the v0.2.0 build every code
    # is BUILT (0 roadmap), so this guards future placeholders without failing today.
    for name in K.list_surrogates():
        if K.get(name).status == "ROADMAP":
            with pytest.raises(NotImplementedError):
                K.run(name, None)


def test_tagged_provenance_requires_retired_by():
    t = K.get("KYRO").predict([3.0, 1.0]).as_tagged("CGYRO (nonlinear gyrokinetic)")
    assert t.tag == "T" and t.retired_by


def test_metrics_calibration():
    from kronos_ml import metrics
    rng = np.random.default_rng(0)
    y = rng.normal(size=2000); mean = np.zeros(2000); std = np.ones(2000)
    # well-calibrated unit Gaussian -> ~90% coverage, small ECE
    assert abs(metrics.coverage_at_level(y, mean, std, 0.90) - 0.90) < 0.03
    assert metrics.ece(y, mean, std) < 0.03
