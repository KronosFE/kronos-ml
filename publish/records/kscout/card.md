# KODEX KSCOUT — card

- **name:** KSCOUT
- **function:** MF-CAMPAIGN
- **status:** BUILT
- **phase:** 3
- **provenance:** SIM
- **retired_by:** the full multi-fidelity CGYRO campaign (twin + mu=400 + real-mass gold)
- **gates:** ['BR-L2-A1e']
- **note:** multi-fidelity Bayesian-optimization campaign manager — extends KSEEK: chooses which operating point AND which fidelity (reduced twin / mu=400 CGYRO / real-mass gold) to run next, greedily maximizing truth-weighted information per GPU-hour under a budget
- **available:** True

Benchmark headline: **multi-fidelity campaign**: tight 15.0 GPU-h over 16 hotspots → mean confidence ρ 0.638 vs 0.281 all-μ=400 (2.27×); triage-then-escalate
