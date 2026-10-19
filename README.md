# KODEX — the Kronos Family of Codes

[![PyPI](https://img.shields.io/pypi/v/kronos-fusion-ml?color=blue)](https://pypi.org/project/kronos-fusion-ml/)
[![Downloads](https://img.shields.io/pypi/dm/kronos-fusion-ml?color=blue)](https://pypi.org/project/kronos-fusion-ml/)
[![Python](https://img.shields.io/pypi/pyversions/kronos-fusion-ml)](https://pypi.org/project/kronos-fusion-ml/)
[![License](https://img.shields.io/github/license/KronosFE/kronos-ml?color=blue)](LICENSE)

A benchmarked AI/ML surrogate suite for fusion. Fast, calibrated stand-ins for
expensive physics — **every code knows when it is out of its depth.**

- **One contract:** `predict(x) -> Prediction(y, uncertainty, in_domain)`
- **Calibrated uncertainty** on every prediction (the **KHALO** layer)
- **An abstention gate** on every prediction (the **KGATE** trust boundary)
- **Honest provenance** in the type: every surrogate is `[T]` tagged, naming the
  real code it stands in for (`retired_by`) — nothing is ever mistaken for
  high-fidelity, and nothing is called "live."

30 codes across three rollout **phases** (Phase 1 = data-backed, ships first;
Phase 2 = partial data; Phase 3 = roadmap). See [`BENCHMARKS.md`](BENCHMARKS.md)
for the honest per-code numbers and [`FAMILY.md`](FAMILY.md) for the technical note.

## Install

```bash
pip install -e ".[ml,data,nn]"     # kronos-toolkit + numpy/scipy are the core
```

The core imports with **no torch and no sklearn**; heavy dependencies load only
when a surrogate actually runs.

## Quickstart

```python
import kronos_ml as K

K.list_surrogates()                 # the 30 K-codes
K.fleet(phase=1)                    # Phase-1 cards

# a real CGYRO transport prediction (representative fidelity, mu=400)
p = K.run("KYRO", [3.5, 1.6])       # a/L_T, magnetic shear
p.y, p.uncertainty, p.in_domain     # ion heat flux Q_i, 1-sigma, trust gate

# the model-free safety gate
K.get("KGATE").clamp([2.0, 0.3, 9.0, 5.0, 0.9, 60.0])   # -> projected onto the envelope

# provenance travels with the value
p.as_tagged("CGYRO (nonlinear gyrokinetic)")            # [T] retired_by CGYRO
```

## The fleet (30 codes)

**Spine:** `KHALO` (uncertainty) · `KGATE` (abstention gate).
**Phase-1 members:** `KOIL` (magnet twin + quench) · `KFLOW` (state estimation) ·
`KMAT` (materials screen) · `KYRO` (CGYRO transport) · `KWARD` (disruption) ·
`KORE` (equilibrium) · `KAIROS` (control) · `KQUBIT` (quantum-ML, no-advantage).
**Phase-2/3:** breeder/burner/isotope/neutronics/operational codes, more quantum
(`KQROSS`, `KSENSE`, `KTENSOR`), multi-fidelity (`KFUSE`), and roadmap codes
(`KFORGE`, `KDRIVE`, `KLAW`, `KGEN`, `KPILOT`, …). `KECON` is a **generic** open
techno-economics calculator with a **financial firewall** — it carries no Kronos
cost/price/valuation data, ever.

## Honesty

Every pre-registered miss is kept (KFLOW AC-18, KWARD <30 ms warning, KORE 1%
bar). CGYRO-backed results are at **representative fidelity (reduced electron
mass μ=400)**; the real-mass "gold" validation point is the single open item.
No economics or Kronos financials in any physics code. Apache-2.0.
