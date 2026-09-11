# The Kronos Family of Codes (KODEX) — technical note

*Methods-paper seed. Draft — not published. 2026-09-09.*

## Abstract

KODEX is a unified suite of machine-learning surrogates for fusion design and
control. Each code replaces an expensive first-principles computation with a fast
approximation that carries **calibrated uncertainty** and an **abstention gate**,
so a consumer always knows both the prediction's error bar and whether the input
is inside the region the surrogate is trusted on. The suite consolidates
previously scattered research prototypes behind one API,
`predict(x) -> (y, uncertainty, in_domain)`, ships as the `kronos-ml` package
(depending on the published `kronos-toolkit` physics engine), and reports honest,
reproducible benchmarks on data already on disk.

## Architecture

A two-layer **spine** wraps every predictive member:

- **KHALO** — the shared uncertainty layer. A Gaussian-process head (Matérn-5/2 ×
  constant + white-noise kernel, optional delta-learning on a cheap baseline) for
  regression, and a soft-vote deep ensemble (HistGB + RandomForest + MLP) for
  classification. Calibration is a first-class output: reliability curve, ECE,
  90% coverage, sharpness, and NLL from one canonical module.
- **KGATE** — the shared trust boundary. A deterministic, model-free projection of
  any command onto a frozen physical envelope (Sabbagh β_N, Greenwald density,
  q95, REBCO strain), fail-closing to a hold on non-finite input. Safety does not
  depend on any model being correct.

Predictive members attach both layers and expose the same contract, so the
toolkit's existing UQ/Study machinery consumes any KODEX surrogate unchanged.

## Data and honest provenance

All training/validation data is on disk; nothing is regenerated on gated
hardware. The flagship transport surrogate **KYRO** is trained and validated on
**real CGYRO nonlinear kinetic-electron turbulence flux** — the **complete 16/16
A1e map (12 turbulent / 4 quiet)** at **representative fidelity, reduced electron
mass μ=400**. The turbulent structure is clean: ITG (ion-dominated) at high
gradient/shear, electron-dominated (TEM-like, Q_i≈0, Q_e carries transport) along
the low-shear line, and a quiet basin at high shear / low gradient — so the
**operating point (a/L_T ≈ 1.3–1.6) sits below the entire map: subcritical**. A
fully-converged real-mass (μ=3672) "gold" validation point is a separate deferred
check. Every surrogate is tagged `[T]`, naming the real code that
retires it; results that stand in for the reduced-order twin or an analytic
closure say so.

## Benchmarks (honest; see BENCHMARKS.md)

- **KGATE** — 0 escapes over 150,000 injected-violation steps (model-free).
- **KYRO** — on the complete 16/16 CGYRO map: **turbulent/quiet classification
  16/16 correct** (leave-one-out), R² = 0.86 on log total heat flux with 88% 90%-interval
  coverage; **~0.26 ms per prediction versus ~2.9 GPU-hours per point** for the CGYRO
  solve (μ=400). At the subcritical operating point it returns `in_domain=False` —
  it abstains rather than extrapolate below the map.
- **KHALO** — ECE 0.033, 90% coverage 0.888 (sourced); live GP calibration on the
  CGYRO points confirms the machinery.
- **KOIL** — magnet-twin strain surrogate rel-L2 ≈ 0.4% live; quench precursor
  AUC 0.9998, 47 ms lead (sourced).
- **KFLOW** — sensor imputation beats the naive baseline ~2×; the trained GNN
  missed the pre-registered AC-18 0.05 bar (0.084) — the miss is kept.
- **KWARD** — twin AUC 0.99 but warning 22.3 ms < the 30 ms actionability bar —
  kept; probability calibration is the KODEX net-new addition.
- **KQUBIT / KQROSS** — no quantum advantage this decade (rigor cards, not speedups).

## Honest limitations

Most members are trained against the reduced-order twin or analytic closures, not
operating-plant data (Kronos has no plant). CGYRO fidelity is representative
(μ=400). Several pre-registered accuracy bars were missed and are reported as
such. The hardware failsafe behind KGATE has an unmeasured trip-time (the largest
residual). KECON is a generic community calculator only — it holds no Kronos
economics.

## Reproducibility

Fixed seed (20260726), CPU-only, no network. `PYTHONPATH=<kronos-ml>:<kronos-toolkit>
python benchmarks/run_all.py` regenerates `BENCHMARKS.md` and `benchmarks.json`;
`pytest tests/` checks the contract, the calibration math, the KGATE fail-closed
behaviour, and the KECON financial firewall. Apache-2.0.
