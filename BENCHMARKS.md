# KODEX — BENCHMARKS

Honest numbers only. Each figure is computed live (fixed seed, CPU, no
network) or **sourced** from a named on-disk file. Pre-registered misses
are kept, not hidden. Nothing here is published until go-live.

| # | Code | Status | Headline benchmark |
|--|--|--|--|
| 01 | **KAIROS** | BUILT | MPC tracking 0.0461 (<5%); Exp-B **0 escapes / 150,000**; 76.8 µs/step |
| 02 | **KFLOW** | BUILT | GNN RMSE@k8 0.0844 vs naive 0.1721 (beats naive; AC-18 0.05 bar **MISSED**, kept) |
| 03 | **KGATE** | BUILT | model-free clamp: **0 escapes / 150,000 steps** |
| 04 | **KHALO** | BUILT | sourced ECE 0.0327 +/- 0.0136 / cov90 0.888 +/- 0.034; live GP ECE 0.0771 |
| 05 | **KMAT** | BUILT | DFT-validated CHGNet screen, 8 candidates |
| 06 | **KOIL** | BUILT | live strain surrogate rel-L2 0.0038; sourced 0.24% @0.15 ms, quench AUC 0.9998 |
| 07 | **KORE** | BUILT | learned NN equilibrium: rel-L2 0.0035 vs analytic, 0.421 ms/field; sourced FNO 2.5% vs 1% bar |
| 08 | **KQUBIT** | BUILT | **statevector 0.0 mHa (PASS); noisy 14x-1305x over chem-acc (sim-only)** — MANDATORY: no quantum advantage this decade |
| 09 | **KWARD** | BUILT | real-device AUC 0.98 (591 MAST shots), ECE 0.0349; **independent-precursor AUC 0.975** (n=1 Mirnov + P_rad ONLY, label-independent); labels heuristic (Ip-quench) |
| 10 | **KYRO** | BUILT | complete 16/16 map (12 turbulent / 4 quiet (16/16 complete)); turbulent/quiet **16/16 correct**, R²=0.858 / cov90=0.875; **0.265 ms vs 2.89 GPU-h/pt** (μ=400) |
| 11 | **KBREED** | BUILT | TBR surrogate R²=1.0 vs neutronics engine; nominal net TBR 0.742; OpenMC = auto fidelity upgrade |
| 12 | **KBURN** | BUILT | dispatch surrogate R²=1.0 (fuel mix → captured power); +22% dynamic |
| 13 | **KEYE** | BUILT | virtual-sensor fault detection 0.767 at FAR 0.0 |
| 14 | **KFLUX** | BUILT | coil-fluence surrogate R²=1.0 vs neutronics engine; OpenMC = auto fidelity upgrade |
| 15 | **KFUSE** | BUILT | multi-fidelity: RMSE 1.481→1.05 (R²=0.787); 3rd fidelity = real-mass gold (deferred) |
| 16 | **KHEAT** | PARTIAL | RF / NBI heating & current-drive actuator-response surrogate |
| 17 | **KISO** | BUILT | servo R²=0.723 + **medical-isotope yield** from real FENDL-3.2 σ(E) (6 isotopes, Mo-99/Tc-99m flagship) |
| 18 | **KPATH** | BUILT | MPC-imitation controller R²=0.879 over 6 axes |
| 19 | **KQROSS** | BUILT | **no crossover this decade; survives optimistic corner** — MANDATORY: no quantum advantage this decade |
| 20 | **KSENSE** | BUILT | dAUC ~ -0.04..+0.003, d-warning ~0 ms vs conventional coil (validated null result) |
| 21 | **KTENSOR** | PARTIAL | tensor-network results on disk; surrogate wrapper is thin |
| 22 | **KDRIVE** | BUILT | RL policy (cross-entropy): tracking **0.0551** vs naive 0.1225, twin-in-the-loop |
| 23 | **KECON** | BUILT | generic LCOE calculator + breakdown — **FINANCIAL FIREWALL (no Kronos numbers)** |
| 24 | **KEDGE** | ROADMAP | divertor / SOL / edge heat-flux transport surrogate |
| 25 | **KFORGE** | BUILT | inverse-design chained KYRO+KORE → **found a quiet operating point** (a/L_T=2.186, shear=1.406, Q_tot≈0.0) |
| 26 | **KFUEL** | BUILT | fuel-cycle balance: self-sufficient=True, doubling 5.17 yr (illustrative TBR) |
| 27 | **KGEN** | BUILT | generative (PCA latent, dim 8); samples plausible equilibrium fields (diffusion = roadmap) |
| 28 | **KLAW** | BUILT | equation discovery: 7 terms, R²=0.903 — **rediscovered the critical-gradient onset** from the CGYRO map |
| 29 | **KPILOT** | BUILT | fleet router: plain-language query → the right code (LLM agent = roadmap upgrade) |
| 30 | **KRAD** | ROADMAP | impurity seeding / radiation / divertor-detachment control |

_Regenerate: `PYTHONPATH=<kronos-ml>:<kronos-toolkit> python benchmarks/run_all.py`._
