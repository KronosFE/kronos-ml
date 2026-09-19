# KODEX — BENCHMARKS

Honest numbers only. Each figure is computed live (fixed seed, CPU, no
network) or **sourced** from a named on-disk file. Pre-registered misses
are kept, not hidden. Nothing here is published until go-live.

| # | Code | Status | Headline benchmark |
|--|--|--|--|
| 01 | **KAIROS** | BUILT | MPC tracking 0.0461 (<5%); Exp-B **0 escapes / 150,000**; 77.8 µs/step |
| 02 | **KFLOW** | BUILT | GNN RMSE@k8 0.0844 vs naive 0.1721 (beats naive; AC-18 0.05 bar **MISSED**, kept) |
| 03 | **KGATE** | BUILT | model-free clamp: **0 escapes / 150,000 steps** |
| 04 | **KHALO** | BUILT | sourced ECE 0.0327 +/- 0.0136 / cov90 0.888 +/- 0.034; live GP ECE 0.0771 |
| 05 | **KMAT** | BUILT | DFT-validated CHGNet screen, 8 candidates |
| 06 | **KOIL** | BUILT | live strain surrogate rel-L2 0.0038; sourced 0.24% @0.15 ms, quench AUC 0.9998 |
| 07 | **KORE** | BUILT | learned NN equilibrium: rel-L2 0.0035 vs analytic, 0.318 ms/field; sourced FNO 2.5% vs 1% bar |
| 08 | **KQUBIT** | BUILT | **REAL VQE** (PennyLane): recovers H2 ground state to 0.0002 mHa on default; runs on real QC hardware — no advantage yet |
| 09 | **KWARD** | BUILT | real-device AUC 0.98 (591 MAST shots), ECE 0.0349; **independent-precursor AUC 0.975** (n=1 Mirnov + P_rad ONLY, label-independent); labels heuristic (Ip-quench) |
| 10 | **KYRO** | BUILT | complete 16/16 map (12 turbulent / 4 quiet (16/16 complete)); turbulent/quiet **16/16 correct**, R²=0.858 / cov90=0.875; **0.257 ms vs 2.89 GPU-h/pt** (μ=400) |
| 11 | **KBREED** | BUILT | TBR surrogate R²=1.0 vs neutronics engine; nominal net TBR 0.742; OpenMC = auto fidelity upgrade |
| 12 | **KBURN** | BUILT | dispatch surrogate R²=1.0 (fuel mix → captured power); +22% dynamic |
| 13 | **KEYE** | BUILT | virtual-sensor fault detection 0.767 at FAR 0.0 |
| 14 | **KFLUX** | BUILT | coil-fluence surrogate R²=1.0 vs neutronics engine; OpenMC = auto fidelity upgrade |
| 15 | **KFUSE** | BUILT | multi-fidelity: RMSE 1.481→1.05 (R²=0.787); 3rd fidelity = real-mass gold (deferred) |
| 16 | **KHEAT** | BUILT | heating/current-drive surrogate: driven-current I_cd **R²=0.949** over 3888 configs (reduced CD; RF/NBI ray-tracing = upgrade) |
| 17 | **KISO** | BUILT | servo R²=0.723 + **medical-isotope yield** from real FENDL-3.2 σ(E) (6 isotopes, Mo-99/Tc-99m flagship) |
| 18 | **KMIT** | BUILT | **REAL error mitigation**: ZNE cuts H2 energy error 44.38→5.029 mHa (0.887 removed); readout TV 0.1472→0.0 — real NISQ tool today, role post-~2036 |
| 19 | **KPATH** | BUILT | MPC-imitation controller R²=0.879 over 6 axes |
| 20 | **KQERN** | BUILT | **REAL quantum kernel**: MAST-disruption AUC 0.919 vs classical 0.929 (ties — honest null); runnable on real QC hardware |
| 21 | **KQROSS** | BUILT | **REAL FT resource estimator**: classical↔quantum crossover ~N=50 needs ~6e+04 physical qubits, roadmap ~2032 — no FT advantage this decade |
| 22 | **KSENSE** | BUILT | **REAL quantum metrology** (GHZ, PennyLane): ideal Heisenberg 4.0× vs dephased 2.05× ≈ SQL 2.01× — dephasing erases the advantage (honest null) |
| 23 | **KTENSOR** | BUILT | SVD/MPS ROM of the real CGYRO flux DB: **effective rank ~2.93 of 4 — modestly compressible, NOT strongly low-rank** (honest characterization; classical, no quantum advantage) |
| 24 | **KBENCH** | BUILT | open fusion-ML benchmark suite: **3 citable tasks** (CGYRO turbulence, MAST disruption, flux ROM) with real data + KODEX baselines |
| 25 | **KDRIVE** | BUILT | RL policy (cross-entropy): tracking **0.0551** vs naive 0.1225, twin-in-the-loop |
| 26 | **KDYN** | BUILT | **REAL Trotter quantum dynamics**: 1st-order error 1.34637→0.03381 over steps (~1/n); runs on real QC hardware — honest cost curve, no advantage yet |
| 27 | **KECON** | BUILT | generic LCOE calculator + breakdown — **FINANCIAL FIREWALL (no Kronos numbers)** |
| 28 | **KEDGE** | BUILT | divertor edge surrogate: heat-flux→target-temp **R²=1.0**, CuCrZr limit q~13.5 MW/m² (reduced 0-D; SOLPS/EIRENE = upgrade) |
| 29 | **KFORGE** | BUILT | inverse-design chained KYRO+KORE → **found a quiet operating point** (a/L_T=2.228, shear=1.511, Q_tot≈0.0) |
| 30 | **KFUEL** | BUILT | fuel-cycle balance: self-sufficient=True, doubling 5.17 yr (illustrative TBR) |
| 31 | **KGEN** | BUILT | generative (PCA latent, dim 8); samples plausible equilibrium fields (diffusion = roadmap) |
| 32 | **KLAW** | BUILT | equation discovery: 7 terms, R²=0.903 — **rediscovered the critical-gradient onset** from the CGYRO map |
| 33 | **KLINQ** | BUILT | **REAL quantum linear solver** (VQLS): 4×4 reduced-MHD solve fidelity 1.0 vs classical (rel-resid 0.0); FT-HHL ~4e+04 phys qubits — no advantage, role post-~2036 |
| 34 | **KMIND** | BUILT | **foundation model**: one shared trunk across 4 domains (66958 params); transfer vs matched single-task — pos ['equil'], neg none, neutral ['state', 'disrupt'] (transport data-starved) — honest multi-task result |
| 35 | **KPILOT** | BUILT | fleet router: plain-language query → the right code (LLM agent = roadmap upgrade) |
| 36 | **KQOPT** | BUILT | **REAL QAOA optimizer**: 1.0 of optimum on a design QUBO (6 qubits); plug in your QUBO, run on real QC hardware — no speedup yet |
| 37 | **KRAD** | BUILT | impurity-seeding radiation evaluator: 3 impurities (Ar, N, Ne); least core dilution = Ar |
| 38 | **KSCOUT** | BUILT | **multi-fidelity campaign**: tight 15.0 GPU-h over 16 hotspots → mean confidence ρ 0.638 vs 0.281 all-μ=400 (2.27×); triage-then-escalate |
| 39 | **KSEEK** | BUILT | active-learning: proposes next CGYRO run (a/L_T=2.225, shear=0.58); LOO std-vs-error corr -0.283 (honest: weak on 16 pts) |

_Regenerate: `PYTHONPATH=<kronos-ml>:<kronos-toolkit> python benchmarks/run_all.py`._
