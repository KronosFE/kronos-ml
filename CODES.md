# KODEX codes — reference

One section per code: brand, technical function, rollout phase, status, the real code it stands in for, and provenance. Full benchmarks in BENCHMARKS.md.

## Phase 1 — data-backed core (ships first)

### KAIROS — CONTROL  `[BUILT]`
real-time closed-loop MPC control; supplies the model-free L4 clamp that KGATE enforces.
*provenance TWIN · retired_by **validated real-time plant controller (device-in-the-loop)** · gates AC-24, AC-25.*

### KFLOW — STATE  `[BUILT]`
twin state-estimation / dropped-sensor imputation on the 76-node diagnostic sensor graph.
*provenance SIM · retired_by **trained message-passing GNN (gnn_seed0..4.pt)** · gates AC-18, AC-19.*

### KGATE — SAFE  `[BUILT]`
trust-boundary / abstention gate — a model-free envelope clamp that decides in_domain and fail-closes on bad input.
*provenance TWIN · retired_by **formal CBF verification + hardware failsafe**.*

### KHALO — UQ  `[BUILT]`
shared calibrated-uncertainty layer (GP posterior + deep-ensemble) + the canonical reliability report.
*provenance SIM · retired_by **full ensemble UQ / conformal prediction** · gates KX-L1-A14, KX-L1-A5.*

### KMAT — MATERIALS  `[BUILT]`
ML+DFT alloy / HEA screen (CHGNet) for structural & functional materials; screen BUILT, dpa/lifetime roadmap.
*provenance SIM · retired_by **DFT (VASP / Quantum ESPRESSO)** · gates KX-L1-A4, BR-L2-A8.*

### KOIL — MAGNET  `[BUILT]`
magnet-twin stress-field surrogate + quench-precursor early-warning; REBCO per-turn Wilson-form hoop stress.
*provenance SIM · retired_by **full FE electromechanical solve** · gates BR-MT1, BR-MT3.*

### KORE — EQUIL  `[BUILT]`
fast MHD-equilibrium accelerator (Fourier Neural Operator / DeepONet); Grad-Shafranov, not turbulence.
*provenance ANALYTIC · retired_by **classical Grad-Shafranov solver** · gates AC-16, AC-L1.*

### KQUBIT — QML  `[BUILT]`
**REAL VQE** (PennyLane) — variational quantum eigensolver for a molecular (H₂) Hamiltonian; runs on a simulator now and on real quantum hardware (`KODEX_QC_BACKEND=ibm` + token), same code path. Recovers the exact ground state to **0.0002 mHa**; a bundled depolarizing-noise sweep shows p=0.5% → 5.56 mHa (why there's no advantage yet). Honest: no quantum advantage this decade (~8–10 yr out); the tooling is real today.
*provenance SIM · retired_by **fault-tolerant quantum hardware (not available this decade)** · gates AC-44, KX-L3-A4.*

### KWARD — DISRUPT  `[BUILT]`
disruption / stability-boundary early-warning (soft-vote ensemble); advisory, downstream of the KGATE clamp.
*provenance TWIN · retired_by **real-device disruption-labelled training (C-Mod/DIII-D/EAST)** · gates AC-20, AC-21.*

### KYRO — TRANSPORT  `[BUILT]`
CGYRO turbulence-transport surrogate (ion heat flux Q_i vs a/L_T x shear); representative fidelity mu=400, real-mass gold pending.
*provenance CGYRO-urep · retired_by **CGYRO (nonlinear gyrokinetic)** · gates BR-L2-A12, BR-L2-A1e, BR-L2-A1c.*

## Phase 2 — fleet extensions

### KBREED — breeder  `[BUILT]`
TBR / neutronics / isotope-yield surrogate for the Hyperion breeder.
*provenance SIM · retired_by **OpenMC / high-fidelity neutronics** · gates BR-neut.*

### KBURN — burner  `[BUILT]`
net-power balance / dispatch / plant surrogate (Aegis/MetroVolt).
*provenance SIM · retired_by **full plant systems model**.*

### KEYE — DIAG  `[BUILT]`
diagnostics / virtual-sensor / signal-validation observer — builds on KFLOW's graph imputation + quantum-sensing; flags faulty channels.
*provenance SIM · retired_by **physical diagnostic suite + validated sensor fusion** · gates AC-36.*

### KFLUX — neutronics  `[BUILT]`
neutronics surrogate — shielding / activation / flux maps.
*provenance SIM · retired_by **OpenMC**.*

### KFUSE — MULTIFID  `[BUILT]`
multi-fidelity surrogate fusing reduced-twin + mu=400 CGYRO + (pending) real-mass gold, with fidelity-aware uncertainty.
*provenance CGYRO-urep · retired_by **CGYRO real-mass converged (mu=3672)** · gates BR-L2-A1e, BR-L2-A1c-MS.*

### KHEAT — heating&CD  `[BUILT]`
heating & current-drive actuator-response surrogate — RandomForest over a 3888-point current-drive design scan; predicts driven current I_cd from RF/NBI drive parameters + plasma state (**R²=0.949**). Feeds KAIROS heating control. Reduced CD model — RF/NBI ray-tracing is the fidelity upgrade.
*provenance SIM · retired_by **full RF/NBI ray-tracing (GENRAY/TORAY/NUBEAM)** · gates H10.*

### KISO — isotopes  `[BUILT]`
isotope-production surrogate incl. medical isotopes.
*provenance SIM · retired_by **activation transport + separation model**.*

### KPATH — operational  `[BUILT]`
pulse-trajectory optimizer (ramp-up -> flat-top -> ramp-down).
*provenance SIM · retired_by **operational scenario optimizer**.*

### KQROSS — QRE  `[BUILT]`
**REAL FT resource estimator** — computes the classical↔quantum crossover, surface-code physical-qubit overhead, and logical-qubit roadmap year for a fusion electronic-structure kernel (order-of-magnitude, literature-scaled). Crossover ~N=50 → **~6e4 physical qubits**, roadmap ~2032; today's hardware ~1e2–1e3 physical, no logical. Validated negative: no FT advantage this decade.
*provenance SIM · retired_by **fault-tolerant quantum hardware (not available this decade)** · gates AC-43, BR-SX-08.*

### KSENSE — QSENSE  `[BUILT]`
**REAL quantum-metrology** (PennyLane mixed-state GHZ) — quantum Fisher information: ideal GHZ gives Heisenberg scaling (gain ~N), but realistic NV/SQUID/SERF dephasing collapses it to the standard quantum limit (over N=2→8, ideal grows 4× vs dephased 2.05× = SQL rate). Honest null: no quantum-sensing advantage for fusion disruptions this decade.
*provenance SIM · retired_by **deployed physical diagnostic hardware** · gates HX-29.*

### KTENSOR — TN  `[BUILT]`
tensor-network / low-rank (MPS-style SVD) ROM of the real CGYRO A1e flux database. **Honest measured result:** the flux is only *modestly* compressible — effective rank ~2.93 of 4; bond-dim-2 keeps ~91% of the variance but ~30% rel-L2, and the turbulent↔quiet transition is a sharp high-rank feature. A genuine ROM characterization that *corrects* the naive "flux is trivially low-rank" expectation. Classical SVD/MPS — **no quantum-advantage claim**.
*provenance SIM · retired_by **exact many-body / quantum simulation** · gates KX-L3.*

### KQERN — QKERNEL  `[BUILT]`  ✦v0.2
**REAL quantum kernel** (PennyLane) — embeds real MAST disruption features into a quantum feature map and classifies on the fidelity kernel; runs on a simulator now, real QC hardware pluggable. MAST-disruption AUC **0.919** vs classical RBF 0.929 (ties — a validated null); a real, runnable quantum-ML pipeline on real fusion data.
*provenance SIM · retired_by **fault-tolerant quantum hardware (not available this decade)** · gates KX-L3.*

### KMIT — QMITIGATION  `[BUILT]`  ✦v0.3
**REAL quantum error mitigation** — zero-noise extrapolation (ZNE, global unitary folding) + inverse-confusion readout-error mitigation applied to the real KQUBIT H₂ VQE circuit (PennyLane default.mixed; sim now, real hardware pluggable). ZNE removes ~89% of the noise-induced energy error (44.4 → 5.0 mHa); readout mitigation cuts the measurement TV distance to ~0. Mitigation is the genuine NISQ tool TODAY — it recovers signal from noise, not a speedup. Honest: quantum's meaningful role in fusion is post-~2036.
*provenance SIM · retired_by **fault-tolerant quantum hardware (not available this decade)** · gates KX-L3.*

## Phase 3 — cutting-edge + capstones

### KDRIVE — RL-control  `[BUILT]`
reinforcement-learning plasma controller (learned policy).
*provenance n/a · retired_by **validated RL controller**.*

### KECON — techno-economics  `[BUILT]`
GENERIC open techno-economics (LCOE) on the USER's inputs — FINANCIAL FIREWALL: no Kronos numbers, ever.
*provenance n/a · retired_by **detailed engineering cost model (never public)**.*

### KEDGE — edge-transport  `[BUILT]`  ✦v0.2
divertor / edge heat-flux → target-thermal surrogate — maps divertor heat flux to target surface temperature + material limits (W / CuCrZr) from the H9 exhaust scan. q→T_surf fit **R²=1.0** (77 pts); CuCrZr material limit **q~13.5 MW/m²**. Reduced 0-D thermal model — SOLPS-ITER/EIRENE edge campaign is the fidelity upgrade.
*provenance SIM · retired_by **SOLPS-ITER / EIRENE edge campaign** · gates H9.*

### KFORGE — inverse-design  `[BUILT]`
inverse-design optimizer that chains the whole fleet to search designs.
*provenance n/a · retired_by **the fleet itself (capstone)**.*

### KFUEL — fuel-cycle  `[BUILT]`
tritium fuel-cycle & inventory surrogate.
*provenance n/a · retired_by **fuel-cycle systems code**.*

### KGEN — generative  `[BUILT]`
generative / diffusion surrogate for states, equilibria & fields.
*provenance n/a · retired_by **validated generative model**.*

### KLAW — eqn-discovery  `[BUILT]`
equation discovery (sparse / symbolic regression, SINDy-class).
*provenance n/a · retired_by **verified reduced closures**.*

### KPILOT — agentic-AI  `[BUILT]`
agentic AI copilot orchestrating the fleet in natural language.
*provenance n/a · retired_by **the fleet + an agent runtime**.*

### KRAD — radiation-control  `[BUILT]`  ✦v0.2
impurity-seeding radiation evaluator — maps a seeded impurity + radiated-power fraction to core Zeff penalty and P_rad from the H9 seeding scan (N / Ne / Ar; Ar gives the least core Zeff dilution at f_rad 0.58). Small scan; full impurity-transport (SOLPS + impurity) is the fidelity upgrade.
*provenance SIM · retired_by **impurity transport (SOLPS + impurity) + radiation control** · gates H9.*

### KQOPT — QOPT  `[BUILT]`  ✦v0.2
**REAL QAOA optimizer** (PennyLane) — solves a combinatorial design QUBO on a real quantum circuit; plug in your own QUBO. Runs on a simulator now, real QC hardware pluggable. Reaches 100% of the brute-force optimum on a 6-qubit design QUBO (p=2). No speedup vs classical at this size; advantage ~8–10 yr out.
*provenance SIM · retired_by **fault-tolerant quantum hardware (not available this decade)** · gates KX-L3.*

### KDYN — QDYN  `[BUILT]`  ✦v0.2
**REAL quantum dynamics** (PennyLane) — Trotterized time-evolution of a transverse-field Ising "kinetic" Hamiltonian; runs on a simulator now, real QC hardware pluggable. 1st-order Trotter error falls 1.35→0.034 over 1→32 steps (~1/n) — the honest cost curve for simulating plasma-like dynamics on a quantum computer.
*provenance SIM · retired_by **fault-tolerant quantum hardware (not available this decade)** · gates KX-L3.*

### KSEEK — ACTIVE  `[BUILT]`  ✦v0.2
active-learning acquisition — proposes the next most-informative CGYRO run from KYRO's GP posterior (max-variance / uncertainty sampling), so expensive GPU-hours go where the surrogate is least sure. Honest: leave-one-out std↔error correlation is weak (−0.28) on the 16-point map — directional guidance that strengthens with data.
*provenance SIM · retired_by **the full CGYRO parameter scan** · gates BR-L2-A1e.*

### KBENCH — BENCHMARK  `[BUILT]`  ✦v0.2
open, citable fusion-ML benchmark suite — 3 tasks (CGYRO turbulence, MAST disruption, flux ROM) with real on-disk data, fixed reproducible splits, metrics, and KODEX baselines to beat. A community leaderboard starting point, not a private result; honest that it's small by mainstream-ML standards (pilot fusion-ML benchmarks).
*provenance n/a · retired_by **community-standard fusion-ML benchmarks**.*

### KMIND — FOUNDATION  `[BUILT]`  ✦v0.3
**whole-device foundation surrogate** — ONE shared-trunk multi-task neural net (per-domain input adapters → a shared plasma-state trunk → per-domain heads, ~67k params) trained JOINTLY across transport (KYRO), equilibrium (KORE), state (KFLOW) and disruption (KWARD) on the data those specialists use. Benchmarked head-by-head against a MATCHED single-task baseline on the same split: on the well-populated domains the shared representation shows **no negative transfer** (equilibrium slight positive, state/disruption neutral); transport is **data-starved (16 pts) and reported, not hidden**. MC-dropout uncertainty; per-domain trust gate (physical box for transport/equil, Mahalanobis for state/disrupt). The honest question a foundation model must answer — does one shared representation help or hurt each task? — answered with matched baselines.
*provenance SIM · retired_by **a device-scale multi-task foundation model on real multi-diagnostic data** · gates BR-L2-A1e, AC-16, AC-18, AC-20.*

### KLINQ — QLINSOLVE  `[BUILT]`  ✦v0.3
**REAL quantum linear solver** — a variational quantum linear solver (VQLS) prepares the solution of a 4×4 reduced-MHD (discrete-Laplacian, SPD) linear system with a real quantum circuit (PennyLane; sim now, hardware pluggable), recovering the classical solution at fidelity ~1.0. Bundled with an honest fault-tolerant HHL resource estimate: a fusion-scale (N~1e6) sparse solve needs ~4e4 physical qubits — no advantage at real scale. Honest: quantum's meaningful role in fusion is post-~2036.
*provenance SIM · retired_by **fault-tolerant quantum hardware (not available this decade)** · gates KX-L3.*

### KSCOUT — MF-CAMPAIGN  `[BUILT]`  ✦v0.3
**multi-fidelity Bayesian-optimization campaign manager** — extends KSEEK from a single next-point to a budget-constrained campaign that chooses which operating point AND which fidelity (reduced twin / μ=400 CGYRO / real-mass gold) to run next. Triage-then-escalate: cheap twin on every uncertain hotspot, escalate the most-uncertain to μ=400, reserve real-mass gold. Under a tight GPU-hour budget it reaches ~2.3× the mean truth-confidence of a naive all-μ=400 plan (which exhausts the budget and leaves most hotspots unexamined). Honest: fidelity correlations are assumptions (gold not yet run); GPU-hours only, no cost figures.
*provenance SIM · retired_by **the full multi-fidelity CGYRO campaign (twin + μ=400 + real-mass gold)** · gates BR-L2-A1e.*

