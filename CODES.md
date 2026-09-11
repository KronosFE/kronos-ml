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
quantum-ML surrogate (VQE) — honest no-advantage rigor card; statevector exact, noisy never reaches chemical accuracy.
*provenance SIM · retired_by **fault-tolerant quantum hardware (not available this decade)** · gates AC-44, KX-L3-A4.*

### KWARD — DISRUPT  `[BUILT]`
disruption / stability-boundary early-warning (soft-vote ensemble); advisory, downstream of the KGATE clamp.
*provenance TWIN · retired_by **real-device disruption-labelled training (C-Mod/DIII-D/EAST)** · gates AC-20, AC-21.*

### KYRO — TRANSPORT  `[BUILT]`
CGYRO turbulence-transport surrogate (ion heat flux Q_i vs a/L_T x shear); representative fidelity mu=400, real-mass gold pending.
*provenance CGYRO-urep · retired_by **CGYRO (nonlinear gyrokinetic)** · gates BR-L2-A12, BR-L2-A1e, BR-L2-A1c.*

## Phase 2 — fleet extensions (partial data)

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

### KHEAT — heating&CD  `[PARTIAL]`
RF / NBI heating & current-drive actuator-response surrogate.
*provenance SIM · retired_by **RF/NBI physics codes**.*

### KISO — isotopes  `[BUILT]`
isotope-production surrogate incl. medical isotopes.
*provenance SIM · retired_by **activation transport + separation model**.*

### KPATH — operational  `[BUILT]`
pulse-trajectory optimizer (ramp-up -> flat-top -> ramp-down).
*provenance SIM · retired_by **operational scenario optimizer**.*

### KQROSS — QRE  `[BUILT]`
fault-tolerant quantum resource estimator + crossover analysis — when/if a quantum computer beats the best classical method.
*provenance SIM · retired_by **fault-tolerant quantum hardware (not available this decade)** · gates AC-43, BR-SX-08.*

### KSENSE — QSENSE  `[BUILT]`
quantum-sensor diagnostic evaluation (NV/SQUID/SERF); honest null gain where disruptions live (model-limited).
*provenance SIM · retired_by **deployed physical diagnostic hardware** · gates HX-29.*

### KTENSOR — TN  `[PARTIAL]`
tensor-network simulation & compression — a classical many-body method and a bridge to quantum for kernel problems.
*provenance SIM · retired_by **exact many-body / quantum simulation** · gates KX-L3.*

## Phase 3 — roadmap

### KDRIVE — RL-control  `[BUILT]`
reinforcement-learning plasma controller (learned policy).
*provenance n/a · retired_by **validated RL controller**.*

### KECON — techno-economics  `[BUILT]`
GENERIC open techno-economics (LCOE) on the USER's inputs — FINANCIAL FIREWALL: no Kronos numbers, ever.
*provenance n/a · retired_by **detailed engineering cost model (never public)**.*

### KEDGE — edge-transport  `[ROADMAP]`
divertor / SOL / edge heat-flux transport surrogate.
*provenance n/a · retired_by **SOLPS-ITER / EIRENE**.*

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

### KRAD — radiation-control  `[ROADMAP]`
impurity seeding / radiation / divertor-detachment control.
*provenance n/a · retired_by **radiation transport + control**.*

