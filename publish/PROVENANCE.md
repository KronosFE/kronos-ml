# KODEX provenance & traceability (v0.1.0, 2026-09-10)

Each BUILT code traces: **code -> real benchmark -> on-disk data/method -> Kronos corpus DOI -> /kodex page.**

| # | Code | Ph | Provenance | Real source/method | Corpus anchor | Page |
|--|--|--|--|--|--|--|
| 01 | **KAIROS** | 1 | TWIN | receding-horizon MPC | references:10.5281/zenodo.21842371; references:10.5281/zenodo.22645689 | kairos |
| 02 | **KBREED** | 2 | SIM | openmc | references:10.5281/zenodo.22645689 | kbreed |
| 03 | **KBURN** | 2 | TWIN | plant systems model | references:10.5281/zenodo.21842371; references:10.5281/zenodo.22645689 | kburn |
| 04 | **KDRIVE** | 3 | TWIN | reinforcement learning | references:10.5281/zenodo.21842371; references:10.5281/zenodo.22645689 | kdrive |
| 05 | **KECON** | 3 | n/a | open LCOE methodology | references:10.5281/zenodo.21842371; references:10.5281/zenodo.22645689 | kecon |
| 06 | **KEYE** | 2 | SIM | sensor fusion, signal validation | references:10.5281/zenodo.21842371; references:10.5281/zenodo.22645689 | keye |
| 07 | **KFLOW** | 1 | SIM | message-passing GNN | references:10.5281/zenodo.21842371; references:10.5281/zenodo.22645689 | kflow |
| 08 | **KFLUX** | 2 | SIM | openmc | references:10.5281/zenodo.22645689 | kflux |
| 09 | **KFORGE** | 3 | CGYRO-urep | differential evolution over the fleet | isDerivedFrom:10.5281/zenodo.22136279 | kforge |
| 10 | **KFUEL** | 3 | SIM | fuel-cycle systems model | references:10.5281/zenodo.21842371; references:10.5281/zenodo.22645689 | kfuel |
| 11 | **KFUSE** | 2 | CGYRO-urep | multi-fidelity GP, CGYRO | isDerivedFrom:10.5281/zenodo.22136279 | kfuse |
| 12 | **KGATE** | 1 | TWIN | CBF certificate, hardware failsafe | references:10.5281/zenodo.21842371; references:10.5281/zenodo.22645689 | kgate |
| 13 | **KGEN** | 3 | ANALYTIC | generative model | references:10.5281/zenodo.21842371; references:10.5281/zenodo.22645689 | kgen |
| 14 | **KHALO** | 1 | SIM | deep-ensemble, conformal | references:10.5281/zenodo.21842371; references:10.5281/zenodo.22645689 | khalo |
| 15 | **KISO** | 2 | SIM | isotope PI servo, FENDL-3.2 activation | references:10.5281/zenodo.22645689; isDerivedFrom:https://www-nds.iaea.org/fendl/ | kiso |
| 16 | **KLAW** | 3 | CGYRO-urep | sparse regression / SINDy | isDerivedFrom:10.5281/zenodo.22136279 | klaw |
| 17 | **KMAT** | 1 | SIM | CHGNet, DFT | references:10.5281/zenodo.21842371; references:10.5281/zenodo.22645689 | kmat |
| 18 | **KOIL** | 1 | SIM | finite-element stress solve | references:10.5281/zenodo.21842371; references:10.5281/zenodo.22645689 | koil |
| 19 | **KORE** | 1 | ANALYTIC | Grad-Shafranov FD solver | references:10.5281/zenodo.21842371; references:10.5281/zenodo.22645689 | kore |
| 20 | **KPATH** | 2 | TWIN | MPC scenario optimizer | references:10.5281/zenodo.21842371; references:10.5281/zenodo.22645689 | kpath |
| 21 | **KPILOT** | 3 | n/a | fleet router / tool dispatcher | references:10.5281/zenodo.21842371; references:10.5281/zenodo.22645689 | kpilot |
| 22 | **KQROSS** | 2 | SIM | FT resource estimation | isDerivedFrom:10.5281/zenodo.21842371 | kqross |
| 23 | **KQUBIT** | 1 | SIM | VQE, quantum chemistry | isDerivedFrom:10.5281/zenodo.21842371 | kqubit |
| 24 | **KSENSE** | 2 | SIM | NV/SQUID/SERF magnetometry | isDerivedFrom:10.5281/zenodo.21842371 | ksense |
| 25 | **KWARD** | 1 | REAL-ANCHOR | real MAST shots (FAIR-MAST) | isDerivedFrom:https://mastapp.site; references:10.5281/zenodo.22645689 | kward |
| 26 | **KYRO** | 1 | CGYRO-urep | cgyro, gene, gs2, tglf | isDerivedFrom:10.5281/zenodo.22136279 | kyro |

## Held as ROADMAP (4)

- **KEDGE** (ROADMAP, phase 3) — divertor / SOL / edge heat-flux transport surrogate
- **KHEAT** (PARTIAL, phase 2) — RF / NBI heating & current-drive actuator-response surrogate
- **KRAD** (ROADMAP, phase 3) — impurity seeding / radiation / divertor-detachment control
- **KTENSOR** (PARTIAL, phase 2) — tensor-network simulation & compression — a classical many-body method and a bridge to quantum for kernel problems

Anchor DOIs: CGYRO `10.5281/zenodo.22136279` · Register `10.5281/zenodo.22645689` · AI+Quantum `10.5281/zenodo.21842371` · toolkit https://github.com/KronosFE/kronos-toolkit

