# KODEX KHEAT — card

- **name:** KHEAT
- **function:** heating&CD
- **status:** BUILT
- **phase:** 2
- **provenance:** SIM
- **retired_by:** full RF/NBI ray-tracing (GENRAY/TORAY/NUBEAM)
- **gates:** ['H10']
- **note:** heating & current-drive actuator-response surrogate — fast RandomForest over a 3888-point current-drive design scan; predicts driven current I_cd from the RF/NBI drive parameters + plasma state. Reduced CD model; ray-tracing = fidelity upgrade
- **available:** True

Benchmark headline: heating/current-drive surrogate: driven-current I_cd **R²=0.949** over 3888 configs (reduced CD; RF/NBI ray-tracing = upgrade)
