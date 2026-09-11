# KODEX KFUSE — card

- **name:** KFUSE
- **function:** MULTIFID
- **status:** BUILT
- **phase:** 2
- **provenance:** CGYRO-urep
- **retired_by:** CGYRO real-mass converged (mu=3672)
- **gates:** ['BR-L2-A1e', 'BR-L2-A1c-MS']
- **note:** multi-fidelity surrogate: cheap analytic low-fidelity + a residual GP correction from the mu=400 CGYRO points; real-mass gold is the 3rd fidelity
- **available:** True

Benchmark headline: multi-fidelity: RMSE 1.481→1.05 (R²=0.787); 3rd fidelity = real-mass gold (deferred)
