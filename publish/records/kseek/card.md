# KODEX KSEEK — card

- **name:** KSEEK
- **function:** ACTIVE
- **status:** BUILT
- **phase:** 3
- **provenance:** SIM
- **retired_by:** the full CGYRO parameter scan (once every point is simulated)
- **gates:** ['BR-L2-A1e']
- **note:** active-learning acquisition — proposes the NEXT most-informative CGYRO run from KYRO's GP posterior (max-variance / uncertainty sampling), so expensive GPU-hours go where the surrogate is least sure. A real experimental-design tool
- **available:** True

Benchmark headline: active-learning: proposes next CGYRO run (a/L_T=2.225, shear=0.58); LOO std-vs-error corr -0.283 (honest: weak on 16 pts)
