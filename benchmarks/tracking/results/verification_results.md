# lptlib particle-tracking accuracy verification

Generated 2026-08-22 18:36:41 EDT

lptlib's adaptive tracker is compared against a high-accuracy SciPy DOP853 reference that integrates the identical particle equation of motion, using lptlib's own drag-coefficient routine and the identical piecewise-linear velocity field. A particle relaxes across a smooth velocity ramp from 12 to 4 m/s.

| Drag model | dp (m) | rho_p | tau (s) | lptlib steps | final vp (m/s) | rel L2 error | max abs error (m/s) |
|---|---|---|---|---|---|---|---|
| stokes | 1.0e-06 | 1000 | 3.010e-06 | 2634 | 4.0000 | 3.82e-04 | 4.32e-03 |
| sphere | 1.0e-06 | 1000 | 3.010e-06 | 2618 | 4.0000 | 3.72e-04 | 4.22e-03 |

The reference downstream velocity is 4 m/s; agreement to a small relative L2 error confirms the tracker integrates the particle dynamics correctly.
