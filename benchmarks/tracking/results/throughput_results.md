# lptlib particle-tracking throughput

Generated 2026-08-22 18:38:59 EDT

- Platform: Linux-6.8.0-136-generic-x86_64-with-glibc2.35
- Python: 3.10.12
- NumPy: 2.2.6
- Particles: 20
- OceanParcels run: False

| Method | Particles | Seconds | Particles/sec | Total steps |
|---|---|---|---|---|
| lptlib (adaptive, curvilinear search+interp) | 20 | 36.183 | 0.55 | 52680 |
| reference NumPy (vectorized, structured 1D) | 20 | 0.008 | 2626.31 | 80020 |

These comparators solve different problems and are not a like-for-like contest. The reference NumPy integrator is a specialized structured-1D vectorized solver with no point location, so it is far faster per particle and marks an upper bound. OceanParcels, when present, advects passive particles on a flat mesh and is included only as cross-library context. lptlib's cost reflects the general curvilinear point-location, interpolation, compressible drag suite, and adaptive stepping that it performs for every particle. A physics-matched cross-code comparison lives under `openfoam/`.
