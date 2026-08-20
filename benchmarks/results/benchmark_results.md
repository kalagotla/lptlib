# PLOT3D I/O benchmark results

Generated 2026-08-20 16:20:27 EDT

## Environment

- Platform: Linux-6.8.0-136-generic-x86_64-with-glibc2.35
- Processor: x86_64
- Python: 3.10.12
- NumPy: 2.2.6
- Fortran: GNU Fortran (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
- Grid: 4 blocks, 2,039,250 points, 24.5 MB (single precision, synthetic)
- Block shapes: [(100, 90, 60), (110, 80, 55), (95, 85, 70), (120, 75, 50)]
- Repetitions: 15 (naive reader 7), warm cache
- Checksum verified across readers: True

## Read-and-reconstruct task

All readers perform one buffered read of the file and reconstruct the per-block coordinate arrays. The I/O is identical, so the difference isolates the reconstruction strategy.

| Reader | Reps | Mean (s) | Min (s) | Stdev (s) |
|---|---|---|---|---|
| lptlib strided read (fromfile + F-order reshape) | 15 | 0.0697 | 0.0676 | 0.0019 |
| naive Python nested-loop reconstruct | 7 | 0.5039 | 0.4938 | 0.0067 |
| Fortran stream read (gfortran -O2) | 15 | 0.0565 | 0.0493 | 0.0073 |

## Speedups (mean time)

- lptlib strided reader vs naive Python reader: **7.2x faster**
- lptlib strided reader vs compiled Fortran reader: **1.23x slower** (ratio 0.81, i.e. the same order of magnitude)

## Full public API (for reference)

`GridIO.read_grid` additionally allocates a double-precision padded array and computes per-block coordinate bounds, which is grid-metric bookkeeping beyond the read technique above.

| Reader | Reps | Mean (s) | Min (s) | Stdev (s) |
|---|---|---|---|---|
| lptlib GridIO.read_grid (full: read + reshape + metric bounds) | 15 | 0.1590 | 0.1513 | 0.0065 |
