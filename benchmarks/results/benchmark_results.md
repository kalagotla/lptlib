# PLOT3D I/O benchmark results

Generated 2026-08-21 10:06:55 EDT

## Environment

- Platform: Linux-6.8.0-136-generic-x86_64-with-glibc2.35
- Processor: x86_64
- Python: 3.10.12
- NumPy: 2.2.6
- Fortran: GNU Fortran (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
- Grid: 4 blocks, 2,039,250 points, 24.5 MB (single precision on disk, synthetic)
- Block shapes: [(100, 90, 60), (110, 80, 55), (95, 85, 70), (120, 75, 50)]
- Repetitions: 15 (naive reader 7), warm cache
- Checksum verified across readers: True

## Single precision (read technique)

The PLOT3D file is single precision. All three readers do one buffered read and reconstruct the per-block float32 coordinate arrays, so the timing difference isolates the reconstruction strategy at matched precision.

| Reader | Reps | Mean (s) | Min (s) | Stdev (s) |
|---|---|---|---|---|
| lptlib strided read (float32) | 15 | 0.0687 | 0.0634 | 0.0048 |
| naive Python nested-loop (float32) | 7 | 0.5266 | 0.5081 | 0.0137 |
| Fortran stream read (float32, gfortran -O2) | 15 | 0.0534 | 0.0480 | 0.0070 |

- lptlib strided vs naive Python reader: **7.7x faster**
- lptlib strided vs compiled Fortran reader: **1.28x slower (ratio 0.78)**

## Double precision (library's full read)

`GridIO.read_grid` returns a float64 grd array and per-block coordinate bounds. For a like-for-like comparison the Fortran reader upcasts each block to float64 after reading. This is the cost the library actually pays.

| Reader | Reps | Mean (s) | Min (s) | Stdev (s) |
|---|---|---|---|---|
| lptlib GridIO.read_grid (float64, incl. metric bounds) | 15 | 0.0851 | 0.0755 | 0.0070 |
| Fortran stream read (float64, gfortran -O2) | 15 | 0.0597 | 0.0554 | 0.0065 |

- lptlib read_grid vs compiled Fortran, both float64: **1.43x slower (ratio 0.70)**
