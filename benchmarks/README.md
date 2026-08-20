# PLOT3D I/O benchmark

This directory holds a reproducible benchmark for the `lptlib` vectorized
PLOT3D reader. It measures how fast `lptlib` reads a multi-block structured
grid and reconstructs the per-block coordinate arrays, and it compares that
against two baselines: a naive pure-Python reader and a compiled Fortran
reader. The results back the performance statement in the JOSS paper
(`paper.md`) and the README.

## What is measured

All readers perform one well-defined task. They read the same PLOT3D grid file
with a single buffered read and reconstruct the per-block `(ni, nj, nk, 3)`
coordinate arrays in memory. Because the file I/O is identical, the timing
difference isolates the reconstruction strategy.

1. **lptlib strided read**  one buffered `numpy.fromfile` call, then a strided
   Fortran-order reshape of each block. This is the read path used by
   `lptlib.io.GridIO.read_grid` and the technique described in the paper.
2. **naive Python reader** (`naive_reader.py`)  the same single buffered read,
   then nested Python loops over every grid point and coordinate. This is the
   "per-point Python overhead" baseline and isolates the striding speedup.
3. **Fortran reader** (`plot3d_read_fortran.f90`)  reads the same file with
   stream access, compiled with `gfortran -O2`. This is the "vs Fortran"
   comparison.

For transparency the benchmark also times the full public
`GridIO.read_grid`, which additionally allocates a double-precision padded
array and computes per-block coordinate bounds. That extra time is grid-metric
bookkeeping, not part of the read technique, so it is reported in a separate
"full public API" section rather than in the head-to-head comparison.

Every reader is verified against a shared checksum (the float64 sum of all
coordinates), so the timings compare identical reconstructed data.

## The grid

The benchmark grid is **synthetic**. `generate_grid.py` writes a smoothly warped
four-block structured mesh of about 2.04 million points (24.5 MB, single
precision) in the exact little-endian binary layout that `GridIO.read_grid`
consumes. It is synthetic so a reviewer can reproduce the numbers without any
large or restricted data file. To benchmark a real grid instead, pass
`--gridfile /path/to/grid.x`; any little-endian multi-block PLOT3D grid in the
`lptlib` format works.

## Reproduce

From this directory, with `lptlib` importable (the runner adds `../src` to the
path) and NumPy, matplotlib, and `gfortran` available:

```
python run_benchmark.py
```

That single command generates the grid, warms the OS cache, times every reader
over multiple repetitions, verifies the checksums, and writes the outputs
below. Useful options:

- `--reps N` repetitions for the fast readers (default 15).
- `--naive-reps N` repetitions for the slow naive reader (default 7).
- `--gridfile PATH` use an existing grid instead of the synthetic one.
- `--keep-grid` keep the generated grid file after the run.
- `--outdir DIR` change where results are written.

If `gfortran` is not on the `PATH`, set the `GFORTRAN` environment variable to
the compiler command (for example `GFORTRAN="gfortran-11"`), or point it at a
WSL or system install. If no Fortran compiler is found the benchmark still runs
the lptlib-vs-naive comparison and simply omits the Fortran row.

## Outputs

Written to `results/`:

- `benchmark_results.csv`  timings for every reader.
- `benchmark_results.md`  formatted table, environment, and speedups.
- `benchmark_results.json`  full machine-readable record.
- `benchmark_bar.png`  bar chart of the read-and-reconstruct times.

The generated grid (`results/synthetic_grid.x`) and the compiled Fortran binary
are ignored by git.

## Representative result

On the machine recorded in `results/benchmark_results.md` (Linux, Python 3.10,
NumPy 2.2, `gfortran` 11.4, 2.04M-point four-block grid, warm cache), the
read-and-reconstruct times were about 70 ms for the lptlib strided reader,
about 504 ms for the naive Python reader, and about 57 ms for the compiled
Fortran reader. That is roughly **7x faster than the naive Python reader** and
**within about 25% of the compiled Fortran reader** (ratio about 0.8, the same
order of magnitude). The full `GridIO.read_grid`, including the per-block
coordinate-bound metrics, was about 159 ms. Absolute times vary with hardware
and disk, but the relative ordering is stable. Re-run the command to regenerate
the numbers for your own machine.
