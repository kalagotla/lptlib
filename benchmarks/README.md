# PLOT3D I/O benchmark

This directory holds a reproducible benchmark for the `lptlib` vectorized
PLOT3D reader. It measures how fast `lptlib` reads a multi-block structured
grid and reconstructs the per-block coordinate arrays, and it compares that
against two baselines: a naive pure-Python reader and a compiled Fortran
reader. The results back the performance statement in the JOSS paper
(`paper.md`) and the README.

## What is measured

The PLOT3D file is single precision on disk. `lptlib.io.GridIO.read_grid`
stores the grid as float64 for downstream use, so comparisons are made at
matched precision in two groups.

Single-precision group (the read technique, all produce float32 block arrays):

1. **lptlib strided read**  one buffered `numpy.fromfile` call, then a strided
   Fortran-order reshape of each block materialized to a contiguous float32
   array. This is the read path used by `GridIO.read_grid`.
2. **naive Python reader** (`naive_reader.py`)  the same single buffered read,
   then nested Python loops over every grid point and coordinate. It isolates
   the striding speedup, because the file I/O is identical.
3. **Fortran reader** (`plot3d_read_fortran.f90`, precision 4)  reads the same
   file with stream access, compiled with `gfortran -O2`.

Double-precision group (matches what the library actually returns):

4. **full `GridIO.read_grid`**  builds the float64 padded grd array and the
   per-block coordinate bounds.
5. **Fortran reader** (precision 8)  reads the file and upcasts each block to
   float64, so it is compared against `read_grid` at the same precision.

Comparing float64 Python against float32 Fortran would not be meaningful, so the
two groups are kept separate and every row is labelled with its precision.

Every reader is verified against a shared checksum (the float64 sum of all
coordinates), so the timings compare identical reconstructed data.

## The grid

The benchmark grid is **synthetic**. `generate_grid.py` writes a smoothly warped
four-block structured mesh in the exact little-endian binary layout that
`GridIO.read_grid` consumes. It is synthetic so a reviewer can reproduce the
numbers without any large or restricted data file. The default grid is about
2.04 million points (24.5 MB); `--large` uses about 28 million points (338 MB).
To benchmark a real grid instead, pass `--gridfile /path/to/grid.x`; any
little-endian multi-block PLOT3D grid in the `lptlib` format works.

## Reproduce

From this directory, with `lptlib` importable (the runner adds `../src` to the
path) and NumPy, matplotlib, and `gfortran` available:

```
python run_benchmark.py            # ~2.04M-point grid
python run_benchmark.py --large    # ~28M-point grid
```

That single command generates the grid, warms the OS cache, times every reader
over multiple repetitions, verifies the checksums, and writes the outputs
below. Useful options:

- `--reps N` repetitions for the fast readers (default 15).
- `--naive-reps N` repetitions for the slow naive reader (default 7).
- `--large` use the ~28M-point grid.
- `--gridfile PATH` use an existing grid instead of the synthetic one.
- `--keep-grid` keep the generated grid file after the run.
- `--outdir DIR` change where results are written.

If `gfortran` is not on the `PATH`, set the `GFORTRAN` environment variable to
the compiler command (for example `GFORTRAN="gfortran-11"`), or point it at a
WSL or system install. If no Fortran compiler is found the benchmark still runs
the lptlib-vs-naive comparison and omits the Fortran rows.

## Outputs

Written to `results/`:

- `benchmark_results.csv`  timings for every reader.
- `benchmark_results.md`  formatted tables, environment, and speedups.
- `benchmark_results.json`  full machine-readable record.
- `benchmark_bar.png`  bar chart of the read-and-reconstruct times.

The generated grid (`results/synthetic_grid.x`) and the compiled Fortran binary
are ignored by git.

## Representative result

On the machine recorded in `results/benchmark_results.md` (Linux, Python 3.10,
NumPy 2.2, `gfortran` 11.4, warm cache) the default 2.04M-point grid gave, at
single precision, about 69 ms for the lptlib strided reader, about 527 ms for
the naive Python reader, and about 53 ms for the compiled Fortran reader. The
strided reader is therefore about **7.7x faster than the naive Python reader**
and within about 30 percent of the compiled Fortran reader (ratio 0.78, the same
order of magnitude). The naive-reader margin grows with grid size, reaching
about **20x on the 28M-point grid**, because the naive path is linear in Python.

At double precision, the full `GridIO.read_grid` took about 85 ms, about
**1.4x the time of the same Fortran reader upcast to float64** (ratio 0.70). The
per-block coordinate bounds in `read_grid` are computed on the single-precision
block views rather than on the padded float64 array, which keeps that metric
step cheap; without it the double-precision read was roughly three times slower
on large grids. Absolute times vary with hardware and disk, but the relative
ordering is stable. Re-run the command to regenerate the numbers for your own
machine.
