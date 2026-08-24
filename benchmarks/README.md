# PLOT3D I/O benchmark

A reproducible benchmark for the `lptlib` vectorized PLOT3D reader. It measures
how fast `lptlib` reads a multi-block structured grid and reconstructs the
per-block coordinate arrays, against a naive pure-Python reader and a compiled
Fortran reader. The numbers behind the performance statement in the JOSS paper
(`paper.md`) come from here.

Particle-tracking benchmarks, which characterize lptlib's central
Lagrangian-tracking capability against a reference ODE, a throughput baseline,
and a matched OpenFOAM parcel-tracking case, live in
[`tracking/`](tracking/README.md).

**Read this first.** The absolute read times in this benchmark move by more than
an order of magnitude between machines, for reasons that have nothing to do with
the readers: page-cache state, memory bandwidth, and whether the filesystem can
cache the file at all. Absolute milliseconds from a single machine are not a
defensible claim. What transfers between machines is the **ratio** between two
readers doing **matched work**, and even that needs an uncertainty attached. The
benchmark is built around those two facts.

---

## What changed, and why

An earlier version of this benchmark reported that the lptlib reader was "within
about a third of the compiled Fortran reader" (ratio 0.78) from a single
recorded run. Re-running the same code on a second machine gave ratio 0.09, a
factor of ten the other way. Both numbers were wrong, for two separate reasons,
and both are fixed here.

**1. The comparison was not like-for-like.** The Fortran baseline did a bare
stream read: it filled a native column-major `(ni,nj,nk,3)` array straight from
the file and stopped. The Python readers additionally reordered that memory,
either transposing each block to C-contiguous layout or scattering it into a
padded float64 array and reducing the per-block coordinate bounds. On a
2.04M-point grid the read itself costs about 2 ms in NumPy and about 3.5 ms in
Fortran; the reordering the Python side was additionally doing costs 28 to 44 ms.
The old benchmark charged all of that to "Python versus Fortran". It was a
workload difference, not a language difference.

The Fortran reader now has three explicit modes, and a Python reader is only
ever compared with the Fortran mode that does the same work:

| Group | Work done | Python reader | Fortran mode |
|---|---|---|---|
| `f32-contiguous` | read, then materialize each block C-contiguous, float32 | lptlib strided read; naive Python reader | `contig` |
| `f64-full` | read, then scatter into the padded float64 `(nimax,njmax,nkmax,3,nb)` array and reduce per-block bounds | `GridIO.read_grid` | `full` |
| `io-floor` | read only, no reorder, no upcast | `numpy.fromfile` plus zero-copy `order='F'` views | `raw` |

The `io-floor` group is a **diagnostic, not a language comparison**. It exists so
a reader can see how much of each real reader's time is file I/O rather than
memory reordering. Do not quote it as a Python-versus-Fortran result.

The Fortran `full` mode declares its array with the index order reversed
(`grd(nb,3,nk,nj,ni)`) so that the Fortran memory layout is byte-for-byte the
same as the C-ordered NumPy array `GridIO.read_grid` builds, and the scatter it
performs has the same access pattern. Its loop nest and its bounds reduction were
each measured against the obvious alternatives and the faster one kept, so the
baseline is not a straw man. Comments in `plot3d_read_fortran.f90` record which
alternatives were tried.

**2. The statistics were wrong, and the cache state was assumed rather than
measured.** The old runner reported the mean of a heavily right-skewed sample
(one reader had min 27 ms against a mean of 53 ms) and asserted a warm cache it
never checked. The recorded 53 ms for a 24.5 MB file works out to about
460 MB/s, which is cold-disk throughput, not page-cache throughput. Measured on
one machine here, the same Fortran binary takes 3.5 ms warm and 48.5 ms on a
genuinely cold cache -- and the old record's minimum was 48.0 ms. The old run was
almost certainly reading from disk on every repetition. A constant additive I/O
term of ~48 ms lands on every reader equally and pushes every ratio toward 1,
which is exactly how "within 30 percent of Fortran" was manufactured.

The old README also claimed "the relative ordering is stable" across hardware.
That claim was false and has been removed. The ordering flipped between the two
machines precisely because the cache state differed.

---

## Methodology

**Median, not mean.** Every reader is summarized by the median of its
repetitions, with the interquartile range and the median absolute deviation as
dispersion. Timings of a memory-bound reader on a shared machine are
right-skewed: a few repetitions get descheduled and land far above the body of
the distribution, and the mean chases them. The mean is still recorded in the
JSON so the skew is visible, but nothing in the report or the aggregation uses
it.

**Repetitions and warmups.** By default 31 timed repetitions for the fast
readers and 9 for the slow naive reader (5 with `--large`), after 5 discarded
warmup repetitions (1 for the naive reader). Both counts are recorded in every
result file. Change them with `--reps`, `--naive-reps`, `--warmups`,
`--naive-warmups`.

**Readers are interleaved, not run in sequence.** Every reader executes one
repetition at a time, round-robin, so all of them sample the same stretch of
wall-clock time. Running all of reader A's repetitions and then all of reader
B's makes every ratio hostage to anything that drifts over the run: another
process starting, thermal throttling, the page cache filling up. Interleaving
spreads that evenly, so it cancels out of the ratios instead of landing on
whichever reader happened to run while the machine was busy. The naive reader
gets fewer repetitions because each takes a second or more, and they are spread
across the run rather than bunched at the front.

Interleaving also removes an artifact. Thirty-one back-to-back repetitions of
the same reader leave a 24.5 MB grid sitting in the CPU's last-level cache, so
the reader is timed against L3 rather than against main memory. That is not
what an application does: it reads a grid once. With the readers interleaved,
each repetition finds the caches in a realistic state, and the measured I/O
floor drops from an implausible 11 GB/s to a believable 4-5 GB/s. Ratios are
also markedly more reproducible run to run.

**Uncertainty on the ratios, not just the timings.** The paper cites ratios, so
the ratios carry their own error bars. For each comparison the runner computes a
95 percent percentile bootstrap interval by resampling each reader's repetition
times with replacement 10 000 times and re-forming the ratio of medians. A ratio
of two noisy medians is noisier than either.

**Interference is reported, not hidden.** If a reader's interquartile range
exceeds 15 percent of its median, the run prints a `NOISY` flag next to that
reader, lists it under a noise warning, and records it in the result file under
`noisy_readers`. The pooled summary repeats the warning.

**Cache state is an explicit parameter.** `--cache-state warm` (the default)
performs 3 untimed whole-file priming reads before the timed repetitions and
records how many. `--cache-state cold` performs none. The benchmark never drops
the page cache itself: that needs root and is not portable. For a genuine
cold-cache run the operator must do one of these, and which one was done is
recorded:

```bash
# have the runner drop the cache before every timed repetition
python run_benchmark.py --cache-state cold \
    --drop-caches-cmd "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"

# or drop it by hand immediately before launching, and say so
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
python run_benchmark.py --cache-state cold --caches-dropped-manually
```

A `--cache-state cold` run with neither flag is recorded as
`cache_state_verified: false` and warned about, because the page cache is
certainly still warm from writing the grid.

**The declared cache state is checked against the data.** The benchmark does
not take `--cache-state warm` on trust. It records the whole-file read
throughput achieved by the I/O floor reader, and compares the first third of
its repetitions against the last third:

- A read served from the page cache is a memcpy and runs at gigabytes per
  second. A read served from a device runs at hundreds of megabytes per second.
  A run declaring a warm cache that achieves under 500 MB/s is flagged, because
  it is not warm whatever it declares. This is exactly the check whose absence
  let the old record be published as "warm cache" at an effective 460 MB/s.
- If the last third of the I/O floor's repetitions is more than about 1.7x
  slower than the first third, the grid file was evicted from the page cache
  part way through the run: the readers' own working set pushed it out. The run
  then mixes warm and cold reads and is flagged. This is a real risk with
  `--large`, whose 338 MB grid needs a machine with room for the file plus a
  working set of roughly 1.2 GB.

Both outcomes are recorded under `cache_check` in the result file and repeated
in the pooled summary.

**Cross-reader verification is kept.** Every reader is checked against a shared
checksum (the float64 sum of all coordinates, produced by the grid generator and
by an untimed pass in the Fortran program), so the timings are known to compare
identical reconstructed data. `checksum_verified` is recorded in every result
file.

---

## The grid

The benchmark grid is **synthetic**. `generate_grid.py` writes a smoothly warped
four-block structured mesh in the exact little-endian binary layout that
`GridIO.read_grid` consumes, so a reviewer can reproduce the numbers without any
large or restricted data file. The default grid is about 2.04 million points
(24.5 MB); `--large` uses about 28 million points (338 MB). To benchmark a real
grid instead, pass `--gridfile /path/to/grid.x`; any little-endian multi-block
PLOT3D grid in the `lptlib` format works.

---

## Running it

From this directory, with NumPy available and `lptlib` importable (the runner
adds `../src` to the path):

```bash
python run_benchmark.py                  # ~2.04M-point grid, warm cache
python run_benchmark.py --large          # ~28M-point grid, warm cache
python aggregate_results.py              # pool every machine's results
```

The default grid finishes in well under a minute. `--large` takes a few minutes,
mostly in the naive reader and the Fortran `full` mode, and needs roughly 2 GB
of free RAM for the padded double-precision array.

Useful options:

- `--machine-label LABEL` name this machine in the pooled results. Defaults to
  `$LPTLIB_BENCH_MACHINE`, then the hostname.
- `--reps N`, `--naive-reps N`, `--warmups N`, `--naive-warmups N` repetition counts.
- `--skip-naive` drop the slow naive reader (it dominates the wall-clock time).
- `--priming-reads N` untimed warm-cache priming reads (default 3).
- `--cache-state warm|cold`, `--drop-caches-cmd CMD`, `--caches-dropped-manually`.
- `--gfortran-flags "-O2"` optimization flags for the Fortran baseline. Recorded
  in the result file. (Flags barely matter here: `-O0` through
  `-O3 -march=native` span 3.5 to 5.3 ms on the `raw` mode.)
- `--bootstrap-resamples N` (default 10 000).
- `--gridfile PATH`, `--keep-grid`, `--outdir DIR`.

**Without gfortran.** If no Fortran compiler is found the benchmark prints
`gfortran not found; skipping Fortran baseline.` and still runs the
lptlib-versus-naive comparison. Set `GFORTRAN` to point at a specific compiler
(for example `GFORTRAN=gfortran-13`) if it is not on `PATH` under a standard
name.

---

## Contributing a result from another machine

The summary is only worth citing once more than one machine has reported. To
contribute:

1. Check out this branch and make `lptlib` importable (`pip install -e .` from
   the repository root, or rely on the runner's `../src` fallback). NumPy is
   required; `gfortran` and matplotlib are optional.
2. Run both grid sizes with a label that identifies your machine:

   ```bash
   cd benchmarks
   python run_benchmark.py          --machine-label my-laptop
   python run_benchmark.py --large  --machine-label my-laptop
   ```

   Use the same label for both runs. Pick something that distinguishes the
   machine, not the person: `epyc-7763-nvme`, `m2-macbook-air`, `hpc-login-node`.
3. Run on an otherwise idle machine. If the output prints a `NOISE WARNING` or
   a `CACHE WARNING`, re-run: a noisy result widens the pooled range without
   adding information, and a run that failed its own cache check is not
   measuring what it says it is. If a warning persists across re-runs it is a
   property of the machine, not bad luck; submit the result anyway, since the
   summary flags it and a flagged result still says something true about that
   machine, but do not let anyone quote it as a clean number.
4. Optionally add a cold-cache run, but only if you actually drop the page cache
   (see the two commands above). A cold run without a dropped cache is worse than
   no cold run.
5. `results/` will now contain
   `<your-label>_2Mpts_warm.json` / `.csv` and `<your-label>_28Mpts_warm.json` /
   `.csv`. The machine label is in the filename so results from different
   machines never collide.
6. Run `python aggregate_results.py` to regenerate `results/SUMMARY.md` and
   `results/benchmark_bar.png`, then commit the new result files together with
   the regenerated summary and figure.

Nothing about your machine beyond what is listed under "Outputs" below is
collected, and everything collected is visible in the JSON.

---

## Representative results

From `results/SUMMARY.md`, one machine (`lptlib-ci-xeon-2c`: 2-core Intel Xeon
at 2.10 GHz, 7.8 GiB RAM, ext4 on a virtualised block device, Python 3.11,
NumPy 2.4, gfortran 13.3 at `-O2`), 2.04M-point grid, warm cache verified at
5.0 GB/s whole-file read throughput:

| Matched-work comparison | Result | 95% CI |
|---|---|---|
| lptlib strided vs naive Python (f32) | **27.8x faster** | 26.1 - 28.8 |
| lptlib strided vs Fortran (f32, both C-contiguous) | 1.40x slower (ratio 0.71) | 0.69 - 0.75 |
| `GridIO.read_grid` vs Fortran (f64 padded + bounds) | **1.75x faster** | 1.68 - 1.85 |

The headline is that the two matched-work comparisons land on **the same order
of magnitude as compiled Fortran in both directions**: NumPy loses the
single-precision block transpose by about 40 percent and wins the
double-precision padded-array-plus-bounds reconstruction by about 75 percent,
because NumPy's copy and reduction kernels beat a straightforward hand-written
Fortran loop nest on that awkward strided scatter. The old "10x slower than
Fortran" and "within 30 percent of Fortran" figures were both artifacts, of the
workload mismatch and of the cold cache respectively.

**One machine is not enough.** These are single-machine numbers and the summary
says so on every range column. The ratios are the part expected to transfer;
please contribute a second machine.

**This machine could not measure the 28M-point grid validly.** With 7.8 GiB of
RAM it cannot hold a 338 MB grid in the page cache alongside a working set of
roughly 1.2 GB, so the `--large` run failed its own cache-residency check
(0.41 GB/s, device rather than page-cache throughput) and is recorded flagged.
The cold-cache bucket is likewise flagged: this is a virtual machine, and
dropping the guest page cache leaves the hypervisor's host-side cache intact, so
those reads still came from RAM at 2.5 GB/s. Both flags are in the summary. A
machine with more RAM, or a bare-metal machine for the cold bucket, would fix
each.

---

## Outputs

`run_benchmark.py` writes, per machine and per configuration:

- `results/<machine-label>_<grid-size>_<cache-state>.json` -- the full record:
  every repetition time, the median/IQR/MAD/min/max per reader, the bootstrap
  intervals on every ratio, the checksums, and a machine header.
- `results/<machine-label>_<grid-size>_<cache-state>.csv` -- the same per-reader
  summary in flat form.

The machine header in each JSON records: OS, kernel release and version, CPU
model, logical and physical core counts, total RAM, the filesystem type and
device backing the benchmark directory and a coarse storage class
(`nvme`, `ssd-or-nvme`, `rotational`, `memory-backed`, `overlay (container)`,
`network-or-fuse`), Python version and implementation, NumPy version, gfortran
version and the exact compile command including optimization flags, the load
average at the start of the run, the grid's block shapes, point count and byte
size, the declared cache state and whether it was verified, the priming-read
count, the repetition and warmup counts, and the machine label.

`aggregate_results.py` writes:

- `results/SUMMARY.md` -- one pooled markdown table per (grid size, cache state)
  bucket, giving for each reader the median across machines and the min-to-max
  range across machines, and the same for every ratio plus the widest
  per-machine bootstrap interval. This is the document the paper cites.
- `results/benchmark_bar.png` -- bar chart of the matched-work groups, with
  error bars spanning the across-machine range. Skip it with `--no-figure` or if
  matplotlib is unavailable; the summary is written either way.

Warm-cache and cold-cache results, and different grid sizes, are pooled into
separate buckets and never mixed. With a single machine reporting, every range
column reads `single machine` rather than implying a spread that does not exist.

The generated grid (`results/synthetic_grid.x`) and the compiled Fortran binary
are ignored by git.

---

## Files

| File | Purpose |
|---|---|
| `run_benchmark.py` | the runner: readers, timing, statistics, result files |
| `aggregate_results.py` | pools every result file into `results/SUMMARY.md` |
| `machine_info.py` | machine-metadata probes recorded in every result file |
| `generate_grid.py` | writes the synthetic multi-block PLOT3D grid |
| `naive_reader.py` | the nested-Python-loop baseline reader |
| `plot3d_read_fortran.f90` | the compiled Fortran baseline, three work modes |
