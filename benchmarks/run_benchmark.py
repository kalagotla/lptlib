"""Reproducible PLOT3D I/O benchmark for lptlib.

WHAT THIS MEASURES
------------------
Several readers reconstruct the same multi-block PLOT3D grid file, and the
read-and-reconstruct work is timed. The point of the benchmark is the *ratio*
between readers, not the absolute milliseconds, because the absolutes move by
an order of magnitude with page-cache state and memory bandwidth.

Readers are grouped by the amount of work they do. A reader is only ever
compared against another reader in the same group. This matters: an earlier
version of this benchmark compared the lptlib reader, which reorders memory,
against a Fortran reader that only streamed bytes into a native column-major
array, and reported the resulting 10x gap as if it were a language gap. It was
a workload gap. The groups below exist so that cannot happen again.

  group "io-floor" (not a fair language comparison, reported for diagnosis)
    numpy raw read       one numpy.fromfile plus zero-copy order='F' reshape
                         views. No block is materialized.
    fortran raw read     one stream read per block into a native (ni,nj,nk,3)
                         column-major array. No reorder, no upcast.
    Both are essentially a memcpy out of the page cache. The gap between this
    group and the groups below is how much of each real reader's time is
    memory reordering rather than file I/O.

  group "f32-contiguous" (matched work, single precision)
    lptlib strided       the GridIO.read_grid read path: one buffered
                         numpy.fromfile, then a strided order='F' reshape of
                         each block materialized C-contiguous.
    naive Python         the same single buffered read, then nested Python
                         loops over every point and coordinate.
    fortran contig       stream read, then the same column-major to row-major
                         block transpose, written out by hand in Fortran.

  group "f64-full" (matched work, double precision, what the library returns)
    lptlib read_grid     the full public GridIO.read_grid: the padded float64
                         (nimax,njmax,nkmax,3,nb) array plus per-block
                         coordinate bounds.
    fortran full         stream read, scatter into an array with byte-identical
                         layout, plus the same per-block bounds.

STATISTICS
----------
Timings of a memory-bound reader on a shared machine are right-skewed: a few
repetitions get descheduled or hit a page fault storm and land far above the
body of the distribution. The mean chases those, so this benchmark reports the
*median* with the interquartile range and the median absolute deviation, over
a stated number of repetitions after a stated number of discarded warmups.

Readers are interleaved round-robin rather than run one after another, so all
of them sample the same stretch of wall-clock time and anything that drifts
over the run cancels out of the ratios. Interleaving also stops a reader being
timed against a last-level cache that thirty back-to-back repetitions of itself
have left conveniently full.

The paper cites ratios, so the ratios carry their own uncertainty: a
percentile bootstrap confidence interval is computed by resampling each
reader's repetition times with replacement and re-forming the ratio of medians.
A ratio of two noisy medians is noisier than either.

If a reader's IQR exceeds a fraction of its median the run is flagged as noisy
in the output and in the result file, rather than being hidden behind a mean.

CACHE STATE
-----------
Cache state is an explicit recorded parameter, never an assumption.

  --cache-state warm (default)  N untimed priming reads of the whole file
                                before the timed repetitions, N recorded.
  --cache-state cold            no priming reads. The benchmark does NOT drop
                                the page cache itself: that needs root and is
                                not portable. Either pass
                                --drop-caches-cmd "<command>" to have the
                                operator's own command run before every timed
                                repetition, or drop the cache by hand before
                                launching and pass --caches-dropped-manually.
                                Whichever was done is recorded, and a cold run
                                with neither is recorded as unverified.

Whichever state is declared, it is checked against the data: the whole-file
read throughput of the I/O floor reader is recorded, and its first third of
repetitions is compared against its last third. A "warm" run achieving device
rather than page-cache throughput, or one whose grid file gets evicted part way
through by the readers' own working set, is flagged rather than believed.

Run:
    python run_benchmark.py                     # ~2.0M-point grid, warm cache
    python run_benchmark.py --large             # ~28M-point grid, warm cache
    python run_benchmark.py --machine-label my-laptop
    python aggregate_results.py                 # pool every machine's results

author: benchmark tooling for lptlib (Dilip Kalagotla)
"""

import argparse
import csv
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import generate_grid  # noqa: E402
import machine_info  # noqa: E402
from naive_reader import read_grid_naive  # noqa: E402

# ~2.0M points, four uneven blocks.
DEFAULT_BLOCKS = [(100, 90, 60), (110, 80, 55), (95, 85, 70), (120, 75, 50)]
# ~28M points, four uneven blocks.
LARGE_BLOCKS = [(400, 320, 60), (360, 300, 64), (340, 310, 66), (380, 300, 58)]

# A run whose interquartile range exceeds this fraction of its median is
# flagged as noisy. 0.15 is loose enough not to fire on a quiet machine and
# tight enough to catch a machine that is doing something else at the time.
NOISE_IQR_FRACTION = 0.15

# A whole-file read served from the page cache is a memcpy and runs at
# gigabytes per second on any machine this benchmark is likely to meet. A read
# served from a device runs at hundreds of megabytes per second. A warm-cache
# run whose I/O floor falls below this rate is not warm, whatever it declares:
# this is the check whose absence let an earlier recorded run be published as
# "warm cache" at an effective 460 MB/s.
WARM_CACHE_MIN_BYTES_PER_S = 500e6

# The readers allocate hundreds of megabytes each repetition. On a machine
# without enough RAM to hold the grid file and that working set at once, the
# kernel evicts the file part way through the run and the later repetitions are
# reading from the device. Re-measuring the I/O floor at the end catches it: a
# throughput that has fallen by more than this fraction means the run mixed
# warm and cold reads and its cache state is not what it declares.
CACHE_RETENTION_MIN_FRACTION = 0.6

RESULT_FORMAT_VERSION = 2

# Reader group names, used to keep unlike readers from being compared.
GROUP_IO_FLOOR = "io-floor"
GROUP_F32 = "f32-contiguous"
GROUP_F64 = "f64-full"


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------

def load_gridio():
    """Return lptlib's GridIO class.

    Prefer the public import. If lptlib.io pulls in optional dependencies that
    are not installed (mpi4py via DataIO, say), fall back to loading the
    plot3dio submodule directly, where GridIO is defined. The reader under test
    is identical either way.
    """
    try:
        from lptlib.io import GridIO
        return GridIO
    except Exception:
        import importlib.util
        mod_path = SRC / "lptlib" / "io" / "plot3dio.py"
        spec = importlib.util.spec_from_file_location("lptlib_plot3dio", mod_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.GridIO


def _read_header(handle):
    nb = int(np.fromfile(handle, dtype="i4", count=1)[0])
    dims = np.fromfile(handle, dtype="i4", count=3 * nb)
    return nb, dims[0::3], dims[1::3], dims[2::3]


def read_numpy_raw(filename, data_type="f4"):
    """I/O floor: one buffered read plus zero-copy order='F' reshape views.

    Nothing is materialized, so this is the numpy counterpart of the Fortran
    ``raw`` mode: both end up with per-block column-major arrays over bytes
    that came straight off the file. It is here to separate file I/O cost from
    memory-reorder cost, not as a usable reader.
    """
    with open(filename, "rb") as grid:
        nb, ni, nj, nk = _read_header(grid)
        nt = ni * nj * nk * 3
        buf = np.fromfile(grid, dtype=data_type, count=int(nt.sum()))
    blocks = []
    for i in range(nb):
        start = int(nt[:i].sum())
        end = start + int(nt[i])
        blocks.append(buf[start:end].reshape(
            (int(ni[i]), int(nj[i]), int(nk[i]), 3), order="F"))
    return blocks


def read_grid_strided(filename, data_type="f4"):
    """The lptlib strided read technique, materialized C-contiguous, float32.

    This is the read path of ``GridIO.read_grid``: one buffered numpy.fromfile
    for the whole float block, then a Fortran-order reshape of each block's
    slice. The slice is materialized to a C-contiguous array so that this,
    the naive Python reader, and the Fortran ``contig`` mode all end with the
    same bytes in the same layout and the comparison is like-for-like. The
    materialization is a full column-major to row-major transpose and is the
    dominant cost; see the io-floor group for the read-only cost.
    """
    with open(filename, "rb") as grid:
        nb, ni, nj, nk = _read_header(grid)
        nt = ni * nj * nk * 3
        buf = np.fromfile(grid, dtype=data_type, count=int(nt.sum()))
        blocks = []
        for i in range(nb):
            start = int(nt[:i].sum())
            end = start + int(nt[i])
            blocks.append(np.ascontiguousarray(
                buf[start:end].reshape(
                    (int(ni[i]), int(nj[i]), int(nk[i]), 3), order="F")))
    return blocks


def checksum_blocks(blocks):
    return float(sum(b.sum(dtype=np.float64) for b in blocks))


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

class Timer:
    """Holds the run's cache policy: priming reads and per-repetition action.

    The timing loop itself lives in ``run_interleaved``; this only decides what
    happens to the page cache before the run and between repetitions.
    """

    def __init__(self, cache_state, priming_reads, drop_caches_cmd, gridfile):
        self.cache_state = cache_state
        self.priming_reads = priming_reads
        self.drop_caches_cmd = drop_caches_cmd
        self.gridfile = Path(gridfile)

    def prime(self):
        """Untimed whole-file reads to bring the file into the page cache."""
        if self.cache_state != "warm":
            return 0
        for _ in range(self.priming_reads):
            with open(self.gridfile, "rb") as handle:
                while handle.read(1 << 22):
                    pass
        return self.priming_reads

    def before_rep(self):
        """Per-repetition cache action, if the operator supplied one."""
        if self.drop_caches_cmd:
            subprocess.run(self.drop_caches_cmd, shell=True, check=True)



class Job:
    """One reader under measurement, executed one repetition at a time.

    Repetitions are executed one at a time so that the readers can be
    interleaved. Running all of reader A's repetitions and then all of reader
    B's makes every ratio hostage to anything that drifts over the span of the
    run: another process starting, thermal throttling, the page cache filling
    up. Interleaving spreads any such drift evenly over every reader, so it
    cancels out of the ratios instead of being attributed to whichever reader
    happened to run while the machine was busy.
    """

    def __init__(self, key, label, group, reps, warmups, implementation):
        self.key = key
        self.label = label
        self.group = group
        self.reps = reps
        self.warmups = warmups
        self.total = reps + warmups
        self.implementation = implementation
        self.times = []
        self.checksum = None
        self._executed = 0

    def run_one(self):
        raise NotImplementedError

    def step(self):
        elapsed, checksum = self.run_one()
        if self.checksum is None and checksum is not None:
            self.checksum = checksum
        if self._executed >= self.warmups:
            self.times.append(elapsed)
        self._executed += 1

    def stats(self):
        summary = summarize(self.times)
        summary.update({"label": self.label, "group": self.group,
                        "warmups_discarded": self.warmups,
                        "implementation": self.implementation})
        return summary


class PythonJob(Job):
    def __init__(self, key, label, group, func, reps, warmups):
        super().__init__(key, label, group, reps, warmups, "python")
        self.func = func

    def run_one(self):
        t0 = time.perf_counter()
        result = self.func()
        elapsed = time.perf_counter() - t0
        checksum = _result_checksum(result) if self.checksum is None else None
        del result
        return elapsed, checksum


class FortranJob(Job):
    """One Fortran mode, invoked once per repetition.

    The timing is taken inside the compiled program with system_clock, so
    process startup stays outside every measurement. The untimed verification
    pass is requested only on the first invocation; later invocations pass
    "nocheck" so a full extra read-and-sum is not paid on every repetition.
    """

    def __init__(self, key, label, group, exe, gridfile, mode, reps, warmups):
        super().__init__(key, label, group, reps, warmups, "fortran")
        self.exe = exe
        self.gridfile = gridfile
        self.mode = mode

    def stats(self):
        summary = super().stats()
        summary["fortran_mode"] = self.mode
        return summary

    def run_one(self):
        want = "check" if self.checksum is None else "nocheck"
        cmd = [str(self.exe), str(self.gridfile), "1", self.mode, want]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("Fortran reader failed:\n" + proc.stdout + proc.stderr)
        times, checksum = _parse_fortran_output(proc.stdout)
        if not times:
            raise RuntimeError("Fortran reader produced no timing:\n" + proc.stdout)
        return times[0], checksum


def _schedule(jobs):
    """Round indices at which each job runs, spread over the whole run.

    Jobs with fewer repetitions than the longest job (the naive reader has far
    fewer, because each of its repetitions is a second or more) are spread
    evenly across the rounds rather than bunched at the start, so they sample
    the same stretch of wall-clock time as everything else.
    """
    total_rounds = max(job.total for job in jobs)
    plan = {}
    for job in jobs:
        if job.total >= total_rounds:
            rounds = list(range(total_rounds))
        elif job.total == 1:
            rounds = [0]
        else:
            step = (total_rounds - 1) / (job.total - 1)
            rounds = [int(round(index * step)) for index in range(job.total)]
        for index in rounds:
            plan.setdefault(index, []).append(job)
    return total_rounds, plan


def run_interleaved(jobs, timer, progress=None):
    """Execute every job's repetitions round-robin over the whole run."""
    total_rounds, plan = _schedule(jobs)
    for index in range(total_rounds):
        for job in plan.get(index, []):
            timer.before_rep()
            job.step()
        if progress:
            progress(index + 1, total_rounds)
    return jobs


def _result_checksum(result):
    if isinstance(result, list):
        return checksum_blocks(result)
    return float(np.float64(result.sum(dtype=np.float64)))


def summarize(times):
    """Median-centered summary of a timing sample.

    The mean is deliberately absent from the headline fields. It is kept as
    ``mean_s`` only so a reader can see how far the distribution is skewed;
    nothing in the report or the aggregation uses it.
    """
    ordered = sorted(times)
    n = len(ordered)
    median = statistics.median(ordered)
    if n >= 4:
        q1 = _percentile(ordered, 25.0)
        q3 = _percentile(ordered, 75.0)
    else:
        q1, q3 = ordered[0], ordered[-1]
    mad = statistics.median([abs(t - median) for t in ordered])
    iqr = q3 - q1
    return {
        "n": n,
        "median_s": median,
        "q1_s": q1,
        "q3_s": q3,
        "iqr_s": iqr,
        "mad_s": mad,
        "min_s": ordered[0],
        "max_s": ordered[-1],
        "mean_s": statistics.mean(ordered),
        "rel_iqr": (iqr / median) if median > 0 else None,
        "noisy": bool(median > 0 and (iqr / median) > NOISE_IQR_FRACTION),
        # Run order, not sorted: the cache-residency check compares the first
        # third of the run against the last third, which only works if the
        # order survives.
        "times_s": list(times),
    }


def _percentile(ordered, pct):
    """Linear-interpolation percentile of an already-sorted sample."""
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def bootstrap_ratio(numer_times, denom_times, resamples=10000, seed=20240921):
    """Percentile bootstrap CI for median(numer) / median(denom).

    Each reader's repetitions are resampled with replacement independently,
    the two medians are recomputed, and their ratio recorded. The 2.5th and
    97.5th percentiles of those ratios are the 95 percent interval. The point
    estimate is the ratio of the observed medians, not the bootstrap mean.
    """
    if not numer_times or not denom_times:
        return None
    rng = random.Random(seed)
    n_num, n_den = len(numer_times), len(denom_times)
    ratios = []
    for _ in range(resamples):
        a = statistics.median([numer_times[rng.randrange(n_num)] for _ in range(n_num)])
        b = statistics.median([denom_times[rng.randrange(n_den)] for _ in range(n_den)])
        if b > 0:
            ratios.append(a / b)
    if not ratios:
        return None
    ratios.sort()
    point = statistics.median(numer_times) / statistics.median(denom_times)
    return {
        "point": point,
        "ci95_low": _percentile(ratios, 2.5),
        "ci95_high": _percentile(ratios, 97.5),
        "iqr_low": _percentile(ratios, 25.0),
        "iqr_high": _percentile(ratios, 75.0),
        "resamples": len(ratios),
    }


# --------------------------------------------------------------------------
# Fortran baseline
# --------------------------------------------------------------------------

def find_gfortran():
    """Return a runnable gfortran command, or None.

    Honors the GFORTRAN environment variable so a reviewer can point at a
    specific compiler. Falls back to gfortran / gfortran-13 / gfortran-11.
    """
    env = os.environ.get("GFORTRAN")
    if env:
        return env.split()
    for name in ("gfortran", "gfortran-13", "gfortran-12", "gfortran-11"):
        if shutil.which(name):
            return [name]
    return None


def compile_fortran(gfortran, flags):
    """Compile the Fortran reader once. Returns (exe_path, compile_cmd)."""
    src = HERE / "plot3d_read_fortran.f90"
    exe = HERE / "plot3d_read_fortran"
    compile_cmd = list(gfortran) + list(flags) + [str(src), "-o", str(exe)]
    proc = subprocess.run(compile_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("gfortran compilation failed:\n"
                           + " ".join(compile_cmd) + "\n" + proc.stdout + proc.stderr)
    return exe, " ".join(compile_cmd)


def _parse_fortran_output(stdout):
    times, checksum = [], None
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "rep":
            times.append(float(parts[2]))
        elif len(parts) == 2 and parts[0] == "checksum":
            checksum = float(parts[1])
    return times, checksum


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def ratio_phrase(ratio, subject="lptlib"):
    """Describe a ratio of the form baseline_time / subject_time."""
    if ratio >= 1.0:
        return f"{subject} {ratio:.2f}x faster"
    return f"{subject} {1.0 / ratio:.2f}x slower"


def ms(value):
    return value * 1000.0


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=31,
                        help="timed repetitions for the fast readers (default 31)")
    parser.add_argument("--naive-reps", type=int, default=None,
                        help="timed repetitions for the slow naive reader "
                             "(default 9, or 5 with --large)")
    parser.add_argument("--warmups", type=int, default=5,
                        help="leading repetitions discarded before timing (default 5)")
    parser.add_argument("--naive-warmups", type=int, default=1,
                        help="discarded repetitions for the naive reader (default 1)")
    parser.add_argument("--priming-reads", type=int, default=3,
                        help="untimed whole-file reads before timing, warm cache "
                             "only (default 3)")
    parser.add_argument("--cache-state", choices=("warm", "cold"), default="warm",
                        help="explicitly declare the page-cache state of the run")
    parser.add_argument("--drop-caches-cmd", default=None,
                        help="shell command run before every timed repetition, for "
                             "cold-cache runs, e.g. "
                             "\"sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'\"")
    parser.add_argument("--caches-dropped-manually", action="store_true",
                        help="record that the operator dropped the page cache by hand "
                             "immediately before launching a cold-cache run")
    parser.add_argument("--machine-label", default=None,
                        help="label for this machine in the pooled results; defaults "
                             "to $LPTLIB_BENCH_MACHINE, then the hostname")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000,
                        help="bootstrap resamples for the ratio intervals (default 10000)")
    parser.add_argument("--large", action="store_true",
                        help="use a ~28M-point grid instead of the ~2M default")
    parser.add_argument("--outdir", default=str(HERE / "results"),
                        help="directory for the result files")
    parser.add_argument("--gridfile", default=None,
                        help="path to the grid file (generated if missing)")
    parser.add_argument("--keep-grid", action="store_true",
                        help="do not delete the generated grid file at the end")
    parser.add_argument("--gfortran-flags", default=None,
                        help="optimization flags for the Fortran baseline "
                             "(default -O2, or $GFORTRAN_FLAGS)")
    parser.add_argument("--skip-naive", action="store_true",
                        help="skip the slow naive Python reader")
    return parser


def main():
    args = build_parser().parse_args()

    naive_reps = args.naive_reps
    if naive_reps is None:
        naive_reps = 5 if args.large else 9

    blocks_spec = LARGE_BLOCKS if args.large else DEFAULT_BLOCKS
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gridfile = Path(args.gridfile) if args.gridfile else (outdir / "synthetic_grid.x")
    gridfile.parent.mkdir(parents=True, exist_ok=True)

    label = machine_info.machine_label(args.machine_label)

    print("Generating synthetic multi-block PLOT3D grid ...")
    nb, blocks, total_points, gen_checksum = generate_grid.write_grid(
        str(gridfile), blocks=blocks_spec)
    file_size = gridfile.stat().st_size
    print(f"  blocks={nb} points={total_points:,} size={file_size/1e6:.1f} MB")

    if args.cache_state == "cold" and not (args.drop_caches_cmd
                                           or args.caches_dropped_manually):
        print("  WARNING: --cache-state cold with neither --drop-caches-cmd nor\n"
              "           --caches-dropped-manually. The page cache is almost\n"
              "           certainly still warm from writing the grid. This run is\n"
              "           recorded as cache_state_verified=false.")

    timer = Timer(args.cache_state, args.priming_reads, args.drop_caches_cmd, gridfile)
    primed = timer.prime()
    print(f"  cache state: {args.cache_state} "
          f"({primed} untimed priming reads"
          + (", per-rep drop command set" if args.drop_caches_cmd else "") + ")")

    gfortran = find_gfortran()
    flags = (args.gfortran_flags.split() if args.gfortran_flags
             else os.environ.get("GFORTRAN_FLAGS", "-O2").split())
    exe = compile_cmd = None
    if gfortran:
        try:
            exe, compile_cmd = compile_fortran(gfortran, flags)
        except RuntimeError as exc:
            print("  Fortran baseline could not be compiled:", exc)
            gfortran = None
    else:
        print("  gfortran not found; skipping Fortran baseline.")

    GridIO = load_gridio()

    def read_grid_full():
        grid = GridIO(str(gridfile))
        grid.read_grid(data_type="f4")
        return grid.grd

    jobs = [
        PythonJob("numpy_raw_read", "numpy raw read (f32 views, no reorder)",
                  GROUP_IO_FLOOR, lambda: read_numpy_raw(str(gridfile)),
                  args.reps, args.warmups),
        PythonJob("lptlib_strided", "lptlib strided read (f32, C-contiguous blocks)",
                  GROUP_F32, lambda: read_grid_strided(str(gridfile)),
                  args.reps, args.warmups),
        PythonJob("lptlib_read_grid", "lptlib GridIO.read_grid (f64 padded + bounds)",
                  GROUP_F64, read_grid_full, args.reps, args.warmups),
    ]
    if not args.skip_naive:
        jobs.append(PythonJob(
            "naive_python", "naive Python nested-loop (f32, C-contiguous blocks)",
            GROUP_F32, lambda: read_grid_naive(str(gridfile)),
            naive_reps, args.naive_warmups))
    if gfortran and exe:
        jobs.extend([
            FortranJob("fortran_raw", "Fortran raw stream read (f32, no reorder)",
                       GROUP_IO_FLOOR, exe, gridfile, "raw", args.reps, args.warmups),
            FortranJob("fortran_contig",
                       "Fortran stream read + transpose (f32, C-contiguous)",
                       GROUP_F32, exe, gridfile, "contig", args.reps, args.warmups),
            FortranJob("fortran_full",
                       "Fortran stream read + padded f64 + bounds",
                       GROUP_F64, exe, gridfile, "full", args.reps, args.warmups),
        ])

    print(f"Timing {len(jobs)} readers, interleaved round-robin so that any drift "
          f"over the\nrun spreads evenly across them and cancels out of the ratios.")
    for job in jobs:
        print(f"  {job.label:<52s} {job.reps} reps + {job.warmups} warmups")

    def progress(done, total):
        if done == total or done % max(1, total // 10) == 0:
            print(f"  round {done}/{total}", flush=True)

    try:
        run_interleaved(jobs, timer, progress)
    except RuntimeError as exc:
        print("  A reader failed:", exc)
        raise

    readers = {job.key: job.stats() for job in jobs}
    checksums = {"generator": gen_checksum}
    checksums.update({job.key: job.checksum for job in jobs})

    # Cache-residency diagnostic. Two questions, both answered from the
    # interleaved run itself rather than assumed.
    #
    # 1. Is the declared cache state consistent with the observed throughput?
    #    A whole-file read served from the page cache is a memcpy; served from
    #    a device it is an order of magnitude slower. This is the check whose
    #    absence let an earlier run be published as "warm cache" at an
    #    effective 460 MB/s.
    # 2. Did the file stay resident for the whole run? The readers allocate
    #    hundreds of megabytes per repetition, and on a machine without the
    #    headroom to hold both, the kernel evicts the grid part way through.
    #    Comparing the first third of the I/O floor's repetitions against the
    #    last third catches that; interleaving means both thirds sample the
    #    same readers under the same conditions.
    io_floor = readers["numpy_raw_read"]
    io_times = io_floor["times_s"]
    third = max(1, len(io_times) // 3)
    early = statistics.median(io_times[:third])
    late = statistics.median(io_times[-third:])
    cache_check = {
        "io_floor_reader": "numpy_raw_read",
        "median_bytes_per_s": file_size / io_floor["median_s"],
        "early_median_s": early,
        "late_median_s": late,
        "early_bytes_per_s": file_size / early,
        "late_bytes_per_s": file_size / late,
        "retention_fraction": early / late,
        "warm_threshold_bytes_per_s": WARM_CACHE_MIN_BYTES_PER_S,
        "retention_threshold": CACHE_RETENTION_MIN_FRACTION,
    }
    cache_check["throughput_consistent_with_warm_cache"] = bool(
        cache_check["median_bytes_per_s"] >= WARM_CACHE_MIN_BYTES_PER_S)
    cache_check["cache_retained_through_run"] = bool(
        cache_check["retention_fraction"] >= CACHE_RETENTION_MIN_FRACTION)
    # The check is symmetric. A run declaring a cold cache but achieving
    # page-cache throughput is not cold, and that is easy to end up with: in a
    # virtual machine, dropping the guest's page cache leaves the hypervisor's
    # host-side cache untouched, so the "cold" read is still served from RAM,
    # just somebody else's. Saying so is the difference between a result that
    # describes itself and one that misleads.
    cache_check["throughput_consistent_with_cold_cache"] = bool(
        cache_check["median_bytes_per_s"] < WARM_CACHE_MIN_BYTES_PER_S)
    if args.cache_state == "warm":
        cache_check["ok_for_declared_state"] = bool(
            cache_check["throughput_consistent_with_warm_cache"]
            and cache_check["cache_retained_through_run"])
    else:
        cache_check["ok_for_declared_state"] = cache_check[
            "throughput_consistent_with_cold_cache"]

    # Cross-reader verification: every reader must reconstruct the same data.
    tol = abs(gen_checksum) * 1e-4 + 1.0
    verified = all(value is not None and abs(value - gen_checksum) <= tol
                   for value in checksums.values())

    # Ratios. Every comparison stays inside one work group.
    ratio_specs = [
        ("naive_python_over_lptlib_strided", "naive_python", "lptlib_strided",
         GROUP_F32, "lptlib strided speedup over the naive Python reader",
         "lptlib strided"),
        ("fortran_over_lptlib_strided", "fortran_contig", "lptlib_strided",
         GROUP_F32, "lptlib strided vs Fortran, matched f32 C-contiguous work",
         "lptlib strided"),
        ("fortran_over_lptlib_read_grid", "fortran_full", "lptlib_read_grid",
         GROUP_F64, "lptlib read_grid vs Fortran, matched f64 padded+bounds work",
         "lptlib read_grid"),
        ("fortran_over_numpy_raw", "fortran_raw", "numpy_raw_read",
         GROUP_IO_FLOOR, "numpy vs Fortran at the I/O floor, no reorder either side",
         "numpy raw read"),
    ]
    ratios = {}
    for key, numer, denom, group, description, subject in ratio_specs:
        if numer in readers and denom in readers:
            boot = bootstrap_ratio(readers[numer]["times_s"], readers[denom]["times_s"],
                                   resamples=args.bootstrap_resamples)
            if boot:
                boot.update({"numerator": numer, "denominator": denom,
                             "group": group, "description": description,
                             "subject": subject})
                ratios[key] = boot

    meta = machine_info.collect(HERE, gfortran_cmd=gfortran, compile_cmd=compile_cmd,
                                numpy_version=np.__version__)
    record = {
        "result_format_version": RESULT_FORMAT_VERSION,
        "machine_label": label,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "machine": meta,
        "measurement": {
            "cache_state": args.cache_state,
            "cache_state_verified": bool(
                args.cache_state == "warm"
                or args.drop_caches_cmd or args.caches_dropped_manually),
            "priming_reads": primed,
            "drop_caches_cmd": args.drop_caches_cmd,
            "caches_dropped_manually": bool(args.caches_dropped_manually),
            "reps": args.reps,
            "naive_reps": naive_reps,
            "warmups_discarded": args.warmups,
            "naive_warmups_discarded": args.naive_warmups,
            "bootstrap_resamples": args.bootstrap_resamples,
            "noise_iqr_fraction_threshold": NOISE_IQR_FRACTION,
            "timer": "time.perf_counter (Python) / system_clock (Fortran)",
            "gfortran_flags": " ".join(flags) if gfortran else None,
        },
        "grid": {
            "synthetic": args.gridfile is None,
            "path": str(gridfile),
            "blocks": nb,
            "block_shapes": [list(shape) for shape in blocks],
            "total_points": total_points,
            "file_size_bytes": file_size,
            "on_disk_precision": "f4",
        },
        "readers": readers,
        "ratios": ratios,
        "cache_check": cache_check,
        "checksums": checksums,
        "checksum_verified": verified,
        "noisy_readers": sorted(k for k, v in readers.items() if v["noisy"]),
    }

    size_tag = f"{round(total_points / 1e6)}Mpts"
    stem = f"{label}_{size_tag}_{args.cache_state}"
    json_path = outdir / f"{stem}.json"
    csv_path = outdir / f"{stem}.csv"
    write_json(json_path, record)
    write_csv(csv_path, record)

    if not args.keep_grid:
        try:
            gridfile.unlink(missing_ok=True)
        except OSError:
            pass

    print_report(record)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    print("Pool this with other machines' results: python aggregate_results.py")


def print_report(record):
    print("\n" + "=" * 78)
    print(f"machine={record['machine_label']}  "
          f"grid={record['grid']['total_points']:,} points "
          f"({record['grid']['file_size_bytes']/1e6:.1f} MB)  "
          f"cache={record['measurement']['cache_state']}"
          + ("" if record["measurement"]["cache_state_verified"] else " (UNVERIFIED)"))
    print("=" * 78)
    for group, title in ((GROUP_IO_FLOOR, "I/O floor (diagnostic; NOT a fair "
                                          "language comparison)"),
                         (GROUP_F32, "Matched work, single precision, "
                                     "C-contiguous blocks"),
                         (GROUP_F64, "Matched work, double precision, "
                                     "padded array + per-block bounds")):
        rows = [(k, v) for k, v in record["readers"].items() if v["group"] == group]
        if not rows:
            continue
        print(f"\n{title}")
        print(f"  {'reader':<52s} {'n':>3s} {'median':>9s} {'IQR':>9s} {'min':>9s}")
        for _, stats in rows:
            flag = "  <- NOISY" if stats["noisy"] else ""
            print(f"  {stats['label']:<52s} {stats['n']:>3d} "
                  f"{ms(stats['median_s']):>7.2f}ms {ms(stats['iqr_s']):>7.2f}ms "
                  f"{ms(stats['min_s']):>7.2f}ms{flag}")
    if record["ratios"]:
        print("\nRatios (95% percentile bootstrap on the ratio of medians)")
        for key, ratio in record["ratios"].items():
            print(f"  {ratio['description']}")
            print(f"    ratio {ratio['point']:.2f}  "
                  f"95% CI [{ratio['ci95_low']:.2f}, {ratio['ci95_high']:.2f}]  "
                  f"-> {ratio_phrase(ratio['point'], ratio.get('subject', 'lptlib'))}")
    check = record.get("cache_check")
    if check:
        print("\nPage-cache residency check (whole-file read throughput, "
              "from the I/O floor)")
        print(f"  over the whole run      : "
              f"{check['median_bytes_per_s']/1e9:.2f} GB/s")
        print(f"  first third / last third: "
              f"{check['early_bytes_per_s']/1e9:.2f} GB/s -> "
              f"{check['late_bytes_per_s']/1e9:.2f} GB/s "
              f"({check['retention_fraction']*100:.0f}% retained)")
        if not check["throughput_consistent_with_warm_cache"]:
            print("  CACHE WARNING: the I/O floor ran at device throughput, not "
                  "page-cache\n                 throughput. This run is not warm "
                  "whatever it declares.")
        if (record["measurement"]["cache_state"] == "cold"
                and not check.get("throughput_consistent_with_cold_cache", True)):
            print("  CACHE WARNING: declared cold, but the I/O floor ran at "
                  "page-cache speed.\n                 The reads were not cold. "
                  "In a virtual machine, dropping the guest\n                 "
                  "page cache leaves the hypervisor's host-side cache intact, so "
                  "the\n                 file still comes from RAM. Treat this "
                  "bucket as partially warm.")
        if record["measurement"]["cache_state"] == "warm" \
                and not check["cache_retained_through_run"]:
            print("  CACHE WARNING: the grid file was evicted from the page cache "
                  "part way\n                 through the run, so later "
                  "repetitions read from the device.\n"
                  "                 This run mixes warm and cold reads. Use a "
                  "machine with enough\n                 RAM to hold the grid "
                  "file alongside the readers' working set.")
    if record["noisy_readers"]:
        print("\n  NOISE WARNING: interquartile range exceeded "
              f"{int(NOISE_IQR_FRACTION*100)}% of the median for: "
              + ", ".join(record["noisy_readers"]))
        print("  Treat this run as indicative only, or re-run on an idle machine.")
    print(f"\n  checksum verified across all readers : {record['checksum_verified']}")


def write_json(path, record):
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2, default=str)


CSV_FIELDS = ["machine_label", "cache_state", "cache_state_verified", "total_points",
              "file_size_bytes", "reader", "label", "group", "implementation", "n",
              "median_s", "q1_s", "q3_s", "iqr_s", "mad_s", "min_s", "max_s",
              "rel_iqr", "noisy"]


def write_csv(path, record):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for key, stats in record["readers"].items():
            row = {field: stats.get(field) for field in CSV_FIELDS}
            row.update({
                "machine_label": record["machine_label"],
                "cache_state": record["measurement"]["cache_state"],
                "cache_state_verified": record["measurement"]["cache_state_verified"],
                "total_points": record["grid"]["total_points"],
                "file_size_bytes": record["grid"]["file_size_bytes"],
                "reader": key,
            })
            writer.writerow(row)


if __name__ == "__main__":
    main()
