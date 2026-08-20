"""Reproducible PLOT3D I/O benchmark for lptlib.

The benchmark compares readers of the same multi-block PLOT3D grid file on one
well-defined task: read the file and reconstruct the per-block coordinate
arrays in memory. Three readers perform that task:

1. lptlib strided read  a single buffered ``numpy.fromfile`` call plus a
   strided Fortran-order reshape. This is the read path used by
   ``lptlib.io.GridIO.read_grid`` and the technique the JOSS paper describes.
2. a naive pure-Python reader (``naive_reader.read_grid_naive``) that does the
   same single buffered read, then reconstructs the arrays with nested loops
   over every grid point. It isolates the striding speedup, because the file
   I/O is identical and only the reconstruction differs.
3. a compiled Fortran reader (``plot3d_read_fortran.f90``, gfortran) that reads
   the same file with stream access. This is the "vs Fortran" comparison.

All three read the identical bytes with one buffered read, so the timing
difference reflects reconstruction strategy, not disk work. For transparency the
benchmark also times the full public ``GridIO.read_grid``, which additionally
allocates a double-precision padded array and computes per-block coordinate
bounds; those metrics are reported separately from the read technique.

All readers are verified against a common checksum. Timings use multiple
repetitions after a warm-up read, and the machine, Python, NumPy, and gfortran
versions are recorded.

Run with a single command:

    python run_benchmark.py
"""

import argparse
import json
import os
import platform
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

import generate_grid  # noqa: E402
from naive_reader import read_grid_naive  # noqa: E402


def load_gridio():
    """Return lptlib's GridIO class.

    Prefer the public import ``from lptlib.io import GridIO``. If the lptlib.io
    package pulls in optional dependencies that are not installed (for example
    mpi4py via DataIO), fall back to importing the reader directly from the
    plot3dio submodule, which is where GridIO is defined. The reader under test
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


def read_grid_strided(filename, data_type="f4"):
    """Read a multi-block PLOT3D grid with the lptlib strided technique.

    This reproduces the read path of ``GridIO.read_grid``: one buffered
    ``numpy.fromfile`` for the whole float block, then a Fortran-order reshape
    of each block's slice. It returns a list of (ni, nj, nk, 3) arrays. It does
    not compute the grid-metric bounds, so that this measures the reader
    technique on the same task as the naive and Fortran readers.
    """
    with open(filename, "r") as grid:
        nb = np.fromfile(grid, dtype="i4", count=1)[0]
        _temp = np.fromfile(grid, dtype="i4", count=3 * nb)
        ni, nj, nk = _temp[0::3], _temp[1::3], _temp[2::3]
        _nt = ni * nj * nk * 3
        buf = np.fromfile(grid, dtype=data_type, count=int(sum(_nt)))
        blocks = []
        for _i in range(nb):
            start = int(sum(_nt[0:_i]))
            end = start + int(_nt[_i])
            blocks.append(
                buf[start:end].reshape(
                    (int(ni[_i]), int(nj[_i]), int(nk[_i]), 3), order="F"
                ).copy()
            )
    return blocks


def _checksum_blocks(blocks):
    return float(sum(b.sum(dtype=np.float64) for b in blocks))


def time_reader(func, gridfile, reps):
    """Time a reader that returns a list of block arrays. Returns (times, checksum)."""
    times = []
    checksum = None
    for _ in range(reps):
        t0 = time.perf_counter()
        blocks = func(str(gridfile), data_type="f4")
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if checksum is None:
            checksum = _checksum_blocks(blocks)
    return times, checksum


def time_lptlib_full(gridfile, reps):
    """Time the full public GridIO.read_grid (read + reshape + metric bounds)."""
    GridIO = load_gridio()
    times = []
    checksum = None
    for _ in range(reps):
        t0 = time.perf_counter()
        grid = GridIO(str(gridfile))
        grid.read_grid(data_type="f4")
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if checksum is None:
            checksum = float(np.float64(grid.grd.sum(dtype=np.float64)))
    return times, checksum


def find_gfortran():
    """Return a runnable gfortran command, or None.

    Honors the GFORTRAN environment variable so a reviewer can point at a
    specific compiler. Falls back to gfortran / gfortran-11 on PATH.
    """
    env = os.environ.get("GFORTRAN")
    if env:
        return env.split()
    for name in ("gfortran", "gfortran-11"):
        if shutil.which(name):
            return [name]
    return None


def build_and_time_fortran(gridfile, reps, gfortran):
    """Compile and run the Fortran reader. Returns (times, checksum, info)."""
    src = HERE / "plot3d_read_fortran.f90"
    exe = HERE / "plot3d_read_fortran"
    extra = os.environ.get("GFORTRAN_FLAGS", "").split()
    compile_cmd = gfortran + ["-O2", str(src), "-o", str(exe)] + extra

    proc = subprocess.run(compile_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "gfortran compilation failed:\n"
            + " ".join(compile_cmd) + "\n" + proc.stdout + proc.stderr
        )

    run_cmd = [str(exe), str(gridfile), str(reps)]
    proc = subprocess.run(run_cmd, capture_output=True, text=True, env=dict(os.environ))
    if proc.returncode != 0:
        raise RuntimeError("Fortran reader failed:\n" + proc.stdout + proc.stderr)

    times = []
    checksum = None
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "rep":
            times.append(float(parts[2]))
        elif len(parts) == 2 and parts[0] == "checksum":
            checksum = float(parts[1])

    version = subprocess.run(
        gfortran + ["--version"], capture_output=True, text=True
    ).stdout.splitlines()[0]
    info = {"gfortran": version, "compile_cmd": " ".join(compile_cmd)}
    return times, checksum, info


def summarize(times):
    return {
        "n": len(times),
        "mean_s": statistics.mean(times),
        "min_s": min(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=15,
                        help="repetitions for the fast readers (strided, Fortran, full)")
    parser.add_argument("--naive-reps", type=int, default=7,
                        help="repetitions for the slow naive reader")
    parser.add_argument("--outdir", default=str(HERE / "results"),
                        help="directory for CSV, markdown, and figure outputs")
    parser.add_argument("--gridfile", default=str(HERE / "results" / "synthetic_grid.x"),
                        help="path to the grid file (generated if missing)")
    parser.add_argument("--keep-grid", action="store_true",
                        help="do not delete the generated grid file at the end")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gridfile = Path(args.gridfile)
    gridfile.parent.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic multi-block PLOT3D grid ...")
    nb, blocks, total_points, gen_checksum = generate_grid.write_grid(str(gridfile))
    file_size = gridfile.stat().st_size
    print(f"  blocks={nb} points={total_points:,} size={file_size/1e6:.1f} MB")

    # Warm the OS page cache so timings reflect reader work, not first-touch I/O.
    with open(gridfile, "rb") as f:
        while f.read(1 << 20):
            pass

    print(f"Timing lptlib strided read ({args.reps} reps) ...")
    strided_times, strided_checksum = time_reader(read_grid_strided, gridfile, args.reps)

    print(f"Timing naive Python reader ({args.naive_reps} reps) ...")
    naive_times, naive_checksum = time_reader(read_grid_naive, gridfile, args.naive_reps)

    print(f"Timing full GridIO.read_grid ({args.reps} reps) ...")
    full_times, full_checksum = time_lptlib_full(gridfile, args.reps)

    gfortran = find_gfortran()
    fortran_times = None
    fortran_checksum = None
    fortran_info = {}
    if gfortran:
        print(f"Compiling and timing Fortran reader ({args.reps} reps) ...")
        try:
            fortran_times, fortran_checksum, fortran_info = build_and_time_fortran(
                gridfile, args.reps, gfortran
            )
        except RuntimeError as exc:
            print("  Fortran baseline could not be run:", exc)
    else:
        print("  gfortran not found; skipping Fortran baseline.")

    # Verify all readers agree on the reconstructed data.
    tol = abs(gen_checksum) * 1e-4 + 1.0
    checks = {
        "generator": gen_checksum,
        "lptlib_strided": strided_checksum,
        "naive": naive_checksum,
        "lptlib_full": full_checksum,
    }
    if fortran_checksum is not None:
        checks["fortran"] = fortran_checksum
    verified = all(abs(v - gen_checksum) <= tol for v in checks.values())

    strided_stats = summarize(strided_times)
    naive_stats = summarize(naive_times)
    full_stats = summarize(full_times)
    fortran_stats = summarize(fortran_times) if fortran_times else None

    speedup_vs_naive = naive_stats["mean_s"] / strided_stats["mean_s"]
    speedup_vs_fortran = (
        fortran_stats["mean_s"] / strided_stats["mean_s"] if fortran_stats else None
    )

    env = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "grid_blocks": nb,
        "grid_block_shapes": blocks,
        "grid_total_points": total_points,
        "grid_file_size_bytes": file_size,
        "reps": args.reps,
        "naive_reps": args.naive_reps,
        "checksums": checks,
        "checksum_verified": verified,
    }
    env.update(fortran_info)

    # Rows for the read-and-reconstruct task (fair comparison).
    task_rows = [
        ("lptlib strided read (fromfile + F-order reshape)", strided_stats),
        ("naive Python nested-loop reconstruct", naive_stats),
    ]
    if fortran_stats:
        task_rows.append(("Fortran stream read (gfortran -O2)", fortran_stats))
    # Transparency row for the full public API.
    extra_rows = [("lptlib GridIO.read_grid (full: read + reshape + metric bounds)", full_stats)]

    write_csv(outdir / "benchmark_results.csv", task_rows + extra_rows)
    write_markdown(outdir / "benchmark_results.md", env, task_rows, extra_rows,
                   speedup_vs_naive, speedup_vs_fortran, nb, total_points,
                   file_size, blocks, args)
    write_json(outdir / "benchmark_results.json", env, task_rows, extra_rows,
               speedup_vs_naive, speedup_vs_fortran)

    try:
        make_figure(task_rows, outdir / "benchmark_bar.png",
                    speedup_vs_naive, speedup_vs_fortran)
    except Exception as exc:  # pragma: no cover
        print("  Could not render figure:", exc)

    if not args.keep_grid:
        try:
            gridfile.unlink(missing_ok=True)
        except OSError:
            # Some mounts (for example OneDrive) disallow unlink; leave the file.
            pass

    print("\nRead-and-reconstruct task (identical I/O, differing reconstruction):")
    for name, s in task_rows:
        print(f"  {name:52s} mean={s['mean_s']*1000:8.2f} ms  min={s['min_s']*1000:8.2f} ms")
    print("\nFor reference (full public API):")
    for name, s in extra_rows:
        print(f"  {name:52s} mean={s['mean_s']*1000:8.2f} ms")
    print(f"\n  lptlib strided vs naive Python : {speedup_vs_naive:.1f}x faster")
    if speedup_vs_fortran is not None:
        if speedup_vs_fortran >= 1.0:
            print(f"  lptlib strided vs Fortran      : {speedup_vs_fortran:.2f}x faster")
        else:
            print(f"  lptlib strided vs Fortran      : {1.0/speedup_vs_fortran:.2f}x slower "
                  f"(ratio {speedup_vs_fortran:.2f})")
    print(f"\n  checksum verified              : {verified}")


def write_csv(path, rows):
    with open(path, "w") as f:
        f.write("reader,reps,mean_s,min_s,stdev_s\n")
        for name, s in rows:
            f.write(f"{name},{s['n']},{s['mean_s']:.6f},{s['min_s']:.6f},{s['stdev_s']:.6f}\n")


def write_markdown(path, env, task_rows, extra_rows, sp_naive, sp_fortran,
                   nb, total_points, file_size, blocks, args):
    with open(path, "w") as f:
        f.write("# PLOT3D I/O benchmark results\n\n")
        f.write(f"Generated {env['timestamp']}\n\n")
        f.write("## Environment\n\n")
        f.write(f"- Platform: {env['platform']}\n")
        f.write(f"- Processor: {env['processor']}\n")
        f.write(f"- Python: {env['python']}\n")
        f.write(f"- NumPy: {env['numpy']}\n")
        if "gfortran" in env:
            f.write(f"- Fortran: {env['gfortran']}\n")
        f.write(f"- Grid: {nb} blocks, {total_points:,} points, "
                f"{file_size/1e6:.1f} MB (single precision, synthetic)\n")
        f.write(f"- Block shapes: {blocks}\n")
        f.write(f"- Repetitions: {args.reps} (naive reader {args.naive_reps}), warm cache\n")
        f.write(f"- Checksum verified across readers: {env['checksum_verified']}\n\n")
        f.write("## Read-and-reconstruct task\n\n")
        f.write("All readers perform one buffered read of the file and reconstruct "
                "the per-block coordinate arrays. The I/O is identical, so the "
                "difference isolates the reconstruction strategy.\n\n")
        f.write("| Reader | Reps | Mean (s) | Min (s) | Stdev (s) |\n")
        f.write("|---|---|---|---|---|\n")
        for name, s in task_rows:
            f.write(f"| {name} | {s['n']} | {s['mean_s']:.4f} | "
                    f"{s['min_s']:.4f} | {s['stdev_s']:.4f} |\n")
        f.write("\n## Speedups (mean time)\n\n")
        f.write(f"- lptlib strided reader vs naive Python reader: "
                f"**{sp_naive:.1f}x faster**\n")
        if sp_fortran is not None:
            if sp_fortran >= 1.0:
                f.write(f"- lptlib strided reader vs compiled Fortran reader: "
                        f"**{sp_fortran:.2f}x faster**\n")
            else:
                f.write(f"- lptlib strided reader vs compiled Fortran reader: "
                        f"**{1.0/sp_fortran:.2f}x slower** (ratio {sp_fortran:.2f}, "
                        f"i.e. the same order of magnitude)\n")
        f.write("\n## Full public API (for reference)\n\n")
        f.write("`GridIO.read_grid` additionally allocates a double-precision "
                "padded array and computes per-block coordinate bounds, which is "
                "grid-metric bookkeeping beyond the read technique above.\n\n")
        f.write("| Reader | Reps | Mean (s) | Min (s) | Stdev (s) |\n")
        f.write("|---|---|---|---|---|\n")
        for name, s in extra_rows:
            f.write(f"| {name} | {s['n']} | {s['mean_s']:.4f} | "
                    f"{s['min_s']:.4f} | {s['stdev_s']:.4f} |\n")


def write_json(path, env, task_rows, extra_rows, sp_naive, sp_fortran):
    record = {
        "environment": env,
        "task_timings": {name: s for name, s in task_rows},
        "full_api_timings": {name: s for name, s in extra_rows},
        "speedup_vs_naive": sp_naive,
        "speedup_vs_fortran": sp_fortran,
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)


def make_figure(rows, path, sp_naive, sp_fortran):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = {
        "lptlib strided read (fromfile + F-order reshape)": "lptlib\nstrided",
        "naive Python nested-loop reconstruct": "naive\nPython",
        "Fortran stream read (gfortran -O2)": "Fortran\n(gfortran)",
    }
    names = [short.get(r[0], r[0]) for r in rows]
    means = [r[1]["mean_s"] for r in rows]
    stdevs = [r[1]["stdev_s"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    colors = ["#2c7fb8", "#d95f0e", "#2ca25f"][: len(names)]
    bars = ax.bar(range(len(names)), means, yerr=stdevs, capsize=4, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Mean read time (s, log scale)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    # Headroom above the tallest bar so its value label does not collide.
    ax.set_ylim(min(means) / 2.0, max(means) * 2.2)
    for bar, m, s in zip(bars, means, stdevs):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s, f"{m*1000:.1f} ms",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("PLOT3D multi-block grid: read-and-reconstruct time", pad=10)
    caption = f"lptlib strided read is {sp_naive:.0f}x faster than naive Python"
    if sp_fortran is not None and sp_fortran >= 1.0:
        caption += f" and {sp_fortran:.2f}x faster than compiled Fortran"
    elif sp_fortran is not None:
        caption += f"; on par with compiled Fortran (ratio {sp_fortran:.2f})"
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.045, caption, ha="center", va="center", fontsize=9, color="#444")
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
