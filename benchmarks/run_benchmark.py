"""Reproducible PLOT3D I/O benchmark for lptlib.

The benchmark reads the same multi-block PLOT3D grid file with several readers
and times the read-and-reconstruct work. Comparisons are always made at matched
precision, because the PLOT3D file is single precision on disk while
``lptlib.io.GridIO.read_grid`` stores the grid as float64 for downstream use.

Single-precision group (the read technique, all produce float32 block arrays):

1. lptlib strided read  one buffered ``numpy.fromfile`` call plus a strided
   Fortran-order reshape, the read path used by ``GridIO.read_grid``.
2. a naive pure-Python reader (``naive_reader.read_grid_naive``) that does the
   same single buffered read, then reconstructs with nested Python loops. It
   isolates the striding speedup, because the file I/O is identical.
3. a compiled Fortran reader (``plot3d_read_fortran.f90``, precision 4).

Double-precision group (matches what the library actually returns):

4. the full public ``GridIO.read_grid``, which builds a float64 padded array and
   computes per-block coordinate bounds.
5. the same compiled Fortran reader at precision 8, which upcasts each block to
   float64 after reading, so it is compared against read_grid at float64.

All readers are checksum-verified to reconstruct identical data. Timings use
multiple repetitions after a warm-up read, and the machine, Python, NumPy, and
gfortran versions are recorded.

Run with a single command:

    python run_benchmark.py            # ~2.0M-point grid
    python run_benchmark.py --large    # ~28M-point grid
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

# ~2.0M points, four uneven blocks.
DEFAULT_BLOCKS = [(100, 90, 60), (110, 80, 55), (95, 85, 70), (120, 75, 50)]
# ~28M points, four uneven blocks.
LARGE_BLOCKS = [(400, 320, 60), (360, 300, 64), (340, 310, 66), (380, 300, 58)]


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
    ``numpy.fromfile`` for the whole float block, then a Fortran-order reshape of
    each block's slice, materialized to a contiguous single-precision array so
    the comparison against the Fortran reader is like-for-like. It does not
    compute the grid-metric bounds or upcast to float64, so it measures the read
    technique on the same single-precision task as the naive and Fortran readers.
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
                np.ascontiguousarray(
                    buf[start:end].reshape(
                        (int(ni[_i]), int(nj[_i]), int(nk[_i]), 3), order="F"
                    )
                )
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
    """Time the full public GridIO.read_grid (float64 grd + metric bounds)."""
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


def compile_fortran(gfortran):
    """Compile the Fortran reader once and return (exe_path, version, cmd)."""
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
    version = subprocess.run(
        gfortran + ["--version"], capture_output=True, text=True
    ).stdout.splitlines()[0]
    return exe, version, " ".join(compile_cmd)


def time_fortran(exe, gridfile, reps, precision):
    """Run the compiled Fortran reader. Returns (times, checksum)."""
    run_cmd = [str(exe), str(gridfile), str(reps), str(precision)]
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
    return times, checksum


def summarize(times):
    return {
        "n": len(times),
        "mean_s": statistics.mean(times),
        "min_s": min(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def ratio_phrase(ratio):
    """Describe strided-vs-other ratio (other_time / strided_time)."""
    if ratio >= 1.0:
        return f"{ratio:.2f}x faster"
    return f"{1.0 / ratio:.2f}x slower (ratio {ratio:.2f})"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=15,
                        help="repetitions for the fast readers")
    parser.add_argument("--naive-reps", type=int, default=7,
                        help="repetitions for the slow naive reader")
    parser.add_argument("--large", action="store_true",
                        help="use a ~28M-point grid instead of the ~2M default")
    parser.add_argument("--outdir", default=str(HERE / "results"),
                        help="directory for CSV, markdown, and figure outputs")
    parser.add_argument("--gridfile", default=None,
                        help="path to the grid file (generated if missing)")
    parser.add_argument("--keep-grid", action="store_true",
                        help="do not delete the generated grid file at the end")
    args = parser.parse_args()

    blocks_spec = LARGE_BLOCKS if args.large else DEFAULT_BLOCKS
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gridfile = Path(args.gridfile) if args.gridfile else (outdir / "synthetic_grid.x")
    gridfile.parent.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic multi-block PLOT3D grid ...")
    nb, blocks, total_points, gen_checksum = generate_grid.write_grid(
        str(gridfile), blocks=blocks_spec
    )
    file_size = gridfile.stat().st_size
    print(f"  blocks={nb} points={total_points:,} size={file_size/1e6:.1f} MB")

    # Warm the OS page cache so timings reflect reader work, not first-touch I/O.
    with open(gridfile, "rb") as f:
        while f.read(1 << 20):
            pass

    print(f"Timing lptlib strided read, float32 ({args.reps} reps) ...")
    strided_times, strided_checksum = time_reader(read_grid_strided, gridfile, args.reps)

    print(f"Timing naive Python reader, float32 ({args.naive_reps} reps) ...")
    naive_times, naive_checksum = time_reader(read_grid_naive, gridfile, args.naive_reps)

    print(f"Timing full GridIO.read_grid, float64 ({args.reps} reps) ...")
    full_times, full_checksum = time_lptlib_full(gridfile, args.reps)

    gfortran = find_gfortran()
    fortran4 = fortran8 = None
    f4_checksum = f8_checksum = None
    fortran_info = {}
    if gfortran:
        print(f"Compiling and timing Fortran reader ({args.reps} reps, f4 and f8) ...")
        try:
            exe, version, cmd = compile_fortran(gfortran)
            fortran4, f4_checksum = time_fortran(exe, gridfile, args.reps, 4)
            fortran8, f8_checksum = time_fortran(exe, gridfile, args.reps, 8)
            fortran_info = {"gfortran": version, "compile_cmd": cmd}
        except RuntimeError as exc:
            print("  Fortran baseline could not be run:", exc)
    else:
        print("  gfortran not found; skipping Fortran baseline.")

    # Verify all readers agree on the reconstructed data.
    tol = abs(gen_checksum) * 1e-4 + 1.0
    checks = {
        "generator": gen_checksum,
        "lptlib_strided_f4": strided_checksum,
        "naive_f4": naive_checksum,
        "lptlib_full_f8": full_checksum,
    }
    if f4_checksum is not None:
        checks["fortran_f4"] = f4_checksum
    if f8_checksum is not None:
        checks["fortran_f8"] = f8_checksum
    verified = all(abs(v - gen_checksum) <= tol for v in checks.values())

    strided_stats = summarize(strided_times)
    naive_stats = summarize(naive_times)
    full_stats = summarize(full_times)
    f4_stats = summarize(fortran4) if fortran4 else None
    f8_stats = summarize(fortran8) if fortran8 else None

    speedup_vs_naive = naive_stats["mean_s"] / strided_stats["mean_s"]
    ratio_strided_vs_f4 = (f4_stats["mean_s"] / strided_stats["mean_s"]) if f4_stats else None
    ratio_full_vs_f8 = (f8_stats["mean_s"] / full_stats["mean_s"]) if f8_stats else None

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

    single_rows = [
        ("lptlib strided read (float32)", strided_stats),
        ("naive Python nested-loop (float32)", naive_stats),
    ]
    if f4_stats:
        single_rows.append(("Fortran stream read (float32, gfortran -O2)", f4_stats))
    double_rows = [
        ("lptlib GridIO.read_grid (float64, incl. metric bounds)", full_stats),
    ]
    if f8_stats:
        double_rows.append(("Fortran stream read (float64, gfortran -O2)", f8_stats))

    results = {
        "env": env,
        "single_rows": single_rows,
        "double_rows": double_rows,
        "speedup_vs_naive": speedup_vs_naive,
        "ratio_strided_vs_f4": ratio_strided_vs_f4,
        "ratio_full_vs_f8": ratio_full_vs_f8,
        "nb": nb, "total_points": total_points, "file_size": file_size,
        "blocks": blocks, "args": args,
    }

    write_csv(outdir / "benchmark_results.csv", single_rows + double_rows)
    write_markdown(outdir / "benchmark_results.md", results)
    write_json(outdir / "benchmark_results.json", results)
    try:
        make_figure(single_rows, double_rows, outdir / "benchmark_bar.png",
                    speedup_vs_naive, ratio_strided_vs_f4, ratio_full_vs_f8)
    except Exception as exc:  # pragma: no cover
        print("  Could not render figure:", exc)

    if not args.keep_grid:
        try:
            gridfile.unlink(missing_ok=True)
        except OSError:
            pass

    print("\nSingle precision (read technique, matched float32):")
    for name, s in single_rows:
        print(f"  {name:44s} mean={s['mean_s']*1000:9.2f} ms  min={s['min_s']*1000:9.2f} ms")
    print("\nDouble precision (library's full read, matched float64):")
    for name, s in double_rows:
        print(f"  {name:44s} mean={s['mean_s']*1000:9.2f} ms  min={s['min_s']*1000:9.2f} ms")
    print(f"\n  strided vs naive Python (f4) : {speedup_vs_naive:.1f}x faster")
    if ratio_strided_vs_f4 is not None:
        print(f"  strided vs Fortran (f4)      : {ratio_phrase(ratio_strided_vs_f4)}")
    if ratio_full_vs_f8 is not None:
        print(f"  read_grid vs Fortran (f8)    : {ratio_phrase(ratio_full_vs_f8)}")
    print(f"\n  checksum verified            : {verified}")


def write_csv(path, rows):
    with open(path, "w") as f:
        f.write("reader,reps,mean_s,min_s,stdev_s\n")
        for name, s in rows:
            f.write(f"{name},{s['n']},{s['mean_s']:.6f},{s['min_s']:.6f},{s['stdev_s']:.6f}\n")


def _table(f, rows):
    f.write("| Reader | Reps | Mean (s) | Min (s) | Stdev (s) |\n")
    f.write("|---|---|---|---|---|\n")
    for name, s in rows:
        f.write(f"| {name} | {s['n']} | {s['mean_s']:.4f} | "
                f"{s['min_s']:.4f} | {s['stdev_s']:.4f} |\n")


def write_markdown(path, r):
    env = r["env"]
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
        f.write(f"- Grid: {r['nb']} blocks, {r['total_points']:,} points, "
                f"{r['file_size']/1e6:.1f} MB (single precision on disk, synthetic)\n")
        f.write(f"- Block shapes: {r['blocks']}\n")
        f.write(f"- Repetitions: {r['args'].reps} (naive reader {r['args'].naive_reps}), warm cache\n")
        f.write(f"- Checksum verified across readers: {env['checksum_verified']}\n\n")
        f.write("## Single precision (read technique)\n\n")
        f.write("The PLOT3D file is single precision. All three readers do one "
                "buffered read and reconstruct the per-block float32 coordinate "
                "arrays, so the timing difference isolates the reconstruction "
                "strategy at matched precision.\n\n")
        _table(f, r["single_rows"])
        f.write("\n")
        f.write(f"- lptlib strided vs naive Python reader: **{r['speedup_vs_naive']:.1f}x faster**\n")
        if r["ratio_strided_vs_f4"] is not None:
            f.write(f"- lptlib strided vs compiled Fortran reader: **{ratio_phrase(r['ratio_strided_vs_f4'])}**\n")
        f.write("\n## Double precision (library's full read)\n\n")
        f.write("`GridIO.read_grid` returns a float64 grd array and per-block "
                "coordinate bounds. For a like-for-like comparison the Fortran "
                "reader upcasts each block to float64 after reading. This is the "
                "cost the library actually pays.\n\n")
        _table(f, r["double_rows"])
        if r["ratio_full_vs_f8"] is not None:
            f.write(f"\n- lptlib read_grid vs compiled Fortran, both float64: "
                    f"**{ratio_phrase(r['ratio_full_vs_f8'])}**\n")


def write_json(path, r):
    record = {
        "environment": r["env"],
        "single_precision_timings": {name: s for name, s in r["single_rows"]},
        "double_precision_timings": {name: s for name, s in r["double_rows"]},
        "speedup_strided_vs_naive": r["speedup_vs_naive"],
        "ratio_strided_vs_fortran_f4": r["ratio_strided_vs_f4"],
        "ratio_read_grid_vs_fortran_f8": r["ratio_full_vs_f8"],
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)


def make_figure(single_rows, double_rows, path, sp_naive, ratio_f4, ratio_f8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = {
        "lptlib strided read (float32)": "lptlib\nstrided f4",
        "naive Python nested-loop (float32)": "naive\nPython f4",
        "Fortran stream read (float32, gfortran -O2)": "Fortran\nf4",
        "lptlib GridIO.read_grid (float64, incl. metric bounds)": "read_grid\nf8",
        "Fortran stream read (float64, gfortran -O2)": "Fortran\nf8",
    }
    rows = single_rows + double_rows
    names = [short.get(n, n) for n, _ in rows]
    means = [s["mean_s"] for _, s in rows]
    stdevs = [s["stdev_s"] for _, s in rows]
    n_single = len(single_rows)
    colors = (["#2c7fb8", "#d95f0e", "#2ca25f"][:n_single]
              + ["#6a51a3", "#41ab5d"][:len(double_rows)])

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(range(len(names)), means, yerr=stdevs, capsize=4,
                  color=colors[:len(names)])
    ax.set_yscale("log")
    ax.set_ylabel("Mean read time (s, log scale)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(min(means) / 2.0, max(means) * 2.4)
    ax.axvline(n_single - 0.5, color="#999", linestyle="--", linewidth=0.8)
    ax.text(0.02, 0.96, "single precision", transform=ax.transAxes, fontsize=8, color="#555", va="top")
    ax.text(0.98, 0.96, "double precision", transform=ax.transAxes, fontsize=8, color="#555", va="top", ha="right")
    for bar, m, s in zip(bars, means, stdevs):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s, f"{m*1000:.1f} ms",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("PLOT3D multi-block grid: read-and-reconstruct time", pad=10)
    cap = f"strided is {sp_naive:.0f}x faster than naive Python (f4)"
    if ratio_f4 is not None:
        cap += f", {ratio_phrase(ratio_f4).split(' (')[0]} than Fortran (f4)"
    fig.subplots_adjust(bottom=0.20)
    fig.text(0.5, 0.04, cap, ha="center", va="center", fontsize=9, color="#444")
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
