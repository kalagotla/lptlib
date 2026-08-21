"""Pool PLOT3D benchmark results from several machines into one summary.

``run_benchmark.py`` writes one JSON file per (machine, grid size, cache state).
This script reads every such file in ``results/`` and emits a single markdown
summary, ``results/SUMMARY.md``, which is the document the JOSS paper cites.

Why pooling matters here. The absolute read times in this benchmark move by
more than an order of magnitude between machines, and they move for reasons
that have nothing to do with the readers: page-cache state, memory bandwidth,
and whether the filesystem can cache the file at all. A single machine's
absolute milliseconds are therefore not a defensible claim. The ratio between
two readers doing matched work is far more stable, so the summary leads with
ratios and reports the across-machine range for everything.

For each reader the summary reports the median of the per-machine medians and
the min-to-max range across machines. For each ratio it reports the same, and
additionally the widest per-machine 95 percent bootstrap interval, so a reader
can see both the between-machine spread and the within-machine uncertainty.

Results are pooled only within a (grid size, cache state) bucket. A warm-cache
result and a cold-cache result measure different things and are never mixed.

With a single machine reporting, the range collapses to that machine's value
and the summary says so explicitly rather than implying a spread it does not
have.

Usage:
    python aggregate_results.py
    python aggregate_results.py --results-dir results --out results/SUMMARY.md
    python aggregate_results.py --no-figure

author: benchmark tooling for lptlib (Dilip Kalagotla)
"""

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Display order and headings for the reader groups.
GROUP_ORDER = [
    ("f32-contiguous",
     "Matched work, single precision (C-contiguous block arrays)",
     "Every reader in this group ends with the same bytes in the same layout: "
     "one C-contiguous float32 `(ni, nj, nk, 3)` array per block. The "
     "comparison is like-for-like."),
    ("f64-full",
     "Matched work, double precision (padded array plus per-block bounds)",
     "Every reader in this group produces the padded float64 "
     "`(nimax, njmax, nkmax, 3, nb)` array and the per-block coordinate "
     "bounds that `GridIO.read_grid` returns. This is what the library "
     "actually costs."),
    ("io-floor",
     "I/O floor (diagnostic only, NOT a language comparison)",
     "Both rows only stream bytes into a column-major array and reorder "
     "nothing. They exist to show how much of each real reader's time is file "
     "I/O rather than memory reordering. Do not quote this group as a "
     "Python-versus-Fortran result."),
]

READER_ORDER = [
    "lptlib_strided", "naive_python", "fortran_contig",
    "lptlib_read_grid", "fortran_full",
    "numpy_raw_read", "fortran_raw",
]

RATIO_ORDER = [
    "naive_python_over_lptlib_strided",
    "fortran_over_lptlib_strided",
    "fortran_over_lptlib_read_grid",
    "fortran_over_numpy_raw",
]


def load_records(results_dir):
    """Load every benchmark JSON in ``results_dir``, newest first per machine."""
    records = []
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            with open(path) as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skipping {path.name}: {exc}")
            continue
        if "readers" not in data or "machine_label" not in data:
            print(f"  skipping {path.name}: not a benchmark result file")
            continue
        data["_path"] = path
        records.append(data)
    return records


def bucket_key(record):
    points = record["grid"]["total_points"]
    return (round(points / 1e6), record["measurement"]["cache_state"])


def across(values):
    """Median, min and max of a list of per-machine values."""
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def fmt_ms(value):
    return f"{value * 1000:.1f}"


def fmt_range_ms(agg):
    if agg["n"] == 1:
        return "single machine"
    return f"{fmt_ms(agg['min'])} - {fmt_ms(agg['max'])}"


def fmt_range_ratio(agg):
    if agg["n"] == 1:
        return "single machine"
    return f"{agg['min']:.2f} - {agg['max']:.2f}"


def ratio_phrase(ratio, subject):
    if ratio >= 1.0:
        return f"{subject} {ratio:.2f}x faster"
    return f"{subject} {1.0 / ratio:.2f}x slower"


def summarize_bucket(records):
    """Aggregate one (grid size, cache state) bucket across machines."""
    readers = {}
    for record in records:
        for key, stats in record["readers"].items():
            entry = readers.setdefault(key, {"label": stats["label"],
                                             "group": stats["group"],
                                             "medians": [], "rel_iqrs": [],
                                             "machines": [], "noisy": []})
            entry["medians"].append(stats["median_s"])
            entry["rel_iqrs"].append(stats.get("rel_iqr") or 0.0)
            entry["machines"].append(record["machine_label"])
            entry["noisy"].append(bool(stats.get("noisy")))

    ratios = {}
    for record in records:
        for key, ratio in record.get("ratios", {}).items():
            entry = ratios.setdefault(key, {"description": ratio["description"],
                                            "subject": ratio.get("subject", "lptlib"),
                                            "group": ratio.get("group", ""),
                                            "points": [], "ci_low": [], "ci_high": [],
                                            "machines": []})
            entry["points"].append(ratio["point"])
            entry["ci_low"].append(ratio["ci95_low"])
            entry["ci_high"].append(ratio["ci95_high"])
            entry["machines"].append(record["machine_label"])
    return readers, ratios


def write_summary(path, buckets, records):
    machines = sorted({record["machine_label"] for record in records})
    lines = []
    add = lines.append

    add("# PLOT3D I/O benchmark: pooled summary")
    add("")
    add("Generated by `aggregate_results.py` from every result file in "
        "`benchmarks/results/`.")
    add("")
    add(f"**Machines reporting: {len(machines)}** "
        f"({', '.join(f'`{m}`' for m in machines)}). "
        f"Result files pooled: {len(records)}.")
    add("")
    if len(machines) == 1:
        add("> Only one machine has reported. Every across-machine range below "
            "collapses to that machine's single value, and the ranges are "
            "marked `single machine` rather than pretending to a spread. "
            "Absolute times on this benchmark move by more than an order of "
            "magnitude between machines, so treat the absolutes as indicative "
            "of this machine only. The ratios within a matched-work group are "
            "the transferable quantity. See `benchmarks/README.md` for how to "
            "contribute a second machine.")
        add("")
    add("Each reader's per-machine number is a **median** over repeated reads "
        "after discarded warmups; the spread within a machine is its "
        "interquartile range. The columns below pool those per-machine medians: "
        "`median across machines` is the median of them, `range across machines` "
        "is min to max. Ratios carry a per-machine 95 percent percentile "
        "bootstrap interval on the ratio of medians, and the widest of those is "
        "shown so within-machine uncertainty is visible next to "
        "between-machine spread.")
    add("")
    add("Buckets are never mixed: a warm-cache result and a cold-cache result "
        "measure different things, and so do two different grid sizes.")
    add("")

    for (points_m, cache_state), bucket_records in sorted(buckets.items()):
        readers, ratios = summarize_bucket(bucket_records)
        bucket_machines = sorted({r["machine_label"] for r in bucket_records})
        size = bucket_records[0]["grid"]["file_size_bytes"] / 1e6
        exact = bucket_records[0]["grid"]["total_points"]
        unverified = [r["machine_label"] for r in bucket_records
                      if not r["measurement"]["cache_state_verified"]]

        add("---")
        add("")
        add(f"## ~{points_m}M-point grid, {cache_state} cache")
        add("")
        add(f"{exact:,} points, {size:.1f} MB on disk (single precision), "
            f"{bucket_records[0]['grid']['blocks']} blocks. "
            f"Machines in this bucket: {len(bucket_machines)} "
            f"({', '.join(f'`{m}`' for m in bucket_machines)}).")
        add("")
        if unverified:
            add(f"> Cache state **unverified** on: "
                f"{', '.join(f'`{m}`' for m in unverified)}. "
                f"A cold-cache run is only meaningful if the operator dropped "
                f"the page cache; see `README.md`.")
            add("")

        throughputs = []
        for record in bucket_records:
            check = record.get("cache_check")
            if check:
                throughputs.append(
                    f"`{record['machine_label']}` "
                    f"{check['median_bytes_per_s']/1e9:.2f} GB/s")
        if throughputs:
            add(f"Whole-file read throughput measured during this run (the I/O "
                f"floor reader, the direct test of the declared cache state): "
                f"{', '.join(throughputs)}. Page-cache reads run at gigabytes per "
                f"second; device reads at hundreds of megabytes per second.")
            add("")

        cache_bad = []
        for record in bucket_records:
            check = record.get("cache_check")
            if check and not check.get("ok_for_declared_state", True):
                reasons = []
                if not check.get("throughput_consistent_with_warm_cache", True):
                    reasons.append(
                        f"I/O floor ran at "
                        f"{check['median_bytes_per_s']/1e9:.2f} GB/s, device rather "
                        f"than page-cache throughput")
                if not check.get("throughput_consistent_with_cold_cache", True) \
                        and record["measurement"]["cache_state"] == "cold":
                    reasons.append(
                        f"declared cold but the I/O floor ran at "
                        f"{check['median_bytes_per_s']/1e9:.2f} GB/s, which is "
                        f"page-cache speed, so the reads were not cold (in a VM "
                        f"the hypervisor's cache survives a guest drop_caches)")
                if not check.get("cache_retained_through_run", True) \
                        and record["measurement"]["cache_state"] == "warm":
                    reasons.append(
                        f"only {check['retention_fraction']*100:.0f}% of the "
                        f"whole-file read throughput survived the run, so the grid "
                        f"file was evicted part way through")
                cache_bad.append(f"`{record['machine_label']}` ("
                                 + "; ".join(reasons) + ")")
        if cache_bad:
            add(f"> Cache-residency check **failed** on: "
                f"{', '.join(cache_bad)}. The declared cache state does not match "
                f"what the machine actually did, so these numbers mix warm and "
                f"cold reads. Treat them as a lower bound on this machine's "
                f"performance, not as a clean warm-cache measurement.")
            add("")

        noisy = sorted({f"`{key}` on `{machine}`"
                        for key, entry in readers.items()
                        for machine, flag in zip(entry["machines"], entry["noisy"])
                        if flag})
        if noisy:
            add(f"> Dispersion warning: interquartile range exceeded 15 percent "
                f"of the median for {', '.join(noisy)}. Those numbers are "
                f"indicative rather than tight.")
            add("")

        add("### Reader times")
        add("")
        for group, title, blurb in GROUP_ORDER:
            group_keys = [k for k in READER_ORDER
                          if k in readers and readers[k]["group"] == group]
            group_keys += sorted(k for k, v in readers.items()
                                 if v["group"] == group and k not in READER_ORDER)
            if not group_keys:
                continue
            add(f"**{title}**")
            add("")
            add(blurb)
            add("")
            add("| Reader | Machines | Median across machines (ms) | "
                "Range across machines (ms) | Worst within-machine rel. IQR |")
            add("|---|---|---|---|---|")
            for key in group_keys:
                entry = readers[key]
                agg = across(entry["medians"])
                worst_iqr = max(entry["rel_iqrs"]) if entry["rel_iqrs"] else 0.0
                add(f"| {entry['label']} | {agg['n']} | {fmt_ms(agg['median'])} | "
                    f"{fmt_range_ms(agg)} | {worst_iqr*100:.1f}% |")
            add("")

        if ratios:
            add("### Ratios (the transferable result)")
            add("")
            add("Every ratio compares two readers doing **matched work**. "
                "A ratio above 1 means the lptlib/numpy side is faster.")
            add("")
            add("| Comparison | Machines | Median ratio | Range across machines | "
                "Widest per-machine 95% CI | Reads as |")
            add("|---|---|---|---|---|---|")
            keys = [k for k in RATIO_ORDER if k in ratios]
            keys += sorted(k for k in ratios if k not in RATIO_ORDER)
            for key in keys:
                entry = ratios[key]
                agg = across(entry["points"])
                ci_low = min(entry["ci_low"])
                ci_high = max(entry["ci_high"])
                add(f"| {entry['description']} | {agg['n']} | "
                    f"{agg['median']:.2f} | {fmt_range_ratio(agg)} | "
                    f"[{ci_low:.2f}, {ci_high:.2f}] | "
                    f"{ratio_phrase(agg['median'], entry['subject'])} |")
            add("")

    add("---")
    add("")
    add("## Machines")
    add("")
    add("| Label | OS / kernel | CPU | Cores | RAM (GiB) | Filesystem | Storage | "
        "Python | NumPy | gfortran | Flags |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    seen = set()
    for record in sorted(records, key=lambda r: r["machine_label"]):
        label = record["machine_label"]
        if label in seen:
            continue
        seen.add(label)
        machine = record["machine"]
        storage = machine.get("storage", {})
        add(f"| `{label}` | {machine.get('os')} {machine.get('os_release')} | "
            f"{machine.get('cpu_model')} | "
            f"{machine.get('cpu_logical_cores')} | "
            f"{machine.get('total_ram_gib')} | "
            f"{storage.get('fstype')} | {storage.get('storage_class')} | "
            f"{machine.get('python_version')} | {machine.get('numpy_version')} | "
            f"{(machine.get('gfortran_version') or 'not available').split('(')[0].strip()} | "
            f"`{record['measurement'].get('gfortran_flags') or 'n/a'}` |")
    add("")
    add("## Result files pooled")
    add("")
    for record in sorted(records, key=lambda r: (r["machine_label"],
                                                 r["grid"]["total_points"])):
        measurement = record["measurement"]
        add(f"- `{Path(record['_path']).name}` -- {record['timestamp_utc']}, "
            f"{measurement['reps']} reps (+{measurement['warmups_discarded']} "
            f"warmups discarded), naive reader {measurement['naive_reps']} reps, "
            f"{measurement['priming_reads']} priming reads, "
            f"cache `{measurement['cache_state']}`, "
            f"checksum verified: {record.get('checksum_verified')}")
    add("")

    Path(path).write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def make_figure(buckets, path):
    """Bar chart of the matched-work groups, one panel per bucket."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(buckets.items())
    fig, axes = plt.subplots(1, len(ordered), figsize=(6.2 * len(ordered), 4.8),
                             squeeze=False)
    colors = {"lptlib_strided": "#2c7fb8", "naive_python": "#d95f0e",
              "fortran_contig": "#2ca25f", "lptlib_read_grid": "#6a51a3",
              "fortran_full": "#41ab5d"}
    short = {"lptlib_strided": "lptlib\nstrided\nf32",
             "naive_python": "naive\nPython\nf32",
             "fortran_contig": "Fortran\nf32",
             "lptlib_read_grid": "lptlib\nread_grid\nf64",
             "fortran_full": "Fortran\nf64"}
    keys = ["lptlib_strided", "naive_python", "fortran_contig",
            "lptlib_read_grid", "fortran_full"]

    for axis, ((points_m, cache_state), records) in zip(axes[0], ordered):
        readers, _ = summarize_bucket(records)
        present = [k for k in keys if k in readers]
        medians = [statistics.median(readers[k]["medians"]) for k in present]
        lows = [min(readers[k]["medians"]) for k in present]
        highs = [max(readers[k]["medians"]) for k in present]
        errs = [[m - lo for m, lo in zip(medians, lows)],
                [hi - m for m, hi in zip(medians, highs)]]
        bars = axis.bar(range(len(present)), medians, yerr=errs, capsize=4,
                        color=[colors.get(k, "#888") for k in present])
        axis.set_yscale("log")
        axis.set_xticks(range(len(present)))
        axis.set_xticklabels([short.get(k, k) for k in present], fontsize=8)
        axis.set_ylabel("Median read time (s, log scale)")
        n_machines = len({r["machine_label"] for r in records})
        flagged = any(not (r.get("cache_check") or {}).get("ok_for_declared_state", True)
                      for r in records)
        title = (f"~{points_m}M points, {cache_state} cache\n"
                 f"({n_machines} machine{'s' if n_machines != 1 else ''})")
        if flagged:
            title += "\ncache check FAILED, see SUMMARY.md"
        axis.set_title(title, fontsize=10,
                       color="#a11" if flagged else "black")
        axis.axvline(2.5, color="#999", linestyle="--", linewidth=0.8)
        for bar, value in zip(bars, medians):
            axis.text(bar.get_x() + bar.get_width() / 2, value,
                      f"{value*1000:.0f} ms", ha="center", va="bottom", fontsize=8)
        axis.set_ylim(min(medians) / 2.5, max(medians) * 3.0)

    fig.suptitle("PLOT3D multi-block grid: matched-work read-and-reconstruct time",
                 fontsize=11)
    fig.text(0.5, 0.015,
             "Left of the dashed line: matched float32 C-contiguous work. "
             "Right: matched float64 padded-array-plus-bounds work. "
             "Error bars span the min-to-max across machines.",
             ha="center", fontsize=8, color="#444")
    fig.subplots_adjust(bottom=0.20, top=0.84)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(HERE / "results"),
                        help="directory holding the per-machine JSON result files")
    parser.add_argument("--out", default=None,
                        help="markdown summary path (default <results-dir>/SUMMARY.md)")
    parser.add_argument("--figure", default=None,
                        help="bar chart path (default <results-dir>/benchmark_bar.png)")
    parser.add_argument("--no-figure", action="store_true",
                        help="do not render the bar chart")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out = Path(args.out) if args.out else results_dir / "SUMMARY.md"
    figure = Path(args.figure) if args.figure else results_dir / "benchmark_bar.png"

    records = load_records(results_dir)
    if not records:
        print(f"No benchmark result files found in {results_dir}.")
        print("Run:  python run_benchmark.py")
        return 1

    buckets = {}
    for record in records:
        buckets.setdefault(bucket_key(record), []).append(record)

    write_summary(out, buckets, records)
    machines = {record["machine_label"] for record in records}
    print(f"Pooled {len(records)} result file(s) from {len(machines)} machine(s): "
          + ", ".join(sorted(machines)))
    for (points_m, cache_state), bucket_records in sorted(buckets.items()):
        print(f"  bucket ~{points_m}M points / {cache_state} cache: "
              f"{len(bucket_records)} file(s)")
    print(f"Wrote {out}")

    if not args.no_figure:
        try:
            make_figure(buckets, figure)
            print(f"Wrote {figure}")
        except Exception as exc:
            print(f"  Could not render figure ({exc}); summary written anyway.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
