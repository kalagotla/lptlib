"""Collect the machine metadata recorded in every benchmark result file.

A benchmark number is only interpretable next to the machine that produced it,
and results from two machines can only be pooled if the two machines can be
told apart. Everything gathered here is written into the header of each result
file: OS and kernel, CPU model and core count, total RAM, the filesystem the
benchmark directory lives on and whether it looks like rotating rust, SSD/NVMe,
a RAM-backed filesystem, or a network mount, the Python and NumPy versions, and
the gfortran version and exact optimisation flags.

The storage class matters more than usual for this particular benchmark. The
readers under test are dominated by file I/O when the page cache is cold and by
memory bandwidth when it is warm, so a result recorded on a network mount or on
a machine that could not cache the file is measuring something quite different
from a result recorded on a warm local NVMe.

Nothing here is required for the benchmark to run. Every probe degrades to
None or "unknown" rather than raising, so the benchmark works on machines where
/proc and /sys are unavailable.

author: benchmark tooling for lptlib (Dilip Kalagotla)
"""

import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

NETWORK_FSTYPES = {
    "nfs", "nfs4", "cifs", "smb3", "smbfs", "afs", "9p", "fuse.sshfs",
    "glusterfs", "lustre", "ceph", "beegfs",
}
MEMORY_FSTYPES = {"tmpfs", "ramfs", "devtmpfs"}


def _read_text(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def _run(cmd):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def cpu_model():
    """Best-effort CPU model string."""
    if sys.platform.startswith("linux"):
        for line in _read_text("/proc/cpuinfo").splitlines():
            if line.startswith(("model name", "Model name", "Processor")):
                return line.split(":", 1)[1].strip()
    elif sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
    return platform.processor() or platform.machine() or "unknown"


def cpu_counts():
    """Return (logical_cores, physical_cores_or_None)."""
    logical = os.cpu_count()
    physical = None
    if sys.platform.startswith("linux"):
        seen = set()
        core_id = pkg_id = None
        for line in _read_text("/proc/cpuinfo").splitlines():
            if line.startswith("physical id"):
                pkg_id = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":", 1)[1].strip()
            elif not line.strip():
                if pkg_id is not None and core_id is not None:
                    seen.add((pkg_id, core_id))
                core_id = pkg_id = None
        if pkg_id is not None and core_id is not None:
            seen.add((pkg_id, core_id))
        physical = len(seen) or None
    return logical, physical


def total_ram_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out.isdigit():
            return int(out)
    return None


def _mount_for(path):
    """Return (mount_point, device, fstype) for the mount containing path."""
    best = ("", "", "unknown")
    target = os.path.realpath(str(path))
    best_len = -1
    for line in _read_text("/proc/mounts").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        device, mount_point, fstype = parts[0], parts[1].replace("\\040", " "), parts[2]
        if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
            if len(mount_point) > best_len:
                best_len = len(mount_point)
                best = (mount_point, device, fstype)
    return best


def _rotational(device):
    """Return True/False/None for the rotational flag of a block device."""
    name = os.path.basename(device)
    if not name:
        return None
    candidates = [name]
    # nvme0n1p3 -> nvme0n1, sda2 -> sda, vda1 -> vda
    stripped = name.rstrip("0123456789")
    if stripped.endswith("p") and "nvme" in stripped:
        stripped = stripped[:-1]
    else:
        stripped = name.rstrip("0123456789")
    candidates.append(stripped)
    for cand in candidates:
        flag = _read_text(f"/sys/block/{cand}/queue/rotational").strip()
        if flag in ("0", "1"):
            return flag == "1"
    return None


def storage_info(path):
    """Describe the storage backing ``path``.

    Returns a dict with the mount point, device, filesystem type, and a coarse
    ``storage_class`` of network / memory / rotational / ssd-or-nvme / unknown.
    """
    mount_point, device, fstype = _mount_for(path)
    info = {
        "mount_point": mount_point or None,
        "device": device or None,
        "fstype": fstype,
        "storage_class": "unknown",
        "rotational": None,
    }
    if fstype in NETWORK_FSTYPES or fstype.startswith("fuse."):
        info["storage_class"] = "network-or-fuse"
        return info
    if fstype in MEMORY_FSTYPES:
        info["storage_class"] = "memory-backed"
        return info
    if fstype == "overlay":
        info["storage_class"] = "overlay (container)"
        return info
    rot = _rotational(device)
    info["rotational"] = rot
    if rot is True:
        info["storage_class"] = "rotational"
    elif rot is False:
        info["storage_class"] = "nvme" if "nvme" in (device or "") else "ssd-or-nvme"
    if os.path.basename(device or "").startswith(("vd", "xvd")):
        # virtio and Xen block devices report the host's default rotational
        # flag, which is usually 1 whatever the physical medium is. Say so
        # rather than letting a reader trust "rotational" on a VM.
        info["note"] = ("virtualised block device; the kernel rotational flag "
                        "does not reflect the physical medium")
    return info


def gfortran_version(gfortran_cmd):
    if not gfortran_cmd:
        return None
    out = _run(list(gfortran_cmd) + ["--version"])
    return out.splitlines()[0] if out else None


def machine_label(explicit=None):
    """Resolve the machine label: --machine-label, then env, then hostname."""
    if explicit:
        return sanitize_label(explicit)
    env = os.environ.get("LPTLIB_BENCH_MACHINE")
    if env:
        return sanitize_label(env)
    return sanitize_label(socket.gethostname() or platform.node() or "unknown-machine")


def sanitize_label(label):
    """Make a label safe to embed in a filename."""
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in label.strip())
    safe = "-".join(part for part in safe.split("-") if part)
    return safe or "unknown-machine"


def collect(bench_dir, gfortran_cmd=None, compile_cmd=None, numpy_version=None):
    """Return the full machine-metadata dict written into every result file."""
    logical, physical = cpu_counts()
    ram = total_ram_bytes()
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(),
        "kernel_version": platform.version(),
        "machine_arch": platform.machine(),
        "cpu_model": cpu_model(),
        "cpu_logical_cores": logical,
        "cpu_physical_cores": physical,
        "total_ram_bytes": ram,
        "total_ram_gib": round(ram / (1 << 30), 2) if ram else None,
        "storage": storage_info(bench_dir),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": numpy_version,
        "gfortran_version": gfortran_version(gfortran_cmd),
        "gfortran_compile_cmd": compile_cmd,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "shutil_disk_free_bytes": (
            shutil.disk_usage(str(bench_dir)).free
            if os.path.isdir(str(bench_dir)) else None
        ),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(collect(Path(__file__).resolve().parent), indent=2))
