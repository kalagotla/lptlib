"""Compare OpenFOAM parcel tracking against lptlib and the reference ODE.

After ``./Allrun`` has produced the lagrangian cloud VTK output, this script
collects every (x, particle-velocity) sample from every parcel at every write
time and overlays them on the same relaxation curve as the reference ODE and,
if lptlib is importable, the lptlib 'sphere' trajectory. Because all parcels are
seeded identically and the field is steady, every sample should fall on the same
relaxation curve, so the comparison does not depend on tracking individual
parcels across time steps.

Run from the openfoam case directory after Allrun:
    python3 compare_openfoam.py
"""

import glob
import re
import sys
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
# Reuse the shared field/reference helpers from the parent tracking package.
sys.path.insert(0, str(CASE.parent))
import common  # noqa: E402


def parse_legacy_vtk(path):
    """Return (positions Nx3, U Nx3 or None) from a legacy ASCII VTK file."""
    text = Path(path).read_text(errors="ignore")
    # POINTS n type ... then 3*n floats
    mp = re.search(r"POINTS\s+(\d+)\s+\w+\s*\n", text)
    if not mp:
        return None, None
    n = int(mp.group(1))
    after = text[mp.end():]
    nums = re.findall(r"[-+0-9.eE]+", after)
    pts = np.array(nums[:3 * n], dtype=float).reshape(n, 3)
    # Find a vector field named U in POINT_DATA (VECTORS U or FIELD ... U 3 n)
    U = None
    mv = re.search(r"VECTORS\s+U\s+\w+\s*\n", text)
    if mv:
        af = text[mv.end():]
        un = re.findall(r"[-+0-9.eE]+", af)
        U = np.array(un[:3 * n], dtype=float).reshape(n, 3)
    else:
        mf = re.search(r"^U\s+3\s+(\d+)\s+\w+\s*\n", text, re.MULTILINE)
        if mf:
            af = text[mf.end():]
            un = re.findall(r"[-+0-9.eE]+", af)
            U = np.array(un[:3 * n], dtype=float).reshape(n, 3)
    return pts, U


def collect_openfoam_samples():
    patterns = [
        str(CASE / "VTK" / "lagrangian" / "kinematicCloud" / "*.vtk"),
        str(CASE / "VTK" / "**" / "lagrangian" / "kinematicCloud" / "*.vtk"),
    ]
    files = []
    for p in patterns:
        files += glob.glob(p, recursive=True)
    files = sorted(set(files))
    xs, us = [], []
    for f in files:
        pts, U = parse_legacy_vtk(f)
        if pts is None or U is None or len(pts) == 0:
            continue
        xs.append(pts[:, 0])
        us.append(U[:, 0])
    if not xs:
        return None, None
    return np.concatenate(xs), np.concatenate(us)


def main():
    x_of, u_of = collect_openfoam_samples()
    if x_of is None:
        print("No OpenFOAM lagrangian VTK found. Run ./Allrun first, then "
              "ensure foamToVTK produced VTK/lagrangian/kinematicCloud/*.vtk.")
        return

    # Reference ODE (sphere drag) through the identical ramp field.
    ni = 500
    x_nodes = np.linspace(0.0, 5e-4, ni)
    u_nodes = common.ramp_velocity_nodes(x_nodes, 1e-4, 2e-4, 12.0, 4.0)
    xr, vr = common.reference_relaxation(x_nodes, u_nodes, x0=2e-5, dp=1e-6,
                                         rho_p=1000.0, rho_f=1.0, temperature=300.0,
                                         model="sphere", t_end=8e-5, max_step=1e-9)

    # Optional lptlib curve for direct overlay.
    x_lpt = v_lpt = None
    try:
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        gp, fp = str(tmp / "r.x"), str(tmp / "r.q")
        common.write_ramp_field(gp, fp)
        x_lpt, v_lpt, _ = common.track_lptlib(gp, fp, x0=2e-5, dp=1e-6,
                                              rho_p=1000.0, drag_model="sphere")
    except Exception as exc:
        print("lptlib overlay skipped:", exc)

    # Error of OpenFOAM samples against the reference curve.
    u_ref_at_of = np.interp(x_of, xr, vr)
    resid = u_of - u_ref_at_of
    rel_l2 = float(np.linalg.norm(resid) / np.linalg.norm(u_ref_at_of))
    print(f"OpenFOAM samples: {len(x_of)}  rel L2 vs reference ODE = {rel_l2:.3e}")

    out_csv = CASE / "openfoam_vs_reference.csv"
    with open(out_csv, "w") as f:
        f.write("x_m,u_x_openfoam,u_x_reference\n")
        order = np.argsort(x_of)
        for i in order:
            f.write(f"{x_of[i]:.8e},{u_of[i]:.6f},{u_ref_at_of[i]:.6f}\n")
    print("wrote", out_csv.name)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        ax.plot(xr * 1e3, vr, "-", color="#888", lw=3, label="reference ODE (sphere)")
        if x_lpt is not None:
            ax.plot(x_lpt * 1e3, v_lpt, "--", color="#2c7fb8", lw=1.5, label="lptlib adaptive")
        ax.plot(x_of * 1e3, u_of, "o", ms=3, color="#d95f0e", alpha=0.6,
                label="OpenFOAM parcels")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("particle velocity $v_x$ (m/s)")
        ax.set_title(f"Particle relaxation: OpenFOAM vs lptlib vs reference\n"
                     f"OpenFOAM rel L2 vs reference = {rel_l2:.1e}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(CASE / "openfoam_comparison.png", dpi=150)
        print("wrote openfoam_comparison.png")
    except Exception as exc:
        print("figure skipped:", exc)


if __name__ == "__main__":
    main()
