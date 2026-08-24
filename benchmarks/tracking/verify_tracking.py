"""Accuracy verification for lptlib inertial-particle tracking.

This is the physics-verification half of the tracking benchmark. It checks that
lptlib's adaptive Lagrangian tracker reproduces a trusted reference solution of
the identical particle equation of motion. The reference is a high-accuracy
SciPy DOP853 integration that uses lptlib's own drag-coefficient routine and
the identical piecewise-linear fluid-velocity field, so the comparison isolates
the accuracy of lptlib's integrator and adaptive time-stepping rather than any
difference in the drag law or the flow field.

Two cases are run:

- Stokes drag, where the drag is linear in the slip and the relaxation is a
  clean exponential with time constant tau = rho_p dp^2 / (18 mu). This is the
  strongest check because the reference is effectively analytic.
- Standard sphere drag, where the drag coefficient is a nonlinear function of
  the slip Reynolds number, exercising the integrator on a nonlinear law.

For each case the particle is seeded in the uniform upstream, advects with zero
slip, then relaxes across a smooth velocity ramp toward the downstream value.
The script reports the relative L2 and maximum velocity error against the
reference and writes a table and a figure.

Run:

    python verify_tracking.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import common

HERE = Path(__file__).resolve().parent


def run_case(model, dp, rho_p, tmpdir, ni=500, u1=12.0, u2=4.0,
             rho=1.0, temperature=300.0, lx=5e-4, x1=1e-4, x2=2e-4, x0=2e-5):
    grid_path = str(tmpdir / f"ramp_{model}.x")
    flow_path = str(tmpdir / f"ramp_{model}.q")
    x_nodes, u_nodes = common.write_ramp_field(
        grid_path, flow_path, ni=ni, lx=lx, x1=x1, x2=x2, u1=u1, u2=u2,
        rho=rho, temperature=temperature)

    t0 = time.perf_counter()
    xp, vp, n_steps = common.track_lptlib(grid_path, flow_path, x0, dp, rho_p,
                                          drag_model=model)
    wall = time.perf_counter() - t0

    xr, vr = common.reference_relaxation(
        x_nodes, u_nodes, x0, dp, rho_p, rho, temperature, model=model,
        t_end=8e-5, max_step=1e-9)

    # Compare on the overlapping x-range, mapping the reference onto lptlib's x.
    xmin = max(xp.min(), xr.min())
    xmax = min(xp.max(), xr.max())
    mask = (xp >= xmin) & (xp <= xmax)
    vp_ref = np.interp(xp[mask], xr, vr)
    diff = vp[mask] - vp_ref
    rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(vp_ref))
    max_abs = float(np.max(np.abs(diff)))

    mu = common.sutherland_viscosity(temperature)
    tau = rho_p * dp ** 2 / (18.0 * mu)
    return {
        "model": model,
        "diameter_m": dp,
        "density_kg_m3": rho_p,
        "tau_s": tau,
        "u_upstream": u1,
        "u_downstream": u2,
        "lptlib_steps": n_steps,
        "lptlib_wall_s": wall,
        "final_vp": float(vp[-1]),
        "rel_l2_error": rel_l2,
        "max_abs_error_m_s": max_abs,
        "_curves": {"xp": xp, "vp": vp, "xr": xr, "vr": vr},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=str(HERE / "results"))
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import tempfile
    tmpdir = Path(tempfile.mkdtemp())

    cases = [
        run_case("stokes", dp=1e-6, rho_p=1000.0, tmpdir=tmpdir),
        run_case("sphere", dp=1e-6, rho_p=1000.0, tmpdir=tmpdir),
    ]

    # CSV + markdown
    csv_path = outdir / "verification_results.csv"
    with open(csv_path, "w") as f:
        f.write("model,diameter_m,density_kg_m3,tau_s,lptlib_steps,final_vp,"
                "rel_l2_error,max_abs_error_m_s\n")
        for c in cases:
            f.write(f"{c['model']},{c['diameter_m']:.3e},{c['density_kg_m3']:.1f},"
                    f"{c['tau_s']:.4e},{c['lptlib_steps']},{c['final_vp']:.5f},"
                    f"{c['rel_l2_error']:.4e},{c['max_abs_error_m_s']:.4e}\n")

    md_path = outdir / "verification_results.md"
    with open(md_path, "w") as f:
        f.write("# lptlib particle-tracking accuracy verification\n\n")
        f.write(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n")
        f.write("lptlib's adaptive tracker is compared against a high-accuracy "
                "SciPy DOP853 reference that integrates the identical particle "
                "equation of motion, using lptlib's own drag-coefficient routine "
                "and the identical piecewise-linear velocity field. A particle "
                "relaxes across a smooth velocity ramp from "
                f"{cases[0]['u_upstream']:.0f} to {cases[0]['u_downstream']:.0f} m/s.\n\n")
        f.write("| Drag model | dp (m) | rho_p | tau (s) | lptlib steps | "
                "final vp (m/s) | rel L2 error | max abs error (m/s) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for c in cases:
            f.write(f"| {c['model']} | {c['diameter_m']:.1e} | {c['density_kg_m3']:.0f} | "
                    f"{c['tau_s']:.3e} | {c['lptlib_steps']} | {c['final_vp']:.4f} | "
                    f"{c['rel_l2_error']:.2e} | {c['max_abs_error_m_s']:.2e} |\n")
        f.write("\nThe reference downstream velocity is "
                f"{cases[0]['u_downstream']:.0f} m/s; agreement to a small "
                "relative L2 error confirms the tracker integrates the particle "
                "dynamics correctly.\n")

    record = [{k: v for k, v in c.items() if k != "_curves"} for c in cases]
    with open(outdir / "verification_results.json", "w") as f:
        json.dump(record, f, indent=2)

    try:
        make_figure(cases, outdir / "verification_relaxation.png")
    except Exception as exc:  # pragma: no cover
        print("Could not render figure:", exc)

    print("Accuracy verification")
    for c in cases:
        print(f"  {c['model']:8s} rel L2 = {c['rel_l2_error']:.3e}  "
              f"max abs = {c['max_abs_error_m_s']:.3e} m/s  "
              f"steps = {c['lptlib_steps']}  final vp = {c['final_vp']:.4f}")
    print(f"  wrote {csv_path.name}, {md_path.name}, verification_relaxation.png in {outdir}")


def make_figure(cases, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.2), squeeze=False)
    for ax, c in zip(axes[0], cases):
        cu = c["_curves"]
        ax.plot(cu["xr"] * 1e3, cu["vr"], "-", color="#888", lw=3,
                label="reference ODE (DOP853)")
        ax.plot(cu["xp"] * 1e3, cu["vp"], "--", color="#2c7fb8", lw=1.5,
                label="lptlib adaptive")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("particle velocity $v_x$ (m/s)")
        ax.set_title(f"{c['model']} drag\nrel L2 = {c['rel_l2_error']:.1e}")
        ax.legend(fontsize=8)
    fig.suptitle("Inertial-particle relaxation: lptlib vs reference ODE", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
