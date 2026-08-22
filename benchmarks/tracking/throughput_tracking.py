"""Throughput comparison for lptlib inertial-particle tracking.

This is the speed half of the tracking benchmark. It tracks many inertial
particles through the same frozen velocity-ramp field and reports throughput in
particles tracked per second.

Two comparators are used, and the caveats matter:

- A reference vectorized NumPy integrator (in ``reference_integrator.py``) that
  advances all particles together with a fixed-step scheme through the identical
  field and drag law. This is a specialized, in-memory, structured-1D solver
  with no point-location search, so it is expected to be much faster per
  particle. It sets an upper bound and shows the cost of lptlib's general
  curvilinear machinery, not a like-for-like contest.
- OceanParcels, if installed. OceanParcels is a widely used Lagrangian particle
  tracking library, but it targets incompressible ocean and atmospheric fields
  and advects passive or simply-forced particles, so any number here is an
  order-of-magnitude context point across a different problem domain, not a
  physics-matched comparison. If OceanParcels is not installed the script skips
  it and says so.

A fair, physics-matched cross-code comparison against a compressible parcel
tracker (OpenFOAM icoUncoupledKinematicParcelFoam) is provided separately under
``openfoam/`` because OpenFOAM does not run in every environment.

Run:

    python throughput_tracking.py --particles 25
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import common
from reference_integrator import track_particles_reference

HERE = Path(__file__).resolve().parent


def time_lptlib(grid_path, flow_path, n_particles, dp, rho_p, y_positions):
    """Track n_particles one at a time with lptlib and return (seconds, total_steps)."""
    t0 = time.perf_counter()
    total_steps = 0
    for y0 in y_positions:
        _x, _vp, n = common.track_lptlib(grid_path, flow_path, x0=2e-5, dp=dp,
                                         rho_p=rho_p, drag_model="stokes", y0=float(y0))
        total_steps += n
    return time.perf_counter() - t0, total_steps


def try_oceanparcels(x_nodes, u_nodes, n_particles, dp, rho_p, rho_f, temperature):
    """Time OceanParcels advecting n_particles through the same u(x), or None."""
    try:
        import parcels  # noqa: F401
    except Exception:
        return None
    try:
        from parcels import FieldSet, ParticleSet, ScipyParticle, Variable, AdvectionRK4
        import numpy as _np

        # Build a 2D FieldSet with U(x) from the same nodal profile and V=0.
        nx = len(x_nodes)
        ny = 4
        lon = x_nodes
        lat = _np.linspace(0.0, 6e-5, ny)
        U = _np.tile(u_nodes.reshape(1, nx), (ny, 1))
        V = _np.zeros((ny, nx))
        fieldset = FieldSet.from_data({"U": U, "V": V},
                                      {"lon": lon, "lat": lat}, mesh="flat")
        pset = ParticleSet(fieldset=fieldset, pclass=ScipyParticle,
                           lon=_np.full(n_particles, x_nodes[1]),
                           lat=_np.linspace(1e-5, 5e-5, n_particles))
        t0 = time.perf_counter()
        pset.execute(AdvectionRK4, runtime=8e-5, dt=1e-8)
        return time.perf_counter() - t0
    except Exception as exc:  # pragma: no cover
        print("  OceanParcels present but the run failed:", exc)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--particles", type=int, default=25)
    parser.add_argument("--outdir", default=str(HERE / "results"))
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import tempfile
    tmpdir = Path(tempfile.mkdtemp())
    grid_path = str(tmpdir / "ramp.x")
    flow_path = str(tmpdir / "ramp.q")
    dp, rho_p, rho_f, temperature = 1e-6, 1000.0, 1.0, 300.0
    x_nodes, u_nodes = common.write_ramp_field(grid_path, flow_path)

    n = args.particles
    y_positions = np.linspace(1e-5, 5e-5, n)

    print(f"Timing lptlib on {n} particles ...")
    lpt_s, lpt_steps = time_lptlib(grid_path, flow_path, n, dp, rho_p, y_positions)

    print(f"Timing reference vectorized NumPy integrator on {n} particles ...")
    ref_s, ref_steps = track_particles_reference(
        x_nodes, u_nodes, n, dp, rho_p, rho_f, temperature,
        model="stokes", dt=2e-8, t_end=8e-5)

    print("Trying OceanParcels ...")
    op_s = try_oceanparcels(x_nodes, u_nodes, n, dp, rho_p, rho_f, temperature)

    rows = [
        ("lptlib (adaptive, curvilinear search+interp)", lpt_s, n / lpt_s, lpt_steps),
        ("reference NumPy (vectorized, structured 1D)", ref_s, n / ref_s, ref_steps),
    ]
    if op_s is not None:
        rows.append(("OceanParcels (passive advection, flat mesh)", op_s, n / op_s, None))

    env = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "n_particles": n,
        "oceanparcels_run": op_s is not None,
    }

    csv_path = outdir / "throughput_results.csv"
    with open(csv_path, "w") as f:
        f.write("method,n_particles,seconds,particles_per_sec,total_steps\n")
        for name, s, pps, steps in rows:
            f.write(f"{name},{n},{s:.4f},{pps:.3f},{steps if steps is not None else ''}\n")

    md_path = outdir / "throughput_results.md"
    with open(md_path, "w") as f:
        f.write("# lptlib particle-tracking throughput\n\n")
        f.write(f"Generated {env['timestamp']}\n\n")
        f.write(f"- Platform: {env['platform']}\n- Python: {env['python']}\n")
        f.write(f"- NumPy: {env['numpy']}\n- Particles: {n}\n")
        f.write(f"- OceanParcels run: {env['oceanparcels_run']}\n\n")
        f.write("| Method | Particles | Seconds | Particles/sec | Total steps |\n")
        f.write("|---|---|---|---|---|\n")
        for name, s, pps, steps in rows:
            f.write(f"| {name} | {n} | {s:.3f} | {pps:.2f} | "
                    f"{steps if steps is not None else 'n/a'} |\n")
        f.write("\nThese comparators solve different problems and are not a "
                "like-for-like contest. The reference NumPy integrator is a "
                "specialized structured-1D vectorized solver with no point "
                "location, so it is far faster per particle and marks an upper "
                "bound. OceanParcels, when present, advects passive particles on "
                "a flat mesh and is included only as cross-library context. "
                "lptlib's cost reflects the general curvilinear point-location, "
                "interpolation, compressible drag suite, and adaptive stepping "
                "that it performs for every particle. A physics-matched "
                "cross-code comparison lives under `openfoam/`.\n")

    with open(outdir / "throughput_results.json", "w") as f:
        json.dump({"environment": env,
                   "rows": [{"method": n_, "seconds": s, "particles_per_sec": pps,
                             "total_steps": steps} for n_, s, pps, steps in rows]},
                  f, indent=2)

    print("\nThroughput")
    for name, s, pps, steps in rows:
        print(f"  {name:48s} {pps:8.2f} particles/sec  ({s:.2f} s)")
    print(f"  wrote {csv_path.name}, {md_path.name} in {outdir}")


if __name__ == "__main__":
    main()
