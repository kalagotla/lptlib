# Particle-tracking benchmarks

These benchmarks characterize lptlib's Lagrangian inertial-particle tracking,
which is the library's central capability, along three axes: accuracy against a
trusted reference, throughput, and a physics-matched cross-code comparison
against OpenFOAM. They complement the PLOT3D I/O benchmark in the parent
directory.

Comparing particle-tracking libraries head to head is inherently delicate
because they target different problems. OpenFOAM's Lagrangian solvers advect
parcels on an unstructured mesh, usually while solving the carrier flow;
OceanParcels advects passive or simply-forced particles on incompressible ocean
fields; lptlib does one-way, offline tracking of inertial particles through
structured curvilinear PLOT3D fields with a compressible and rarefied drag
suite. A raw particles-per-second race across these mixes different physics and
machinery. These benchmarks therefore lead with accuracy verification, report
throughput with explicit caveats, and make the cross-code comparison physics
matched rather than a speed contest.

## The test field

All three parts use a controlled field (`common.py`) in which the streamwise
fluid velocity is uniform at 12 m/s, ramps linearly to 4 m/s, then stays at
4 m/s, with uniform density and temperature. lptlib reconstructs a
piecewise-linear nodal velocity exactly, and uniform density and temperature
keep the viscosity and Stokes relaxation time constant, so a reference solution
is clean. A particle seeded in the uniform upstream advects with zero slip, then
relaxes across the ramp, which is the same behavior a tracer shows crossing a
shock.

## Accuracy verification

```
python verify_tracking.py
```

lptlib's adaptive tracker is compared against a high-accuracy SciPy DOP853
integration of the identical particle equation of motion,

    dv/dt = -0.75 Cd rho_f/(rho_p dp) |v-u| (v-u),

using lptlib's own drag-coefficient routine and the identical velocity field, so
the comparison isolates the integrator. Representative result on the recorded
machine: for both Stokes drag (linear, effectively analytic reference) and
standard sphere drag (nonlinear in the slip Reynolds number), the adaptive
tracker matches the reference to a relative L2 velocity error of about
**4e-4**, relaxing to the correct downstream velocity. Outputs are written to
`results/verification_results.{md,csv,json}` and `verification_relaxation.png`.

## Throughput

```
python throughput_tracking.py --particles 25
```

Reports particles tracked per second for lptlib and a specialized vectorized
NumPy reference integrator (`reference_integrator.py`) on the same frozen field
and drag law, and for OceanParcels if it is installed. These are not a
like-for-like contest. The reference integrator is a structured-1D vectorized
solver with no point-location search, so it is far faster per particle and marks
an upper bound; lptlib's cost reflects the general curvilinear search,
interpolation, compressible drag suite, and adaptive stepping it performs for
every particle. On the recorded machine lptlib tracked on the order of one
particle per second for this fully adaptive relaxation trajectory while the
reference integrator reached thousands per second. Outputs are written to
`results/throughput_results.{md,csv,json}`.

## OpenFOAM cross-code comparison

See `openfoam/`. It contains a complete `icoUncoupledKinematicParcelFoam` case
matched to the verification field (same geometry, frozen carrier ramp, carrier
density and viscosity, particle density and diameter, and sphere drag), plus
`compare_openfoam.py` which overlays the OpenFOAM parcel relaxation on the
lptlib and reference-ODE curves. OpenFOAM is not run automatically; run it on
your own OpenFOAM install and then run the comparison.

## Notes

- `common.py` installs a lightweight `mpi4py` stub only if a working MPI runtime
  is absent, so the tracking benchmarks import lptlib without system MPI. On a
  normal install with mpi4py present the stub is ignored.
- The generated PLOT3D fields are written to a temporary directory and are not
  committed.
