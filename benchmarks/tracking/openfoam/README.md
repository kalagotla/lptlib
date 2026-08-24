# OpenFOAM cross-code comparison (one-way parcel tracking)

This case is a physics-matched, cross-code comparison of lptlib's inertial
particle tracking against OpenFOAM's Lagrangian parcel tracker. It is provided
as a ready-to-run case rather than run automatically, because OpenFOAM is not
available in every environment (for example the CI container). Run it on your
WSL or HPC OpenFOAM install.

## What is matched

The case reproduces the same verification field used by
`../verify_tracking.py`, so the OpenFOAM result can be overlaid directly on the
lptlib and reference-ODE curves:

- Geometry: a box 5e-4 m (x) by 6e-5 m (y) by 6e-5 m (z), meshed 500 x 4 x 4,
  the same streamwise resolution as the lptlib PLOT3D grid.
- Carrier field: frozen, one-way. The streamwise velocity is uniform at
  12 m/s for x <= 1e-4 m, ramps linearly to 4 m/s by x = 2e-4 m, then stays at
  4 m/s. Density 1.0 kg/m^3 and viscosity 1.846e-5 kg/m/s (Sutherland at 300 K)
  are constant, matching the lptlib field.
- Particles: density 1000 kg/m^3, diameter 1e-6 m, standard sphere drag
  (`sphereDrag`), seeded at x = 2e-5 m with the upstream fluid velocity so they
  advect with zero slip until the ramp. This matches the lptlib `sphere` case.
- Solver: `icoUncoupledKinematicParcelFoam`, which advects the kinematic cloud
  through the prescribed carrier field without solving the flow. That is exactly
  the one-way, offline mode lptlib operates in.

Because both codes integrate the same particle equation of motion through the
same field with the same drag law and particle properties, the relaxation
curves should agree. Any difference reflects integration scheme and drag-curve
implementation, not the problem setup.

## Requirements

Targeted at OpenFOAM v2006 or newer from openfoam.com. The core dictionaries
(`blockMeshDict`, `controlDict`, `fvSchemes`, `fvSolution`, `transportProperties`)
are build-independent. `constant/kinematicCloudProperties` uses the v2006+
sub-model layout. On an OpenFOAM Foundation (openfoam.org) build, the same
sub-models exist but a few keywords differ; the likely edits are:

- `sizeDistribution`/`fixedValueDistribution` may be `sizeDistribution { type fixedValue; fixedValueDistribution { value 1e-6; } }` on .com and a slightly different spelling on .org; set a fixed 1e-6 m diameter however your build expects.
- `particleForces { sphereDrag; }` is common to both.
- if `icoUncoupledKinematicParcelFoam` is unavailable, `uncoupledKinematicParcelFoam` (or `kinematicParcelFoam` with `coupled false`) is the equivalent frozen-carrier solver.

## Run

```
./Allrun
python3 compare_openfoam.py
```

`Allrun` builds the mesh, writes the cell centres, prescribes the frozen ramp
carrier field with `set_U.py`, tracks the parcels, and exports the cloud to VTK.
`compare_openfoam.py` collects every (x, particle velocity) sample from the VTK
output and overlays them on the reference ODE and the lptlib trajectory,
reporting the relative L2 difference and writing `openfoam_comparison.png` and
`openfoam_vs_reference.csv`.

## Interpreting the result

All parcels are seeded identically in a steady field, so every (x, v_x) sample
should fall on a single relaxation curve. Agreement with the reference ODE and
the lptlib curve demonstrates that lptlib and OpenFOAM produce the same
one-way particle relaxation for matched physics. This is the meaningful
cross-code comparison; a raw particles-per-second race between the two is not,
because OpenFOAM couples to its own mesh and solver machinery and lptlib works
on structured PLOT3D fields.
