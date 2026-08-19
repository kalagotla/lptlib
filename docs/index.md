# lptlib documentation

lptlib is a Python library for one-way coupled Lagrangian particle tracking in steady and unsteady compressible CFD data. This page gives an overview of the modules, a reference to the public API, and pointers to worked examples. For a quick orientation and installation instructions, see the [README](../README.md).

## Concepts

A tracking run moves through four stages. The flow field is read from PLOT3D files, a point is located inside the curvilinear grid, the flow state is interpolated to that point, and the point is advanced in time to trace a streamline or a particle path. For inertial particles the advance step integrates the particle momentum equation under a selected drag law, which lets a particle lag the fluid across shocks and through steep gradients. After a large ensemble of particles has been tracked, the scattered tracks are reduced back onto a structured mesh to produce Eulerian, PIV-like fields.

The governing parameter for tracer response is the Stokes number, the ratio of the particle response time to a characteristic timescale of the surrounding flow. As particle inertia grows, the tracer deviates more strongly from the true fluid motion, which is the bias that lptlib was built to quantify.

## Package layout

lptlib is organized into three subpackages plus a set of synthetic test cases.

`lptlib.io` handles file input and output. `GridIO` and `FlowIO` read and write PLOT3D grids and solutions and compute grid metrics. `DataIO` performs the MPI-parallel Lagrangian-to-Eulerian reduction and writes PLOT3D grid, fluid, and particle files.

`lptlib.streamlines` carries the tracking pipeline. `Search` locates a point inside the curvilinear grid, `Interpolation` evaluates the flow at that point, and `Integration` advances streamlines and particle paths. `Streamlines` chains these stages into a single call, and `StochasticModel`, `Particle`, and `SpawnLocations` seed and run large particle ensembles in parallel.

`lptlib.function` provides supporting computation. `Variables` derives velocity, temperature, pressure, Mach number, and viscosity from the conserved flow state, and `Plots` and `Timer` are plotting and timing helpers.

`lptlib.test_cases` supplies analytic cases such as `ObliqueShock` and `ObliqueShockData` that synthesize a controlled grid and flow for validation and for tracer-response studies.

## Public API reference

Every symbol below is importable directly from the top-level package, for example `from lptlib import GridIO`.

### lptlib.io

`GridIO(filename)` reads a PLOT3D grid. Key methods are `read_grid(data_type='f4')` for a multi-block grid, `read_grid_fortran_2d(precision, plane)` for a two-dimensional Fortran-record plane, `compute_metrics()` for grid metrics, and `mgrd_to_p3d(...)` for writing a grid back to PLOT3D.

`FlowIO(filename)` reads a PLOT3D solution. Key methods are `read_flow(data_type='f4')`, `read_unsteady_flow(...)` for a time sequence, `read_formatted_txt(...)`, and `mgrd_to_p3d(...)` for export.

`DataIO(grid, flow, percent_data=100, read_file=None, location='.', ...)` reduces Lagrangian tracks to Eulerian fields. Call `compute()` to run the end-to-end reduction and PLOT3D export.

### lptlib.streamlines

`Search(grid, ppoint)` locates a point. Call `compute(method=...)` with `distance`, `block_distance`, `p-space`, or `c-space`. Helpers `p2c(ppoint)` and `c2p(cpoint)` convert between physical and computational space.

`Interpolation(flow, idx)` interpolates the flow to the located point. Call `compute(method=...)` with `p-space`, `c-space`, radial-basis-function options, regular-grid options, or `simple_oblique_shock` for the analytic test case.

`Integration(interp)` advances a point. Call `compute(method=..., time_step=...)` with `pRK2`, `pRK4`, `cRK2`, `cRK4`, or their unsteady variants. Particle dynamics use `compute_ppath(...)` with a `drag_model` argument.

`Streamlines(grid_file, flow_file, point, ...)` is the high-level orchestrator. Call `compute(method=...)` and read the results from `streamline`, `fvelocity`, `svelocity`, and `time`.

`StochasticModel(particles, spawn_locations, method='adaptive-p-space', grid=None, flow=None)` runs many particles in parallel. Execution backends are `serial()`, `multi_thread()`, `multi_process()`, and `mpi_run()`. `Particle()` defines a size distribution through `compute_distribution()`, and `SpawnLocations(particles)` defines seed points through `compute()`.

### lptlib.function

`Variables(flow, gamma=1.4, gas_constant=287.052874)` derives flow quantities. Methods include `compute_velocity()`, `compute_temperature()`, `compute_pressure()`, `compute_mach()`, and `compute_viscosity()`, with `compute()` running the full set.

## Drag models

The particle-path integrator selects a spherical-particle drag closure through the `drag_model` argument. The available closures are `stokes`, `oseen`, `schiller-nauman`, `melling`, `melling-2`, `cunningham`, `henderson`, `subramaniam-balachandar`, `loth`, `tedeschi`, the rigid-sphere standard-drag curve `sphere`, and `zero-drag` for a massless fluid tracer. Stokes applies in the creeping-flow limit, Henderson and Loth span the continuum to rarefied and compressible regimes, and the zero-drag mode reproduces the fluid streamline for reference. Because the model is a single argument, the same particle definition can be tracked under several closures and the responses compared directly. Air viscosity is evaluated through the Sutherland or Keyes laws.

## Worked examples

`main.py` in the repository root is a complete oblique-shock tracer-response run. It synthesizes a Mach 7.6 oblique shock, seeds a constant-diameter particle distribution, and launches an adaptive parallel simulation with the Loth drag model.

`test/test_dataio.py` is a minimal, runnable example of the Lagrangian-to-Eulerian reduction. The other files under `test/` double as focused examples of search, interpolation, integration, streamlines, and the drag models.

## Running the tests

Install pytest and run the suite from the repository root with `pytest test -v`. The suite exercises search, interpolation for steady and unsteady flow, integration, the drag models, streamlines, the DataIO reduction, plotting, and the MPI helpers.
