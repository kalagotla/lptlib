# lptlib documentation

lptlib is a Python library for one-way coupled Lagrangian particle tracking in steady and unsteady compressible CFD data. This page gives an overview of the modules, a reference to the public API, and pointers to worked examples. For a quick orientation and installation instructions, see the [README](../README.md).

## Concepts

A tracking run moves through four stages. The flow field is read from PLOT3D files, a point is located inside the curvilinear grid, the flow state is interpolated to that point, and the point is advanced in time to trace a streamline or a particle path. For inertial particles the advance step integrates the particle momentum equation under a selected drag law, which lets a particle lag the fluid across shocks and through steep gradients. After a large ensemble of particles has been tracked, the scattered tracks are reduced back onto a structured mesh to produce Eulerian, PIV-like fields.

The governing parameter for tracer response is the Stokes number, the ratio of the particle response time to a characteristic timescale of the surrounding flow. As particle inertia grows, the tracer deviates more strongly from the true fluid motion, which is the bias that lptlib was built to quantify.

## Package layout

lptlib is organized into four subpackages: `lptlib.io`, `lptlib.streamlines`, `lptlib.function`, and `lptlib.test_cases`.

`lptlib.io` handles file input and output. `GridIO` and `FlowIO` read and write PLOT3D grids and solutions and compute grid metrics. `DataIO` performs the Lagrangian-to-Eulerian reduction and writes PLOT3D grid, fluid, and particle files. Its `compute()` runs over MPI, so that end-to-end call needs the `mpi` extra even on a single rank, while the reduction stages it is built from are plain NumPy and SciPy.

`lptlib.streamlines` carries the tracking pipeline. `Search` locates a point inside the curvilinear grid, `Interpolation` evaluates the flow at that point, and `Integration` advances streamlines and particle paths. `Streamlines` chains these stages into a single call, and `StochasticModel`, `Particle`, and `SpawnLocations` seed and run large particle ensembles in parallel.

`lptlib.function` provides supporting computation. `Variables` derives velocity, temperature, pressure, Mach number, and viscosity from the conserved flow state, and `Plots` and `Timer` are plotting and timing helpers.

`lptlib.test_cases` supplies analytic cases such as `ObliqueShock`, `ObliqueShockData`, and `ObliqueShockAlignedData` that synthesize a controlled grid and flow for validation and for tracer-response studies.

## Public API reference

Every symbol below is importable directly from the top-level package, for example `from lptlib import GridIO`. The top level re-exports exactly sixteen classes, listed in its `__all__` alongside `__version__`: `DataIO`, `FlowIO`, `GridIO`, `Integration`, `Interpolation`, `ObliqueShock`, `ObliqueShockAlignedData`, `ObliqueShockData`, `Particle`, `Plots`, `Search`, `SpawnLocations`, `StochasticModel`, `Streamlines`, `Timer`, and `Variables`. Those re-exports are explicit rather than star imports, so the four subpackage names survive: `lptlib.io`, `lptlib.streamlines`, `lptlib.function`, and `lptlib.test_cases` all resolve to the subpackages, and `from lptlib.streamlines import Search` works as well as `from lptlib import Search`.

### lptlib.io

`GridIO(filename)` reads a PLOT3D grid. Key methods are `read_grid(data_type='f4', store_type=None)` for a multi-block grid, `read_grid_fortran_2d(precision='single', plane='i')` for a two-dimensional Fortran-record plane, `compute_metrics()` for grid metrics, and `mgrd_to_p3d(xi, yi, out_file='mgrd_to_p3d.x', steps=5, step_size=None, data_type='f4')` for writing a grid back to PLOT3D.

`read_grid` takes two independent precision arguments. `data_type` describes the file on disk and is `'f4'` for single precision or `'f8'` for double. `store_type` selects the dtype of the reconstructed `grd` array in memory. The default, `None`, stores `grd` as float64, which matches the historical behaviour and gives `compute_metrics` full double-precision headroom for the Jacobian and inverse-metric terms. Passing `store_type='f4'` keeps `grd` at the on-disk single precision and roughly halves the read time on large grids by avoiding the upcast copy; use it for coordinate-only work, or for very large grids where single-precision metrics are acceptable. The `grd_min` and `grd_max` bounds are always returned as float64.

`FlowIO(filename)` reads a PLOT3D solution. Key methods are `read_flow(data_type='f4', print_progress=True)`, `read_unsteady_flow(data_type='f4')` for a time sequence, `read_formatted_txt(grid, data_type='f8')`, `read_flow_fortran_2d(precision='single', plane='i', rho_ref=1.0, vel_ref=1.0, p_ref=1.0)` and `read_flow_piv_fortran_2d(precision='double', plane='k', vel_ref=1.0)` for two-dimensional Fortran-record planes, and `mgrd_to_p3d(q, out_file='mgrd_to_p3d', mode=None, steps=5, data_type='f4')` for export.

`DataIO(grid, flow, percent_data=100, location='.', x_refinement=50, y_refinement=40, seed=None)` reduces Lagrangian tracks to Eulerian fields. `location` is the directory of per-particle `.npy` track files, `x_refinement` and `y_refinement` set the resolution of the output structured mesh, `percent_data` below 100 takes a stratified spatial subsample of the tracks, and `seed` makes that subsample reproducible. Call `compute()` to run the end-to-end reduction and PLOT3D export.

### lptlib.streamlines

`Search(grid, ppoint, warm_start=None)` locates a point. Call `compute(method=...)` with `distance`, `block_distance` (the default), `p-space`, or `c-space`. `warm_start` seeds the Newton-Raphson search of `p-space` and `c-space` with the computational-space point found for the previous step, which is what the trajectory integrators pass so a walk along a path does not restart the search from scratch each step. Helpers `p2c(ppoint)` and `c2p(cpoint)` convert between physical and computational space.

`Interpolation(flow, idx)` interpolates the flow to the located point. Call `compute(method=...)` with `p-space` (the default), `c-space`, `rbf-p-space`, `rbf-c-space`, `rgi-p-space`, `rgi-c-space`, or `simple_oblique_shock` for the analytic test case. The `rgi-p-space` variant needs a rectilinear, axis-aligned block and raises `ValueError` naming the alternatives on a curvilinear one.

`Integration(interp)` advances a point. Call `compute(method=..., time_step=1e-6)` with `p-space` (the default), `c-space`, `pRK2`, `cRK2`, `pRK4`, `cRK4`, or `unsteady-pRK4`. Inertial-particle dynamics use `compute_ppath(diameter=1e-6, density=1000, velocity=None, method='pRK4', time_step=1e-4, drag_model='stokes')`, whose `method` is one of `pRK4`, `cRK4`, or `unsteady-pRK4`.

`Streamlines(grid_file=None, flow_file=None, point=None, ...)` is the high-level orchestrator. Pass file paths in `grid_file` and `flow_file` to have it read the PLOT3D files itself, or pass already-populated `grid` and `flow` objects to `compute` instead. The remaining constructor arguments are `search='p-space'`, `interpolation='p-space'`, `integration='pRK4'`, `diameter=1e-7`, `density=1000`, `time_step=1e-3`, `max_time_step=1`, `drag_model='stokes'`, `adaptivity=0.001`, `magnitude_adaptivity=0.001`, `filepath=None`, `task=None`, and `debug=False`. Call `compute(method='p-space', grid=None, flow=None)`, where `method` is one of `p-space`, `adaptive-p-space`, `c-space`, `adaptive-c-space`, `ppath`, `adaptive-ppath`, `ppath-c-space`, `adaptive-ppath-c-space`, `unsteady-p-space`, or `unsteady-ppath`, and read the results from `streamline`, `fvelocity`, `svelocity`, and `time`.

`StochasticModel(particles, spawn_locations, method='adaptive-p-space', grid=None, flow=None)` runs many particles in parallel. Execution backends are `serial()`, `multi_thread()`, `multi_process()`, and `mpi_run()`; only `mpi_run()` needs the `mpi` extra. `Particle(seed=None)` defines a size distribution through `compute_distribution()`, and `SpawnLocations(particles)` defines seed points through `compute()`.

### lptlib.test_cases

`ObliqueShock(mach=None, deflection=None, shock_angle=None)` solves the oblique-shock relations. Call `compute()` and read `shock_angle`, `pressure_ratio`, `density_ratio`, `temperature_ratio`, and `mach_ratio`; each is a two-element array holding the weak and strong solutions.

`ObliqueShockData(oblique_shock=None, nx_max=100e-3, ny_max=500e-3, nz_max=1e-4, xpoints=200, ypoints=500, zpoints=5, inlet_temperature=48.20, inlet_density=0.07747)` builds a shock-normal test case. The shock sits on the plane `x = 0`, so the grid runs from `-nx_max` to `+nx_max` and the freestream enters at the shock angle. Pass the solved `ObliqueShock` as `oblique_shock`, or set it as an attribute after construction. The default is `None` rather than a shared `ObliqueShock()`: a mutable default argument is built once at import and then mutated in place by every `compute()` call, so every instance would have silently shared one shock solution. When `oblique_shock` is left at `None` the constructor builds a fresh `ObliqueShock()` for that instance. Set `nx_max`, `ny_max`, `nz_max`, `xpoints`, `ypoints`, `zpoints`, `inlet_temperature`, `inlet_density`, and `shock_strength` (`'weak'` or `'strong'`) as constructor arguments or as attributes, then call `create_grid()` and `create_flow()`. The results are a populated `GridIO` in `grid` and `FlowIO` in `flow`, ready to pass to `Streamlines` or `StochasticModel`.

`ObliqueShockAlignedData(oblique_shock=None, ...)` takes the same arguments with the same defaults and is the shock-aligned counterpart. The incoming flow is horizontal and the shock plane is instead tilted to the computed shock angle, passing through `(0, ny_max/2)` and separating the pre- and post-shock states by the signed distance `s = sin(beta) * x - cos(beta) * (y - ny_max/2)`. It takes the same attributes and the same `create_grid()` and `create_flow()` calls, and it is the right case when a trajectory should approach the shock along the x-axis rather than at an angle.

### lptlib.function

`Variables(flow, gamma=1.4, gas_constant=287.052874)` derives flow quantities. Methods include `compute_velocity()`, `compute_temperature()`, `compute_pressure()`, `compute_mach()`, and `compute_viscosity(law='keyes')`, with `compute()` running the full set. `compute_drag_coefficient(_re=None, _mach=None, _model='stokes')` is the single implementation of the drag suite listed below, and the particle-path integrator calls straight into it.

`Plots(file, grid=None, flow=None)` reads a saved particle-track file and draws the standard diagnostics: `plot_paths()`, `plot_velocity()`, `plot_fluid_velocity()`, `plot_relative_mach()`, `plot_relative_reynolds()`, `plot_drag_coefficient()`, and `plot_drag(particle_density=4230)`. Each takes an optional matplotlib `ax` and passes any further keyword arguments through to the plot call.

`Timer` is a context manager and decorator around `time.perf_counter` for timing a block or a function.

## Drag models

The particle-path integrator selects a spherical-particle drag closure through the `drag_model` argument. It accepts twelve values. Nine are published closures: `stokes`, `oseen`, `schiller-nauman`, `melling`, `cunningham`, `henderson`, `subramaniam-balachandar`, `loth`, and `tedeschi`; these are the nine the JOSS paper counts. The remaining three are `melling-2`, which is the Melling correction with the alternative 2.7 prefactor on the Knudsen term, the rigid-sphere standard-drag curve `sphere`, and `zero-drag` for a massless fluid tracer. Stokes applies in the creeping-flow limit, Henderson and Loth span the continuum to rarefied and compressible regimes, and the zero-drag mode reproduces the fluid streamline for reference. The Knudsen number the slip-corrected closures use is formed as `Kn = M/Re * sqrt(pi*gamma/2)`, and `compute_drag_coefficient` returns 0 for `Re <= 1e-9` so the creeping-flow limit stays finite. Because the model is a single argument, the same particle definition can be tracked under several closures and the responses compared directly. Air viscosity is evaluated through the Sutherland or Keyes laws.

## Worked examples

The [Quickstart in the README](../README.md#quickstart) is the place to start. Every snippet there runs against the synthetic `ObliqueShockData` case, needs no external data, and the whole sequence runs in about thirteen seconds.

`main.py` in the repository root is a research-scale version of the same case: a Mach 7.6 oblique shock over a 100 mm by 500 mm domain and no step cap. Its `n_concentration` defaults to 4 particles so that a first run finishes in minutes rather than hours; the published cloud used 1000, and `oblique_shock_response(n_concentration=1000)` reproduces it. It is a worked example rather than a quickstart, and a full run takes a long time and writes output files.

`test/test_dataio_reduction.py` exercises the Lagrangian-to-Eulerian reduction on a tiny synthetic particle set, and the other files under `test/` double as focused examples of search, interpolation, integration, streamlines, the drag models, and the plotting helpers. Tests that need the large PLOT3D datasets from the original research cases skip when those files are absent, which is the normal state of a fresh clone.

## Running the tests

Install the test extra and run the suite from the repository root with `pip install -e ".[test,mpi]"` followed by `pytest test -v`. Bare `pytest` is not enough, because several test modules import `parameterized`. The `mpi` extra matters here even though the MPI launcher tests are opt-in: one test in `test/test_dataio_reduction.py` drives `DataIO.compute()`, which runs over MPI on a single rank, and it errors rather than skipping without `mpi4py`. The suite exercises search, interpolation for steady and unsteady flow, integration, the drag models, streamlines, the DataIO reduction, plotting, and the MPI helpers. Set `LPTLIB_RUN_MPI=1` to include the tests that launch MPI processes with `mpiexec`; they also need a system MPI runtime, and under Open MPI as root they additionally need `OMPI_ALLOW_RUN_AS_ROOT=1` and `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`.
