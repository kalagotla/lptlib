# lptlib

**A parallel Lagrangian particle tracking library for compressible CFD data and tracer-response analysis in optical velocimetry.**

[![PyPI version](https://img.shields.io/pypi/v/lptlib.svg)](https://pypi.org/project/lptlib/)
[![Python versions](https://img.shields.io/pypi/pyversions/lptlib.svg)](https://pypi.org/project/lptlib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/kalagotla/lptlib/actions/workflows/ci.yml/badge.svg)](https://github.com/kalagotla/lptlib/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22006302.svg)](https://doi.org/10.5281/zenodo.22006302)

lptlib reads structured, multi-block, curvilinear PLOT3D grid and flow fields, locates arbitrary points inside the curvilinear domain, interpolates the flow state to those points, and integrates fluid streamlines and inertial particle trajectories through the field. Particle motion is advanced with a spherical-particle drag law chosen from a broad set of compressible and rarefied closures, so a single particle definition can be tracked under different drag models and the results compared directly. Large ensembles of particles run in parallel through multiprocessing, thread pools, or MPI, with adaptive time stepping for the stiff dynamics near shocks.

The library was built to quantify how inertial tracer particles lag the fluid in high-speed particle image velocimetry (PIV) and particle tracking velocimetry (PTV), and the same machinery applies to any one-way coupled particle-laden flow where inertial particles are advected through a resolved compressible field. It was previously developed under the name project-arrakis.

## Highlights

Three capabilities set lptlib apart.

- **Vectorized PLOT3D input and output.** Each grid or flow file is read in a single buffered `numpy.fromfile` call and reconstructed into multi-block arrays through strided slicing and Fortran-order reshaping. This removes the per-point Python overhead that dominates naive readers. Readers are only ever compared when they do matched work, since a bare stream read and a read followed by a full array reorder are not the same task. On a 2.04 million-point four-block grid with the page-cache state verified by measurement, the strided single-precision read is 27.80 times faster than an otherwise identical naive Python reader, with a 95 percent bootstrap confidence interval of 26.09 to 28.76. Against a compiled Fortran reader doing the identical transpose it is about 1.40 times slower, while the full `read_grid`, which also builds the padded double-precision array and the per-block coordinate bounds, is about 1.75 times faster than Fortran doing that same work. The margin over the naive reader narrows as the grid grows rather than widening. On a 28.2 million-point grid it falls to 13.85 times, and at that size the Fortran reader is ahead on both matched-work comparisons. These come from one machine and are quoted as ratios, because absolute read times move by more than an order of magnitude between machines while the matched-work ratios largely do not. Result files, machine metadata, and per-run uncertainty live in [`benchmarks/results/`](benchmarks/results/SUMMARY.md), and [`benchmarks/README.md`](benchmarks/README.md) explains how to rerun the comparison and how to add a machine.
- **MPI-parallel Lagrangian-to-Eulerian reduction.** Scattered particle tracks are binned and interpolated back onto a user-defined structured mesh in parallel with MPI, then exported as PLOT3D grid, fluid, and particle files. This produces the PIV-like Eulerian fields that a downstream synthetic-imaging tool consumes to render frames.
- **A twelve-model drag suite behind one argument.** `drag_model` accepts twelve values: the nine published closures `stokes`, `oseen`, `schiller-nauman`, `melling`, `cunningham`, `henderson`, `loth`, `subramaniam-balachandar`, and `tedeschi`, the nine the accompanying paper counts, plus `melling-2` for the alternative Knudsen prefactor, the rigid-sphere standard-drag curve `sphere`, and a `zero-drag` mode for fluid tracers. Tracer response can be studied as a function of particle size, density, and drag closure without changing anything else.

## Full feature list

At a glance, lptlib provides:

- PLOT3D input and output for multi-block, structured, curvilinear grids and flow solutions, in single and double precision, including unsteady flow sequences and two-dimensional Fortran-record planes.
- Point location in curvilinear grids with physical-space and computational-space search, and conversion between the two spaces.
- Interpolation of flow variables to arbitrary points, with physical-space, computational-space, radial-basis-function, and regular-grid options, plus an analytic oblique-shock interpolant for controlled test cases.
- Streamline and particle-path integration with second- and fourth-order Runge-Kutta schemes in physical and computational space, unsteady variants, and adaptive time stepping.
- The twelve `drag_model` values described above, with air viscosity evaluated through Sutherland or Keyes laws.
- Stochastic seeding of particle-size distributions and spawn locations, with parallel execution over many particles through multiprocessing, thread pools, and MPI.
- Derived-variable computation for velocity, temperature, pressure, Mach number, and viscosity.
- The MPI-parallel Lagrangian-to-Eulerian reduction that writes PLOT3D fluid and particle fields for visualization and for synthetic PIV imaging.

## Installation

lptlib is on PyPI.

```bash
pip install lptlib
```

Python 3.10 or newer is required. The core dependencies are `numpy`, `scipy`, `matplotlib`, `pandas`, and `tqdm`, and nothing else is needed for the library's serial and multiprocessing paths.

The MPI execution backends (`StochasticModel.mpi_run` and the `DataIO` Lagrangian-to-Eulerian reduction) are optional and live behind an extra.

```bash
pip install "lptlib[mpi]"
```

`mpi4py` needs a system MPI implementation. On Debian or Ubuntu install one with `sudo apt-get install libopenmpi-dev openmpi-bin`, and on macOS with `brew install open-mpi`. Without the extra, `import lptlib` and every non-MPI code path still work; calling an MPI backend raises a clear error telling you what to install.

## Quickstart

Every snippet below runs against a synthetic oblique-shock case that lptlib builds in memory, so no external data files, downloads, or CFD solutions are needed. The snippets build on each other: run them in order in one interactive session, or paste them into a single file. The whole quickstart runs in about thirteen seconds on the two-core machine these timings were measured on; the per-example times quoted below vary with core count for example 4, which runs its particles in parallel.

### 1. Build a synthetic test case (about one second)

`ObliqueShock` solves the oblique-shock relations, and `ObliqueShockData` turns that solution into a structured 3D grid and a matching PLOT3D flow state.

```python
from lptlib import ObliqueShock, ObliqueShockData

# Oblique-shock relations for a Mach 7.6 flow deflected by 20 degrees
shock = ObliqueShock()
shock.mach = 7.6
shock.deflection = 20                 # degrees
shock.compute()
print(f'weak-shock angle   : {shock.shock_angle[0]:.2f} deg')
print(f'density ratio      : {shock.density_ratio[0]:.3f}')
print(f'temperature ratio  : {shock.temperature_ratio[0]:.3f}')

# Synthesize a structured 3D grid and the matching PLOT3D flow state
osd = ObliqueShockData()
osd.oblique_shock = shock
osd.nx_max, osd.ny_max, osd.nz_max = 5e-3, 30e-3, 1e-4   # metres
osd.inlet_temperature = 48.20         # K
osd.inlet_density = 0.07747           # kg/m^3
osd.xpoints, osd.ypoints, osd.zpoints = 20, 60, 5
osd.shock_strength = 'weak'
osd.create_grid()
osd.create_flow()
print('grid:', osd.grid.grd.shape, ' flow:', osd.flow.q.shape)
```

```text
weak-shock angle   : 26.85 deg
density ratio      : 4.213
temperature ratio  : 3.224
grid: (40, 60, 5, 3, 1)  flow: (40, 60, 5, 5, 1)
```

### 2. Locate a point and interpolate the flow to it (under a second)

`Search`, `Interpolation`, and `Variables` are the three stages behind every tracking run, and they can be called directly.

```python
import numpy as np
from lptlib import Search, Interpolation, Variables

for x in (-2e-3, 2e-3):                      # upstream and downstream of the shock
    idx = Search(osd.grid, [x, 5e-3, 5e-5])
    idx.compute(method='p-space')            # locate the point in the curvilinear grid
    interp = Interpolation(osd.flow, idx)
    interp.compute(method='p-space')         # interpolate the conserved state
    q = Variables(interp)
    q.compute()                              # derive velocity, T, p, Mach, viscosity
    print(f'x = {x * 1e3:+5.1f} mm  '
          f'|u| = {np.linalg.norm(q.velocity):7.1f} m/s  '
          f'T = {q.temperature.ravel()[0]:6.2f} K  '
          f'M = {q.mach.ravel()[0]:4.2f}')
```

```text
x =  -2.0 mm  |u| =  1057.8 m/s  T =  48.20 K  M = 7.60
x =  +2.0 mm  |u| =   950.5 m/s  T = 155.41 K  M = 3.80
```

### 3. Track an inertial particle and compare drag models (about three seconds)

This is the measurement lptlib exists to make. A 1.94 micron tracer is released upstream and integrated across the shock under three closures. `zero-drag` reproduces the fluid path exactly, while the inertial closures leave the particle over-speeding the decelerated post-shock gas by roughly 105 m/s, which is the velocity bias a PIV or PTV measurement would report.

`max_steps` caps the integration so the example finishes quickly.

```python
import numpy as np
from lptlib import Streamlines

for model in ('zero-drag', 'stokes', 'loth'):
    sl = Streamlines(point=[-5e-3, 1e-3, 5e-5],      # seed upstream of the shock
                     diameter=1.94e-6, density=950,  # 1.94 um TiO2-like tracer
                     drag_model=model,
                     time_step=1e-8,
                     interpolation='simple_oblique_shock')
    sl.max_steps = 400                                # keep the example short
    sl.compute(method='adaptive-ppath', grid=osd.grid, flow=osd.flow)
    vp = np.linalg.norm(sl.svelocity[-1])             # particle speed
    uf = np.linalg.norm(sl.fvelocity[-1])             # local fluid speed
    print(f'{model:>10}: particle {vp:7.1f} m/s   fluid {uf:7.1f} m/s   slip {vp - uf:6.1f} m/s')
```

```text
 zero-drag: particle   950.5 m/s   fluid   950.5 m/s   slip    0.0 m/s
    stokes: particle  1055.9 m/s   fluid   950.5 m/s   slip  105.4 m/s
      loth: particle  1056.1 m/s   fluid   950.5 m/s   slip  105.6 m/s
```

The two inertial runs each also print a line saying the step cap was reached, which is `max_steps` doing its job. Drop `max_steps` and the particle runs to the domain boundary instead.

### 4. Run an ensemble in parallel (about nine seconds on two cores)

`Particle` defines a size distribution, `SpawnLocations` defines where the particles enter, and `StochasticModel` runs them. The backends are `serial()`, `multi_thread()`, `multi_process()`, and `mpi_run()`; only the last needs the `mpi` extra.

```python
import numpy as np
from lptlib import StochasticModel, Particle, SpawnLocations

particle = Particle()
particle.min_dia = particle.max_dia = particle.mean_dia = 1.94e-6
particle.std_dia = 0
particle.density = 950
particle.n_concentration = 8
particle.distribution = 'gaussian'
particle.compute_distribution()

spawn = SpawnLocations(particle)
spawn.x_min, spawn.z_min = -5e-3, 5e-5
spawn.y_min, spawn.y_max = 1e-3, 5e-3
spawn.compute()

model = StochasticModel(particle, spawn, grid=osd.grid, flow=osd.flow)
model.method = 'adaptive-ppath'
model.interpolation = 'simple_oblique_shock'
model.drag_model = 'loth'
model.time_step = 1e-8
model.max_steps = 300                 # keep the example short
tracks = model.multi_process()        # one process per core

print(f'tracked {len(tracks)} particles')
xyz = np.array(tracks[0].streamline)
print(f'first track: {len(xyz)} points, x from {xyz[0, 0]:+.4f} to {xyz[-1, 0]:+.4f} m')
```

```text
tracked 8 particles
first track: 300 points, x from -0.0050 to +0.0000 m
```

As in example 3, the run also prints a `tqdm` progress bar and one step-cap notice per particle alongside the two lines above.

A production run raises `n_concentration` to thousands and removes `max_steps`, which is where the parallel backends and the MPI reduction earn their keep. Expect minutes to hours rather than seconds.

### 5. Read and write PLOT3D files (under a second)

`GridIO` and `FlowIO` are the vectorized PLOT3D readers. The snippet writes the synthetic case out in PLOT3D format and reads it back, so it doubles as a description of the binary layout the readers expect.

```python
import numpy as np
from lptlib import GridIO, FlowIO

ni, nj, nk = int(osd.grid.ni[0]), int(osd.grid.nj[0]), int(osd.grid.nk[0])

# PLOT3D multi-block binary: block count, per-block dimensions, then the data
with open('shock.x', 'wb') as f:
    np.array([1, ni, nj, nk], dtype='i4').tofile(f)
    osd.grid.grd[..., 0].ravel(order='F').astype('f8').tofile(f)
with open('shock.q', 'wb') as f:
    np.array([1, ni, nj, nk], dtype='i4').tofile(f)
    np.array([shock.mach, 0.0, 0.0, 0.0], dtype='f8').tofile(f)   # mach, alpha, Re, time
    osd.flow.q[..., 0].ravel(order='F').astype('f8').tofile(f)

grid = GridIO('shock.x')
flow = FlowIO('shock.q')
grid.read_grid(data_type='f8')      # store_type='f4' halves the read time on large grids
flow.read_flow(data_type='f8')
grid.compute_metrics()              # Jacobians and inverse metrics for c-space work

print('blocks:', grid.nb, ' dims:', grid.ni[0], grid.nj[0], grid.nk[0])
print('bounds (m):', grid.grd_min[0], 'to', grid.grd_max[0])
print('freestream Mach from file:', flow.mach)
print('max round-trip error:', np.abs(grid.grd - osd.grid.grd).max())
```

```text
blocks: 1  dims: 40 60 5
bounds (m): [-0.005  0.     0.   ] to [0.005  0.03   0.0001]
freestream Mach from file: 7.6
max round-trip error: 0.0
```

Real grids are read the same way: point `GridIO` and `FlowIO` at an existing `.x` and `.q` pair, and pass `data_type='f4'` for single-precision files.

### Reduce particle tracks to an Eulerian field

`DataIO` reads the scattered particle tracks a run writes out, interpolates the flow to those points, removes outliers, interpolates both the flow and particle fields onto a structured mesh, and writes PLOT3D grid, fluid, and particle files. This is the stage that parallelizes over MPI. `DataIO.compute()` distributes the track files across ranks and gathers the result, so a production reduction over a large ensemble is launched under `mpiexec`. Because the whole of `compute()` goes through a communicator, it needs the `mpi` extra even when it runs on a single rank. The reduction stages underneath it are plain NumPy and SciPy.

`test/test_dataio_reduction.py` exercises the reduction on a tiny synthetic particle set built by the test fixtures. It needs no external data and no `mpiexec` launcher, and `pytest test/test_dataio_reduction.py -q` reports `8 passed` on a fresh clone with the `mpi` extra installed. Seven of those tests cover the reduction stages directly and run without `mpi4py` at all; the eighth drives the full `compute()` pipeline and so needs `mpi4py` present. See [Testing](#testing).

### A larger script

`main.py` in the repository root is a longer, research-scale version of examples 1, 3, and 4 combined: a 100 mm by 500 mm oblique-shock domain and no step cap. Its `n_concentration` defaults to 4 particles so a first run finishes in minutes; the published cloud used 1000, which `oblique_shock_response(n_concentration=1000)` reproduces. It is a real production run, not a quickstart, and it takes a long time and writes output files. Read it as a worked example rather than as something to run first.

## Architecture

lptlib is organized into four subpackages under `lptlib`.

`lptlib.io` handles file I/O. `GridIO` and `FlowIO` read and write PLOT3D grids and solutions and compute grid metrics, and `DataIO` performs the MPI-parallel Lagrangian-to-Eulerian reduction and PLOT3D export.

`lptlib.streamlines` carries the tracking pipeline. `Search` locates a point inside the curvilinear grid, `Interpolation` evaluates the flow state at that point, and `Integration` advances streamlines and particle paths, including the drag-model particle dynamics. `Streamlines` chains these three stages into a single call, and `StochasticModel`, `Particle`, and `SpawnLocations` seed and run large particle ensembles in parallel.

`lptlib.function` provides supporting computation, including `Variables` for derived quantities such as velocity, temperature, pressure, Mach number, and viscosity, along with plotting and timing helpers. Synthetic test cases such as `ObliqueShock`, `ObliqueShockData`, and `ObliqueShockAlignedData` live in `lptlib.test_cases`.

A typical run flows from a PLOT3D solution through search, interpolation, and integration to a set of particle tracks, then through the `DataIO` reduction to Eulerian PLOT3D fields ready for visualization or synthetic PIV imaging.

## Documentation

An overview of the modules, the public API, and worked examples is in [docs/index.md](docs/index.md). Deeper docstrings on the public classes and methods are available through Python's built-in `help()`.

## Testing

Install the test and `mpi` extras and run the suite from the repository root. `pytest` alone is not enough; several test modules import `parameterized`, and collection fails without it.

```bash
pip install -e ".[test,mpi]"
pytest test -v
```

The suite covers search, interpolation for steady and unsteady flow, integration, drag models, streamlines, the stochastic model, the PLOT3D readers and round trips, the DataIO reduction, plotting, and the MPI helpers. Continuous integration runs the same tests on Ubuntu and macOS for Python 3.10 through 3.13 on every push and pull request, with a coverage floor of 60 percent.

The `mpi` extra belongs in that install even though the MPI launcher tests are opt-in. One test drives `DataIO.compute()`, which goes through a communicator on a single rank, and it errors rather than skipping when `mpi4py` is absent. With `pip install -e ".[test]"` alone the suite reports one failure in `test/test_dataio_reduction.py`.

Two groups of tests do not run by default.

- **Data-backed tests skip.** Some tests need large PLOT3D grids and solutions from the original research cases. Those datasets are not published and are not part of this repository, so the tests that need them skip with a message naming the missing path. This is expected on a fresh clone, and the rest of the suite still runs and still covers the library. Nothing in the quickstart above needs them.
- **MPI tests are opt-in.** Set `LPTLIB_RUN_MPI=1` to enable the tests that launch MPI processes with `mpiexec -np 2`. They also need `mpiexec` on `PATH`, and they add a few minutes to the run.

```bash
LPTLIB_RUN_MPI=1 pytest test -v
```

Open MPI refuses to launch as root, so in a container running as root those tests also need `OMPI_ALLOW_RUN_AS_ROOT=1` and `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` in the environment. Neither is needed on a normal user account or on a GitHub-hosted runner.

Matplotlib is forced to a headless backend by the test configuration, so no display or `MPLBACKEND` setting is needed.

## How to cite

If lptlib supports your research, please cite both the software archive and the accompanying paper.

The versioned software archive is deposited on Zenodo.

> Kalagotla, D. (2026). *lptlib: A parallel Lagrangian particle tracking library for compressible CFD data and tracer-response analysis in optical velocimetry* [Software]. Zenodo. https://doi.org/10.5281/zenodo.22006302

A companion paper has been prepared for the Journal of Open Source Software. Citation details will be added once the review is complete. See `paper.md` for the current draft.

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff), which GitHub renders as a "Cite this repository" button.

## Support

If something is unclear or does not work, please open an issue at https://github.com/kalagotla/lptlib/issues.

- **Questions and usage help.** Open an issue and apply the `question` label, or start a thread in [GitHub Discussions](https://github.com/kalagotla/lptlib/discussions). Please include what you are trying to do, the snippet you ran, and the full output.
- **Bug reports and feature requests.** Open an issue. [CONTRIBUTING.md](CONTRIBUTING.md) lists what to include so the problem can be reproduced quickly.
- **Anything that does not belong in public.** Email the maintainer at dilipkalagotla@gmail.com.

Issues are the preferred channel, because the answer then helps the next person with the same question. Expect a response within a few working days.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for how to report bugs, set up a development environment, run the tests, and open a pull request. All participants are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

lptlib is distributed under the MIT License. See [LICENSE](LICENSE) for the full text.

## Acknowledgements

Developed by Dilip Kalagotla with Paul D. Orkwis in the Department of Aerospace Engineering and Engineering Mechanics at the University of Cincinnati. Thanks to Harpreet Chhabra for contributions to early test cases. The vectorized PLOT3D reader builds on NumPy.
