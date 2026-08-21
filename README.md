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

- **Vectorized PLOT3D input and output.** Each grid or flow file is read in a single buffered `numpy.fromfile` call and reconstructed into multi-block arrays through strided slicing and Fortran-order reshaping. This removes the per-point Python overhead that dominates naive readers and was designed to be competitive with a compiled Fortran PLOT3D reader on the same files. On a 2.04 million-point four-block grid the strided read is about 7 times faster than an equivalent naive Python reader and stays within about a quarter of a compiled Fortran reader, on the same order of magnitude. The comparison is reproducible with a single command in [`benchmarks/`](benchmarks/README.md).
- **MPI-parallel Lagrangian-to-Eulerian reduction.** Scattered particle tracks are binned and interpolated back onto a user-defined structured mesh in parallel with MPI, then exported as PLOT3D grid, fluid, and particle files. This produces the PIV-like Eulerian fields that a downstream synthetic-imaging tool consumes to render frames.
- **A twelve-model drag suite behind one argument.** Stokes, Oseen, Schiller-Naumann, Melling, Cunningham, Henderson, Loth, Subramaniam-Balachandar, and Tedeschi closures, together with a rigid-sphere standard-drag curve and a zero-drag mode for fluid tracers, are all selectable through a single `drag_model` argument. Tracer response can be studied as a function of particle size, density, and drag closure without changing anything else.

## Full feature list

At a glance, lptlib provides:

- PLOT3D input and output for multi-block, structured, curvilinear grids and flow solutions, in single and double precision, including unsteady flow sequences and two-dimensional Fortran-record planes.
- Point location in curvilinear grids with physical-space and computational-space search, and conversion between the two spaces.
- Interpolation of flow variables to arbitrary points, with physical-space, computational-space, radial-basis-function, and regular-grid options, plus an analytic oblique-shock interpolant for controlled test cases.
- Streamline and particle-path integration with second- and fourth-order Runge-Kutta schemes in physical and computational space, unsteady variants, and adaptive time stepping.
- The twelve-model spherical-particle drag suite described above, with air viscosity evaluated through Sutherland or Keyes laws.
- Stochastic seeding of particle-size distributions and spawn locations, with parallel execution over many particles through multiprocessing, thread pools, and MPI.
- Derived-variable computation for velocity, temperature, pressure, Mach number, and viscosity.
- The MPI-parallel Lagrangian-to-Eulerian reduction that writes PLOT3D fluid and particle fields for visualization and for synthetic PIV imaging.

## Installation

lptlib is on PyPI.

```bash
pip install lptlib
```

Python 3.10 or newer is required. Core dependencies are `numpy`, `scipy`, `matplotlib`, `pandas`, `seaborn`, `tqdm`, `scikit-learn`, `h5py`, and `mpi4py`. The `mpi4py` dependency needs a system MPI implementation. On Debian or Ubuntu install one with `sudo apt-get install libopenmpi-dev openmpi-bin`, and on macOS with `brew install open-mpi`.

## Quickstart

### Read a PLOT3D grid and flow

```python
from lptlib import GridIO, FlowIO

grid = GridIO('data/plate_data/plate.sp.x')
flow = FlowIO('data/plate_data/sol-0000010.q')
grid.read_grid()
flow.read_flow()
grid.compute_metrics()
```

### Extract a single streamline

The high-level `Streamlines` orchestrator locates the seed point, interpolates, and integrates in one call.

```python
from lptlib import Streamlines

sl = Streamlines('data/plate_data/plate.sp.x',
                 'data/plate_data/sol-0000010.q',
                 point=[0.5, 0.5, 0.01])
sl.compute(method='p-space')
coords = sl.streamline   # list of points along the path
```

### Track inertial particles through a synthetic oblique shock

`main.py` is a complete, runnable example. It synthesizes an oblique-shock grid and flow, seeds a particle-size distribution, and launches an adaptive parallel simulation.

```python
from lptlib import ObliqueShock, ObliqueShockData
from lptlib import StochasticModel, Particle, SpawnLocations

shock = ObliqueShock()
shock.mach = 7.6
shock.deflection = 20          # degrees
shock.compute()

osd = ObliqueShockData()
osd.oblique_shock = shock
osd.create_grid()
osd.create_flow()

particle = Particle()
particle.min_dia = particle.max_dia = particle.mean_dia = 1.94e-6
particle.std_dia = 0
particle.density = 950
particle.n_concentration = 1000
particle.distribution = 'gaussian'
particle.compute_distribution()

spawn = SpawnLocations(particle)
spawn.x_min, spawn.z_min = -50e-3, 5e-5
spawn.y_min, spawn.y_max = 0, osd.ny_max
spawn.compute()

model = StochasticModel(particle, spawn, grid=osd.grid, flow=osd.flow)
model.method = 'adaptive-ppath'
model.interpolation = 'simple_oblique_shock'
model.drag_model = 'loth'
model.time_step = 1e-10
data = model.multi_process()
```

Run the packaged example directly with:

```bash
python main.py
```

### Reduce particle tracks to an Eulerian field

`DataIO` reads scattered particle tracks, interpolates the flow to those points, removes outliers, interpolates both the flow and particle fields onto a structured mesh, and writes PLOT3D grid, fluid, and particle files. A minimal, runnable example lives in `test/test_dataio.py`.

## Architecture

lptlib is organized into three subpackages under `lptlib`.

`lptlib.io` handles file I/O. `GridIO` and `FlowIO` read and write PLOT3D grids and solutions and compute grid metrics, and `DataIO` performs the MPI-parallel Lagrangian-to-Eulerian reduction and PLOT3D export.

`lptlib.streamlines` carries the tracking pipeline. `Search` locates a point inside the curvilinear grid, `Interpolation` evaluates the flow state at that point, and `Integration` advances streamlines and particle paths, including the drag-model particle dynamics. `Streamlines` chains these three stages into a single call, and `StochasticModel`, `Particle`, and `SpawnLocations` seed and run large particle ensembles in parallel.

`lptlib.function` provides supporting computation, including `Variables` for derived quantities such as velocity, temperature, pressure, Mach number, and viscosity, along with plotting and timing helpers. Synthetic test cases such as `ObliqueShock` and `ObliqueShockData` live in `lptlib.test_cases`.

A typical run flows from a PLOT3D solution through search, interpolation, and integration to a set of particle tracks, then through the `DataIO` reduction to Eulerian PLOT3D fields ready for visualization or synthetic PIV imaging.

## Documentation

An overview of the modules, the public API, and worked examples is in [docs/index.md](docs/index.md). Deeper docstrings on the public classes and methods are available through Python's built-in `help()`.

## Testing

Install the test dependency and run the suite from the repository root.

```bash
pip install pytest
pytest test -v
```

The suite covers search, interpolation for steady and unsteady flow, integration, drag models, streamlines, the DataIO reduction, plotting, and the MPI helpers. Continuous integration runs the same tests on Python 3.10, 3.11, and 3.12 for every push and pull request.

## How to cite

If lptlib supports your research, please cite both the software archive and the accompanying paper.

The versioned software archive is deposited on Zenodo.

> Kalagotla, D. (2026). *lptlib: A parallel Lagrangian particle tracking library for compressible CFD data and tracer-response analysis in optical velocimetry* [Software]. Zenodo. https://doi.org/10.5281/zenodo.22006302

A companion paper has been prepared for the Journal of Open Source Software. Citation details will be added once the review is complete. See `paper.md` for the current draft.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for how to report bugs, set up a development environment, run the tests, and open a pull request.

## License

lptlib is distributed under the MIT License. See [LICENSE](LICENSE) for the full text.

## Acknowledgements

Developed by Dilip Kalagotla with Paul D. Orkwis in the Department of Aerospace Engineering and Engineering Mechanics at the University of Cincinnati. Thanks to Harpreet Chhabra for contributions to early test cases. The vectorized PLOT3D reader builds on NumPy.
