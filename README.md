# lptlib (Lagrangian Particle Tracking Library)
### Previously project-arrakis

Python based particle tracking algorithms for CFD data

A highly parallelized set of Lagrangian Particle Tracking (LPT) algorithms based on Python to post-process steady and unsteady CFD data. An advanced programming interface (API) is developed for uncertainty quantification of optical velocimetry data.

## Installation

```bash
pip install lptlib
```

Python >= 3.10 is required. Core dependencies include `numpy`, `scipy`, `matplotlib`, `pandas`, `seaborn`, `tqdm`, `mpi4py`, and `scikit-learn`.

## Overview

lptlib provides building blocks to:
- Read Plot3D grid/flow data (`GridIO`, `FlowIO`)
- Locate points in structured curvilinear grids (`Search`)
- Interpolate flow variables at arbitrary locations (`Interpolation`)
- Integrate particle paths and streamlines with multiple schemes (`Integration`, `Streamlines`)
- Run stochastic, parallel particle simulations (`StochasticModel`, `Particle`, `SpawnLocations`)
- Compute derived variables like velocity, temperature, pressure, Mach, viscosity (`Variables`)
- Post-process LPT outputs to Eulerian fields and Plot3D files (`DataIO`)

## Quickstart

### Read Plot3D grid and flow

```python
from lptlib.io.plot3dio import GridIO, FlowIO

grid = GridIO('path/to/grid.sp.x')
flow = FlowIO('path/to/sol-0000010.q')
grid.read_grid()
flow.read_flow()
grid.compute_metrics()
```

### Interpolate and integrate a streamline

```python
import numpy as np
from lptlib.streamlines.search import Search
from lptlib.streamlines.interpolation import Interpolation
from lptlib.streamlines.integration import Integration

point = np.array([0.1, 0.05, 0.0])
idx = Search(grid, point)
idx.compute(method='p-space')

interp = Interpolation(flow, idx)
interp.compute(method='p-space')

intg = Integration(interp)
new_point, u = intg.compute(method='pRK4', time_step=1e-3)
```

### One-shot streamline extraction

```python
from lptlib.streamlines.streamlines import Streamlines

sl = Streamlines('path/to/grid.sp.x', 'path/to/sol-0000010.q', [0.1, 0.05, 0.0])
sl.compute(method='p-space')
coords = sl.streamline  # list of points
```

### Stochastic parallel run (oblique shock example)

The repository includes a fully working example in `main.py` that generates an oblique shock test case and launches an adaptive particle tracking simulation in parallel:

```bash
python main.py
```

Key objects used in the example:
- `ObliqueShock`, `ObliqueShockData` to synthesize grid/flow for a controlled shock case
- `Particle`, `SpawnLocations` to define particle size distribution and seed locations
- `StochasticModel` to run many particles in parallel with adaptive time stepping

## DataIO pipeline (Lagrangian → Eulerian)

`DataIO` reads scattered particle tracks (as `.npy` per particle), interpolates flow to those points, removes outliers, then interpolates both flow and particle fields onto a structured mesh and writes Plot3D outputs for visualization and downstream tools.

Essential steps:
1. Scatter interpolation of flow to particle locations (MPI-parallel)
2. Outlier removal and caching of intermediate `.npy` files under `dataio/`
3. Grid interpolation to a user-defined mesh
4. Export to Plot3D: `mgrd_to_p3d.x`, `mgrd_to_p3d_fluid.q`, `mgrd_to_p3d_particle.q`

See `test/test_dataio.py` for a minimal, runnable example.

## Core API

- `lptlib.io.plot3dio.GridIO`
  - `read_grid(data_type='f4')`, `compute_metrics()`, `mgrd_to_p3d(...)`
- `lptlib.io.plot3dio.FlowIO`
  - `read_flow(data_type='f4')`, `read_unsteady_flow(...)`, `mgrd_to_p3d(...)`, `read_formatted_txt(...)`
- `lptlib.streamlines.Search`
  - `compute(method=...)`, `p2c(ppoint)`, `c2p(cpoint)`
- `lptlib.streamlines.Interpolation`
  - `compute(method=...)` with options: `p-space`, `c-space`, `rbf-*`, `rgi-*`, `simple_oblique_shock`
- `lptlib.streamlines.Integration`
  - `compute(method=..., time_step=...)` with `pRK2/4`, `cRK2/4`, unsteady variants
  - `compute_ppath(...)` for particle dynamics with drag models (`stokes`, `loth`, etc.)
- `lptlib.streamlines.Streamlines`
  - High-level orchestrator: `compute(method=...)`, exposes `streamline`, `fvelocity`, `svelocity`, `time`
- `lptlib.streamlines.StochasticModel`
  - Parallel execution over many particles: `multi_process()`, `multi_thread()`, `mpi_run()`, `serial()`
- `lptlib.function.Variables`
  - `compute_velocity()`, `compute_temperature()`, `compute_pressure()`, `compute_mach()`, `compute_viscosity()`
- `lptlib.io.DataIO`
  - `compute()` end-to-end Lagrangian→Eulerian conversion and Plot3D export

## Testing

Run the test suite from the repo root:

```bash
pytest -q
```

Tests cover search, interpolation (steady/unsteady), integration, DataIO, streamlines, plotting, and MPI helpers.

## License

Distributed under MIT AND (Apache-2.0 OR BSD-2-Clause). See `LICENSE`.