"""Synthetic, in-memory fixtures for the lptlib test suite.

The large PLOT3D grid and solution files that the historical tests depend on
are not tracked in the repository, so those tests skip on a clean checkout.
The helpers here build small, deterministic grids and flow fields entirely in
memory (or as tiny temporary PLOT3D files) so the core library logic is
exercised on every CI run rather than skipped.

Two families of helpers live here.

``write_plot3d_grid`` and ``write_plot3d_flow`` serialize small numpy arrays to
the exact little-endian, stream (no Fortran record markers) PLOT3D layout that
``GridIO.read_grid`` and ``FlowIO.read_flow`` consume. They let the I/O tests do
true round trips: write a known array, read it back, assert equality.

``make_oblique_shock_case`` wraps the library's own ``ObliqueShockData``
generator to produce a fully populated ``GridIO``/``FlowIO`` pair (grid metrics
computed, piecewise-constant pre/post-shock flow) without touching disk. That
single case drives the search, interpolation, integration, particle-path,
DataIO reduction, and stochastic-model tests.
"""

import numpy as np

from lptlib.test_cases import ObliqueShock, ObliqueShockData


def write_plot3d_grid(path, blocks, dtype="f4"):
    """Write a multiblock PLOT3D grid file from a list of block arrays.

    Each block is an array of shape ``(ni, nj, nk, 3)``. The on-disk layout is
    the number of blocks, then ``ni, nj, nk`` for every block, then the block
    coordinate data flattened in Fortran order. This mirrors exactly what
    ``GridIO.read_grid`` expects.
    """
    with open(path, "wb") as handle:
        np.array([len(blocks)], dtype="i4").tofile(handle)
        for block in blocks:
            np.array(block.shape[:3], dtype="i4").tofile(handle)
        for block in blocks:
            block.astype(dtype).flatten(order="F").tofile(handle)


def write_plot3d_flow(path, q_blocks, mach=2.0, alpha=0.0, rey=1.0e5,
                      time=1.0, dtype="f4"):
    """Write a multiblock PLOT3D solution (q) file from a list of block arrays.

    Each block is an array of shape ``(ni, nj, nk, 5)``. The layout is the
    number of blocks, then ``ni, nj, nk`` for every block, then for each block
    the four dimensionless header values followed by the block flow data in
    Fortran order. This mirrors what ``FlowIO.read_flow`` expects.
    """
    with open(path, "wb") as handle:
        np.array([len(q_blocks)], dtype="i4").tofile(handle)
        for q in q_blocks:
            np.array(q.shape[:3], dtype="i4").tofile(handle)
        for q in q_blocks:
            np.array([mach, alpha, rey, time], dtype=dtype).tofile(handle)
            q.astype(dtype).flatten(order="F").tofile(handle)


def make_oblique_shock_case(xpoints=12, ypoints=12, zpoints=5,
                            mach=2.0, deflection=8.0,
                            inlet_temperature=152.778, inlet_density=1.2663,
                            nx_max=15e-3, ny_max=15e-3, nz_max=1e-4,
                            shock_strength="weak"):
    """Build a small synthetic oblique-shock grid and flow field in memory.

    Returns the populated ``ObliqueShockData`` instance. Its ``grid`` attribute
    is a ``GridIO`` with coordinates, block sizes, bounds, and metrics filled
    in, and its ``flow`` attribute is a ``FlowIO`` with a piecewise-constant
    pre/post-shock ``q`` field. The grid is a uniform Cartesian mesh spanning
    ``[-nx_max, nx_max] x [0, ny_max] x [0, nz_max]`` with ``2*xpoints`` points
    in x. Everything is deterministic for fixed inputs.
    """
    shock = ObliqueShock()
    shock.mach = mach
    shock.deflection = deflection
    shock.compute()

    osd = ObliqueShockData()
    osd.nx_max = nx_max
    osd.ny_max = ny_max
    osd.nz_max = nz_max
    osd.inlet_temperature = inlet_temperature
    osd.inlet_density = inlet_density
    osd.xpoints = xpoints
    osd.ypoints = ypoints
    osd.zpoints = zpoints
    osd.oblique_shock = shock
    osd.shock_strength = shock_strength
    osd.create_grid()
    osd.create_flow()
    return osd
