"""Synthetic, in-memory fixtures for the lptlib test suite.

The large PLOT3D grid and solution files that the historical tests depend on
are not tracked in the repository, so those tests skip on a clean checkout.
The helpers here build small, deterministic grids and flow fields entirely in
memory (or as tiny temporary PLOT3D files) so the core library logic is
exercised on every CI run rather than skipped.

Three families of helpers live here.

``write_plot3d_grid`` and ``write_plot3d_flow`` serialize small numpy arrays to
the exact little-endian, stream (no Fortran record markers) PLOT3D layout that
``GridIO.read_grid`` and ``FlowIO.read_flow`` consume. They let the I/O tests do
true round trips: write a known array, read it back, assert equality.

``make_oblique_shock_case`` wraps the library's own ``ObliqueShockData``
generator to produce a fully populated ``GridIO``/``FlowIO`` pair (grid metrics
computed, piecewise-constant pre/post-shock flow) without touching disk. That
single case drives the search, interpolation, integration, particle-path,
DataIO reduction, and stochastic-model tests.

``make_curvilinear_annulus_grid`` builds a body-fitted quarter-annulus block.
Every other fixture here is Cartesian, which means each block coincides with
its own axis-aligned bounding box; the annulus deliberately breaks that so the
search can be tested against points that lie inside a block's bounding box but
outside the block itself. Its ``radial_stretch`` option additionally breaks the
one accidental linearity the uniform annulus retains, which is what makes the
cell-indexing tests able to see a wrong cell at all.

``analytic_vortex_field``, ``make_analytic_flow`` and ``make_coordinate_flow``
attach known fields to any grid, so interpolation can be checked against an
answer that is known in closed form rather than against another interpolation.
"""

import numpy as np

from lptlib.io.plot3dio import FlowIO, GridIO
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


def make_curvilinear_annulus_grid(nr=9, ntheta=13, nz=5,
                                  r_in=1.0, r_out=2.0,
                                  theta_max=np.pi / 2, z_max=0.25,
                                  radial_stretch=1.0):
    """Build a curvilinear quarter-annulus block as a populated ``GridIO``.

    The oblique-shock fixture above is a uniform Cartesian mesh, so each of its
    blocks *is* its own axis-aligned bounding box. That makes it blind to a
    whole class of search behavior: ``Search._find_block`` locates a point by
    testing it against the block bounding box, which on a Cartesian block is an
    exact containment test, so no point that is outside the block ever reaches
    the cell search.

    This fixture breaks that coincidence. The block is a body-fitted sector of
    an annulus -- node ``(i, j, k)`` sits at radius ``r_i``, angle
    ``theta_j`` and height ``z_k``, so the ``i`` axis runs radially, the ``j``
    axis runs around the arc, and the block is curved. Its bounding box is
    ``[0, r_out] x [0, r_out] x [0, z_max]``, which strictly contains two large
    regions that are *not* in the block:

    * the hole inside the inner radius, e.g. ``(0.3, 0.3, z)`` at ``r = 0.42``;
    * the corner beyond the outer radius, e.g. ``(1.9, 1.9, z)`` at
      ``r = 2.69``.

    Those points pass ``_find_block`` and reach the cell search, which is
    exactly the case a Cartesian fixture cannot produce. See
    ``test_search_curvilinear.py``.

    The mapping is smooth and has a strictly positive Jacobian everywhere
    (``J`` is proportional to ``r``, bounded away from zero because
    ``r_in > 0``), so ``GridIO.compute_metrics`` produces finite ``m1``, ``m2``
    and ``J`` and the Newton-Raphson searches in ``Search.p2c`` are well posed.

    Everything is deterministic for fixed inputs and nothing touches disk.

    Parameters
    ----------
    nr, ntheta, nz : int
        Node counts along the radial, circumferential and axial directions.
    r_in, r_out : float
        Inner and outer radii of the sector. ``r_in`` must be positive to keep
        the mapping non-degenerate.
    theta_max : float
        Angular extent of the sector, in radians, starting from ``theta = 0``.
    z_max : float
        Height of the extrusion.
    radial_stretch : float
        Exponent applied to the normalized radial coordinate, so that node
        ``i`` sits at ``r_in + (r_out - r_in) * (i/(nr-1))**radial_stretch``.
        The default of 1.0 gives the uniform spacing every existing test
        assumes.

        A value other than 1.0 matters for one specific reason. With uniform
        spacing the radial node lines are straight rays *and* ``r`` is linear
        in ``i``, so the mapping is exactly linear along the i axis. That
        accident hides errors: interpolating from the wrong cell along i, at a
        local fraction outside ``[0, 1]``, still reproduces a linear field
        exactly, so a misindexed cell can be invisible in the interpolated
        value even while the index is plainly wrong. Stretching the radius
        removes the accident and the misindexing shows up as a real error in
        the interpolated field -- see ``test_curvilinear_cell_indexing.py``.

    Returns
    -------
    GridIO
        With ``nb``, ``ni``, ``nj``, ``nk``, ``grd``, ``grd_min``, ``grd_max``,
        ``m1``, ``m2`` and ``J`` populated, matching what ``read_grid`` plus
        ``compute_metrics`` would produce for a real PLOT3D file.
    """
    if r_in <= 0.0:
        raise ValueError("r_in must be positive; a zero inner radius collapses "
                         "the mapping and makes the Jacobian singular.")

    if radial_stretch <= 0.0:
        raise ValueError("radial_stretch must be positive; a non-positive "
                         "exponent does not give a monotonic radial "
                         "distribution.")

    grid = GridIO("<synthetic-curvilinear-annulus>")
    _s = np.linspace(0.0, 1.0, nr)
    _r = r_in + (r_out - r_in) * _s ** radial_stretch
    _theta = np.linspace(0.0, theta_max, ntheta)
    _z = np.linspace(0.0, z_max, nz)
    _rr, _tt, _zz = np.meshgrid(_r, _theta, _z, indexing="ij")

    grid.nb = 1
    grid.ni = np.array([nr], dtype="i4")
    grid.nj = np.array([ntheta], dtype="i4")
    grid.nk = np.array([nz], dtype="i4")
    grid.grd = np.stack(((_rr * np.cos(_tt))[..., None],
                         (_rr * np.sin(_tt))[..., None],
                         _zz[..., None]), axis=3)
    grid.grd_min = grid.grd[..., 0].min(axis=(0, 1, 2)).reshape(1, 3)
    grid.grd_max = grid.grd[..., 0].max(axis=(0, 1, 2)).reshape(1, 3)

    GridIO.compute_metrics(grid)
    return grid


def analytic_vortex_field(x, y, z):
    """A smooth, closed-form ``q`` state as a function of physical position.

    Five components in PLOT3D order (rho, rho*u, rho*v, rho*w, e) built from a
    free-vortex-like flow around the annulus axis. Every component varies
    non-linearly with radius, which is the point: a field that is linear in
    ``x``, ``y`` and ``z`` is reproduced exactly by tri-linear extrapolation
    from a neighboring cell along a straight grid line, so a linear field
    cannot tell a correctly indexed cell from a wrong one. This one can.

    The inputs may be scalars or arrays of any matching shape; the return has
    one extra trailing axis of length 5.
    """
    x, y, z = np.asarray(x, dtype="f8"), np.asarray(y, dtype="f8"), np.asarray(z, dtype="f8")
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return np.stack([1.0 + 1.0 / r,                       # rho
                     -np.sin(theta) / r,                  # rho * u
                     np.cos(theta) / r,                   # rho * v
                     0.10 + 0.20 * z,                     # rho * w
                     2.0 + 0.5 / r ** 2 + 0.3 * np.cos(2.0 * theta)],
                    axis=-1)


def _empty_flow(grid, name):
    """A ``FlowIO`` carrying ``grid``'s block sizes and dimensionless header."""
    flow = FlowIO(name)
    flow.nb = grid.nb
    flow.ni, flow.nj, flow.nk = grid.ni, grid.nj, grid.nk
    flow.mach, flow.alpha, flow.rey, flow.time = 2.0, 0.0, 1.0e5, 1.0
    return flow


def _block_coordinates(grid):
    """The single block's node coordinates, trimmed to its own extent."""
    ni, nj, nk = int(grid.ni[0]), int(grid.nj[0]), int(grid.nk[0])
    return ni, nj, nk, grid.grd[:ni, :nj, :nk, :, 0]


def make_analytic_flow(grid, field=analytic_vortex_field):
    """Sample ``field`` at every node of a single-block ``grid``.

    Returns a ``FlowIO`` whose ``q`` is what ``read_flow`` would have produced
    for a solution file holding that field, so ``Interpolation`` can be checked
    against ``field`` evaluated at the query point.
    """
    flow = _empty_flow(grid, "<synthetic-analytic-field>")
    _, _, _, grd = _block_coordinates(grid)
    flow.q = field(grd[..., 0], grd[..., 1], grd[..., 2])[..., None]
    return flow


def make_coordinate_flow(grid):
    """A ``q`` field whose first three components are the node coordinates.

    Interpolating it answers the sharpest question that can be asked of a
    search-plus-interpolation pair: *where does the interpolation think the
    query point is?* When the cell holding the point is indexed correctly, the
    tri-linear weights that ``Search.p2c`` converged on reconstruct the point
    itself, so the answer comes back equal to the query point to round-off.
    When the cell is wrong the answer is displaced, and the size of the
    displacement is the interpolation's positional error in meters rather than
    in units of some particular flow variable.

    The last two components are zero; nothing reads them.
    """
    flow = _empty_flow(grid, "<synthetic-coordinate-field>")
    ni, nj, nk, grd = _block_coordinates(grid)
    q = np.zeros((ni, nj, nk, 5, 1))
    q[..., 0, 0], q[..., 1, 0], q[..., 2, 0] = grd[..., 0], grd[..., 1], grd[..., 2]
    flow.q = q
    return flow
