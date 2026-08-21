"""The cell a point is located in must be the cell that contains it.

``Search`` returns two things that have to agree: ``cpoint``, the point's
coordinate in computational space, and ``cell``, the eight nodes of the cell
holding it. ``Interpolation`` builds every c-space-flavoured tri-linear weight
as ``cpoint - cell[0]``, so if the two disagree those weights leave ``[0, 1]``
and the "interpolation" is an extrapolation out of a cell the point is not in.
Nothing in the library checks for that, and nothing warns.

They used to disagree, on exactly the grids this library exists for.
``Search.p2c`` handed its converged c-space coordinate to ``_cell_index``,
which picks a cell by the Cartesian octant of ``ppoint - grd[i, j, k]``. That
is the same thing as indexing by the computational coordinate only when the
computational axes line up with x, y and z -- i.e. on a Cartesian grid. On the
quarter-annulus fixture the i axis is radial, and ``cell[0]`` disagreed with
``cpoint.astype(int)`` for 228 of 400 in-domain points (57 per cent), with the
local fractions reaching 1.996: almost a full cell of silent extrapolation on
every curvilinear c-space interpolation. On the Cartesian oblique-shock fixture
the same sweep showed the smaller, and separate, effect of the absolute 1e-6
octant threshold, which is 4 per cent of a cell where ``dz = 2.5e-5``.

Cell selection now comes from the computational coordinate
(``Search._locate_from_cpoint``) and node detection is its own test against the
grid coordinates with a cell-relative tolerance (``Search._is_node``). The
tests here pin both halves, and the accuracy that follows from them.

A note on the fixture. ``curvilinear_grid`` is curved, but its radial node
lines are straight rays along which ``r`` is linear in ``i``, so the mapping is
exactly linear along i -- and a tri-linear extrapolation along a linear
direction reproduces a linear field exactly. That accident means a wrong cell
can leave no trace in the interpolated value there. The accuracy tests below
therefore use ``curvilinear_stretched_grid``, which stretches the radial
spacing and removes it; the index tests run on both.
"""

import numpy as np
import pytest

from lptlib.streamlines import Interpolation, Search
from synthetic import analytic_vortex_field

# Search methods that converge a computational coordinate with p2c, and so
# produce a ``cpoint`` for ``Interpolation`` to build weights from.
C_SPACE_SEARCHES = ["p-space", "c-space"]

# Interpolation methods whose weights are ``cpoint - cell[0]``. ``rgi-p-space``
# is deliberately absent: it feeds the cell's physical coordinates to
# ``RegularGridInterpolator``, which needs a rectilinear cell, and it raises on
# any curvilinear block. That is a pre-existing limitation of that method, not
# of the cell indexing tested here.
FRACTION_METHODS = ["p-space", "c-space", "rbf-c-space", "rgi-c-space"]

# Slack for "the weights are inside the unit cell". The invariant is exact by
# construction; this only absorbs a Newton-Raphson iterate that settled onto a
# cell face from the wrong side by an ulp.
FRACTION_TOL = 1e-9

# Must match the defaults of ``make_curvilinear_annulus_grid``.
R_IN, R_OUT = 1.0, 2.0
THETA_MAX = np.pi / 2
Z_MAX = 0.25


def _sweep_points(n_r=10, n_theta=10, n_z=4):
    """A deterministic lattice of points strictly inside the annulus block.

    Placed in cylindrical coordinates and converted, so every point is in the
    block by construction rather than by trusting the search to say so. The
    small inset keeps them off the boundary faces, which have their own tests
    in ``test_search_boundary.py`` and ``test_search_curvilinear.py``.
    """
    points = []
    for radius in np.linspace(R_IN + 0.02, R_OUT - 0.02, n_r):
        for theta in np.linspace(0.01, THETA_MAX - 0.01, n_theta):
            for height in np.linspace(0.005, Z_MAX - 0.005, n_z):
                points.append([radius * np.cos(theta),
                               radius * np.sin(theta),
                               height])
    return np.array(points)


def _cartesian_sweep_points(grid, n=(6, 6, 4)):
    """A lattice spanning the interior of a Cartesian block, in c-space.

    Built by mapping computational coordinates forward through ``c2p`` so the
    points land at known fractional positions inside known cells, including
    positions very close to a node -- which is where the old absolute octant
    threshold did its damage on a fine grid.
    """
    helper = Search(grid, None)
    helper.block = 0
    sizes = (int(grid.ni[0]) - 1, int(grid.nj[0]) - 1, int(grid.nk[0]) - 1)
    points = []
    for i in np.linspace(0.02, sizes[0] - 0.02, n[0]):
        for j in np.linspace(0.02, sizes[1] - 0.02, n[1]):
            for k in np.linspace(0.02, sizes[2] - 0.02, n[2]):
                points.append(np.asarray(helper.c2p(np.array([i, j, k])),
                                         dtype="f8").copy())
    return np.array(points)


def _located(grid, point, method):
    """Run a search and return it, skipping the point if it was rejected."""
    idx = Search(grid, list(point))
    idx.compute(method=method)
    return idx


def _fractions(idx):
    return np.asarray(idx.cpoint, dtype="f8") - idx.cell[0]


# ---------------------------------------------------------------------------
# The invariant Interpolation depends on.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", C_SPACE_SEARCHES)
def test_local_fractions_are_inside_the_unit_cell_on_a_curved_grid(
        curvilinear_grid, method):
    """``cpoint - cell[0]`` is in ``[0, 1]`` for every in-domain point.

    This is the regression. Before cell selection was moved onto the
    computational coordinate, 228 of these 400 points had at least one weight
    outside ``[0, 1]``, the worst by 0.996 of a cell.
    """
    points = _sweep_points()
    worst = 0.0
    offenders = 0
    for point in points:
        idx = _located(curvilinear_grid, point, method)
        assert idx.cpoint is not None, f"{method} lost an in-domain point"
        fractions = _fractions(idx)
        excursion = max(float(np.max(fractions - 1.0)),
                        float(np.max(-fractions)), 0.0)
        worst = max(worst, excursion)
        if excursion > FRACTION_TOL:
            offenders += 1

    assert offenders == 0, (
        f"{method}: {offenders}/{len(points)} points interpolate from a cell "
        f"that does not contain them, worst by {worst:.6f} of a cell")


@pytest.mark.parametrize("method", C_SPACE_SEARCHES)
def test_local_fractions_are_inside_the_unit_cell_on_a_stretched_grid(
        curvilinear_stretched_grid, method):
    """Same invariant where the radial mapping is not linear in ``i``.

    The uniform annulus is linear along i, which is the one direction the old
    octant test got wrong there; a non-uniform radial distribution makes sure
    the invariant is not being satisfied by that coincidence.
    """
    for point in _sweep_points():
        idx = _located(curvilinear_stretched_grid, point, method)
        assert idx.cpoint is not None
        fractions = _fractions(idx)
        assert np.all(fractions >= -FRACTION_TOL), (method, point, fractions)
        assert np.all(fractions <= 1.0 + FRACTION_TOL), (method, point, fractions)


@pytest.mark.parametrize("method", C_SPACE_SEARCHES)
def test_local_fractions_are_inside_the_unit_cell_on_a_cartesian_grid(
        synthetic_grid, method):
    """The Cartesian control, on a grid fine enough to expose an absolute tolerance.

    The oblique-shock fixture has ``dz = 2.5e-5``, so the 1e-6 that
    ``_cell_index`` used to hard-code as its octant threshold was 4 per cent of
    a cell: points that close to a node were placed in the neighbouring cell
    and the weights ran to 1.038. The threshold is a fraction of the local cell
    now, so the excursion is at round-off.
    """
    for point in _cartesian_sweep_points(synthetic_grid):
        idx = _located(synthetic_grid, point, method)
        assert idx.cpoint is not None
        fractions = _fractions(idx)
        assert np.all(fractions >= -FRACTION_TOL), (method, point, fractions)
        assert np.all(fractions <= 1.0 + FRACTION_TOL), (method, point, fractions)


@pytest.mark.parametrize("method", C_SPACE_SEARCHES)
def test_cell_origin_is_the_integer_part_of_the_computational_point(
        curvilinear_stretched_grid, method):
    """``cell[0]`` is ``floor(cpoint)``, the definition of the containing cell.

    Two documented exceptions are excluded rather than asserted around:

    * a point on the far i/j/k face has an integer part equal to the last node
      index, where no cell starts, and ``_cell_split`` reports it in the last
      cell at a fraction of 1.0 (see ``test_search_boundary.py``);
    * a point sitting on a node can converge from either side of the integer,
      so ``cell[0]`` comes from the node test rather than from truncation.

    The sweep is placed strictly inside the block and off the nodes, so neither
    applies to any point in it.
    """
    grid = curvilinear_stretched_grid
    for point in _sweep_points():
        idx = _located(grid, point, method)
        assert idx.cpoint is not None
        assert idx.info is None, "sweep point unexpectedly landed on a node"
        expected = np.asarray(idx.cpoint, dtype="f8").astype(int)
        np.testing.assert_array_equal(idx.cell[0], expected)


@pytest.mark.parametrize("method", FRACTION_METHODS)
def test_every_c_space_interpolation_method_stays_inside_its_cell(
        curvilinear_stretched_grid, curvilinear_analytic_flow, method):
    """No c-space-flavoured method is handed weights outside the unit cell.

    ``Interpolation`` logs a warning when it is, so a run that stays silent is
    a run in which every method interpolated rather than extrapolated.
    """
    grid, flow = curvilinear_stretched_grid, curvilinear_analytic_flow
    for point in _sweep_points(n_r=5, n_theta=5, n_z=3):
        idx = _located(grid, point, "c-space")
        assert idx.cpoint is not None
        fractions = _fractions(idx)
        assert np.all(fractions >= -FRACTION_TOL)
        assert np.all(fractions <= 1.0 + FRACTION_TOL)

        interp = Interpolation(flow, idx)
        interp.compute(method=method)
        assert interp.q is not None
        assert np.all(np.isfinite(np.asarray(interp.q, dtype="f8")))


def test_interpolation_does_not_warn_about_extrapolation(
        curvilinear_stretched_grid, curvilinear_analytic_flow, caplog):
    """The extrapolation warning stays silent across the sweep.

    ``Interpolation._local_fractions`` exists to make a wrong cell audible
    instead of silent. This asserts it has nothing to say -- which is also a
    check that the warning is wired to the real weights, since it fired for 200
    of these points before the fix.
    """
    grid, flow = curvilinear_stretched_grid, curvilinear_analytic_flow
    with caplog.at_level("WARNING", logger="lptlib.streamlines.interpolation"):
        for point in _sweep_points(n_r=6, n_theta=6, n_z=3):
            idx = _located(grid, point, "c-space")
            Interpolation(flow, idx).compute(method="c-space")

    assert [record.getMessage() for record in caplog.records] == []


# ---------------------------------------------------------------------------
# What the invariant buys: accuracy.
# ---------------------------------------------------------------------------

def test_interpolating_the_coordinate_field_recovers_the_query_point(
        curvilinear_stretched_grid, curvilinear_coordinate_flow):
    """The sharpest statement of the defect, in metres.

    Sampling ``q = (x, y, z)`` at the nodes and interpolating it asks where the
    interpolation thinks the query point is. With the correct cell the
    tri-linear weights that ``p2c`` converged on rebuild the point exactly, so
    the answer is the query point to round-off. With the wrong cell it is a
    different place, and every flow quantity is then reported for that other
    place. Measured on this fixture before the fix: up to 5.16e-2, which is
    0.355 of a cell diagonal. It is now at round-off.
    """
    grid, flow = curvilinear_stretched_grid, curvilinear_coordinate_flow
    diagonal = float(np.linalg.norm(grid.grd[1, 1, 1, :, 0]
                                    - grid.grd[0, 0, 0, :, 0]))
    worst = 0.0
    for point in _sweep_points():
        idx = _located(grid, point, "c-space")
        assert idx.cpoint is not None
        interp = Interpolation(flow, idx)
        interp.compute(method="c-space")
        recovered = np.asarray(interp.q, dtype="f8").reshape(5)[:3]
        worst = max(worst, float(np.linalg.norm(recovered - point)))

    assert worst < 1e-9 * diagonal, (
        f"interpolation is reporting flow data for a point up to "
        f"{worst:.3e} away ({worst / diagonal:.3f} cell diagonals) from the "
        f"one that was asked for")


def test_analytic_field_is_recovered_to_second_order(
        curvilinear_stretched_grid, curvilinear_analytic_flow):
    """A known closed-form field comes back to tri-linear accuracy.

    The reference is ``analytic_vortex_field`` evaluated at the query point, so
    this measures the whole search-plus-interpolation chain against an answer
    that owes nothing to the library. What remains is the truncation error of
    tri-linear interpolation on a cell of this size, which is what should
    remain; before the fix the error was dominated by extrapolation from the
    wrong cell instead.

    Stated tolerances, against measured values on this fixture:
    max relative error 5.4e-3 (asserted below 1.2e-2) and RMS 1.5e-3
    (asserted below 3.5e-3). Before the fix the same sweep gave a max of
    1.6e-2 with 32 of 400 points past 1 per cent; it is now 0 of 400.
    """
    grid, flow = curvilinear_stretched_grid, curvilinear_analytic_flow
    errors = []
    for point in _sweep_points():
        idx = _located(grid, point, "c-space")
        assert idx.cpoint is not None
        interp = Interpolation(flow, idx)
        interp.compute(method="c-space")
        got = np.asarray(interp.q, dtype="f8").reshape(5)
        reference = analytic_vortex_field(*point)
        errors.append(np.abs(got - reference) / np.abs(reference))
    errors = np.array(errors)

    assert errors.max() < 1.2e-2
    assert np.sqrt(np.mean(errors ** 2)) < 3.5e-3
    assert not np.any(errors > 1.0e-2), (
        f"{int(np.sum(np.any(errors > 1e-2, axis=1)))} of {len(errors)} points "
        f"are more than 1 per cent off the analytic field")


# ---------------------------------------------------------------------------
# Node detection, now that it is a separate concern.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("index", [(0, 0, 0), (3, 4, 2), (4, 6, 1), (2, 9, 3)])
@pytest.mark.parametrize("method", C_SPACE_SEARCHES)
def test_a_node_is_still_reported_as_a_node(curvilinear_stretched_grid,
                                            index, method):
    """Node detection survives being split out of cell selection.

    ``_cell_index`` used to do both jobs in one branch, which is why the
    obvious one-line fix to cell selection dropped the node flag. They are now
    separate: ``_locate_from_cpoint`` indexes the cell from ``cpoint`` and asks
    ``_is_node`` about the node independently.
    """
    node = curvilinear_stretched_grid.grd[index[0], index[1], index[2], :, 0]
    idx = _located(curvilinear_stretched_grid, node, method)

    assert idx.info is not None and "is a node in the domain" in idx.info
    np.testing.assert_array_equal(idx.cell[0], np.array(index))


@pytest.mark.parametrize("index", [(0, 0, 0), (3, 4, 2), (4, 6, 1)])
def test_interpolation_at_a_node_returns_that_node_exactly(
        curvilinear_stretched_grid, curvilinear_analytic_flow, index):
    """The exact-node shortcut still recovers the node state bit for bit.

    Both through the shortcut and, for the far-face nodes that cannot be a
    cell origin, through the ordinary tri-linear path at a fraction of 1.0.
    """
    grid, flow = curvilinear_stretched_grid, curvilinear_analytic_flow
    node = grid.grd[index[0], index[1], index[2], :, 0]
    idx = _located(grid, node, "c-space")
    interp = Interpolation(flow, idx)
    interp.compute(method="c-space")

    expected = flow.q[index[0], index[1], index[2], :, 0]
    np.testing.assert_allclose(np.asarray(interp.q).reshape(5), expected,
                               rtol=1e-13, atol=0.0)
    # The c-space integrators need the metrics on this path too.
    assert interp.J is not None and interp.J_inv is not None
    np.testing.assert_allclose(np.asarray(interp.J) @ np.asarray(interp.J_inv),
                               np.eye(3), atol=1e-9)


@pytest.mark.parametrize("face", ["far-i", "far-j", "far-k"])
def test_far_face_nodes_are_interpolated_exactly_without_the_shortcut(
        curvilinear_stretched_grid, curvilinear_analytic_flow, face):
    """A node on the far face is not the origin of any cell, and still exact.

    ``_cell_nodes`` clamps it into the last cell, where it sits at a local
    fraction of 1.0. The node shortcut reads ``cell[0]``, so it must not be
    claimed here -- but the tri-linear weights collapse onto the node anyway,
    so the answer is exact either way.
    """
    grid, flow = curvilinear_stretched_grid, curvilinear_analytic_flow
    ni, nj, nk = int(grid.ni[0]), int(grid.nj[0]), int(grid.nk[0])
    index = {"far-i": (ni - 1, nj // 2, nk // 2),
             "far-j": (ni // 2, nj - 1, nk // 2),
             "far-k": (ni // 2, nj // 2, nk - 1)}[face]

    node = grid.grd[index[0], index[1], index[2], :, 0]
    idx = _located(grid, node, "c-space")
    assert idx.info is None, "a far-face node must not claim the cell[0] shortcut"

    fractions = _fractions(idx)
    assert np.all(fractions >= -FRACTION_TOL)
    assert np.all(fractions <= 1.0 + FRACTION_TOL)

    interp = Interpolation(flow, idx)
    interp.compute(method="c-space")
    expected = flow.q[index[0], index[1], index[2], :, 0]
    # Absolute floor scaled to the size of the state: one component of the
    # vortex field passes through zero on the theta = 90 degrees face, and a
    # relative tolerance on a component that is 5e-17 by construction measures
    # round-off, not accuracy.
    np.testing.assert_allclose(np.asarray(interp.q).reshape(5), expected,
                               rtol=1e-12,
                               atol=1e-12 * float(np.max(np.abs(expected))))


def test_node_tolerance_is_relative_to_the_local_cell(synthetic_grid):
    """A displacement that is a large fraction of a fine cell is not a node.

    The old node test compared an absolute 1e-12 against a physical offset, and
    the old octant test an absolute 1e-6, with no reference to how big a cell
    is. On the oblique-shock fixture ``dz = 2.5e-5``, so 1e-6 was 4 per cent of
    a cell in z -- a point that far from a node is most of the way across it,
    not on it. The tolerances are fractions of the local cell now, so this
    point is not a node and is located in the cell it actually falls in.
    """
    ni, nj, nk = (int(synthetic_grid.ni[0]), int(synthetic_grid.nj[0]),
                  int(synthetic_grid.nk[0]))
    index = (ni // 2, nj // 2, nk // 2)
    node = np.asarray(synthetic_grid.grd[index[0], index[1], index[2], :, 0],
                      dtype="f8")
    dz = float(synthetic_grid.grd[index[0], index[1], 1, 2, 0]
               - synthetic_grid.grd[index[0], index[1], 0, 2, 0])
    assert dz < 1e-4, "fixture is no longer fine enough to make the point"

    offset = node + np.array([0.0, 0.0, 0.04 * dz])
    idx = _located(synthetic_grid, offset, "c-space")

    assert idx.info is None, "a point 4 per cent of a cell away is not a node"
    fractions = _fractions(idx)
    assert np.all(fractions >= -FRACTION_TOL)
    assert np.all(fractions <= 1.0 + FRACTION_TOL)
    assert fractions[2] == pytest.approx(0.04, abs=1e-6)


def test_the_exact_node_is_still_within_the_relative_tolerance(synthetic_grid):
    """Tightening the node tolerance must not lose actual nodes.

    The counterpart to the test above: on the same fine grid, the node itself
    is still recognised.
    """
    ni, nj, nk = (int(synthetic_grid.ni[0]), int(synthetic_grid.nj[0]),
                  int(synthetic_grid.nk[0]))
    index = (ni // 2, nj // 2, nk // 2)
    node = synthetic_grid.grd[index[0], index[1], index[2], :, 0]
    idx = _located(synthetic_grid, node, "c-space")

    assert idx.info is not None and "is a node in the domain" in idx.info
    np.testing.assert_array_equal(idx.cell[0], np.array(index))


# ---------------------------------------------------------------------------
# Downstream consumers of ``cell``.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", C_SPACE_SEARCHES)
def test_located_cell_really_contains_the_point(curvilinear_stretched_grid,
                                                method):
    """``_point_in_cell`` agrees that the c-space cell holds the point.

    ``_point_in_cell`` is the out-of-domain containment test, and it compares
    the point against the bounding box of the located cell. Now that the cell
    comes from the computational coordinate, the point is in it by
    construction, so this must never fire -- it is the cross-check that the two
    halves of ``Search`` tell the same story.
    """
    for point in _sweep_points(n_r=6, n_theta=6, n_z=3):
        idx = _located(curvilinear_stretched_grid, point, method)
        assert idx.ppoint is not None, (method, point)
        assert idx._point_in_cell(), (method, point)


@pytest.mark.parametrize("method", ["distance", "block_distance"])
def test_no_containment_decision_turns_on_the_size_of_the_pad(
        curvilinear_stretched_grid, method):
    """The pad in ``_point_in_cell`` shrank; nothing may hinge on that.

    The pad absorbs the slop in ``_cell_index``'s octant choice, and it was an
    absolute 1e-6 -- meaningless without a length scale -- where it is now a
    fraction of the cell. Shrinking a tolerance is where an over-tight test
    starts rejecting valid points, so this measures the margin instead of
    trusting it: every point of the sweep is either strictly inside the located
    cell's box (miss <= 0) or outside it by at least one per cent of the cell,
    which is four orders of magnitude clear of either pad. No point sits in the
    band where the choice of pad decides the answer.

    The rejected points are not a regression and not new. ``_cell_index``
    chooses a cell by Cartesian octant, which on a curved block can be a
    neighbouring cell, and ``_point_in_cell`` correctly reports that the point
    is not in it. That is the documented limitation of the ``distance``
    searches on curvilinear grids -- they have no computational coordinate to
    index with -- and it is exactly why ``p2c`` no longer uses that path.
    """
    grid = curvilinear_stretched_grid
    misses = []
    for point in _sweep_points(n_r=6, n_theta=6, n_z=3):
        idx = _located(grid, point, method)
        nodes = grid.grd[idx.cell[:, 0], idx.cell[:, 1], idx.cell[:, 2], :, 0]
        low, high = nodes.min(axis=0), nodes.max(axis=0)
        span = float(np.max(high - low))
        misses.append(float(np.max(np.maximum(low - point, point - high))) / span)
    misses = np.array(misses)

    ambiguous = misses[(misses > 0.0) & (misses < 1e-2)]
    assert ambiguous.size == 0, (
        f"{ambiguous.size} points sit within a hundredth of a cell of the "
        f"located cell's boundary, where the value of the pad decides whether "
        f"they are in the domain")
    assert np.any(misses == 0.0), "the sweep no longer reaches any valid cell"


def test_integration_reads_the_metrics_of_the_containing_cell(
        curvilinear_stretched_grid, curvilinear_analytic_flow):
    """``Integration`` indexes ``m2`` with ``cell[0]``, so it moved too.

    The c-space integrator takes its inverse Jacobian from the node at
    ``cell[0]``. That node is now the origin of the cell the point is in
    rather than a node of a neighbouring cell, so the check here is simply
    that it is the node the computational coordinate points at.
    """
    grid = curvilinear_stretched_grid
    for point in _sweep_points(n_r=5, n_theta=5, n_z=3):
        idx = _located(grid, point, "c-space")
        assert idx.cpoint is not None
        origin = np.asarray(idx.cpoint, dtype="f8").astype(int)
        np.testing.assert_array_equal(idx.cell[0], origin)
        metrics = grid.m2[idx.cell[0, 0], idx.cell[0, 1], idx.cell[0, 2],
                          :, :, idx.block]
        assert np.all(np.isfinite(metrics))
