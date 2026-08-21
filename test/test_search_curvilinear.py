"""Out-of-domain behaviour of ``Search`` on a genuinely curvilinear block.

``Search._find_block`` decides whether a point is in the grid at all by testing
it against each block's axis-aligned bounding box. On a Cartesian block that is
an exact containment test, because the block fills its own bounding box. Every
other synthetic fixture in this suite is Cartesian, so no test here could ever
present the search with a point that passes ``_find_block`` and is still
outside the block.

That matters because of the far-face clamp in ``Search._cell_nodes`` (see
``test_search_boundary.py``). The clamp is required -- without it a point
landing exactly on the maximum i/j/k face asked for a cell starting at the last
node and raised ``IndexError``. But it also guarantees that the index-range
tests the ``distance`` searches used to make,
``max(cell[:, 0]) > ni - 1`` and friends, can never fire again. On a Cartesian
grid that loss is invisible: nothing outside the block gets that far. On a
curvilinear grid it is not, and the searches would locate a point that is
outside the block in the nearest boundary cell and silently extrapolate.

The fixture is a quarter-annulus sector, ``r`` in ``[1, 2]``, ``theta`` in
``[0, 90]`` degrees. Its bounding box is ``[0, 2] x [0, 2] x [0, z_max]``, so
the hole inside the inner radius and the corner beyond the outer radius are
both inside the bounding box and outside the block.

These tests pin down all four search methods against four classes of point:
well inside, exactly on each of the six boundary faces, inside the bounding box
but outside the block, and outside the bounding box entirely.
"""

import numpy as np
import pytest

from lptlib.streamlines import Search

METHODS = ["distance", "block_distance", "p-space", "c-space"]

# Must match the defaults of ``make_curvilinear_annulus_grid``.
R_IN, R_OUT = 1.0, 2.0
THETA_MAX = np.pi / 2
Z_MAX = 0.25


def _cyl(r, theta, z):
    """Physical point at the given radius, angle and height."""
    return np.array([r * np.cos(theta), r * np.sin(theta), z])


# A point at the parametric centre of the block.
INTERIOR = _cyl(0.5 * (R_IN + R_OUT), 0.5 * THETA_MAX, 0.5 * Z_MAX)

# One point exactly on each of the six boundary faces, placed at the middle of
# the face so it is not also a node.
FACE_POINTS = {
    "inner-radius": _cyl(R_IN, 0.5 * THETA_MAX, 0.5 * Z_MAX),
    "outer-radius": _cyl(R_OUT, 0.5 * THETA_MAX, 0.5 * Z_MAX),
    "theta-min": _cyl(0.5 * (R_IN + R_OUT), 0.0, 0.5 * Z_MAX),
    "theta-max": _cyl(0.5 * (R_IN + R_OUT), THETA_MAX, 0.5 * Z_MAX),
    "z-min": _cyl(0.5 * (R_IN + R_OUT), 0.5 * THETA_MAX, 0.0),
    "z-max": _cyl(0.5 * (R_IN + R_OUT), 0.5 * THETA_MAX, Z_MAX),
}

# Inside the block's bounding box, outside the block. This is the case a
# Cartesian fixture cannot express.
OUTSIDE_INSIDE_BBOX = {
    # r = 0.42, in the hole inside the inner radius.
    "hole": np.array([0.30, 0.30, 0.5 * Z_MAX]),
    # r = 2.69, in the corner beyond the outer radius.
    "corner": np.array([1.90, 1.90, 0.5 * Z_MAX]),
}

# Outside the bounding box as well; ``_find_block`` rejects these outright.
OUTSIDE_BBOX = {
    "far": np.array([5.0, 5.0, 0.5 * Z_MAX]),
    "below-z": np.array([1.0, 1.0, -1.0]),
}


def _radius(point):
    return float(np.hypot(point[0], point[1]))


def _in_bounding_box(grid, point):
    return bool(np.all(grid.grd_min[0] <= point)
                and np.all(grid.grd_max[0] >= point))


def _located_cell_bounds(grid, search):
    """Axis-aligned bounds of the eight nodes of the located cell."""
    nodes = grid.grd[search.cell[:, 0], search.cell[:, 1], search.cell[:, 2],
                     :, search.block]
    return nodes.min(axis=0), nodes.max(axis=0)


# ---------------------------------------------------------------------------
# The fixture has the property the rest of the file depends on.
# ---------------------------------------------------------------------------

def test_annulus_metrics_are_finite_and_non_singular(curvilinear_grid):
    """The curved mapping produces usable metrics.

    ``Search.p2c`` divides by the Jacobian and interpolates ``m2``, so a
    fixture with a singular or non-finite metric would make the c-space
    searches meaningless rather than testing them.
    """
    ni, nj, nk = (int(curvilinear_grid.ni[0]), int(curvilinear_grid.nj[0]),
                  int(curvilinear_grid.nk[0]))
    jac = curvilinear_grid.J[:ni, :nj, :nk, 0]
    assert np.all(np.isfinite(jac))
    assert np.all(jac > 0.0)
    assert np.all(np.isfinite(curvilinear_grid.m1[:ni, :nj, :nk, :, :, 0]))
    assert np.all(np.isfinite(curvilinear_grid.m2[:ni, :nj, :nk, :, :, 0]))


def test_bounding_box_is_strictly_larger_than_the_block(curvilinear_grid):
    """The premise: bounding-box containment is not block containment here.

    Both out-of-domain probes sit inside the block's axis-aligned bounding box
    while being outside the block, which is precisely what a Cartesian fixture
    cannot produce and what ``_find_block`` alone cannot catch.
    """
    for name, point in OUTSIDE_INSIDE_BBOX.items():
        assert _in_bounding_box(curvilinear_grid, point), name
        radius = _radius(point)
        assert radius < R_IN or radius > R_OUT, name

    for name, point in OUTSIDE_BBOX.items():
        assert not _in_bounding_box(curvilinear_grid, point), name


# ---------------------------------------------------------------------------
# Points that are in the domain must be located, by every method.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_interior_point_is_located(curvilinear_grid, method):
    """A point well inside the curved block is found and not rejected."""
    idx = Search(curvilinear_grid, list(INTERIOR))
    idx.compute(method=method)

    assert idx.block == 0
    assert idx.ppoint is not None
    assert idx.cell is not None and idx.cell.shape == (8, 3)
    assert np.all(idx.cell >= 0)
    assert np.all(idx.cell[:, 0] < curvilinear_grid.ni[0])
    assert np.all(idx.cell[:, 1] < curvilinear_grid.nj[0])
    assert np.all(idx.cell[:, 2] < curvilinear_grid.nk[0])


@pytest.mark.parametrize("face", sorted(FACE_POINTS))
@pytest.mark.parametrize("method", METHODS)
def test_boundary_face_point_is_located(curvilinear_grid, face, method):
    """A point exactly on any of the six faces is in the domain.

    This is the boundary of the block, not the outside of it, so no method may
    reject it -- and the far i/j/k faces must not raise ``IndexError`` either,
    which is what the clamp in ``_cell_nodes`` guarantees.
    """
    point = FACE_POINTS[face]
    idx = Search(curvilinear_grid, list(point))
    idx.compute(method=method)

    assert idx.ppoint is not None, f"{method} rejected the {face} face"
    assert idx.cell is not None and idx.cell.shape == (8, 3)
    assert np.all(idx.cell >= 0)
    assert np.all(idx.cell[:, 0] < curvilinear_grid.ni[0])
    assert np.all(idx.cell[:, 1] < curvilinear_grid.nj[0])
    assert np.all(idx.cell[:, 2] < curvilinear_grid.nk[0])


@pytest.mark.parametrize("face", sorted(FACE_POINTS))
def test_boundary_face_point_round_trips_through_c_space(curvilinear_grid,
                                                         face):
    """``p2c`` then ``c2p`` returns a boundary-face point on the curved grid."""
    point = FACE_POINTS[face]
    idx = Search(curvilinear_grid, list(point))
    idx.compute(method="c-space")
    assert idx.cpoint is not None

    back = Search(curvilinear_grid, list(point))
    back.block = 0
    np.testing.assert_allclose(
        back.c2p(np.asarray(idx.cpoint, dtype="f8").copy()), point,
        rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# Points that are not in the domain must be rejected, by every method.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("probe", sorted(OUTSIDE_INSIDE_BBOX))
@pytest.mark.parametrize("method", METHODS)
def test_point_inside_bounding_box_but_outside_block_is_rejected(
        curvilinear_grid, probe, method):
    """The regression this file exists for.

    The point passes ``_find_block``, so it reaches the cell search. It is not
    in the block, so the search must report that rather than extrapolate from
    the nearest boundary cell.

    ``ppoint is None`` is the rejection signal every caller in the library
    tests (``DataIO._flow_data``, ``Integration``), so that is what is asserted
    here; ``cell`` deliberately keeps the last cell that was examined.
    """
    point = OUTSIDE_INSIDE_BBOX[probe]
    idx = Search(curvilinear_grid, list(point))
    idx.compute(method=method)

    # It really did get past the block test -- otherwise this would be testing
    # the same path as ``test_point_outside_bounding_box_is_rejected``.
    assert idx.block == 0

    assert idx.ppoint is None, (
        f"{method} accepted {probe}, which is outside the block")
    assert idx.cpoint is None


@pytest.mark.parametrize("probe", sorted(OUTSIDE_INSIDE_BBOX))
@pytest.mark.parametrize("method", ["distance", "block_distance"])
def test_rejected_point_was_not_silently_extrapolated(curvilinear_grid, probe,
                                                      method):
    """The rejected point is provably outside the cell that was examined.

    Without the containment test these searches returned this cell with
    ``ppoint`` still set, and the caller would have interpolated flow data at a
    location the cell does not cover.
    """
    point = OUTSIDE_INSIDE_BBOX[probe]
    idx = Search(curvilinear_grid, list(point))
    idx.compute(method=method)

    assert idx.cell is not None
    low, high = _located_cell_bounds(curvilinear_grid, idx)
    assert np.any(point < low) or np.any(point > high)


@pytest.mark.parametrize("probe", sorted(OUTSIDE_BBOX))
@pytest.mark.parametrize("method", METHODS)
def test_point_outside_bounding_box_is_rejected(curvilinear_grid, probe,
                                                method):
    """A point outside every block bounding box is rejected by ``_find_block``.

    This is the pre-existing path and it clears every attribute, including
    ``cell`` and ``block``.
    """
    point = OUTSIDE_BBOX[probe]
    idx = Search(curvilinear_grid, list(point))
    idx.compute(method=method)

    assert idx.cell is None
    assert idx.block is None
    assert idx.ppoint is None
    assert idx.cpoint is None
    assert idx.info is not None and "not in the domain" in idx.info


@pytest.mark.parametrize("method", ["distance", "block_distance"])
def test_rejection_starts_within_one_cell_of_the_curved_boundary(
        curvilinear_grid, method):
    """How far outside the block a point has to be before it is rejected.

    ``_point_in_cell`` compares the point against the bounding box of the
    located cell, not against the curved cell itself, so it is conservative: it
    never rejects a point that is genuinely inside, and it can accept a point
    just outside a curved face. This pins down that the accepted sliver is
    small -- a point a hundredth of a cell beyond the outer radius is already
    rejected -- so the documented behaviour does not quietly widen.
    """
    d_r = (R_OUT - R_IN) / (int(curvilinear_grid.ni[0]) - 1)

    on_face = _cyl(R_OUT, 0.25 * THETA_MAX, 0.5 * Z_MAX)
    idx = Search(curvilinear_grid, list(on_face))
    idx.compute(method=method)
    assert idx.ppoint is not None

    for overshoot in (0.01, 0.1, 1.0):
        outside = _cyl(R_OUT + overshoot * d_r, 0.25 * THETA_MAX, 0.5 * Z_MAX)
        assert _in_bounding_box(curvilinear_grid, outside)
        idx = Search(curvilinear_grid, list(outside))
        idx.compute(method=method)
        assert idx.ppoint is None, (
            f"{method} accepted a point {overshoot} cells outside the block")
