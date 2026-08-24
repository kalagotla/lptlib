"""A point on the far i/j/k boundary must be located, not crash the search.

``Search._cell_nodes`` builds a cell from its lowest node and that node's
``+1`` neighbors. A point landing exactly on the maximum index along an axis
asked for a cell starting at the last node -- a cell that does not exist -- and
``Search.p2c`` raised ``IndexError: index 500 is out of bounds`` while looking
up the grid metrics. That is what killed ``main.py`` on its last particle, and
before the fix, seeding exactly on a far face raised IndexError.

The fix clamps the cell index to the last valid cell, so a point on the far
face is located in the adjacent cell at a local fraction of 1.0 -- the same
physical location.
"""

import numpy as np
import pytest

from lptlib.streamlines import Interpolation, Search

FACES = ["far-i", "far-j", "far-k", "far-corner"]


def _far_face_node(grid, face):
    """Index triple of a node sitting on the requested far face."""
    ni, nj, nk = int(grid.ni[0]), int(grid.nj[0]), int(grid.nk[0])
    return {
        "far-i": (ni - 1, nj // 2, nk // 2),
        "far-j": (ni // 2, nj - 1, nk // 2),
        "far-k": (ni // 2, nj // 2, nk - 1),
        "far-corner": (ni - 1, nj - 1, nk - 1),
    }[face]


@pytest.mark.parametrize("face", FACES)
@pytest.mark.parametrize("method", ["p-space", "c-space", "distance",
                                    "block_distance"])
def test_search_on_the_far_boundary_does_not_raise(synthetic_grid, face, method):
    """Seeding exactly on each far face locates a cell instead of raising."""
    i, j, k = _far_face_node(synthetic_grid, face)
    point = synthetic_grid.grd[i, j, k, :, 0]

    idx = Search(synthetic_grid, list(point))
    idx.compute(method=method)

    assert idx.cell is not None
    assert idx.cell.shape == (8, 3)
    # Every node of the located cell is a real node of the grid.
    assert np.all(idx.cell >= 0)
    assert np.all(idx.cell[:, 0] < synthetic_grid.ni[0])
    assert np.all(idx.cell[:, 1] < synthetic_grid.nj[0])
    assert np.all(idx.cell[:, 2] < synthetic_grid.nk[0])
    # The point itself is one of the located cell's corners.
    corners = synthetic_grid.grd[idx.cell[:, 0], idx.cell[:, 1],
                                 idx.cell[:, 2], :, 0]
    assert np.min(np.linalg.norm(corners - point, axis=1)) < 1e-12


@pytest.mark.parametrize("face", FACES)
def test_far_boundary_c_space_point_round_trips(synthetic_grid, face):
    """p2c then c2p returns the original far-face point."""
    i, j, k = _far_face_node(synthetic_grid, face)
    point = np.asarray(synthetic_grid.grd[i, j, k, :, 0], dtype=float)

    idx = Search(synthetic_grid, list(point))
    cpoint = idx.compute(method="c-space") or idx.cpoint
    assert cpoint is not None

    back = Search(synthetic_grid, list(point))
    back.block = 0
    np.testing.assert_allclose(back.c2p(np.asarray(cpoint, dtype=float)),
                               point, rtol=0, atol=1e-12)


@pytest.mark.parametrize("face", FACES)
def test_far_boundary_interpolation_recovers_the_node_state(synthetic_grid,
                                                            synthetic_flow,
                                                            face):
    """Interpolating on the far face gives that node's flow state exactly.

    The clamped cell puts the point at a local fraction of 1.0, so the
    tri-linear weights collapse onto the node it sits on.
    """
    i, j, k = _far_face_node(synthetic_grid, face)
    point = synthetic_grid.grd[i, j, k, :, 0]
    expected = synthetic_flow.q[i, j, k, :, 0]

    idx = Search(synthetic_grid, list(point))
    idx.compute(method="p-space")
    interp = Interpolation(synthetic_flow, idx)
    interp.compute(method="p-space")

    np.testing.assert_allclose(interp.q.reshape(5), expected, rtol=1e-9,
                               atol=1e-12)
