"""Round-trip and structural tests for the PLOT3D reader on synthetic files.

These tests write small grids and solutions to disk in the exact binary layout
the readers expect, read them back, and assert the arrays and header values
survive the trip. They run everywhere because they generate their own data, so
they cover the I/O path that the large-file tests skip on a clean checkout.
"""

import numpy as np
import pytest

from lptlib.io import GridIO, FlowIO
from synthetic import write_plot3d_grid, write_plot3d_flow


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_grid_single_block_round_trip(tmp_path, dtype):
    """A single non-cubic block survives a write/read round trip exactly."""
    rng = np.random.default_rng(0)
    block = rng.random((4, 3, 2, 3))
    path = str(tmp_path / "single.x")
    write_plot3d_grid(path, [block], dtype=dtype)

    grid = GridIO(path)
    grid.read_grid(data_type=dtype)

    assert grid.nb == 1
    assert grid.grd.shape == (4, 3, 2, 3, 1)
    tol = 0.0 if dtype == "f8" else 1e-6
    assert np.allclose(grid.grd[:4, :3, :2, :, 0], block, atol=tol)


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_grid_multiblock_odd_shapes(tmp_path, dtype):
    """Multiblock grids with differing, odd block shapes read back correctly.

    The reader strides the interleaved ``ni, nj, nk`` header and pads every
    block into a common array. This checks the strided header parse and that
    each block lands in the right slice with the right values.
    """
    rng = np.random.default_rng(1)
    block0 = rng.random((5, 3, 2, 3))
    block1 = rng.random((2, 4, 2, 3))
    path = str(tmp_path / "multi.x")
    write_plot3d_grid(path, [block0, block1], dtype=dtype)

    grid = GridIO(path)
    grid.read_grid(data_type=dtype)

    assert grid.nb == 2
    assert list(grid.ni) == [5, 2]
    assert list(grid.nj) == [3, 4]
    assert list(grid.nk) == [2, 2]
    # Padded common array covers the largest extent in every dimension.
    assert grid.grd.shape == (5, 4, 2, 3, 2)
    tol = 0.0 if dtype == "f8" else 1e-6
    assert np.allclose(grid.grd[:5, :3, :2, :, 0], block0, atol=tol)
    assert np.allclose(grid.grd[:2, :4, :2, :, 1], block1, atol=tol)


def test_grid_known_values(tmp_path):
    """A grid built from an analytic coordinate map reads back node-for-node."""
    ni, nj, nk = 3, 4, 2
    block = np.zeros((ni, nj, nk, 3))
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                block[i, j, k] = [i * 1.0, j * 2.0, k * 3.0]
    path = str(tmp_path / "known.x")
    write_plot3d_grid(path, [block], dtype="f8")

    grid = GridIO(path)
    grid.read_grid(data_type="f8")

    assert grid.grd[2, 3, 1, 0, 0] == pytest.approx(2.0)
    assert grid.grd[2, 3, 1, 1, 0] == pytest.approx(6.0)
    assert grid.grd[2, 3, 1, 2, 0] == pytest.approx(3.0)
    # Block coordinate bounds are recorded per block.
    assert grid.grd_min[0] == pytest.approx([0.0, 0.0, 0.0])
    assert grid.grd_max[0] == pytest.approx([2.0, 6.0, 3.0])


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_flow_round_trip(tmp_path, dtype):
    """A solution block and its dimensionless header survive a round trip."""
    rng = np.random.default_rng(2)
    q = rng.random((4, 3, 2, 5))
    path = str(tmp_path / "sol.q")
    write_plot3d_flow(path, [q], mach=2.5, alpha=1.0, rey=3.2e5, time=7.0,
                      dtype=dtype)

    flow = FlowIO(path)
    flow.read_flow(data_type=dtype)

    assert flow.nb == 1
    assert flow.q.shape == (4, 3, 2, 5, 1)
    assert flow.mach == pytest.approx(2.5, rel=1e-5)
    assert flow.alpha == pytest.approx(1.0, rel=1e-5)
    assert flow.rey == pytest.approx(3.2e5, rel=1e-5)
    assert flow.time == pytest.approx(7.0, rel=1e-5)
    tol = 0.0 if dtype == "f8" else 1e-6
    assert np.allclose(flow.q[:4, :3, :2, :, 0], q, atol=tol)


def test_flow_multiblock_header_parse(tmp_path):
    """A multiblock solution file parses its block count and strided header.

    The reader recovers the number of blocks and the interleaved ``ni, nj, nk``
    sizes for every block and allocates the padded ``q`` array to the largest
    extent. Only the block-count and header parse is asserted here: the
    per-block dimensionless-quantity removal in ``read_flow`` is exercised for
    the single-block case in ``test_flow_round_trip``.
    """
    rng = np.random.default_rng(3)
    q0 = rng.random((3, 3, 2, 5))
    q1 = rng.random((2, 2, 2, 5))
    path = str(tmp_path / "sol_mb.q")
    write_plot3d_flow(path, [q0, q1], dtype="f8")

    flow = FlowIO(path)
    flow.read_flow(data_type="f8")

    assert flow.nb == 2
    assert list(flow.ni) == [3, 2]
    assert list(flow.nj) == [3, 2]
    assert list(flow.nk) == [2, 2]
    assert flow.q.shape == (3, 3, 2, 5, 2)


def test_truncated_grid_file_raises(tmp_path):
    """A truncated grid file is rejected rather than silently mis-read.

    The reader reshapes the coordinate stream into the declared block shape, so
    a file whose body is cut short cannot satisfy the shape and raises.
    """
    rng = np.random.default_rng(4)
    block = rng.random((4, 3, 2, 3))
    good = str(tmp_path / "good.x")
    write_plot3d_grid(good, [block], dtype="f4")

    raw = open(good, "rb").read()
    bad = str(tmp_path / "truncated.x")
    with open(bad, "wb") as handle:
        handle.write(raw[: len(raw) // 2])

    grid = GridIO(bad)
    with pytest.raises(ValueError):
        grid.read_grid(data_type="f4")


def test_grid_metrics_shapes_on_synthetic(oblique_case):
    """Grid metric arrays have the documented shapes on the synthetic case."""
    grid = oblique_case.grid
    ni, nj, nk, nb = grid.ni[0], grid.nj[0], grid.nk[0], grid.nb
    assert grid.m1.shape == (ni, nj, nk, 3, 3, nb)
    assert grid.m2.shape == (ni, nj, nk, 3, 3, nb)
    assert grid.J.shape == (ni, nj, nk, nb)
    # The Jacobian determinant is strictly positive for this right-handed mesh.
    assert np.all(grid.J > 0)
