"""Regression test for GridIO.read_grid coordinate bounds.

read_grid computes per-block min/max coordinate bounds from the single-precision
block views for speed. This test writes a small multi-block PLOT3D grid, reads
it, and checks that grd, grd_min, and grd_max match an independent reference
computed directly from the written coordinates. It guards the optimization
against silently changing the reported bounds.
"""

import struct

import numpy as np


def _write_grid(path, blocks, seed=0):
    """Write a multi-block single-precision PLOT3D grid and return the block arrays."""
    rng = np.random.default_rng(seed)
    arrays = []
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(blocks)))
        for (ni, nj, nk) in blocks:
            f.write(struct.pack("<iii", ni, nj, nk))
        for (ni, nj, nk) in blocks:
            grd = rng.standard_normal((ni, nj, nk, 3)).astype("f4")
            arrays.append(grd)
            f.write(grd.tobytes(order="F"))
    return arrays


def test_read_grid_bounds(tmp_path):
    from lptlib.io import GridIO

    blocks = [(6, 5, 4), (7, 3, 5), (4, 6, 3)]
    path = tmp_path / "bounds_test.x"
    arrays = _write_grid(str(path), blocks, seed=7)

    grid = GridIO(str(path))
    grid.read_grid()

    assert grid.nb == len(blocks)
    assert grid.grd_min.shape == (len(blocks), 3)
    assert grid.grd_max.shape == (len(blocks), 3)

    for b, (ni, nj, nk) in enumerate(blocks):
        ref = arrays[b]
        # Reconstructed coordinates match the written block.
        np.testing.assert_allclose(
            grid.grd[:ni, :nj, :nk, :, b], ref, rtol=0, atol=0
        )
        # Bounds match an independent reduction over the written coordinates.
        np.testing.assert_allclose(
            grid.grd_min[b], ref.min(axis=(0, 1, 2)).astype(np.float64), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            grid.grd_max[b], ref.max(axis=(0, 1, 2)).astype(np.float64), rtol=0, atol=0
        )

    # The default stores grd as float64.
    assert grid.grd.dtype == np.float64


def test_read_grid_single_precision_store(tmp_path):
    """store_type='f4' keeps grd single precision with identical coordinates."""
    from lptlib.io import GridIO

    blocks = [(6, 5, 4), (7, 3, 5), (4, 6, 3)]
    path = tmp_path / "bounds_store_f4.x"
    arrays = _write_grid(str(path), blocks, seed=11)

    grid = GridIO(str(path))
    grid.read_grid(store_type="f4")

    assert grid.grd.dtype == np.float32
    # Bounds are still returned as float64.
    assert grid.grd_min.dtype == np.float64
    assert grid.grd_max.dtype == np.float64

    for b, (ni, nj, nk) in enumerate(blocks):
        ref = arrays[b]
        # Coordinates are bit-exact because the file is already single precision.
        np.testing.assert_array_equal(grid.grd[:ni, :nj, :nk, :, b], ref)
        np.testing.assert_array_equal(
            grid.grd_min[b], ref.min(axis=(0, 1, 2)).astype(np.float64)
        )
        np.testing.assert_array_equal(
            grid.grd_max[b], ref.max(axis=(0, 1, 2)).astype(np.float64)
        )


def test_read_grid_is_idempotent(tmp_path):
    """Calling read_grid twice on the same object gives identical results.

    ``read_grid`` appends per-block bounds to ``grd_min``/``grd_max`` and then
    converts them to arrays. Without resetting the accumulators the second call
    raised ``AttributeError: 'numpy.ndarray' object has no attribute 'append'``.
    """
    from lptlib.io import GridIO

    blocks = [(6, 5, 4), (7, 3, 5)]
    path = tmp_path / "idempotent.x"
    _write_grid(str(path), blocks, seed=3)

    grid = GridIO(str(path))
    grid.read_grid()
    first_grd = grid.grd.copy()
    first_min = grid.grd_min.copy()
    first_max = grid.grd_max.copy()

    grid.read_grid()  # must not raise

    assert grid.grd_min.shape == first_min.shape
    assert grid.grd_max.shape == first_max.shape
    np.testing.assert_array_equal(grid.grd, first_grd)
    np.testing.assert_array_equal(grid.grd_min, first_min)
    np.testing.assert_array_equal(grid.grd_max, first_max)


def test_read_grid_uses_binary_mode(tmp_path):
    """The PLOT3D reader must open the file in binary mode.

    Text mode is a silent corruption on Windows, where the C runtime translates
    CRLF byte pairs inside what is really binary coordinate data.
    """
    import inspect

    from lptlib.io import GridIO

    source = inspect.getsource(GridIO.read_grid)
    assert "open(self.filename, 'rb')" in source
    assert "open(self.filename, 'r')" not in source
