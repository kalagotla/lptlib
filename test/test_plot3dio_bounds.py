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
