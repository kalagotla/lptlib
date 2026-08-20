"""Generate a synthetic multi-block structured PLOT3D grid file.

The file is written in the exact little-endian binary layout that
``lptlib.io.GridIO.read_grid`` consumes, so the same file can be read by the
lptlib strided reader, by the naive Python reader, and by the Fortran reader in
this benchmark. This grid is synthetic. It is a smoothly warped structured mesh
of a realistic multi-block size, and it exists only so that a reviewer can
reproduce the I/O benchmark without needing any large or restricted data files.

File layout (all little-endian):

    int32                         number of blocks, nb
    int32 * (3 * nb)              ni, nj, nk for each block, interleaved
    float32                       block data, concatenated block by block,
                                  each block an (ni, nj, nk, 3) array written
                                  in Fortran (column-major) order

author: benchmark generated for lptlib (Dilip Kalagotla)
"""

import argparse
import struct

import numpy as np


# Default multi-block configuration. Four blocks of differing sizes so the
# grid exercises the multi-block reconstruction path rather than a single
# uniform reshape. Total is a little over two million grid points.
DEFAULT_BLOCKS = [
    (100, 90, 60),
    (110, 80, 55),
    (95, 85, 70),
    (120, 75, 50),
]


def build_block(ni, nj, nk, seed):
    """Return an (ni, nj, nk, 3) float32 array for one smoothly warped block."""
    rng = np.random.default_rng(seed)
    i = np.linspace(0.0, 1.0, ni)
    j = np.linspace(0.0, 1.0, nj)
    k = np.linspace(0.0, 1.0, nk)
    ii, jj, kk = np.meshgrid(i, j, k, indexing="ij")

    # A smooth analytic warping keeps the coordinates deterministic and
    # physically plausible. A small block-specific offset separates the blocks.
    offset = float(seed)
    x = ii + 0.05 * np.sin(2.0 * np.pi * jj) + offset
    y = jj + 0.05 * np.sin(2.0 * np.pi * kk)
    z = kk + 0.05 * np.sin(2.0 * np.pi * ii)

    grd = np.empty((ni, nj, nk, 3), dtype="f4")
    grd[..., 0] = x
    grd[..., 1] = y
    grd[..., 2] = z
    # Silence the unused rng (kept for anyone who wants to add jitter).
    _ = rng
    return grd


def write_grid(filename, blocks=DEFAULT_BLOCKS):
    """Write the multi-block grid and return (nb, blocks, total_points, checksum).

    ``checksum`` is the float64 sum of every coordinate value written, used by
    the benchmark to confirm that all three readers reconstruct the same data.
    """
    nb = len(blocks)
    checksum = np.float64(0.0)
    total_points = 0

    with open(filename, "wb") as f:
        f.write(struct.pack("<i", nb))
        for (ni, nj, nk) in blocks:
            f.write(struct.pack("<iii", ni, nj, nk))
        for b, (ni, nj, nk) in enumerate(blocks):
            grd = build_block(ni, nj, nk, seed=b + 1)
            checksum += np.float64(grd.sum(dtype=np.float64))
            total_points += ni * nj * nk
            # Fortran order so the first axis (i) varies fastest, matching the
            # order='F' reshape in GridIO.read_grid and a direct Fortran read.
            f.write(grd.tobytes(order="F"))

    return nb, blocks, total_points, float(checksum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="path to the grid file to write")
    args = parser.parse_args()
    nb, blocks, total_points, checksum = write_grid(args.output)
    print(f"Wrote {args.output}")
    print(f"  blocks       : {nb}")
    print(f"  block shapes : {blocks}")
    print(f"  total points : {total_points:,}")
    print(f"  checksum     : {checksum:.6f}")


if __name__ == "__main__":
    main()
