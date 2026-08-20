"""A deliberately naive, pure-Python PLOT3D grid reader.

This reader exists as a baseline for the benchmark. It reads the same file as
``lptlib.io.GridIO.read_grid`` but reconstructs the multi-block arrays with
explicit nested Python loops over every grid point and coordinate, instead of
the single strided Fortran-order reshape that lptlib uses. It therefore
isolates the cost of the per-point Python overhead that the vectorized reader
avoids.

The file I/O is intentionally kept identical to the lptlib path (one buffered
read of the float block), so the timing difference reflects the reconstruction
strategy, not different amounts of disk work.
"""

import struct

import numpy as np


def read_grid_naive(filename, data_type="f4"):
    """Read a multi-block PLOT3D grid with nested Python loops.

    Returns a list of (ni, nj, nk, 3) numpy arrays, one per block.
    """
    with open(filename, "rb") as f:
        nb = struct.unpack("<i", f.read(4))[0]

        dims = []
        for _ in range(nb):
            ni, nj, nk = struct.unpack("<iii", f.read(12))
            dims.append((ni, nj, nk))

        blocks = []
        for (ni, nj, nk) in dims:
            count = ni * nj * nk * 3
            # Same single buffered read as the vectorized reader.
            flat = np.frombuffer(f.read(count * 4), dtype=data_type)

            grd = np.zeros((ni, nj, nk, 3), dtype=data_type)
            # Nested loops reproducing the Fortran-order layout by hand. The
            # first axis (i) varies fastest, so the flat index is
            # i + ni*(j + nj*(k + nk*c)).
            p = 0
            for c in range(3):
                for k in range(nk):
                    for j in range(nj):
                        for i in range(ni):
                            grd[i, j, k, c] = flat[p]
                            p += 1
            blocks.append(grd)

    return blocks


if __name__ == "__main__":
    import sys

    blks = read_grid_naive(sys.argv[1])
    total = sum(b.sum(dtype=np.float64) for b in blks)
    print(f"blocks={len(blks)} checksum={total:.6f}")
