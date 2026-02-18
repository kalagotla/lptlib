# This file contains classes for plot3d data io
#  Each output function should be added to respective GridIO/FlowIO classes
#  Change docstrings for doctest in test_io

import numpy as np
import os


class GridIO:
    """Module to read-in a grid file and output grid parameters

    ...

    Attributes
    ----------
    Input :
        filename : str
            name of the grid file
    Output :
        nb : int
            number of blocks in the grid
        ni, nj, nk : int
            shape of the domain
        grd : numpy.ndarray
            Grid data of shape (ni, nj, nk, 3, nb)
        grd_min: numpy.ndarray
            min co-ordinates of each block (3, nb)
        grd_max: numpy.ndarray
            max co-ordinates of each block (3, nb)
        m1 : numpy.ndarray
            xi, eta, zeta derivatives wrt x, y, z --> shape (ni, nj, nk, 3, 3, nb)
        m2 : numpy.ndarray
            x, y, z derivatives wrt xi, eta, zeta --> shape (ni, nj, nk, 3, 3, nb)
        J : numpy.ndarray
            Jacobian determinant of m1 --> shape (ni, nj, nk, nb)


    Methods
    -------
    read_grid()
        returns the output attributes

    Example:
        grid = GridIO('plate.sp.x')  # Assume file is in the path
        grid.read_grid()  # Call method to read the data
        grid.compute_metrics()  # Computes m1, m2, J arrays
        print(grid)  # prints the docstring for grid
        # Instance attributes
        print(grid.grd.shape)  # shape of the grid data
        print(grid.nb)  # Number of blocks

    author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
    date: 10-05/2021
    co-author: Harpreet Singh @ chhabrhh@mail.uc.edu
    date: 01-20/2024
    """

    def __init__(self, filename):
        self.filename = filename
        self.nb = None
        self.ni, self.nj, self.nk = None, None, None
        self.grd = None
        self.grd_min, self.grd_max = [], []
        self.m1, self.m2 = None, None
        self.J = None

    def __str__(self):
        doc = "This instance has the filename " + self.filename + "\n" + \
              "grd attribute is of shape (ni, nj, nk, 3, nb), rows representing x,y,z coordinates for each block\n" \
              "For example, x = grid.grd(...,0, 0), grid being the object.\n" \
              "Use method 'read_grid' to compute grd attributes:\n" \
              "ni, nj, nk, ng\n" \
              "Use method 'compute_metrics' to compute grid metric attributes:\n" \
              "m1, m2, J\n" \
              "grd -- The grid data\n"
        return doc

    def read_grid(self, data_type='f4'):
        """Reads in the grid file and changes the instance attributes

        Parameters
        -----------
        data_type: str
            Specify the data type of the grid file specified
            Default is 'f4' for single-precision
            For double-precision use 'f8'

        Returns
        -------
        None

        author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
        credit: Paul Orkwis
        date: 11-04/2021
        """
        with open(self.filename, 'r') as grid:
            # Read-in number of blocks
            self.nb = np.fromfile(grid, dtype='i4', count=1)[0]

            # Read-in i, j, k for all blocks
            _temp = np.fromfile(grid, dtype='i4', count=3 * self.nb)
            self.ni, self.nj, self.nk = _temp[0::3], _temp[1::3], _temp[2::3]

            # length of grd data
            _nt = self.ni * self.nj * self.nk * 3
            # read the grd data from file to temp_array
            _temp = np.fromfile(grid, dtype=data_type, count=sum(_nt))

            # pre-define grd to reduce calling pad and concatenate
            self.grd = np.zeros((self.ni.max(), self.nj.max(), self.nk.max(), 3, self.nb))

            # Reshape and assign data to grd
            for _i in range(self.nb):
                self.grd[0:self.ni[_i], 0:self.nj[_i], 0:self.nk[_i], 0:3, _i] = \
                    _temp[sum(_nt[0:_i]):sum(_nt[0:_i]) + _nt[_i]] \
                    .reshape((self.ni[_i], self.nj[_i], self.nk[_i], 3), order='F')

            print("Grid data reading is successful for " + self.filename + "\n")

            # Setup some parameters for further processing
            # Find out the min and max of each block
            for _i, _j, _k, _b in zip(self.ni, self.nj, self.nk, range(self.nb)):
                self.grd_min.append(np.amin(self.grd[:_i, :_j, :_k, :, _b], axis=(0, 1, 2)))
                self.grd_max.append(np.amax(self.grd[:_i, :_j, :_k, :, _b], axis=(0, 1, 2)))

            # Convert lists to arrays
            self.grd_min = np.array(self.grd_min)
            self.grd_max = np.array(self.grd_max)

    def read_grid_fortran_2d(
        self,
        precision: str = "single",
        plane: str = "i",
    ) -> None:
        """Read a 2D Fortran-record Plot3D-style grid plane and populate attributes.

        This method is a Python analogue of the MATLAB ``gridread2d`` helper,
        but instead of returning the coordinates directly it fills the
        :class:`GridIO` attributes in a way that is compatible with the rest of
        the API (``nb``, ``ni``, ``nj``, ``nk``, ``grd``, ``grd_min``,
        ``grd_max``).

        The file is assumed to contain a single 2D plane written as big-endian
        unformatted Fortran records with two coordinate components.

        Parameters
        ----------
        precision:
            Either ``'single'`` (32-bit) or ``'double'`` (64-bit) floating-point
            precision used for the coordinate data in the file.
        plane:
            Which plane the file represents:

            - ``'i'``: i-plane, varying :math:`(j, k)`
            - ``'j'``: j-plane, varying :math:`(i, k)`
            - ``'k'``: k-plane, varying :math:`(i, j)`

        Returns
        -------
        None
        """
        _plane = plane.lower()
        if _plane not in {"i", "j", "k"}:
            raise ValueError("plane must be one of 'i', 'j', or 'k'")

        _prec = precision.lower()
        if _prec == "single":
            float_dtype = ">f4"
        elif _prec == "double":
            float_dtype = ">f8"
        else:
            raise ValueError("precision must be 'single' or 'double'")

        int_dtype = ">i4"

        with open(self.filename, "rb") as f:
            # Leading record marker
            hdr = np.fromfile(f, dtype=int_dtype, count=1)
            if hdr.size != 1:
                raise IOError("Failed to read record header from grid file.")

            if _plane in {"i", "j"}:
                ng = np.fromfile(f, dtype=int_dtype, count=3)
                if ng.size != 3:
                    raise IOError("Failed to read (imax, jmax, kmax) from grid file.")
                imax, jmax, kmax = (int(v) for v in ng)
            else:  # 'k' plane stores only imax, jmax
                ng = np.fromfile(f, dtype=int_dtype, count=2)
                if ng.size != 2:
                    raise IOError("Failed to read (imax, jmax) from grid file.")
                imax, jmax = (int(v) for v in ng)

            # Trailing record marker for the header
            _ = np.fromfile(f, dtype=int_dtype, count=1)

            # Leading record marker for the coordinate data block
            _ = np.fromfile(f, dtype=int_dtype, count=1)

            if _plane == "i":
                count = jmax * kmax * 2
                shape = (jmax, kmax, 2)
            elif _plane == "j":
                count = imax * kmax * 2
                shape = (imax, kmax, 2)
            else:  # _plane == "k"
                count = imax * jmax * 2
                shape = (imax, jmax, 2)

            f2 = np.fromfile(f, dtype=float_dtype, count=count)
            if f2.size != count:
                raise IOError(
                    "Coordinate data block is incomplete or file format does not "
                    "match expected 2D plane layout."
                )

            # Trailing record marker for the coordinate data block
            _ = np.fromfile(f, dtype=int_dtype, count=1)

        # Reshape to 2D coordinates (MATLAB uses column-major ordering)
        f3 = f2.reshape(shape, order="F")
        a = f3[..., 0]
        b = f3[..., 1]

        # Make this compatible with the rest of GridIO by treating the plane
        # as a single block with one cell in the thin direction (nk = 1).
        self.nb = 1
        if _plane == "i":
            # i-plane: fixed i, varying (j, k)
            ni, nj, nk = 1, a.shape[0], a.shape[1]
        elif _plane == "j":
            # j-plane: fixed j, varying (i, k)
            ni, nj, nk = a.shape[0], 1, a.shape[1]
        else:  # 'k' plane: fixed k, varying (i, j)
            ni, nj, nk = a.shape[0], a.shape[1], 1

        self.ni = np.array([ni], dtype=int)
        self.nj = np.array([nj], dtype=int)
        self.nk = np.array([nk], dtype=int)

        # Allocate grid in the same layout as read_grid: (ni, nj, nk, 3, nb)
        self.grd = np.zeros((ni, nj, nk, 3, self.nb), dtype=float)

        if _plane == "i":
            # Map (j, k) -> (0, j, k)
            self.grd[0, :, :, 0, 0] = a
            self.grd[0, :, :, 1, 0] = b
        elif _plane == "j":
            # Map (i, k) -> (i, 0, k)
            self.grd[:, 0, :, 0, 0] = a
            self.grd[:, 0, :, 1, 0] = b
        else:  # 'k'
            # Map (i, j) -> (i, j, 0)
            self.grd[:, :, 0, 0, 0] = a
            self.grd[:, :, 0, 1, 0] = b

        # z-coordinate is zero for a strictly 2D plane
        # (already zero from initialization)

        # Compute min and max coordinates for the single block
        self.grd_min = np.array(
            [np.amin(self.grd[:ni, :nj, :nk, :, 0], axis=(0, 1, 2))]
        )
        self.grd_max = np.array(
            [np.amax(self.grd[:ni, :nj, :nk, :, 0], axis=(0, 1, 2))]
        )

    def plot_grid_2d(
        self,
        plane: str = "k",
        block: int = 0,
        ax=None,
        line_color="k",
        line_width=0.4,
        alpha=0.8,
    ):
        """
        Plot structured grid lines.
        """

        import matplotlib.pyplot as plt
        import numpy as np

        if self.grd is None:
            raise ValueError("Grid data is not loaded.")

        if not (0 <= block < self.nb):
            raise ValueError(f"Block index {block} out of range.")

        plane = plane.lower()
        if plane not in {"i", "j", "k"}:
            raise ValueError("plane must be 'i', 'j', or 'k'")

        ni, nj, nk = int(self.ni[block]), int(self.nj[block]), int(self.nk[block])

        # --- Extract plane ---
        if plane == "i":
            idx = 0 if ni == 1 else ni // 2
            x = self.grd[idx, :nj, :nk, 0, block]
            y = self.grd[idx, :nj, :nk, 1, block]
        elif plane == "j":
            idx = 0 if nj == 1 else nj // 2
            x = self.grd[:ni, idx, :nk, 0, block]
            y = self.grd[:ni, idx, :nk, 1, block]
        else:
            idx = 0 if nk == 1 else nk // 2
            x = self.grd[:ni, :nj, idx, 0, block]
            y = self.grd[:ni, :nj, idx, 1, block]

        x2d = np.squeeze(x)
        y2d = np.squeeze(y)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        # --- Draw structured grid lines ---
        if x2d.ndim == 2:
            ni2, nj2 = x2d.shape

            # Lines in i-direction
            for i in range(ni2):
                ax.plot(
                    x2d[i, :],
                    y2d[i, :],
                    color=line_color,
                    linewidth=line_width,
                    alpha=alpha,
                )

            # Lines in j-direction
            for j in range(nj2):
                ax.plot(
                    x2d[:, j],
                    y2d[:, j],
                    color=line_color,
                    linewidth=line_width,
                    alpha=alpha,
                )

        elif x2d.ndim == 1:
            ax.plot(
                x2d,
                y2d,
                color=line_color,
                linewidth=line_width,
                alpha=alpha,
            )
        else:
            raise ValueError("Slice not suitable for 1D/2D plotting.")

        # --- Make it look like a CFD grid window ---
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Grid plane {plane.upper()} (block {block})")
        ax.grid(False)

        # Remove top/right spines (more CFD-like)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        return ax

    def compute_metrics(self):
        """Calculate grid metrics from grid data.
        Need to call read_grid() before computing metrics

        Parameters
        ----------

        Returns
        -------
        None

        Example:
            grid = GridIO(filename)  # Assume grid is the object from GridIO
            grid.read_grid()  # Call method to read-in grid data
            grid.compute_metrics()  # Call method to compute grid metrics
            print(grid)  # prints the docstring for grid metrics
            # Instance attributes
            print(grid.m1)  # derivatives xi, eta, zeta
            print(grid.m2)  # derivatives x, y, z

        author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
        credit: Jacob Welsh for providing the code for single block
        date: 12-23/2021
        """
        self.m1 = np.zeros((self.ni.max(), self.nj.max(), self.nk.max(), 3, 3, self.nb))
        self.m2 = np.zeros((self.ni.max(), self.nj.max(), self.nk.max(), 3, 3, self.nb))
        self.J = np.zeros((self.ni.max(), self.nj.max(), self.nk.max(), self.nb))

        #  The for loop is to compute x, y, z derivatives wrt xi, eta, zeta
        # for i in range(3):
        #     self.m1[..., i, :] = np.gradient(self.grd, axis=i)

        for b in range(self.nb):
            print(f"Computing Jacobian for block {b}...")
            for i in range(3):
                self.m1[:self.ni[b], :self.nj[b], :self.nk[b], :, i, b] = \
                    np.gradient(self.grd[:self.ni[b], :self.nj[b], :self.nk[b], :, b], axis=i)
        print("Done computing Jacobian for all the blocks!!\n")

        # compute Jacobian
        for b in range(self.nb):
            print(f"Computing Jacobian determinant for block {b}...")
            self.J[:self.ni[b], :self.nj[b], :self.nk[b], b] = \
                self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 0, b] * \
                (self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 1, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 2, b] -
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 2, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 1, b]) - \
                self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 0, b] * \
                (self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 1, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 2, b] -
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 2, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 1, b]) + \
                self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 0, b] * \
                (self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 1, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 2, b] -
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 2, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 1, b])
        print("Done computing Jacobian determinant for all the blocks!!")

        # x derivatives
        for b in range(self.nb):
            print(f"Computing Inverse Jacobian for block {b}...")
            self.m2[:self.ni[b], :self.nj[b], :self.nk[b], 0, 0, b] = \
                (self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 1, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 2, b] -
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 2, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 1, b]) /\
                self.J[:self.ni[b], :self.nj[b], :self.nk[b], b]
            self.m2[:self.ni[b], :self.nj[b], :self.nk[b], 0, 1, b] = \
                (self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 2, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 1, b] -
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 1, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 2, b]) /\
                self.J[:self.ni[b], :self.nj[b], :self.nk[b], b]
            self.m2[:self.ni[b], :self.nj[b], :self.nk[b], 0, 2, b] = \
                (self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 1, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 2, b] -
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 2, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 1, b]) /\
                self.J[:self.ni[b], :self.nj[b], :self.nk[b], b]

            # y derivative
            self.m2[:self.ni[b], :self.nj[b], :self.nk[b], 1, 0, b] = \
                (self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 2, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 0, b] -
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 0, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 2, b]) /\
                self.J[:self.ni[b], :self.nj[b], :self.nk[b], b]
            self.m2[:self.ni[b], :self.nj[b], :self.nk[b], 1, 1, b] = \
                (self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 0, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 2, b] -
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 2, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 0, b]) /\
                self.J[:self.ni[b], :self.nj[b], :self.nk[b], b]
            self.m2[:self.ni[b], :self.nj[b], :self.nk[b], 1, 2, b] = \
                (self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 2, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 0, b] -
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 0, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 2, b]) /\
                self.J[:self.ni[b], :self.nj[b], :self.nk[b], b]

            # z derivatives
            self.m2[:self.ni[b], :self.nj[b], :self.nk[b], 2, 0, b] = \
                (self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 0, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 1, b] -
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 1, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 0, b]) /\
                self.J[:self.ni[b], :self.nj[b], :self.nk[b], b]
            self.m2[:self.ni[b], :self.nj[b], :self.nk[b], 2, 1, b] = \
                (self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 1, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 0, b] -
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 0, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 2, 1, b]) /\
                self.J[:self.ni[b], :self.nj[b], :self.nk[b], b]
            self.m2[:self.ni[b], :self.nj[b], :self.nk[b], 2, 2, b] = \
                (self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 0, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 1, b] -
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 0, 1, b] *
                 self.m1[:self.ni[b], :self.nj[b], :self.nk[b], 1, 0, b]) /\
                self.J[:self.ni[b], :self.nj[b], :self.nk[b], b]

        print("Done computing Inverse Jacobian for all the blocks!!")
        print("All Grid metrics computed successfully!")

    def two_to_three(self, steps: int = 5, step_size: float = None, data_type='f4'):
        """
        Converts 2D plot3d grid file to 3D format
        TODO: Current limitation is that the code works only for 3d written 2d files and single block
        Note: This works only on single block grids.
        Returns: None

        """
        if step_size is None:
            print("Step size is not provided; Using minimum grid size")
            step_size = abs(min(np.diff(self.grd[:, 0, 0, 0, 0])))

        # check self.ni, self.nj dtype -- This is to keep the old functionality working.
        self.ni = self.ni[0] if type(self.ni) == np.ndarray else self.ni
        self.nj = self.nj[0] if type(self.nj) == np.ndarray else self.nj
        self.nk = self.nk[0] if type(self.nk) == np.ndarray else self.nk
        # old code:
        _a_temp = np.array([self.nb, self.ni, self.nj, steps], dtype='i4')
        _x_temp = self.grd[..., 0, 0].repeat(steps, axis=2)
        _y_temp = self.grd[..., 1, 0].repeat(steps, axis=2)
        _z_temp = np.ones((int(self.ni), int(self.nj), steps)) * np.linspace(0, steps*step_size, steps)
        _b_temp = np.array([_x_temp.T, _y_temp.T, _z_temp.T], dtype=data_type)

        # Build output filename cleanly: "<original_path>_3D<ext>"
        _base, _ext = os.path.splitext(self.filename)
        _temp_filename = _base + '_3D' + _ext
        with open(_temp_filename, 'wb') as f:
            f.write(_a_temp.tobytes())
            f.write(_b_temp.tobytes())

        print(f'\n File is successfully written in the working directory as {_temp_filename}')

        return

    def mgrd_to_p3d(self, xi, yi, out_file: str = 'mgrd_to_p3d.x',
                    steps: int = 5, step_size: float = None, data_type='f4'):
        """
        Creates a 3d plot3d grid file;
        This can be later used to be converted to 3d format by two_to_three
        Returns:

        """
        if step_size is None:
            print("Step size is not provided; Using minimum grid size")
            step_size = abs(min(np.diff(self.grd[:, 0, 0, 0, 0])))

        # Number of blocks is always 1 for this function
        _ng, _ni, _nj, _nk = np.array([1, xi.shape[0], yi.shape[1], steps], dtype='i4')
        _xx, _yy, _zz = np.meshgrid(xi[:, 0], yi[0], np.linspace(0, steps * step_size, steps), indexing='ij')
        _grd = np.array([_xx.T, _yy.T, _zz.T], dtype=data_type)

        with open(out_file, 'wb') as f:
            f.write(_ng)
            f.write(_ni)
            f.write(_nj)
            f.write(_nk)
            f.write(_grd.tobytes())

        print(f'\n File is successfully written in the working directory as {out_file}')

        return



class FlowIO:
    """Module to read-in a flow file and output flow parameters

    ...

    Attributes
    ----------
    Input :
        filename : str
            name of the flow file
    Output :
        nb : int
            number of blocks in the grid
        ni, nj, nk : int
            shape of the domain
        mach, alpha, rey, time: float
            extra numerics needed for post-processing
        q : numpy.ndarray
            Flow data of shape (ni, nj, nk, 5, nb)

    Methods
    -------
    read_flow()
        returns the output attributes
    two_to_three()
        converts 2D data to 3D
    mgrd_to_p3d()
        converts and returns meshgrid data to plot3D data
    read_formatted_txt()
        Reads Tecplot returned formatted text. Must contain five columns without headers
        rho, rho-u, rho-v, rho-w, e are the five columns

    Example:
        grid = FlowIO('plate.sp.x')  # Assume file is in the path\n
        flow.read_flow()  # Call method to read the data\n
        print(flow)  # prints the docstring for grid\n
        # Instance attributes\n
        print(flow.q.shape)  # shape of the flow data\n
        print(flow.ng)  # Number of blocks


    author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
    date: 10-05/2021
    """

    def __init__(self, filename):
        self.filename = filename
        self.nb = None
        self.ni, self.nj, self.nk = None, None, None
        self.mach = None
        self.alpha = None
        self.rey = None
        self.time = None
        self.q = None
        self.unsteady_flow = None
        self.gamma = 1.4

    def __str__(self):
        doc = "This instance has the filename " + self.filename + "\n" + \
              "q attribute is of shape (ni, nj, nk, 5, nb), rows representing density, momentum[*3], energy" \
              "for every block\n" \
              "For example, rho = flow.q(...,0,0), flow being the object for block-1.\n" \
              "Use method 'read_flow' to compute attributes:\n" \
              "nb, ni, nj, nk\n" \
              "mach, alpha, rey, time\n" \
              "q -- The flow data\n"
        return doc

    def read_flow(self, data_type='f4', print_progress=True):
        """Reads in the flow file and changes the instance attributes

        Parameters
        ----------
        data_type: str
            Specify the data type of the flow file specified
            Default is 'f4' for single-precision
            For double-precision use 'f8'

        Returns
        -------
        None

        author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
        credit: Paul Orkwis
        date: 10-05/2021
        """
        with open(self.filename, 'r') as data:
            self.nb = np.fromfile(data, dtype='i4', count=1)[0]

            # Read in the i, j, k values for blocks
            _temp = np.fromfile(data, dtype='i4', count=3 * self.nb)
            self.ni, self.nj, self.nk = _temp[0::3], _temp[1::3], _temp[2::3]

            # Read-in flow data into a temp array
            # Less use of fromfile is more speed
            _nt = self.ni * self.nj * self.nk * 5
            _temp = np.fromfile(data, dtype=data_type, count=sum(_nt) + 4 * self.nb)

            # Assign the dimensionless attributes
            self.mach, self.alpha, self.rey, self.time = _temp[0:4]

            # Create a mask to remove dimensionless quantities from q
            # Indices of the first four dimensionless quantities
            _index_array = np.array([0, 1, 2, 3])
            for i in range(len(_nt) - 1):
                _term = sum(_temp[0:i]) + (i + 1) * 4
                _index_array = np.concatenate((_index_array, np.arange(_term, _term + 4)))
            _mask = np.ones(len(_temp), dtype='bool')
            _mask[_index_array] = 0
            # Use the mask to remove dimensionless quantities
            _temp = _temp[_mask]

            # Pre-define q array
            self.q = np.zeros((self.ni.max(), self.nj.max(), self.nk.max(), 5, self.nb))

            # Reshape and assign data to q
            for _i in range(self.nb):
                self.q[0:self.ni[_i], 0:self.nj[_i], 0:self.nk[_i], 0:5, _i] = \
                    _temp[sum(_nt[0:_i]):sum(_nt[0:_i]) + _nt[_i]] \
                    .reshape((self.ni[_i], self.nj[_i], self.nk[_i], 5), order='F')

            if print_progress:
                print("Flow data reading is successful for " + self.filename + "\n")

    def read_flow_fortran_2d(
        self,
        precision: str = "single",
        plane: str = "i",
    ) -> None:
        """Read a 2D Fortran-record RANS flow plane and populate attributes.

        This is a 2D analogue of :meth:`read_flow` for legacy unformatted
        Fortran files that store a single ``i``, ``j``, or ``k`` plane of
        primitive variables (for example ``u, v, p, w, k, epsilon, mu``).

        The file is assumed to be big-endian and to begin with a record
        containing four integers:

        - ``imax``, ``jmax``, ``kmax`` (or a subset, depending on ``plane``)
        - ``nfun`` number of primitive variables

        followed by a record containing ``imax * jmax * kmax * nfun`` values.

        Parameters
        ----------
        precision:
            Either ``'single'`` (32-bit) or ``'double'`` (64-bit) floating-point
            precision used for the flow data.
        plane:
            Which plane the file represents:

            - ``'i'``: i-plane, varying :math:`(j, k)`
            - ``'j'``: j-plane, varying :math:`(i, k)`
            - ``'k'``: k-plane, varying :math:`(i, j)`

        Returns
        -------
        None
        """
        _plane = plane.lower()
        if _plane not in {"i", "j", "k"}:
            raise ValueError("plane must be one of 'i', 'j', or 'k'")

        _prec = precision.lower()
        if _prec == "single":
            float_dtype = ">f4"
        elif _prec == "double":
            float_dtype = ">f8"
        else:
            raise ValueError("precision must be 'single' or 'double'")

        int_dtype = ">i4"  # this needs to be updated with try catch block to handle big endian and little endian

        with open(self.filename, "rb") as f:
            # Leading record marker
            hdr = np.fromfile(f, dtype=int_dtype, count=1)
            if hdr.size != 1:
                raise IOError("Failed to read record header from flow file.")

            # Header with dimensions and number of functions
            ng = np.fromfile(f, dtype=int_dtype, count=3)
            if ng.size != 3:
                raise IOError("Failed to read header (imax, jmax, nfun) from flow file.")

            # Trailing record marker for the header
            _ = np.fromfile(f, dtype=int_dtype, count=1)

            # Leading record marker for the data block
            _ = np.fromfile(f, dtype=int_dtype, count=1)

            if _plane == "i":
                jmax, kmax, nfun = (int(v) for v in ng)
                ni, nj, nk = 1, jmax, kmax
                count = jmax * kmax * nfun
                shape = (jmax, kmax, nfun)
            elif _plane == "j":
                imax, kmax, nfun = (int(v) for v in ng)
                ni, nj, nk = imax, 1, kmax
                count = imax * kmax * nfun
                shape = (imax, kmax, nfun)
            else:  # "k"
                imax, jmax, nfun = (int(v) for v in ng)
                ni, nj, nk = imax, jmax, 1
                count = imax * jmax * nfun
                shape = (imax, jmax, nfun)

            f2 = np.fromfile(f, dtype=float_dtype, count=count)
            if f2.size != count:
                raise IOError(
                    "Flow data block is incomplete or file format does not "
                    "match expected 2D plane layout."
                )

            # Trailing record marker for the data block
            _ = np.fromfile(f, dtype=int_dtype, count=1)

        # Reshape to (.., nfun) in column-major sense
        f3 = f2.reshape(shape, order="F")
        # reshape to match the shape of the flow data object
        f3 = f3.reshape(ni, nj, nk, nfun, 1)

        # Update FlowIO attributes to be consistent with the rest of the API
        self.nb = 1
        self.ni = np.array([ni], dtype=int)
        self.nj = np.array([nj], dtype=int)
        self.nk = np.array([nk], dtype=int)

        # Store all primitive variables in q with nfun components
        self.q = np.zeros((ni, nj, nk, nfun, self.nb), dtype=float)

        # convert this to be compatible with the rest of the API
        # self.q is made up of (ni, nj, nk, 5, nb), where rho, rho-u, rho-v, rho-w, e are the five columns
        # File format assumption from Gargi Doshara: u
        # For different planes, the velocity components vary:
        # - i-plane: fixed i, varying (j, k) -> in-plane velocities are v and w
        # - j-plane: fixed j, varying (i, k) -> in-plane velocities are u and w  
        # - k-plane: fixed k, varying (i, j) -> in-plane velocities are u and v
        # Note: Original k-plane code used f3[..., 1, 0] = u, f3[..., 2, 0] = v, but that conflicts
        # with f3[..., 2, 0] being used as pressure. Updated to use consistent indexing.
        
        rho = f3[..., 3, 0]
        p = f3[..., 2, 0]
        
        if _plane == "i":
            # i-plane: f3[..., 0, 0] = v, f3[..., 1, 0] = w (in-plane velocities)
            v_vel = f3[..., 0, 0]
            w_vel = f3[..., 1, 0]
            u_vel = np.zeros_like(v_vel)  # u is zero (normal to plane)
            self.q[..., 0, 0] = rho
            self.q[..., 1, 0] = 0  # rho-u = 0
            self.q[..., 2, 0] = v_vel * rho  # rho-v
            self.q[..., 3, 0] = w_vel * rho  # rho-w
            # energy = p/(gamma - 1) + 0.5 * rho * (v^2 + w^2)
            self.q[..., 4, 0] = p / (self.gamma - 1) + 0.5 * rho * (v_vel**2 + w_vel**2)
        elif _plane == "j":
            # j-plane: f3[..., 0, 0] = u, f3[..., 1, 0] = w (in-plane velocities)
            u_vel = f3[..., 0, 0]
            w_vel = f3[..., 1, 0]
            v_vel = np.zeros_like(u_vel)  # v is zero (normal to plane)
            self.q[..., 0, 0] = rho
            self.q[..., 1, 0] = u_vel * rho  # rho-u
            self.q[..., 2, 0] = 0  # rho-v = 0
            self.q[..., 3, 0] = w_vel * rho  # rho-w
            # energy = p/(gamma - 1) + 0.5 * rho * (u^2 + w^2)
            self.q[..., 4, 0] = p / (self.gamma - 1) + 0.5 * rho * (u_vel**2 + w_vel**2)
        else:  # _plane == "k"
            # k-plane: f3[..., 0, 0] = u, f3[..., 1, 0] = v (in-plane velocities)
            # Updated from original: f3[..., 1, 0] = u, f3[..., 2, 0] = v to use consistent indexing
            u_vel = f3[..., 0, 0]
            v_vel = f3[..., 1, 0]
            w_vel = np.zeros_like(u_vel)  # w is zero (normal to plane)
            self.q[..., 0, 0] = rho
            self.q[..., 1, 0] = u_vel * rho  # rho-u
            self.q[..., 2, 0] = v_vel * rho  # rho-v
            self.q[..., 3, 0] = 0  # rho-w = 0
            # energy = p/(gamma - 1) + 0.5 * rho * (u^2 + v^2)
            self.q[..., 4, 0] = p / (self.gamma - 1) + 0.5 * rho * (u_vel**2 + v_vel**2)

    def plot_contour(
        self,
        variable,
        grid=None,
        plane: str = "k",
        block: int = 0,
        index: int = None,
        ax=None,
        levels=None,
        cmap=None,
        colorbar=True,
        **kwargs
    ):
        """
        Plot contour of any variable from q array.

        Parameters
        ----------
        variable : int or str
            Variable to plot:
            - int (0-4): Direct index into q array
                - 0: density (rho)
                - 1: rho-u (momentum x)
                - 2: rho-v (momentum y)
                - 3: rho-w (momentum z)
                - 4: energy (e)
            - str: Variable name ('density', 'rho-u', 'rho-v', 'rho-w', 'energy')
        grid : GridIO, optional
            Grid object for coordinates. If None, uses index-based coordinates.
        plane : str, default 'k'
            Plane to extract: 'i', 'j', or 'k'
        block : int, default 0
            Block index to plot
        index : int, optional
            Index along the plane direction (e.g., i-index for i-plane).
            If None, uses middle of the domain.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        levels : int or array-like, optional
            Number of contour levels or array of specific levels.
            If None, uses default matplotlib levels.
        cmap : str or Colormap, optional
            Colormap to use for contours.
        colorbar : bool, default True
            Whether to add a colorbar.
        **kwargs
            Additional keyword arguments passed to matplotlib contour/contourf.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the contour plot.

        Examples
        --------
        >>> flow = FlowIO('flow.q')
        >>> flow.read_flow()
        >>> grid = GridIO('grid.x')
        >>> grid.read_grid()
        >>> flow.plot_contour(0, grid=grid)  # Plot density
        >>> flow.plot_contour('density', grid=grid, plane='j', index=50)
        >>> flow.plot_contour(4, grid=grid, levels=20, cmap='viridis')
        """
        import matplotlib.pyplot as plt

        if self.q is None:
            raise ValueError("Flow data is not loaded. Call read_flow() first.")

        if not (0 <= block < self.nb):
            raise ValueError(f"Block index {block} out of range [0, {self.nb-1}].")

        plane = plane.lower()
        if plane not in {"i", "j", "k"}:
            raise ValueError("plane must be 'i', 'j', or 'k'")

        # Map variable name to index if string provided
        var_map = {
            'density': 0, 'rho': 0,
            'rho-u': 1, 'rhou': 1, 'momentum_x': 1,
            'rho-v': 2, 'rhov': 2, 'momentum_y': 2,
            'rho-w': 3, 'rhow': 3, 'momentum_z': 3,
            'energy': 4, 'e': 4
        }

        if isinstance(variable, str):
            variable = var_map.get(variable.lower())
            if variable is None:
                raise ValueError(f"Unknown variable name '{variable}'. "
                               f"Valid names: {list(var_map.keys())}")

        if not (0 <= variable < 5):
            raise ValueError(f"Variable index must be between 0 and 4, got {variable}")

        ni, nj, nk = int(self.ni[block]), int(self.nj[block]), int(self.nk[block])

        # Extract 2D slice of the variable
        if plane == "i":
            idx = index if index is not None else (0 if ni == 1 else ni // 2)
            if not (0 <= idx < ni):
                raise ValueError(f"Index {idx} out of range [0, {ni-1}] for i-plane")
            var_2d = self.q[idx, :nj, :nk, variable, block]
            if grid is not None:
                x = grid.grd[idx, :nj, :nk, 0, block]
                y = grid.grd[idx, :nj, :nk, 1, block]
            else:
                x, y = np.meshgrid(np.arange(nj), np.arange(nk), indexing='ij')
        elif plane == "j":
            idx = index if index is not None else (0 if nj == 1 else nj // 2)
            if not (0 <= idx < nj):
                raise ValueError(f"Index {idx} out of range [0, {nj-1}] for j-plane")
            var_2d = self.q[:ni, idx, :nk, variable, block]
            if grid is not None:
                x = grid.grd[:ni, idx, :nk, 0, block]
                y = grid.grd[:ni, idx, :nk, 1, block]
            else:
                x, y = np.meshgrid(np.arange(ni), np.arange(nk), indexing='ij')
        else:  # plane == "k"
            idx = index if index is not None else (0 if nk == 1 else nk // 2)
            if not (0 <= idx < nk):
                raise ValueError(f"Index {idx} out of range [0, {nk-1}] for k-plane")
            var_2d = self.q[:ni, :nj, idx, variable, block]
            if grid is not None:
                x = grid.grd[:ni, :nj, idx, 0, block]
                y = grid.grd[:ni, :nj, idx, 1, block]
            else:
                x, y = np.meshgrid(np.arange(ni), np.arange(nj), indexing='ij')

        # Squeeze to remove singleton dimensions
        x = np.squeeze(x)
        y = np.squeeze(y)
        var_2d = np.squeeze(var_2d)

        if x.ndim != 2 or y.ndim != 2 or var_2d.ndim != 2:
            raise ValueError(f"Slice not suitable for 2D contour plotting. "
                             f"Shapes: x={x.shape}, y={y.shape}, var={var_2d.shape}")

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Plot contours
        var_names = ['Density (ρ)', 'Momentum-x (ρu)', 'Momentum-y (ρv)', 
                     'Momentum-z (ρw)', 'Energy (e)']
        var_name = var_names[variable]

        # Plot filled contours
        cs = ax.contourf(x, y, var_2d, levels=levels, cmap=cmap, **kwargs)
        
        # Add contour lines
        ax.contour(x, y, var_2d, levels=levels, colors='k', linewidths=0.5, alpha=0.3)

        # Add colorbar
        if colorbar:
            cbar = fig.colorbar(cs, ax=ax)
            cbar.set_label(var_name, rotation=90, labelpad=15)

        # Format axes
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{var_name} contour - Plane {plane.upper()} (block {block}, index {idx})")
        ax.grid(False)

        # Remove top/right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        return ax

    def two_to_three(self, steps: int = 5, data_type='f4'):
        """
        Converts 2D plot3d flow file to 3D format
        TODO: Current limitation is that the code works only for 3d written 2d files and single block
        Returns: None

        """
        # TODO: The code below needs debugging. It's not tested. mgrd_to_p3d might help!
        # check self.ni, self.nj dtype -- This is to keep the old functionality working.
        self.ni = self.ni[0] if type(self.ni) == np.ndarray else self.ni
        self.nj = self.nj[0] if type(self.nj) == np.ndarray else self.nj
        self.nk = self.nk[0] if type(self.nk) == np.ndarray else self.nk
        _a_temp = np.array([self.nb, self.ni, self.nj, int(steps)], dtype='i4')
        _b_temp = np.array([self.mach, self.alpha, self.rey, self.time], dtype=data_type)
        _q0_temp = self.q[..., 0, 0].repeat(int(steps), axis=2)
        _q1_temp = self.q[..., 1, 0].repeat(int(steps), axis=2)
        _q2_temp = self.q[..., 2, 0].repeat(int(steps), axis=2)
        _q3_temp = self.q[..., 3, 0].repeat(int(steps), axis=2)
        _q4_temp = self.q[..., 4, 0].repeat(int(steps), axis=2)
        _q_temp = np.array([_q0_temp, _q1_temp, _q2_temp, _q3_temp, _q4_temp], dtype=data_type)

        _base, _ext = os.path.splitext(self.filename)
        _temp_filename = _base + '_3D' + _ext
        with open(_temp_filename, 'wb') as f:
            f.write(_a_temp.tobytes())
            f.write(_b_temp.tobytes())
            f.write(_q_temp.tobytes())

        return

    def mgrd_to_p3d(self, q, out_file: str = 'mgrd_to_p3d', mode: str = None, steps: int = 5, data_type='f4'):
        """
        Writes out data generated from scattered data interpolation to plot3d format
        Args:
            q:
            out_file:
            mode:
            steps:
            data_type:

        Returns:

        """
        _a_temp = np.array([1, q.shape[1], q.shape[-1], int(steps)], dtype='i4')
        _b_temp = np.array([self.mach, self.alpha, self.rey, self.time], dtype=data_type)
        _q0 = np.expand_dims(q[0, ...], axis=2).T.repeat(int(steps), axis=0)
        _q1 = np.expand_dims(q[1, ...], axis=2).T.repeat(int(steps), axis=0)
        _q2 = np.expand_dims(q[2, ...], axis=2).T.repeat(int(steps), axis=0)
        _q3 = np.expand_dims(q[3, ...], axis=2).T.repeat(int(steps), axis=0)
        _q4 = np.expand_dims(q[4, ...], axis=2).T.repeat(int(steps), axis=0)
        _q = np.array([_q0, _q1, _q2, _q3, _q4], dtype=data_type)

        with open(out_file+'_'+mode+'.q', 'wb') as f:
            f.write(_a_temp.tobytes())
            f.write(_b_temp.tobytes())
            f.write(_q.tobytes())

        return

    def read_formatted_txt(self, grid, data_type='f8'):
        import os
        """
        Reads the formatted flow file generated from Tecplot. Needs grid object
        Generate data from tecplot without the header, delimiter as space and have 5 variables
        rho, rho-u, rho-v, rho-w, and e
        Manually add all the required variables for flow object
        Args:
            grid: grid object related to the flow file
            data_type: data type of the formatted text from Tecplot

        Returns:
            None

        """
        try:
            # load the flow file if available
            self.q = np.load(self.filename + '_temp/flow_data.npy')
            print('**IMPORTANT** Read data from existing temp flow_data.npy file.\n')
        except:
            try:
                # Read the formatted data till the file ends and reshape it to (ni, nj, nk, 5, nb)
                print('Starting flow read from the formatted text. Please wait...\n')
                with open(self.filename, 'r') as flow:
                    self.q = np.fromfile(flow, sep=' ', dtype=data_type, count=-1)\
                        .reshape((int(grid.nk), int(grid.nj), int(grid.ni), 5, 1)).transpose(2, 1, 0, 3, 4)
                print('Flow data reading is successful for ' + self.filename + '\n')
            except ValueError:
                # check for 2D formatted data
                # read the formatted data till the file ends and reshape it to (ni, nj, 1, 5, nb)
                # and expand it to (ni, nj, nk, 5, nb)
                with open(self.filename, 'r') as flow:
                    self.q = np.fromfile(flow, sep=' ', dtype=data_type, count=-1)\
                        .reshape((1, int(grid.nj), int(grid.ni), 5, 1)).transpose(2, 1, 0, 3, 4)
                self.q = self.q.repeat(int(grid.nk), axis=2)
                print('Flow data reading is successful for ' + self.filename + '\n')


            # Save as numpy file for future computations
            try:
                # Try creating the directory; if exists errors out and except
                print('Trying to save to npy for future use\n')
                os.mkdir(self.filename + '_temp')
                np.save(self.filename + '_temp/flow_data', self.q)
                print('Created _temp folder and saved flow data for future use.\n')
            except:
                np.save(self.filename + '_temp/flow_data', self.q)
                print('Saved file for future use.\n')

        # Fill in other variables
        if self.mach is not None and self.rey is not None and self.alpha is not None and self.time is not None:
            print('\nYour flow object is ready for further computations!\n')
        else:
            print('\nPlease fill out mach, rey, alpha, and time variables in the object\n')
            print('\n** ERROR **: Try again; one of the required variables is not filled\n')
            exit()

        return

    def read_unsteady_flow(self, data_type='f4'):
        """
        Reads in the unsteady flow files and changes the instance attributes
        Args:
            data_type: data type of the flow file

        Returns:
            None
            Fills up unsteady_flow attribute with a list of flow objects

        """
        import os
        import glob
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None
        self.read_flow(data_type=data_type)

        # Find all the files relevant to initial flow file
        # get file extension
        _ext = os.path.splitext(self.filename)[-1]
        # get directory
        _dir = os.path.dirname(self.filename)
        # extract all the files with the same extension in the working directory
        _filenames = glob.glob(os.path.join(_dir, '*' + _ext))
        # sort filenames
        _filenames.sort()
        # Read all the flow files into a list as objects
        self.unsteady_flow = [FlowIO(filename) for filename in _filenames]
        # Read all the flow files
        _iterator = self.unsteady_flow
        if tqdm is not None:
            _iterator = tqdm(self.unsteady_flow, desc='Reading unsteady flow', unit='file')
        for _flowfile in _iterator:
            _flowfile.read_flow(data_type=data_type, print_progress=False)

