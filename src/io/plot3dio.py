# This file contains classes for plot3d data io
# TODO: implement output for p3d data - Assigned to Harpreet.
#  Each output function should be added to respective GridIO/FlowIO classes
#  Change docstrings for doctest in test_io

import numpy as np


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
        self.q = []

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

    def read_flow(self, data_type='f4'):
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

            print("Flow data reading is successful for " + self.filename + "\n")
