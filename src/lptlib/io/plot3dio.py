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
        Returns: None

        """
        if step_size is None:
            print("Step size is not provided; Using minimum grid size")
            step_size = abs(min(np.diff(self.grd[:, 0, 0, 0, 0])))

        _a_temp = np.array([self.nb, self.ni, self.nj, steps], dtype='i4')
        _x_temp = self.grd[..., 0, 0].repeat(steps, axis=2)
        _y_temp = self.grd[..., 1, 0].repeat(steps, axis=2)
        _z_temp = np.ones((int(self.ni), int(self.nj), steps)) * np.linspace(0, steps*step_size, steps)
        _b_temp = np.array([_x_temp.T, _y_temp.T, _z_temp.T], dtype=data_type)

        _temp_filename = self.filename.replace('.x', '_3D.x')
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

    def two_to_three(self, steps: int = 5, data_type='f4'):
        """
        Converts 2D plot3d flow file to 3D format
        TODO: Current limitation is that the code works only for 3d written 2d files and single block
        Returns: None

        """
        # TODO: The code below needs debugging. It's not tested. mgrd_to_p3d might help!
        _a_temp = np.array([self.nb, self.ni, self.nj, int(steps)], dtype='i4')
        _b_temp = np.array([self.mach, self.alpha, self.rey, self.time], dtype=data_type)
        _q0_temp = self.q[..., 0, 0].repeat(int(steps), axis=2)
        _q1_temp = self.q[..., 1, 0].repeat(int(steps), axis=2)
        _q2_temp = self.q[..., 2, 0].repeat(int(steps), axis=2)
        _q3_temp = self.q[..., 3, 0].repeat(int(steps), axis=2)
        _q4_temp = self.q[..., 4, 0].repeat(int(steps), axis=2)
        _q_temp = np.array([_q0_temp, _q1_temp, _q2_temp, _q3_temp, _q4_temp], dtype=data_type)

        _temp_filename = self.filename.replace('.q', '_3D.q')
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
        for _flowfile in self.unsteady_flow:
            _flowfile.read_flow(data_type=data_type)

