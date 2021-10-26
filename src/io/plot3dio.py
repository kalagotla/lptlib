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
        ng : int
            number of domains in the grid
        ni, nj, nk : int
            shape of the domain
        grd : numpy.ndarray
            Grid data of shape (ni, nj, nk, 3)

    Methods
    -------
    read_grid()
        returns the output attributes

    Example:
        grid = GridIO('plate.sp.x')  # Assume file is in the path
        grid.read_grid()  # Call method to read the data
        print(grid)  # prints the docstring for grid
        # Instance attributes
        print(grid.grd.shape)  # shape of the grid data
        print(grid.ng)  # Number of blocks

    author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
    date: 10-05/2021
    """

    def __init__(self, filename):
        self.filename = filename
        self.ng = None
        self.ni, self.nj, self.nk = None, None, None
        self.grd = None

    def __str__(self):
        doc = "This instance has the filename " + self.filename + "\n" + \
              "grd attribute is of shape (ni, nj, nk, 3), rows representing x,y,z coordinates\n" \
              "For example, x = grid.grd(...,0), grid being the object.\n" \
              "Use method 'read_grid' to compute attributes:\n" \
              "ni, nj, nk, ng\n" \
              "grd -- The grid data\n"
        return doc

    def read_grid(self):
        """Reads in the grid file and changes the instance attributes

        :parameter

        :return
        None

        author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
        credit: Paul Orkwis
        date: 10-05/2021
        """
        with open(self.filename, 'r') as grid:
            # Number of blocks
            self.ng = np.fromfile(grid, dtype='i4', count=1)[0]

            # Should be looped for multiple blocks
            # ni, nj, nk values for one block
            self.ni, self.nj, self.nk = np.fromfile(grid, dtype='i4', count=3)

            # Read-in grid data
            self.grd = np.zeros((self.ni, self.nj, self.nk, 3))

            # These loops cannot be avoided due to fortran file formatting
            # This is where the read-in is slow
            # Could be made faster using hdf/cgns formats
            for m in range(3):
                for k in range(self.nk):
                    for j in range(self.nj):
                        self.grd[:, j, k, m] = np.fromfile(grid, dtype='f4', count=self.ni)

            print("Grid data reading is successful for " + self.filename + "\n")


class FlowIO:
    """Module to read-in a flow file and output flow parameters

    ...

    Attributes
    ----------
    Input :
        filename : str
            name of the flow file
    Output :
        ng : int
            number of domains in the grid
        ni, nj, nk : int
            shape of the domain
        mach, alpha, rey, time: float
            extra numerics needed for post-processing
        q : numpy.ndarray
            Flow data of shape (ni, nj, nk, 5)

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
        self.ng = None
        self.ni, self.nj, self.nk = None, None, None
        self.mach = None
        self.alpha = None
        self.rey = None
        self.time = None
        self.q = []

    def __str__(self):
        doc = "This instance has the filename " + self.filename + "\n" + \
              "q attribute is of shape (ni, nj, nk, 5), rows representing density, momentum[*3], energy\n" \
              "For example, rho = flow.q(...,0), flow being the object.\n" \
              "Use method 'read_flow' to compute attributes:\n" \
              "ng, ni, nj, nk\n" \
              "mach, alpha, rey, time\n" \
              "q -- The flow data\n"
        return doc

    def read_flow(self):
        """Reads in the flow file and changes the instance attributes

        Parameters
        ----------

        Returns
        -------
        None

        author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
        credit: Paul Orkwis
        date: 10-05/2021
        """
        with open(self.filename, 'r') as data:
            self.ng = np.fromfile(data, dtype='i4', count=1)[0]

            # Should be looped for multiple blocks
            self.ni, self.nj, self.nk = np.fromfile(data, dtype='i4', count=3)

            self.mach, self.alpha, self.rey, self.time = np.fromfile(data, dtype='f4', count=4)

            # Read-in flow data
            self.q = np.zeros((self.ni, self.nj, self.nk, 5))

            for m in range(5):
                for k in range(self.nk):
                    for j in range(self.nj):
                        self.q[:, j, k, m] = np.fromfile(data, dtype='f4', count=self.ni)

            print("Flow data reading is successful for " + self.filename + "\n")
