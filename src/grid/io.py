import numpy as np


class GridIO:
    """Module to read-in a grid file and output grid parameters

    author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
    date: 10-05/2021

    Example:
        grid = GridIO('plate.sp.x')  # Assume file is in the path
        grid.read_grid()  # Call method to read the data
        print(grid)  # prints the docstring for grid
        # Instance attributes
        print(grid.grd.shape)  # shape of the grid data
        print(grid.ng)  # Number of blocks
    """

    def __init__(self, filename):
        self.filename = filename
        self.ng = None
        self.ni, self.nj, self.nk = None, None, None
        self.grd = None

    def __str__(self):
        doc = "This instance has the filename " + self.filename + "\n" + \
              "grd attribute is of shape 3xgrid_shape, rows representing x,y,z coordinates\n" + \
              "For example, x = grid.grd(0,...), grid being the object"
        return doc

    def read_grid(self):
        """Reads in the grid file and changes the instance attributes

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

            print('Grid data reading is successful for ' + self.filename)
