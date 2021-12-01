# This file searches for the cell in which the given point is present in the grid
# TODO: Implement KDTree or interpolation search for faster times

import numpy as np


class Search:
    """Module to search for the cell in which the given point is present

    ...

    Attributes
    ----------
    Input :
        grid : src.io.plot3dio.GridIO
            Grid object created from GridIO
        point: list
            A float list of shape 3 -- Representing x, y, z of a point
    Output :
        index: numpy.ndarray
            Indices of closest node to the given point
        cell: numpy.ndarray
            Indices of the cell in which the given point is present
        info: str
            Information about the point location

    Methods
    -------
    read_grid()
        returns the output attributes

        Example:
            grid = GridIO('plate.sp.x')  # Assume file is in the path
            nodes = Search(grid, [0.5, 0.7, -10.7])  # grid object is created from GridIO
            print(nodes)  # prints the docstring for grid
            nodes.compute()  # Call method to search for the cell
            # Instance attributes
            print(nodes.index)  # closest node in the grid to the given point
            print(nodes.cell)  # prints the nodes of the cell

        author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
        date: 10-24/2021
        """

    def __init__(self, grid, point):
        self.grid = grid
        self.point = point
        self.index = None
        self.cell = None
        self.info = None
        self.block = None

    def __str__(self):
        doc = "This instance takes in the grid of shape " + self.grid.grd.shape + \
              "\nand the searches for the point " + self.point + " in the grid.\n" \
              "Use method 'compute' to find (attributes) the closest 'index' and the nodes of the 'cell'.\n"
        return doc

    def compute(self, method='distance'):
        """
        Use the method to compute index and cell attributes

        :parameter:

        :return:
        None

        author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
        date: 10-24/2021
        """

        # Setup to compute block number in which the point is present
        _bool_min = self.grid.grd_min <= self.point
        _bool_max = self.grid.grd_max >= self.point
        _bool = _bool_max == _bool_min

        # Test if the given point is in domain or not
        if np.all(_bool.all(axis=1) == False):
            self.info = 'Given point is not in the domain. The cell attribute will return "None"\n'
            self.cell = None
            print(self.info)
            return
        # Assign the block number to the attribute
        self.block = int(np.where(_bool.all(axis=1))[0][0])

        if method == 'distance':
            # Compute the distance from all nodes in the grid
            _dist = np.sqrt((self.grid.grd[..., 0, :] - self.point[0]) ** 2 +
                            (self.grid.grd[..., 1, :] - self.point[1]) ** 2 +
                            (self.grid.grd[..., 2, :] - self.point[2]) ** 2)

            # Find the closest node to the point --> index.ndim = 4
            self.index = np.array(np.unravel_index(_dist.argmin(), _dist.shape))
            i, j, k, self.block = self.index[0], self.index[1], self.index[2], self.index[3]
            _node = self.grid.grd[i, j, k, :, self.block]

        if method == 'block_distance':
            # Compute distance inside the block to get the nearest node
            _i, _j, _k = self.grid.ni[self.block], self.grid.nj[self.block], self.grid.nk[self.block]
            _dist = np.sqrt((self.grid.grd[:_i, :_j, :_k, 0, self.block] - self.point[0]) ** 2 +
                            (self.grid.grd[:_i, :_j, :_k, 1, self.block] - self.point[1]) ** 2 +
                            (self.grid.grd[:_i, :_j, :_k, 2, self.block] - self.point[2]) ** 2)

            self.index = np.array(np.unravel_index(_dist.argmin(), _dist.shape))
            i, j, k = self.index[0], self.index[1], self.index[2]
            _node = self.grid.grd[i, j, k, :, self.block]

        def _cell_nodes(_i, _j, _k):
            # _Internal method to get the nodes of a cell
            _cell = np.array([[_i, _j, _k],
                              [_i + 1, _j, _k],
                              [_i + 1, _j + 1, _k],
                              [_i, _j + 1, _k],
                              [_i, _j, _k + 1],
                              [_i + 1, _j, _k + 1],
                              [_i + 1, _j + 1, _k + 1],
                              [_i, _j + 1, _k + 1]])
            return _cell

        # Transform to found node to find the location of point
        # Basically looking at which quadrant the point is located
        # to find the nodes of the respective cell
        _point_transform = self.point - _node
        # Check if point is a node in the domain
        if np.all(abs(_point_transform) <= 1e-6):
            self.cell = _cell_nodes(i, j, k)
            self.info = 'Given point is a node in the domain with a tol of 1e-6.\n' \
                        'Interpolation will assign node properties for integration.\n' \
                        'Index of the node will be returned by cell attribute\n'
            print(self.info)
            return
        # ON BOUNDARY FOR A GENERALIZED HEXA IS SAME AS DEFAULT SEARCH
        # Removed the code for on the boundary case
        # Start the main cell modes code
        if np.all(_point_transform >= 0):
            self.cell = _cell_nodes(i, j, k)
            return
        if _point_transform[0] <= 0 and _point_transform[1] >= 0 and _point_transform[2] >= 0:
            self.cell = _cell_nodes(i - 1, j, k)
            return
        if _point_transform[0] <= 0 and _point_transform[1] <= 0 and _point_transform[2] >= 0:
            self.cell = _cell_nodes(i - 1, j - 1, k)
            return
        if _point_transform[0] >= 0 and _point_transform[1] <= 0 and _point_transform[2] >= 0:
            self.cell = _cell_nodes(i, j - 1, k)
            return
        if _point_transform[0] >= 0 and _point_transform[1] >= 0 and _point_transform[2] <= 0:
            self.cell = _cell_nodes(i, j, k - 1)
            return
        if _point_transform[0] <= 0 and _point_transform[1] >= 0 and _point_transform[2] <= 0:
            self.cell = _cell_nodes(i - 1, j, k - 1)
            return
        if np.all(_point_transform <= 0):
            self.cell = _cell_nodes(i - 1, j - 1, k - 1)
            return
        if _point_transform[0] >= 0 and _point_transform[1] <= 0 and _point_transform[2] <= 0:
            self.cell = _cell_nodes(i, j - 1, k - 1)
            return
