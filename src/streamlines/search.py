# This file searches for the cell in which the given point is present in the grid

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

    @staticmethod
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

    @staticmethod
    def _cell_index(self, i, j, k):
        # _Internal method to obtain the nodes of the cell in which the given point is present

        # Transform to found node to find the location of point
        # Basically looking at which quadrant the point is located
        # to find the nodes of the respective cell
        _node = self.grid.grd[i, j, k, :, self.block]
        _point_transform = self.point - _node

        # Check if point is a node in the domain
        if np.all(abs(_point_transform) <= 1e-6):
            self.cell = self._cell_nodes(i, j, k)
            self.info = 'Given point is a node in the domain with a tol of 1e-6.\n' \
                        'Interpolation will assign node properties for integration.\n' \
                        'Index of the node will be returned by cell attribute\n'
            print(self.info)
            return

        # ON BOUNDARY FOR A GENERALIZED HEXA IS SAME AS DEFAULT SEARCH
        # Removed the code for on the boundary case
        # Start the main cell modes code
        if np.all(_point_transform >= 0):
            self.cell = self._cell_nodes(i, j, k)
            return
        if _point_transform[0] <= 0 and _point_transform[1] >= 0 and _point_transform[2] >= 0:
            self.cell = self._cell_nodes(i - 1, j, k)
            return
        if _point_transform[0] <= 0 and _point_transform[1] <= 0 and _point_transform[2] >= 0:
            self.cell = self._cell_nodes(i - 1, j - 1, k)
            return
        if _point_transform[0] >= 0 and _point_transform[1] <= 0 and _point_transform[2] >= 0:
            self.cell = self._cell_nodes(i, j - 1, k)
            return
        if _point_transform[0] >= 0 and _point_transform[1] >= 0 and _point_transform[2] <= 0:
            self.cell = self._cell_nodes(i, j, k - 1)
            return
        if _point_transform[0] <= 0 and _point_transform[1] >= 0 and _point_transform[2] <= 0:
            self.cell = self._cell_nodes(i - 1, j, k - 1)
            return
        if np.all(_point_transform <= 0):
            self.cell = self._cell_nodes(i - 1, j - 1, k - 1)
            return
        if _point_transform[0] >= 0 and _point_transform[1] <= 0 and _point_transform[2] <= 0:
            self.cell = self._cell_nodes(i, j - 1, k - 1)
            return

        return

    @staticmethod
    def _find_block(self):
        # Setup to compute block number in which the point is present
        _bool_min = self.grid.grd_min <= self.point
        _bool_max = self.grid.grd_max >= self.point
        _bool = _bool_max == _bool_min

        # Test if the given point is in domain or not
        if np.all(_bool.all(axis=1) == False) or np.all(_bool_min == False) or np.all(_bool_max == False):
            self.info = 'Given point is not in the domain. The cell attribute will return "None" in search algorithm\n'
            self.cell = None
            self.point = None
            self.block = None
            print(self.info)
            return
        # Assign the block number to the attribute
        self.block = int(np.where(_bool.all(axis=1))[0][0])

        return self.block

    def compute(self, method='block_distance'):
        """
        Use the method to compute index and cell attributes

        :parameter:
            method: str
                Currently distance or block_distance

        :return:
        None

        author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
        date: 10-24/2021
        """

        # Find the block number
        self.block = self._find_block(self)
        # To check for point out-of-domain case
        if self.block is None:
            return

        if method == 'distance':
            # Compute the distance from all nodes in the grid
            _dist = np.sqrt((self.grid.grd[..., 0, :] - self.point[0]) ** 2 +
                            (self.grid.grd[..., 1, :] - self.point[1]) ** 2 +
                            (self.grid.grd[..., 2, :] - self.point[2]) ** 2)

            # Find the closest node to the point --> index.ndim = 4
            self.index = np.array(np.unravel_index(_dist.argmin(), _dist.shape))
            i, j, k, self.block = self.index[0], self.index[1], self.index[2], self.index[3]
            self._cell_index(self, i, j, k)

        if method == 'block_distance':
            # Compute distance inside the block to get the nearest node
            _i, _j, _k = self.grid.ni[self.block], self.grid.nj[self.block], self.grid.nk[self.block]
            _dist = np.sqrt((self.grid.grd[:_i, :_j, :_k, 0, self.block] - self.point[0]) ** 2 +
                            (self.grid.grd[:_i, :_j, :_k, 1, self.block] - self.point[1]) ** 2 +
                            (self.grid.grd[:_i, :_j, :_k, 2, self.block] - self.point[2]) ** 2)

            # Other methods to calculate distance. The above method is faster
            # _grd_min_point = self.grid.grd[:_i, :_j, :_k, :, self.block] - self.point
            # # _dist = np.linalg.norm(_grd_min_point, axis=-1)
            # _dist = np.sqrt(np.einsum("ijkl,ijkl->ijk", _grd_min_point, _grd_min_point))

            self.index = np.array(np.unravel_index(_dist.argmin(), _dist.shape))
            i, j, k = self.index[0], self.index[1], self.index[2]
            self._cell_index(self, i, j, k)

        if method == 'c-space':
            # To run c-space global point location is needed
            # credit: Sadarjoen et al.
            # title: Particle tracing algorithms for 3D Curvilinear grids

            # Transform given point from p-space to c-space using newton-raphson
            # This is only performed once to get the initial c-space point
            self.point = self.p2c(self.point)

    def c2p(self, eps):
        """
        Method to convert c-space point to p-space

        Args:
            eps: c-space co-ordinates

        Returns:
            point: p-space co-ordinates

        """
        _eps0, _eps1, _eps2 = eps.astype(int)
        _alpha, _beta, _gamma = np.modf(eps)[0]  # same as eps % 1.0

        self.cell = self._cell_nodes(_eps0, _eps1, _eps2)
        _cell_grd = self.grid.grd[self.cell[:, 0], self.cell[:, 1], self.cell[:, 2], :, self.block]

        point = (1 - _alpha) * (1 - _beta) * (1 - _gamma) * _cell_grd[0] + \
                     _alpha  * (1 - _beta) * (1 - _gamma) * _cell_grd[1] + \
                     _alpha  *      _beta  * (1 - _gamma) * _cell_grd[2] + \
                (1 - _alpha) *      _beta  * (1 - _gamma) * _cell_grd[3] + \
                (1 - _alpha) * (1 - _beta) *      _gamma  * _cell_grd[4] + \
                     _alpha  * (1 - _beta) *      _gamma  * _cell_grd[5] + \
                     _alpha  *      _beta *       _gamma  * _cell_grd[6] + \
                (1 - _alpha) *      _beta  *      _gamma  * _cell_grd[7]

        return point

    def p2c(self, point):
        """
        Method to convert p-space point to c-space
        As there is no direct analytical equation. We use Newton-Raphson
        Args:
            point: p-space co-ordinates

        Returns:
            eps: c-space co-ordinates
        """

        self.compute(method='block_distance')
        # Setup initial variables
        _cell_grd = self.grid.grd[self.cell[:, 0], self.cell[:, 1], self.cell[:, 2], :, self.block]

        # Start Newton-Raphson
        _iter = 0
        # Initial guess
        _eps = self.cell[0] + np.random.rand(3)

        # TODO: Replace the compute method with a better initial guess to speed up
        # Initial guess
        # _eps0 = np.random.randint(0, self.grid.ni[self.block])
        # _eps1 = np.random.randint(0, self.grid.nj[self.block])
        # _eps2 = np.random.randint(0, self.grid.nk[self.block])
        # _eps = np.array((_eps0, _eps1, _eps2))

        while True:
            # Check if taking too long
            if _iter >= 1e3:
                print('Newton-Raphson did not converge. Try again!')
                return

            # Currently, using one Jacobian per cell
            # TODO: Need to use tri-linear interpolation to get Jacobian
            _eps0, _eps1, _eps2 = _eps.astype(int)
            _J_inv = self.grid.m2[_eps0, _eps1, _eps2, :, :, self.block]

            # Transform from c to p-space
            _pred_point = self.c2p(_eps)

            # Difference b/w predicted point to given point
            _delta_point = point - _pred_point

            # End newton-raphson if condition is met
            if sum(abs(_delta_point)) <= 1e-6:
                _eps0, _eps1, _eps2 = _eps.astype(int)
                self.cell = self._cell_nodes(_eps0, _eps1, _eps2)
                return _eps

            # Transform from p to c-space
            _delta_eps = np.matmul(_J_inv, _delta_point)
            if sum(abs(_delta_eps)) <= 1e-6:
                _eps0, _eps1, _eps2 = _eps.astype(int)
                self.cell = self._cell_nodes(_eps0, _eps1, _eps2)
                return _eps

            # New point
            _eps += _delta_eps
            _iter += 1

            pass


