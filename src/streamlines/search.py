# This file searches for the cell in which the given point is present in the grid

import numpy as np


# noinspection SpellCheckingInspection
class Search:
    """Module to search for the cell in which the given point is present

    ...

    Attributes
    ----------
    Input :
        grid : src.io.plot3dio.GridIO
            Grid object created from GridIO
        ppoint: list
            A float list of shape 3 -- Representing x, y, z of a point
    Output :
        cpoint: numpy.ndarray
            ppoint location in c-space. Will be computed if calculating in c-space
        index: numpy.ndarray
            Indices of closest node to the given point
        cell: numpy.ndarray
            Indices of the cell in which the given point is present
        info: str
            Information about the point location

    Methods
    -------
    compute()
        Finds the location of given point in the grid using the given search-method
        search-methods:
            distance, block-distance, p-space, c-space

    p2c()
        Method to convert location of point in physical space to computational space
        Rarely used

    c2p()
        Method to convert from c-space to p-space
        Used in integration algorithms to get new cell

    author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
    date: 10-24/2021
        """

    def __init__(self, grid, ppoint):
        self.grid = grid
        self.ppoint = ppoint
        self.cpoint = None
        self.index = None
        self.cell = None
        self.info = None
        self.block = None

    def __str__(self):
        doc = "This instance takes in the grid of shape " + self.grid.grd.shape + \
              "\nand the searches for the point " + self.ppoint + " in the grid.\n" \
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
                          [_i, _j + 1, _k + 1]], dtype=int)
        return _cell

    @staticmethod
    def _cell_index(self, i, j, k):
        # _Internal method to obtain the nodes of the cell in which the given point is present

        # Transform to found node to find the location of point
        # Basically looking at which quadrant the point is located
        # to find the nodes of the respective cell
        _node = self.grid.grd[i, j, k, :, self.block]
        _point_transform = self.ppoint - _node

        # Check if point is a node in the domain
        if np.all(abs(_point_transform) <= 1e-12):
            self.cell = self._cell_nodes(i, j, k)
            self.info = 'Given point is a node in the domain with a tol of 1e-12.\n' \
                        'Interpolation will assign node properties for integration.\n' \
                        'Index of the node will be returned by cell attribute\n'
            # print(self.info)
            return

        # ON BOUNDARY FOR A GENERALIZED HEXA IS SAME AS DEFAULT SEARCH
        # Removed the code for on the boundary case
        # Start the main cell nodes code
        if np.all(_point_transform >= 1e-6):
            self.cell = self._cell_nodes(i, j, k)
            return
        if _point_transform[0] <= 1e-6 and _point_transform[1] >= 1e-6 and _point_transform[2] >= 1e-6:
            self.cell = self._cell_nodes(i - 1, j, k)
            return
        if _point_transform[0] <= 1e-6 and _point_transform[1] <= 1e-6 and _point_transform[2] >= 1e-6:
            self.cell = self._cell_nodes(i - 1, j - 1, k)
            return
        if _point_transform[0] >= 1e-6 and _point_transform[1] <= 1e-6 and _point_transform[2] >= 1e-6:
            self.cell = self._cell_nodes(i, j - 1, k)
            return
        if _point_transform[0] >= 1e-6 and _point_transform[1] >= 1e-6 and _point_transform[2] <= 1e-6:
            self.cell = self._cell_nodes(i, j, k - 1)
            return
        if _point_transform[0] <= 1e-6 and _point_transform[1] >= 1e-6 and _point_transform[2] <= 1e-6:
            self.cell = self._cell_nodes(i - 1, j, k - 1)
            return
        if np.all(_point_transform <= 1e-6):
            self.cell = self._cell_nodes(i - 1, j - 1, k - 1)
            return
        if _point_transform[0] >= 1e-6 and _point_transform[1] <= 1e-6 and _point_transform[2] <= 1e-6:
            self.cell = self._cell_nodes(i, j - 1, k - 1)
            return

        return

    @staticmethod
    def _find_block(self):
        # _Internal method to find the block
        # Setup to compute block number in which the point is present
        _bool_min = self.grid.grd_min <= self.ppoint
        _bool_max = self.grid.grd_max >= self.ppoint
        _bool = _bool_max == _bool_min

        # Test if the given point is in domain or not
        if np.all(_bool.all(axis=1) == False) or np.all(_bool_min == False) or np.all(_bool_max == False):
            self.info = 'Given point is not in the domain. The cell attribute will return "None" in search algorithm\n'
            self.cell = None
            self.ppoint = None
            self.cpoint = None
            self.block = None
            # print(self.info)
            return
        # Assign the block number to the attribute
        self.block = int(np.where(_bool.all(axis=1))[0][0])

        return self.block

    def compute(self, method='block_distance'):
        """
        Use the method to compute index and cell attributes

        parameter:
            method: str
                Currently distance or block_distance

        return:
        None

        author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
        date: 10-24/2021
        """

        # Find the block number
        self.block = self._find_block(self)
        # To check for point out-of-domain case
        if self.block is None:
            return

        match method:

            case 'distance':
                # Compute the distance from all nodes in the grid
                _dist = np.sqrt((self.grid.grd[..., 0, :] - self.ppoint[0]) ** 2 +
                                (self.grid.grd[..., 1, :] - self.ppoint[1]) ** 2 +
                                (self.grid.grd[..., 2, :] - self.ppoint[2]) ** 2)

                # Find the closest node to the point --> index.ndim = 4
                self.index = np.array(np.unravel_index(_dist.argmin(), _dist.shape))
                i, j, k, self.block = self.index[0], self.index[1], self.index[2], self.index[3]
                self._cell_index(self, i, j, k)
                # Check for the end of the domain case
                if max(self.cell[:, 0]) > self.grid.ni[self.block] - 1 or \
                        max(self.cell[:, 1]) > self.grid.nj[self.block] - 1 or \
                        max(self.cell[:, 2]) > self.grid.nk[self.block] - 1:
                    self.cpoint = None
                    self.ppoint = None
                    return

            case 'block_distance':
                # Compute distance inside the block to get the nearest node
                _i, _j, _k = self.grid.ni[self.block], self.grid.nj[self.block], self.grid.nk[self.block]
                _dist = np.sqrt((self.grid.grd[:_i, :_j, :_k, 0, self.block] - self.ppoint[0]) ** 2 +
                                (self.grid.grd[:_i, :_j, :_k, 1, self.block] - self.ppoint[1]) ** 2 +
                                (self.grid.grd[:_i, :_j, :_k, 2, self.block] - self.ppoint[2]) ** 2)

                # Other methods to calculate distance. The above method is faster
                # _grd_min_point = self.grid.grd[:_i, :_j, :_k, :, self.block] - self.point
                # _dist = np.linalg.norm(_grd_min_point, axis=-1)
                # _dist = np.sqrt(np.einsum("ijkl,ijkl->ijk", _grd_min_point, _grd_min_point))

                self.index = np.array(np.unravel_index(_dist.argmin(), _dist.shape))
                i, j, k = self.index
                self._cell_index(self, i, j, k)
                # Check for the end of the domain case
                if max(self.cell[:, 0]) > self.grid.ni[self.block] - 1 or \
                        max(self.cell[:, 1]) > self.grid.nj[self.block] - 1 or \
                        max(self.cell[:, 2]) > self.grid.nk[self.block] - 1 or np.any(self.cell < 0):
                    print('Block search returned wrong cell! Point position lost.\n')
                    self.cpoint = None
                    self.ppoint = None
                    return

            case 'p-space':
                # Search for given point using newton-raphson
                # credit: Sadarjoen et al.
                # title: Particle tracing algorithms for 3D Curvilinear grids
                # This search is performed every single time to find the given point
                self.cpoint = self.p2c(self.ppoint)

            case 'c-space':
                # To run c-space global point location is needed
                # credit: Sadarjoen et al.
                # title: Particle tracing algorithms for 3D Curvilinear grids

                # Transform given point from p-space to c-space using newton-raphson
                # This is only performed once to get the initial c-space point
                self.cpoint = self.p2c(self.ppoint)

    def c2p(self, _cpoint):
        """
        Method to convert c-space point to p-space
        self.block is kept constant through the c-space algos
        self.block is checked and switched in streamlines algo
        This method is commonly used to test point location as well
        in different algorithms

        Args:
            _cpoint: c-space co-ordinates

        Returns:
            _ppoint: p-space co-ordinates

        """
        self.cpoint = _cpoint
        _eps0, _eps1, _eps2 = _cpoint.astype(int)
        _alpha, _beta, _gamma = np.modf(_cpoint)[0]  # same as eps % 1.0

        # Check if the given point is in the domain
        if _eps0 >= self.grid.ni[self.block]-1 or _eps1 >= self.grid.nj[self.block]-1 or \
                _eps2 >= self.grid.nk[self.block]-1:
            self.cpoint = None
            self.ppoint = None
            return self.ppoint

        # Determine in which cell the current point is present
        self.cell = self._cell_nodes(_eps0, _eps1, _eps2)
        _cell_grd = self.grid.grd[self.cell[:, 0], self.cell[:, 1], self.cell[:, 2], :, self.block]

        # Calculate the location in p-space
        self.ppoint = (1 - _alpha) * (1 - _beta) * (1 - _gamma) * _cell_grd[0] + \
                _alpha  * (1 - _beta) * (1 - _gamma) * _cell_grd[1] + \
                _alpha  *      _beta  * (1 - _gamma) * _cell_grd[2] + \
                (1 - _alpha) *      _beta  * (1 - _gamma) * _cell_grd[3] + \
                (1 - _alpha) * (1 - _beta) *      _gamma  * _cell_grd[4] + \
                _alpha  * (1 - _beta) *      _gamma  * _cell_grd[5] + \
                _alpha  *      _beta *       _gamma  * _cell_grd[6] + \
                (1 - _alpha) *      _beta  *      _gamma  * _cell_grd[7]

        return self.ppoint

    def p2c(self, _ppoint):
        """
        Method to convert p-space point to c-space
        As there is no direct analytical equation. We use Newton-Raphson
        Args:
            _ppoint: p-space co-ordinates

        Returns:
            eps: c-space co-ordinates
        """
        global _cpoint
        self.ppoint = _ppoint

        if self.block is None:
            self.block = self._find_block(self)

        # Start Newton-Raphson
        _iter = 0
        # Initial guess
        try:
            _cpoint is not None
        except:
            _cpoint = np.array([self.grid.ni[self.block], self.grid.nj[self.block], self.grid.nk[self.block]]) / 2 + \
                np.random.randn(3)

        while True:
            # Check if taking too long
            if _iter >= 1e3:
                print('**ERROR** Newton-Raphson did not converge. Try again!\n'
                      'Possible reason might be the point might be too close to the end of a domain')
                self.ppoint, self.cpoint = None, None
                return

            # Check for out-of-domain case and reset the point to in-domain
            _eps0, _eps1, _eps2 = _cpoint.astype(int)
            _alpha, _beta, _gamma = np.modf(_cpoint)[0]
            if _eps0 + 1 >= self.grid.ni[self.block]:
                _cpoint[0] = self.grid.ni[self.block] - 1 - _alpha
            if _eps1 + 1 >= self.grid.nj[self.block]:
                _cpoint[1] = self.grid.nj[self.block] - 1 - _beta
            if _eps2 + 1 >= self.grid.nk[self.block]:
                _cpoint[2] = self.grid.nk[self.block] - 1 - _gamma

            # Compute eps after the point is reset
            _eps0, _eps1, _eps2 = _cpoint.astype(int)
            self.cell = self._cell_nodes(_eps0, _eps1, _eps2)

            # Calculate J_inv for the point --> Uses tri-linear interpolation
            _cell_J_inv = self.grid.m2[self.cell[:, 0], self.cell[:, 1], self.cell[:, 2], :, :, self.block]
            _J_inv = (1 - _alpha) * (1 - _beta) * (1 - _gamma) * _cell_J_inv[0] + \
                     _alpha * (1 - _beta) * (1 - _gamma) * _cell_J_inv[1] + \
                     _alpha * _beta * (1 - _gamma) * _cell_J_inv[2] + \
                     (1 - _alpha) * _beta * (1 - _gamma) * _cell_J_inv[3] + \
                     (1 - _alpha) * (1 - _beta) * _gamma * _cell_J_inv[4] + \
                     _alpha * (1 - _beta) * _gamma * _cell_J_inv[5] + \
                     _alpha * _beta * _gamma * _cell_J_inv[6] + \
                     (1 - _alpha) * _beta * _gamma * _cell_J_inv[7]

            # Transform from c to p-space
            _pred_ppoint = self.c2p(_cpoint)

            # Difference b/w predicted point to given point
            _delta_ppoint = _ppoint - _pred_ppoint

            # End newton-raphson if condition is met
            # TODO: Condition needs to be adapted based on Jacobian
            # TODO: Need to improve by normalizing the data
            _tol = 1e-12 * self.grid.J[self.cell[0, 0], self.cell[0, 1], self.cell[0, 2], self.block]
            if _tol <= 1e-12:
                _tol = 1e-12
            if sum(abs(_delta_ppoint)) <= _tol:
                _eps0, _eps1, _eps2 = _cpoint.astype(int)
                # same as self.cell = self._cell_nodes(_eps0, _eps1, _eps2)
                self._cell_index(self, _eps0, _eps1, _eps2)
                self.cpoint = _cpoint
                self.ppoint = _pred_ppoint
                return _cpoint

            # Transform from p to c-space
            _delta_cpoint = np.matmul(_J_inv, _delta_ppoint)

            # Save old point
            _cpoint_old = _cpoint.copy()

            # Update point
            _cpoint += _delta_cpoint
            # Update the point to zero if less than zero
            _cpoint[_cpoint < 0] = 0
            _cpoint = abs(_cpoint)
            _iter += 1

            pass


