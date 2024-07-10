# Use tri-linear interpolation to get data at the given point

import numpy as np
from ..function.variables import Variables


class Interpolation:
    """
    Module to do the cell data interpolation to the given point

    ...

    Attributes
    ----------
    Input:
        flow : .io.plot3dio.FlowIO
            Flow object created from FlowIO
        idx: .streamlines.Search
            Index object created from Search
    Output:
        q : ndarray
            Interpolated data at the given point
        nb, ni, nj, nk: int
            Dimensionless data from flow
        mach, alpha, rey, time: float
            Dimensionless data from flow
        J_inv: ndarray
            Inverse grid metrics at the given point

    Methods
    -------
    compute(method='p-space')
        Interpolates the data onto a given point
        p-space or c-space can be specified

    Example:
        grid = GridIO('../data/plate_data/plate.sp.x')
        flow = FlowIO('../data/plate_data/sol-0000010.q')
        idx = Search(grid, [8.5, 0.5, 0.01])
        interp_data = Interpolation(grid, flow, idx)

        grid.read_grid()
        flow.read_flow()
        idx.compute()
        interp_data.compute()  # method='p-space' is default

    author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
    date: 10-29/2021
    """

    def __init__(self, flow, idx):
        self.flow = flow
        self.idx = idx
        self.q = None
        self.nb = None
        self.ni, self.nj, self.nk = [None] * 3
        self.mach, self.alpha, self.rey, self.time = [None] * 4
        self.J = None
        self.J_inv = None
        self.level = [0, 0, 0]
        self.rbf_kernel = 'thin_plate_spline'
        self.adaptive = None
        self.rbf_epsilon = 1  # Default for RBF interpolation
        self.method = None
        # for unsteady case
        self.flow_old = None
        self.time = []

    def __str__(self):
        doc = "This instance uses " + self.flow.filename + " as the flow file " \
                                                           "to compute properties at " + self.idx.ppoint + "\n"
        return doc

    @staticmethod
    def _shock_cell_check(self):
        # Inspect shock cell and assign nearest interpolation
        i0, j0, k0 = self.idx.cell[0, 0], self.idx.cell[0, 1], self.idx.cell[0, 2]
        i1, j1, k1 = self.idx.cell[1, 0], self.idx.cell[1, 1], self.idx.cell[1, 2]
        # compute velocity, mach
        _var = Variables(self.flow)
        _var.compute_mach()
        # _grad_v = _J_inv * (v1 - v0)
        _grad_v = self.idx.grid.m2[i0, j0, k0, :, self.idx.block] * (
                _var.velocity[i1, j1, k1, :, self.idx.block]
                - _var.velocity[i0, j0, k0, :, self.idx.block]
        )
        # compute norm to get the unit vector
        _grad_v = _grad_v / np.linalg.norm(_grad_v)
        # mach vector -- mach * unit velocity vector
        _mach0 = (_var.mach[i0, j0, k0] * _var.velocity[i0, j0, k0, :, self.idx.block]
                  / _var.velocity_magnitude[i0, j0, k0, self.idx.block])
        _mach1 = (_var.mach[i1, j1, k1] * _var.velocity[i1, j1, k1, :, self.idx.block]
                  / _var.velocity_magnitude[i1, j1, k1, self.idx.block])
        # normal mach vector
        _mach_n0 = np.linalg.norm(np.dot(_mach0, _grad_v))
        _mach_n1 = np.linalg.norm(np.dot(_mach1, _grad_v))

        return _mach_n0, _mach_n1

    def compute(self, method='p-space'):
        """
        Find interpolated plot3d data and grid metrics at a given point
        Args:
            method: Available methods for interpolation
            1. p-space: Tri-linear interpolation in physical co-ordinates
            2. c-space: Tri-linear interpolation in c-space. Must-use for the c-space algo

        Returns:
            q --> Attribute with the interpolated flow data.
            Has same ndim as q attribute from flow object

        """
        # this object is used for integration without changing much of the code
        self.method = method

        # Assign data from q file to keep the format for further computations
        self.nb = self.flow.nb
        self.ni, self.nj, self.nk = self.flow.ni, self.flow.nj, self.flow.nk
        self.mach, self.alpha, self.rey, self.time = self.flow.mach, self.flow.alpha, self.flow.rey, self.flow.time

        # If out of domain return
        if self.idx.ppoint is None and self.idx.cpoint is None:
            print('Cannot run interpolation. The given point is out of domain.\n')
            self.q = None
            return

        # Obtain grid data from nodes found from the search method
        _cell_grd = np.zeros(self.idx.cell.shape)
        _cell_grd = self.idx.grid.grd[self.idx.cell[:, 0], self.idx.cell[:, 1], self.idx.cell[:, 2], :, self.idx.block]

        # Obtain flow data from nodes found
        _cell_q = np.zeros((self.idx.cell.shape[0], 5))
        _cell_q = self.flow.q[self.idx.cell[:, 0], self.idx.cell[:, 1], self.idx.cell[:, 2], :, self.idx.block]

        match method:
            # simple oblique shock -- use nearest neighbor
            case 'simple_oblique_shock':
                if self.idx.cell.shape == (8, 3) and self.idx.info is None:
                    _distance = np.sqrt(np.sum((_cell_grd - self.idx.ppoint) ** 2, axis=1))
                    # nearest neighbor index
                    _nn = np.argmin(_distance)
                    # assign the nearest neighbor to the given point
                    self.q = _cell_q[_nn]
                    self.q = self.q.reshape((1, 1, 1, -1, 1))

                # if the point is node return node data
                if self.idx.info == 'Given point is a node in the domain with a tol of 1e-12.\n' \
                                    'Interpolation will assign node properties for integration.\n' \
                                    'Index of the node will be returned by cell attribute\n':
                    self.q = _cell_q[0]  # the first node is the point based on search method
                    self.q = self.q.reshape((1, 1, 1, -1, 1))

            # "Tri"Linear interpolation
            case 'p-space':
                # If the node is found in a cell
                if self.idx.cell.shape == (8, 3) and self.idx.info is None:
                    def _f(self, _m, _n, _arr1, _arr2, _axis, _paxis, _data1, _data2):
                        """
                        Internal function used for linearly interpolating data
                        :param _m: int
                            point-1 for interpolation
                        :param _n: int
                            point-2 for interpolation
                        :param _arr1: ndarray
                            data with point-1
                        :param _arr2: ndarray
                            data with point-2
                        :param _axis: int
                            assign x, y or z axis to consider for interpolation
                        :param _paxis: int
                            assign point's x, y, or z for interpolation
                        :param _data1: ndarray/list
                            data at point-1 to be interpolated. List is provided when moving across axis
                        :param _data2: ndarray/list
                            data at point-2 to be interpolated. List is provided when moving across axis
                        :return: None

                        author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
                        date: 10-29/2021
                        """
                        _a = (_arr2[_n, _axis] - self.idx.ppoint[_paxis]) / (_arr2[_n, _axis] - _arr1[_m, _axis]) * _data1[_m]
                        _b = (self.idx.ppoint[_paxis] - _arr1[_m, _axis]) / (_arr2[_n, _axis] - _arr1[_m, _axis]) * _data2[_n]
                        return _a + _b

                    # 0123 are vertices of quadrilateral
                    # Doing interpolation to x location in self.idx.ppoint
                    # Values in the function help set the linear interpolation in x-direction
                    _cell_grd_01 = _f(self, 0, 1, _cell_grd, _cell_grd, 0, 0, _cell_grd, _cell_grd)
                    _cell_grd_32 = _f(self, 3, 2, _cell_grd, _cell_grd, 0, 0, _cell_grd, _cell_grd)
                    _cell_q_01 = _f(self, 0, 1, _cell_grd, _cell_grd, 0, 0, _cell_q, _cell_q)
                    _cell_q_32 = _f(self, 3, 2, _cell_grd, _cell_grd, 0, 0, _cell_q, _cell_q)

                    # Doing interpolation to the point projection on the face
                    _cell_grd_0123 = _f(self, 1, 1, _cell_grd_01, _cell_grd_32, None, 1,
                                        [None, _cell_grd_01], [None, _cell_grd_32])
                    _cell_q_0123 = _f(self, 1, 1, _cell_grd_01, _cell_grd_32, None, 1,
                                      [None, _cell_q_01], [None, _cell_q_32])

                    # 4567 face
                    # Doing interpolation to x location in self.idx.ppoint
                    # Values in the function help set the linear interpolation in y-direction
                    _cell_grd_45 = _f(self, 4, 5, _cell_grd, _cell_grd, 0, 0, _cell_grd, _cell_grd)
                    _cell_grd_76 = _f(self, 7, 6, _cell_grd, _cell_grd, 0, 0, _cell_grd, _cell_grd)
                    _cell_q_45 = _f(self, 4, 5, _cell_grd, _cell_grd, 0, 0, _cell_q, _cell_q)
                    _cell_q_76 = _f(self, 7, 6, _cell_grd, _cell_grd, 0, 0, _cell_q, _cell_q)

                    # Doing interpolation to the point projection on the face
                    _cell_grd_4567 = _f(self, 1, 1, _cell_grd_45, _cell_grd_76, None, 1, [None, _cell_grd_45],
                                        [None, _cell_grd_76])
                    _cell_q_4567 = _f(self, 1, 1, _cell_grd_45, _cell_grd_76, None, 1, [None, _cell_q_45],
                                      [None, _cell_q_76])

                    # Do the shock cell check
                    if self.idx.cell.shape == (8, 3) and self.idx.info is None:
                        if self.adaptive == 'shock':
                            _mach_n0, _mach_n1 = self._shock_cell_check(self)
                            # if shock is in the cell _mach_n0 > 1 > _mach_n1
                            if _mach_n0 > 1 > _mach_n1:
                                _distance = np.sqrt(np.sum((_cell_grd - self.idx.ppoint) ** 2, axis=1))
                                # nearest neighbor index
                                _nn = np.argmin(_distance)
                                # assign the nearest neighbor to the given point
                                self.q = _cell_q[_nn]
                                self.q = self.q.reshape((1, 1, 1, -1, 1))
                                return
                            else:
                                pass

                    # Doing data interpolation to the given point based on the face points
                    self.q = _f(self, 2, 2, _cell_grd_0123, _cell_grd_4567, None, 2,
                                [None, None, _cell_q_0123], [None, None, _cell_q_4567])
                    self.q = self.q.reshape((1, 1, 1, -1, 1))

                # if the point is node return node data
                if self.idx.info == 'Given point is a node in the domain with a tol of 1e-12.\n' \
                                    'Interpolation will assign node properties for integration.\n'\
                                    'Index of the node will be returned by cell attribute\n':
                    self.q = _cell_q[0]  # the first node is the point based on search method
                    self.q = self.q.reshape((1, 1, 1, -1, 1))

            case 'c-space':
                """
                Point from idx is in c-space. Tri-linear interpolation is performed.
                Equation can be found in Sadarjoen et al.
                """

                # if the point is node return node data
                if self.idx.info == 'Given point is a node in the domain with a tol of 1e-12.\n' \
                                    'Interpolation will assign node properties for integration.\n' \
                                    'Index of the node will be returned by cell attribute\n':
                    self.q = _cell_q[0]
                    self.q = self.q.reshape((1, 1, 1, -1, 1))
                    return

                # Interpolate m1 and m2 for use in integration; m1 is used in ppath only!
                # m1 -- indicated as J (determinant of m1 is J)
                _cell_J = self.idx.grid.m1[self.idx.cell[:, 0], self.idx.cell[:, 1], self.idx.cell[:, 2], :, :, self.idx.block]
                # m2 -- indicated as J_inv (determinant of m2 is J_inv)
                _cell_J_inv = self.idx.grid.m2[self.idx.cell[:, 0], self.idx.cell[:, 1], self.idx.cell[:, 2], :, :, self.idx.block]
                _alpha, _beta, _gamma = self.idx.cpoint - self.idx.cell[0]

                # Do the shock cell check
                if self.idx.cell.shape == (8, 3) and self.idx.info is None:
                    if self.adaptive == "shock":
                        _mach_n0, _mach_n1 = self._shock_cell_check(self)
                        # if shock is in the cell _mach_n0 > 1 > _mach_n1
                        if _mach_n0 > 1 > _mach_n1:
                            _distance = np.sqrt(np.sum((self.idx.cell - self.idx.cpoint) ** 2, axis=1))
                            # nearest neighbor index
                            _nn = np.argmin(_distance)
                            # assign the nearest neighbor to the given point
                            self.q = _cell_q[_nn]
                            self.J = _cell_J[_nn]
                            self.J_inv = _cell_J_inv[_nn]
                            self.q = self.q.reshape((1, 1, 1, -1, 1))
                            return
                        else:
                            pass

                def _eqn(_alpha, _beta, _gamma, _var):
                    _fun = (1 - _alpha) * (1 - _beta) * (1 - _gamma) * _var[0] + \
                                _alpha  * (1 - _beta) * (1 - _gamma) * _var[1] + \
                                _alpha  *      _beta  * (1 - _gamma) * _var[2] + \
                           (1 - _alpha) *      _beta  * (1 - _gamma) * _var[3] + \
                           (1 - _alpha) * (1 - _beta) *      _gamma  * _var[4] + \
                                _alpha  * (1 - _beta) *      _gamma  * _var[5] + \
                                _alpha  *      _beta *       _gamma  * _var[6] + \
                           (1 - _alpha) *      _beta  *      _gamma  * _var[7]

                    return _fun

                self.q = _eqn(_alpha, _beta, _gamma, _cell_q)
                self.J = _eqn(_alpha, _beta, _gamma, _cell_J)
                self.J_inv = _eqn(_alpha, _beta, _gamma, _cell_J_inv)

                self.q = self.q.reshape((1, 1, 1, -1, 1))

            case 'rbf-p-space':
                """
                Raidal basis function interpolation in physical space
                """
                # if the point is node return node data
                if self.idx.info == 'Given point is a node in the domain with a tol of 1e-12.\n' \
                                    'Interpolation will assign node properties for integration.\n' \
                                    'Index of the node will be returned by cell attribute\n':
                    self.q = _cell_q[0]
                    self.q = self.q.reshape((1, 1, 1, -1, 1))
                    return

                from scipy.interpolate import RBFInterpolator as rbf
                # add adjacent cells to the cell list
                # TODO: debug for multiblock case -- currently defaults to single block; if not enough cells in
                #  the block, it default to using rbf
                if np.any(np.array(self.level) > 0):
                    _level_cell = self.idx.cell
                    if (np.any(self.idx.cell[:, 0] + self.level[0] >= self.idx.grid.ni - 1) or
                            np.any(self.idx.cell[:, 0] - self.level[0] <= 0)):
                        pass
                    else:
                        for _i in range(self.level[0]):
                            _level_cell = np.vstack((_level_cell, self.idx.cell[1] + [_i+1, 0, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[2] + [_i+1, 0, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[5] + [_i+1, 0, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[6] + [_i+1, 0, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[0] - [_i+1, 0, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[3] - [_i+1, 0, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[4] - [_i+1, 0, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[7] - [_i+1, 0, 0]))
                    if (np.any(self.idx.cell[:, 1] + self.level[1] >= self.idx.grid.nj - 1) or
                            np.any(self.idx.cell[:, 1] - self.level[1] <= 0)):
                        pass
                    else:
                        for _i in range(self.level[1]):
                            _level_cell = np.vstack((_level_cell, self.idx.cell[2] + [0, _i+1, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[3] + [0, _i+1, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[6] + [0, _i+1, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[7] + [0, _i+1, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[1] - [0, _i+1, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[0] - [0, _i+1, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[5] - [0, _i+1, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[4] - [0, _i+1, 0]))
                    if (np.any(self.idx.cell[:, 2] + self.level[2] >= self.idx.grid.nk - 1) or
                            np.any(self.idx.cell[:, 2] - self.level[2] <= 0)):
                        pass
                    else:
                        for _i in range(self.level[2]):
                            _level_cell = np.vstack((_level_cell, self.idx.cell[4] + [0, 0, _i+1]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[5] + [0, 0, _i+1]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[6] + [0, 0, _i+1]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[7] + [0, 0, _i+1]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[0] - [0, 0, _i+1]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[1] - [0, 0, _i+1]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[2] - [0, 0, _i+1]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[3] - [0, 0, _i+1]))
                    # only unique rows to remove singularity issue -- This doesn't happen anymore
                    # Leave it here for now until more testing is done
                    # _level_cell = np.unique(_level_cell, axis=0)
                    # remove rows with negative values -- This doesn't happen anymore
                    # Leave it here for now until more testing is done
                    # _level_cell = _level_cell[~np.any(_level_cell < 0, axis=1)]
                    _cell_grd = self.idx.grid.grd[_level_cell[:, 0], _level_cell[:, 1], _level_cell[:, 2],
                                :, self.idx.block]
                    _cell_q = self.flow.q[_level_cell[:, 0], _level_cell[:, 1], _level_cell[:, 2],
                                :, self.idx.block]
                    # print('**SUCCESS Interpolating with {} levels'.format(self.level))
                # if -ve values; use the whole grid
                elif np.any(np.array(self.level) < 0):
                    _cell_grd = self.idx.grid.grd[..., self.idx.block].reshape(-1, 3)
                    _cell_q = self.flow.q[..., self.idx.block].reshape(-1, 5)
                _rbf = rbf(_cell_grd, _cell_q, kernel=self.rbf_kernel, epsilon=self.rbf_epsilon)
                self.q = _rbf(np.array(self.idx.ppoint).reshape(1, -1))
                self.q = self.q.reshape((1, 1, 1, -1, 1))

            case 'rbf-c-space':
                """
                Raidal basis function interpolation in c-space
                """
                # if the point is node return node data
                if self.idx.info == 'Given point is a node in the domain with a tol of 1e-12.\n' \
                                    'Interpolation will assign node properties for integration.\n' \
                                    'Index of the node will be returned by cell attribute\n':
                    self.q = _cell_q[0]
                    self.q = self.q.reshape((1, 1, 1, -1, 1))
                    return

                # Interpolate m1 and m2 for use in integration; m1 is used in ppath only!
                # m1 -- indicated as J (determinant of m1 is J)
                _cell_J = self.idx.grid.m1[self.idx.cell[:, 0], self.idx.cell[:, 1], self.idx.cell[:, 2], :, :,
                          self.idx.block]
                # m2 -- indicated as J_inv (determinant of m2 is J_inv)
                _cell_J_inv = self.idx.grid.m2[self.idx.cell[:, 0], self.idx.cell[:, 1], self.idx.cell[:, 2], :, :,
                              self.idx.block]
                # _alpha, _beta, _gamma and reshape for rbf intepolator to work
                _fractions = (self.idx.cpoint - self.idx.cell[0]).reshape(1, -1)

                # Start RBF interpolation
                from scipy.interpolate import RBFInterpolator as rbf
                # add adjacent cells to the cell list
                # TODO: debug for multiblock case -- currently defaults to single block
                _unit_cell = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                                       [0, 1, 0], [0, 0, 1], [1, 0, 1],
                                       [1, 1, 1], [0, 1, 1]])
                if np.any(np.array(self.level) > 0):
                    _level_cell = self.idx.cell
                    if (np.any(self.idx.cell[:, 0] + self.level[0] >= self.idx.grid.ni - 1) or
                            np.any(self.idx.cell[:, 0] - self.level[0] <= 0)):
                        pass
                    else:
                        for _i in range(self.level[0]):
                            _level_cell = np.vstack((_level_cell, self.idx.cell[1] + [_i+1, 0, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[2] + [_i+1, 0, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[5] + [_i+1, 0, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[6] + [_i+1, 0, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[0] - [_i+1, 0, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[3] - [_i+1, 0, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[4] - [_i+1, 0, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[7] - [_i+1, 0, 0]))
                            _unit_cell = np.vstack((_unit_cell, [_i + 2, 0, 0]))
                            _unit_cell = np.vstack((_unit_cell, [_i + 2, 1, 0]))
                            _unit_cell = np.vstack((_unit_cell, [_i + 2, 0, 1]))
                            _unit_cell = np.vstack((_unit_cell, [_i + 2, 1, 1]))
                            _unit_cell = np.vstack((_unit_cell, [-_i -1, 0, 0]))
                            _unit_cell = np.vstack((_unit_cell, [-_i -1, 1, 0]))
                            _unit_cell = np.vstack((_unit_cell, [-_i -1, 0, 1]))
                            _unit_cell = np.vstack((_unit_cell, [-_i -1, 1, 1]))

                    if (np.any(self.idx.cell[:, 1] + self.level[1] >= self.idx.grid.nj - 1) or
                            np.any(self.idx.cell[:, 1] - self.level[1] <= 0)):
                        pass
                    else:
                        for _i in range(self.level[1]):
                            _level_cell = np.vstack((_level_cell, self.idx.cell[2] + [0, _i+1, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[3] + [0, _i+1, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[6] + [0, _i+1, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[7] + [0, _i+1, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[0] - [0, _i+1, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[1] - [0, _i+1, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[4] - [0, _i+1, 0]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[5] - [0, _i+1, 0]))
                            _unit_cell = np.vstack((_unit_cell, [1, _i + 2, 0]))
                            _unit_cell = np.vstack((_unit_cell, [0, _i + 2, 0]))
                            _unit_cell = np.vstack((_unit_cell, [1, _i + 2, 1]))
                            _unit_cell = np.vstack((_unit_cell, [0, _i + 2, 1]))
                            _unit_cell = np.vstack((_unit_cell, [0, -_i - 1, 0]))
                            _unit_cell = np.vstack((_unit_cell, [1, -_i - 1, 0]))
                            _unit_cell = np.vstack((_unit_cell, [0, -_i - 1, 1]))
                            _unit_cell = np.vstack((_unit_cell, [1, -_i - 1, 1]))
                    if (np.any(self.idx.cell[:, 2] + self.level[2] >= self.idx.grid.nk - 1) or
                            np.any(self.idx.cell[:, 2] - self.level[2] <= 0)):
                        pass
                    else:
                        for _i in range(self.level[2]):
                            _level_cell = np.vstack((_level_cell, self.idx.cell[4] + [0, 0, _i+1]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[5] + [0, 0, _i+1]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[6] + [0, 0, _i+1]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[7] + [0, 0, _i+1]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[0] - [0, 0, _i+1]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[1] - [0, 0, _i+1]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[2] - [0, 0, _i+1]))
                            _level_cell = np.vstack((_level_cell, self.idx.cell[3] - [0, 0, _i+1]))
                            _unit_cell = np.vstack((_unit_cell, [0, 0, _i + 2]))
                            _unit_cell = np.vstack((_unit_cell, [1, 0, _i + 2]))
                            _unit_cell = np.vstack((_unit_cell, [1, 1, _i + 2]))
                            _unit_cell = np.vstack((_unit_cell, [0, 1, _i + 2]))
                            _unit_cell = np.vstack((_unit_cell, [0, 0, -_i - 1]))
                            _unit_cell = np.vstack((_unit_cell, [1, 0, -_i - 1]))
                            _unit_cell = np.vstack((_unit_cell, [1, 1, -_i - 1]))
                            _unit_cell = np.vstack((_unit_cell, [0, 1, -_i - 1]))
                    _cell_grd = self.idx.grid.grd[
                        _level_cell[:, 0], _level_cell[:, 1], _level_cell[:, 2], :, self.idx.block
                    ]
                    _cell_q = self.flow.q[
                        _level_cell[:, 0], _level_cell[:, 1], _level_cell[:, 2], :, self.idx.block
                    ]
                    _cell_J = self.idx.grid.m1[
                        _level_cell[:, 0], _level_cell[:, 1], _level_cell[:, 2], :, :, self.idx.block
                    ]
                    _cell_J_inv = self.idx.grid.m2[
                        _level_cell[:, 0], _level_cell[:, 1], _level_cell[:, 2], :, :, self.idx.block
                    ]
                    # print('**SUCCESS Interpolating with {} levels'.format(self.level))
                _rbf_q = rbf(_unit_cell, _cell_q, kernel=self.rbf_kernel, epsilon=self.rbf_epsilon)
                self.q = _rbf_q(_fractions)
                self.q = self.q.reshape((1, 1, 1, -1, 1))

                _rbf_J = rbf(_unit_cell, _cell_J, kernel=self.rbf_kernel, epsilon=self.rbf_epsilon)
                self.J = _rbf_J(_fractions).reshape(3, 3)
                _rbf_J_inv = rbf(_unit_cell, _cell_J_inv, kernel=self.rbf_kernel, epsilon=self.rbf_epsilon)
                self.J_inv = _rbf_J_inv(_fractions).reshape(3, 3)


            case 'rgi-p-space':
                # implement cublic spline interpolation in p-space
                # _cell_grd and _cell_q are known
                # get _level_grd and _level_q
                # TODO: debug for multiblock case -- currently defaults to single block; if not enough cells in
                #  the block, it defaults to the single cell case
                if np.any(np.array(self.level) > 0):
                    # add cells in i-direction
                    if np.any(self.idx.cell[:, 0] + self.level[0] >= self.idx.grid.ni - 1) or np.any(
                        self.idx.cell[:, 0] - self.level[0] <= 0
                    ):
                        _i_cell = np.unique(self.idx.cell[:, 0])
                        pass
                    else:
                        _i_cell = np.unique(self.idx.cell[:, 0])
                        for _i in range(self.level[0]):
                            _i_cell = np.hstack((_i_cell, self.idx.cell[1, 0] + _i + 1))
                            _i_cell = np.hstack((_i_cell, self.idx.cell[0, 0] - _i - 1))
                    # add nodes in j-direction
                    if np.any(self.idx.cell[:, 1] + self.level[1] >= self.idx.grid.nj - 1) or np.any(
                        self.idx.cell[:, 1] - self.level[1] <= 0
                    ):
                        _j_cell = np.unique(self.idx.cell[:, 1])
                        pass
                    else:
                        _j_cell = np.unique(self.idx.cell[:, 1])
                        for _i in range(self.level[1]):
                            _j_cell = np.hstack((_j_cell, self.idx.cell[2, 1] + _i + 1))
                            _j_cell = np.hstack((_j_cell, self.idx.cell[0, 1] - _i - 1))
                    # add nodes in k-direction
                    if np.any(self.idx.cell[:, 2] + self.level[2] >= self.idx.grid.nk - 1) or np.any(
                        self.idx.cell[:, 2] - self.level[2] <= 0
                    ):
                        _k_cell = np.unique(self.idx.cell[:, 2])
                        pass
                    else:
                        _k_cell = np.unique(self.idx.cell[:, 2])
                        for _i in range(self.level[2]):
                            _k_cell = np.hstack((_k_cell, self.idx.cell[4, 2] + _i + 1))
                            _k_cell = np.hstack((_k_cell, self.idx.cell[0, 2] - _i - 1))
                    # create a meshgrid of all possible combinations
                    _level_cell = np.meshgrid(np.sort(_i_cell), np.sort(_j_cell), np.sort(_k_cell))
                    # re-order back to an array with coordinates
                    _level_cell = np.moveaxis(np.array(_level_cell), 0, _level_cell[0].ndim).reshape(-1, len(_level_cell))
                    _level_cell = np.unique(_level_cell, axis=0)
                    # remove rows with negative values -- This doesn't happen anymore
                    # Leave it here for now until more testing is done
                    _level_cell = _level_cell[~np.any(_level_cell < 0, axis=1)]
                    _cell_grd = self.idx.grid.grd[
                        _level_cell[:, 0], _level_cell[:, 1], _level_cell[:, 2], :, self.idx.block
                    ]
                    _cell_q = self.flow.q[
                        _level_cell[:, 0], _level_cell[:, 1], _level_cell[:, 2], :, self.idx.block
                    ]
                # if -ve values; use the whole grid
                elif np.any(np.array(self.level) < 0):
                    _cell_grd = self.idx.grid.grd[..., self.idx.block].reshape(-1, 3)
                    _cell_q = self.flow.q[..., self.idx.block].reshape(-1, 5)
                # if all zeros; use the cell
                elif np.all(np.array(self.level) == 0):
                    _level_cell = np.unique(self.idx.cell, axis=0)
                    _cell_grd = self.idx.grid.grd[
                                _level_cell[:, 0], _level_cell[:, 1], _level_cell[:, 2], :, self.idx.block]
                    _cell_q = self.flow.q[
                                _level_cell[:, 0], _level_cell[:, 1], _level_cell[:, 2], :, self.idx.block]

                # Start the RGI
                from scipy.interpolate import RegularGridInterpolator
                # Find the unique values in each direction
                _x = np.unique(_cell_grd[:, 0])
                _y = np.unique(_cell_grd[:, 1])
                _z = np.unique(_cell_grd[:, 2])
                # Set the shape for reshaping q
                _shape = np.array([len(_x), len(_y), len(_z)])

                if self.adaptive =='shock':
                    _mach_n0, _mach_n1 = self._shock_cell_check(self)
                    # if shock is in the cell _mach_n0 > 1 > _mach_n1
                    if _mach_n0 > 1 > _mach_n1:
                        _method = 'nearest'
                    else:
                        if np.all(_shape >= 6):
                            _method = "quintic"
                        elif np.all(_shape >= 4):
                            _method = "cubic"
                        else:
                            _method = "linear"
                # Depending on the shape set the best possible interpolation method
                else:
                    if np.all(_shape >= 6):
                        _method = 'quintic'
                    elif np.all(_shape >= 4):
                        _method = 'cubic'
                    else:
                        _method = 'linear'

                # Create the RGI for each variable
                _rgi_rho = RegularGridInterpolator((_x, _y, _z), _cell_q[:, 0].reshape(_shape), method=_method)
                _rgi_rho_u = RegularGridInterpolator((_x, _y, _z), _cell_q[:, 1].reshape(_shape), method=_method)
                _rgi_rho_v = RegularGridInterpolator((_x, _y, _z), _cell_q[:, 2].reshape(_shape), method=_method)
                _rgi_rho_w = RegularGridInterpolator((_x, _y, _z), _cell_q[:, 3].reshape(_shape), method=_method)
                _rgi_e = RegularGridInterpolator((_x, _y, _z), _cell_q[:, 4].reshape(_shape), method=_method)

                # Find the values at the point and reshape
                self.q = np.array([_rgi_rho(self.idx.ppoint), _rgi_rho_u(self.idx.ppoint),
                                   _rgi_rho_v(self.idx.ppoint), _rgi_rho_w(self.idx.ppoint),
                                   _rgi_e(self.idx.ppoint)])
                self.q = self.q.reshape((1, 1, 1, -1, 1))

            case 'rgi-c-space':
                # implement rgi interpolation in c-space
                # if the point is node return node data
                if self.idx.info == 'Given point is a node in the domain with a tol of 1e-12.\n' \
                                    'Interpolation will assign node properties for integration.\n' \
                                    'Index of the node will be returned by cell attribute\n':
                    self.q = _cell_q[0]
                    self.q = self.q.reshape((1, 1, 1, -1, 1))
                    return

                # Interpolate m1 and m2 for use in integration; m1 is used in ppath only!
                # _alpha, _beta, _gamma and reshape for rbf intepolator to work
                _fractions = (self.idx.cpoint - self.idx.cell[0]).reshape(1, -1)

                # add adjacent cells to the cell list
                # TODO: debug for multiblock case -- currently defaults to single block
                _unit_cell = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                                       [0, 1, 0], [0, 0, 1], [1, 0, 1],
                                       [1, 1, 1], [0, 1, 1]])

                # Add cells in _i, _j, _k directions
                if np.any(np.array(self.level) > 0):
                    # add cells in i-direction
                    if np.any(self.idx.cell[:, 0] + self.level[0] >= self.idx.grid.ni - 1) or np.any(
                        self.idx.cell[:, 0] - self.level[0] <= 0
                    ):
                        _i_cell = np.unique(self.idx.cell[:, 0])
                        _i_unit_cell = _unit_cell[:, 0]
                        pass
                    else:
                        _i_cell = np.unique(self.idx.cell[:, 0])
                        _i_unit_cell = _unit_cell[:, 0]
                        for _i in range(self.level[0]):
                            _i_cell = np.hstack((_i_cell, self.idx.cell[1, 0] + _i + 1))
                            _i_cell = np.hstack((_i_cell, self.idx.cell[0, 0] - _i - 1))
                            _i_unit_cell = np.hstack((_i_unit_cell, _unit_cell[:, 0] + _i + 1))
                            _i_unit_cell = np.hstack((_i_unit_cell, _unit_cell[:, 0] - _i - 1))
                    # add nodes in j-direction
                    if np.any(self.idx.cell[:, 1] + self.level[1] >= self.idx.grid.nj - 1) or np.any(
                            self.idx.cell[:, 1] - self.level[1] <= 0
                    ):
                        _j_cell = np.unique(self.idx.cell[:, 1])
                        _j_unit_cell = _unit_cell[:, 1]
                        pass
                    else:
                        _j_cell = np.unique(self.idx.cell[:, 1])
                        _j_unit_cell = _unit_cell[:, 1]
                        for _i in range(self.level[1]):
                            _j_cell = np.hstack((_j_cell, self.idx.cell[2, 1] + _i + 1))
                            _j_cell = np.hstack((_j_cell, self.idx.cell[0, 1] - _i - 1))
                            _j_unit_cell = np.hstack((_j_unit_cell, _unit_cell[:, 1] + _i + 1))
                            _j_unit_cell = np.hstack((_j_unit_cell, _unit_cell[:, 1] - _i - 1))
                    # add nodes in k-direction
                    if np.any(self.idx.cell[:, 2] + self.level[2] >= self.idx.grid.nk - 1) or np.any(
                            self.idx.cell[:, 2] - self.level[2] <= 0
                    ):
                        _k_cell = np.unique(self.idx.cell[:, 2])
                        _k_unit_cell = _unit_cell[:, 2]
                        pass
                    else:
                        _k_cell = np.unique(self.idx.cell[:, 2])
                        _k_unit_cell = _unit_cell[:, 2]
                        for _i in range(self.level[2]):
                            _k_cell = np.hstack((_k_cell, self.idx.cell[4, 2] + _i + 1))
                            _k_cell = np.hstack((_k_cell, self.idx.cell[0, 2] - _i - 1))
                            _k_unit_cell = np.hstack((_k_unit_cell, _unit_cell[:, 2] + _i + 1))
                            _k_unit_cell = np.hstack((_k_unit_cell, _unit_cell[:, 2] - _i - 1))
                    # create a meshgrid of all possible combinations
                    _level_cell = np.meshgrid(np.sort(_i_cell), np.sort(_j_cell), np.sort(_k_cell))
                    _level_unit_cell = np.meshgrid(np.sort(_i_unit_cell), np.sort(_j_unit_cell), np.sort(_k_unit_cell))
                    # re-order back to an array with coordinates
                    _level_cell = np.moveaxis(np.array(_level_cell), 0, _level_cell[0].ndim).reshape(-1, len(_level_cell))
                    _level_cell = np.unique(_level_cell, axis=0)
                    _level_unit_cell = np.moveaxis(np.array(_level_unit_cell), 0, _level_unit_cell[0].ndim).reshape(-1, len(_level_unit_cell))
                    _level_unit_cell = np.unique(_level_unit_cell, axis=0)
                # if all zeros; use the cell
                elif np.all(np.array(self.level) == 0):
                    _level_cell = np.unique(self.idx.cell, axis=0)
                    _level_unit_cell = _unit_cell

                # assign cell values
                _cell_grd = self.idx.grid.grd[
                            _level_cell[:, 0], _level_cell[:, 1], _level_cell[:, 2], :, self.idx.block]
                _cell_q = self.flow.q[
                            _level_cell[:, 0], _level_cell[:, 1], _level_cell[:, 2], :, self.idx.block]
                _cell_J = self.idx.grid.m1[
                          _level_cell[:, 0], _level_cell[:, 1], _level_cell[:, 2], :, :, self.idx.block]
                _cell_J_inv = self.idx.grid.m2[
                            _level_cell[:, 0], _level_cell[:, 1], _level_cell[:, 2], :, :, self.idx.block]

                # Start the RGI
                from scipy.interpolate import RegularGridInterpolator
                # Find the unique values in each direction
                _x = np.unique(_level_unit_cell[:, 0])
                _y = np.unique(_level_unit_cell[:, 1])
                _z = np.unique(_level_unit_cell[:, 2])
                # Set the shape for reshaping q
                _shape = np.array([len(_x), len(_y), len(_z)])

                if self.adaptive =='shock':
                    _mach_n0, _mach_n1 = self._shock_cell_check(self)
                    # if shock is in the cell _mach_n0 > 1 > _mach_n1
                    if _mach_n0 > 1 > _mach_n1:
                        _method = 'nearest'
                    else:
                        if np.all(_shape >= 6):
                            _method = "quintic"
                        elif np.all(_shape >= 4):
                            _method = "cubic"
                        else:
                            _method = "linear"
                # Depending on the shape set the best possible interpolation method
                else:
                    if np.all(_shape >= 6):
                        _method = 'quintic'
                    elif np.all(_shape >= 4):
                        _method = 'cubic'
                    else:
                        _method = 'linear'

                # Create the RGI for each variable
                _rgi_rho = RegularGridInterpolator((_x, _y, _z), _cell_q[:, 0].reshape(_shape), method=_method)
                _rgi_rho_u = RegularGridInterpolator((_x, _y, _z), _cell_q[:, 1].reshape(_shape), method=_method)
                _rgi_rho_v = RegularGridInterpolator((_x, _y, _z), _cell_q[:, 2].reshape(_shape), method=_method)
                _rgi_rho_w = RegularGridInterpolator((_x, _y, _z), _cell_q[:, 3].reshape(_shape), method=_method)
                _rgi_e = RegularGridInterpolator((_x, _y, _z), _cell_q[:, 4].reshape(_shape), method=_method)

                # Find the values at the point and reshape
                self.q = np.array([_rgi_rho(_fractions), _rgi_rho_u(_fractions),
                                      _rgi_rho_v(_fractions), _rgi_rho_w(_fractions),
                                      _rgi_e(_fractions)])
                self.q = self.q.reshape((1, 1, 1, -1, 1))

                _rgi_J00 = RegularGridInterpolator((_x, _y, _z), _cell_J[:, 0, 0].reshape(_shape), method=_method)
                _rgi_J01 = RegularGridInterpolator((_x, _y, _z), _cell_J[:, 0, 1].reshape(_shape), method=_method)
                _rgi_J02 = RegularGridInterpolator((_x, _y, _z), _cell_J[:, 0, 2].reshape(_shape), method=_method)
                _rgi_J10 = RegularGridInterpolator((_x, _y, _z), _cell_J[:, 1, 0].reshape(_shape), method=_method)
                _rgi_J11 = RegularGridInterpolator((_x, _y, _z), _cell_J[:, 1, 1].reshape(_shape), method=_method)
                _rgi_J12 = RegularGridInterpolator((_x, _y, _z), _cell_J[:, 1, 2].reshape(_shape), method=_method)
                _rgi_J20 = RegularGridInterpolator((_x, _y, _z), _cell_J[:, 2, 0].reshape(_shape), method=_method)
                _rgi_J21 = RegularGridInterpolator((_x, _y, _z), _cell_J[:, 2, 1].reshape(_shape), method=_method)
                _rgi_J22 = RegularGridInterpolator((_x, _y, _z), _cell_J[:, 2, 2].reshape(_shape), method=_method)
                _J00, _J01, _J02 = _rgi_J00(_fractions), _rgi_J01(_fractions), _rgi_J02(_fractions)
                _J10, _J11, _J12 = _rgi_J10(_fractions), _rgi_J11(_fractions), _rgi_J12(_fractions)
                _J20, _J21, _J22 = _rgi_J20(_fractions), _rgi_J21(_fractions), _rgi_J22(_fractions)
                self.J = np.array([[_J00, _J01, _J02], [_J10, _J11, _J12], [_J20, _J21, _J22]]).reshape(3, 3)

                _rgi_J00_inv = RegularGridInterpolator((_x, _y, _z), _cell_J_inv[:, 0, 0].reshape(_shape), method=_method)
                _rgi_J01_inv = RegularGridInterpolator((_x, _y, _z), _cell_J_inv[:, 0, 1].reshape(_shape), method=_method)
                _rgi_J02_inv = RegularGridInterpolator((_x, _y, _z), _cell_J_inv[:, 0, 2].reshape(_shape), method=_method)
                _rgi_J10_inv = RegularGridInterpolator((_x, _y, _z), _cell_J_inv[:, 1, 0].reshape(_shape), method=_method)
                _rgi_J11_inv = RegularGridInterpolator((_x, _y, _z), _cell_J_inv[:, 1, 1].reshape(_shape), method=_method)
                _rgi_J12_inv = RegularGridInterpolator((_x, _y, _z), _cell_J_inv[:, 1, 2].reshape(_shape), method=_method)
                _rgi_J20_inv = RegularGridInterpolator((_x, _y, _z), _cell_J_inv[:, 2, 0].reshape(_shape), method=_method)
                _rgi_J21_inv = RegularGridInterpolator((_x, _y, _z), _cell_J_inv[:, 2, 1].reshape(_shape), method=_method)
                _rgi_J22_inv = RegularGridInterpolator((_x, _y, _z), _cell_J_inv[:, 2, 2].reshape(_shape), method=_method)
                _J00_inv, _J01_inv, _J02_inv = _rgi_J00_inv(_fractions), _rgi_J01_inv(_fractions), _rgi_J02_inv(_fractions)
                _J10_inv, _J11_inv, _J12_inv = _rgi_J10_inv(_fractions), _rgi_J11_inv(_fractions), _rgi_J12_inv(_fractions)
                _J20_inv, _J21_inv, _J22_inv = _rgi_J20_inv(_fractions), _rgi_J21_inv(_fractions), _rgi_J22_inv(_fractions)
                self.J_inv = np.array([[_J00_inv, _J01_inv, _J02_inv],
                                      [_J10_inv, _J11_inv, _J12_inv],
                                      [_J20_inv, _J21_inv, _J22_inv]]).reshape(3, 3)

            # TODO: Thorough testing needed
            case 'unsteady-rbf-p-space':
                """
                Raidal basis function interpolation in physical space for unsteady problems
                """
                # if the point is node return node data
                if self.idx.info == 'Given point is a node in the domain with a tol of 1e-12.\n' \
                                    'Interpolation will assign node properties for integration.\n' \
                                    'Index of the node will be returned by cell attribute\n':
                    self.q = _cell_q[0]
                    self.q = self.q.reshape((1, 1, 1, -1, 1))
                    return

                from scipy.interpolate import RBFInterpolator as rbf
                _rbf = rbf(_cell_grd, _cell_q)
                self.q = _rbf(np.array(self.idx.ppoint).reshape(1, -1))
                self.q = self.q.reshape((1, 1, 1, -1, 1))
                # Equation is linear interpolation between time steps in unsteady data
                # use of try-except is to avoid error in the first time step
                # instead of keeping track of it in every step
                try:
                    _tau = (np.sum(self.time) - self.flow_old.time) / (self.flow.time - self.flow_old.time)
                    _cell_q_old = self.flow_old.q[
                                  self.idx.cell[:, 0], self.idx.cell[:, 1], self.idx.cell[:, 2], :, self.idx.block
                                  ]
                    _rbf_old = rbf(_cell_grd, _cell_q_old)
                    _q_old = _rbf_old(np.array(self.idx.ppoint).reshape(1, -1))
                    _q_old = _q_old.reshape((1, 1, 1, -1, 1))
                    self.q = _tau * self.q + (1 - _tau) * _q_old
                except AttributeError:
                    return
