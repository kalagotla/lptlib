# Use tri-linear interpolation to get data at the given point

import numpy as np


class Interpolation:
    """
    Module to do the cell data interpolation to the given point

    ...

    Attributes
    ----------
    Input:
        flow : src.io.plot3dio.FlowIO
            Flow object created from FlowIO
        idx: src.streamlines.Search
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

    def __str__(self):
        doc = "This instance uses " + self.flow.filename + " as the flow file " \
                                                           "to compute properties at " + self.idx.ppoint + "\n"
        return doc

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

                    # Doing data interpolation to the given point based on the face points
                    self.q = _f(self, 2, 2, _cell_grd_0123, _cell_grd_4567, None, 2,
                                [None, None, _cell_q_0123], [None, None, _cell_q_4567])
                    self.q = self.q.reshape((1, 1, 1, -1, 1))

                # if the point is node return node data
                if self.idx.info == 'Given point is a node in the domain with a tol of 1e-6.\n' \
                                    'Interpolation will assign node properties for integration.\n'\
                                    'Index of the node will be returned by cell attribute\n':
                    self.q = _cell_q
                    self.q = self.q.reshape((1, 1, 1, -1, 1))

            case 'c-space':
                """
                Point from idx is in c-space. Tri-linear interpolation is performed.
                Equation can be found in Sadarjoen et al.
                """

                # Interpolate m1 and m2 for use in integration; m1 is used in ppath only!
                # m1 -- indicated as J (determinant of m1 is J)
                _cell_J = self.idx.grid.m1[self.idx.cell[:, 0], self.idx.cell[:, 1], self.idx.cell[:, 2], :, :, self.idx.block]
                # m2 -- indicated as J_inv (determinant of m2 is J_inv)
                _cell_J_inv = self.idx.grid.m2[self.idx.cell[:, 0], self.idx.cell[:, 1], self.idx.cell[:, 2], :, :, self.idx.block]
                _alpha, _beta, _gamma = self.idx.cpoint - self.idx.cell[0]

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

                # if the point is node return node data
                if self.idx.info == 'Given point is a node in the domain with a tol of 1e-6.\n' \
                                    'Interpolation will assign node properties for integration.\n' \
                                    'Index of the node will be returned by cell attribute\n':
                    self.q = _cell_q
                    self.q = self.q.reshape((1, 1, 1, -1, 1))
