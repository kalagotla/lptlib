# This file contains DataIO class to read and write particle data

import numpy as np
from src.streamlines.search import Search
from src.streamlines.interpolation import Interpolation
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp
from scipy.interpolate import griddata


class DataIO:
    """
    Use to read/write particle data
    """

    def __init__(self, grid, flow, read_file=None, write_file=None, refinement: int = 50):
        self.grid = grid
        self.flow = flow
        self.read_file = read_file
        self.write_file = write_file
        self.refinement = refinement

    def compute(self):
        """
        This should interpolate the scattered particle data onto a 2D grid
        Return a file to be used for syPIV
        Returns:
        """
        # read particle data
        _p_data = np.load(self.read_file)
        _x_min, _x_max = _p_data[:, 0].min(), _p_data[:, 0].max()
        _y_min, _y_max = _p_data[:, 1].min(), _p_data[:, 1].max()

        # Get density and energy for plot3d file at locations
        _locations = _p_data[:, :3]

        def _flow_data(_point):
            _idx = Search(self.grid, _point)
            _idx.compute(method='c-space')
            _interp = Interpolation(self.flow, _idx)
            _interp.compute(method='c-space')

            return _interp.q.reshape(-1)

        _pool = Pool(mp.cpu_count() - 2)
        _q_list = _pool.map(_flow_data, _locations)
        _pool.close()

        # Fluid data at scattered points/particle locations
        _q_list = np.array(_q_list)

        # Particle data at scattered points/particle locations
        _q_p_list = np.hstack((_q_list[:, 0].reshape(-1, 1), _p_data[:, 3:6], _q_list[:, 4].reshape(-1, 1)))

        # Create the grid to interpolate to
        _xi, _yi = np.linspace(_x_min, _x_max, self.refinement), np.linspace(_y_min, _y_max, self.refinement)
        _xi, _yi = np.meshgrid(_xi, _yi, indexing='ij')

        # Function to loop through the scattered data
        def _grid_interp(_points=_p_data[:, :2], _x_grid=_xi, _y_grid=_yi, _data=None, method='linear'):
            _data = _data.reshape(-1)
            _q = griddata(_points, _data, (_x_grid, _y_grid),
                          method=method, fill_value=_data.max()*1.01)

            return _q

        # Interpolate scattered data onto the grid
        _qf, _qp = [], []
        for _i in range(5):
            _qf.append(_grid_interp(_data=_q_list[:, _i]))
            _qp.append(_grid_interp(_data=_q_p_list[:, _i]))

        _qf = np.array(_qf)
        _qp = np.array(_qp)

        # Write out to plot3d format for further processing
        self.grid.mgrd_to_p3d(_xi, _yi)
        self.flow.mgrd_to_p3d(_qf, mode='fluid')

        return
