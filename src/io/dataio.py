# This file contains DataIO class to read and write particle data

import numpy as np
from src.streamlines.search import Search
from src.streamlines.interpolation import Interpolation
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp
from scipy.interpolate import griddata
import os


class DataIO:
    """
    Use to read/write particle data
    The process happens in four steps. Each step is described below
        1. Reads in the scattered particle data obtained from LPT code. Two ways are available:
            If a combined file is given, data is obtained from that --> faster
            Else, data is read from a folder of files where each file corresponds to a particle --> slow due to for loop
        2. Interpolates fluid data to these scattered locations. Two temp files are generated
            interpolated_q_data.npy --> has the flow data at the scattered locations
            new_p_data.npy --> has the particle data at scattered locations (WITH few outlier points removed)
        3. Lagrangian to Eulerian interpolation. Two temp files are generated
            flow_data.npy --> Fluid data interpolated to the grid
            particle_data.npy --> Particle data interpolated to the grid
        4. mgrid data to plot3d for visualization and syPIV usage. Output tunnel. Three files are generated
            mgrd_to_p3d.x --> Grid file
            mgrd_to_p3d_fluid.q --> fluid data
            mgrd_to_p3d_particle.q --> particle data in Eulerian interpretation

    Attributes
    ----------
    Input :
        grid : src.io.plot3dio.GridIO
            Grid object created from GridIO
        flow : src.io.plot3dio.FlowIO
            Flow object created from FlowIO
        read_file : str
            Combined file of all the particles generated from LPT code
        location : str
            Files location for individual particle data generated from the LPT code
        x_refinement: int
            Refinement of grid in horizontal direction
        y_refinement: int
            Refinement of grid in vertical direction
    Output:
        None

    Methods
    -------
        compute()
            saves the output files in the working directory

        Examples
            The test case is in test/test_dataio.py


    author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
    date: 12-28/2022
    """

    def __init__(self, grid, flow, read_file=None, location=None,
                 x_refinement: int = 50, y_refinement: int = 40):
        self.grid = grid
        self.flow = flow
        self.read_file = read_file
        self.location = location
        self.x_refinement = x_refinement
        self.y_refinement = y_refinement

    def compute(self):
        """
        This should interpolate the scattered particle data onto a 2D grid
        Return a file to be used for syPIV
        Returns:
        """
        # read particle files from a folder
        # read particle data
        try:
            print('Trying to read from a combined file...')
            _p_data = np.load(self.read_file)
            print('Read from the combined file!!')
        except:
            print('Reading from a group of files... This will take a while!')
            _files = os.listdir(self.location)
            _p_data = np.load(self.location + _files[0])
            for infile in _files[1:]:
                if infile.endswith(".npy"):
                    _temp = np.load(self.location + infile)
                    _p_data = np.vstack((_p_data, _temp))
            print('Done reading files into an array from a group of files!')

        _x_min, _x_max = _p_data[:, 0].min(), _p_data[:, 0].max()
        _y_min, _y_max = _p_data[:, 1].min(), _p_data[:, 1].max()

        # Get density and energy for plot3d file at locations
        _locations = _p_data[:, :3]

        def _flow_data(_point):
            """
            Internal function that interpolates data to scattered points
            Args:
                _point: One of the scattered points

            Returns:
                Interpolated data at each scattered point location given

            """
            _idx = Search(self.grid, _point)
            _idx.compute(method='p-space')
            _interp = Interpolation(self.flow, _idx)
            _interp.compute(method='c-space')

            try:
                print(f'Done with {_point}')
                return _interp.q.reshape(-1)
            except:
                print('Returned None. Exception occurred')
                return np.array([1], dtype=int)

        try:
            # Read back the saved file
            _q_list = np.load('interpolated_q_data.npy', allow_pickle=False)
            _p_data = np.load('new_p_data.npy', allow_pickle=False)
            print('Read the available interpolated data to continue with the griddata algorithm')
        except:
            print('Interpolated data file is unavailable. Continuing with interpolation to scattered data!\n'
                  'This is going to take sometime. Sit back and relax!\n'
                  'Your PC will take off because of multi-process. Let it breathe...\n')
            _pool = Pool(mp.cpu_count())
            _q_list = _pool.map(_flow_data, _locations)
            _pool.close()

            # Fluid data at scattered points/particle locations
            # Some searches return None. This helps remove those locations!
            _remove_index = [j for j in range(len(_q_list)) if np.all(_q_list[j] == 1)]
            _q_list = np.vstack(np.delete(_q_list, _remove_index, axis=0))
            _p_data = np.delete(_p_data, _remove_index, axis=0)
            # _q_list = np.array([i for i in _q_list if np.all(i != 1)])
            # Save both interpolated data and new particle data for easy future computations
            np.save('interpolated_q_data', _q_list)
            np.save('new_p_data', _p_data)
            print('Done with interpolating flow data to scattered points. ')

        # Particle data at the scattered points/particle locations
        # rho, x,y,z - momentum, energy per unit volume (q-file data)
        _q_p_list = np.hstack((_q_list[:, 0].reshape(-1, 1), _p_data[:, 3:6] * _q_list[:, 0].reshape(-1, 1),
                               _q_list[:, 4].reshape(-1, 1)))
        # Fluid data at the scattered points/particle locations
        _q_f_list = np.hstack((_q_list[:, 0].reshape(-1, 1), _p_data[:, 6:9] * _q_list[:, 0].reshape(-1, 1),
                               _q_list[:, 4].reshape(-1, 1)))

        # Create the grid to interpolate to
        _xi, _yi = np.linspace(_x_min, _x_max, self.x_refinement), np.linspace(_y_min, _y_max, self.y_refinement)
        _xi, _yi = np.meshgrid(_xi, _yi, indexing='ij')

        # Function to loop through the scattered data
        def _grid_interp(_data=None, _points=_p_data[:, :2], _x_grid=_xi, _y_grid=_yi, method='linear'):
            """
            Interpolate to grid data from scatter points
            Args:
                _points: scattered data - This is fixed, usually
                _x_grid: grid data - fixed usually
                _y_grid: grid data - fixed usually
                _data: data set to be interpolated
                method: set to linear but can be changed if needed

            Returns:
                Interpolated flow data

            """
            _data = _data.reshape(-1)
            # Transposing to keep consistency with default xy indexing of meshgrid
            _q = griddata(_points, _data, (_x_grid, _y_grid), method=method, fill_value=_data.max()*2)

            return _q


        try:
            # Read to see if data is available
            _qf = np.load('flow_data.npy')
            _qp = np.load('particle_data.npy')
            print('Loaded available flow/particle data from numpy residual files\n')
        except:
            print('Interpolating data to the grid provided...\n')
            # Interpolate scattered data onto the grid -- for flow
            _pool = Pool(mp.cpu_count())
            _qf = _pool.map(_grid_interp,
                            [_q_f_list[:, 0], _q_f_list[:, 1], _q_f_list[:, 2], _q_f_list[:, 3], _q_f_list[:, 4]])
            _pool.close()
            print(f'Done with flow data interpolation to grid.\n')

            # Interpolate scattered data onto the grid -- for particles
            _pool = Pool(mp.cpu_count())
            _qp_123 = _pool.map(_grid_interp, [_q_p_list[:, 1], _q_p_list[:, 2], _q_p_list[:, 3]])
            _pool.close()
            print(f'Done with particle data interpolation to grid.\n')

            # Create _qp array from known values
            _qp = np.dstack((_qf[0], _qp_123[0], _qp_123[1], _qp_123[2], _qf[-1])).transpose((2, 0, 1))

            # Save data to a temporary file
            _qf = np.array(_qf)
            _qp = np.array(_qp)
            np.save('flow_data', _qf)
            np.save('particle_data', _qp)

        # Write out to plot3d format for further processing
        self.grid.mgrd_to_p3d(_xi, _yi)
        self.flow.mgrd_to_p3d(_qf, mode='fluid')
        self.flow.mgrd_to_p3d(_qp, mode='particle')
        print('Files written to the working directory')

        return
