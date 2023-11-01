# This file contains DataIO class to read and write particle data

import numpy as np
from src.streamlines.search import Search
from src.streamlines.interpolation import Interpolation
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp
from scipy.interpolate import griddata
import os
import re
rng = np.random.default_rng()


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

    def __init__(self, grid, flow, percent_data=100, read_file=None, location='.',
                 x_refinement: int = 50, y_refinement: int = 40):
        self.grid = grid
        self.flow = flow
        self.percent_data = percent_data
        self.read_file = read_file
        self.location = location
        self.x_refinement = x_refinement
        self.y_refinement = y_refinement

    @staticmethod
    def _natural_sort(_l: list):
        """
        Sorts a list in natural order of things. Humanity is weird
        Args:
            _l: list of strings

        Returns:
            sorted list in natural order

        """
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(_l, key=alphanum_key)

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
            # TODO: Dask might save time here!
            print('Reading from a group of files... This will take a while!')
            # Sort in natural order to stack particles in order
            _files = np.array(self._natural_sort(os.listdir(self.location)))
            _bool = []
            for _i, _name in enumerate(_files):
                if not _name.endswith(".npy"):
                    _bool.append(_i)
            _files = np.delete(_files, _bool)

            _p_data = np.load(self.location + _files[0])
            for infile in _files[1:]:
                _temp = np.load(self.location + infile)
                _p_data = np.vstack((_p_data, _temp))
                print('Done with - ' + infile)
            print('Done reading files into an array from a group of files!')
            np.save(self.location + 'combined_file', _p_data)
            print('**SUCCESS** combined_file.npy saved to the given location.')

        # p-data has the following columns
        # x, y, z, vx, vy, vz, ux, uy, uz, time, integrated (ux, uy, uz), diameter, density
        _x_min, _x_max = self.grid.grd_min.reshape(-1)[0], self.grid.grd_max.reshape(-1)[0]
        _y_min, _y_max = self.grid.grd_min.reshape(-1)[1], self.grid.grd_max.reshape(-1)[1]

        # Get density and energy for plot3d file at locations
        if self.percent_data == 100:
            pass
        else:
            # Get a uniform distribution of the sample
            _p_data = rng.choice(_p_data, size=int(_p_data.shape[0] * self.percent_data / 100))
        _locations = _p_data[:, :3]

        def _flow_data(_point, _index, _size):
            """
            Internal function that interpolates data to scattered points
            Args:
                _point: One of the scattered points

            Returns:
                Interpolated data at each scattered point location given

            """
            try:
                _idx = Search(self.grid, _point)
                _idx.compute(method='distance')
                _interp = Interpolation(self.flow, _idx)
                _interp.compute(method='p-space')
                print(f'Done with flow data interpolation {_index}/{_size}')
                return _interp.q.reshape(-1)
            except:
                print(f'**Exception occurred with {_index}**')
                return np.array([1], dtype=int)

        try:
            # Read if saved files are available
            _q_list = np.load(self.location + 'dataio/interpolated_q_data.npy', allow_pickle=False)
            _p_data = np.load(self.location + 'dataio/new_p_data.npy', allow_pickle=False)
            print('Read the available interpolated data to continue with the griddata algorithm')
        except:
            # Run through the process of creating interpolation files
            try:
                # Read old interpolation files before removing outliers if available
                _q_list = np.load(self.location + 'dataio/_old_interpolated_q_data.npy', allow_pickle=True)
                _p_data = np.load(self.location + 'dataio/_old_p_data.npy', allow_pickle=True)
                print('Read the available old interpolated data to continue with the outliers algorithm')
            except:
                # Run the interpolation process on all the scattered points
                print('Interpolated data file is unavailable. Continuing with interpolation to scattered data!\n'
                      'This is going to take sometime. Sit back and relax!\n'
                      'Your PC will take off because of multi-process. Let it breathe...\n')
                _processors = mp.cpu_count()
                _pool = Pool(_processors)
                _loc_len = len(_locations)
                # Passing extra parameters to keep track of the progress. Chunk-size helps to keep it orderly
                _q_list = _pool.starmap(_flow_data, zip(_locations, np.arange(0, _loc_len), np.repeat(_loc_len, _loc_len)),
                                        chunksize=1)
                _pool.close()

                # Intermediate save of the data -- if the process is interrupted we can restart it from here
                try:
                    # Try creating the directory; if exists errors out and except
                    os.mkdir(self.location + 'dataio')
                    np.save(self.location + 'dataio/_old_interpolated_q_data', _q_list)
                    np.save(self.location + 'dataio/_old_p_data', _p_data)
                    print('Created dataio folder and saved old interpolated flow data to scattered points.\n')
                except:
                    np.save(self.location + 'dataio/_old_interpolated_q_data', _q_list)
                    np.save(self.location + 'dataio/_old_new_p_data', _p_data)
                print('Done with interpolating flow data to scattered points.\n'
                      'Removing outliers from the data...\n')

            # Fluid data at scattered points/particle locations
            # Some searches return None. This helps remove those locations!
            _remove_index = [j for j in range(len(_q_list)) if np.all(_q_list[j] == 1)]
            _q_list = np.vstack(np.delete(_q_list, _remove_index, axis=0))
            _p_data = np.delete(_p_data, _remove_index, axis=0)
            # Save both interpolated data and new particle data for easy future computations
            try:
                # Save interpolated data to files
                np.load(self.location + 'dataio/interpolated_q_data', allow_pickle=False)
                np.load(self.location + 'dataio/new_p_data', allow_pickle=False)
                print('Loaded particle and flow interpolated data from existing files.\n')
            except:
                np.save(self.location + 'dataio/interpolated_q_data', _q_list)
                np.save(self.location + 'dataio/new_p_data', _p_data)
            print('Done with interpolating flow data to scattered points.\n')

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
            # Where there's no data, fill with twice the max value
            _q = griddata(_points, _data, (_x_grid, _y_grid), method=method, fill_value=_data.max() * 2)

            return _q

        try:
            # Read to see if data is available
            _qf = np.load(self.location + 'dataio/flow_data.npy')
            _qp = np.load(self.location + 'dataio/particle_data.npy')
            print('Loaded available flow/particle data from numpy residual files\n')
        except:
            print('Interpolating data to the grid provided...\n')
            # Interpolate scattered data onto the grid -- for flow
            _pool = Pool(mp.cpu_count())
            _qf = _pool.map(_grid_interp,
                            [_q_f_list[:, 0], _q_f_list[:, 1], _q_f_list[:, 2], _q_f_list[:, 3], _q_f_list[:, 4]],
                            chunksize=1)
            _pool.close()
            print(f'Done with flow data interpolation to grid.\n')

            # Save the array to a file
            # This will only happen when there are files in dataio directory
            _qf = np.array(_qf)
            np.save(self.location + 'dataio/flow_data', _qf)

            # Interpolate scattered data onto the grid -- for particles
            _pool = Pool(mp.cpu_count())
            _qp_123 = _pool.map(_grid_interp, [_q_p_list[:, 1], _q_p_list[:, 2], _q_p_list[:, 3]], chunksize=1)
            _pool.close()
            print(f'Done with particle data interpolation to grid.\n')

            # Create _qp array from known values
            _qp = np.dstack((_qf[0], _qp_123[0], _qp_123[1], _qp_123[2], _qf[-1])).transpose((2, 0, 1))

            # Save data to a temporary file
            _qp = np.array(_qp)

            # This will only happen when there are files in dataio directory
            np.save(self.location + 'dataio/particle_data', _qp)

        # Write out to plot3d format for further processing
        self.grid.mgrd_to_p3d(_xi, _yi, out_file=self.location + 'dataio/mgrd_to_p3d.x')
        self.flow.mgrd_to_p3d(_qf, mode='fluid', out_file=self.location + 'dataio/mgrd_to_p3d')
        self.flow.mgrd_to_p3d(_qp, mode='particle', out_file=self.location + 'dataio/mgrd_to_p3d')
        print('Files written to the working directory')

        return
