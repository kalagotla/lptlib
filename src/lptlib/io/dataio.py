# This file contains DataIO class to read and write particle data

import numpy as np
from ..streamlines.search import Search
from ..streamlines.interpolation import Interpolation
from scipy.interpolate import griddata, LinearNDInterpolator, RBFInterpolator
import os
import re
from tqdm import tqdm
from mpi4py import MPI
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
rng = np.random.default_rng(7)


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

    def _flow_data(self, _point):
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
            return _interp.q.reshape(-1)
        except:
            return np.array([1], dtype=int)

    # Function to loop through the scattered data
    @staticmethod
    def _grid_interp(_points, _data, _x_grid, _y_grid, fill_value, method='linear'):
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
        _q = griddata(_points, _data, (_x_grid, _y_grid), method=method, fill_value=fill_value)

        return _q

    @staticmethod
    def _sample_data(self, _data, _percent):
        """
            Uniformly samples the given percentage of data spread on an XY plane.

            Parameters:
            data (numpy.ndarray): An array of shape (n, 15) where the first two columns are x and y coordinates.
            percent (float): The percentage of data to sample (0 < percent <= 100).

            Returns:
            numpy.ndarray: The uniformly sampled subset of the data.
            """
        # Ensure percent is between 0 and 100
        if _percent <= 0 or _percent > 100:
            raise ValueError("Percent must be between 0 and 100")

        # Calculate the number of samples to take
        n_samples = int(len(_data[:, 0]) * _percent / 100)

        # Uniformly sample indices
        sampled_indices = np.random.choice(len(_data), n_samples, replace=False)

        # create a xy plot and save it
        fig, ax = plt.subplots()
        ax.scatter(_data[:, 0], _data[:, 1], s=1, label=f'Original data: {len(_data[:, 0])} points')
        ax.scatter(_data[sampled_indices, 0], _data[sampled_indices, 1], s=1, color='red',
                   label=f'Sampled data: {n_samples} points')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(_data[:, 0].min(), _data[:, 0].max())
        ax.set_ylim(_data[:, 1].min(), _data[:, 1].max())
        ax.legend(loc='upper right')
        try:
            # Try creating the directory; if exists errors out and except
            os.mkdir(self.location + 'dataio')
            plt.savefig(self.location + 'dataio/sampled_data.png', dpi=300)
        except FileExistsError:
            plt.savefig(self.location + 'dataio/sampled_data.png', dpi=300)

        # Return the sampled data
        return _data[sampled_indices]

    def _mpi_read(self, _files, comm):
        """
        Read files using MPI
        Args:
            _files: list of files to be read
            comm: communicator object from MPI

        Returns:

        """
        rank = comm.Get_rank()
        size = comm.Get_size()
        _data = []
        # scatter
        _files = np.array_split(_files, size)[rank]
        for _file in tqdm(_files, desc=f'Reading files on Rank {rank}'):
            if np.load(self.location + _file).shape[0] == 0:
                continue
            _data.append(np.load(self.location + _file))
        _data = np.vstack(_data)
        # gather -- fails if there are a lot of files
        _data = comm.gather(_data, root=0)
        if rank == 0:
            _data = np.vstack(_data)
        else:
            _data = None
        return _data

    def compute(self):
        """
        This should interpolate the scattered particle data onto a 2D grid
        Return a file to be used for syPIV
        Returns:
        """
        # MPI
        comm = MPI.COMM_WORLD
        # read particle files from a folder
        # read particle data
        try:
            print('Trying to read from a combined file...')
            _p_data = np.load(self.read_file)
            print('Read from the combined file!!')
        except FileNotFoundError:
            # Sort in natural order to stack particles in order and track progress
            _files = np.array(self._natural_sort(os.listdir(self.location)))
            _bool = []
            for _file in tqdm(_files, desc='Checking files'):
                _bool.append('npy' not in _file)
            _files = np.delete(_files, _bool)

            # Read and stack files using MPI
            # cut the files into smaller chunks to avoid memory issues
            n = len(_files) // 1500 + 1  # ~2 sets for 3000-4000 files as tested
            _files = np.array_split(_files, n)
            _p_data = self._mpi_read(_files[0], comm)
            for _file in _files[1:]:
                _data = self._mpi_read(_file, comm)
                if comm.Get_rank() == 0:
                    _p_data = np.vstack((_p_data, _data))
                else:
                    _p_data = np.vstack((_p_data, _data))
            _p_data = comm.bcast(_p_data, root=0)

            # pause until all processes are done
            comm.Barrier()
            print('Read from the group of files!!')

            # Save the combined file for future use
            try:
                if comm.Get_rank() == 0:
                    np.save(self.location + 'combined_file', _p_data)
                    print('Saved the combined file for future use!\n')
            except FileNotFoundError:
                print('Could not save the combined file for future use!\n')

        # p-data has the following columns
        # x, y, z, vx, vy, vz, ux, uy, uz, time, integrated (ux, uy, uz), diameter, density
        _x_min, _x_max = self.grid.grd_min.reshape(-1)[0], self.grid.grd_max.reshape(-1)[0]
        _y_min, _y_max = self.grid.grd_min.reshape(-1)[1], self.grid.grd_max.reshape(-1)[1]

        # Get density and energy for plot3d file at locations
        if self.percent_data == 100:
            pass
        else:
            # Get a uniform distribution of the sample using stratified sampling in x and y
            if comm.Get_rank() == 0:
                _p_data = self._sample_data(self, _p_data, self.percent_data)
            else:
                _p_data = None
        _p_data = comm.bcast(_p_data, root=0)
        _locations = _p_data[:, :3]
        comm.Barrier()

        try:
            # Read if saved files are available
            _q_list = np.load(self.location + 'dataio/interpolated_q_data.npy', allow_pickle=False)
            _p_data = np.load(self.location + 'dataio/new_p_data.npy', allow_pickle=False)
            print('Read the available interpolated data to continue with the griddata algorithm')
        except FileNotFoundError:
            # Run through the process of creating interpolation files
            try:
                # Read old interpolation files before removing outliers if available
                _q_list = np.load(self.location + 'dataio/_old_interpolated_q_data.npy', allow_pickle=True)
                _p_data = np.load(self.location + 'dataio/_old_p_data.npy', allow_pickle=True)
                print('Read the available old interpolated data to continue with the outliers algorithm')
            except FileNotFoundError:
                # Run the interpolation process on all the scattered points
                print('Interpolated data file is unavailable. Continuing with interpolation to scattered data!\n'
                      'This is going to take sometime. Sit back and relax!\n'
                      'Your PC will take off because of multi-process. Let it breathe...\n')

                # MPI
                _q_list = []
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                size = comm.Get_size()
                _locations = np.array_split(_locations, size)[rank]
                for _point in tqdm(_locations, desc=f'Data interpolation on Rank {rank}'):
                    _q_list.append(self._flow_data(_point))
                _q_list = comm.gather(_q_list, root=0)
                if rank == 0:
                    _q_list = np.vstack(_q_list)
                else:
                    _q_list = None
                _q_list = comm.bcast(_q_list, root=0)
                # synchronize the processes
                comm.Barrier()

                # Intermediate save of the data -- if the process is interrupted we can restart it from here
                try:
                    # Try creating the directory; if exists errors out and except
                    os.mkdir(self.location + 'dataio')
                    np.save(self.location + 'dataio/_old_interpolated_q_data', _q_list)
                    np.save(self.location + 'dataio/_old_p_data', _p_data)
                    print('Created dataio folder and saved old interpolated flow data to scattered points.\n')
                except:
                    np.save(self.location + 'dataio/_old_interpolated_q_data', _q_list)
                    np.save(self.location + 'dataio/_old_p_data', _p_data)
                print('Removing outliers from the data...\n')

            # Fluid data at scattered points/particle locations
            # Some searches return None. This helps remove those locations!
            _remove_index = [j for j in range(len(_q_list)) if np.all(_q_list[j] == 1)]
            _q_list = np.vstack(np.delete(_q_list, _remove_index, axis=0))
            _p_data = np.delete(_p_data, _remove_index, axis=0)
            # Remove outliers due to bad interpolation -- density cannot go beyond the flow limits
            _remove_index = np.where(_q_list[:, 0] < self.flow.q[..., 0, :].min())
            _q_list = np.vstack(np.delete(_q_list, _remove_index, axis=0))
            _p_data = np.delete(_p_data, _remove_index, axis=0)
            _remove_index = np.where(_q_list[:, 0] > self.flow.q[..., 0, :].max())
            _q_list = np.vstack(np.delete(_q_list, _remove_index, axis=0))
            _p_data = np.delete(_p_data, _remove_index, axis=0)
            # Save both interpolated data and new particle data for easy future computations
            try:
                # Save interpolated data to files
                _q_list = np.load(self.location + 'dataio/interpolated_q_data', allow_pickle=False)
                _p_data = np.load(self.location + 'dataio/new_p_data', allow_pickle=False)
                print('Loaded particle and flow interpolated data from existing files.\n')
            except:
                np.save(self.location + 'dataio/interpolated_q_data', _q_list)
                np.save(self.location + 'dataio/new_p_data', _p_data)

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

        try:
            # Read to see if data is available
            _qf = np.load(self.location + 'dataio/flow_data.npy')
            _qp = np.load(self.location + 'dataio/particle_data.npy')
            print('Loaded available flow/particle data from numpy residual files\n')
        except:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            # set fill value to twice the max value for scalars and 0 for vectors
            _fill_value = [2 * np.nanmax(self.flow.q[..., 0, :]), 0, 0, 0, 2 * np.nanmax(self.flow.q[..., -1, :])]
            if rank < 5:
                # Interpolate scattered data onto the grid -- for flow using MPI
                _qf = []
                # run each variable of the flow data separately on each process
                _q_f_list = np.array_split(_q_f_list, 5, axis=1)[rank]
                for _q_f in tqdm(_q_f_list.T, desc=f'Flow data interpolation on Rank {rank}'):
                    _qf.append(self._grid_interp(_p_data[:, :2], _q_f, _xi, _yi, _fill_value[rank]))
            else:
                # For ranks >= 5, participate in gather with a dummy value to ensure no deadlock
                _qf = [np.empty(_xi.shape)]  # Ensure the dummy value is consistent with the expected data structure
            # Ensure all processes reach this point before proceeding
            comm.Barrier()
            _qf = comm.gather(_qf, root=0)
            if rank == 0:
                # stack _qf from first 5 ranks
                _qf = [data[0] for i, data in enumerate(_qf) if i < 5]  # list of arrays
                print(f'Flow data shape list: {len(_qf)} and {_qf[0].shape}')
                # Save the array to a file
                # This will only happen when there are files in dataio directory
                _qf = np.stack(_qf)  # shape (5, _xi.shape[0], _xi.shape[1])
                # fill the missing values with the fill value
                np.save(self.location + 'dataio/flow_data', _qf)
                print(f'Flow data shape stack: {_qf.shape}')
            else:
                _qf = None
            _qf = comm.bcast(_qf, root=0)
            # synchronize the processes
            comm.Barrier()

            # Particle data interpolation
            if rank < 3:
                # Interpolate scattered data onto the grid -- for particles using MPI
                _qp_123 = []
                _q_p_list = np.array_split(_q_p_list[:, 1:4], 3, axis=1)[rank]
                for _q_p in tqdm(_q_p_list.T, desc=f'Particle data interpolation on Rank {rank}'):
                    # 0 fill value because we are interpolating vectors
                    _qp_123.append(self._grid_interp(_p_data[:, :2], _q_p, _xi, _yi, 0))
            else:
                # For ranks >= 3, participate in gather with a dummy value to ensure no deadlock
                _qp_123 = [np.empty(_xi.shape)]  # Ensure the dummy value is consistent with the expected data structure
            # Ensure all processes reach this point before proceeding
            comm.Barrier()
            _qp_123 = comm.gather(_qp_123, root=0)
            if rank == 0:
                # stack _qf from first 5 ranks
                _qp_123 = [data[0] for i, data in enumerate(_qp_123) if i < 3]  # list of lists
                # Create _qp array from known values
                _qp = np.stack((_qf[0], _qp_123[0], _qp_123[1], _qp_123[2], _qf[-1]))
                # Save data to a temporary file
                _qp = np.array(_qp)
                # This will only happen when there are files in dataio directory
                np.save(self.location + 'dataio/particle_data', _qp)
            else:
                _qp_123 = None
            _qp_123 = comm.bcast(_qp_123, root=0)
            # synchronize the processes
            comm.Barrier()

            if rank == 0:
                # Write out to plot3d format for further processing
                self.grid.mgrd_to_p3d(_xi, _yi, out_file=self.location + 'dataio/mgrd_to_p3d.x')
                self.flow.mgrd_to_p3d(_qf, mode='fluid', out_file=self.location + 'dataio/mgrd_to_p3d')
                self.flow.mgrd_to_p3d(_qp, mode='particle', out_file=self.location + 'dataio/mgrd_to_p3d')

        return
