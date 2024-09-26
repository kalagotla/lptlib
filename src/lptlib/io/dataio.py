# This file contains DataIO class to read and write particle data

import numpy as np
from ..streamlines.search import Search
from ..streamlines.interpolation import Interpolation
from scipy.interpolate import griddata, LinearNDInterpolator, RBFInterpolator
import os
import re
from tqdm import tqdm
from mpi4py import MPI
import psutil
import matplotlib.pyplot as plt
rng = np.random.default_rng(7)


class DataIO:
    """
    Use to read/write particle data
    The process happens in four steps. Each step is described below
        1. Reads in the scattered particle data obtained from LPT code.
            Data is read from a folder of files where each file corresponds to a particle; uses MPI
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
        self.location = location
        self.x_refinement = x_refinement
        self.y_refinement = y_refinement
        self.file_number_split = 10

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

        # # create a xy plot and save it
        # fig, ax = plt.subplots()
        # ax.scatter(_data[:, 0], _data[:, 1], s=1, label=f'Original data: {len(_data[:, 0])} points')
        # ax.scatter(_data[sampled_indices, 0], _data[sampled_indices, 1], s=1, color='red',
        #            label=f'Sampled data: {n_samples} points')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_xlim(_data[:, 0].min(), _data[:, 0].max())
        # ax.set_ylim(_data[:, 1].min(), _data[:, 1].max())
        # ax.legend(loc='upper right')
        # try:
        #     # Try creating the directory; if exists errors out and except
        #     os.mkdir(self.location + 'dataio')
        #     plt.savefig(self.location + 'dataio/sampled_data.png', dpi=300)
        # except FileExistsError:
        #     plt.savefig(self.location + 'dataio/sampled_data.png', dpi=300)

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
        for _file in tqdm(_files, desc=f'Reading files on Rank {rank}', position=1):
            if np.load(self.location + _file).shape[0] == 0:
                continue
            _data.append(np.load(self.location + _file))
        # gather -- fails if there are a lot of files
        _data = comm.gather(_data, root=0)
        if rank == 0:
            # remove empty arrays -- happens when there are more files than processes
            _data = [data for data in _data if len(data) > 0]
            # flatten the list
            _data = [data for sublist in _data for data in sublist]
            # stack the data
            _data = np.vstack(_data)
        else:
            _data = None
        # synchronize the processes
        comm.Barrier()
        return _data

    def compute(self):
        """
        This should interpolate the scattered particle data onto a 2D grid
        Return a file to be used for syPIV
        Returns:
        """
        # MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        # read particle files from a folder
        # Sort in natural order to stack particles in order and track progress
        _files = np.array(self._natural_sort(os.listdir(self.location)))
        _bool = []
        for _file in tqdm(_files, desc='Checking files'):
            _bool.append('npy' not in _file)
        _files = np.delete(_files, _bool)

        # Read and stack files using MPI
        # cut the files into smaller chunks to avoid memory issues
        n = self.file_number_split
        _files = np.array_split(_files, n)
        # remove empty arrays
        _files = [files for files in _files if len(files) > 0]
        _p_data = self._mpi_read(_files[0], comm)
        for _file in tqdm(_files[1:], desc='Reading files', position=0):
            _data = self._mpi_read(_file, comm)
            if comm.Get_rank() == 0:
                _p_data = np.vstack((_p_data, _data))
            else:
                _p_data = np.vstack((_p_data, _data))

        # pause until all processes are done
        comm.Barrier()
        print('Read from the group of files!!')

        # p-data has the following columns
        # x, y, z, vx, vy, vz, ux, uy, uz, time, integrated (ux, uy, uz), diameter, density
        # set the grid limits on all processes
        _x_min, _x_max = self.grid.grd_min.reshape(-1)[0], self.grid.grd_max.reshape(-1)[0]
        _y_min, _y_max = self.grid.grd_min.reshape(-1)[1], self.grid.grd_max.reshape(-1)[1]

        # Get density and energy for plot3d file at locations
        if self.percent_data == 100:
            pass
        else:
            # Get a uniform distribution of the sample using stratified sampling in x and y
            if rank == 0:
                _p_data = self._sample_data(self, _p_data, self.percent_data)
            else:
                _p_data = None
        # broadcast the data to all processes
        if rank == 0:
            _locations = _p_data[:, :3]
        else:
            _locations = None
        _locations = comm.bcast(_locations, root=0)
        comm.Barrier()

        # Load interpolated data after outlier removal if available
        try:
            # Read if saved files are available
            _q_list = np.load(self.location + 'dataio/interpolated_q_data.npy', allow_pickle=False)
            _p_data = np.load(self.location + 'dataio/new_p_data.npy', allow_pickle=False)
            print('Read the available interpolated data to continue with the griddata algorithm')
        except FileNotFoundError:
            # Run through the process of creating interpolation files
            # check first if the old interpolated data is available
            try:
                # Read old interpolation files before removing outliers if available
                _q_list = np.load(self.location + 'dataio/_old_interpolated_q_data.npy', allow_pickle=True)
                _p_data = np.load(self.location + 'dataio/_old_p_data.npy', allow_pickle=True)
                print('Read the available old interpolated data to continue with the outliers algorithm')
            except FileNotFoundError:
                # Run the interpolation process on all the scattered points
                if rank == 0:
                    print('Interpolated data file is unavailable. Continuing with interpolation to scattered data!\n'
                          'This is going to take sometime. Sit back and relax!\n'
                          'Your PC will take off because of multi-process. Let it breathe...\n')
                else:
                    pass

                # MPI
                _q_list = []

                # Prepare Scatterv
                if rank == 0:
                    # Split the locations array into subarrays along the first axis
                    split_sizes = np.array_split(_locations, size)

                    # Flatten the subarrays into a 1D array for Scatterv
                    _locations_flat = np.concatenate(split_sizes).flatten()

                    # Calculate the number of elements (not rows) for each process
                    split_sizes = [len(split) * 3 for split in split_sizes]  # Each row has 3 elements

                    # Calculate the displacements: the starting index of each process's subarray in the flattened array
                    split_displacements = [0] + np.cumsum(split_sizes[:-1]).tolist()
                else:
                    _locations_flat = None
                    split_sizes = None
                    split_displacements = None

                # Broadcast sizes and displacements to all ranks
                split_sizes = comm.bcast(split_sizes, root=0)
                split_displacements = comm.bcast(split_displacements, root=0)

                # Allocate space for the local subarray on each process
                local_size = split_sizes[rank]
                local_locations = np.empty(local_size, dtype=np.float64)

                # Scatter the flattened data
                comm.Scatterv([_locations_flat, split_sizes, split_displacements, MPI.DOUBLE], local_locations, root=0)

                # Reshape the local data back into 2D (each process gets a subarray of shape (m, 3))
                local_locations = local_locations.reshape(-1, 3)

                # Synchronize the processes
                comm.Barrier()

                # Run the interpolation process on each process
                _q_list = []
                for _point in tqdm(local_locations, desc=f'Data interpolation on Rank {rank}'):
                    _q_list.append(self._flow_data(_point))  # Assuming self._flow_data returns a (5,) array

                # Convert _q_list to a 1D numpy array for MPI communication (flatten the list of (5,) arrays)
                _q_list_flat = np.concatenate(_q_list).astype(np.float64)  # Shape will be (len(_q_list) * 5,)

                # Gather the sizes of the flattened _q_list from all processes
                local_q_list_size = len(_q_list_flat)
                all_q_list_sizes = comm.gather(local_q_list_size, root=0)

                # Prepare Gatherv variables for rank 0
                if rank == 0:
                    gatherv_displacements = [0] + np.cumsum(all_q_list_sizes[:-1]).tolist()
                    gathered_q_list = np.empty(sum(all_q_list_sizes), dtype=np.float64)  # Flat array to gather data
                else:
                    gathered_q_list = None
                    gatherv_displacements = None

                # Use Gatherv to gather the interpolated data
                comm.Gatherv(_q_list_flat, [gathered_q_list, all_q_list_sizes, gatherv_displacements, MPI.DOUBLE],
                             root=0)

                # Rank 0 should reshape the gathered data
                if rank == 0:
                    # Reshape gathered_q_list back into a 2D array where each row is of shape (5,)
                    _q_list = gathered_q_list.reshape(-1, 5)

                # Synchronize the processes
                comm.Barrier()

                # Intermediate save of the data -- if the process is interrupted we can restart it from here
                try:
                    # Try creating the directory; if exists errors out and except
                    os.mkdir(self.location + 'dataio')
                    np.save(self.location + 'dataio/_old_interpolated_q_data', _q_list)
                    np.save(self.location + 'dataio/_old_p_data', _p_data)
                    print('Created dataio folder and saved old interpolated flow data to scattered points.\n')
                except FileExistsError:
                    np.save(self.location + 'dataio/_old_interpolated_q_data', _q_list)
                    np.save(self.location + 'dataio/_old_p_data', _p_data)

            # Run the outlier removal process and save the data
            if rank == 0:
                print('Removing outliers from the data using one process...\n')
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
                _locations = _p_data[:, :3]
                # Save both interpolated data and new particle data for easy future computations
                np.save(self.location + 'dataio/interpolated_q_data', _q_list)
                np.save(self.location + 'dataio/new_p_data', _p_data)
            else:
                _q_list, _p_data, _locations = None, None, None
            # synchronize the processes
            comm.Barrier()

        # Create the grid to interpolate the Lagrangian data and save to plot3d format
        _xi, _yi = np.linspace(_x_min, _x_max, self.x_refinement), np.linspace(_y_min, _y_max,
                                                                               self.y_refinement)
        _xi, _yi = np.meshgrid(_xi, _yi, indexing='ij')
        if rank == 0:
            # save the grid to plot3d format
            self.grid.mgrd_to_p3d(_xi, _yi, out_file=self.location + 'dataio/mgrd_to_p3d.x')
        else:
            pass

        # Interpolate to grid
        try:
            # Read to see if data is available
            _qf = np.load(self.location + 'dataio/flow_data.npy')
            _qp = np.load(self.location + 'dataio/particle_data.npy')
            print('Loaded available flow/particle data from numpy residual files\n')
            # save to plot3d format
            self.flow.mgrd_to_p3d(_qf, mode='fluid', out_file=self.location + 'dataio/mgrd_to_p3d_fluid.q')
            self.flow.mgrd_to_p3d(_qp, mode='particle', out_file=self.location + 'dataio/mgrd_to_p3d_particle.q')
        except FileNotFoundError:
            def distribute_grid(grid, rank, size):
                # Determine the number of chunks in x and y directions
                num_x_chunks = int(np.sqrt(size))
                num_y_chunks = size // num_x_chunks

                # Ensure that the number of processes divides evenly
                assert num_x_chunks * num_y_chunks == size, "Size must be a perfect square or a product of two integers."

                # Get the shape of the grid
                grid_x = np.unique(grid[:, 0])
                grid_y = np.unique(grid[:, 1])
                x_chunk_size = len(grid_x) // num_x_chunks
                y_chunk_size = len(grid_y) // num_y_chunks

                # Determine the x and y index range for this rank
                rank_x = rank % num_x_chunks
                rank_y = rank // num_x_chunks

                # Get the start and end indices for x and y based on the rank
                x_start = rank_x * x_chunk_size
                x_end = (rank_x + 1) * x_chunk_size if rank_x != num_x_chunks - 1 else len(grid_x)

                y_start = rank_y * y_chunk_size
                y_end = (rank_y + 1) * y_chunk_size if rank_y != num_y_chunks - 1 else len(grid_y)

                # Extract the grid chunk corresponding to this rank
                x_chunk = grid_x[x_start:x_end]
                y_chunk = grid_y[y_start:y_end]

                # Create a grid chunk by combining the x and y meshgrid
                grid_chunk = np.array(np.meshgrid(x_chunk, y_chunk)).T.reshape(-1, 2)

                return grid_chunk, (x_start, x_end), (y_start, y_end)

            def distribute_points(grid_chunk, locations, data, data_particle):
                x_min, y_min = grid_chunk.min(axis=0)
                x_max, y_max = grid_chunk.max(axis=0)
                # Filter points that lie within the grid chunk
                indices = np.where((locations[:, 0] >= x_min) & (locations[:, 0] <= x_max) &
                                   (locations[:, 1] >= y_min) & (locations[:, 1] <= y_max))[0]
                locations = locations[indices]
                data = data[indices]
                data_particle = data_particle[indices]
                return locations, data, data_particle
            # Initialize MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            # create plot3d format lists
            if rank == 0:
                # Particle data at the scattered points/particle locations
                # rho, x,y,z - momentum, energy per unit volume (q-file data)
                _q_p_list = np.hstack((_q_list[:, 0].reshape(-1, 1), _p_data[:, 3:6] * _q_list[:, 0].reshape(-1, 1),
                                       _q_list[:, 4].reshape(-1, 1)))
                # Fluid data at the scattered points/particle locations
                _q_f_list = np.hstack((_q_list[:, 0].reshape(-1, 1), _p_data[:, 6:9] * _q_list[:, 0].reshape(-1, 1),
                                       _q_list[:, 4].reshape(-1, 1)))

                # Flatten grid for RBF interpolation
                grid = np.column_stack([_xi.ravel(), _yi.ravel()])
                # Split the scattered data points and grid across MPI processes on rank 0
                _grid_chunks = []
                _scatter_chunks = []
                _split_indices = []
                for i in range(size):
                    # Distribute the grid for each rank
                    grid_chunk, (x_start, x_end), (y_start, y_end) = distribute_grid(grid, i, size)
                    _xy_chunk, _qf_chunk, _qp_chunk = distribute_points(grid_chunk, _locations[:, :2],
                                                                        _q_f_list, _q_p_list)
                    _scatter_chunks.append((_xy_chunk, _qf_chunk, _qp_chunk))
                    _grid_chunks.append(grid_chunk)
                    _split_indices.append((x_start, x_end, y_start, y_end))
            else:
                _xi, _yi = None, None
                _q_f_list, _q_p_list = None, None
                _grid_chunks = []
                _scatter_chunks = []
                _scatter_particle_chunks = []
                _split_indices = []
            # synchronize the processes
            comm.Barrier()

            # Split the data across MPI processes
            grid_chunk = comm.scatter([chunk for chunk in _grid_chunks], root=0)
            # Lagrangian frame data
            _xy_chunk = comm.scatter([chunk[0] for chunk in _scatter_chunks], root=0)
            _qf_chunk = comm.scatter([chunk[1] for chunk in _scatter_chunks], root=0)
            _qp_chunk = comm.scatter([chunk[2] for chunk in _scatter_chunks], root=0)
            # debug statements
            # print(f'Rank {rank} has {_xy_chunk.shape} scattered points and {_qf_chunk.shape} values')
            # print(f'Rank {rank} has {grid_chunk.shape} grid points')

            # # Debugging plot
            # fig, ax = plt.subplots()
            # ax.scatter(_xy_chunk[:, 0], _xy_chunk[:, 1], s=1, label='Scattered points')
            # ax.scatter(grid_chunk[:, 0], grid_chunk[:, 1], s=1, label='Grid points')
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # ax.legend(loc='upper right')
            # plt.show()

            # Perform RBF interpolation on each process
            _fill_value = [2 * np.nanmax(self.flow.q[..., 0, :]), 0, 0, 0, 2 * np.nanmax(self.flow.q[..., -1, :])]
            if _qf_chunk.shape[0] <= 3:
                print(f'Found only {_qf_chunk.shape[0]} scattered points on Rank {rank}. Skipping RBF interpolation...')
                rho_result_chunk = np.full(grid_chunk.shape[0], _fill_value[0])
                ux_result_chunk = np.full(grid_chunk.shape[0], _fill_value[1])
                uy_result_chunk = np.full(grid_chunk.shape[0], _fill_value[2])
                uz_result_chunk = np.full(grid_chunk.shape[0], _fill_value[3])
                e_result_chunk = np.full(grid_chunk.shape[0], _fill_value[4])
                # print(f'shape of rho_result_chunk: {rho_result_chunk.shape} on Rank {rank}')
            else:
                rho_interpolator = RBFInterpolator(_xy_chunk, _qf_chunk[:, 0])
                ux_interpolator = RBFInterpolator(_xy_chunk, _qf_chunk[:, 1])
                uy_interpolator = RBFInterpolator(_xy_chunk, _qf_chunk[:, 2])
                uz_interpolator = RBFInterpolator(_xy_chunk, _qf_chunk[:, 3])
                e_interpolator = RBFInterpolator(_xy_chunk, _qf_chunk[:, 4])
                # Interpolated values on this chunk of the grid
                rho_result_chunk = rho_interpolator(grid_chunk)
                ux_result_chunk = ux_interpolator(grid_chunk)
                uy_result_chunk = uy_interpolator(grid_chunk)
                uz_result_chunk = uz_interpolator(grid_chunk)
                e_result_chunk = e_interpolator(grid_chunk)
                # print(f'shape of rho_result_chunk in RBF: {rho_result_chunk.shape} on Rank {rank}')

            # Gather the results from all processes
            # Gather chunk sizes
            chunk_size = np.array(rho_result_chunk.size)
            chunk_sizes = np.zeros(size, dtype=int)
            comm.Allgather(chunk_size, chunk_sizes)

            # Define displacements for gathering
            displacements = np.insert(np.cumsum(chunk_sizes[:-1]), 0, 0)

            # Gather all interpolated data
            if rank == 0:
                rho_result = np.empty(np.sum(chunk_sizes), dtype=rho_result_chunk.dtype)
                ux_result = np.empty(np.sum(chunk_sizes), dtype=ux_result_chunk.dtype)
                uy_result = np.empty(np.sum(chunk_sizes), dtype=uy_result_chunk.dtype)
                uz_result = np.empty(np.sum(chunk_sizes), dtype=uz_result_chunk.dtype)
                e_result = np.empty(np.sum(chunk_sizes), dtype=e_result_chunk.dtype)
            else:
                rho_result, ux_result, uy_result, uz_result, e_result = None, None, None, None, None

            comm.Gatherv(rho_result_chunk, [rho_result, chunk_sizes, displacements, MPI.DOUBLE], root=0)
            comm.Gatherv(ux_result_chunk, [ux_result, chunk_sizes, displacements, MPI.DOUBLE], root=0)
            comm.Gatherv(uy_result_chunk, [uy_result, chunk_sizes, displacements, MPI.DOUBLE], root=0)
            comm.Gatherv(uz_result_chunk, [uz_result, chunk_sizes, displacements, MPI.DOUBLE], root=0)
            comm.Gatherv(e_result_chunk, [e_result, chunk_sizes, displacements, MPI.DOUBLE], root=0)

            # Rank 0 processes final results
            if rank == 0:
                rho_result_save = np.zeros((self.x_refinement, self.y_refinement))
                ux_result_save = np.zeros((self.x_refinement, self.y_refinement))
                uy_result_save = np.zeros((self.x_refinement, self.y_refinement))
                uz_result_save = np.zeros((self.x_refinement, self.y_refinement))
                e_result_save = np.zeros((self.x_refinement, self.y_refinement))

                def reshape_to_save(variable=rho_result, variable_save=rho_result_save, _split_indices=_split_indices):
                    # Take in the interpolated grid variable and reshape it to save it
                    length_old = 0  # Initialize the length of the data
                    # print(f'{variable.shape} is the shape of the variable')
                    for indices in _split_indices:
                        # Assign split indices
                        x_start, x_end, y_start, y_end = indices
                        # Calculate the length of the data to be split
                        length = length_old + (x_end - x_start) * (y_end - y_start)
                        # print(f'length: {length}, length_old: {length_old}')
                        variable_save[x_start:x_end, y_start:y_end] = (
                            variable[length_old:length].reshape(x_end - x_start, y_end - y_start))
                        length_old = length
                    return variable_save

                rho_result_save = reshape_to_save(rho_result, rho_result_save)
                ux_result_save = reshape_to_save(ux_result, ux_result_save)
                uy_result_save = reshape_to_save(uy_result, uy_result_save)
                uz_result_save = reshape_to_save(uz_result, uz_result_save)
                e_result_save = reshape_to_save(e_result, e_result_save)

                _qf = np.stack([rho_result_save, ux_result_save, uy_result_save, uz_result_save, e_result_save])
                np.save(self.location + 'dataio/flow_data', _qf)
                self.flow.mgrd_to_p3d(_qf, mode='fluid', out_file=self.location + 'dataio/mgrd_to_p3d')
                print('Saved interpolated flow data to grid\n')

            comm.Barrier()

            # Particle data interpolation
            # debug statements
            # print(f'Rank {rank} has {_xy_chunk.shape} scattered points and {_qp_chunk.shape} values')
            # print(f'Rank {rank} has {grid_chunk.shape} grid points')

            # # Debugging plot
            # fig, ax = plt.subplots()
            # ax.scatter(_xy_chunk[:, 0], _xy_chunk[:, 1], s=1, label='Scattered points')
            # ax.scatter(grid_chunk[:, 0], grid_chunk[:, 1], s=1, label='Grid points')
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # ax.legend(loc='upper right')
            # plt.show()

            # Perform RBF interpolation on each process for particle data
            if _qp_chunk.shape[0] <= 3:
                print(f'Found only {_qp_chunk.shape[0]} scattered points on Rank {rank}. Skipping RBF interpolation...')
                ux_result_chunk = np.full(grid_chunk.shape[0], _fill_value[1])
                uy_result_chunk = np.full(grid_chunk.shape[0], _fill_value[2])
                uz_result_chunk = np.full(grid_chunk.shape[0], _fill_value[3])
            else:
                ux_interpolator = RBFInterpolator(_xy_chunk, _qp_chunk[:, 1])
                uy_interpolator = RBFInterpolator(_xy_chunk, _qp_chunk[:, 2])
                uz_interpolator = RBFInterpolator(_xy_chunk, _qp_chunk[:, 3])
                # Interpolated values on this chunk of the grid
                ux_result_chunk = ux_interpolator(grid_chunk)
                uy_result_chunk = uy_interpolator(grid_chunk)
                uz_result_chunk = uz_interpolator(grid_chunk)

            # Gather the results from all processes
            # Gather chunk sizes
            chunk_size = np.array(rho_result_chunk.size)
            chunk_sizes = np.zeros(size, dtype=int)
            comm.Allgather(chunk_size, chunk_sizes)

            # Define displacements for gathering
            displacements = np.insert(np.cumsum(chunk_sizes[:-1]), 0, 0)

            # Gather all interpolated data
            if rank == 0:
                ux_result = np.empty(np.sum(chunk_sizes), dtype=ux_result_chunk.dtype)
                uy_result = np.empty(np.sum(chunk_sizes), dtype=uy_result_chunk.dtype)
                uz_result = np.empty(np.sum(chunk_sizes), dtype=uz_result_chunk.dtype)
            else:
                ux_result, uy_result, uz_result = None, None, None

            comm.Gatherv(ux_result_chunk, [ux_result, chunk_sizes, displacements, MPI.DOUBLE], root=0)
            comm.Gatherv(uy_result_chunk, [uy_result, chunk_sizes, displacements, MPI.DOUBLE], root=0)
            comm.Gatherv(uz_result_chunk, [uz_result, chunk_sizes, displacements, MPI.DOUBLE], root=0)

            # Rank 0 processes final results
            if rank == 0:
                ux_result_save = np.zeros((self.x_refinement, self.y_refinement))
                uy_result_save = np.zeros((self.x_refinement, self.y_refinement))
                uz_result_save = np.zeros((self.x_refinement, self.y_refinement))

                ux_result_save = reshape_to_save(variable=ux_result, variable_save=ux_result_save)
                uy_result_save = reshape_to_save(variable=uy_result, variable_save=uy_result_save)
                uz_result_save = reshape_to_save(variable=uz_result, variable_save=uz_result_save)

                _qp = np.stack([rho_result_save, ux_result_save, uy_result_save, uz_result_save, e_result_save])
                np.save(self.location + 'dataio/particle_data', _qp)
                self.flow.mgrd_to_p3d(_qp, mode='particle', out_file=self.location + 'dataio/mgrd_to_p3d')
                print('Saved interpolated particle data to grid\n')
            else:
                pass

            comm.Barrier()

        return
