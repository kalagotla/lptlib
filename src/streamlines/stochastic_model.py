# Class to run streamlines script in parallel
# Stochastic model for tracers is implemented

import numpy as np
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp
from src.streamlines.streamlines import Streamlines

rng = np.random.default_rng(7)


class StochasticModel(Streamlines):
    """Module to spawn and run LPT on given tracers parallely

    ...

    Attributes
    ----------

    """

    def __init__(self, particles, spawn_locations, method='adaptive-p-space',
                 grid=None, flow=None, point=None,
                 search='p-space', interpolation='p-space', integration='pRK4',
                 diameter=1e-7, density=1000,
                 time_step=1e-3, max_time_step=1, drag_model='henderson', adaptivity=0.001,
                 magnitude_adaptivity=0.001,
                 filepath: str = None
                 ):
        super().__init__(point=point,
                         search=search, interpolation=interpolation, integration=integration,
                         diameter=diameter, density=density,
                         time_step=time_step, max_time_step=max_time_step, drag_model=drag_model)
        self.particles = particles
        self.spawn_locations = spawn_locations
        # Read-in grid and flow files
        self.grid = grid
        self.flow = flow
        self.method = method
        self.filepath = filepath
        self.adaptivity = adaptivity
        self.magnitude_adaptivity = magnitude_adaptivity

    def setup(self, spawn_location, particle_dia, task):
        """
        Sets up the function to be run in parallel
        Args:
            self:
            spawn_location:
            particle_dia:
            task: same as particle.n_concentration, used to track progress of computation

        Returns:

        """
        # TODO: Have to use inheritance properties. Currently, just calling in another object
        print(f'Execution started for particle number - {task}')
        sl = Streamlines(None, None, point=spawn_location, diameter=particle_dia, time_step=self.time_step,
                         task=task)
        sl.density = self.particles.density
        sl.drag_model = self.drag_model
        sl.max_time_step = self.max_time_step
        sl.filepath = self.filepath
        sl.search = self.search
        sl.interpolation = self.interpolation
        sl.integration = self.integration
        sl.adaptivity = self.adaptivity
        sl.magnitude_adaptivity = self.magnitude_adaptivity
        sl.compute(method=self.method, grid=self.grid, flow=self.flow)

        return sl

    def multi_process(self):
        """
        To parallelize using multiprocessing approach; the setup function
        Returns:

        """
        with mp.Pool(mp.cpu_count() - 1) as pool:
            lpt_data = pool.starmap(self.setup, zip(self.spawn_locations.locations, self.particles.particle_field,
                                                    np.arange(self.particles.n_concentration)), chunksize=1)

        return lpt_data

    def multi_thread(self):
        """
        To parallelize using multithreading approach; the setup function
        Returns:

        """
        with Pool(mp.cpu_count() - 1) as pool:
            lpt_data = pool.starmap(self.setup, zip(self.spawn_locations.locations, self.particles.particle_field,
                                                    np.arange(self.particles.n_concentration)), chunksize=1)

        return lpt_data

    def serial(self):
        """
        To run setup in serial
        Returns:

        """
        # Run setup function in serial
        lpt_data = []
        for i in range(self.particles.n_concentration):
            lpt_data.append(self.setup(self.spawn_locations.locations[i], self.particles.particle_field[i], i))

        return lpt_data


class Particle:
    """
    Class holds details for particles used in a PIV experiment
    ---
    User has to provide all the information to generate size distribution
    """

    def __init__(self):
        self.distribution = "gaussian"
        self.min_dia = None
        self.max_dia = None
        self.mean_dia = None
        self.std_dia = None
        self.density = None
        self.n_concentration = None
        self.particle_field = None

    def compute_distribution(self):
        """
        Run this method to return a distribution of particle diameters
        :return: numpy.ndarray
        A 1d array of particle diameters
        """
        if self.distribution == "gaussian":
            print("When Gaussian distribution is used,"
                  " the particle statistics are computed using mean and std diameters\n"
                  "Particle min and max are cutoffs for the distribution")
            self.particle_field = rng.normal(self.mean_dia, self.std_dia, int(self.n_concentration))
            self.particle_field = np.clip(self.particle_field, self.min_dia, self.max_dia)
            np.random.shuffle(self.particle_field)
            return

        # TODO: Add Uniform distribution

    pass


class SpawnLocations:
    """
    Creates spawn locations array based on number of particles
    """
    def __init__(self, particles):
        self.x_min, self.x_max = None, None
        self.y_min, self.y_max = None, None
        self.z_min, self.z_max = None, None
        self.particles = particles
        self.locations = None

    def compute(self):
        """
        Computes the locations array to be passed into parallel
        Returns:

        """
        _size = self.particles.n_concentration
        # Draw a straight line between given points
        if self.x_max is None and self.z_max is None:
            _x_temp = np.repeat(self.x_min, _size).reshape(_size, 1)
            _z_temp = np.repeat(self.z_min, _size).reshape(_size, 1)
            _y_temp = np.linspace(self.y_min, self.y_max, _size).reshape(_size, 1)

            self.locations = np.hstack((_x_temp, _y_temp, _z_temp))
