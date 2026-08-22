# Class to run streamlines script in parallel
# Stochastic model for tracers is implemented

import logging
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp
from ..streamlines.streamlines import Streamlines
from scipy.stats import skewnorm, lognorm
from tqdm import tqdm
import socket

try:  # mpi4py needs a system MPI runtime; keep it optional at import time
    import mpi4py
    # See the identical note in lptlib.io.dataio: importing mpi4py.MPI would
    # otherwise call MPI_Init as a side effect of importing lptlib.
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = True
    from mpi4py import MPI
except (ImportError, RuntimeError):  # pragma: no cover - depends on the host
    MPI = None

logger = logging.getLogger(__name__)


def _require_mpi():
    """
    Raise a clear, actionable error when mpi4py/MPI is unavailable.

    Returns:
        The imported ``mpi4py.MPI`` module.
    """
    if MPI is None:
        raise ImportError(
            "This feature requires mpi4py and a working MPI runtime, which could not "
            "be imported. Install a system MPI implementation (e.g. "
            "'sudo apt-get install libopenmpi-dev openmpi-bin' on Debian/Ubuntu or "
            "'brew install open-mpi') and then reinstall mpi4py with "
            "'pip install --no-binary mpi4py --force-reinstall mpi4py'. "
            "Use StochasticModel.serial(), multi_thread() or multi_process() to run "
            "without MPI."
        )
    return MPI


def _init_mpi():
    """
    Ensure MPI is available *and* initialised before any communicator is used.

    ``mpi4py.rc.initialize`` is turned off at import time so that importing
    lptlib has no side effects, which means the first MPI entry point reached
    has to call ``MPI_Init`` itself.

    Returns:
        The imported ``mpi4py.MPI`` module, with MPI initialised.
    """
    _require_mpi()
    if not MPI.Is_initialized():
        MPI.Init()
    return MPI


class StochasticModel(Streamlines):
    """Module to spawn and run LPT on given tracers parallely

    ...

    Attributes
    ----------

    """

    def __init__(self, particles, spawn_locations, method='adaptive-p-space', grid=None, flow=None):
        super().__init__()
        self.particles = particles
        self.spawn_locations = spawn_locations
        # Read-in grid and flow files
        self.grid = grid
        self.flow = flow
        self.method = method
        self.chunksize = 32
        self.cpu_count = mp.cpu_count()

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
        if self.debug is True:
            logger.debug(f'Execution started for particle number - {task}')
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
        sl.adaptive_interpolation = self.adaptive_interpolation
        sl.max_loop_check = self.max_loop_check
        sl.max_steps = self.max_steps
        sl.debug = self.debug
        sl.compute(method=self.method, grid=self.grid, flow=self.flow)

        return sl

    def multi_process(self):
        """
        To parallelize using multiprocessing approach; the setup function
        Returns:

        """
        inputs = zip(self.spawn_locations.locations, self.particles.particle_field, np.arange(self.particles.n_concentration))
        with mp.Pool(self.cpu_count) as pool:
            # Wrap the *result* iterator, not the input one: starmap consumes
            # its inputs up front, so a bar around `inputs` filled to 100% as
            # soon as the work was dispatched rather than when it finished.
            # imap keeps the same ordering as starmap and yields one item per
            # completed particle, so the bar tracks real progress.
            lpt_data = list(tqdm(pool.imap(self._starmap_setup, inputs, chunksize=self.chunksize),
                                 total=self.particles.n_concentration))

        return lpt_data

    def _starmap_setup(self, args):
        """Unpack one ``(location, diameter, task)`` tuple into :meth:`setup`.

        ``Pool.imap`` passes a single argument, unlike ``starmap``.
        """
        return self.setup(*args)

    def multi_thread(self):
        """
        To parallelize using multithreading approach; the setup function
        Returns:

        """
        with Pool(self.cpu_count) as pool:
            lpt_data = pool.starmap(self.setup, zip(self.spawn_locations.locations, self.particles.particle_field,
                                                    np.arange(self.particles.n_concentration)), chunksize=self.chunksize)

        return lpt_data

    def serial(self):
        """
        To run setup in serial
        Returns:

        """
        # Run setup function in serial using tqdm
        lpt_data = []
        for i, (loc, dia) in tqdm(enumerate(zip(self.spawn_locations.locations, self.particles.particle_field)),
                                  total=self.particles.n_concentration):
            lpt_data.append(self.setup(loc, dia, i))

        return lpt_data

    def mpi_run(self):
        """
        To run setup in parallel using MPI.
        Run using mpiexec -np 8 python main.py
        To run in an IDE use another python file with subprocess.run
        Returns:

        """
        _init_mpi()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Split the data
        data_indices = np.array_split(np.arange(self.particles.n_concentration), size)
        data = data_indices[rank]

        # Run the setup function in parallel
        lpt_data = []
        for i, (loc, dia) in tqdm(enumerate(zip(self.spawn_locations.locations[data],
                                                self.particles.particle_field[data])),
                                  total=len(data), desc=f'{socket.gethostname()} Rank: {rank}'):
            lpt_data.append(self.setup(loc, dia, data[i]))

        return lpt_data


class Particle:
    """
    Class holds details for particles used in a PIV experiment
    ---
    User has to provide all the information to generate size distribution

    Parameters
    ----------
    seed : None, int or numpy.random.Generator, optional
        Controls the random draw in :meth:`compute_distribution`. ``None``
        (the default) seeds from OS entropy, so every run produces a different
        particle field. Pass an integer for a reproducible field, or an
        existing ``numpy.random.Generator`` to draw from a stream you own.

    Attributes
    ----------
    rng : numpy.random.Generator
        The generator used by the last :meth:`compute_distribution` call.

    Notes
    -----
    Reproducibility: a single ``numpy.random.Generator`` drives *all* of the
    draws -- gaussian, uniform, skewnorm, lognorm and the final shuffle -- and
    it is rebuilt from ``seed`` at the start of every
    :meth:`compute_distribution` call. So with an integer ``seed`` the same
    inputs always give the same diameters in the same order, on any machine,
    for any number of ``Particle`` objects, and repeated calls on one object
    agree too. With ``seed=None`` every call differs.

    Earlier versions were reproducible in neither direction: the gaussian and
    uniform draws came from a module-level generator with a hardcoded seed of
    7 that advanced between calls, skewnorm and lognorm hardcoded
    ``random_state=7``, and the final ordering came from the unseeded legacy
    ``numpy.random`` global. The diameters were frozen while their order was
    not, and two identically configured ``Particle`` objects disagreed.
    """

    def __init__(self, seed=None):
        self.distribution = "gaussian"
        self.min_dia = None
        self.max_dia = None
        self.mean_dia = None
        self.std_dia = None
        self.density = None
        self.n_concentration = None
        self.particle_field = None
        self.distribution_parameter = None  # skew - skewnorm, shape - lognorm, etc
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def compute_distribution(self):
        """
        Run this method to return a distribution of particle diameters
        :return: numpy.ndarray
        A 1d array of particle diameters

        The draw and the shuffle both come from ``self.rng``, rebuilt here from
        ``self.seed``; see the class docstring for the reproducibility contract.
        """
        # One generator for every draw below, including the shuffle. Rebuilding
        # it here (rather than reusing an advanced one) is what makes a seeded
        # Particle give the same field on every call and in every process.
        self.rng = np.random.default_rng(self.seed)
        rng = self.rng

        if self.distribution == "gaussian":
            logger.info("When Gaussian distribution is used,"
                  " the particle statistics are computed using mean and std diameters\n"
                  "Particle min and max are cutoffs for the distribution")
            self.particle_field = rng.normal(self.mean_dia, self.std_dia, int(self.n_concentration))

        if self.distribution == 'skewnorm':
            logger.info("When Skewnorm distribution is used,"
                  " the particle statistics are computed using mean and std diameters\n"
                  "Particle min and max are cutoffs for the distribution")
            try:
                a = self.distribution_parameter
                self.particle_field = skewnorm.rvs(a, loc=self.mean_dia, scale=self.std_dia,
                                                   size=int(self.n_concentration), random_state=rng)
            except ValueError:
                logger.error("Skewness parameter is not provided")
                raise ValueError

        if self.distribution == 'lognorm':
            logger.info("When Lognorm distribution is used,"
                  " the particle statistics are computed using mean and std diameters\n"
                  "Particle min and max are cutoffs for the distribution")
            try:
                s = self.distribution_parameter
                self.particle_field = lognorm.rvs(s, loc=self.mean_dia, scale=self.std_dia,
                                                  size=int(self.n_concentration), random_state=rng)
            except ValueError:
                logger.error("Shape parameter is not provided")
                raise ValueError

        if self.distribution == 'uniform':
            logger.info("When Uniform distribution is used,"
                  " the particle statistics are computed using min and max diameters\n"
                  "Particle min and max are cutoffs for the distribution")
            self.particle_field = rng.uniform(self.min_dia, self.max_dia, int(self.n_concentration))

        # Continue to clip the distribution to min and max diameters
        self.particle_field = np.clip(self.particle_field, self.min_dia, self.max_dia)
        # Shuffle through the same generator: the legacy np.random.shuffle
        # global is unseeded, which left the ordering irreproducible even when
        # the diameters themselves were fixed.
        rng.shuffle(self.particle_field)


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

        if self.y_max is None and self.z_max is None:
            _x_temp = np.linspace(self.x_min, self.x_max, _size).reshape(_size, 1)
            _z_temp = np.repeat(self.z_min, _size).reshape(_size, 1)
            _y_temp = np.repeat(self.y_min, _size).reshape(_size, 1)

            self.locations = np.hstack((_x_temp, _y_temp, _z_temp))

        if self.x_max is None and self.y_max is None:
            _x_temp = np.repeat(self.x_min, _size).reshape(_size, 1)
            _z_temp = np.linspace(self.z_min, self.z_max, _size).reshape(_size, 1)
            _y_temp = np.repeat(self.y_min, _size).reshape(_size, 1)

            self.locations = np.hstack((_x_temp, _y_temp, _z_temp))
