# Script to create an oblique shock and run the stochastic model

import logging
import os
import time

import seaborn as sns
import matplotlib.pyplot as plt
from lptlib import ObliqueShock, ObliqueShockData
from lptlib import StochasticModel, Particle, SpawnLocations

logger = logging.getLogger(__name__)


# One particle takes a few minutes to cross the domain at the time step below,
# so the default is a small demonstration ensemble rather than the 1000-particle
# cloud used for the published results.
DEFAULT_N_CONCENTRATION = 4
MINUTES_PER_PARTICLE = 4.0


def oblique_shock_response(filepath='./tio2_particle/', dp=5.272e-6, rhop=182.225,
                           n_concentration=DEFAULT_N_CONCENTRATION):
    """Track a monodisperse particle cloud through a Mach 7.6 oblique shock.

    Args:
        filepath: directory to write the per-particle trajectories into. It is
            created if it does not exist, along with a ``_temp`` sibling for
            the diagnostic plots.
        dp: particle diameter in meters.
        rhop: particle material density in kg/m^3.
        n_concentration: number of particles to track. Each particle takes a
            few minutes to cross the domain, and particles run in parallel
            across CPU cores, so the wall-clock cost is roughly
            ``4 min * ceil(n_concentration / cpu_count)``. The default of 4 is
            a quick demonstration; the published results use 1000.
    """
    # Create oblique shock
    os1 = ObliqueShock()
    os1.mach = 7.6
    os1.deflection = 20  # degrees
    os1.compute()

    # Create grid and flow files
    osd = ObliqueShockData()
    osd.oblique_shock = os1
    osd.nx_max = 100e-3  # 100 mm
    osd.ny_max = 500e-3  # 500 mm
    osd.nz_max = 1e-4  # 0.1 mm
    osd.inlet_temperature = 48.20  # K
    osd.inlet_density = 0.07747  # kg/m^3
    osd.xpoints = 200  # 200 points
    osd.ypoints = 500  # 500 points
    osd.zpoints = 5  # 5 points
    osd.shock_strength = 'weak'
    osd.create_grid()
    osd.create_flow()

    # Test particle class
    p = Particle()
    # Constant particle size
    p.min_dia = dp
    p.max_dia = dp
    p.mean_dia = dp
    p.std_dia = 0
    p.density = rhop
    p.n_concentration = n_concentration
    p.distribution = 'gaussian'
    p.compute_distribution()

    # Output directories. makedirs creates the parent too, and exist_ok makes
    # a re-run a no-op instead of an error.
    os.makedirs(filepath, exist_ok=True)
    os.makedirs(os.path.join(filepath, '_temp'), exist_ok=True)
    sns.displot(p.particle_field, bins=50)
    plt.savefig(os.path.join(filepath, '_temp', 'particle_distribution.svg'),
                format='svg', dpi=1200)
    plt.close('all')

    # Test SpawnLocations class
    spawn = SpawnLocations(p)
    spawn.x_min = -50e-3
    spawn.z_min = 5e-5
    # Spawn across the full height of the domain. Setting y_min == y_max
    # instead spawns every particle at one point, which is the ideal setup for
    # a response analysis.
    spawn.y_min, spawn.y_max = 0, osd.ny_max
    spawn.compute()

    # Run the model in parallel
    grid = osd.grid
    flow = osd.flow
    sm = StochasticModel(p, spawn, grid=grid, flow=flow)
    sm.method = 'adaptive-ppath'
    sm.search = 'p-space'
    sm.time_step = 1e-10
    sm.max_time_step = 1
    sm.interpolation = 'simple_oblique_shock'
    # sm.adaptive_interpolation = 'shock'
    sm.drag_model = 'loth'
    # save to the filepath
    sm.filepath = filepath
    # One task per worker rather than the default chunk of 32, so a small
    # ensemble actually spreads across the available cores.
    sm.chunksize = 1

    _rounds = -(-n_concentration // max(sm.cpu_count, 1))
    logger.info('Tracking %d particles through the shock on %d cores. '
                'Expect roughly %.0f minutes (about %.0f min per particle, '
                '%d round(s)). Pass a smaller n_concentration to shorten this, '
                'or n_concentration=1000 to reproduce the published cloud.',
                n_concentration, sm.cpu_count,
                _rounds * MINUTES_PER_PARTICLE, MINUTES_PER_PARTICLE, _rounds)
    _start = time.perf_counter()
    lpt_data = sm.multi_process()
    logger.info('Done. Tracked %d particles in %.1f s. Trajectories written to %s',
                n_concentration, time.perf_counter() - _start,
                os.path.abspath(filepath))

    return lpt_data


if __name__ == '__main__':
    # Applications configure logging; the library itself never calls basicConfig.
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    oblique_shock_response(filepath='./exp_estimated_particle/', dp=1.94e-6,
                           rhop=950)
