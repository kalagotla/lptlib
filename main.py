# Script to create an oblique shock and run the stochastic model

import os
import seaborn as sns
import matplotlib.pyplot as plt
from src.lptlib import ObliqueShock, ObliqueShockData
from src.lptlib import StochasticModel, Particle, SpawnLocations


def oblique_shock_response(filepath='./tio2_particle/', dp=5.272e-6, rhop=182.225):

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
    p.n_concentration = 1000  # number of particles 2 per cell (y-direction)
    p.distribution = 'gaussian'
    p.compute_distribution()
    try:
        os.mkdir(filepath + '_temp')
        sns.displot(p.particle_field, bins=50)  # doesn't show the plot until plt.show() is called
        plt.savefig(filepath + '_temp/particle_distribution.svg', format='svg', dpi=1200)
    except:
        sns.displot(p.particle_field, bins=50)  # doesn't show the plot until plt.show() is called
        plt.savefig(filepath + '_temp/particle_distribution.svg', format='svg', dpi=1200)

    # Test SpawnLocations class
    l = SpawnLocations(p)
    l.x_min = -50e-3
    l.z_min = 5e-5
    l.y_min, l.y_max = 0, osd.ny_max  # same values spawn particles at that point ideal for response analysis
    l.compute()

    # Run the model in parallel
    grid = osd.grid
    flow = osd.flow
    sm = StochasticModel(p, l, grid=grid, flow=flow)
    sm.method = 'adaptive-ppath'
    sm.search = 'p-space'
    sm.time_step = 1e-10
    sm.max_time_step = 1
    sm.interpolation = 'simple_oblique_shock'
    # sm.adaptive_interpolation = 'shock'
    sm.drag_model = 'loth'
    # save to the filepath
    sm.filepath = filepath
    lpt_data = sm.multi_process()


if __name__ == '__main__':
    oblique_shock_response(filepath='./exp_estimated_particle/', dp=1.94e-6, rhop=950)

