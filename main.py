# TODO: Data files to run this script are to be added to the repository

import numpy as np
import matplotlib.pyplot as plt
from src import StochasticModel, Particle, SpawnLocations, GridIO, FlowIO


def shock_interaction():
    # Test particle class
    p = Particle()
    p.min_dia = 250e-9
    p.max_dia = 550e-9
    p.mean_dia = 340e-9
    p.std_dia = 10e-9
    p.density = 4200
    p.n_concentration = 1
    p.compute_distribution()

    # Test SpawnLocations class
    l = SpawnLocations(p)
    l.x_min = 0.0001
    l.z_min = 0.0005
    l.y_min, l.y_max = 0.001, 0.0016
    l.compute()

    # Run the model in parallel
    path = './data/shock_interaction/final_grid_coarse/'
    grid_file, flow_file = path + 'coarse_python.x', path + '37500_overflow_eqn_corrected.txt'
    grid = GridIO(grid_file)
    grid.read_grid(data_type='f8')
    grid.compute_metrics()
    flow = FlowIO(flow_file)
    flow.mach = 5.0
    flow.rey = 1.188e8
    flow.alpha = 0.0
    flow.time = 1.0
    flow.read_formatted_txt(grid=grid, data_type='f8')
    sm = StochasticModel(p, l, grid=grid, flow=flow)
    sm.method = 'adaptive-ppath'
    # sm.method = 'ppath-c-space'
    sm.drag_model = 'henderson'
    sm.search = 'p-space'
    sm.time_step = 1e-10
    sm.max_time_step = 1
    sm.adaptivity = 0.001
    sm.magnitude_adaptivity = 0.001
    # this saves data after every process is done. This will open up memory as well
    # To test multiple drag models
    sm.filepath = path + 'drag_models/'

    # Run multiprocess
    lpt_data = sm.multi_process()

    print('**** DONE ******'*5)

    ax = plt.axes()
    fig1 = plt.figure()
    ax1 = plt.axes()
    for i in range(p.n_concentration):
        xdata = np.array(lpt_data[i].streamline)
        vdata = np.array(lpt_data[i].svelocity)
        udata = np.array(lpt_data[i].fvelocity)
        xp, yp, zp = xdata[:, 0], xdata[:, 1], xdata[:, 2]
        vx, vy, vz = vdata[:, 0], vdata[:, 1], vdata[:, 2]
        ux, uy, uz = udata[:, 0], udata[:, 1], udata[:, 2]
        #
        ax.plot(xp, vx, '.-r', label='Particle')
        ax.plot(xp, ux, '.-b', label='Fluid')
        ax1.plot(xp, yp, '.-', label='Path')
    ax.legend()
    ax.set_title(sm.method)
    plt.show()


if __name__ == '__main__':
    shock_interaction()
