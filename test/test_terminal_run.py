#!/usr/bin/env python3

import numpy as np
import sys

path = '/Users/kal/Library/CloudStorage/OneDrive-UniversityofCincinnati' \
       '/Desktop/University of Cincinnati/DoctoralWork/Codes/project-arrakis'
sys.path.append(path)
import matplotlib.pyplot as plt
from src.lptlib.streamlines import StochasticModel, Particle, SpawnLocations
from src.lptlib.io import GridIO, FlowIO


def test_stochastic_model():
    # Test particle class
    p = Particle()
    p.min_dia = 177e-9
    p.max_dia = 573e-9
    p.mean_dia = 281e-9
    p.std_dia = 97e-9
    p.density = 813
    p.n_concentration = 4
    p.compute_distribution()

    # Test SpawnLocations class
    l = SpawnLocations(p)
    l.x_min = 9e-4
    l.z_min = 2e-4
    l.y_min, l.y_max = 2e-4, 15e-4
    l.compute()

    # Run the model in parallel
    grid_file, flow_file = path + '/data/shocks/shock_test.sb.sp.x', path + '/data/shocks/shock_test.sb.sp.q'
    grid = GridIO(grid_file)
    grid.read_grid()
    grid.compute_metrics()
    flow = FlowIO(flow_file)
    flow.read_flow()
    sm = StochasticModel(p, l, grid=grid, flow=flow)
    sm.method = 'adaptive-ppath'
    sm.drag_model = "henderson"
    sm.search = 'p-space'
    sm.time_step = 1e-10
    sm.max_time_step = 1e-10
    sm.filepath = path + '/data/shocks/particle_data/multi_process_test/'

    # Run multiprocess
    lpt_data = sm.multi_process()


if __name__ == '__main__':
    test_stochastic_model()
