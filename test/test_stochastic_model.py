import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.lptlib.streamlines import StochasticModel, Particle, SpawnLocations
from src.lptlib.io import GridIO, FlowIO


class TestStochasticModel(unittest.TestCase):
    def test_stochastic_model(self):
        # Test particle class
        p = Particle()
        p.min_dia = 281e-9
        p.max_dia = 281e-9
        p.mean_dia = 281e-9
        p.std_dia = 0
        p.density = 813
        p.n_concentration = 1300
        p.compute_distribution()

        # Test SpawnLocations class
        l = SpawnLocations(p)
        l.x_min = 9e-4
        l.z_min = 2e-4
        l.y_min, l.y_max = 2e-4, 15e-4
        l.compute()

        # Run the model in parallel
        grid_file, flow_file = '../data/shocks/shock_test.sb.sp.x', '../data/shocks/shock_test.sb.sp.q'
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
        sm.max_time_step = 1
        sm.adaptivity = 0.001
        # this saves data after every process is done. This will open up memory as well
        sm.filepath = '../data/shocks/particle_data/281nm_time_step_adaptive/'

        # Run multiprocess
        lpt_data = sm.multi_process()

        # save data
        # for i in range(p.n_concentration):
        #     xdata = np.array(lpt_data[i].streamline)
        #     vdata = np.array(lpt_data[i].svelocity)
        #     udata = np.array(lpt_data[i].fvelocity)
        #     data_save = np.hstack((xdata, vdata, udata))
            # np.save('../data/shocks/particle_data/multi_process_test/final_data/' + 'particle_number_' + str(i),
            #         data_save)


if __name__ == '__main__':
    unittest.main()
