import unittest
from src.lptlib import GridIO, FlowIO
from src.lptlib import Streamlines
import numpy as np
import matplotlib.pyplot as plt

class TestUnsteadyStreamlines(unittest.TestCase):
    def test_unsteady_streamlines(self):
        path = './test_unsteady/cylinder_data/'
        grid_file, flow_file = path + "cylinder.sp.x", path + "sol-0000010.q"
        grid = GridIO(grid_file)
        grid.read_grid(data_type="f4")
        grid.compute_metrics()
        flow = FlowIO(flow_file)
        flow.read_unsteady_flow(data_type="f4")
        start_point = [1, 2.5, 0.5]
        sl = Streamlines(point=start_point)
        sl.time_step = 0.1
        sl.unsteady_time_step = 0.1
        sl.compute(grid=grid, flow=flow, method='unsteady-ppath')

        xdata = np.array(sl.streamline)
        vdata = np.array(sl.svelocity)
        udata = np.array(sl.fvelocity)
        tdata = np.array(sl.time).reshape(-1, 1)

        xp, yp, zp = xdata[:, 0], xdata[:, 1], xdata[:, 2]
        vx, vy, vz = vdata[:, 0], vdata[:, 1], vdata[:, 2]
        ux, uy, uz = udata[:, 0], udata[:, 1], udata[:, 2]

        ax = plt.axes()
        ax.plot(xp, vx, '.-r', label='Particle')
        ax.plot(xp, ux, '.-b', label='Fluid')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()

        plt.figure()
        plt.plot(xp, yp, '.-', label='Path')
        plt.show()


if __name__ == '__main__':
    unittest.main()
