import unittest
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp
import random
from src.io.plot3dio import GridIO, FlowIO


class TestParallelFast(unittest.TestCase):
    def test_parallel_fast(self):
        """
        Parallel implementation for streamlines using gid and flow objects.
        This makes the code run faster as the GridIO and FlowIO are used only once

        Returns:
        """
        grid_file = '../../data/vortex/vortex.sb.sp.x'
        flow_file = '../../data/vortex/vortex.sb.sp.q'
        grid = GridIO(grid_file)
        flow = FlowIO(flow_file)

        # Read in the grid and flow data
        grid.read_grid()
        flow.read_flow()
        grid.compute_metrics()

        def test_vortex(start_point):
            """
            Local function to run in parallel
            """
            from src.streamlines.streamlines import Streamlines
            sl = Streamlines(None, None, start_point, time_step=1e-3)
            sl.compute(method='adaptive-p-space', grid=grid, flow=flow)

            return np.array(sl.streamline)

        # Points at which the streamline should start
        point_list = []
        npoints = 5
        for i in range(npoints):
            point_list.append([random.uniform(-npoints, npoints) / npoints,
                               random.uniform(-npoints, npoints) / npoints, 5])

        # Use two processes to run in parallel
        pool = Pool(mp.cpu_count() - 2)
        streamline_list = pool.map(test_vortex, point_list)
        pool.close()

        point_list = np.array(point_list)
        plt.scatter(point_list[:, 0], point_list[:, 1])

        for i in range(len(streamline_list)):
            plt.plot(streamline_list[i][:, 0], streamline_list[i][:, 1])
        plt.show()


if __name__ == '__main__':
    unittest.main()
