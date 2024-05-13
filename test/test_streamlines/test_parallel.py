# Tested the code in jupyter-notebook. It works!
# Find the file in vortex_test.ipynb
# Having issues with pycharm IDE

import unittest
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp
import random


class TestParallel(unittest.TestCase):
    def test_parallel(self):
        """
        Parallel implementation for streamlines

        Returns:
        """

        def test_vortex(start_point):
            """
            Local function to run in parallel
            """
            from src.lptlib.streamlines import Streamlines
            sl = Streamlines('../../data/vortex/vortex.sb.sp.x', '../../data/vortex/vortex.sb.sp.q', start_point,
                             time_step=1e-3)
            sl.compute(method='adaptive-p-space')

            return np.array(sl.streamline)

        # Points at which the streamline should start
        point_list = []
        npoints = 500
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
