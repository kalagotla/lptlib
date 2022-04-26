# Tested the code in jupyter-notebook. It works!
# Find the file in vortex_test.ipynb
# Having issues with pycharm IDE

import unittest
import numpy as np
from multiprocess import Pool


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
            from src.streamlines.streamlines import Streamlines
            sl = Streamlines('../../data/vortex/vortex.sb.sp.x', '../../data/vortex/vortex.sb.sp.q', start_point,
                             time_step=1)
            sl.compute(method='p-space')

            import matplotlib.pyplot as plt
            data = np.array(sl.streamline)
            xp, yp, zp = data[:, 0], data[:, 1], data[:, 2]

            ax = plt.axes(projection='3d')
            ax.plot3D(xp, yp, zp, 'b', label='SB-P')
            ax.set_title('Comparing different particle path algorithms')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.legend()
            plt.show()

            return

        # Points at which the streamline should start
        point_list = [[-0.05, 0.05, 5], [-0.06, 0.05, 5]]

        # Use two processes to run in parallel
        pool = Pool(2)
        streamline_list = pool.map(test_vortex, point_list)
        pool.close()


if __name__ == '__main__':
    unittest.main()