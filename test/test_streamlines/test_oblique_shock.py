import unittest
import numpy as np
from parameterized import parameterized
from src.function.timer import Timer
import matplotlib.pyplot as plt


class TestObliqueShock(unittest.TestCase):
    @parameterized.expand([
        # ('adaptive-ppath-p-space', 'adaptive-ppath', 1e-7),
        # ('ppath-p-space', 'ppath', 1e-7),
        ('adaptive-p-space', 'adaptive-p-space', 1e-7),
        # ('p-space', 'p-space', 1e-7),
        # ('adaptive-c-space', 'adaptive-c-space', 1e-9),
        # ('c-space', 'c-space', 1e-7),
    ])
    @Timer()
    def test_oblique_shock(self, name, method='pRK4', time_step=1e-4):
        from src.streamlines.streamlines import Streamlines
        sl = Streamlines('../../data/shocks/shock_test.sb.sp.x', '../../data/shocks/shock_test.sb.sp.q',
                         [18e-4, 2e-4, 2e-4])
        sl.diameter = 5e-7
        sl.density = 1000
        sl.time_step = time_step
        sl.max_time_step = 1
        sl.compute(method=method)

        xdata = np.array(sl.streamline)
        vdata = np.array(sl.svelocity)
        udata = np.array(sl.fvelocity)
        np.save('../../data/shocks/' + name + str(sl.diameter), xdata)
        print('Data written to file: ' + name + str(sl.diameter))
        xp, yp, zp = xdata[:, 0], xdata[:, 1], xdata[:, 2]
        vx, vy, vz = vdata[:, 0], vdata[:, 1], vdata[:, 2]
        ux, uy, uz = udata[:, 0], udata[:, 1], udata[:, 2]

        ax = plt.axes()
        ax.plot(xp, vx, 'r', label='Particle')
        ax.plot(xp, ux, 'b', label='Fluid')
        # ax.plot(xp, yp, label='Path')
        ax.set_title('Shock Normal Velocity')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 38e-4)
        ax.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
