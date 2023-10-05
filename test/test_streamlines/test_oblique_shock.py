import unittest
import numpy as np
from parameterized import parameterized
from src.function.timer import Timer
import matplotlib.pyplot as plt


class TestObliqueShock(unittest.TestCase):
    @parameterized.expand([
        ('adaptive-ppath-p-space', 'adaptive-ppath', 1e-7),
        ('ppath-p-space', 'ppath', 1e-8),
        ('ppath-c-space', 'ppath-c-space', 1e-8),
        ('adaptive-ppath-c-space', 'adaptive-ppath-c-space', 1e-8),
        ('adaptive-p-space', 'adaptive-p-space', 1e-8),
        ('p-space', 'p-space', 1e-8),
        ('adaptive-c-space', 'adaptive-c-space', 1e-7),
        ('c-space', 'c-space', 1e-9),
    ])
    @Timer()
    def test_oblique_shock(self, name, method='adaptive-ppath', time_step=1e-4):
        from src.streamlines.streamlines import Streamlines
        sl = Streamlines('../../data/shocks/m5_d20_strong.sb.sp.x', '../../data/shocks/m5_d20_strong.sb.sp.q',
                         [15e-4, 2e-4, 2e-4])
        # Best TiO2 specs
        sl.diameter = 250e-9
        sl.density = 4200
        sl.time_step = time_step
        # sl.max_time_step = 1e-10
        sl.adaptivity = 0.01
        sl.magnitude_adaptivity = 0.01
        sl.drag_model = 'henderson'
        sl.interpolation = 'p-space'
        sl.adaptive_interpolation = 'shock'
        sl.compute(method=method)

        xdata = np.array(sl.streamline)
        vdata = np.array(sl.svelocity)
        udata = np.array(sl.fvelocity)
        tdata = np.array(sl.time).reshape(-1, 1)

        data = np.hstack((xdata, vdata, udata, tdata))
        np.save('../../data/shocks/' + name + str(sl.diameter) + '_' + sl.drag_model, data)
        print('Data written to file: ' + name + str(sl.diameter) + sl.drag_model)
        xp, yp, zp = xdata[:, 0], xdata[:, 1], xdata[:, 2]
        vx, vy, vz = vdata[:, 0], vdata[:, 1], vdata[:, 2]
        ux, uy, uz = udata[:, 0], udata[:, 1], udata[:, 2]

        ax = plt.axes()
        ax.plot(xp, vx, '.-r', label='Particle')
        ax.plot(xp, ux, '.-b', label='Fluid')
        ax.set_title(name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 38e-4)
        ax.legend()

        plt.figure()
        plt.plot(xp, yp, '.-', label='Path')
        plt.xlim(0, 38e-4)
        plt.show()


if __name__ == '__main__':
    unittest.main()
