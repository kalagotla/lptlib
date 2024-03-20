import unittest
import numpy as np
from parameterized import parameterized
from src.function.timer import Timer
import matplotlib.pyplot as plt


class TestDragModel(unittest.TestCase):
    @parameterized.expand([
        ('henderson', 'adaptive-ppath', 1e-8, 'henderson'),
        ('stokes', 'adaptive-ppath', 1e-8, 'stokes'),
        ('oseen', 'adaptive-ppath', 1e-8, 'oseen'),
        ('schiller-nauman', 'adaptive-ppath', 1e-8, 'schiller-nauman'),
        ('cunningham', 'adaptive-ppath', 1e-8, 'cunningham'),
        ('tedeschi', 'adaptive-ppath', 1e-8, 'tedeschi'),
    ])
    def test_drag_model(self, name, method='pRK4', time_step=1e-4, drag_model='stokes'):
        from src.streamlines.streamlines import Streamlines
        sl = Streamlines('../../data/shocks/shock_test.sb.sp.x', '../../data/shocks/shock_test.sb.sp.q',
                         [15e-4, 2e-4, 2e-4])
        sl.diameter = 5e-7
        sl.density = 1000
        sl.time_step = time_step
        sl.max_time_step = 10
        sl.drag_model = drag_model
        sl.compute(method=method)

        xdata = np.array(sl.streamline)
        vdata = np.array(sl.svelocity)
        udata = np.array(sl.fvelocity)
        data_save = np.hstack((xdata, vdata, udata))
        # np.save('../../data/shocks/' + name + str(sl.diameter), data_save)
        print('Data written to file: ' + name + str(sl.diameter))
        xp, yp, zp = xdata[:, 0], xdata[:, 1], xdata[:, 2]
        vx, vy, vz = vdata[:, 0], vdata[:, 1], vdata[:, 2]
        ux, uy, uz = udata[:, 0], udata[:, 1], udata[:, 2]

        ax = plt.axes()
        # ax.plot(xp, vx, 'r', label='Particle')
        # ax.plot(xp, ux, 'b', label='Fluid')
        ax.plot(xp, yp, '.-', label='Path')
        ax.set_title(name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 38e-4)
        ax.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
