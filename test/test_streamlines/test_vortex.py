import unittest
import numpy as np
from parameterized import parameterized
from src.lptlib.function import Timer
import matplotlib.pyplot as plt


class TestVortex(unittest.TestCase):
    # noinspection DuplicatedCode
    @parameterized.expand([
        # ('adaptive-ppath-p-space', 'adaptive-ppath', 1e-4),
        # ('ppath-p-space', 'ppath', 5),
        # ('ppath-c-space', 'ppath-c-space', 1e-10),
        # ('adaptive-ppath-c-space', 'adaptive-ppath-c-space', 1e-9),
        # ('adaptive-p-space', 'adaptive-p-space', 1e-6),
        ('p-space', 'p-space', 1e-2),
        # ('adaptive-c-space', 'adaptive-c-space', 1e-2),  # Workaround based on adaptivity 0.11
        # ('c-space', 'c-space', 1e-6),
    ])
    @Timer()
    def test_vortex(self, name, method='p-space', time_step=1e-1):
        """
        Applies streamlines algo for vortex field.
        Details of the vortex can be found in Murman and Powell, 1987

        Returns:
        """
        from src.lptlib.streamlines import Streamlines
        sl = Streamlines('../../data/vortex/vortex.sb.sp.x', '../../data/vortex/vortex.sb.sp.q', [-0.05, 0.05, 5],
                         time_step=time_step)
        sl.compute(method=method)

        xdata = np.array(sl.streamline)
        vdata = np.array(sl.svelocity)
        udata = np.array(sl.fvelocity)
        tdata = np.array(sl.time).reshape(-1, 1)

        data = np.hstack((xdata, vdata, udata, tdata))
        np.save('../../data/vortex/' + name + str(sl.diameter) + '_' + sl.drag_model + '_time' + str(time_step), data)
        print('Data written to file: ' + name + str(sl.diameter) + sl.drag_model)
        xp, yp, zp = xdata[:, 0], xdata[:, 1], xdata[:, 2]
        vx, vy, vz = vdata[:, 0], vdata[:, 1], vdata[:, 2]
        ux, uy, uz = udata[:, 0], udata[:, 1], udata[:, 2]

        ax = plt.axes()
        ax.plot(xp, vx, '-r', label='Particle')
        ax.plot(xp, ux, '-b', label='Fluid')
        ax.set_title(name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()

        plt.figure()
        plt.plot(xp, yp, '-', label='Path')
        plt.show()


if __name__ == '__main__':
    unittest.main()
