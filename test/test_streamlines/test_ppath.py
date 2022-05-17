import unittest
import numpy as np
from parameterized import parameterized
from src.function.timer import Timer


class TestPPath(unittest.TestCase):
    # noinspection DuplicatedCode
    @parameterized.expand([
        ('adaptive-ppath-p-space', 'adaptive-ppath', 1e-4),
        ('ppath-p-space', 'ppath', 1e-2),
        ('adaptive-p-space', 'adaptive-p-space', 1e-4),
        ('p-space', 'p-space', 1e-2),
        ('adaptive-c-space', 'adaptive-c-space', 1e-4)
    ])
    @Timer()
    def test_ppath(self, name, method='pRK4', time_step=1e-4):
        """
        Applies streamlines algo for vortex field.
        Details of the vortex can be found in Murman and Powell, 1987

        Returns:
        """
        from src.streamlines.streamlines import Streamlines
        sl = Streamlines('../../data/vortex/vortex.sb.sp.x', '../../data/vortex/vortex.sb.sp.q', [-0.05, 0.05, 5])
        sl.diameter = 0
        sl.density = 1000
        sl.time_step = time_step
        sl.max_time_step = 1
        sl.compute(method=method)

        import matplotlib.pyplot as plt
        data = np.array(sl.streamline)
        np.save('../../data/vortex/' + name, data)
        print('Data written to file: ' + name)
        xp, yp, zp = data[:, 0], data[:, 1], data[:, 2]

        ax = plt.axes(projection='3d')
        ax.plot3D(xp, yp, zp, 'b', label='SB-P')
        ax.set_title('Comparing different particle path algorithms')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
