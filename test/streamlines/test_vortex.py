import unittest
import numpy as np


class TestVortex(unittest.TestCase):
    def test_vortex(self):
        """
        Applies streamlines algo for vortex field.
        Details of the vortex can be found in Murman and Powell, 1987

        Returns:
        """
        from src.streamlines.streamlines import Streamlines
        sl = Streamlines('../../data/vortex/vortex.sb.sp.x', '../../data/vortex/vortex.sb.sp.q', [-0.05, 0.05, 5],
                         integration_method='pRK4', time_step=1)
        sl.compute()

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


if __name__ == '__main__':
    unittest.main()
