import unittest
import numpy as np
import matplotlib.pyplot as plt


class TestPlot(unittest.TestCase):
    def test_plot(self):
        # filepath = '../data/shocks/particle_data/random_seed_7/raw_data/'
        # filepath = '../data/shocks/particle_data/multi_process_test/final_data/'
        filepath = '../data/shock_interaction/final_grid/particle_data/'
        fig, ax = plt.subplots(2, 2)
        for i in range(2):
            data = np.load(filepath + 'particle_number_' + str(i) + '.npy')
            # data = np.load(filepath + 'ppath_' + str(i) + '.npy')
            xp, yp, zp = data[:, 0], data[:, 1], data[:, 2]
            vx, vy, vz = data[:, 3], data[:, 4], data[:, 5]
            ux, uy, uz = data[:, 6], data[:, 7], data[:, 8]

            # Plot in a single window
            ax[0, 0].plot(xp, yp, '.r')
            ax[0, 0].set_title('paths')
            ax[0, 1].plot(xp, vx, 'r')
            ax[0, 1].plot(xp, ux, 'b')
            ax[0, 1].set_title('x-velocity')
            ax[1, 0].plot(xp, vy, 'r')
            ax[1, 0].plot(xp, uy, 'b')
            ax[1, 0].set_title('y-velocity')
            ax[1, 1].plot(xp, vz, 'r')
            ax[1, 1].plot(xp, uz, 'b')
            ax[1, 1].set_title('z-velocity')
        plt.show()


if __name__ == '__main__':
    unittest.main()
