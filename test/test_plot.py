import unittest
import numpy as np
import matplotlib.pyplot as plt


class TestPlot(unittest.TestCase):
    def test_plot(self):
        # filepath = '../data/shocks/particle_data/random_seed_7/raw_data/'
        filepath = '../data/shocks/particle_data/multi_process_test/'
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        for i in range(5):
            # data = np.load(filepath + 'particle_number_' + str(i) + '.npy')
            data = np.load(filepath + 'ppath_' + str(i) + '.npy')
            xp, yp, zp = data[:, 0], data[:, 1], data[:, 2]
            vx, vy, vz = data[:, 3], data[:, 4], data[:, 5]
            ux, uy, uz = data[:, 6], data[:, 7], data[:, 8]

            # plot velocities
            ax.plot(xp, vx, 'r', label='Particle')
            ax.plot(xp, ux, 'b', label='Fluid')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(0, 38e-4)

            # plot paths in a separate window
            ax1.plot(xp, yp, '.-', label='Path')
            ax1.set_xlim(0, 38e-4)

        plt.show()


if __name__ == '__main__':
    unittest.main()
