import unittest
import numpy as np
import matplotlib.pyplot as plt


class TestPlot(unittest.TestCase):
    def test_plot(self):
        # filepath = '../data/shocks/particle_data/random_seed_7/raw_data/'
        # filepath = '../data/shocks/particle_data/random_seed_7_time_step_adaptive/'
        filepath = '../data/shock_interaction/final_grid/particle_data/'
        fig, ax = plt.subplots(2, 2)
        fig1, ax1 = plt.subplots(2, 2)
        fig2 = plt.figure()
        ax2 = plt.axes(projection='3d')
        for i in range(5):
            # data = np.load(filepath + 'particle_number_' + str(i) + '.npy')
            data = np.load(filepath + 'ppath_' + str(i) + '.npy')
            xp, yp, zp = data[:, 0], data[:, 1], data[:, 2]
            vx, vy, vz = data[:, 3], data[:, 4], data[:, 5]
            ux, uy, uz = data[:, 6], data[:, 7], data[:, 8]
            time = data[:, 9]
            xf, yf, zf = data[:, 10], data[:, 11], data[:, 12]

            # Plot in a single window
            ax[0, 0].plot(xp, yp, 'r')
            ax[0, 0].plot(xf, yf, 'b')
            ax[0, 0].set_title('paths')
            ax[0, 1].plot(xp, vx, '.-r')
            ax[0, 1].plot(xp, ux, '.-b')
            ax[0, 1].set_title('x-velocity')
            ax[1, 0].plot(xp, vy, 'r')
            ax[1, 0].plot(xp, uy, 'b')
            ax[1, 0].set_title('y-velocity')
            ax[1, 1].plot(xp, vz, 'r')
            ax[1, 1].plot(xp, uz, 'b')
            ax[1, 1].set_title('z-velocity')

            print('Error in particle to fluid path deviation: ', np.linalg.norm(data[:, 0:3] - data[:, 3:6]))

            ax1[0, 0].plot(xp, abs(xp-xf), 'k')
            ax1[0, 0].set_title('Difference in paths')
            ax1[0, 1].plot(xp, abs(vx-ux), 'k')
            ax1[0, 1].set_title('Difference in x-velocity')

            ax2.plot3D(xp, yp, zp, 'r')
            ax2.plot3D(xf, yf, zf, 'b')
            ax2.set_xlim(0, 0.005)
            ax2.set_ylim(0, 0.003)
            ax2.set_zlim(0, 0.001)
        plt.show()


if __name__ == '__main__':
    unittest.main()
