# analysis for different data sets produced throughout the project

import numpy as np
import re
from fastdtw import fastdtw as fdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt


class Analysis:
    """
    Module to analyze data
    ----
    Attributes:
        compare_with_files: list of files to compare with
        files: list of files to compare
        plot_f1: file to plot
        plot_f2: file to plot
    """
    def __init__(self, compare_with_files, files):
        self.compare_with_files = compare_with_files
        self.files = files
        self.plot_f1 = None
        self.plot_f2 = None

    def modified_dtw(self):
        """
        Compute the modified dtw
        References: https://github.com/slaypni/fastdtw
        Returns:

        """
        for f1 in self.compare_with_files:
            for f2 in self.files:
                # load the data
                data1 = np.load(f1)
                data2 = np.load(f2)
                # extract the data
                x1, y1, z1 = data1[:, 0], data1[:, 1], data1[:, 2]
                vx1, vy1, vz1 = data1[:, 3] / data1[:, 3].max(), data1[:, 4] / data1[:, 4].max(), data1[:, 5]
                x2, y2, z2 = data2[:, 0], data2[:, 1], data2[:, 2]
                vx2, vy2, vz2 = data2[:, 3] / data2[:, 3].max(), data2[:, 4] / data2[:, 4].max(), data2[:, 5]
                # compute the dtw
                error_distance, warp_path = fdtw(np.vstack((x1, y1)).T, np.vstack((x2, y2)).T, dist=euclidean)
                error_xvelocity, warp_path_vel = fdtw(np.vstack((x1, vx1)).T, np.vstack((x2, vx2)).T, dist=euclidean)
                print('dtw distance between {} and {}: {}'.format(f1, f2, error_distance/len(warp_path)))
                print('dtw x-velocity between {} and {}: {}'.format(f1, f2, error_xvelocity/len(warp_path_vel)))

        return

    def plot_dtw(self):
        """
        Plot the dtw
        Returns:

        """
        d1 = np.load(self.plot_f1)
        d2 = np.load(self.plot_f2)

        # extract the data
        x1, y1, z1 = d1[:, 0], d1[:, 1], d1[:, 2]
        x2, y2, z2 = d2[:, 0], d2[:, 1], d2[:, 2]

        # compute the dtw
        # xp, yp are used to extract the paths for plotting
        xp = np.vstack((x1, y1)).T
        yp = np.vstack((x2, y2)).T
        error_distance, warp_path = fdtw(xp, yp, dist=euclidean)

        # plot the dtw
        fig, ax = plt.subplots(figsize=(16, 12))

        # Remove the border and axes ticks
        fig.patch.set_visible(False)
        ax.axis("off")
        for [map_x, map_y] in warp_path:
            ax.plot([xp[map_x][0], yp[map_y][0]], [xp[map_x][1], yp[map_y][1]], "-k")

        ax.plot(
            xp[:, 0],
            xp[:, 1],
            color="blue",
            marker="o",
            markersize=10,
            linewidth=5,
            label="dataset1",
        )
        ax.plot(
            yp[:, 0],
            yp[:, 1],
            color="g",
            marker="o",
            markersize=10,
            linewidth=5,
            label="dataset2",
        )
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.legend()
        ax.set_title("Set title using ax")
        plt.show()

        return ax

