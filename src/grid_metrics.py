import numpy as np


class GridMetrics:
    """Module to calculate grid metrics from grid data

        author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
        date: 10-05/2021

        Example:
            gm = GridMetrics(grid)  # Assume grid is the object from GridIO
            gm.compute()  # Call method to compute grid metrics
            print(flow)  # prints the docstring for grid metrics
            # Instance attributes
            print(gm.m1)  # derivatives xi, eta, zeta
            print(gm.m2)  # derivatives x, y, z
            """

    def __init__(self, grid):
        self.grid = grid
        self.m1 = None
        self.m2 = None
        self.J = None

    def compute(self):
        self.m1 = np.zeros((self.grid.ni, self.grid.nj, self.grid.nk, 3, 3))
        self.m2 = np.zeros((self.grid.ni, self.grid.nj, self.grid.nk, 3, 3))
        self.J = np.zeros((self.grid.ni, self.grid.nj, self.grid.nk))

        #  The for loop is to compute xi, eta, zeta derivatives
        for i in range(3):
            self.m1[..., i] = np.gradient(self.grid.grd[...], axis=i)

        # compute Jacobian
        self.J = self.m1[..., 0, 0] * (
                    self.m1[..., 1, 1] * self.m1[..., 2, 2] - self.m1[..., 1, 2] * self.m1[..., 2, 1]) - \
                 self.m1[..., 1, 0] * (
                             self.m1[..., 0, 1] * self.m1[..., 2, 2] - self.m1[..., 0, 2] * self.m1[..., 2, 1]) + \
                 self.m1[..., 2, 0] * (
                             self.m1[..., 0, 1] * self.m1[..., 1, 2] - self.m1[..., 0, 2] * self.m1[..., 1, 1])

        # x derivatives
        self.m2[..., 0, 0] = (self.m1[..., 1, 1] * self.m1[..., 2, 2] - self.m1[..., 1, 2] * self.m1[..., 2, 1]) / self.J
        self.m2[..., 1, 0] = (self.m1[..., 1, 2] * self.m1[..., 2, 0] - self.m1[..., 1, 0] * self.m1[..., 2, 2]) / self.J
        self.m2[..., 2, 0] = (self.m1[..., 1, 0] * self.m1[..., 2, 1] - self.m1[..., 1, 1] * self.m1[..., 2, 0]) / self.J

        # y derivative
        self.m2[..., 0, 1] = (self.m1[..., 0, 2] * self.m1[..., 2, 1] - self.m1[..., 0, 1] * self.m1[..., 2, 2]) / self.J
        self.m2[..., 1, 1] = (self.m1[..., 0, 0] * self.m1[..., 2, 2] - self.m1[..., 0, 2] * self.m1[..., 2, 0]) / self.J
        self.m2[..., 2, 1] = (self.m1[..., 0, 1] * self.m1[..., 2, 0] - self.m1[..., 0, 0] * self.m1[..., 2, 1]) / self.J

        # z derivatives
        self.m2[..., 0, 2] = (self.m1[..., 0, 0] * self.m1[..., 1, 2] - self.m1[..., 0, 2] * self.m1[..., 1, 1]) / self.J
        self.m2[..., 1, 2] = (self.m1[..., 0, 1] * self.m1[..., 1, 0] - self.m1[..., 0, 0] * self.m1[..., 1, 2]) / self.J
        self.m2[..., 2, 2] = (self.m1[..., 0, 2] * self.m1[..., 1, 1] - self.m1[..., 0, 1] * self.m1[..., 1, 0]) / self.J
