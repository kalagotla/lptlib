# This test case is for plate data plot3d file
# TODO: Add the docstring test

import unittest
import doctest
import numpy as np
from src.lptlib.io import GridIO, FlowIO


class TestIO(unittest.TestCase):
    def test_grid_io(self):

        # Test with the plate data
        grid = GridIO('../data/plate_data/plate.sp.x')
        # Print the doc string
        print(grid)
        grid.read_grid()

        # Simple assertion to test the import
        self.assertEqual(grid.grd.shape, (grid.ni, grid.nj, grid.nk, 3, 1))

        return

    def test_flow_io(self):

        # Test with the plate data
        flow = FlowIO('../data/plate_data/sol-0000010.q')
        # Print the doc string
        print(flow)
        flow.read_flow()

        # Simple assertion to test the import
        self.assertEqual(flow.q.shape, (flow.ni, flow.nj, flow.nk, 5, 1))

    def test_multi_block(self):
        """
        This tests the basic slicing functionality in the code
        Returns:

        """

        # Testing with an 8-block 2x2x2 grid
        grid = GridIO('../data/multi_block/cube/cube.mb.x')
        # Print the doc string
        print(grid)
        grid.read_grid()

        # Simple assertion to test the import
        self.assertEqual(grid.grd.shape, (grid.ni.max(), grid.nj.max(), grid.nk.max(), 3, grid.nb))

    def test_mb_plate(self):
        """
        This tests the padding functionality as well
        Returns:

        """

        # Testing with a multi-block plate grid
        grid = GridIO('../data/multi_block/plate/plate.mb.sp.x')
        # Print the doc string
        print(grid)
        grid.read_grid(data_type='f4')

        # Compute metrics from the test setup function
        test_grid = grid_metrics(grid)

        # Simple assertion to test the import
        self.assertEqual(grid.grd.shape, (grid.ni.max(), grid.nj.max(), grid.nk.max(), 3, grid.nb))
        self.assertEqual(np.all(grid.m1 == test_grid.m1), True)
        self.assertEqual(np.all(grid.m2 == test_grid.m2), True)
        self.assertEqual(np.all(grid.J == test_grid.J), True)

    def test_two_to_three(self):
        """
        Tests the conversion of data from 2d to 3d
        Returns: None

        """

        # Import the shock interaction case
        grid = GridIO('../data/shock_interaction/shock_interaction_coarse.x')
        grid.read_grid()
        grid.two_to_three()

        test_grid = GridIO('../data/shock_interaction/shock_interaction_coarse_3D.x')
        test_grid.read_grid()

    def test_read_formatted_txt(self):
        """
        Tests the code to read formatted text obtained from Tecplot
        Returns:

        """
        path = '../data/shock_interaction/fine/'
        grid = GridIO(path + 'fine_python.x')
        grid.read_grid()

        flow = FlowIO(path + '42500.txt')
        # Fill out required variables for the q-file
        flow.mach = 2.3
        flow.rey = 32033863.98
        flow.alpha = 0.0
        flow.time = 1.0
        flow.read_formatted_txt(grid=grid, data_type='f8')


def grid_metrics(grid):
    """
    Computes grid metrics traditionally
    Returns:
         m1, m2, J
    """

    grid.m1 = np.zeros((grid.ni.max(), grid.nj.max(), grid.nk.max(), 3, 3, grid.nb))
    grid.m2 = np.zeros((grid.ni.max(), grid.nj.max(), grid.nk.max(), 3, 3, grid.nb))
    grid.J = np.zeros((grid.ni.max(), grid.nj.max(), grid.nk.max(), grid.nb))

    # xi derivatives
    for b in range(grid.nb):
        for k in range(grid.nk[b]):
            for j in range(grid.nj[b]):
                grid.m1[0, j, k, 0, 0, b] = grid.grd[1, j, k, 0, b] - grid.grd[0, j, k, 0, b]
                grid.m1[0, j, k, 1, 0, b] = grid.grd[1, j, k, 1, b] - grid.grd[0, j, k, 1, b]
                grid.m1[0, j, k, 2, 0, b] = grid.grd[1, j, k, 2, b] - grid.grd[0, j, k, 2, b]
                for i in range(1, grid.ni[b] - 1):
                    grid.m1[i, j, k, 0, 0, b] = 0.5 * (grid.grd[i + 1, j, k, 0, b] - grid.grd[i - 1, j, k, 0, b])
                    grid.m1[i, j, k, 1, 0, b] = 0.5 * (grid.grd[i + 1, j, k, 1, b] - grid.grd[i - 1, j, k, 1, b])
                    grid.m1[i, j, k, 2, 0, b] = 0.5 * (grid.grd[i + 1, j, k, 2, b] - grid.grd[i - 1, j, k, 2, b])
                grid.m1[grid.ni[b] - 1, j, k, 0, 0, b] = grid.grd[grid.ni[b] - 1, j, k, 0, b] - grid.grd[
                    grid.ni[b] - 2, j, k, 0, b]
                grid.m1[grid.ni[b] - 1, j, k, 1, 0, b] = grid.grd[grid.ni[b] - 1, j, k, 1, b] - grid.grd[
                    grid.ni[b] - 2, j, k, 1, b]
                grid.m1[grid.ni[b] - 1, j, k, 2, 0, b] = grid.grd[grid.ni[b] - 1, j, k, 2, b] - grid.grd[
                    grid.ni[b] - 2, j, k, 2, b]
        print('xi derivatives computed')
        # eta derivatives
        for k in range(grid.nk[b]):
            for i in range(grid.ni[b]):
                grid.m1[i, 0, k, 0, 1, b] = grid.grd[i, 1, k, 0, b] - grid.grd[i, 0, k, 0, b]
                grid.m1[i, 0, k, 1, 1, b] = grid.grd[i, 1, k, 1, b] - grid.grd[i, 0, k, 1, b]
                grid.m1[i, 0, k, 2, 1, b] = grid.grd[i, 1, k, 2, b] - grid.grd[i, 0, k, 2, b]
                for j in range(1, grid.nj[b] - 1):
                    grid.m1[i, j, k, 0, 1, b] = 0.5 * (grid.grd[i, j + 1, k, 0, b] - grid.grd[i, j - 1, k, 0, b])
                    grid.m1[i, j, k, 1, 1, b] = 0.5 * (grid.grd[i, j + 1, k, 1, b] - grid.grd[i, j - 1, k, 1, b])
                    grid.m1[i, j, k, 2, 1, b] = 0.5 * (grid.grd[i, j + 1, k, 2, b] - grid.grd[i, j - 1, k, 2, b])
                grid.m1[i, grid.nj[b] - 1, k, 0, 1, b] = grid.grd[i, grid.nj[b] - 1, k, 0, b] - grid.grd[
                    i, grid.nj[b] - 2, k, 0, b]
                grid.m1[i, grid.nj[b] - 1, k, 1, 1, b] = grid.grd[i, grid.nj[b] - 1, k, 1, b] - grid.grd[
                    i, grid.nj[b] - 2, k, 1, b]
                grid.m1[i, grid.nj[b] - 1, k, 2, 1, b] = grid.grd[i, grid.nj[b] - 1, k, 2, b] - grid.grd[
                    i, grid.nj[b] - 2, k, 2, b]
        print('eta derivatives computed')
        # zeta derivatives
        for j in range(grid.nj[b]):
            for i in range(grid.ni[b]):
                grid.m1[i, j, 0, 0, 2, b] = grid.grd[i, j, 1, 0, b] - grid.grd[i, j, 0, 0, b]
                grid.m1[i, j, 0, 1, 2, b] = grid.grd[i, j, 1, 1, b] - grid.grd[i, j, 0, 1, b]
                grid.m1[i, j, 0, 2, 2, b] = grid.grd[i, j, 1, 2, b] - grid.grd[i, j, 0, 2, b]
                for k in range(1, grid.nk[b] - 1):
                    grid.m1[i, j, k, 0, 2, b] = 0.5 * (grid.grd[i, j, k + 1, 0, b] - grid.grd[i, j, k - 1, 0, b])
                    grid.m1[i, j, k, 1, 2, b] = 0.5 * (grid.grd[i, j, k + 1, 1, b] - grid.grd[i, j, k - 1, 1, b])
                    grid.m1[i, j, k, 2, 2, b] = 0.5 * (grid.grd[i, j, k + 1, 2, b] - grid.grd[i, j, k - 1, 2, b])
                grid.m1[i, j, grid.nk[b] - 1, 0, 2, b] = grid.grd[i, j, grid.nk[b] - 1, 0, b] - grid.grd[
                    i, j, grid.nk[b] - 2, 0, b]
                grid.m1[i, j, grid.nk[b] - 1, 1, 2, b] = grid.grd[i, j, grid.nk[b] - 1, 1, b] - grid.grd[
                    i, j, grid.nk[b] - 2, 1, b]
                grid.m1[i, j, grid.nk[b] - 1, 2, 2, b] = grid.grd[i, j, grid.nk[b] - 1, 2, b] - grid.grd[
                    i, j, grid.nk[b] - 2, 2, b]
        print('zeta derivatives')

        return grid


if __name__ == '__main__':
    unittest.main()
