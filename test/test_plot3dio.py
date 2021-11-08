# This test case is for plate data plot3d file
# TODO: Add the docstring test

import unittest
import doctest


class TestIO(unittest.TestCase):
    def test_grid_io(self):
        from src.io.plot3dio import GridIO

        # Test with the plate data
        grid = GridIO('../data/plate_data/plate.sp.x')
        # Print the doc string
        print(grid)
        grid.read_grid()

        # Simple assertion to test the import
        self.assertEqual(grid.grd.shape, (grid.ni, grid.nj, grid.nk, 3, 1))

        return

    def test_flow_io(self):
        from src.io.plot3dio import FlowIO

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
        from src.io.plot3dio import GridIO

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
        from src.io.plot3dio import GridIO

        # Testing with a multi-block plate grid
        grid = GridIO('../data/multi_block/plate/plate.mb.x')
        # Print the doc string
        print(grid)
        grid.read_grid(data_type='f8')

        # Simple assertion to test the import
        self.assertEqual(grid.grd.shape, (grid.ni.max(), grid.nj.max(), grid.nk.max(), 3, grid.nb))


if __name__ == '__main__':
    unittest.main()
