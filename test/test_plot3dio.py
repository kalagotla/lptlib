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
        self.assertEqual(grid.grd.shape, (grid.ni, grid.nj, grid.nk, 3))

        return

    def test_flow_io(self):
        from src.io.plot3dio import FlowIO

        # Test with the plate data
        flow = FlowIO('../data/plate_data/sol-0000010.q')
        # Print the doc string
        print(flow)
        flow.read_flow()

        # Simple assertion to test the import
        self.assertEqual(flow.q.shape, (flow.ni, flow.nj, flow.nk, 5))


if __name__ == '__main__':
    unittest.main()
