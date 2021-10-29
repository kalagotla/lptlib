# Test the data interpolation

import unittest


class TestInterpolation(unittest.TestCase):
    def test_interpolation(self):
        from src.io.plot3dio import GridIO, FlowIO
        from src.streamlines.search import Search
        from src.streamlines.interpolation import Interpolation

        # Read the grid data
        grid = GridIO('../data/plate_data/plate.sp.x')
        flow = FlowIO('../data/plate_data/sol-0000010.q')
        idx = Search(grid, [8.5, 0.5, 0.01])
        point_data = Interpolation(grid, flow, idx)

        grid.read_grid()
        flow.read_flow()
        idx.compute()
        point_data.compute()

        print(point_data.q)


if __name__ == '__main__':
    unittest.main()
