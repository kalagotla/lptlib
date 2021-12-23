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
        point_data = Interpolation(flow, idx)

        grid.read_grid()
        flow.read_flow()
        idx.compute()
        point_data.compute()

        self.assertEqual(
            sum(abs(point_data.q.reshape(5)
                    - [9.99767442e-01, 1.02352604e-01, -5.38538464e-06, 6.40554753e-09, 1.79096631e+00]))
            <= 1e-6,
            True)


if __name__ == '__main__':
    unittest.main()
