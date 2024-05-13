# Test the data interpolation

import unittest


class TestInterpolation(unittest.TestCase):
    def test_interpolation(self):
        from src.lptlib.io import GridIO, FlowIO
        from src.lptlib.streamlines import Search, Interpolation

        # Read the grid data
        grid = GridIO('../data/plate_data/plate.sp.x')
        flow = FlowIO('../data/plate_data/sol-0000010.q')
        idx = Search(grid, [8.5, 0.5, 0.01])
        point_data = Interpolation(flow, idx)

        grid.read_grid()
        grid.compute_metrics()
        flow.read_flow()
        idx.compute(method='c-space')
        point_data.compute(method='rbf-c-space')

        self.assertEqual(
            sum(abs(point_data.q.reshape(5)
                    - [9.99767442e-01, 1.02352604e-01, -5.38538464e-06, 6.40554753e-09, 1.79096631e+00]))
            <= 1e-6,
            True)


        # Test if the point is a node interpolation
        from src.lptlib.test_cases import ObliqueShock, ObliqueShockData

        # Create oblique shock
        os = ObliqueShock()
        os.mach = 2
        os.deflection = 9
        os.compute()

        # Create grid and flow files
        osd = ObliqueShockData()
        osd.nx_max = 15e-3
        osd.ny_max = 15e-3
        osd.nz_max = 1e-4
        osd.inlet_temperature = 152.778
        # osd.inlet_pressure = 55535.59
        osd.inlet_density = 1.2663
        osd.xpoints = 100
        osd.ypoints = 100
        osd.zpoints = 5
        osd.oblique_shock = os
        osd.shock_strength = 'weak'
        osd.create_grid()
        osd.create_flow()

        # search and interpolation
        grid = osd.grid
        flow = osd.flow
        idx = Search(grid, [7.61402532e-03, 1.42415589e-02, 5.00000000e-05])
        interp = Interpolation(flow, idx)
        idx.compute(method='distance')
        interp.compute(method='p-space')
        self.assertEqual(interp.q.shape == (1, 1, 1, 5, 1), True)


if __name__ == '__main__':
    unittest.main()
