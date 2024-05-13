import unittest
import numpy as np


class TestObliqueShockResponse(unittest.TestCase):
    def test_oblique_shock(self):
        from src.lptlib.test_cases import ObliqueShock
        os = ObliqueShock()
        os.mach = 2.3
        os.deflection = 10
        os.compute()
        self.assertAlmostEqual(os.shock_angle.all(), np.array([34.32642717, 85.02615188]).all(), places=4)

    def test_oblique_shock_relation(self):
        from src.lptlib.test_cases import ObliqueShock
        from src.lptlib.test_cases import ObliqueShockData
        os = ObliqueShock()
        os.mach = 2.3
        os.deflection = 10
        os.compute()

        # Create grid and flow files
        osd = ObliqueShockData()
        osd.nx_max = 10
        osd.ny_max = 10
        osd.nz_max = 10
        osd.inlet_density = 1.273
        osd.inlet_temperature = 300
        osd.inlet_pressure = 101325
        osd.xpoints = 100  # creates 100 * 2 points
        osd.ypoints = 100  # creates 100 points
        osd.zpoints = 8  # creates 8 points
        osd.oblique_shock = os
        osd.create_grid()
        osd.create_flow()


if __name__ == '__main__':
    unittest.main()
