import unittest
import sys
import subprocess
from src.lptlib import StochasticModel, Particle, SpawnLocations, ObliqueShock, ObliqueShockData
from mpi4py import MPI


class TestMPI(unittest.TestCase):
    @staticmethod
    def create_oblique_shock():
        os1 = ObliqueShock()
        os1.mach = 7.6
        os1.deflection = 20
        os1.compute()

        osd = ObliqueShockData()
        osd.oblique_shock = os1
        osd.nx_max = 1000e-3
        osd.ny_max = 2000e-3
        osd.nz_max = 1e-4
        osd.inlet_temperature = 48.20
        osd.inlet_density = 0.07747
        osd.xpoints = 1000
        osd.ypoints = 2000
        osd.zpoints = 3
        osd.shock_strength = 'weak'
        osd.create_grid()
        osd.create_flow()
        return osd

    @staticmethod
    def create_particle():
        p = Particle()
        p.min_dia = 1000e-9
        p.max_dia = 3000e-9
        p.mean_dia = 1940e-9
        p.std_dia = 25e-9
        p.density = 950
        p.n_concentration = 2
        p.distribution = 'gaussian'
        p.distribution_parameter = -7
        p.compute_distribution()
        return p

    @staticmethod
    def create_spawn_locations(p):
        l = SpawnLocations(p)
        l.x_min = -50e-3
        l.z_min = 5e-5
        l.y_min, l.y_max = 1e-4, 1e-4
        l.compute()
        l.compute()
        return l

    def lpt_code(self):
        osd = self.create_oblique_shock()
        p = self.create_particle()
        l = self.create_spawn_locations(p)
        grid = osd.grid
        flow = osd.flow
        sm = StochasticModel(p, l, grid=grid, flow=flow)
        sm.method = 'adaptive-ppath'
        sm.search = 'p-space'
        sm.time_step = 1e-10
        sm.max_time_step = 10
        sm.interpolation = 'simple_oblique_shock'
        sm.drag_model = 'loth'
        sm.mpi_run()

    def test_mpi(self):
        command = ['mpiexec', '-np', '2', sys.executable, 'test_mpi.py', '--mpi']
        result = subprocess.run(command, capture_output=False, text=True)

        if result.returncode == 0:
            print("MPI script ran successfully.")
        else:
            print("MPI script failed.")
            print("Error:")
            print(result.stderr)
        self.assertEqual(result.returncode, 0, "MPI script failed with errors.")


if __name__ == '__main__':
    if '--mpi' in sys.argv:
        # We are running under MPI
        test = TestMPI()
        test.lpt_code()
    else:
        # Regular unit test execution
        unittest.main()
