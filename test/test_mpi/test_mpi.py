import unittest
import sys
import subprocess
from lptlib import StochasticModel, Particle, SpawnLocations, ObliqueShock, ObliqueShockData


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
        locations = SpawnLocations(p)
        locations.x_min = -50e-3
        locations.z_min = 5e-5
        locations.y_min, locations.y_max = 1e-4, 1e-4
        locations.compute()
        locations.compute()
        return locations

    def lpt_code(self):
        osd = self.create_oblique_shock()
        p = self.create_particle()
        locations = self.create_spawn_locations(p)
        grid = osd.grid
        flow = osd.flow
        sm = StochasticModel(p, locations, grid=grid, flow=flow)
        sm.method = 'adaptive-ppath'
        sm.search = 'p-space'
        sm.time_step = 1e-10
        sm.max_time_step = 10
        sm.interpolation = 'simple_oblique_shock'
        sm.drag_model = 'loth'
        sm.mpi_run()

    def test_mpi(self):
        import os
        import shutil
        # This launches a nested MPI job that builds a large synthetic flow field
        # on every rank, so it is a heavy integration test. It is skipped by default
        # and can be enabled by setting LPTLIB_RUN_MPI=1 in an environment with enough
        # memory and an MPI launcher.
        if not os.environ.get('LPTLIB_RUN_MPI'):
            self.skipTest("MPI integration test disabled: set the environment variable "
                          "LPTLIB_RUN_MPI=1 to enable it")
        if shutil.which('mpiexec') is None:
            self.skipTest("MPI integration test requires an mpiexec launcher on PATH "
                          "(LPTLIB_RUN_MPI=1 is set but mpiexec was not found)")

        script = os.path.abspath(__file__)
        command = ['mpiexec', '-np', '2', sys.executable, script, '--mpi']
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
