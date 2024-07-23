# unittest for dataio mpi

import unittest
import sys
import subprocess
from src.lptlib import DataIO
from src.lptlib import GridIO, FlowIO
from src.lptlib import ObliqueShock, ObliqueShockData
from src.lptlib import StochasticModel, Particle, SpawnLocations
import matplotlib.pyplot as plt
import numpy as np


class TestDataIOMPI(unittest.TestCase):
    def test_dataio(self):
        # Create oblique shock
        os1 = ObliqueShock()
        os1.mach = 7.6
        os1.deflection = 20
        os1.compute()

        # Create grid and flow files
        osd = ObliqueShockData()
        osd.oblique_shock = os1
        osd.nx_max = 100e-3
        osd.ny_max = 500e-3
        osd.nz_max = 1e-4
        osd.inlet_temperature = 48.20
        osd.inlet_density = 0.07747
        osd.xpoints = 200
        osd.ypoints = 500
        osd.zpoints = 5
        osd.shock_strength = 'weak'
        osd.create_grid()
        osd.create_flow()

        # data module test
        # data = DataIO(grid, flow, location='../data/shocks/particle_data/multi_process_test/')
        data = DataIO(osd.grid, osd.flow, location='../../data/shocks/new_start/williams_data/constant_particle_specs/',
                      read_file='../../data/shocks/new_start/williams_data/constant_particle_specs/combined_file.npy')
        data.percent_data = 10
        # Increased refinement for better resolution
        data.x_refinement = 500
        data.y_refinement = 400
        data.compute()

        return

    def test_dataio_mpi(self):
        # Run the test_dataio_mpi.py script
        command = ['mpiexec', '-np', '30', sys.executable, 'test_dataio_mpi.py', '--mpi']
        result = subprocess.run(command, capture_output=False, text=True)

        if result.returncode == 0:
            print("MPI script ran successfully.")
        else:
            print("MPI script failed.")
            print("Error:")
            print(result.stderr)
        self.assertEqual(result.returncode, 0, "MPI script failed with errors.")

        return

    def test_debug_plots(self):
        # path
        path = '../../data/shocks/new_start/williams_data/constant_particle_specs/dataio_old_published/'
        # load the old data
        p_data = np.load(path + 'new_p_data.npy')
        # load the flow data
        grid = GridIO(path + 'mgrd_to_p3d.x')
        flow = FlowIO(path + 'mgrd_to_p3d_fluid.q')
        grid.read_grid()
        grid.compute_metrics()
        flow.read_flow()
        # plot the data
        fig, ax = plt.subplots()
        contour = ax.contour(grid.grd[..., 1, 0, 0], grid.grd[..., 1, 1, 0],
                             flow.q[..., 1, 1, 0] / flow.q[..., 1, 0, 0], 100, cmap='jet')
        # # colorbar
        # cbar = fig.colorbar(contour, ax=ax)
        # cbar.set_label('Velocity')
        # # inline numbers
        # ax.clabel(contour, inline=True, fontsize=12)
        # color by velocity
        ax.scatter(p_data[:, 0], p_data[:, 1], c=p_data[:, 3], s=1, cmap='jet')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
        return


if __name__ == '__main__':
    # run the test_dataio_mpi() function to test the MPI script
    if '--mpi' in sys.argv:
        # We are running under MPI
        test = TestDataIOMPI()
        test.test_dataio()
    else:
        # Regular unit test execution
        unittest.main()

