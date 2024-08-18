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

    def sharp_nozzle(self):

        # Run the model
        path = '../../data/dan_data/jet_flowfield/my_work/sharp_nozzle/'
        grid_file, flow_file = path + 'test.x', path + 'npr_4p0_flowdata.txt'
        grid = GridIO(grid_file)
        grid.read_grid(data_type='f4')
        grid.compute_metrics()
        # read flow data from sharp_nozzle folder
        flow = FlowIO(flow_file)
        flow.mach = 1.56
        flow.rey = 2.7e6
        flow.alpha = 0.0
        flow.time = 1.0
        flow.read_formatted_txt(grid=grid, data_type='f4')

        return grid, flow

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

        osd.grid, osd.flow = self.sharp_nozzle()

        # data module test
        # data = DataIO(grid, flow, location='../data/shocks/particle_data/multi_process_test/')
        # path = '../../data/shocks/new_start/williams_data/constant_particle_specs/'
        path = '../../data/dan_data/jet_flowfield/my_work/sharp_nozzle_new/dp_1e-06/'
        data = DataIO(osd.grid, osd.flow, location=path, read_file=path + 'combined_file.npy')
        data.percent_data = 1
        # Increased refinement for better resolution
        data.x_refinement = 500
        data.y_refinement = 400
        data.compute()

        return

    def test_dataio_mpi(self):
        # Run the test_dataio_mpi.py script
        command = ['mpiexec', '-np', '8', sys.executable, 'test_dataio_mpi.py', '--mpi']
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
        # path = '../../data/shocks/new_start/williams_data/constant_particle_specs/dataio_old_published/'
        # path = ('/Users/kal/Library/CloudStorage/OneDrive-UniversityofCincinnati/Desktop/University of Cincinnati/'
        #         'DoctoralWork/Codes/hpc_data/williams_data/constant_particle_specs/dataio/')
        # path = '../../data/shocks/new_start/williams_data/constant_particle_specs/dataio/'
        path = '../../data/dan_data/jet_flowfield/my_work/sharp_nozzle_new/dp_1e-06/dataio/'
        # load the old data
        p_data = np.load(path + 'new_p_data.npy')
        # load the flow data
        grid = GridIO(path + 'mgrd_to_p3d.x')
        flow = FlowIO(path + 'mgrd_to_p3d_particle.q')
        grid.read_grid()
        grid.compute_metrics()
        flow.read_flow()
        # plot the data
        fig, ax = plt.subplots()
        contour = ax.contour(grid.grd[..., 1, 0, 0], grid.grd[..., 1, 1, 0],
                             flow.q[..., 1, 1, 0] / flow.q[..., 1, 0, 0], 100, cmap='jet')
        # colorbar
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Velocity')
        # inline numbers
        ax.clabel(contour, inline=True, fontsize=12)
        # color by velocity
        # ax.scatter(p_data[:, 0], p_data[:, 1], c=p_data[:, 3], s=1, cmap='jet')
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

