# unittest for dataio mpi

import unittest
import sys
import subprocess
from src.lptlib.io import DataIO
from src.lptlib.io import GridIO, FlowIO
import matplotlib.pyplot as plt
import numpy as np


class TestDataIOMPI(unittest.TestCase):
    def test_dataio(self):
        # grid object
        grid = GridIO('../data/shocks/shock_test.sb.sp.x')
        grid.read_grid()
        grid.compute_metrics()

        # flow object
        flow = FlowIO('../data/shocks/shock_test.sb.sp.q')
        flow.read_flow()

        # data module test
        # data = DataIO(grid, flow, location='../data/shocks/particle_data/multi_process_test/')
        data = DataIO(grid, flow, location='../data/shocks/particle_data/281nm_time_step_adaptive/old_data/',
                      read_file='../data/shocks/particle_data/281nm_time_step_adaptive/old_data/combined_file.npy')
        data.percent_data = 0.1
        # Increased refinement for better resolution
        data.x_refinement = 500
        data.y_refinement = 400
        data.compute()

        return

    def test_dataio_mpi(self):
        # Run the test_dataio_mpi.py script
        command = ['mpiexec', '-np', '8', sys.executable, 'test_mpi/test_dataio_mpi.py', '--mpi']
        result = subprocess.run(command, capture_output=False, text=True)

        if result.returncode == 0:
            print("MPI script ran successfully.")
        else:
            print("MPI script failed.")
            print("Error:")
            print(result.stderr)
        self.assertEqual(result.returncode, 0, "MPI script failed with errors.")

        return

    def test_debug_dataio_mpi(self):

        # path
        path = '../data/shocks/particle_data/281nm_time_step_adaptive/old_data/dataio/'
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
        ax.contour(grid.grd[..., 1, 0, 0], grid.grd[..., 1, 1, 0], flow.q[..., 1, 1, 0] / flow.q[..., 1, 0, 0], 100)
        ax.plot(p_data[:, 0], p_data[:, 1], 'r.')
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

