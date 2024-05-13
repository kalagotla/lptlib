import unittest


class TestDataIO(unittest.TestCase):
    def test_dataio(self):
        from src.lptlib.io import DataIO
        from src.lptlib.io import GridIO, FlowIO
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


if __name__ == '__main__':
    unittest.main()
