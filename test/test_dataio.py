import unittest


class TestDataIO(unittest.TestCase):
    def test_dataio(self):
        from lptlib.io import DataIO
        from lptlib.io import GridIO, FlowIO
        from testdata import require_data
        # grid object
        grid = GridIO(require_data('shocks', 'shock_test.sb.sp.x'))
        grid.read_grid()
        grid.compute_metrics()

        # flow object
        flow = FlowIO(require_data('shocks', 'shock_test.sb.sp.q'))
        flow.read_flow()

        # data module test
        location = require_data('shocks', 'particle_data', '281nm_time_step_adaptive', 'old_data')
        read_file = require_data('shocks', 'particle_data', '281nm_time_step_adaptive', 'old_data',
                                 'combined_file.npy')
        data = DataIO(grid, flow, location=location, read_file=read_file)
        data.percent_data = 0.1
        # Increased refinement for better resolution
        data.x_refinement = 500
        data.y_refinement = 400
        data.compute()


if __name__ == '__main__':
    unittest.main()
