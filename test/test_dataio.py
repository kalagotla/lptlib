import unittest


class TestDataIO(unittest.TestCase):
    def test_dataio(self):
        from src.io.dataio import DataIO
        from src.io.plot3dio import GridIO, FlowIO
        # grid object
        grid = GridIO('../data/shocks/shock_test.sb.sp.x')
        grid.read_grid()
        grid.compute_metrics()

        # flow object
        flow = FlowIO('../data/shocks/shock_test.sb.sp.q')
        flow.read_flow()

        # data module test
        data = DataIO(grid, flow, read_file='../data/shocks/adaptive-ppath-p-space5e-07.npy')
        data.compute()


if __name__ == '__main__':
    unittest.main()
