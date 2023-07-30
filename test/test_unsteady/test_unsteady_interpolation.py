import unittest


class TestUnsteadyInterpolation(unittest.TestCase):
    def test_unsteady_interpolation(self):
        from src.io.plot3dio import GridIO, FlowIO
        from src.streamlines.search import Search
        from src.streamlines.interpolation import Interpolation
        import numpy as np

        grid_file, flow_file = "cylinder.sp.x", "sol-0000010.q"
        grid = GridIO(grid_file)
        grid.read_grid(data_type="f4")
        grid.compute_metrics()
        flow = FlowIO(flow_file)
        flow.read_unsteady_flow(data_type="f4")
        idx = Search(grid, [1, 2.5, 0.5])
        idx.method = "p-space"
        idx.compute()
        for i in range(len(flow.unsteady_flow)):
            interp = Interpolation(flow.unsteady_flow[i], idx)
            if i == 0:
                interp.flow_old = None
            else:
                interp.flow_old = flow.unsteady_flow[i-1]
            test_time_step = 1e-6
            interp.time.append(test_time_step)
            interp.compute(method='unsteady-rbf-p-space')


if __name__ == "__main__":
    unittest.main()
