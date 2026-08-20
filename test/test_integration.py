import unittest
from parameterized import parameterized
from lptlib.function import Timer


class TestIntegration(unittest.TestCase):
    @parameterized.expand([
        ('sb_sp_p_space', 'plate_data/plate.sp.x', 'plate_data/sol-0000010.q',
         'f4', [8.5, 0.5, 0.01], 'block_distance', 'p-space', 'p-space'),
        ('sb_sp_c_space', 'plate_data/plate.sp.x', 'plate_data/sol-0000010.q',
         'f4', [8.5, 0.5, 0.01], 'c-space', 'c-space', 'c-space'),
        ('sb_sp_p_space', 'plate_data/plate.sp.x', 'plate_data/sol-0000010.q',
         'f4', [8.5, 0.5, 0.01], 'block_distance', 'p-space', 'RK4'),
        ('sb_sp_c_space', 'plate_data/plate.sp.x', 'plate_data/sol-0000010.q',
         'f4', [8.5, 0.5, 0.01], 'c-space', 'c-space', 'cRK4'),
        ('mb_sp_p_space', 'multi_block/plate/plate.mb.sp.x', 'multi_block/plate/plate.mb.sp.q',
         'f4', [8.5, 0.5, 0.01], 'block_distance', 'p-space', 'RK4'),
        ('mb_sp_c_space', 'multi_block/plate/plate.mb.sp.x', 'multi_block/plate/plate.mb.sp.q',
         'f4', [8.5, 0.5, 0.01], 'c-space', 'c-space', 'cRK4')

    ])
    @Timer()
    def test_integration(self, name, gridfile='plate_data/plate.sp.x',
                         flowfile='plate_data/sol-0000010.q', data_type='f4', point=None,
                         search_method='block_distance', interpolation_method='p-space', integration_method='RK4'):

        from lptlib import GridIO, FlowIO
        from lptlib import Search
        from lptlib import Interpolation
        from lptlib import Integration
        from testdata import require_data

        if point is None:
            point = [8.5, 0.5, 0.01]

        gridfile = require_data(gridfile)
        flowfile = require_data(flowfile)

        # Read the grid data
        grid = GridIO(gridfile)
        grid.read_grid(data_type=data_type)
        grid.compute_metrics()

        # Read the flow data
        flow = FlowIO(flowfile)
        flow.read_flow(data_type=data_type)

        # Search for the given point
        idx = Search(grid, point)
        idx.compute(method=search_method)

        # Do Interpolation
        interp = Interpolation(flow, idx)
        interp.compute(method=interpolation_method)

        # Do Integration
        intg = Integration(interp)
        new_point = intg.compute(method=integration_method, time_step=1e-2)


if __name__ == '__main__':
    unittest.main()
