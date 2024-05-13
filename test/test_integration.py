import unittest
from parameterized import parameterized
from src.lptlib.function import Timer


class TestIntegration(unittest.TestCase):
    @parameterized.expand([
        ('sb_sp_p_space', '../data/plate_data/plate.sp.x', '../data/plate_data/sol-0000010.q',
         'f4', [8.5, 0.5, 0.01], 'block_distance', 'p-space', 'p-space'),
        ('sb_sp_c_space', '../data/plate_data/plate.sp.x', '../data/plate_data/sol-0000010.q',
         'f4', [8.5, 0.5, 0.01], 'c-space', 'c-space', 'c-space'),
        ('sb_sp_p_space', '../data/plate_data/plate.sp.x', '../data/plate_data/sol-0000010.q',
         'f4', [8.5, 0.5, 0.01], 'block_distance', 'p-space', 'RK4'),
        ('sb_sp_c_space', '../data/plate_data/plate.sp.x', '../data/plate_data/sol-0000010.q',
         'f4', [8.5, 0.5, 0.01], 'c-space', 'c-space', 'cRK4'),
        ('mb_sp_p_space', '../data/multi_block/plate/plate.mb.sp.x', '../data/multi_block/plate/plate.mb.sp.q',
         'f4', [8.5, 0.5, 0.01], 'block_distance', 'p-space', 'RK4'),
        ('mb_sp_c_space', '../data/multi_block/plate/plate.mb.sp.x', '../data/multi_block/plate/plate.mb.sp.q',
         'f4', [8.5, 0.5, 0.01], 'c-space', 'c-space', 'cRK4')

    ])
    @Timer()
    def test_integration(self, name, gridfile='../data/plate_data/plate.sp.x',
                         flowfile='../data/plate_data/sol-0000010.q', data_type='f4', point=None,
                         search_method='block_distance', interpolation_method='p-space', integration_method='RK4'):

        from src import GridIO, FlowIO
        from src import Search
        from src import Interpolation
        from src import Integration

        if point is None:
            point = [8.5, 0.5, 0.01]

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
