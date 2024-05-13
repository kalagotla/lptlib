# This test is for the search algorithm using plate plot3d data

import unittest
from parameterized import parameterized
from src.lptlib.function import Timer


# This setup only works with the plate data
# Need to add assertions to parametrized and change them to variables in the function
# to be able to implement
class TestSearch(unittest.TestCase):
    # sb: single block; mb: multi block
    # sp: single precision; dp: double precision
    # Add more cases as a tuple below
    @parameterized.expand([
        # ('sb_sp_distance', '../data/plate_data/plate.sp.x', 'f4', 'distance'),
        # ('sb_sp_block_distance', '../data/plate_data/plate.sp.x', 'f4', 'block_distance'),
        # ('mb_dp_distance', '../data/multi_block/plate/plate.mb.dp.x', 'f8', 'distance'),
        # ('mb_dp_block_distance', '../data/multi_block/plate/plate.mb.dp.x', 'f8', 'block_distance'),
        # ('mb_dp_c_space', '../data/multi_block/plate/plate.mb.dp.x', 'f8', 'c-space'),
        ('mb_dp_p_space', '../data/multi_block/plate/plate.mb.dp.x', 'f8', 'p-space')

    ])
    @Timer()
    def test_search(self, name, filename='../data/plate_data/plate.sp.x', data_type='f4', method='binary'):
        from src import GridIO
        from src import Search

        # Read the grid data
        grid = GridIO(filename)
        grid.read_grid(data_type=data_type)
        grid.compute_metrics()
        flow = None

        # Start test for search algorithm
        # Check if distance and neighbors are working fine
        idx = Search(grid, [8.5, 0.5, 0.01])
        idx.compute(method=method)
        idx_test = Search(grid, [8.5, 0.5, 0.01])
        idx_test.compute(method='distance')
        self.assertEqual(idx.cell.shape, (8, 3))
        self.assertEqual(idx.info, None)
        self.assertEqual((idx.cell == idx_test.cell).all(), True)
        # Check for out of domain point
        idx = Search(grid, [-10, 0.5, 0.01])
        idx.compute(method=method)
        self.assertEqual(idx.cell, None)
        self.assertEqual(idx.info,
                         'Given point is not in the domain. The cell attribute will return "None"'
                         ' in search algorithm\n')
        # Check for on the node case
        # Used a Node from grid numbering
        idx = Search(grid, grid.grd[0, 0, 0, :, 0])
        idx.compute(method=method)
        self.assertEqual(idx.info, 'Given point is a node in the domain with a tol of 1e-12.\n'
                                   'Interpolation will assign node properties for integration.\n'
                                   'Index of the node will be returned by cell attribute\n')
        # Check for on the boundary case
        # Point is created by averaging neighboring nodes
        test_point = [sum(grid.grd[0:2, 0, 0, 0, 0]) / 2, grid.grd[0, 0, 0, 1, 0], grid.grd[0, 0, 0, 2, 0]]
        idx = Search(grid, test_point)
        idx.compute(method=method)
        idx_test = Search(grid, test_point)
        idx_test.compute(method='distance')
        self.assertEqual(idx.cell.shape, (8, 3))
        self.assertEqual(idx.info, None)
        self.assertEqual((idx.cell == idx_test.cell).all(), True)


if __name__ == '__main__':
    unittest.main()
