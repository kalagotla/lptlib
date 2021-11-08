# This test is for the search algorithm using plate plot3d data

# TODO: Need to develop this test case for search algorithm
#  Convert the array data to p3d data. Then implement the search algorithm
#  Most of the code is copied in Notion under "code samples"

import unittest


class TestSearch(unittest.TestCase):
    def test_search(self):
        from src.io.plot3dio import GridIO
        from src.streamlines.search import Search

        # Read the grid data
        grid = GridIO('../data/plate_data/plate.sp.x')
        grid.read_grid()

        # Start test for search algorithm
        # Check if distance and neighbors are working fine
        idx = Search(grid, [8.5, 0.5, 0.01])
        idx.compute()
        self.assertEqual(idx.cell.shape, (8, 3))
        self.assertEqual(idx.info, None)
        # Check for out of domain point
        idx = Search(grid, [-10, 0.5, 0.01])
        idx.compute()
        self.assertEqual(idx.cell, None)
        self.assertEqual(idx.info,
                         'Given point is not in the domain. The cell attribute will return "None"\n')
        # Check for on the node case
        # Used a Node from grid numbering
        idx = Search(grid, grid.grd[0, 0, 0, :, 0])
        idx.compute()
        self.assertEqual(idx.info, 'Given point is a node in the domain with a tol of 1e-6.\n'
                                   'Interpolation will assign node properties for integration.\n'
                                   'Index of the node will be returned by cell attribute\n')
        # Check for on the boundary case
        # Point is created by averaging neighboring nodes
        idx = Search(grid, [sum(grid.grd[0:2, 0, 0, 0, 0]) / 2, grid.grd[0, 0, 0, 1, 0], grid.grd[0, 0, 0, 2, 0]])
        idx.compute()
        self.assertEqual(idx.cell.shape, (8, 3))
        self.assertEqual(idx.info, None)

    def test_multi_search(self):
        from src.io.plot3dio import GridIO
        from src.streamlines.search import Search

        # Read the grid data
        grid = GridIO('../data/multi_block/plate/plate.mb.x')
        grid.read_grid(data_type='f8')

        # Start test for search algorithm
        # Check if distance and neighbors are working fine
        idx = Search(grid, [8.5, 0.5, 0.01])
        idx.compute()
        self.assertEqual(idx.cell.shape, (8, 3))
        self.assertEqual(idx.info, None)
        # Check for out of domain point
        idx = Search(grid, [-10, 0.5, 0.01])
        idx.compute()
        self.assertEqual(idx.cell, None)
        self.assertEqual(idx.info,
                         'Given point is not in the domain. The cell attribute will return "None"\n')
        # Check for on the node case
        # Used a Node from grid numbering
        idx = Search(grid, grid.grd[0, 0, 0, :, 0])
        idx.compute()
        self.assertEqual(idx.info, 'Given point is a node in the domain with a tol of 1e-6.\n'
                                   'Interpolation will assign node properties for integration.\n'
                                   'Index of the node will be returned by cell attribute\n')
        # Check for on the boundary case
        # Point is created by averaging neighboring nodes
        idx = Search(grid, [sum(grid.grd[0:2, 0, 0, 0, 0]) / 2, grid.grd[0, 0, 0, 1, 0], grid.grd[0, 0, 0, 2, 0]])
        idx.compute()
        self.assertEqual(idx.cell.shape, (8, 3))
        self.assertEqual(idx.info, None)


if __name__ == '__main__':
    unittest.main()
