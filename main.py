# This is the main script. Run this and follow the steps
# TODO: Use this to guide a user
#  Ask for the filename, what to do? like get streamline data
#  or compute some properties etc...


def main(grid_file, flow_file):
    from src.io.plot3dio import GridIO
    from src.io.plot3dio import FlowIO
    from src.function.metrics import GridMetrics
    from src.streamlines.search import Search

    # Read-in the data and compute grid metrics
    grid = GridIO(grid_file)
    flow = FlowIO(flow_file)
    # Read in the grid and flow data
    grid.read_grid()
    flow.read_flow()

    # Test the search algorithm using plate grid
    if True:
        print('Testing all the cases for node/cell search')

        def test_search(point):
            idx = Search(grid, point)
            idx.compute()
            print(f'Index of the node closest to given point: {idx.index}')
            print(f'Nodes of the cell in which the point resides-in: \n{idx.cell}')
            return

        # Check if distance and neighbors are working fine
        print('Checking if the algorithm works as expected...')
        test_search([8.5, 0.5, 0.01])
        # Check for out of domain point
        print('Checking if the out of domain case is working...')
        test_search([-10, 0.5, 0.01])
        # Check for on the node case
        print('Checking if "on the node" case is working...')
        test_search(grid.grd[0, 0, 0, :])
        # Check for on the boundary case
        print('Checking if "on the boundary" case is working')
        test_search([sum(grid.grd[0:2, 0, 0, 0])/2, grid.grd[0, 0, 0, 1], grid.grd[0, 0, 0, 2]])

    # Test for grid metrics
    gm = GridMetrics(grid)
    gm.compute()
    print('Shape of grid metrics: ', gm.m1.shape)
    print('Shape of grid: ', grid.grd.shape)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('data/plate_data/plate.sp.x', 'data/plate_data/sol-0000010.q')
