# This is the main script. Run this and follow the steps
# TODO: Use this to guide a user
#  Ask for the filename, what to do? like get streamline data
#  or compute some properties etc...


def main(grid_file, flow_file, point):
    from src.io.plot3dio import GridIO
    from src.io.plot3dio import FlowIO
    from src.function.metrics import GridMetrics
    from src.streamlines.search import Search
    from src.streamlines.interpolation import Interpolation
    from src.streamlines.integration import Integration

    # Read-in the data and compute grid metrics
    grid = GridIO(grid_file)
    flow = FlowIO(flow_file)

    # Read in the grid and flow data
    grid.read_grid()
    flow.read_flow()

    streamline = [point]
    while True:
        idx = Search(grid, point)
        interp = Interpolation(grid, flow, idx)
        intg = Integration(grid, flow, idx, interp)
        idx.compute()
        interp.compute()
        new_point = intg.compute(time_step=1e-3)
        if new_point is None:
            print('Integration complete!')
            return
        streamline.append(new_point)
        point = new_point
        print(len(streamline))

    # Test for grid metrics
    # gm = GridMetrics(grid)
    # gm.compute()
    # print('Shape of grid metrics: ', gm.m1.shape)
    # print('Shape of grid: ', grid.grd.shape)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('data/plate_data/plate.sp.x', 'data/plate_data/sol-0000010.q', [8.5, 0.5, 0.01])
