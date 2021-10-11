# This is the main script. Run this and follow the steps


def main(grid_file, flow_file):
    from src.grid.io import GridIO
    from src.flow.io import FlowIO
    from src.grid.metrics import GridMetrics

    # Read-in the data and compute grid metrics
    grid = GridIO(grid_file)
    flow = FlowIO(flow_file)
    gm = GridMetrics(grid)

    grid.read_grid()
    flow.read_flow()
    gm.compute()

    # Can plot or calculate more variables etc...
    print(gm.m1.shape)  # Testing the main program


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('data/plate_data/plate.sp.x', 'data/plate_data/sol-0000010.q')
