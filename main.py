# This is the main script. Run this and follow the steps
# TODO: Use this to guide a user
#  Ask for the filename, what to do? like get streamline data
#  or compute some properties etc...

from src.function.timer import Timer
import numpy as np


@Timer(text="Elapsed time for main program: {:.4f} seconds")
def main(grid_file, flow_file, point):
    from src.io.plot3dio import GridIO
    from src.io.plot3dio import FlowIO
    from src.streamlines.search import Search
    from src.streamlines.interpolation import Interpolation
    from src.streamlines.integration import Integration

    # Read-in the data and compute grid metrics
    grid = GridIO(grid_file)
    flow = FlowIO(flow_file)

    # Read in the grid and flow data
    grid.read_grid()
    flow.read_flow()
    grid.compute_metrics()

    streamline = [point]
    # while True:
    #     with Timer(text="Elapsed time for loop number " + str(len(streamline)) + ": {:.8f}"):
    #         idx = Search(grid, metrics, point)
    #         interp = Interpolation(grid, flow, metrics, idx)
    #         intg = Integration(grid, flow, idx, interp)
    #         t = Timer(text="Elapsed time for search number " + str(len(streamline)) + ": {:.8f} seconds")
    #         t.start()
    #         idx.compute(method='block_distance')
    #         t.stop()
    #         interp.compute()
    #         new_point = intg.compute(time_step=1e-1)
    #         if new_point is None:
    #             print('Integration complete!')
    #             break
    #         streamline.append(new_point)
    #         point = new_point

    streamline = [point]
    idx = Search(grid, point)
    idx.compute(method='c-space')
    while True:
        with Timer(text="Elapsed time for loop number " + str(len(streamline)) + ": {:.8f}"):
            interp = Interpolation(flow, idx)
            interp.compute(method='c-space')
            intg = Integration(interp)
            new_point = intg.compute(method='c-space', time_step=1e-2)
            if new_point is None:
                print('Integration complete!')
                break
            save_point = idx.c2p(new_point)
            streamline.append(save_point)
            idx.point = new_point

    return np.array(streamline)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sl = main('data/plate_data/plate.sp.x', 'data/plate_data/sol-0000010.q', [8.5, 0.5, 0.01])
    # main('data/multi_block/test/test.mb.sp.x', 'data/multi_block/test/test.mb.sp.q', [-0.5, -0.5, -0.5])
    sl1 = main('data/multi_block/plate/plate.mb.sp.x', 'data/multi_block/plate/plate.mb.sp.q', [8.5, 0.5, 0.01])

    x, y, z = sl[:, 0], sl[:, 1], sl[:, 2]
    x1, y1, z1 = sl1[:, 0], sl1[:, 1], sl1[:, 2]

    import matplotlib.pyplot as plt
    import random

    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'r')
    ax.plot3D(x1, y1, z1, 'b')
    # ax.scatter3D(x, y, z)
    # ax.scatter3D(x1, y1, z1)
    plt.show()

