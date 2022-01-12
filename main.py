# This is the main script. Run this and follow the steps
# TODO: Use this to guide a user
#  Ask for the filename, what to do? like get streamline data
#  or compute some properties etc...

from src.function.timer import Timer
import numpy as np


@Timer(text="Elapsed time for main program: {:.4f} seconds")
def main(grid_file, flow_file, point, method='c-space'):
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
    if method == 'p-space':
        while True:
            with Timer(text="Elapsed time for loop number " + str(len(streamline)) + ": {:.8f}"):
                idx = Search(grid, point)
                interp = Interpolation(flow, idx)
                intg = Integration(interp)
                t = Timer(text="Elapsed time for search number " + str(len(streamline)) + ": {:.8f} seconds")
                t.start()
                idx.compute(method='block_distance')
                t.stop()
                interp.compute()
                new_point = intg.compute(method='pRK4', time_step=1e-1)
                if new_point is None:
                    print('Integration complete!')
                    break
                streamline.append(new_point)
                point = new_point

    if method == 'c-space':
        # Use c-space search to convert and find the location of given point
        # All the idx attributes are converted to c-space -- point, cell, block
        idx = Search(grid, point)
        idx.compute(method='c-space')
        while True:
            with Timer(text="Elapsed time for loop number " + str(len(streamline)) + ": {:.8f}"):
                interp = Interpolation(flow, idx)
                interp.compute(method='c-space')
                intg = Integration(interp)
                new_point = intg.compute(method='cRK4', time_step=1e-1)
                if new_point is None:
                    # For multi-block case if the point is out-of-block
                    # Use previous point and run one-step of p-space algo
                    print('inside loop')
                    idx = Search(grid, save_point)
                    interp = Interpolation(flow, idx)
                    intg = Integration(interp)
                    idx.compute(method='block_distance')
                    interp.compute()
                    new_point = intg.compute(method='pRK4', time_step=1e-1)
                    if new_point is None:
                        print('Integration complete!')
                        break
                    else:
                        new_point = idx.p2c(new_point)  # Move point obtained to c-space
                save_point = idx.c2p(new_point)
                streamline.append(save_point)
                idx.point = new_point

    return np.array(streamline)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # slc = main('data/plate_data/plate.sp.x', 'data/plate_data/sol-0000010.q', [8.5, 0.5, 0.01], method='c-space')
    # slp = main('data/plate_data/plate.sp.x', 'data/plate_data/sol-0000010.q', [8.5, 0.5, 0.01], method='p-space')
    # slc = main('data/cylinder_data/cylinder.sp.x', 'data/cylinder_data/sol-0010000.q', [0.5, 0.5, 0.5], method='c-space')
    # slp = main('data/cylinder_data/cylinder.sp.x', 'data/cylinder_data/sol-0010000.q', [0.5, 0.5, 0.5], method='p-space')
    # main('data/multi_block/test/test.mb.sp.x', 'data/multi_block/test/test.mb.sp.q', [-0.5, -0.5, -0.5])
    slmb = main('data/multi_block/plate/plate.mb.sp.x', 'data/multi_block/plate/plate.mb.sp.q', [8.5, 0.5, 0.01])

    # xc, yc, zc = slc[:, 0], slc[:, 1], slc[:, 2]
    # xp, yp, zp = slp[:, 0], slp[:, 1], slp[:, 2]
    xmb, ymb, zmb = slmb[:-2, 0], slmb[:-2, 1], slmb[:-2, 2]

    import matplotlib.pyplot as plt
    import random

    ax = plt.axes(projection='3d')
    # ax.plot3D(xc, yc, zc, 'r')
    # plt.figure()
    # ax.plot3D(xp, yp, zp, 'b')
    ax.plot3D(xmb, ymb, zmb, 'g')
    plt.show()

