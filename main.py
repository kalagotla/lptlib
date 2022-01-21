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
        grid.compute_metrics()
        save_point = point
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
                    print('Point exited the block! Searching for new position...')
                    idx = Search(grid, save_point)
                    interp = Interpolation(flow, idx)
                    intg = Integration(interp)
                    idx.compute(method='block_distance')
                    interp.compute()
                    new_point = intg.compute(method='pRK4', time_step=1e-1)
                    if new_point is None:
                        print('No location found. Point out-of-domain. Integration complete!')
                        break
                    else:
                        # Update the block in idx
                        idx = Search(grid, new_point)
                        idx.compute(method='c-space')
                        streamline.append(new_point)
                        # new_point = idx.p2c(new_point)  # Move point obtained to c-space
                else:
                    save_point = idx.c2p(new_point)
                    streamline.append(save_point)
                    idx.point = new_point

    return np.array(streamline), grid


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # slc, gridc = main('data/plate_data/plate.sp.x', 'data/plate_data/sol-0000010.q', [0.5, 0.5, 0.01], method='c-space')
    # slp, gridp = main('data/plate_data/plate.sp.x', 'data/plate_data/sol-0000010.q', [0.5, 0.5, 0.01], method='p-space')
    # slmbc, gridmbc = main('data/multi_block/plate/plate.mb.sp.x', 'data/multi_block/plate/plate.mb.sp.q',
    #                       [0.5, 0.5, 0.01], method='c-space')
    # slmbp, gridmbp = main('data/multi_block/plate/plate.mb.sp.x', 'data/multi_block/plate/plate.mb.sp.q',
    #                       [0.5, 0.5, 0.01], method='p-space')
    # Uncomment lines below to test cylinder grid
    # slc, gridc = main('data/cylinder_data/cylinder.sp.x', 'data/cylinder_data/sol-0010000.q',
    #                   [0.5, 1.5, 0.5], method='c-space')
    slp, gridp = main('data/cylinder_data/cylinder.sp.x', 'data/cylinder_data/sol-0010000.q',
                      [0.5, 1.5, 0.5], method='p-space')


    # xc, yc, zc = slc[:, 0], slc[:, 1], slc[:, 2]
    xp, yp, zp = slp[:, 0], slp[:, 1], slp[:, 2]
    # xmbc, ymbc, zmbc = slmbc[:, 0], slmbc[:, 1], slmbc[:, 2]
    # xmbp, ymbp, zmbp = slmbp[:, 0], slmbp[:, 1], slmbp[:, 2]

    import matplotlib.pyplot as plt

    ax = plt.axes(projection='3d')
    # ax.plot3D(xc, yc, zc, 'r', label='SB-C')
    ax.plot3D(xp, yp, zp, 'b', label='SB-P')
    # ax.plot3D(xmbc, ymbc, zmbc, 'g', label='MB-C')
    # ax.plot3D(xmbp, ymbp, zmbp, 'k', label='MB-P')
    # ax.set_xlim([gridc.grd_min[0, 0], gridc.grd_max[0, 0]])
    # ax.set_ylim([gridc.grd_min[0, 1], gridc.grd_max[0, 1]])
    # ax.set_zlim([gridc.grd_min[0, 2], gridc.grd_max[0, 2]])
    ax.set_title('Comparing different streamline algorithms')
    ax.legend()
    plt.show()

