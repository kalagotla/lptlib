# Uses the program API to extract streamlines
import numpy as np


class Streamlines:
    """
    Module to extract particle path information

    ...

    Attributes
    ----------
    Input:
        grid_file : str
            Path to the plot3d grid data file
        flow_file : str
            Path to the plot3d flow data file
        point : list
            Starting point for integration
        search : str
            Default is p-space; can specify c-space
        interpolation : str
            Default is p-space; can specify c-space
        integration : str
            Default is p-space; can specify c-space
        time_step : float
            Default is 1e-3

    Output:
        streamline : numpy.ndarray
            shape is nx3; each column represents x, y, z

    Methods
    -------
    compute()
        integrates and returns the streamline ndarray
    ...

    Example:
        sl = Streamlines('../../data/vortex/vortex.sb.sp.x', '../../data/vortex/vortex.sb.sp.q', [-0.05, 0.05, 5])
        sl.compute()
    -------


    """
    def __init__(self, grid_file, flow_file, point,
                 search='p-space', interpolation='p-space', integration='pRK4',
                 time_step=1e-3):
        self.grid_file = grid_file
        self.flow_file = flow_file
        self.point = np.array(point)
        self.search = search
        self.interpolation = interpolation
        self.integration = integration
        self.time_step = time_step
        self.streamline = []
        self.fvelocity = []
        self.svelocity = []

    # TODO: Need to add doc for streamlines

    @staticmethod
    def angle_btw(v1, v2):
        # If vector are tiny return for handling tiny time steps
        # A lot of trail and error decided 1e-12 would work
        if np.linalg.norm(v1) <= 1e-12 or np.linalg.norm(v2) <= 1e-12: return

        u1 = v1 / np.linalg.norm(v1)
        u2 = v2 / np.linalg.norm(v2)

        y = u1 - u2
        x = u1 + u2

        a0 = 2 * np.arctan(np.linalg.norm(y) / np.linalg.norm(x))

        if (not np.signbit(a0)) or np.signbit(np.pi - a0):
            return np.rad2deg(a0)
        elif np.signbit(a0):
            return 0.0
        else:
            return 180

    def compute(self, method='p-space'):
        from src.function.timer import Timer
        from src.io.plot3dio import GridIO
        from src.io.plot3dio import FlowIO
        from src.streamlines.search import Search
        from src.streamlines.interpolation import Interpolation
        from src.streamlines.integration import Integration

        grid = GridIO(self.grid_file)
        flow = FlowIO(self.flow_file)

        # Read in the grid and flow data
        grid.read_grid()
        flow.read_flow()
        grid.compute_metrics()

        self.streamline.append(self.point)

        if method == 'p-space':
            while True:
                with Timer(text="Elapsed time for loop number " + str(len(self.streamline)) + ": {:.8f}"):
                    idx = Search(grid, self.point)
                    interp = Interpolation(flow, idx)
                    intg = Integration(interp)
                    idx.compute(method=self.search)
                    interp.compute(method=self.interpolation)
                    new_point, new_vel = intg.compute(method=self.integration, time_step=self.time_step)
                    if new_point is None:
                        print('Integration complete!')
                        break
                    self.streamline.append(new_point)
                    self.fvelocity.append(new_vel)
                    self.point = new_point

        if method == 'adaptive-p-space':
            while True:
                with Timer(text="Elapsed time for loop number " + str(len(self.streamline)) + ": {:.8f}"):
                    idx = Search(grid, self.point)
                    interp = Interpolation(flow, idx)
                    intg = Integration(interp)
                    idx.compute(method=self.search)
                    interp.compute(method=self.interpolation)
                    new_point, new_vel = intg.compute(method=self.integration, time_step=self.time_step)
                    if new_point is None:
                        print('Integration complete!')
                        break

                    # Save results and adjust time-step
                    if self.angle_btw(new_point - self.point, new_vel) <= 0.1:
                        self.streamline.append(new_point)
                        self.fvelocity.append(new_vel)
                        self.point = new_point
                        self.time_step = 2 * self.time_step
                    elif self.angle_btw(new_point - self.point, new_vel) >= 1.4:
                        self.time_step = 0.5 * self.time_step
                    else:
                        self.streamline.append(new_point)
                        self.fvelocity.append(new_vel)
                        self.point = new_point

        if method == 'c-space':
            # Use c-space search to convert and find the location of given point
            # All the idx attributes are converted to c-space -- point, cell, block
            save_point = self.point
            idx = Search(grid, self.point)
            idx.compute(method='c-space')
            while True:
                with Timer(text="Elapsed time for loop number " + str(len(self.streamline)) + ": {:.8f}"):
                    interp = Interpolation(flow, idx)
                    interp.compute(method='c-space')
                    intg = Integration(interp)
                    new_point = intg.compute(method='cRK4', time_step=self.time_step)
                    if new_point is None:
                        # For multi-block case if the point is out-of-block
                        # Use previous point and run one-step of p-space algo
                        print('Point exited the block! Searching for new position...')
                        idx = Search(grid, save_point)
                        interp = Interpolation(flow, idx)
                        intg = Integration(interp)
                        idx.compute(method='block_distance')
                        interp.compute()
                        new_point = intg.compute(method='pRK4', time_step=1)
                        if new_point is None:
                            print('No location found. Point out-of-domain. Integration complete!')
                            break
                        else:
                            # Update the block in idx
                            idx = Search(grid, new_point)
                            idx.compute(method='c-space')
                            self.streamline.append(new_point)
                            # new_point = idx.p2c(new_point)  # Move point obtained to c-space
                    else:
                        save_point = idx.c2p(new_point)
                        self.streamline.append(save_point)
                        idx.point = new_point

        if method == 'ppath':
            vel = None
            while True:
                with Timer(text="Elapsed time for loop number " + str(len(self.streamline)) + ": {:.8f}"):
                    idx = Search(grid, self.point)
                    interp = Interpolation(flow, idx)
                    intg = Integration(interp)
                    idx.compute(method=self.search)
                    interp.compute(method=self.interpolation)
                    new_point, new_vel = intg.compute_ppath(diameter=5e-4, density=1000, viscosity=1.827e-5,
                                                            velocity=vel, method='pRK4', time_step=self.time_step)
                    if new_point is None:
                        print('Integration complete!')
                        break

                    # Save results and continue the loop
                    self.streamline.append(new_point)
                    self.svelocity.append(vel)
                    self.point = new_point
                    vel = new_vel.copy()

        if method == 'adaptive-ppath':
            vel = None
            while True:
                with Timer(text="Elapsed time for loop number " + str(len(self.streamline)) + ": {:.8f}"):
                    idx = Search(grid, self.point)
                    interp = Interpolation(flow, idx)
                    intg = Integration(interp)
                    idx.compute(method=self.search)
                    interp.compute(method=self.interpolation)
                    new_point, new_vel = intg.compute_ppath(diameter=5e-4, density=1000, viscosity=1.827e-5,
                                                            velocity=vel, method='pRK4', time_step=self.time_step)
                    if new_point is None:
                        print('Integration complete!')
                        break

                    # Save results and continue the loop
                    if vel is None:
                        # For the first step in the loop
                        self.streamline.append(new_point)
                        self.svelocity.append(new_vel)
                        self.point = new_point
                        vel = new_vel.copy()
                    elif self.angle_btw(new_point - self.point, vel) is None:
                        print('Increasing time step. Successive points are same')
                        self.time_step = 10 * self.time_step
                    elif self.angle_btw(new_point - self.point, vel) <= 0.05:
                        self.streamline.append(new_point)
                        self.svelocity.append(vel)
                        self.point = new_point
                        vel = new_vel.copy()
                        self.time_step = 2 * self.time_step
                    # Check for if the points are identical because of tiny time step and deflection
                    elif self.angle_btw(new_point - self.point, vel) >= 1.4:
                        self.time_step = 0.5 * self.time_step
                    else:
                        self.streamline.append(new_point)
                        self.svelocity.append(vel)
                        self.point = new_point
                        vel = new_vel.copy()

        return
