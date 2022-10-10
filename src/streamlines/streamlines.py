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
        fvelocity : list
            shape is nx3; each column represents ufx, ufy, ufz
        svelocity: list
            shape is nx3; each column represents vpx, vpy, vpz

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
                 search='block_distance', interpolation='p-space', integration='pRK4',
                 diameter=1e-7, density=1000, viscosity=1.827e-5,
                 time_step=1e-3, max_time_step=1, drag_model='stokes'):
        self.grid_file = grid_file
        self.flow_file = flow_file
        self.point = np.array(point)
        self.search = search
        self.interpolation = interpolation
        self.integration = integration
        self.diameter = diameter
        self.density = density
        self.viscosity = viscosity
        self.time_step = time_step
        self.max_time_step = max_time_step
        self.drag_model = drag_model
        self.streamline = []
        self.fvelocity = []
        self.svelocity = []

    # TODO: Need to add doc for streamlines

    @staticmethod
    def angle_btw(v1, v2):
        # If vectors are tiny return for handling tiny time steps
        # A lot of trail and error decided 1e-9 would work fast
        if np.linalg.norm(v1) <= 1e-9 or np.linalg.norm(v2) <= 1e-9:
            return None

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
        from src.function.variables import Variables

        grid = GridIO(self.grid_file)
        flow = FlowIO(self.flow_file)

        # Read in the grid and flow data
        grid.read_grid()
        flow.read_flow()
        grid.compute_metrics()

        # Add data to output at the given point
        # This is the assumption where particle velocity is same as the fluid
        self.streamline.append(self.point)
        idx = Search(grid, self.point)
        interp = Interpolation(flow, idx)
        idx.compute()
        interp.compute()
        q_interp = Variables(interp)
        q_interp.compute_velocity()
        uf = q_interp.velocity.reshape(3)
        self.fvelocity.append(uf)
        self.svelocity.append(uf)
        loop_check = 0

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
                    self.svelocity.append(new_vel)
                    self.point = new_point

        if method == 'adaptive-p-space':
            vel = None
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

                    # Adaptive algorithm starts
                    # Save results and adjust time-step
                    # Details for the algorithm are provided in adaptive-ppath
                    if vel is None:
                        # For the first step in the loop
                        self.streamline.append(new_point)
                        self.fvelocity.append(new_vel)
                        self.svelocity.append(new_vel)
                        self.point = new_point
                        vel = new_vel.copy()
                    elif self.angle_btw(new_point - self.point, vel) is None:
                        print('Increasing time step. Successive points are same')
                        self.time_step = 10 * self.time_step
                        loop_check += 1
                        if loop_check == 70:
                            print('Stuck in the same loop for too long. Integration ends!')
                            return
                    elif self.angle_btw(new_point - self.point, vel) <= 0.1 \
                            and self.time_step <= self.max_time_step:
                        self.streamline.append(new_point)
                        self.fvelocity.append(new_vel)
                        self.svelocity.append(new_vel)
                        self.point = new_point
                        vel = new_vel.copy()
                        self.time_step = 2 * self.time_step
                        loop_check = 0
                    elif self.angle_btw(new_point - self.point, vel) >= 1.4 and self.time_step >= 1e-12:
                        self.time_step = 0.5 * self.time_step
                    else:
                        self.streamline.append(new_point)
                        self.fvelocity.append(new_vel)
                        self.svelocity.append(new_vel)
                        self.point = new_point
                        vel = new_vel.copy()
                        loop_check = 0

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
                    new_point, new_pvel, new_cvel = intg.compute(method='cRK4', time_step=self.time_step)
                    if new_point is None:
                        # For multi-block case if the point is out-of-block
                        # Use previous point and run one-step of p-space algo
                        print('Point exited the block! Searching for new position...')
                        idx = Search(grid, save_point)
                        interp = Interpolation(flow, idx)
                        intg = Integration(interp)
                        idx.compute(method='block_distance')
                        interp.compute()
                        new_point, new_pvel = intg.compute(method='pRK4', time_step=self.time_step)
                        if new_point is None:
                            print('Point out-of-domain. Integration complete!')
                            break
                        else:
                            # Update the block in idx
                            idx = Search(grid, new_point)
                            idx.compute(method='c-space')
                            self.streamline.append(new_point)
                            self.fvelocity.append(new_pvel)
                            self.svelocity.append(new_pvel)
                            # new_point = idx.p2c(new_point)  # Move point obtained to c-space
                    else:
                        save_point = idx.c2p(new_point)
                        self.streamline.append(save_point)
                        self.fvelocity.append(new_pvel)
                        self.svelocity.append(new_pvel)
                        idx.point = new_point

        if method == 'adaptive-c-space':
            # Use c-space search to convert and find the location of given point
            # All the idx attributes are converted to c-space -- point, cell, block
            # pvel signifies physical-space velocity
            save_point = self.point
            idx = Search(grid, self.point)
            idx.compute(method='c-space')
            pvel = None
            while True:
                with Timer(text="Elapsed time for loop number " + str(len(self.streamline)) + ": {:.8f}"):
                    interp = Interpolation(flow, idx)
                    interp.compute(method='c-space')
                    intg = Integration(interp)
                    new_point, new_pvel, new_cvel = intg.compute(method='cRK4', time_step=self.time_step)
                    if new_point is None:
                        # For multi-block case if the point is out-of-block
                        # Use previous point and run one-step of p-space algo
                        print('Point exited the block! Searching for new position...')
                        idx = Search(grid, save_point)
                        interp = Interpolation(flow, idx)
                        intg = Integration(interp)
                        idx.compute(method='block_distance')
                        interp.compute()
                        new_point, new_pvel = intg.compute(method='pRK4', time_step=self.time_step)
                        if new_point is None:
                            print('Point out-of-domain. Integration complete!')
                            break
                        else:
                            # Update the block in idx
                            idx = Search(grid, new_point)
                            idx.compute(method='c-space')
                            self.streamline.append(new_point)
                            self.fvelocity.append(new_pvel)
                            self.svelocity.append(new_pvel)
                            self.point = save_point
                            save_point = new_point
                            pvel = new_pvel.copy()
                            # new_point = idx.p2c(new_point)  # Move point obtained to c-space
                    else:
                        self.point = save_point
                        save_point = idx.c2p(new_point)

                        # Adaptive algorithm starts
                        # Save results and adjust time-step
                        # Details for the algorithm are provided in adaptive-ppath
                        # Decided to use new_pvel for adaptivity because of the in-shock cell RK4 integration
                        # RK4 depends on future points for integration and this adjustment moves it forward
                        if self.angle_btw(save_point - self.point, new_pvel) is None:
                            print('Increasing time step. Successive points are same')
                            self.time_step = 10 * self.time_step
                            loop_check += 1
                            if loop_check == 70:
                                print('Stuck in the same loop for too long. Integration ends!')
                                return
                        elif self.angle_btw(save_point - self.point, new_pvel) <= 0.1 \
                                and self.time_step <= self.max_time_step:
                            self.point = save_point
                            save_point = idx.c2p(new_point)
                            self.streamline.append(save_point)
                            self.fvelocity.append(new_pvel)
                            self.svelocity.append(new_pvel)
                            idx.point = new_point
                            pvel = new_pvel.copy()
                            self.time_step = 2 * self.time_step
                            loop_check = 0
                        elif self.angle_btw(save_point - self.point, new_pvel) >= 1.4 and self.time_step >= 1e-12:
                            self.time_step = 0.5 * self.time_step
                        else:
                            self.point = save_point
                            save_point = idx.c2p(new_point)
                            self.streamline.append(save_point)
                            self.fvelocity.append(new_pvel)
                            self.svelocity.append(new_pvel)
                            idx.point = new_point
                            pvel = new_pvel.copy()
                            loop_check = 0

        if method == 'ppath':
            vel = None
            fvel = None
            while True:
                with Timer(text="Elapsed time for loop number " + str(len(self.streamline)) + ": {:.8f}"):
                    idx = Search(grid, self.point)
                    interp = Interpolation(flow, idx)
                    intg = Integration(interp)
                    idx.compute(method=self.search)
                    interp.compute(method=self.interpolation)
                    new_point, new_vel, new_fvel = intg.compute_ppath(diameter=self.diameter,
                                                                      density=self.density,
                                                                      viscosity=self.viscosity,
                                                                      velocity=vel, method='pRK4',
                                                                      time_step=self.time_step,
                                                                      drag_model=self.drag_model)
                    if new_point is None:
                        print('Integration complete!')
                        break

                    # Save results and continue the loop
                    self.streamline.append(new_point)
                    self.svelocity.append(new_vel)
                    self.fvelocity.append(new_fvel)
                    self.point = new_point
                    vel = new_vel.copy()
                    fvel = new_fvel.copy()

        if method == 'adaptive-ppath':
            # particle velocity
            vel = None
            # fluid velocity
            fvel = None
            while True:
                with Timer(text="Elapsed time for loop number " + str(len(self.streamline)) + ": {:.8f}"):
                    idx = Search(grid, self.point)
                    interp = Interpolation(flow, idx)
                    intg = Integration(interp)
                    idx.compute(method=self.search)
                    interp.compute(method=self.interpolation)
                    new_point, new_vel, new_fvel = intg.compute_ppath(diameter=self.diameter, density=self.density,
                                                                      viscosity=self.viscosity, velocity=vel,
                                                                      method='pRK4', time_step=self.time_step,
                                                                      drag_model=self.drag_model)
                    if new_point is None:
                        print('Integration complete!')
                        break

                    # Check for mid-rk4 blowup
                    if intg.rk4_bool is True:
                        print('Mid-RK4 blow up! Reducing time-step')
                        intg.rk4_bool = False
                        self.time_step = 0.5 * self.time_step

                    # Adaptive algorithm starts
                    # Save results and continue the loop
                    elif vel is None:
                        # For the first step in the loop
                        self.streamline.append(new_point)
                        self.svelocity.append(new_vel)
                        self.fvelocity.append(new_fvel)
                        self.point = new_point
                        vel = new_vel.copy()
                        fvel = new_fvel.copy()
                    # Check for if the points are identical because of tiny time step and deflection
                    elif self.angle_btw(new_point - self.point, vel) is None:
                        print('Increasing time step. Successive points are same')
                        self.time_step = 10 * self.time_step
                        loop_check += 1
                        if loop_check == 70:
                            print('Successive points did not change for too long. Integration ends!')
                            return
                    # Increase time step when angle is below 0.05 degrees
                    elif self.angle_btw(new_point - self.point, vel) <= 0.01 and self.time_step <= self.max_time_step:
                        # print('Increasing time step. Low deflection wrt velocity')
                        self.streamline.append(new_point)
                        self.svelocity.append(new_vel)
                        self.fvelocity.append(new_fvel)
                        self.point = new_point
                        vel = new_vel.copy()
                        fvel = new_fvel.copy()
                        self.time_step = 2 * self.time_step
                        loop_check = 0
                    # Decrease time step when angle is above 1.4 degrees
                    # Make sure time step does not go to zero; 1 pico-second
                    elif self.angle_btw(new_point - self.point, vel) >= 1.4 and self.time_step >= 1e-12:
                        # print('Decreasing time step. High deflection wrt velocity')
                        self.time_step = 0.5 * self.time_step
                    # Save if none of the above conditions meet
                    else:
                        self.streamline.append(new_point)
                        self.svelocity.append(new_vel)
                        self.fvelocity.append(new_fvel)
                        self.point = new_point
                        vel = new_vel.copy()
                        fvel = new_fvel.copy()
                        loop_check = 0

        if method == 'ppath-c-space':
            pvel = None
            fvel = None
            save_point = self.point
            idx = Search(grid, self.point)
            idx.compute(method='c-space')
            while True:
                with Timer(text="Elapsed time for loop number " + str(len(self.streamline)) + ": {:.8f}"):
                    interp = Interpolation(flow, idx)
                    interp.compute(method='c-space')
                    intg = Integration(interp)
                    new_point, new_fvel, new_pvel = intg.compute_ppath(diameter=self.diameter,
                                                                       density=self.density,
                                                                       viscosity=self.viscosity,
                                                                       velocity=pvel, method='cRK4',
                                                                       time_step=self.time_step,
                                                                       drag_model=self.drag_model)
                    if new_point is None:
                        # For multi-block case if the point is out-of-block
                        # Use previous point and run one-step of p-space algo
                        print('Point exited the block! Searching for new position...')
                        idx = Search(grid, save_point)
                        interp = Interpolation(flow, idx)
                        intg = Integration(interp)
                        idx.compute(method='block_distance')
                        interp.compute()
                        new_point, new_fvel, new_pvel = intg.compute_ppath(diameter=self.diameter,
                                                                           density=self.density,
                                                                           viscosity=self.viscosity,
                                                                           velocity=pvel, method='pRK4',
                                                                           time_step=self.time_step,
                                                                           drag_model=self.drag_model)
                        if new_point is None:
                            print('Point out-of-domain. Integration complete!')
                            break
                        else:
                            # Update the block in idx
                            idx = Search(grid, new_point)
                            idx.compute(method='c-space')
                            self.streamline.append(new_point)
                            self.fvelocity.append(new_fvel)
                            self.svelocity.append(new_pvel)
                            pvel = new_pvel.copy()
                            fvel = new_fvel.copy()
                    else:

                        # Check for mid-rk4 blowup
                        if intg.rk4_bool is True:
                            print('Mid-RK4 blow up! Reducing time-step')
                            intg.rk4_bool = False
                            self.time_step = 0.5 * self.time_step

                        else:
                            self.point = save_point
                            save_point = idx.c2p(new_point)
                            self.streamline.append(save_point)
                            self.fvelocity.append(new_fvel)
                            self.svelocity.append(new_pvel)
                            pvel = new_pvel.copy()
                            fvel = new_fvel.copy()
                            idx.point = new_point

        if method == 'adaptive-ppath-c-space':
            pvel = None
            fvel = None
            save_point = self.point
            idx = Search(grid, self.point)
            idx.compute(method='c-space')
            while True:
                with Timer(text="Elapsed time for loop number " + str(len(self.streamline)) + ": {:.8f}"):
                    interp = Interpolation(flow, idx)
                    interp.compute(method='c-space')
                    intg = Integration(interp)
                    new_point, new_fvel, new_pvel = intg.compute_ppath(diameter=self.diameter,
                                                                       density=self.density,
                                                                       viscosity=self.viscosity,
                                                                       velocity=pvel, method='cRK4',
                                                                       time_step=self.time_step,
                                                                       drag_model=self.drag_model)
                    if new_point is None:
                        # For multi-block case if the point is out-of-block
                        # Use previous point and run one-step of p-space algo
                        print('Point exited the block! Searching for new position...')
                        idx = Search(grid, save_point)
                        interp = Interpolation(flow, idx)
                        intg = Integration(interp)
                        idx.compute(method='block_distance')
                        interp.compute()
                        new_point, new_fvel, new_pvel = intg.compute_ppath(diameter=self.diameter,
                                                                           density=self.density,
                                                                           viscosity=self.viscosity,
                                                                           velocity=pvel, method='pRK4',
                                                                           time_step=self.time_step,
                                                                           drag_model=self.drag_model)
                        if new_point is None:
                            print('Point out-of-domain. Integration complete!')
                            break
                        else:
                            # Update the block in idx
                            idx = Search(grid, new_point)
                            idx.compute(method='c-space')
                            # new_point found is in p-space; so, append
                            self.streamline.append(new_point)
                            self.fvelocity.append(new_fvel)
                            self.svelocity.append(new_pvel)
                            pvel = new_pvel.copy()
                            fvel = new_fvel.copy()
                    else:
                        self.point = save_point
                        save_point = idx.c2p(new_point)

                        # Check for mid-rk4 blowup
                        if intg.rk4_bool is True:
                            print('Mid-RK4 blow up! Reducing time-step')
                            intg.rk4_bool = False
                            self.time_step = 0.5 * self.time_step

                        # Adaptive algorithm starts
                        # Save results and adjust time-step
                        # Details for the algorithm are provided in adaptive-ppath
                        elif pvel is None:
                            # For the first step in the loop
                            self.streamline.append(save_point)
                            self.svelocity.append(new_pvel)
                            self.fvelocity.append(new_fvel)
                            self.point = new_point
                            pvel = new_pvel.copy()
                            fvel = new_fvel.copy()
                        elif self.angle_btw(save_point - self.point, pvel) is None:
                            print('Increasing time step. Successive points are same')
                            self.time_step = 10 * self.time_step
                            loop_check += 1
                            if loop_check == 70:
                                print('Stuck in the same loop for too long. Integration ends!')
                                return
                        elif self.angle_btw(save_point - self.point, pvel) <= 0.01 \
                                and self.time_step <= self.max_time_step:
                            self.point = save_point
                            save_point = idx.c2p(new_point)
                            self.streamline.append(save_point)
                            self.fvelocity.append(new_fvel)
                            self.svelocity.append(new_pvel)
                            idx.point = new_point.copy()
                            pvel = new_pvel.copy()
                            fvel = new_fvel.copy()
                            self.time_step = 2 * self.time_step
                            loop_check = 0
                        elif self.angle_btw(save_point - self.point, pvel) >= 1.4 and self.time_step >= 1e-12:
                            self.time_step = 0.5 * self.time_step
                        else:
                            self.point = save_point
                            save_point = idx.c2p(new_point)
                            self.streamline.append(save_point)
                            self.fvelocity.append(new_fvel)
                            self.svelocity.append(new_pvel)
                            idx.point = new_point.copy()
                            pvel = new_pvel.copy()
                            fvel = new_fvel.copy()
                            loop_check = 0

        return
