# Uses the program API to extract streamlines
import numpy as np

from src.function.timer import Timer
from src.function.variables import Variables
from src.io.plot3dio import FlowIO
from src.io.plot3dio import GridIO
from src.streamlines.integration import Integration
from src.streamlines.interpolation import Interpolation
from src.streamlines.search import Search


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
        magnitude_adaptivity : float
            Set to override the adaptivity value

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

    def __init__(self, grid_file=None, flow_file=None, point=None,
                 search='p-space', interpolation='p-space', integration='pRK4',
                 diameter=1e-7, density=1000, viscosity=1.827e-5,
                 time_step=1e-3, max_time_step=1, drag_model='stokes', adaptivity=0.001,
                 magnitude_adaptivity=None, filepath=None, task=None):
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
        self.time = []
        self.adaptivity = adaptivity
        self.magnitude_adaptivity = magnitude_adaptivity
        self.filepath = filepath
        self.task = task

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

    @staticmethod
    def _save_data(self):
        """
        Pass self object to save data if the filepath is provided
        Args:
            self: self object of the class

        Returns:
            None
        """
        if self.filepath is not None:
            p_xdata = np.array(self.streamline)
            vdata = np.array(self.svelocity)
            udata = np.array(self.fvelocity)
            tdata = np.array(self.time).reshape(-1, 1)
            f_xdata = p_xdata.copy()
            for _i in range(1, len(f_xdata)):
                f_xdata[_i] = f_xdata[_i - 1] + udata[_i - 1] * tdata[_i - 1]
            # Data is added towards the end because of the development cycle. Mostly to work with dataio
            _data_save = np.hstack((p_xdata, vdata, udata, tdata, f_xdata))

            np.save(self.filepath + 'ppath_' + str(self.task), _data_save)
            print('** SUCCESS ** Done writing file for particle number - ' + str(self.task) + ' ** SUCCESS **')
            # set self to None to clear up memory after saving required data
            self.streamline = []
            self.svelocity = []
            self.fvelocity = []
            self.time = []
        return

    @staticmethod
    def _magnitude(self, v1, v2):
        # return the magnitude of the vector
        _r = np.linalg.norm(v1 - v2) / min(np.linalg.norm(v1), np.linalg.norm(v2))
        if _r < 1e-6 * self.magnitude_adaptivity:
            _r = 1e-6 * self.magnitude_adaptivity
        return _r

    def compute(self, method='p-space', grid=None, flow=None):
        """
        Method to compute particle paths. Contains multiple algorithms
        Args:
            method: str (default: p-space) - can be p-space, c-space, adaptive-p-space,
             adaptive-c-space, ppath, adaptive-ppath, ppath-c-space, adaptive-ppath-c-space
            grid: GridIO object (default: None) - if grid is already read in and metrics are computed pass the object
            flow: FlowIO object (default: None) - if flow is already read in pass the object

        Returns:
            None - saves the data in the object itself and can be accessed using the object attributes
            streamline, fvelocity, svelocity, time, point, task, search, interpolation, integration, time_step, etc

        """
        if grid is None or flow is None:
            grid = GridIO(self.grid_file)
            flow = FlowIO(self.flow_file)

            # Read in the grid and flow data
            grid.read_grid()
            flow.read_flow()
            grid.compute_metrics()

        if self.magnitude_adaptivity is None:
            self.magnitude_adaptivity = abs(np.min(np.gradient(grid.grd[..., 0])))
        # Add data to output at the given point
        # This is the assumption where particle velocity is same as the fluid
        self.streamline.append(self.point)
        idx = Search(grid, self.point)
        # if list use the first element if not use the same object
        if isinstance(flow.unsteady_flow, list):
            interp = Interpolation(flow.unsteady_flow[0], idx)
        else:
            interp = Interpolation(flow, idx)
        idx.compute(method=self.search)
        interp.compute(method=self.interpolation)
        q_interp = Variables(interp)
        q_interp.compute_velocity()
        uf = q_interp.velocity.reshape(3)
        self.fvelocity.append(uf)
        self.svelocity.append(uf)
        self.time.append(self.time_step)
        loop_check = 0

        if method == 'p-space':
            t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            t.start()
            while True:
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
                self.time.append(self.time_step)
                self.point = new_point

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            t.stop()

        if method == 'adaptive-p-space':
            t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            t.start()
            vel = None
            while True:
                idx = Search(grid, self.point)
                interp = Interpolation(flow, idx)
                intg = Integration(interp)
                idx.compute(method=self.search)
                interp.compute(method=self.interpolation)
                new_point, new_vel = intg.compute(method=self.integration, time_step=self.time_step)
                if new_point is None:
                    print('Checking if the end of the domain is reached...')
                    self.time_step = 1e-9 * self.time_step
                    new_point, new_vel = intg.compute(method=self.integration, time_step=self.time_step)
                    if new_point is not None:
                        print('Continuing integration by decreasing time-step!')
                        continue
                    elif new_point is None:
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
                    self.time.append(self.time_step)
                    self.point = new_point
                    vel = new_vel.copy()
                elif self.angle_btw(new_point - self.point, vel) is None:
                    print('Increasing time step. Successive points are same')
                    self.time_step = 10 * self.time_step
                    loop_check += 1
                    if loop_check == 70:
                        print('Stuck in the same loop for too long. Integration ends!')
                        break
                elif self.angle_btw(new_point - self.point, vel) <= 0.001 \
                        and self.time_step <= self.max_time_step:
                    self.streamline.append(new_point)
                    self.fvelocity.append(new_vel)
                    self.svelocity.append(new_vel)
                    self.time.append(self.time_step)
                    self.point = new_point
                    vel = new_vel.copy()
                    self.time_step = 2 * self.time_step
                    loop_check = 0
                elif self.angle_btw(new_point - self.point, vel) >= 0.01 and self.time_step >= 1e-12:
                    self.time_step = 0.5 * self.time_step
                else:
                    self.streamline.append(new_point)
                    self.fvelocity.append(new_vel)
                    self.svelocity.append(new_vel)
                    self.time.append(self.time_step)
                    self.point = new_point
                    vel = new_vel.copy()
                    loop_check = 0

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            t.stop()

        if method == 'c-space':
            # Use c-space search to convert and find the location of given point
            # All the idx attributes are converted to c-space -- point, cell, block
            t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            t.start()
            save_point = self.point
            idx = Search(grid, self.point)
            idx.compute(method='c-space')
            while True:
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
                        # new_point is in p-space for this else block

                        idx = Search(grid, new_point)
                        idx.compute(method='c-space')
                        self.streamline.append(new_point)
                        self.fvelocity.append(new_pvel)
                        self.svelocity.append(new_pvel)
                        self.time.append(self.time_step)
                        self.point = save_point
                        save_point = new_point
                        pvel = new_pvel.copy()
                else:
                    save_point = idx.c2p(new_point)
                    self.streamline.append(save_point)
                    self.fvelocity.append(new_pvel)
                    self.svelocity.append(new_pvel)
                    self.time.append(self.time_step)
                    idx.point = new_point

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            t.stop()

        if method == 'adaptive-c-space':
            # Use c-space search to convert and find the location of given point
            # All the idx attributes are converted to c-space -- point, cell, block
            # pvel signifies physical-space velocity
            t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            t.start()
            save_point = self.point
            idx = Search(grid, self.point)
            idx.compute(method='c-space')
            pvel = None
            while True:
                interp = Interpolation(flow, idx)
                interp.compute(method='c-space')
                intg = Integration(interp)
                new_point, new_pvel, new_cvel = intg.compute(method='cRK4', time_step=self.time_step)

                # Check for large time-step in the last loop
                if new_point is None:
                    # Checking by decreasing time-step
                    print('Checking if the end of the domain is reached...')
                    self.time_step = 1e-9 * self.time_step
                    new_point, new_pvel, new_cvel = intg.compute(method='cRK4', time_step=self.time_step)
                    if new_point is not None:
                        print('Continuing integration by decreasing time-step!')
                        continue
                    # Check for block switch
                    elif new_point is None:
                        # For multi-block case if the point is out-of-block
                        # Use previous point and run one-step of p-space algo
                        print('Point exited the block! Searching for new position...')
                        idx = Search(grid, save_point)
                        interp = Interpolation(flow, idx)
                        intg = Integration(interp)
                        idx.compute(method='block_distance')
                        interp.compute()
                        new_point, new_pvel = intg.compute(method='pRK4', time_step=self.time_step)
                        # Even after pRK4 if the point is None; End integration
                        if new_point is None:
                            print('End of the domain. Integration Ends!')
                            break
                        # If not none; update to c-space and run
                        else:
                            # Update the block in idx
                            idx = Search(grid, new_point)
                            idx.compute(method='c-space')
                            self.streamline.append(new_point)
                            self.fvelocity.append(new_pvel)
                            self.svelocity.append(new_pvel)
                            self.time.append(self.time_step)
                            self.point = save_point
                            save_point = new_point
                            pvel = new_pvel.copy()
                            # new_point = idx.p2c(new_point)  # Move point obtained to c-space
                # If the point is not none; continue c-space
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
                        self.time_step = 2 * self.time_step
                        loop_check += 1
                        if loop_check == 70:
                            print('Stuck in the same loop for too long. Integration ends!')
                            break
                    elif self.angle_btw(save_point - self.point, new_pvel) <= 0.1 * self.adaptivity \
                            and self.time_step <= self.max_time_step:
                        self.point = save_point
                        save_point = idx.c2p(new_point)
                        self.streamline.append(save_point)
                        self.fvelocity.append(new_pvel)
                        self.svelocity.append(new_pvel)
                        self.time.append(self.time_step)
                        idx.point = new_point
                        pvel = new_pvel.copy()
                        self.time_step = 2 * self.time_step
                        loop_check = 0
                    elif self.angle_btw(save_point - self.point, new_pvel) >= self.adaptivity and\
                            self.time_step >= 1e-12:
                        self.time_step = 0.5 * self.time_step
                    else:
                        self.point = save_point
                        save_point = idx.c2p(new_point)
                        self.streamline.append(save_point)
                        self.fvelocity.append(new_pvel)
                        self.svelocity.append(new_pvel)
                        self.time.append(self.time_step)
                        idx.point = new_point
                        pvel = new_pvel.copy()
                        loop_check = 0
            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            t.stop()

        if method == 'ppath':
            t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            t.start()
            vel = None
            fvel = None
            while True:
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
                self.time.append(self.time_step)
                self.point = new_point
                vel = new_vel.copy()
                fvel = new_fvel.copy()

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            t.stop()

        if method == 'adaptive-ppath':
            t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            t.start()
            # particle velocity
            vel = None
            # fluid velocity
            fvel = None
            while True:
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
                    # print('Checking if the end of the domain is reached...')
                    self.time_step = 1e-9 * self.time_step
                    new_point, new_vel, new_fvel = intg.compute_ppath(diameter=self.diameter, density=self.density,
                                                                      viscosity=self.viscosity, velocity=vel,
                                                                      method='pRK4', time_step=self.time_step,
                                                                      drag_model=self.drag_model)
                    if new_point is not None:
                        # print('Continuing integration by decreasing time-step!')
                        continue
                    elif new_point is None:
                        print('Integration complete!')
                        break

                # Check for mid-rk4 blowup
                if intg.rk4_bool is True:
                    print(f'**WARNING** Large residual. Mid-RK4 blow up! Reducing time-step for particle number'
                          f' - {self.task}')
                    intg.rk4_bool = False
                    self.time_step = 0.5 * self.time_step
                    loop_check += 1
                    if loop_check == 70:
                        print('Stuck in the same loop for too long. Integration ends!')
                        break

                # Adaptive algorithm starts
                # Save results and continue the loop
                elif vel is None:
                    # For the first step in the loop
                    self.streamline.append(new_point)
                    self.svelocity.append(new_vel)
                    self.fvelocity.append(new_fvel)
                    self.time.append(self.time_step)
                    self.point = new_point
                    vel = new_vel.copy()
                    fvel = new_fvel.copy()
                # Check for if the points are identical because of tiny time step and deflection
                elif self.angle_btw(new_vel, vel) is None:
                    # print('Increasing time step. Successive points are same')
                    self.time_step = 2 * self.time_step
                    loop_check += 1
                    if loop_check == 70:
                        print(f'Successive points did not change for too long. Integration ends! for particle '
                              f'{self.task}')
                        break
                # Check for strong acceleration and reduce time-step
                # Increase time step when angle is below 0.05 degrees
                elif self.angle_btw(new_vel, vel) <= 0.1 * self.adaptivity and self.time_step <= self.max_time_step \
                        and self._magnitude(self, new_vel, vel) <= 0.01 * self.magnitude_adaptivity:
                    # print('Increasing time step. Low deflection wrt velocity')
                    self.streamline.append(new_point)
                    self.svelocity.append(new_vel)
                    self.fvelocity.append(new_fvel)
                    self.time.append(self.time_step)
                    self.point = new_point
                    vel = new_vel.copy()
                    fvel = new_fvel.copy()
                    self.time_step = 2 * self.time_step
                    loop_check = 0
                # Decrease time step when angle is above 1.4 degrees
                # Make sure time step does not go to zero; 1 pico-second
                elif self.angle_btw(new_vel, vel) >= self.adaptivity and self.time_step >= 1e-12 \
                        and self._magnitude(self, new_vel, vel) >= 0.1 * self.magnitude_adaptivity:
                    # print('Decreasing time step. High deflection wrt velocity')
                    self.time_step = 0.5 * self.time_step
                # Save if none of the above conditions meet
                else:
                    self.streamline.append(new_point)
                    self.svelocity.append(new_vel)
                    self.fvelocity.append(new_fvel)
                    self.time.append(self.time_step)
                    self.point = new_point
                    vel = new_vel.copy()
                    fvel = new_fvel.copy()
                    loop_check = 0

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            t.stop()

        if method == 'ppath-c-space':
            t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            t.start()
            pvel = None
            fvel = None
            save_point = self.point
            idx = Search(grid, self.point)
            idx.compute(method='c-space')
            while True:
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
                    idx.compute(method='p-space')
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
                        self.time.append(self.time_step)
                        pvel = new_pvel.copy()
                        fvel = new_fvel.copy()
                        self.point = save_point
                        save_point = new_point
                else:

                    # Check for mid-rk4 blowup
                    if intg.rk4_bool is True:
                        print('Mid-RK4 blow up! Reducing time-step')
                        intg.rk4_bool = False
                        self.time_step = 0.5 * self.time_step
                        loop_check += 1
                        if loop_check == 70:
                            print('Stuck in the same loop for too long. Integration ends!')
                            break

                    else:
                        self.point = save_point
                        save_point = idx.c2p(new_point)
                        self.streamline.append(save_point)
                        self.fvelocity.append(new_fvel)
                        self.svelocity.append(new_pvel)
                        self.time.append(self.time_step)
                        pvel = new_pvel.copy()
                        fvel = new_fvel.copy()
                        idx.point = new_point
                        if loop_check > 0:
                            print('Resetting time step')
                            self.time_step = self.time_step / 0.5 ** loop_check
                            loop_check = 0

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            t.stop()

        if method == 'adaptive-ppath-c-space':
            t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            t.start()
            pvel = None
            fvel = None
            save_point = self.point
            idx = Search(grid, self.point)
            idx.compute(method='c-space')
            while True:
                interp = Interpolation(flow, idx)
                interp.compute(method='c-space')
                intg = Integration(interp)
                new_point, new_fvel, new_pvel = intg.compute_ppath(diameter=self.diameter,
                                                                   density=self.density,
                                                                   viscosity=self.viscosity,
                                                                   velocity=pvel, method='cRK4',
                                                                   time_step=self.time_step,
                                                                   drag_model=self.drag_model)
                # Check if the point is out-of-block
                if new_point is None:
                    # Check for large time-step
                    print('Checking if the end of the domain is reached...')
                    self.time_step = 1e-9 * self.time_step
                    new_point, new_fvel, new_pvel = intg.compute_ppath(diameter=self.diameter,
                                                                       density=self.density,
                                                                       viscosity=self.viscosity,
                                                                       velocity=pvel, method='cRK4',
                                                                       time_step=self.time_step,
                                                                       drag_model=self.drag_model)
                    if new_point is not None:
                        print('Continuing integration by decreasing time-step!')
                        continue
                    # Check for out-of-block situation
                    # For multi-block case if the point is out-of-block
                    # Use previous point and run one-step of p-space algo
                    elif new_point is None:
                        print('Point exited the block! Searching for new position...')
                        idx = Search(grid, save_point)
                        interp = Interpolation(flow, idx)
                        intg = Integration(interp)
                        idx.compute(method='p-space')
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
                            self.time.append(self.time_step)
                            pvel = new_pvel.copy()
                            fvel = new_fvel.copy()
                            self.point = save_point
                            save_point = new_point
                else:
                    self.point = save_point
                    save_point = idx.c2p(new_point)

                    # Check for mid-rk4 blowup
                    if intg.rk4_bool is True:
                        print(f'**WARNING** Large residual. Mid-RK4 blow up! Reducing time-step for particle number'
                              f' - {self.task}')
                        intg.rk4_bool = False
                        self.time_step = 0.5 * self.time_step
                        loop_check += 1
                        if loop_check == 70:
                            print('Stuck in the same loop for too long. Integration ends!')
                            break

                    # Adaptive algorithm starts
                    # Save results and adjust time-step
                    # Details for the algorithm are provided in adaptive-ppath
                    elif pvel is None:
                        # For the first step in the loop
                        self.streamline.append(save_point)
                        self.svelocity.append(new_pvel)
                        self.fvelocity.append(new_fvel)
                        self.time.append(self.time_step)
                        self.point = new_point
                        pvel = new_pvel.copy()
                        fvel = new_fvel.copy()
                    elif self.angle_btw(new_fvel, fvel) is None:
                        print('Increasing time step. Successive points are same')
                        self.time_step = 2 * self.time_step
                        loop_check += 1
                        if loop_check == 70:
                            print(f'Successive points did not change for too long. Integration ends! for particle '
                                  f'{self.task}')
                            break
                    elif self.angle_btw(new_fvel,
                                        fvel) <= 0.1 * self.adaptivity and self.time_step <= self.max_time_step:
                        self.point = save_point
                        save_point = idx.c2p(new_point)
                        self.streamline.append(save_point)
                        self.fvelocity.append(new_fvel)
                        self.svelocity.append(new_pvel)
                        self.time.append(self.time_step)
                        idx.point = new_point.copy()
                        pvel = new_pvel.copy()
                        fvel = new_fvel.copy()
                        self.time_step = 2 * self.time_step
                        loop_check = 0
                    elif self.angle_btw(new_fvel, fvel) >= self.adaptivity and self.time_step >= 1e-12:
                        self.time_step = 0.5 * self.time_step
                    else:
                        self.point = save_point
                        save_point = idx.c2p(new_point)
                        self.streamline.append(save_point)
                        self.fvelocity.append(new_fvel)
                        self.svelocity.append(new_pvel)
                        self.time.append(self.time_step)
                        idx.point = new_point.copy()
                        pvel = new_pvel.copy()
                        fvel = new_fvel.copy()
                        loop_check = 0

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            t.stop()

        if method == 'unsteady-p-space':
            import glob
            import os
            t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            t.start()
            while True:
                # break the loop if the loop exceeds length of flow files
                if len(self.streamline) == len(flow.unsteady_flow):
                    print('Integration complete!')
                    break
                idx = Search(grid, self.point)
                # Skip the first object from _flowfiles and change the flow object with every iteration in interp
                interp = Interpolation(flow.unsteady_flow[len(self.streamline)-1], idx)
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
                self.time.append(self.time_step)
                self.point = new_point

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            t.stop()

        if method == 'unsteady-ppath':
            t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            t.start()
            vel = None
            fvel = None
            while True:
                # break the loop if the loop exceeds length of flow files
                if len(self.streamline) == len(flow.unsteady_flow):
                    print('Integration complete!')
                    break
                idx = Search(grid, self.point)
                interp = Interpolation(flow.unsteady_flow[len(self.streamline)-1], idx)
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
                self.time.append(self.time_step)
                self.point = new_point
                vel = new_vel.copy()
                fvel = new_fvel.copy()

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            t.stop()

        return
