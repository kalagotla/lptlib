# Uses the program API to extract streamlines
import logging
import numpy as np
from ..function.timer import Timer
from ..io.plot3dio import GridIO
from ..io.plot3dio import FlowIO
from ..streamlines.search import Search
from ..streamlines.interpolation import Interpolation
from ..streamlines.integration import Integration
from ..function.variables import Variables
import matplotlib.pyplot as plt
import functools

logger = logging.getLogger(__name__)


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
                 diameter=1e-7, density=1000,
                 time_step=1e-3, max_time_step=1, drag_model='stokes', adaptivity=0.001,
                 magnitude_adaptivity=0.001, filepath=None, task=None, debug=False):
        self.grid_file = grid_file
        self.flow_file = flow_file
        self.point = np.array(point)
        self.search = search
        self.interpolation = interpolation
        self.adaptive_interpolation = None
        self.integration = integration
        self.diameter = diameter
        self.density = density
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
        self.debug = debug
        self.max_loop_check = 70
        # Hard cap on the number of stored trajectory points. Honoured by every
        # algorithm, so a runaway trajectory is bounded whichever one is used.
        # None means "no cap"; the trajectory then ends when the point leaves
        # the domain or an algorithm-specific guard fires.
        self.max_steps = None
        self._initial_vel_magnitude = None
        # Book-keeping for time-step reductions forced by the mid-RK4 blow-up
        # guard, so the step can grow back once steps start succeeding again.
        self._blowup_halvings = 0
        self._blowup_ceiling = None
        # Newton-Raphson warm start carried along this particle's trajectory.
        # Held per Streamlines instance -- one instance per particle -- so it is
        # never shared between particles or threads.
        self._last_search = None
        # Live-plot options
        self.show_velocity_contour = False
        # unsteady parameters
        # Shaped based on the interp class q attribute
        self.flow_old = None

    # TODO: Need to add doc for streamlines

    @staticmethod
    def angle_btw(v1, v2):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 <= 1e-9 or n2 <= 1e-9:
            return None

        u1 = v1 / n1
        u2 = v2 / n2

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
            _data_save = np.hstack((p_xdata, vdata, udata, tdata, f_xdata,
                                    np.ones(tdata.shape) * self.diameter,
                                    np.ones(tdata.shape) * self.density))

            # help closes the file after writing
            with open(self.filepath + 'ppath_' + str(self.task) + '.npy', 'wb') as f:
                np.save(f, _data_save)
            self.print_debug(self, '** SUCCESS ** Done writing file for particle number - ' + str(
                self.task) + ' ** SUCCESS **')
            # set self to None to clear up memory after saving required data
            self.streamline = []
            self.svelocity = []
            self.fvelocity = []
            self.time = []
        return

    def _new_search(self, grid, point):
        """Build a Search seeded with this trajectory's previous c-space guess.

        Every step of a tracking loop creates a fresh Search. Without a warm
        start each one falls back to a nearest-node scan over the whole block,
        which dominates the run time on large grids. Seeding from the previous
        search on *this* instance keeps that optimisation while leaving the
        guess private to this particle -- the old module-global cache was
        shared by every thread and every particle.
        """
        _warm = self._last_search._cpoint if self._last_search is not None else None
        _idx = Search(grid, point, warm_start=_warm)
        self._last_search = _idx
        return _idx

    def _step_cap_reached(self, location=None):
        """Report whether ``max_steps`` has been hit, warning once if it has.

        Called at the top of every integration loop. Before this existed only
        ``adaptive-ppath`` looked at ``max_steps``, so a runaway trajectory
        under any other algorithm was unbounded.
        """
        if self.max_steps is None or len(self.streamline) < self.max_steps:
            return False
        _where = self.point if location is None else location
        logger.warning('Particle %s: maximum step count (%s) reached at %s. '
                       'Stopping integration.',
                       self.task, self.max_steps, np.asarray(_where).tolist())
        return True

    def _reduce_step_after_blowup(self):
        """Halve the time step after the mid-RK4 blow-up guard rejected a step.

        Remembers the step size in force before the first reduction so that
        :meth:`_restore_step_after_blowup` can grow it back. Without that a
        single discontinuity crossing permanently crippled the rest of the
        trajectory: every later step ran at the reduced size for ever.
        """
        if self._blowup_halvings == 0:
            self._blowup_ceiling = self.time_step
        self.time_step = 0.5 * self.time_step
        self._blowup_halvings += 1

    def _restore_step_after_blowup(self):
        """Grow the time step back toward its pre-blow-up value after a success.

        One doubling per successful step, never past the size in force before
        the first blow-up reduction, and never a reduction: an algorithm that
        has already grown the step past that ceiling on its own keeps it.
        """
        if self._blowup_halvings > 0:
            self.time_step = max(self.time_step,
                                 min(2.0 * self.time_step, self._blowup_ceiling))
            self._blowup_halvings -= 1

    @staticmethod
    def _magnitude(self, v1, v2):
        # return the magnitude of the vector
        _r = np.linalg.norm(v1 - v2) / min(np.linalg.norm(v1), np.linalg.norm(v2))
        if _r < 1e-6 * self.magnitude_adaptivity:
            _r = 1e-6 * self.magnitude_adaptivity
        return _r

    @staticmethod
    def plot_live(func):
        # Interactive mode for jupyter notebook; preserves zoom/pan across frames
        _saved_limits = []
        _first_call = [True]

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            fig = plt.gcf()
            axes = fig.get_axes()
            # Save current axis limits so zoom/pan are preserved (skip on first call)
            if not _first_call[0] and len(_saved_limits) >= len(axes):
                for i, axis in enumerate(axes):
                    try:
                        _saved_limits[i] = (axis.get_xlim(), axis.get_ylim())
                    except (AttributeError, IndexError, ValueError) as _err:
                        logger.debug('Could not save axis limits for axis %d: %s', i, _err)
            else:
                _saved_limits[:] = [(ax.get_xlim(), ax.get_ylim()) for ax in axes]

            # Clear all axes in the current figure
            for axis in axes:
                axis.cla()

            # Call func to plot something
            result = func(*args, **kwargs)

            # Restore axis limits so user zoom/pan is preserved (skip on first call)
            if len(_saved_limits) >= len(axes) and not _first_call[0]:
                for i, axis in enumerate(axes):
                    if i < len(_saved_limits) and _saved_limits[i] is not None:
                        xlim, ylim = _saved_limits[i]
                        # For the first axis (velocity vs x), let both x- and y-limits
                        # auto-update with data. For other axes, restore both.
                        if i == 0:
                            continue
                        axis.set_xlim(xlim)
                        axis.set_ylim(ylim)

            plt.draw()
            plt.pause(0.001)
            _first_call[0] = False  # after first frame, allow save/restore of limits

            return result

        return new_func

    @plot_live
    def plot_update(self, ax_vel, ax_pos, *args, **kwargs):
        grid = kwargs.pop('grid', None)
        flow = kwargs.pop('flow', None)

        # Optional velocity-magnitude contour overlay on the path subplot
        show_contour = getattr(self, 'show_velocity_contour', False)

        # If contour is turned off, remove any existing contour and colorbar once
        if not show_contour:
            if hasattr(self, "_vel_cs") and self._vel_cs is not None:
                try:
                    for coll in self._vel_cs.collections:
                        coll.remove()
                except (AttributeError, ValueError, NotImplementedError) as _err:
                    logger.debug('Could not remove contour collections: %s', _err)
                self._vel_cs = None
            if hasattr(self, "_vel_cbar") and self._vel_cbar is not None:
                try:
                    self._vel_cbar.remove()
                except (AttributeError, ValueError, KeyError) as _err:
                    logger.debug('Could not remove colorbar: %s', _err)
                self._vel_cbar = None
        elif (flow is not None) and (grid is not None) and hasattr(flow, 'q'):
            try:
                # Use first block and a representative k-plane (mid-plane)
                b = 0
                ni, nj, nk = grid.ni[b], grid.nj[b], grid.nk[b]
                k0 = nk // 2
                # Flow data: q[..., 0:4] = [rho, rho*u, rho*v, rho*w]
                q_slice = flow.q[0:ni, 0:nj, k0, 0:4, b]
                # q_slice has shape (ni, nj, 4): [rho, rho*u, rho*v, rho*w]
                rho = q_slice[..., 0]
                u = q_slice[..., 1] / rho
                v = q_slice[..., 2] / rho
                w = q_slice[..., 3] / rho
                vel_mag = np.sqrt(u**2 + v**2 + w**2)

                x = grid.grd[0:ni, 0:nj, k0, 0, b]
                y = grid.grd[0:ni, 0:nj, k0, 1, b]

                fig = ax_pos.figure

                # Remove old pcolormesh if it exists
                if hasattr(self, "_vel_pmesh") and self._vel_pmesh is not None:
                    try:
                        self._vel_pmesh.remove()
                    except (AttributeError, ValueError, NotImplementedError) as _err:
                        logger.debug('Could not remove velocity pcolormesh: %s', _err)

                # Cell-centred velocity magnitude for smoother shading
                vel_cell = 0.25 * (
                    vel_mag[:-1, :-1]
                    + vel_mag[1:, :-1]
                    + vel_mag[:-1, 1:]
                    + vel_mag[1:, 1:]
                )

                vmin_data = float(np.nanmin(vel_cell))
                vmax_data = float(np.nanmax(vel_cell))
                if not hasattr(self, "_vel_vmin_data") or not hasattr(self, "_vel_vmax_data"):
                    self._vel_vmin_data = vmin_data
                    self._vel_vmax_data = vmax_data
                if not hasattr(self, "_vel_scale"):
                    self._vel_scale = 1.0
                mid = 0.5 * (self._vel_vmin_data + self._vel_vmax_data)
                half_span = 0.5 * (self._vel_vmax_data - self._vel_vmin_data) / max(self._vel_scale, 1e-6)
                vmin = mid - half_span
                vmax = mid + half_span

                import matplotlib.pyplot as plt
                from matplotlib.colors import Normalize

                self._vel_pmesh = ax_pos.pcolormesh(
                    x,
                    y,
                    vel_cell,
                    cmap="viridis",
                    norm=Normalize(vmin=vmin, vmax=vmax),
                    shading="flat",
                    rasterized=True,
                    zorder=0,
                )

                if not hasattr(self, "_vel_cbar") or self._vel_cbar is None:
                    cbar = fig.colorbar(self._vel_pmesh, ax=ax_pos)
                    cbar.set_label("Velocity magnitude")
                    self._vel_cbar = cbar
                else:
                    self._vel_cbar.mappable = self._vel_pmesh
                    self._vel_cbar.mappable.set_clim(vmin, vmax)
                    self._vel_cbar.update_normal(self._vel_cbar.mappable)
            except Exception as _err:
                # Skip the contour overlay if anything about the data is
                # inconsistent -- this is a live-plot nicety, never load bearing.
                logger.debug('Skipping velocity contour overlay: %s: %s',
                             type(_err).__name__, _err)

        # Bottom subplot: grid outline and particle/fluid position on top of any background
        if grid is not None and grid.grd is not None:
            for b in range(grid.nb):
                ni, nj, nk = grid.ni[b], grid.nj[b], grid.nk[b]
                k0 = 0
                x_01 = grid.grd[0:ni, 0, k0, 0, b]
                y_01 = grid.grd[0:ni, 0, k0, 1, b]
                x_12 = grid.grd[ni - 1, 0:nj, k0, 0, b]
                y_12 = grid.grd[ni - 1, 0:nj, k0, 1, b]
                x_23 = grid.grd[ni - 1::-1, nj - 1, k0, 0, b]
                y_23 = grid.grd[ni - 1::-1, nj - 1, k0, 1, b]
                x_34 = grid.grd[0, nj - 1::-1, k0, 0, b]
                y_34 = grid.grd[0, nj - 1::-1, k0, 1, b]
                x_b = np.concatenate([x_01, x_12, x_23, x_34])
                y_b = np.concatenate([y_01, y_12, y_23, y_34])
                ax_pos.plot(x_b, y_b, 'k-', linewidth=0.8, zorder=1)

        # Particle trajectory over the grid
        if hasattr(self, 'streamline') and len(self.streamline) > 0:
            pts = np.asarray(self.streamline)
            if pts.shape[1] >= 2:
                xp = pts[:, 0]
                yp = pts[:, 1]
                ax_pos.plot(xp, yp, 'b-', zorder=2)
                ax_pos.plot(xp[-1], yp[-1], 'bo', label='particle', zorder=3)
                ax_pos.plot(xp[-1], yp[-1], 'rx', label='fluid', zorder=3)

        ax_pos.set_xlabel('x')
        ax_pos.set_ylabel('y')
        try:
            ax_pos.set_aspect('equal', adjustable='box')
        except (NotImplementedError, ValueError) as _err:
            logger.debug('Could not set equal aspect on the position axis: %s', _err)
        # Hint for interactive control
        ax_pos.text(0.01, 0.99, "c: contour, [: narrow, ]: widen",
                    transform=ax_pos.transAxes, ha='left', va='top',
                    fontsize=8, color='0.3')

        # Top subplot: velocity vs x (existing behavior)
        colors = ['b', 'g', 'c', 'm', 'y', 'k']
        for i, (x, y) in enumerate(zip(args[::2], args[1::2])):
            color = colors[i % len(colors)]
            ax_vel.plot(x, y, color)
            ax_vel.plot(x[-5:], y[-5:], 'r-')
            ax_vel.plot(x[-1], y[-1], 'ro')

        ax_vel.set_title(kwargs.get('title', ''))
        ax_vel.set_xlabel('x')
        ax_vel.set_ylabel('u_x')

        return

    @staticmethod
    def print_debug(self, statement):
        if self.debug is True:
            logger.debug(statement)
        return

    def compute(self, method='p-space', grid=None, flow=None):
        """
        Method to compute particle paths. Contains multiple algorithms
        Args:
            method:
            grid:
            flow:

        Returns:

        """
        # Start each trajectory from a cold Newton-Raphson guess so a repeated
        # compute() on the same object is reproducible.
        self._last_search = None
        self._blowup_halvings = 0
        self._blowup_ceiling = None

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
        idx = self._new_search(grid, self.point)
        interp = Interpolation(flow, idx)
        interp.adaptive = self.adaptive_interpolation
        idx.compute(method=self.search)
        interp.compute(method=self.interpolation)
        # If the initial point ended up out-of-domain (e.g. NR search failure),
        # stop cleanly instead of raising inside Variables.
        if interp.q is None:
            self.print_debug(self, 'Initial point is out of domain. Aborting integration.')
            return
        q_interp = Variables(interp)
        q_interp.compute_velocity()
        uf = q_interp.velocity.reshape(3)
        vel = uf.copy()
        pvel = uf.copy()
        fvel = uf.copy()
        self.fvelocity.append(uf)
        self.svelocity.append(uf)
        self.time.append(self.time_step)
        loop_check = 0
        # Check for step back to accommodate better dataio interpolation
        step_back = False
        pre_stepback_time_step = self.time_step

        if method == 'p-space':
            # t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            # t.start()
            while True:
                if self._step_cap_reached():
                    break
                idx = self._new_search(grid, self.point)
                interp = Interpolation(flow, idx)
                interp.adaptive = self.adaptive_interpolation
                intg = Integration(interp)
                idx.compute(method=self.search)
                interp.compute(method=self.interpolation)
                new_point, new_vel = intg.compute(method=self.integration, time_step=self.time_step)
                if new_point is None:
                    self.print_debug(self, 'Integration complete!')
                    break
                self.streamline.append(new_point)
                self.fvelocity.append(new_vel)
                self.svelocity.append(new_vel)
                self.time.append(self.time_step)
                self.point = new_point

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            # t.stop()

        if method == 'adaptive-p-space':
            # t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            # t.start()
            while True:
                if self._step_cap_reached():
                    break
                idx = self._new_search(grid, self.point)
                interp = Interpolation(flow, idx)
                interp.adaptive = self.adaptive_interpolation
                intg = Integration(interp)
                idx.compute(method=self.search)
                interp.compute(method=self.interpolation)
                new_point, new_vel = intg.compute(method=self.integration, time_step=self.time_step)
                if new_point is None:
                    self.print_debug(self, 'Checking if the end of the domain is reached...')
                    self.time_step = 1e-9 * self.time_step
                    new_point, new_vel = intg.compute(method=self.integration, time_step=self.time_step)
                    if new_point is not None:
                        self.print_debug(self, 'Continuing integration by decreasing time-step!')
                        continue
                    elif new_point is None:
                        self.print_debug(self, 'Integration complete!')
                        break

                # Adaptive algorithm starts
                # Save results and adjust time-step
                # Details for the algorithm are provided in adaptive-ppath
                if self.angle_btw(new_point - self.point, vel) is None:
                    self.print_debug(self, 'Increasing time step. Successive points are same')
                    self.time_step = 10 * self.time_step
                    loop_check += 1
                    if loop_check == 70:
                        self.print_debug(self, 'Stuck in the same loop for too long. Integration ends!')
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
            # t.stop()

        if method == 'c-space':
            # Use c-space search to convert and find the location of given point
            # All the idx attributes are converted to c-space -- point, cell, block
            # t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            # t.start()
            save_point = self.point
            idx = self._new_search(grid, self.point)
            idx.compute(method='c-space')
            while True:
                if self._step_cap_reached(save_point):
                    break
                interp = Interpolation(flow, idx)
                interp.adaptive = self.adaptive_interpolation
                interp.compute(method='c-space')
                intg = Integration(interp)
                new_point, new_pvel, new_cvel = intg.compute(method='cRK4', time_step=self.time_step)
                if new_point is None:
                    # For multi-block case if the point is out-of-block
                    # Use previous point and run one-step of p-space algo
                    self.print_debug(self, 'Point exited the block! Searching for new position...')
                    idx = self._new_search(grid, save_point)
                    interp = Interpolation(flow, idx)
                    interp.adaptive = self.adaptive_interpolation
                    intg = Integration(interp)
                    idx.compute(method='block_distance')
                    interp.compute()
                    new_point, new_pvel = intg.compute(method='pRK4', time_step=self.time_step)
                    if new_point is None:
                        self.print_debug(self, 'Point out-of-domain. Integration complete!')
                        break
                    else:
                        # Update the block in idx
                        # new_point is in p-space for this else block

                        idx = self._new_search(grid, new_point)
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
            # t.stop()

        if method == 'adaptive-c-space':
            # Use c-space search to convert and find the location of given point
            # All the idx attributes are converted to c-space -- point, cell, block
            # pvel signifies physical-space velocity
            # t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            # t.start()
            save_point = self.point
            idx = self._new_search(grid, self.point)
            idx.compute(method='c-space')
            while True:
                if self._step_cap_reached(save_point):
                    break
                interp = Interpolation(flow, idx)
                interp.compute(method='c-space')
                interp.adaptive = self.adaptive_interpolation
                intg = Integration(interp)
                new_point, new_pvel, new_cvel = intg.compute(method='cRK4', time_step=self.time_step)

                # Check for large time-step in the last loop
                if new_point is None:
                    # Checking by decreasing time-step
                    self.print_debug(self, 'Checking if the end of the domain is reached...')
                    self.time_step = 1e-9 * self.time_step
                    new_point, new_pvel, new_cvel = intg.compute(method='cRK4', time_step=self.time_step)
                    if new_point is not None:
                        self.print_debug(self, 'Continuing integration by decreasing time-step!')
                        continue
                    # Check for block switch
                    elif new_point is None:
                        # For multi-block case if the point is out-of-block
                        # Use previous point and run one-step of p-space algo
                        self.print_debug(self, 'Point exited the block! Searching for new position...')
                        idx = self._new_search(grid, save_point)
                        interp = Interpolation(flow, idx)
                        interp.adaptive = self.adaptive_interpolation
                        intg = Integration(interp)
                        idx.compute(method='block_distance')
                        interp.compute(method='p-space')
                        new_point, new_pvel = intg.compute(method='pRK4', time_step=self.time_step)
                        # Even after pRK4 if the point is None; End integration
                        if new_point is None:
                            self.print_debug(self, 'End of the domain. Integration Ends!')
                            break
                        # If not none; update to c-space and run
                        else:
                            # Update the block in idx
                            idx = self._new_search(grid, new_point)
                            idx.compute(method='c-space')
                            self.streamline.append(new_point)
                            self.fvelocity.append(new_pvel)
                            self.svelocity.append(new_pvel)
                            self.time.append(self.time_step)
                            save_point = new_point
                            pvel = new_pvel.copy()
                            # new_point = idx.p2c(new_point)  # Move point obtained to c-space
                # If the point is not none; continue c-space
                else:
                    old_ppoint = save_point
                    # This changes the cell attribute in idx, used for interp
                    new_ppoint = idx.c2p(new_point)

                    # Adaptive algorithm starts
                    # Save results and adjust time-step
                    # Details for the algorithm are provided in adaptive-ppath
                    if self.angle_btw(new_ppoint - old_ppoint, pvel) is None:
                        self.print_debug(self, 'Increasing time step. Successive points are same')
                        self.time_step = 2 * self.time_step
                        loop_check += 1
                        if loop_check == 70:
                            self.print_debug(self, 'Stuck in the same loop for too long. Integration ends!')
                            break
                    elif self.angle_btw(new_ppoint - old_ppoint, pvel) <= 0.1 * self.adaptivity \
                            and self.time_step <= self.max_time_step:
                        save_point = new_ppoint
                        self.streamline.append(save_point)
                        self.fvelocity.append(new_pvel)
                        self.svelocity.append(new_pvel)
                        self.time.append(self.time_step)
                        idx.point = new_point
                        pvel = new_pvel.copy()
                        self.time_step = 2 * self.time_step
                        loop_check = 0
                    elif self.angle_btw(new_ppoint - old_ppoint, pvel) >= self.adaptivity and \
                            self.time_step >= 1e-12:
                        self.time_step = 0.5 * self.time_step
                    else:
                        save_point = new_ppoint
                        self.streamline.append(save_point)
                        self.fvelocity.append(new_pvel)
                        self.svelocity.append(new_pvel)
                        self.time.append(self.time_step)
                        idx.point = new_point
                        pvel = new_pvel.copy()
                        loop_check = 0
            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            # t.stop()

        if method == 'ppath':
            # t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            # t.start()
            vel = None
            fvel = None
            _blowup_retries = 0
            while True:
                if self._step_cap_reached():
                    break
                idx = self._new_search(grid, self.point)
                interp = Interpolation(flow, idx)
                intg = Integration(interp)
                interp.adaptive = self.adaptive_interpolation
                idx.compute(method=self.search)
                interp.compute(method=self.interpolation)
                new_point, new_vel, new_fvel = intg.compute_ppath(diameter=self.diameter,
                                                                  density=self.density,
                                                                  velocity=vel, method='pRK4',
                                                                  time_step=self.time_step,
                                                                  drag_model=self.drag_model)
                if new_point is None:
                    self.print_debug(self, 'Integration complete!')
                    break

                # The blow-up guard fired, so no step was taken: what came back
                # is the unchanged input state. Appending it and carrying on
                # stored a duplicate point and spun for ever on any particle
                # that could not cross a discontinuity. Retry with a halved
                # time step instead, a bounded number of times.
                if intg.rk4_bool is True:
                    intg.rk4_bool = False
                    _blowup_retries += 1
                    if _blowup_retries > self.max_loop_check:
                        logger.warning(
                            'Particle %s: mid-RK4 blow up at %s could not be resolved '
                            'after %d time-step reductions (time step now %g). '
                            'Ending this trajectory.',
                            self.task, np.asarray(self.point).tolist(),
                            _blowup_retries - 1, self.time_step)
                        break
                    self.print_debug(self, f'**WARNING** Large residual. Mid-RK4 blow up! '
                                           f'Reducing time-step for particle number - {self.task}')
                    self._reduce_step_after_blowup()
                    continue

                # Save results and continue the loop
                self.streamline.append(new_point)
                self.svelocity.append(new_vel)
                self.fvelocity.append(new_fvel)
                self.time.append(self.time_step)
                self.point = new_point
                vel = new_vel.copy()
                fvel = new_fvel.copy()
                _blowup_retries = 0
                self._restore_step_after_blowup()

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            # t.stop()

        if method == 'adaptive-ppath':
            # t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            # t.start()
            if self.debug:
                fig, (ax_vel, ax_pos) = plt.subplots(2, 1, figsize=(6, 8))
                fig.tight_layout()
                # Keyboard controls:
                #   'c'  -> toggle velocity contour on/off
                #   '['  -> narrow contour value range
                #   ']'  -> widen contour value range
                def _on_key(event, sl=self):
                    if event.key == 'c':
                        sl.show_velocity_contour = not getattr(sl, 'show_velocity_contour', False)
                    elif event.key == '[':
                        if not hasattr(sl, "_vel_scale"):
                            sl._vel_scale = 1.0
                        sl._vel_scale *= 1.2  # narrow range (more contrast)
                    elif event.key == ']':
                        if not hasattr(sl, "_vel_scale"):
                            sl._vel_scale = 1.0
                        sl._vel_scale /= 1.2  # widen range
                fig.canvas.mpl_connect('key_press_event', _on_key)
            _domain_exit_count = 0
            _vel_max = np.linalg.norm(vel) if vel is not None else 1.0  # Track max velocity for near-zero guard
            while True:
                _vel_max = max(_vel_max, np.linalg.norm(vel)) if vel is not None else _vel_max
                if self._step_cap_reached():
                    break
                idx = self._new_search(grid, self.point)
                interp = Interpolation(flow, idx)
                interp.adaptive = self.adaptive_interpolation
                intg = Integration(interp)
                idx.compute(method=self.search)
                interp.compute(method=self.interpolation)
                new_point, new_vel, new_fvel = intg.compute_ppath(diameter=self.diameter, density=self.density,
                                                                  velocity=vel,
                                                                  method='pRK4', time_step=self.time_step,
                                                                  drag_model=self.drag_model)
                if new_point is None:
                    self.print_debug(self, 'Checking if the end of the domain is reached...')
                    _domain_exit_count += 1
                    if self.time_step <= 1e-10 or _domain_exit_count >= 5:
                        self.print_debug(self, 'Integration complete!')
                        break
                    else:
                        self.time_step = 1e-2 * self.time_step
                    new_point, new_vel, new_fvel = intg.compute_ppath(diameter=self.diameter, density=self.density,
                                                                      velocity=vel,
                                                                      method='pRK4', time_step=self.time_step,
                                                                      drag_model=self.drag_model)
                    if new_point is not None:
                        self.print_debug(self, 'Continuing integration by decreasing time-step!')
                        continue
                    elif new_point is None:
                        self.print_debug(self, 'Integration complete!')
                        break

                # Check for mid-rk4 blowup
                if intg.rk4_bool is True:
                    self.print_debug(self,
                                     f'**WARNING** Large residual. Mid-RK4 blow up! Reducing time-step for particle '
                                     f'number'
                                     f' - {self.task}')
                    intg.rk4_bool = False
                    self._reduce_step_after_blowup()
                    loop_check += 1
                    if loop_check == self.max_loop_check:
                        logger.warning(
                            'Particle %s: mid-RK4 blow up at %s could not be resolved '
                            'after %d time-step reductions (time step now %g). '
                            'Ending this trajectory.',
                            self.task, np.asarray(self.point).tolist(),
                            loop_check, self.time_step)
                        break

                # Adaptive algorithm starts
                # Save results and continue the loop
                # Check for if the points are identical because of tiny time step and deflection
                elif self.angle_btw(new_vel, vel) is None:
                    self.print_debug(self, 'Increasing time step. Successive points are same')
                    self.time_step = 2 * self.time_step
                    loop_check += 1
                    if loop_check == 70:
                        self.print_debug(self, f'Successive points did not change for too long. Integration ends! for '
                                               f'particle'
                                               f'{self.task}')
                        break
                # Check for strong acceleration and reduce time-step
                # Increase time step when angle is below 0.05 degrees
                elif self.angle_btw(new_vel, vel) <= 0.1 * self.adaptivity \
                        and self._magnitude(self, new_vel, vel) <= 0.1 * self.magnitude_adaptivity \
                        and self.time_step <= self.max_time_step:
                    self.print_debug(self, 'Increasing time step. Low deflection wrt velocity')
                    self.streamline.append(new_point)
                    self.svelocity.append(new_vel)
                    self.fvelocity.append(new_fvel)
                    self.time.append(self.time_step)
                    self.point = new_point
                    vel = new_vel.copy()
                    fvel = new_fvel.copy()
                    self.time_step = 2 * self.time_step
                    loop_check = 0  # This check might lead to slower integration for some edge cases
                    self._restore_step_after_blowup()
                # Decrease time step when angle OR magnitude change is large
                # Make sure time step does not go to zero; 1 pico-second
                # Guard: skip decrease when velocity is near zero (e.g. separation zones)
                # because angle and magnitude ratio are ill-conditioned there.
                elif (self.angle_btw(new_vel, vel) >= self.adaptivity
                        or self._magnitude(self, new_vel, vel) >= self.magnitude_adaptivity) \
                        and self.time_step >= 1e-12 \
                        and min(np.linalg.norm(vel), np.linalg.norm(new_vel)) > 1e-3 * _vel_max:
                    self.print_debug(self, 'Decreasing time step. High deflection wrt velocity')
                    self.time_step = 0.5 * self.time_step
                    step_back = True
                # check the vector length just before entering an acceleration zone
                elif step_back:
                    self.print_debug(self, 'Step back to accommodate better dataio Delaunay triangulation')
                    saved_time_step = self.time_step
                    step_back_count = 0
                    _vel_norm = np.linalg.norm(vel)
                    _step_back_tol = max(1e-6 * _vel_norm, 1e-6 * _vel_max, 1e-12)
                    while np.linalg.norm(new_vel - vel) >= _step_back_tol:
                        self.time_step = 0.5 * self.time_step
                        new_point, new_vel, new_fvel = intg.compute_ppath(diameter=self.diameter, density=self.density,
                                                                          velocity=vel,
                                                                          method='pRK4', time_step=self.time_step,
                                                                          drag_model=self.drag_model)
                        step_back_count += 1
                        if step_back_count > 70:
                            break
                    # Save the results
                    self.streamline.append(new_point)
                    self.svelocity.append(new_vel)
                    self.fvelocity.append(new_fvel)
                    self.time.append(self.time_step)
                    self.point = new_point
                    vel = new_vel.copy()
                    fvel = new_fvel.copy()
                    loop_check = 0
                    # reset the time-step to the value before step-back
                    self.time_step = saved_time_step
                    self._restore_step_after_blowup()
                    # set step_back to False and continue the loop
                    step_back = False

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
                    self._restore_step_after_blowup()

                # Plot the streamline for debugging
                # add levels to debug; multiple ways of showing plots etc...
                # This will help when working with varying flow fields
                if self.debug:
                    self.plot_update(ax_vel, ax_pos,
                                     [i[0] for i in self.streamline], [i[0] for i in self.svelocity],
                                     [i[0] for i in self.streamline], [i[0] for i in self.fvelocity],
                                     title=f'Particle number - {self.task} and diameter - {self.diameter},\n'
                                           f' density - {self.density} and time-step - {self.time_step}',
                                     grid=grid,
                                     flow=flow)

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            # t.stop()

        if method == 'ppath-c-space':
            t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            t.start()
            save_point = self.point
            idx = self._new_search(grid, self.point)
            idx.compute(method='c-space')
            while True:
                if self._step_cap_reached(save_point):
                    break
                interp = Interpolation(flow, idx)
                interp.adaptive = self.adaptive_interpolation
                interp.compute(method='c-space')
                intg = Integration(interp)
                new_point, new_pvel, new_fvel = intg.compute_ppath(diameter=self.diameter,
                                                                   density=self.density,
                                                                   velocity=pvel, method='cRK4',
                                                                   time_step=self.time_step,
                                                                   drag_model=self.drag_model)
                if new_point is None:
                    # For multi-block case if the point is out-of-block
                    # Use previous point and run one-step of p-space algo
                    self.print_debug(self, 'Point exited the block! Searching for new position...')
                    idx = self._new_search(grid, save_point)
                    interp = Interpolation(flow, idx)
                    interp.adaptive = self.adaptive_interpolation
                    intg = Integration(interp)
                    idx.compute(method='p-space')
                    interp.compute(method='p-space')
                    new_point, new_pvel, new_fvel = intg.compute_ppath(diameter=self.diameter,
                                                                       density=self.density,
                                                                       velocity=pvel, method='pRK4',
                                                                       time_step=self.time_step,
                                                                       drag_model=self.drag_model)
                    if new_point is None:
                        self.print_debug(self, 'Point out-of-domain. Integration complete!')
                        break
                    else:
                        # Update the block in idx
                        idx = self._new_search(grid, new_point)
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
                        self.print_debug(self, 'Mid-RK4 blow up! Reducing time-step')
                        intg.rk4_bool = False
                        self.time_step = 0.5 * self.time_step
                        loop_check += 1
                        if loop_check == 70:
                            self.print_debug(self, 'Stuck in the same loop for too long. Integration ends!')
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
                            self.print_debug(self, 'Resetting time step')
                            self.time_step = self.time_step * (2 ** loop_check)
                            loop_check = 0

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            # t.stop()

        if method == 'adaptive-ppath-c-space':
            # t = Timer(text="Time taken for particle " + str(self.task) + " is {:.2f} seconds")
            # t.start()
            if self.debug:
                fig, (ax_vel, ax_pos) = plt.subplots(2, 1, figsize=(6, 8))
                fig.tight_layout()
                # Keyboard controls:
                #   'c'  -> toggle velocity contour on/off
                #   '['  -> narrow contour value range
                #   ']'  -> widen contour value range
                def _on_key(event, sl=self):
                    if event.key == 'c':
                        sl.show_velocity_contour = not getattr(sl, 'show_velocity_contour', False)
                    elif event.key == '[':
                        if not hasattr(sl, "_vel_scale"):
                            sl._vel_scale = 1.0
                        sl._vel_scale *= 1.2  # narrow range (more contrast)
                    elif event.key == ']':
                        if not hasattr(sl, "_vel_scale"):
                            sl._vel_scale = 1.0
                        sl._vel_scale /= 1.2  # widen range
                fig.canvas.mpl_connect('key_press_event', _on_key)
            save_point = self.point
            idx = self._new_search(grid, self.point)
            idx.compute(method='c-space')
            _resync_counter = 0
            _resync_interval = 10  # Resync c-space via p-space every N saved steps
            while True:
                if self._step_cap_reached(save_point):
                    break
                # Periodic p-space resync to correct accumulated Jacobian drift
                if _resync_counter >= _resync_interval:
                    idx = self._new_search(grid, save_point)
                    idx.compute(method='c-space')
                    _resync_counter = 0
                interp = Interpolation(flow, idx)
                interp.adaptive = self.adaptive_interpolation
                interp.compute(method='c-space')
                intg = Integration(interp)
                new_point, new_pvel, new_fvel = intg.compute_ppath(diameter=self.diameter,
                                                                   density=self.density,
                                                                   velocity=pvel, method='cRK4',
                                                                   time_step=self.time_step,
                                                                   drag_model=self.drag_model)
                # Check if the point is out-of-block
                if new_point is None:
                    # Check for large time-step
                    self.print_debug(self, 'Checking if the end of the domain is reached...')
                    if self.time_step <= 1e-12:
                        break
                    self.time_step = 1e-2 * self.time_step
                    new_point, new_pvel, new_fvel = intg.compute_ppath(diameter=self.diameter,
                                                                       density=self.density,
                                                                       velocity=pvel, method='cRK4',
                                                                       time_step=self.time_step,
                                                                       drag_model=self.drag_model)
                    if new_point is not None:
                        self.print_debug(self, 'Continuing integration by decreasing time-step!')
                        continue
                    # Check for out-of-block situation
                    # For multi-block case if the point is out-of-block
                    # Use previous point and run one-step of p-space algo
                    elif new_point is None:
                        self.print_debug(self, 'Point exited the block! Searching for new position...')
                        idx = self._new_search(grid, save_point)
                        interp = Interpolation(flow, idx)
                        interp.adaptive = self.adaptive_interpolation
                        intg = Integration(interp)
                        idx.compute(method='p-space')
                        interp.compute()
                        new_point, new_pvel, new_fvel = intg.compute_ppath(diameter=self.diameter,
                                                                           density=self.density,
                                                                           velocity=pvel, method='pRK4',
                                                                           time_step=self.time_step,
                                                                           drag_model=self.drag_model)
                        if new_point is None:
                            self.print_debug(self, 'Point out-of-domain. Integration complete!')
                            break
                        else:
                            # Update the block in idx
                            idx = self._new_search(grid, new_point)
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
                            _resync_counter = 0  # just did a p-space search
                else:
                    self.point = save_point
                    save_point = idx.c2p(new_point)

                    # Check for mid-rk4 blowup
                    if intg.rk4_bool is True:
                        self.print_debug(self,
                                         f'**WARNING** Large residual. Mid-RK4 blow up! Reducing time-step for particle number'
                                         f' - {self.task}')
                        intg.rk4_bool = False
                        self._reduce_step_after_blowup()
                        loop_check += 1
                        if loop_check == 70:
                            logger.warning(
                                'Particle %s: mid-RK4 blow up at %s could not be resolved '
                                'after %d time-step reductions (time step now %g). '
                                'Ending this trajectory.',
                                self.task, np.asarray(save_point).tolist(),
                                loop_check, self.time_step)
                            break

                    # Adaptive algorithm starts
                    # Save results and adjust time-step
                    # Details for the algorithm are provided in adaptive-ppath
                    elif self.angle_btw(new_pvel, pvel) is None:
                        self.print_debug(self, 'Increasing time step. Successive points are same')
                        self.time_step = 2 * self.time_step
                        loop_check += 1
                        if loop_check == 70:
                            self.print_debug(self,
                                             f'Successive points did not change for too long. Integration ends! for particle '
                                             f'{self.task}')
                            break
                    elif self.angle_btw(new_pvel, pvel) <= 0.1 * self.adaptivity \
                            and self._magnitude(self, new_pvel, pvel) <= 0.1 * self.magnitude_adaptivity \
                            and self.time_step <= self.max_time_step:
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
                        _resync_counter += 1
                        self._restore_step_after_blowup()
                    elif (self.angle_btw(new_pvel, pvel) >= self.adaptivity
                            or self._magnitude(self, new_pvel, pvel) >= self.magnitude_adaptivity) \
                            and self.time_step >= 1e-12:
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
                        _resync_counter += 1
                        self._restore_step_after_blowup()

                # Plot the streamline for debugging
                if self.debug:
                    self.plot_update(ax_vel, ax_pos,
                                     [i[0] for i in self.streamline], [i[0] for i in self.svelocity],
                                     [i[0] for i in self.streamline], [i[0] for i in self.fvelocity],
                                     title=f'Particle number - {self.task} and diameter - {self.diameter},\n'
                                           f' density - {self.density} and time-step - {self.time_step}',
                                     grid=grid,
                                     flow=flow)

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)
            # t.stop()

        if method == 'unsteady-p-space':
            # _temp is used to keep track of the flow object
            _temp = 0
            if self.debug:
                fig, (ax_vel, ax_pos) = plt.subplots(2, 1, figsize=(6, 8))
                fig.tight_layout()
                # Keyboard controls:
                #   'c'  -> toggle velocity contour on/off
                #   '['  -> narrow contour value range
                #   ']'  -> widen contour value range
                def _on_key(event, sl=self):
                    if event.key == 'c':
                        sl.show_velocity_contour = not getattr(sl, 'show_velocity_contour', False)
                    elif event.key == '[':
                        if not hasattr(sl, "_vel_scale"):
                            sl._vel_scale = 1.0
                        sl._vel_scale *= 1.2
                    elif event.key == ']':
                        if not hasattr(sl, "_vel_scale"):
                            sl._vel_scale = 1.0
                        sl._vel_scale /= 1.2
                fig.canvas.mpl_connect('key_press_event', _on_key)
            while True:
                if self._step_cap_reached():
                    break
                if (flow.unsteady_flow[_temp].time - np.sum(self.time)) < 0:
                    self.flow_old = flow.unsteady_flow[_temp]
                    _temp += 1
                if _temp == len(flow.unsteady_flow):
                    logger.info('Integration complete! for particle ' + str(self.task))
                    break
                idx = self._new_search(grid, self.point)
                # Skip the first object from _flowfiles and change the flow object with every iteration in interp
                interp = Interpolation(flow.unsteady_flow[_temp], idx)
                interp.time = self.time
                interp.flow_old = self.flow_old
                intg = Integration(interp)
                idx.compute(method=self.search)
                interp.compute(method=self.interpolation)
                new_point, new_vel = intg.compute(method='unsteady-pRK4', time_step=self.time_step)
                if new_point is None:
                    logger.info('Integration complete!')
                    break
                self.streamline.append(new_point)
                self.fvelocity.append(new_vel)
                self.svelocity.append(new_vel)
                self.time.append(self.time_step)
                self.point = new_point

                # Plot the streamline for debugging
                # add levels to debug; multiple ways of showing plots etc...
                # This will help when working with varying flow fields
                if self.debug:
                    self.plot_update(ax_vel, ax_pos,
                                     [i[0] for i in self.streamline], [i[0] for i in self.svelocity],
                                     [i[0] for i in self.streamline], [i[0] for i in self.fvelocity],
                                     title=f'Particle number - {self.task} and diameter - {self.diameter},\n'
                                           f' density - {self.density} and time-step - {self.time_step}',
                                     grid=grid,
                                     flow=flow)

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)

        if method == 'unsteady-ppath':
            vel = None
            fvel = None
            _blowup_retries = 0
            # _temp is used to keep track of the flow object
            _temp = 0
            if self.debug:
                fig, (ax_vel, ax_pos) = plt.subplots(2, 1, figsize=(6, 8))
                fig.tight_layout()
                # Keyboard controls:
                #   'c'  -> toggle velocity contour on/off
                #   '['  -> narrow contour value range
                #   ']'  -> widen contour value range
                def _on_key(event, sl=self):
                    if event.key == 'c':
                        sl.show_velocity_contour = not getattr(sl, 'show_velocity_contour', False)
                    elif event.key == '[':
                        if not hasattr(sl, "_vel_scale"):
                            sl._vel_scale = 1.0
                        sl._vel_scale *= 1.2
                    elif event.key == ']':
                        if not hasattr(sl, "_vel_scale"):
                            sl._vel_scale = 1.0
                        sl._vel_scale /= 1.2
                fig.canvas.mpl_connect('key_press_event', _on_key)
            while True:
                if self._step_cap_reached():
                    break
                if (flow.unsteady_flow[_temp].time - np.sum(self.time)) < 0:
                    self.flow_old = flow.unsteady_flow[_temp]
                    _temp += 1
                if _temp == len(flow.unsteady_flow):
                    logger.info('Integration complete! for particle ' + str(self.task))
                    break
                idx = self._new_search(grid, self.point)
                interp = Interpolation(flow.unsteady_flow[_temp], idx)
                interp.time = self.time
                interp.flow_old = self.flow_old
                intg = Integration(interp)
                idx.compute(method=self.search)
                interp.compute(method=self.interpolation)
                new_point, new_vel, new_fvel = intg.compute_ppath(diameter=self.diameter,
                                                                  density=self.density,
                                                                  velocity=vel, method='unsteady-pRK4',
                                                                  time_step=self.time_step,
                                                                  drag_model=self.drag_model)
                if new_point is None:
                    logger.info('Integration complete! Out of domain for particle ' + str(self.task))
                    break

                # Blow-up guard fired: no step was taken, so retry with a
                # halved time step rather than appending a duplicate point for
                # ever (the same defect that made 'ppath' non-terminating).
                if intg.rk4_bool is True:
                    intg.rk4_bool = False
                    _blowup_retries += 1
                    if _blowup_retries > self.max_loop_check:
                        logger.warning(
                            'Particle %s: mid-RK4 blow up at %s could not be resolved '
                            'after %d time-step reductions (time step now %g). '
                            'Ending this trajectory.',
                            self.task, np.asarray(self.point).tolist(),
                            _blowup_retries - 1, self.time_step)
                        break
                    self._reduce_step_after_blowup()
                    continue

                # Save results and continue the loop
                self.streamline.append(new_point)
                self.svelocity.append(new_vel)
                self.fvelocity.append(new_fvel)
                self.time.append(self.time_step)
                self.point = new_point
                vel = new_vel.copy()
                fvel = new_fvel.copy()
                _blowup_retries = 0
                self._restore_step_after_blowup()

                # Plot the streamline for debugging
                # add levels to debug; multiple ways of showing plots etc...
                # This will help when working with varying flow fields
                if self.debug:
                    self.plot_update(ax_vel, ax_pos,
                                     [i[0] for i in self.streamline], [i[0] for i in self.svelocity],
                                     [i[0] for i in self.streamline], [i[0] for i in self.fvelocity],
                                     title=f'Particle number - {self.task} and diameter - {self.diameter},\n'
                                           f' density - {self.density} and time-step - {self.time_step}',
                                     grid=grid,
                                     flow=flow.unsteady_flow[_temp])

            # Save files for each particle; can be used for multiprocessing large number of particles
            self._save_data(self)

        return
