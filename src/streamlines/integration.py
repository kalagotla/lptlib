# Integrate from given point to produce streamlines
import numpy as np
from src.streamlines.interpolation import Interpolation
from src.streamlines.search import Search
from src.function.variables import Variables


class Integration:

    def __init__(self, interp):
        self.interp = interp
        self.ppoint = None
        self.cpoint = None

    def __str__(self):
        doc = "This instance uses data from " + self.interp.flow.filename + \
              " and integrates based on the given time step"
        return doc

    def compute(self, method='p-space', time_step=1e-6):
        match method:
            case 'p-space':
                # For p-space algos; the point-in-domain check was done in search
                if self.interp.idx.ppoint is None:
                    self.ppoint = None
                    return self.ppoint

                # Compute required variables from plot3d data
                q_interp = Variables(self.interp)
                q_interp.compute_velocity()
                # Integration for one time step
                self.ppoint = self.interp.idx.ppoint + q_interp.velocity.reshape(3) * time_step

                return self.ppoint

            case 'c-space':
                # Get inverse Jacobian from the interpolation class
                # Using cell node data. For more accurate calculation refer to cRK4 method
                _J_inv = self.interp.idx.grid.m2[self.interp.idx.cell[0, 0], self.interp.idx.cell[0, 1],
                                                 self.interp.idx.cell[0, 2], :, :, self.interp.idx.block]

                q_interp = Variables(self.interp)
                q_interp.compute_velocity()
                p_velocity = q_interp.velocity.reshape(3)
                c_velocity = np.matmul(_J_inv, p_velocity)
                self.cpoint = self.interp.idx.cpoint + c_velocity * time_step

                # For c-space the point in-domain check is done after integration
                if not np.all([0, 0, 0] <= self.cpoint) or not np.all(
                        self.cpoint + 1 < [self.interp.idx.grid.ni[self.interp.idx.block],
                                           self.interp.idx.grid.nj[self.interp.idx.block],
                                           self.interp.idx.grid.nk[self.interp.idx.block]]):
                    self.cpoint = None
                    return self.cpoint

                return self.cpoint

            case 'pRK4':
                """
                This is a straight forward RK4 integration. Search for the point,
                Interpolate the data to the point, Compute required variables,
                Perform RK4 integration!
                """

                def _rk4_step(self, x):
                    """

                    Args:
                        self:
                        x: ndarray
                            point in p-space

                    Returns:
                        k: ndarray
                            interim RK4 variables

                    """
                    idx = Search(self.interp.idx.grid, x)
                    idx.compute(method='p-space')
                    # For p-space algos; the point-in-domain check was done in search
                    if idx.ppoint is None: return None, None
                    interp = Interpolation(self.interp.flow, idx)
                    interp.compute()
                    q_interp = Variables(interp)
                    q_interp.compute_velocity()
                    u = q_interp.velocity.reshape(3)
                    k = time_step * u
                    return u, k

                # Start RK4 for p-space
                # For p-space algos; the point-in-domain check was done in search
                x0 = self.interp.idx.ppoint
                if x0 is None: return None, None
                q_interp = Variables(self.interp)
                q_interp.compute_velocity()
                u0 = q_interp.velocity.reshape(3)
                k0 = time_step * u0
                x1 = x0 + k0

                u1, k1 = _rk4_step(self, x1)
                if k1 is None: return None, None
                x2 = x0 + 0.5 * k1

                u2, k2 = _rk4_step(self, x2)
                if k2 is None: return None, None
                x3 = x0 + 0.5 * k2

                u3, k3 = _rk4_step(self, x3)
                if k3 is None: return None, None
                x4 = x0 + 1/6 * (k0 + 2*k1 + 2*k2 + k3)

                self.ppoint = x4

                return self.ppoint, u3

            case 'cRK4':
                '''
                This is a block-wise integration. Everytime the point gets out
                a pRK4 is run to find the new block the point is located in.
                That particular step is done in streamlines algorithm.
                
                All the points, x0, x1... are in c-space
                
                Point location is known in c-space, avoiding search.
                Interpolates data to the point
                RK4 integration is performed!
                '''

                def _rk4_step(self, x):
                    """

                    Args:
                        self:
                        x: ndarray
                            point in c-space

                    Returns:
                        k: ndarray
                            interim RK4 variables

                    """
                    idx = Search(self.interp.idx.grid, x)
                    idx.block = self.interp.idx.block
                    idx.c2p(x)  # This will change the cell attribute
                    # In-domain check is done in search
                    if idx.cpoint is None:
                        self.cpoint = None
                        self.ppoint = None
                        return None
                    interp = Interpolation(self.interp.flow, idx)
                    interp.compute(method='c-space')
                    _J_inv = interp.J_inv
                    q_interp = Variables(interp)
                    q_interp.compute_velocity()
                    p_velocity = q_interp.velocity.reshape(3)
                    c_velocity = np.matmul(_J_inv, p_velocity)
                    k = time_step * c_velocity
                    return k

                x0 = self.interp.idx.cpoint
                _J_inv = self.interp.J_inv
                if x0 is None: return None

                q_interp = Variables(self.interp)
                q_interp.compute_velocity()
                p_velocity = q_interp.velocity.reshape(3)
                c_velocity = np.matmul(_J_inv, p_velocity)
                k0 = time_step * c_velocity
                x1 = x0 + k0

                k1 = _rk4_step(self, x1)
                if k1 is None: return None
                x2 = x0 + 0.5 * k1

                k2 = _rk4_step(self, x2)
                if k2 is None: return None
                x3 = x0 + 0.5 * k2

                k3 = _rk4_step(self, x3)
                if k3 is None: return None
                x4 = x0 + 1/6 * (k0 + 2*k1 + 2*k2 + k3)

                self.cpoint = x4

                return self.cpoint

    def compute_ppath(self, diameter=1e-6, density=1000, viscosity=1.827e-5, velocity=None,
                      method='pRK4', time_step=1e-4):

        def _drag_constant(_re):
            """
            Coefficient of drag for a spherical particle
            wrt relative Reynolds number
            ref: Fluid Mechanics, Frank M. White
            Args:
                _re

            Returns:
                coefficient of drag based on local flow/particle properties

            """
            if _re <= 1e-7:
                return 0
            if _re < 1e-3:
                return 24 / _re
            if 1e-3 <= _re < 0.45:
                return 24 / _re * (1 + 3 * _re / 16)
            if 0.45 <= _re < 1:
                # Same as above due to lack of data
                return 24 / _re * (1 + 3 * _re / 16)
            if 1 <= _re < 800:
                return 24 / _re * (1 + _re ** (2 / 3) / 6)
            if 800 <= _re < 3e5:
                return 0.44
            if 3e5 <= _re < 4e5:
                # Same as above due to lack of data
                return 0.44
            if _re >= 4e5:
                return 0.07

        def _viscosity(_mu_ref, _temperature):
            # Sutherland's viscosity law
            # All temperatures must be in kelvin
            _mu = _mu_ref * ((_temperature + 273.15) / 291.15)**1.5 * (291.15 + 120) / (120 + _temperature + 273.15)
            return _mu

        match method:
            case 'pRK4':

                def _rk4_step(self, vp, x):
                    """

                    Args:
                        self:

                    Returns:
                        k: ndarray
                            interim RK4 variables

                    """
                    idx = Search(self.interp.idx.grid, x)
                    idx.compute(method='p-space')
                    # For p-space algos; the point-in-domain check was done in search
                    if idx.ppoint is None:
                        return None, None
                    interp = Interpolation(self.interp.flow, idx)
                    interp.compute()
                    q_interp = Variables(interp)
                    q_interp.compute_velocity()
                    uf = q_interp.velocity.reshape(3)
                    # particle dynamics
                    _rhof = q_interp.density.reshape(-1)
                    dp = diameter
                    rhop = density
                    q_interp.compute_temperature()
                    mu = _viscosity(viscosity, q_interp.temperature.reshape(-1))
                    # Relative Reynolds Number
                    re = _rhof * np.linalg.norm(vp - uf) * dp / mu
                    _cd = _drag_constant(re)
                    # When drag is zero consider particle as a fluid packet
                    if _cd == 0:
                        _vk = np.zeros(3)
                        return _vk, uf
                    _k = -0.75 * _rhof / (rhop * dp)
                    _vk = _cd * _k * (vp - uf) * np.linalg.norm(vp - uf) * time_step
                    # if np.linalg.norm(_vk) >= 1e6 and time_step >= 1e-4:
                    #     print('!!! Large residuals detected. Decreasing time_step !!!')
                    #     # Decreasing _vk effectively reduces the time step too; implicit reduction
                    #     # This will ensure velocity will not blow up
                    #     _vk = _vk * 1e-4
                    return _vk, uf

                # Start RK4 for p-space
                # For p-space algos; the point-in-domain check was done in search
                x0 = self.interp.idx.ppoint
                if x0 is None: return None, None
                q_interp = Variables(self.interp)
                q_interp.compute_velocity()
                u0 = q_interp.velocity.reshape(3)
                # Assign velocity to start Rk4 step
                if velocity is None:
                    v0 = u0.copy()
                else:
                    v0 = velocity.copy()
                vk0, uf0 = _rk4_step(self, v0, x0)
                # Assign fluid velocity when vk is zero
                # Theory: When zero drag particle is massless, hence fluid velocity
                if np.linalg.norm(vk0) == 0:
                    v0 = uf0.copy()
                v1 = v0 + vk0
                xk0 = v1 * time_step
                x1 = x0 + xk0

                vk1, uf1 = _rk4_step(self, v1, x1)
                if vk1 is None:
                    return None, None
                v2 = v0 + 0.5 * vk1
                xk1 = v2 * time_step
                x2 = x0 + 0.5 * xk1

                vk2, uf2 = _rk4_step(self, v2, x2)
                if vk2 is None:
                    return None, None
                v3 = v0 + 0.5 * vk2
                xk2 = v3 * time_step
                x3 = x0 + 0.5 * xk2

                vk3, uf3 = _rk4_step(self, v3, x3)
                if vk3 is None:
                    return None, None
                v4 = v0 + 1 / 6 * (vk0 + 2 * vk1 + 2 * vk2 + vk3)
                xk3 = v4 * time_step
                x4 = x0 + 1 / 6 * (xk0 + 2 * xk1 + 2 * xk2 + xk3)

                self.ppoint = x4

                return x4, v4

