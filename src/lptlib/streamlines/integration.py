# Integrate from given point to produce streamlines
import numpy as np
from ..streamlines.interpolation import Interpolation
from ..streamlines.search import Search
from ..function.variables import Variables
from scipy.optimize import fsolve


class Integration:

    def __init__(self, interp):
        self.interp = interp
        self.ppoint = None
        self.cpoint = None
        self.rk4_bool = False

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

            case 'pRK2':
                """
                This is a straight forward RK2 integration. Search for the point,
                Interpolate the data to the point, Compute required variables,
                Perform RK4 integration!
                """

                def _rk2_step(self, x):
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
                    interp.adaptive = self.interp.adaptive
                    interp.rbf_kernel = self.interp.rbf_kernel
                    interp.compute(method=self.interp.method)
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

                u1, k1 = _rk2_step(self, x1)
                if k1 is None: return None, None

                x_new = x0 + 1/2 * (k0 + k1)
                u_new, k_new = _rk2_step(self, x_new)
                if k_new is None: return None, None

                self.ppoint = x_new

                return self.ppoint, u_new

            case 'cRK2':
                '''
                This is a block-wise integration. Everytime the point gets out
                a pRK4 is run to find the new block the point is located in.
                That particular step is done in streamlines algorithm.

                All the points, x0, x1... are in c-space

                Point location is known in c-space, avoiding search.
                Interpolates data to the point
                RK4 integration is performed!
                '''

                def _rk2_step(self, x):
                    """

                    Args:
                        self:
                        x: ndarray
                            point in c-space

                    Returns:
                        k: ndarray
                            interim RK2 variables

                    """
                    idx = Search(self.interp.idx.grid, x)
                    idx.block = self.interp.idx.block
                    idx.c2p(x)  # This will change the cell attribute
                    # In-domain check is done in search
                    if idx.cpoint is None:
                        self.cpoint = None
                        self.ppoint = None
                        return None, None, None
                    interp = Interpolation(self.interp.flow, idx)
                    interp.adaptive = self.interp.adaptive
                    interp.rbf_kernel = self.interp.rbf_kernel
                    interp.compute(method=self.interp.method)
                    _J_inv = interp.J_inv
                    q_interp = Variables(interp)
                    q_interp.compute_velocity()
                    p_velocity = q_interp.velocity.reshape(3)
                    c_velocity = np.matmul(_J_inv, p_velocity)
                    k = time_step * c_velocity
                    return k, p_velocity, c_velocity

                x0 = self.interp.idx.cpoint
                _J_inv = self.interp.J_inv
                if x0 is None:
                    return None, None, None

                q_interp = Variables(self.interp)
                q_interp.compute_velocity()
                p_velocity = q_interp.velocity.reshape(3)
                c_velocity = np.matmul(_J_inv, p_velocity)
                k0 = time_step * c_velocity
                x1 = x0 + k0

                k1, pv1, cv1 = _rk2_step(self, x1)
                if k1 is None:
                    return None, None, None

                x_new = x0 + 1/2 * (k0 + k1)
                _, pv_new, cv_new = _rk2_step(self, x_new)

                self.cpoint = x_new

                return self.cpoint, pv_new, cv_new

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
                    interp.adaptive = self.interp.adaptive
                    interp.rbf_kernel = self.interp.rbf_kernel
                    interp.compute(method=self.interp.method)
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
                x1 = x0 + 0.5 * k0

                u1, k1 = _rk4_step(self, x1)
                if k1 is None: return None, None
                x2 = x0 + 0.5 * k1

                u2, k2 = _rk4_step(self, x2)
                if k2 is None: return None, None
                x3 = x0 + k2

                u3, k3 = _rk4_step(self, x3)
                if k3 is None: return None, None
                x_new = x0 + 1/6 * (k0 + 2*k1 + 2*k2 + k3)
                u_new, _ = _rk4_step(self, x_new)

                self.ppoint = x_new

                return self.ppoint, u_new

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
                        return None, None, None
                    interp = Interpolation(self.interp.flow, idx)
                    interp.adaptive = self.interp.adaptive
                    interp.rbf_kernel = self.interp.rbf_kernel
                    interp.compute(method=self.interp.method)
                    _J_inv = interp.J_inv
                    q_interp = Variables(interp)
                    q_interp.compute_velocity()
                    p_velocity = q_interp.velocity.reshape(3)
                    c_velocity = np.matmul(_J_inv, p_velocity)
                    k = time_step * c_velocity
                    return k, p_velocity, c_velocity

                x0 = self.interp.idx.cpoint
                _J_inv = self.interp.J_inv
                if x0 is None:
                    return None, None, None

                q_interp = Variables(self.interp)
                q_interp.compute_velocity()
                p_velocity = q_interp.velocity.reshape(3)
                c_velocity = np.matmul(_J_inv, p_velocity)
                k0 = time_step * c_velocity
                x1 = x0 + 0.5 * k0

                k1, pv1, cv1 = _rk4_step(self, x1)
                if k1 is None:
                    return None, None, None
                x2 = x0 + 0.5 * k1

                k2, pv2, cv2 = _rk4_step(self, x2)
                if k2 is None:
                    return None, None, None
                x3 = x0 + k2

                k3, pv3, cv3 = _rk4_step(self, x3)
                if k3 is None:
                    return None, None, None
                x_new = x0 + 1/6 * (k0 + 2*k1 + 2*k2 + k3)
                _, pv_new, cv_new = _rk4_step(self, x_new)

                self.cpoint = x_new

                return self.cpoint, pv_new, cv_new

            case 'unsteady-pRK4':
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
                    interp.adaptive = self.interp.adaptive
                    interp.rbf_kernel = self.interp.rbf_kernel
                    interp.time = self.interp.time
                    interp.flow_old = self.interp.flow_old
                    interp.compute(method='unsteady-rbf-p-space')
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
                x1 = x0 + 0.5 * k0

                u1, k1 = _rk4_step(self, x1)
                if k1 is None: return None, None
                x2 = x0 + 0.5 * k1

                u2, k2 = _rk4_step(self, x2)
                if k2 is None: return None, None
                x3 = x0 + k2

                u3, k3 = _rk4_step(self, x3)
                if k3 is None: return None, None
                x_new = x0 + 1 / 6 * (k0 + 2 * k1 + 2 * k2 + k3)
                u_new, _ = _rk4_step(self, x_new)

                self.ppoint = x_new

                return self.ppoint, u_new

    def compute_ppath(self, diameter=1e-6, density=1000, velocity=None,
                      method='pRK4', time_step=1e-4, drag_model='stokes'):

        def _drag_constant(_re, _q_interp=None, _mach=None, _mu=None, _model=drag_model):
            """
            Coefficient of drag for a spherical particle

            Args:
                _re : Relative Reynolds Number
                _mach : Relative Mach Number
                _model : Drag Model Name
                    Available models are 'sphere', 'stokes', 'oseen', 'schiller_nauman',
                    'cunningham'

            Returns:
                coefficient of drag based on local flow/particle properties

            """
            _gamma = q_interp.gamma
            match _model:
                case 'zero-drag':
                    # zero drag model to simulate fluid
                    return 0

                case 'sphere':
                    # ref: Fluid Mechanics, Frank M. White
                    # This was decided by trail-and-error from VISUAL3 code
                    if _re <= 1e-9:
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

                case 'stokes':
                    # Stokes Drag; for creeping flow regime; Re << 1
                    if _re <= 1e-9:
                        return 0
                    else:
                        return 24/_re

                case 'melling':
                    # The popular melling correction
                    if _re <= 1e-9:
                        return 0
                    else:
                        knd = _mach / _re * np.sqrt(np.pi*_gamma/2)
                        return 24/_re * (1 + knd)**-1

                case 'melling-2':
                    # The popular melling correction
                    if _re <= 1e-9:
                        return 0
                    else:
                        knd = _mach / _re * np.sqrt(np.pi*_gamma/2)
                        return 24/_re * (1 + 2.7*knd)**-1

                case 'oseen':
                    # Oseen's model; for creeping flow regime; Re < 1
                    if _re <= 1e-9:
                        return 0
                    else:
                        return 24/_re * (1 + 3/16 * _re)

                case 'schiller-nauman':
                    # Schiller and Nauman's model; for Re <~ 200 & M <~ 0.25
                    if _re <= 1e-9:
                        return 0
                    else:
                        return 24/_re * (1 + 0.15 * _re**0.687)

                case 'cunningham':
                    # Cunningham model; for Re << 1; M << 1; Kn <~ 0.1
                    # Knudsen number
                    # _r = _q_interp.density * _q_interp.velocity_magnitude * diameter / _mu
                    if _re <= 1e-9:
                        return 0
                    if _re <= 1:
                        _kn = _mach / _re * np.sqrt(_q_interp.gamma * np.pi/2)
                        return 24/_re * (1 + 4.5*_kn)**-1
                    if _re > 1:
                        _kn = _mach / np.sqrt(_re)
                        return 24/_re * (1 + 4.5*_kn)**-1

                case 'henderson':
                    # Henderson model; for all flow regimes
                    # Simplified by ignoring sphere temperature
                    if _re < 1e-9:
                        return 0

                    # For Mach < 1
                    _s = _mach * np.sqrt(_gamma/2)
                    _f1 = 24 * (_re + _s * (5.89688 * np.exp(-0.247 * _re/_s)))**-1
                    _f2 = np.exp(-0.5*_mach/np.sqrt(_re)) * \
                                ((4.5 + 0.38*(0.03*_re + 0.48*np.sqrt(_re))) / (1 + 0.03*_re + 0.48*np.sqrt(_re)) +
                                 0.1*_mach**2 + 0.2*_mach**8)
                    _f3 = (1 - np.exp(-_mach/_re))*0.6*_s
                    _cd1 = _f1 + _f2 + _f3
                    if _mach < 1:
                        return _cd1

                    # For Mach >= 1.75
                    _mach_inf = _mach
                    _re_inf = _re
                    _s_inf = _mach_inf * np.sqrt(_gamma/2)
                    _g1 = 0.9 + 0.34/_mach_inf**2
                    _g2 = 1.86 * np.sqrt(_mach_inf/_re_inf) * (2 + 2/_s_inf**2 + 1.058/_s_inf - 1/_s_inf**4)
                    _g3 = 1 + 1.86 * np.sqrt(_mach_inf/_re_inf)
                    _cd2 = (_g1 + _g2) / _g3
                    if _mach >= 1.75:
                        return _cd2

                    # For 1 <= Mach < 1.75; linear interpolation
                    if 1 <= _mach < 1.75:
                        return _cd1 + 4/3 * (_mach_inf - 1) * (_cd2 - _cd1)

                case 'subramaniam-balachandar':
                    # Model from their new book
                    if _re < 1e-9:
                        return 0

                    if _re < 0.5:
                        # Stokes
                        return 24/_re

                    if _re < 20:
                        # Clift
                        return 24/_re * (1 + 0.1315 * _re**(0.82-0.05*np.log10(_re)))

                    if _re < 800:
                        # Schiller-Naumann
                        return 24/_re * (1 + 0.15 * _re**0.687)

                    if _re < 3e5:
                        # Clift-Gauvin
                        return 24/_re * (1 + 0.15 * _re**0.687 + 0.42/24 * _re * (1 + 4.25e4 * _re**(-1.16))**-1)

                case 'loth':
                    # Loth's model; for all flow regimes
                    if _re < 1e-9:
                        return 0

                    if _re < 45:
                    # Rarefraction dominated domain
                        import math
                        _s = _mach * np.sqrt(_gamma/2)
                        _cd_fm = (1 + 2 * _s**2) * np.exp(-_s**2) / (_s**3 * np.pi**0.5) + \
                                 (4*_s**4 + 4*_s**2 - 1) * math.erf(_s) / (2*_s**4) + 2 * np.pi**0.5 / (3 * _s)
                        _cd_fm_re = _cd_fm / (1 + (_cd_fm/1.63 - 1) * (_re/45)**0.5)
                        _kn = (np.pi * _gamma / 2)**0.5 * _mach / _re
                        _f_kn = (1 + _kn * (2.514 + 0.8 * np.exp(-0.55/_kn)))**-1
                        _cd_kn_re = 24/_re * (1 + 0.15 * _re**0.687) * _f_kn
                        _cd = (_cd_kn_re + _mach**4 * _cd_fm_re) / (1 + _mach**4)
                        return _cd

                    if _re == 45:
                        return 1.63

                    if _re > 45:
                    # compression-dominated regime
                        if _mach <= 1.45:
                            _cm = 5/3 + 2/3 * np.tanh(3 * np.log(_mach + 0.1))
                        if _mach > 1.45:
                            _cm = 2.044 + 0.2 * np.exp(-1.8 * (np.log(_mach/1.5))**2)
                        if _mach <= 0.89:
                            _gm = 1 - 1.525 * _mach**4
                        if _mach > 0.89:
                            _gm = 2e-4 + 8e-4 * np.tanh(12.77 * (_mach - 2.02))
                        _hm = 1 - 0.258 * _cm / (1 + 514 * _gm)
                        _cd = 24/_re * (1 + 0.15 * _re**0.687) * _hm + 0.42 * _cm / (1 + 42000 * _gm / _re**1.16)
                        return _cd

                case 'tedeschi':
                    # Tedeschi's model; for all flow regimes
                    if _re < 1e-9:
                        return 0
                    if _re <= 1:
                        _kn = _mach / _re * np.sqrt(_q_interp.gamma * np.pi/2)
                    else:
                        _kn = _mach / np.sqrt(_re)

                    s = _mach * np.sqrt(_gamma/2)

                    def _solve_k(_k):
                        s_prime = (1 - _k) * s
                        epsilon_prime = 3/8 * (np.pi**2 / s_prime) * (1 + s_prime**2) * s_prime + np.exp(-s_prime**2) /4
                        a1 = 9/4 * 0.15 * 2 * _kn / epsilon_prime * (s * np.pi**0.5 / _kn)**0.687
                        a2 = 1 + 9/4 * 2 * _kn / epsilon_prime
                        return a1 * _k**1.687 + a2 * _k - 1

                    # solve the equation
                    k = fsolve(_solve_k, np.array([0.5]))

                    c = 1 + _re**2 / (_re**2 + 100) * np.e**(-0.225/_mach**2.5)
                    _epsilon_kn = 1.177 + 0.177 * (0.851 * _kn**1.16 - 1) / (0.851 * _kn**1.16 + 1)

                    return 24/_re * k * (1 + 0.15 * (k*_re)**0.687) * c * _epsilon_kn

        def _viscosity(_temperature, law='keyes'):
            """
            Viscosity of air as a function of temperature
            Args:
                _temperature: in Kelvin provided by default in the integration class
                law: 'sutherland' or 'keyes' -- defaults to 'keyes'

            Returns:
                _mu: viscosity of air in kg/m-s

            """
            match law:
                case 'sutherland':
                    # Sutherland's viscosity law
                    # All temperatures must be in kelvin
                    # Formula from cfd-online
                    _c1 = 1.716e-5 * (273.15 + 110.4) / 273.15**1.5
                    _mu = _c1 * _temperature**1.5 * 0.4 / (_temperature + 110.4)
                case 'keyes':
                    # New formula from keyes et al.
                    a0, a, a1 = 1.488, 122.1, 5.0
                    _tau = 1/_temperature
                    _mu = a0 * _temperature**0.5 * 10**-6 / (1 + a * _tau / 10 ** (a1 * _tau))
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
                        return None, None, None
                    interp = Interpolation(self.interp.flow, idx)
                    interp.adaptive = self.interp.adaptive
                    interp.rbf_kernel = self.interp.rbf_kernel
                    interp.compute(method=self.interp.method)
                    q_interp = Variables(interp)
                    # Compute all variables
                    q_interp.compute_mach()
                    uf = q_interp.velocity.reshape(-1)
                    # particle dynamics
                    _rhof = q_interp.density.reshape(-1)
                    dp = diameter
                    rhop = density
                    mu = _viscosity(q_interp.temperature.reshape(-1))
                    # Relative Reynolds Number
                    re = _rhof * np.linalg.norm(vp - uf) * dp / mu
                    _mach = np.linalg.norm(vp - uf) * q_interp.mach.reshape(-1) /\
                            q_interp.velocity_magnitude.reshape(-1)
                    _cd = _drag_constant(re, _q_interp=q_interp, _mach=_mach, _mu=mu, _model=drag_model)
                    _k = -0.75 * _rhof / (rhop * dp)
                    try:
                        _vk = _cd * _k * (vp - uf) * np.linalg.norm(vp - uf) * time_step
                    except TypeError:
                        return None, None, None
                    return _vk, uf, None

                # Start RK4 for p-space
                # For p-space algos; the point-in-domain check was done in search
                x0 = self.interp.idx.ppoint
                if x0 is None:
                    return None, None, None
                q_interp = Variables(self.interp)
                q_interp.compute_velocity()
                u0 = q_interp.velocity.reshape(3)
                # Assign velocity to start Rk4 step
                if velocity is None:
                    v0 = u0.copy()
                else:
                    v0 = velocity.copy()
                vk0, uf0, temp = _rk4_step(self, v0, x0)
                xk0 = v0 * time_step
                # Assign fluid velocity when vk is zero
                # Theory: When zero drag particle is massless, hence fluid velocity
                if np.linalg.norm(vk0) == 0:
                    v0 = uf0.copy()
                v1 = v0 + 0.5 * vk0
                x1 = x0 + 0.5 * xk0

                vk1, uf1, temp = _rk4_step(self, v1, x1)
                xk1 = v1 * time_step
                if vk1 is None:
                    return None, None, None
                if np.linalg.norm(vk1) == 0:
                    v0 = uf1.copy()
                v2 = v0 + 0.5 * vk1
                x2 = x0 + 0.5 * xk1
                # Check for mid-RK4 blow-up issue. Happens when Cd and time-step are high
                if np.linalg.norm(x2 - x0) >= 10 * np.linalg.norm(x1-x0) and np.linalg.norm(x2 - x0) >= 1e-12:
                    self.rk4_bool = True
                    return x0, v0, u0

                vk2, uf2, temp = _rk4_step(self, v2, x2)
                xk2 = v2 * time_step
                if vk2 is None:
                    return None, None, None
                if np.linalg.norm(vk2) == 0:
                    v0 = uf2.copy()
                v3 = v0 + vk2
                x3 = x0 + xk2
                # Check for mid-RK4 blow-up issue. Happens when Cd and time-step are high
                if np.linalg.norm(x3 - x0) >= 10 * np.linalg.norm(x1 - x0) and np.linalg.norm(x3 - x0) >= 1e-12:
                    self.rk4_bool = True
                    return x0, v0, u0

                vk3, uf3, temp = _rk4_step(self, v3, x3)
                xk3 = v3 * time_step
                if vk3 is None:
                    return None, None, None
                if np.linalg.norm(vk3) == 0:
                    v0 = uf3.copy()
                v_new = v0 + 1 / 6 * (vk0 + 2 * vk1 + 2 * vk2 + vk3)
                x_new = x0 + 1 / 6 * (xk0 + 2 * xk1 + 2 * xk2 + xk3)
                # Check for mid-RK4 blow-up issue. Happens when Cd and time-step are high
                if np.linalg.norm(x_new - x0) >= 10 * np.linalg.norm(x1 - x0) and np.linalg.norm(x_new - x0) >= 1e-12:
                    self.rk4_bool = True
                    return x0, v0, u0

                vk_new, uf_new, temp = _rk4_step(self, v_new, x_new)
                # make sure the velocity is going down in a compression case
                # This is done to remove the oscillations in the post-shock region
                if vk_new is None:
                    return None, None, None
                if self.interp.method == 'simple_oblique_shock' and np.dot(v_new - v0, uf_new - v_new) < 0:
                    self.rk4_bool = True
                    return x0, v0, u0

                self.ppoint = x_new

                return x_new, v_new, uf_new

            case 'cRK4':

                def _rk4_step(self, c_vp, x):
                    """
                    Returns required data for the RK4 step

                    Args:
                        self:
                        c_vp: ndarray
                            particle velocity in c-space
                        x: ndarray
                            point in c-space

                    Returns:
                        k: ndarray
                            interim RK4 variables

                    """
                    if np.any(x < 0):
                        return None, None, None, None
                    idx = Search(self.interp.idx.grid, x)
                    idx.block = self.interp.idx.block
                    idx.c2p(x)  # This will change the cell attribute
                    # In-domain check is done in search
                    if idx.cpoint is None:
                        self.cpoint = None
                        self.ppoint = None
                        return None, None, None, None
                    interp = Interpolation(self.interp.flow, idx)
                    interp.adaptive = self.interp.adaptive
                    interp.rbf_kernel = self.interp.rbf_kernel
                    interp.compute(method=self.interp.method)
                    _J_inv = interp.J_inv
                    _J = interp.J
                    q_interp = Variables(interp)
                    # Computing mach get all the necessary variables
                    q_interp.compute_mach()
                    p_uf = q_interp.velocity.reshape(-1)
                    c_uf = np.matmul(_J_inv, p_uf)
                    # particle dynamics
                    _rhof = q_interp.density.reshape(-1)
                    dp = diameter
                    rhop = density
                    # Transform particle velocity to p-space
                    p_vp = np.matmul(_J, c_vp)
                    mu = _viscosity(q_interp.temperature.reshape(-1))
                    # Relative Reynolds Number
                    re = _rhof * np.linalg.norm(p_vp - p_uf) * dp / mu
                    _mach = np.linalg.norm(p_vp - p_uf) * q_interp.mach.reshape(-1) /\
                            q_interp.velocity_magnitude.reshape(-1)
                    _cd = _drag_constant(re, _q_interp=q_interp, _mach=_mach, _mu=mu, _model=drag_model)
                    _k = -0.75 * _rhof / (rhop * dp)
                    p_vk = _cd * _k * (p_vp - p_uf) * np.linalg.norm(p_vp - p_uf) * time_step
                    c_vk = np.matmul(_J_inv, p_vk)
                    return c_vk, p_uf, c_uf, p_vp

                # Start RK4 for c-space
                x0 = self.interp.idx.cpoint
                if x0 is None:
                    return None, None, None
                q_interp = Variables(self.interp)
                q_interp.compute_velocity()
                p_u0 = q_interp.velocity.reshape(-1)
                c_u0 = np.matmul(self.interp.J_inv, p_u0)
                # Assign velocity to start Rk4 step
                if velocity is None:
                    c_v0 = c_u0.copy()
                else:
                    c_v0 = np.matmul(self.interp.J_inv, velocity)
                vk0, p_u0, c_u0, p_v0 = _rk4_step(self, c_v0, x0)
                # Assign fluid velocity when vk is zero
                # Theory: When zero drag particle is massless, hence fluid velocity
                if np.linalg.norm(vk0) == 0:
                    c_v0 = c_u0.copy()
                c_v1 = c_v0 + 0.5 * vk0
                xk0 = c_v0 * time_step
                x1 = x0 + 0.5 * xk0

                # Integration starts
                vk1, p_u1, c_u1, p_v1 = _rk4_step(self, c_v1, x1)
                # if the residual is none return; exited the domain
                if vk1 is None:
                    return None, None, None
                # if zero; particle is acting like massless particle; low relative reynolds number cases
                if np.linalg.norm(vk1) == 0:
                    c_v0 = c_u1.copy()
                c_v2 = c_v0 + 0.5 * vk1
                xk1 = c_v1 * time_step
                x2 = x0 + 0.5 * xk1
                # Check for mid-RK4 blow up due to residuals
                if np.linalg.norm(x2 - x0) >= 10 * np.linalg.norm(x1-x0):
                    self.rk4_bool = True
                    return x0, p_v0, p_u0

                # Repeat three more times; RK4
                vk2, p_u2, c_u2, p_v2 = _rk4_step(self, c_v2, x2)
                if vk2 is None:
                    return None, None, None
                if np.linalg.norm(vk2) == 0:
                    c_v0 = c_u2.copy()
                c_v3 = c_v0 + vk2
                xk2 = c_v2 * time_step
                x3 = x0 + xk2
                if np.linalg.norm(x3 - x0) >= 10 * np.linalg.norm(x1-x0):
                    self.rk4_bool = True
                    return x0, p_v0, p_u0

                vk3, p_u3, c_u3, p_v3 = _rk4_step(self, c_v3, x3)
                if vk3 is None:
                    return None, None, None
                if np.linalg.norm(vk3) == 0:
                    c_v0 = c_u3.copy()
                c_v_new = c_v0 + 1 / 6 * (vk0 + 2 * vk1 + 2 * vk2 + vk3)
                xk3 = c_v3 * time_step
                x_new = x0 + 1 / 6 * (xk0 + 2 * xk1 + 2 * xk2 + xk3)
                if np.linalg.norm(x_new - x0) >= 10 * np.linalg.norm(x1-x0):
                    self.rk4_bool = True
                    return x0, p_v0, p_u0

                _, p_u_new, c_u_new, p_v_new = _rk4_step(self, c_v_new, x_new)

                self.cpoint = x_new

                return x_new, p_u_new, p_v_new

            case 'unsteady-pRK4':

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
                        return None, None, None
                    interp = Interpolation(self.interp.flow, idx)
                    interp.time = self.interp.time
                    interp.flow_old = self.interp.flow_old
                    interp.adaptive = self.interp.adaptive
                    interp.rbf_kernel = self.interp.rbf_kernel
                    interp.compute(method='unsteady-rbf-p-space')
                    q_interp = Variables(interp)
                    # Compute all variables
                    q_interp.compute_mach()
                    uf = q_interp.velocity.reshape(-1)
                    # particle dynamics
                    _rhof = q_interp.density.reshape(-1)
                    dp = diameter
                    rhop = density
                    mu = _viscosity(q_interp.temperature.reshape(-1), law='sutherland')
                    # Relative Reynolds Number
                    re = _rhof * np.linalg.norm(vp - uf) * dp / mu
                    _mach = np.linalg.norm(vp - uf) * q_interp.mach.reshape(-1) / \
                            q_interp.velocity_magnitude.reshape(-1)
                    _cd = _drag_constant(re, _q_interp=q_interp, _mach=_mach, _mu=mu, _model=drag_model)
                    _k = -0.75 * _rhof / (rhop * dp)
                    try:
                        _vk = _cd * _k * (vp - uf) * np.linalg.norm(vp - uf) * time_step
                    except TypeError:
                        return None, None, None
                    return _vk, uf, None

                # Start RK4 for p-space
                # For p-space algos; the point-in-domain check was done in search
                x0 = self.interp.idx.ppoint
                if x0 is None:
                    return None, None, None
                q_interp = Variables(self.interp)
                q_interp.compute_velocity()
                u0 = q_interp.velocity.reshape(3)
                # Assign velocity to start Rk4 step
                if velocity is None:
                    v0 = u0.copy()
                else:
                    v0 = velocity.copy()
                vk0, uf0, temp = _rk4_step(self, v0, x0)
                xk0 = v0 * time_step
                # Assign fluid velocity when vk is zero
                # Theory: When zero drag particle is massless, hence fluid velocity
                if np.linalg.norm(vk0) == 0:
                    v0 = uf0.copy()
                v1 = v0 + 0.5 * vk0
                x1 = x0 + 0.5 * xk0

                vk1, uf1, temp = _rk4_step(self, v1, x1)
                xk1 = v1 * time_step
                if vk1 is None:
                    return None, None, None
                if np.linalg.norm(vk1) == 0:
                    v0 = uf1.copy()
                v2 = v0 + 0.5 * vk1
                x2 = x0 + 0.5 * xk1
                # Check for mid-RK4 blow-up issue. Happens when Cd and time-step are high
                if np.linalg.norm(x2 - x0) >= 10 * np.linalg.norm(x1 - x0) and np.linalg.norm(x2 - x0) >= 1e-12:
                    self.rk4_bool = True
                    return x0, v0, u0

                vk2, uf2, temp = _rk4_step(self, v2, x2)
                xk2 = v2 * time_step
                if vk2 is None:
                    return None, None, None
                if np.linalg.norm(vk2) == 0:
                    v0 = uf2.copy()
                v3 = v0 + vk2
                x3 = x0 + xk2
                # Check for mid-RK4 blow-up issue. Happens when Cd and time-step are high
                if np.linalg.norm(x3 - x0) >= 10 * np.linalg.norm(x1 - x0) and np.linalg.norm(x3 - x0) >= 1e-12:
                    self.rk4_bool = True
                    return x0, v0, u0

                vk3, uf3, temp = _rk4_step(self, v3, x3)
                xk3 = v3 * time_step
                if vk3 is None:
                    return None, None, None
                if np.linalg.norm(vk3) == 0:
                    v0 = uf3.copy()
                v_new = v0 + 1 / 6 * (vk0 + 2 * vk1 + 2 * vk2 + vk3)
                x_new = x0 + 1 / 6 * (xk0 + 2 * xk1 + 2 * xk2 + xk3)
                # Check for mid-RK4 blow-up issue. Happens when Cd and time-step are high
                if np.linalg.norm(x_new - x0) >= 10 * np.linalg.norm(x1 - x0) and np.linalg.norm(
                        x_new - x0) >= 1e-12:
                    self.rk4_bool = True
                    return x0, v0, u0

                vk_new, uf_new, temp = _rk4_step(self, v_new, x_new)
                # make sure the velocity is going down in a compression case
                # This is done to remove the oscillations in the post-shock region
                if vk_new is None:
                    return None, None, None
                if self.interp.method == 'simple_oblique_shock' and np.dot(v_new - v0, uf_new - v_new) < 0:
                    self.rk4_bool = True
                    return x0, v0, u0

                self.ppoint = x_new

                return x_new, v_new, uf_new
