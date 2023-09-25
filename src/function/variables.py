# Calculates different variables from plot3d data
# Equations can be found here: https://www.grc.nasa.gov/WWW/winddocs/towne/plotc/plotc_p3d.html
# TODO: Add more variables
#  Enthalpies, Vorticity, Entropy, Turbulence Parameters, Gradients, Move metrics here?
#  Total quantities
import numpy as np


class Variables:
    """
    Module to compute flow variables

    ...

    Attributes
    ----------
    Input:
        flow : src.io.plot3dio.FlowIO or src.streamlines.interpolation.Interpolation or similar
            object with q --> flow data
        gamma : float
            default is 1.4; can specify
    Output:
        velocity : ndarray
            velocity of the flow at nodes; shape of the flow.q (ni x nj x nk x 3 x nb)
        velocity_magnitude : ndarray
            magnitude of velocity at nodes; shape is (ni x nj x nk x nb)
        temperature : ndarray
            temperature at nodes; shape is (ni x nj x nk x nb)
        pressure : ndarray
            pressure at nodes; shape is (ni x nj x nk x nb)

    Methods
    -------
    compute_velocity()
        computes the velocity and velocity_magnitude
    compute_temperature()
        computes the temperature
    compute_pressure()
        computes the pressure
    compute()
        computes all the variables available

    ...

    Example:
    -------
        variables = Variables(flow)  # Assume flow object is pre-defined
        variables.compute_velocity()  # returns velocity attribute
        variables.compute_temperature()  # fills up the temperature attribute
        variables.compute()  # computes all the attributes available

    """

    def __init__(self, flow, gamma=1.4, gas_constant=287.052874):
        self.flow = flow
        self.gamma = gamma
        self.gas_constant = gas_constant
        self.density = flow.q[..., 0, :]  # q0
        self.velocity = None
        self.mach = None
        self.velocity_magnitude = None
        self.temperature = None
        self.pressure = None
        self.viscosity = None

    def compute_velocity(self):
        """
        Function to compute velocity and velocity magnitude
        :return: None
        """
        # velocity = [q1, q2, q3] / q0
        self.velocity = self.flow.q[..., 1:4, :] / (self.flow.q[..., 0, :, None])
        self.velocity_magnitude = (self.velocity[..., 0, :]**2 + self.velocity[..., 1, :]**2 + self.velocity[..., 2, :]**2)**0.5

        return

    def compute_temperature(self):
        """
        Function to compute temperature.
        This computes velocity first
        :return: None
        """
        self.compute_velocity()
        _q4 = self.flow.q[..., 4, :]
        self.temperature = (self.gamma - 1) * (_q4/self.density - self.velocity_magnitude**2/2) / self.gas_constant

        return

    def compute_mach(self):
        """
        Function to compute local mach number
        Returns: None
        """
        self.compute_temperature()
        self.mach = self.velocity_magnitude / np.sqrt(self.gamma * self.gas_constant * self.temperature)

        return

    def compute_pressure(self):
        """
        Function to compute pressure.
        This computes velocity and temperature first
        :return: None
        """
        self.compute_mach()
        self.pressure = self.density * self.temperature * self.gas_constant

        return

    def compute_viscosity(self, law='keyes'):
        """
        Function to compute viscosity

        Viscosity of air as a function of temperature
        Args:
            _temperature: in Kelvin provided by default in the integration class
            law: 'sutherland' or 'keyes' -- defaults to 'keyes'
        :return: None

        """
        match law:
            case 'sutherland':
                # Sutherland's viscosity law
                # All temperatures must be in kelvin
                # Formula from cfd-online
                _c1 = 1.716e-5 * (273.15 + 110.4) / 273.15**1.5
                self.viscosity = _c1 * self.temperature**1.5 * 0.4 / (self.temperature + 110.4)
            case 'keyes':
                # New formula from keyes et al.
                a0, a, a1 = 1.488, 122.1, 5.0
                _tau = 1/self.temperature
                self.viscosity = a0 * self.temperature**0.5 * 10**-6 / (1 + a * _tau / 10 ** (a1 * _tau))
            case _:
                raise ValueError('Viscosity law not supported')

        return

    def compute_drag_coefficient(self, _re=None, _mach=None, _model='stokes'):
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
                    knd = _mach / _re * np.sqrt(np.pi*self.gamma/2)
                    return 24/_re * (1 + knd)**-1

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
                if _re <= 1e-9:
                    return 0
                if _re <= 1:
                    _kn = _mach / _re * np.sqrt(self.gamma * np.pi/2)
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
                _s = _mach * np.sqrt(self.gamma/2)
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
                _s_inf = _mach_inf * np.sqrt(self.gamma/2)
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
                    _s = _mach * np.sqrt(self.gamma/2)
                    _cd_fm = (1 + 2 * _s**2) * np.exp(-_s**2) / (_s**3 * np.pi**0.5) + \
                             (4*_s**4 + 4*_s**2 - 1) * math.erf(_s) / (2*_s**4) + 2 * np.pi**0.5 / (3 * _s)
                    _cd_fm_re = _cd_fm / (1 + (_cd_fm/1.63 - 1) * (_re/45)**0.5)
                    _kn = (np.pi * self.gamma / 2)**0.5 * _mach / _re
                    _f_kn = (1 + _kn * (2.514 + 0.8 * np.exp(-0.55/_kn)))**-1
                    _cd_kn_re = 24/_re * (1 + 0.15 * _re**0.687) * _f_kn
                    _cd = (_cd_kn_re + _mach**4 * _cd_fm_re) / (1 + _mach**4)
                    return _cd

                if _re == 45:
                    return 1.63

                if _re > 45:
                    # compression-dominated regime
                    if _mach <= 1.45:
                        _cm = 5/3 + 2/3 * np.tanh(3 * np.log(_mach - 0.1))
                    if _mach > 1.45:
                        _cm = 2.044 + 0.2 * np.exp(-1.8 * (np.log(_mach/1.5))**2)
                    if _mach <= 0.89:
                        _gm = 1 - 1.525 * _mach**4
                    if _mach > 0.89:
                        _gm = 2e-4 + 8e-4 * np.tanh(12.77 * (_mach - 2.02))
                    _hm = 1 - 0.258 * _cm / (1 + 514 * _gm)
                    _cd = 24/_re * (1 + 0.15 * _re**0.687) * _hm + 0.42 * _cm / (1 + 42000 * _gm / _re**1.16)
                    return _cd

                return

    def compute(self):
        # implicitly runs compute_velocity() and compute_temperature()
        """
        Function to compute all the attributes in the class
        :return: None
        """
        self.compute_pressure()
        # This computes viscosity using the keyes formula
        # To compute using sutherland's law, call compute_viscosity() separately, which will update the attribute
        self.compute_viscosity()
