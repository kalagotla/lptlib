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
