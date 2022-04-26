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

    def __init__(self, flow, gamma=1.4):
        self.flow = flow
        self.gamma = gamma
        self.density = flow.q[..., 0, :]
        self.velocity = None
        self.velocity_magnitude = None
        self.temperature = None
        self.pressure = None

    def compute_velocity(self):
        """
        Function to compute velocity and velocity magnitude
        :return: None
        """
        self.velocity = self.flow.q[..., 1:4, :] / (self.flow.q[..., 0, :, None] * self.flow.mach)
        self.velocity_magnitude = (self.velocity[..., 0, :]**2 + self.velocity[..., 1, :]**2 + self.velocity[..., 2, :]**2)**0.5

    def compute_temperature(self):
        """
        Function to compute temperature.
        This computes velocity first
        :return: None
        """
        self.compute_velocity()
        _R = 1 / (self.gamma * self.flow.mach**2)
        _E_T = self.flow.q[..., 4, :] / self.flow.mach**2
        self.temperature = (self.gamma - 1) * (_E_T/self.flow.q[..., 0, :] - self.velocity_magnitude/2) / _R

    def compute_pressure(self):
        """
        Function to compute pressure.
        This computes velocity and temperature first
        :return: None
        """
        self.compute_temperature()
        self.pressure = self.flow.q[..., 0, :] * self.temperature

    def compute(self):
        # implicitly runs compute_velocity() and compute_temperature()
        """
        Function to compute all the attributes in the class
        :return: None
        """
        self.compute_pressure()
