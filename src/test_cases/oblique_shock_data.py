# Class to calculate particle response across an oblique shock

import numpy as np
from scipy.optimize import newton
from src.streamlines.stochastic_model import StochasticModel, Particle, SpawnLocations
import matplotlib.pyplot as plt
import os
import glob


# Create a class to calculate oblique shock properties from given mach and deflection angle
class ObliqueShock:
    """
    Class to calculate oblique shock properties from given mach and deflection angle

    ...

    Attributes
    ----------
    Input :
        mach : float
            Mach number
        deflection : float
            Deflection angle in degrees
        shock_angle : float
            Shock angle in degrees
    Output :
        pressure_ratio : float
            Pressure ratio across the shock
        density_ratio : float
            Density ratio across the shock
        temperature_ratio : float
            Temperature ratio across the shock


    Methods
    -------
    oblique_shock_relation(_mach, _deflection, _shock_angle)
        Oblique shock relations for given mach, deflection and theta
    derivative(f, method='central', h=1e-7) -- static method
        Compute the difference formula for f with step size h.
    compute()
        Calculate the oblique shock properties

    Example:
    >>> os = ObliqueShock()
    >>> os.mach = 2.3
    >>> os.deflection = 10
    >>> os.compute()
    >>> print(os.shock_angle)
    [34.32642717 85.02615188]

    """

    def __init__(self, mach=None, deflection=None, shock_angle=None):
        self.mach = mach
        self.deflection = deflection
        self.shock_angle = shock_angle
        self.pressure_ratio = None
        self.density_ratio = None
        self.temperature_ratio = None
        self.mach_ratio = None
        self.gamma = 1.4
        self.gas_constant = 287.058

    def oblique_shock_relation(self, _mach, _deflection, _shock_angle):
        """
        Oblique shock relations for given mach, deflection and theta
        Args:

        Returns:
            delta-theta-mach relation
        """
        return (_mach ** 2 * (self.gamma + np.cos(2 * _shock_angle)) + 2) * np.tan(_deflection) * \
            np.tan(_shock_angle) - 2 * (_mach ** 2 * np.sin(_shock_angle) ** 2 - 1)

    @staticmethod
    def derivative(f, method='central', h=1e-7):
        """Compute the difference formula for f with step size h.

        Parameters
        ----------
        f : function
            Vectorized function of one variable
        method : string
            Difference formula: 'forward', 'backward' or 'central'
        h : number
            Step size in difference formula

        Returns
        -------
        function
            Difference formula:
                central: f(x+h) - f(x-h))/2h
                forward: f(x+h) - f(x))/h
                backward: f(x) - f(x-h))/h
        """

        if method == 'central':
            return lambda x: (f(x + h) - f(x - h)) / (2 * h)
        elif method == 'forward':
            return lambda x: (f(x + h) - f(x)) / h
        elif method == 'backward':
            return lambda x: (f(x) - f(x - h)) / h
        else:
            raise ValueError("Method must be 'central', 'forward' or 'backward'.")

    def compute(self):
        # Use Newton-Raphson method to calculate the shock angle
        # Compute the shock angle if mach and deflection are given
        if self.mach is not None and self.deflection is not None:
            self.deflection = np.radians(self.deflection)

            f_shock_angle = lambda shock_angle: self.oblique_shock_relation(self.mach, self.deflection, shock_angle)
            df_shock_angle = self.derivative(f_shock_angle)

            self.shock_angle = newton(f_shock_angle, (0.5, 1.57), fprime=df_shock_angle)

        # Calculate the pressure ratio
        self.pressure_ratio = 1 + 2 * self.gamma / (self.gamma + 1) * (
                self.mach ** 2 * np.sin(self.shock_angle) ** 2 - 1)

        # Calculate the density ratio
        self.density_ratio = (self.gamma + 1) * self.mach ** 2 * np.sin(self.shock_angle) ** 2 / \
                             (2 + (self.gamma - 1) * self.mach ** 2 * np.sin(self.shock_angle) ** 2)

        # Calculate the temperature ratio
        self.temperature_ratio = self.pressure_ratio / self.density_ratio

        # Calculate the mach ratio
        self.mach_ratio = 1/self.mach * np.sqrt(
            (1 + (self.gamma - 1) / 2 * self.mach ** 2 * np.sin(self.shock_angle) ** 2) /
            (self.gamma * self.mach ** 2 * np.sin(self.shock_angle) ** 2 -
             (self.gamma - 1) / 2)) / np.sin(self.shock_angle - self.deflection)

        # Change the shock angle to degrees
        self.shock_angle = np.degrees(self.shock_angle)
        self.deflection = np.degrees(self.deflection)

        return


# class to create a grid and flow based on oblique shock properties
class ObliqueShockData:
    """
    Class to create a grid and flow based on oblique shock properties
    """
    def __init__(self, oblique_shock=ObliqueShock()):
        from src.io.plot3dio import GridIO
        from src.io.plot3dio import FlowIO
        self.oblique_shock = oblique_shock
        self.nx_max = None
        self.ny_max = None
        self.nz_max = None
        self.xpoints = None
        self.ypoints = None
        self.zpoints = None
        self.grid = GridIO('dummy')
        self.flow = FlowIO('dummy')
        self.shock_strength = 'weak'
        self.inlet_temperature = None
        self.inlet_density = None
        self.inlet_pressure = None

    def create_grid(self):
        # create a structured grid from -nx_max to nx_max, 0 to ny_max, 0 to nz_max
        # spacing is the grid spacing
        _xx, _yy, _zz = np.meshgrid(np.linspace(-self.nx_max, self.nx_max, 2*self.xpoints),
                                    np.linspace(0, self.ny_max, self.ypoints),
                                    np.linspace(0, self.nz_max, self.zpoints), indexing='ij')
        # Create a plot3dio similar object
        # n-blocks - one for the current case
        self.grid.nb = 1
        # ni, nj, nk
        self.grid.ni = np.array([2*self.xpoints], dtype='i4')
        self.grid.nj = np.array([self.ypoints], dtype='i4')
        self.grid.nk = np.array([self.zpoints], dtype='i4')
        # grd
        # expand dimensions and stack along the last axis
        self.grid.grd = np.stack((_xx[..., None], _yy[..., None], _zz[..., None]), axis=3)
        # grd_min and grd_max - 2d array to match the rest of the code
        self.grid.grd_min = np.array([[-self.nx_max, 0, 0]])
        self.grid.grd_max = np.array([[self.nx_max, self.ny_max, self.nz_max]])

        # compute metrics
        from src.io.plot3dio import GridIO
        GridIO.compute_metrics(self.grid)
        return

    def create_flow(self):
        # Create a vector with density, shock-normal, shock-tangential, zero velocities, and energy
        # Have pre-shock before x=0 and post-shock after
        if self.shock_strength == 'weak':
            self.oblique_shock.shock_angle = self.oblique_shock.shock_angle[0]
            self.oblique_shock.density_ratio = self.oblique_shock.density_ratio[0]
            self.oblique_shock.pressure_ratio = self.oblique_shock.pressure_ratio[0]
            self.oblique_shock.temperature_ratio = self.oblique_shock.temperature_ratio[0]
            self.oblique_shock.mach_ratio = self.oblique_shock.mach_ratio[0]
        elif self.shock_strength == 'strong':
            self.oblique_shock.shock_angle = self.oblique_shock.shock_angle[1]
            self.oblique_shock.density_ratio = self.oblique_shock.density_ratio[1]
            self.oblique_shock.pressure_ratio = self.oblique_shock.pressure_ratio[1]
            self.oblique_shock.temperature_ratio = self.oblique_shock.temperature_ratio[1]
            self.oblique_shock.mach_ratio = self.oblique_shock.mach_ratio[1]
        # Compute inlet flow properties -- pre-shock
        _density = self.inlet_density
        _velocity = self.oblique_shock.mach * np.sqrt(self.oblique_shock.gamma * self.oblique_shock.gas_constant *
                                                      self.inlet_temperature)
        _x_velocity = _velocity * np.sin(np.radians(self.oblique_shock.shock_angle))
        _y_velocity = _velocity * np.cos(np.radians(self.oblique_shock.shock_angle))
        _z_velocity = 0
        _energy = _density * (self.oblique_shock.gas_constant * self.inlet_temperature / (self.oblique_shock.gamma - 1)
                              + 0.5 * (_x_velocity ** 2 + _y_velocity ** 2 + _z_velocity ** 2))
        # _pre_shock properties
        _pre_shock = np.array([_density, _x_velocity * _density, _y_velocity * _density, _z_velocity * _density,
                               _energy])

        # post-shock properties
        _density_post = _density * self.oblique_shock.density_ratio
        _velocity_post = self.oblique_shock.mach * self.oblique_shock.mach_ratio * np.sqrt(
            self.oblique_shock.gamma * self.oblique_shock.gas_constant
            * self.inlet_temperature * self.oblique_shock.temperature_ratio)
        _x_velocity_post = _velocity_post * np.sin(np.radians(self.oblique_shock.shock_angle
                                                              - self.oblique_shock.deflection))
        _y_velocity_post = _velocity_post * np.cos(np.radians(self.oblique_shock.shock_angle
                                                              - self.oblique_shock.deflection))
        _z_velocity_post = 0
        _energy_post = _density_post * (self.oblique_shock.gas_constant * self.inlet_temperature *
                                        self.oblique_shock.temperature_ratio / (self.oblique_shock.gamma - 1)
                                        + 0.5 * (_x_velocity_post ** 2 + _y_velocity_post ** 2 + _z_velocity_post ** 2))
        _post_shock = np.array([_density_post, _x_velocity_post * _density_post, _y_velocity_post * _density_post,
                                _z_velocity_post * _density_post, _energy_post])

        # Create a vector with density, shock-normal, shock-tangential, zero velocities, and energy
        # Have pre-shock before x=0 and post-shock after

        # create plot3dio flow similar object
        self.flow.nb = 1
        # ni, nj, nk
        self.flow.ni = np.array([2*self.xpoints], dtype='i4')
        self.flow.nj = np.array([self.ypoints], dtype='i4')
        self.flow.nk = np.array([self.zpoints], dtype='i4')
        # mach, aoa/alpha, re, t
        self.flow.mach = self.oblique_shock.mach
        self.flow.alpha = 0.0
        self.flow.rey = (_density * _velocity * (self.grid.grd[1, 0, 0, 0, 0] - self.grid.grd[0, 0, 0, 0, 0])) / 1.8e-5
        self.flow.time = 1.0
        # flow
        self.flow.q = np.zeros((self.flow.ni[0], self.flow.nj[0], self.flow.nk[0], 5, self.flow.nb), dtype='f8')
        self.flow.q[:self.xpoints, ...] = _pre_shock[..., None]
        self.flow.q[self.xpoints:, ...] = _post_shock[..., None]
        return


