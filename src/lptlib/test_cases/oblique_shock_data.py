# Class to calculate particle response across an oblique shock

import numpy as np
from ..io.plot3dio import GridIO, FlowIO
from ..function.variables import Variables


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

    def compute(self):
        # Use the cubic equation solver to find the shock angle
        # Compute the shock angle if mach and deflection are given
        if self.mach is not None and self.deflection is not None:
            self.deflection = np.radians(self.deflection)

            # calculate coefficients
            A = self.mach ** 2 - 1
            B = 0.5 * (self.gamma + 1) * self.mach ** 4 * np.tan(self.deflection)
            C = (1 + 0.5 * (self.gamma + 1) * self.mach ** 2) * np.tan(self.deflection)
            coeffs = [1, C, -A, (B - A * C)]

            # roots of a cubic equation, two positive solutions
            # (disregard the negative)
            roots = np.array([r for r in np.roots(coeffs) if r > 0])

            thetas = np.arctan(1 / roots)
            self.shock_angle = np.array([np.min(thetas), np.max(thetas)])

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
                              + 0.5 * _velocity**2)
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
                                        + 0.5 * _velocity_post**2)
        _post_shock = np.array([_density_post, _x_velocity_post * _density_post, _y_velocity_post * _density_post,
                                _z_velocity_post * _density_post, _energy_post])

        # Create a vector with density, shock-normal, shock-tangential, zero velocities, and energy
        # Have pre-shock before x=0 and post-shock after

        # compute viscosity using sutherland's law
        _c1 = 1.716e-5 * (273.15 + 110.4) / 273.15 ** 1.5
        viscosity = _c1 * self.inlet_temperature ** 1.5 * 0.4 / (self.inlet_temperature + 110.4)

        # create plot3dio flow similar object
        self.flow.nb = 1
        # ni, nj, nk
        self.flow.ni = np.array([2*self.xpoints], dtype='i4')
        self.flow.nj = np.array([self.ypoints], dtype='i4')
        self.flow.nk = np.array([self.zpoints], dtype='i4')
        # mach, aoa/alpha, re, t
        self.flow.mach = self.oblique_shock.mach
        self.flow.alpha = 0.0
        self.flow.rey = (_density * _velocity * (self.grid.grd[1, 0, 0, 0, 0] - self.grid.grd[0, 0, 0, 0, 0])) / viscosity
        self.flow.time = 1.0
        # flow
        self.flow.q = np.zeros((self.flow.ni[0], self.flow.nj[0], self.flow.nk[0], 5, self.flow.nb), dtype='f8')
        self.flow.q[:self.xpoints, ...] = _pre_shock[..., None]
        self.flow.q[self.xpoints:, ...] = _post_shock[..., None]
        return


class ObliqueShockAlignedData:
    """
    Creates a 3D grid and flow field where the shock plane is aligned along the computed
    shock angle (β) and the incoming flow is horizontal. The shock plane in the x-y plane
    passes through (0, ny_max/2) and is defined by the signed distance
        s = sin(β)*x - cos(β)*(y - ny_max/2)
    with:
      - s < 0 : upstream (pre-shock) state
      - s >= 0 : downstream (post-shock) state
    The z-direction is handled as a full 3D extension.
    """

    def __init__(self, oblique_shock=ObliqueShock()):
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

    def create_grid(self):
        # Create a structured 3D grid: x in [-nx_max, nx_max], y in [0, ny_max], z in [0, nz_max]
        _xx, _yy, _zz = np.meshgrid(np.linspace(-self.nx_max, self.nx_max, 2 * self.xpoints),
                                    np.linspace(0, self.ny_max, self.ypoints),
                                    np.linspace(0, self.nz_max, self.zpoints),
                                    indexing='ij')
        self.grid.nb = 1
        self.grid.ni = np.array([2 * self.xpoints], dtype='i4')
        self.grid.nj = np.array([self.ypoints], dtype='i4')
        self.grid.nk = np.array([self.zpoints], dtype='i4')
        self.grid.grd = np.stack((_xx[..., None], _yy[..., None], _zz[..., None]), axis=3)
        self.grid.grd_min = np.array([[-self.nx_max, 0, 0]])
        self.grid.grd_max = np.array([[self.nx_max, self.ny_max, self.nz_max]])
        GridIO.compute_metrics(self.grid)
        return

    def create_flow(self):
        gamma = self.oblique_shock.gamma
        R = self.oblique_shock.gas_constant
        T1 = self.inlet_temperature
        rho1 = self.inlet_density

        # Select weak or strong shock solution
        if self.shock_strength == 'weak':
            beta_deg = (self.oblique_shock.shock_angle[0] if hasattr(self.oblique_shock.shock_angle, '__iter__')
                        else self.oblique_shock.shock_angle)
            dens_ratio = (self.oblique_shock.density_ratio[0] if hasattr(self.oblique_shock.density_ratio, '__iter__')
                          else self.oblique_shock.density_ratio)
            temp_ratio = (
                self.oblique_shock.temperature_ratio[0] if hasattr(self.oblique_shock.temperature_ratio, '__iter__')
                else self.oblique_shock.temperature_ratio)
            mach_ratio = (self.oblique_shock.mach_ratio[0] if hasattr(self.oblique_shock.mach_ratio, '__iter__')
                          else self.oblique_shock.mach_ratio)
        elif self.shock_strength == 'strong':
            beta_deg = (self.oblique_shock.shock_angle[1] if hasattr(self.oblique_shock.shock_angle, '__iter__')
                        else self.oblique_shock.shock_angle)
            dens_ratio = (self.oblique_shock.density_ratio[1] if hasattr(self.oblique_shock.density_ratio, '__iter__')
                          else self.oblique_shock.density_ratio)
            temp_ratio = (
                self.oblique_shock.temperature_ratio[1] if hasattr(self.oblique_shock.temperature_ratio, '__iter__')
                else self.oblique_shock.temperature_ratio)
            mach_ratio = (self.oblique_shock.mach_ratio[1] if hasattr(self.oblique_shock.mach_ratio, '__iter__')
                          else self.oblique_shock.mach_ratio)
        else:
            raise ValueError("shock_strength must be either 'weak' or 'strong'")

        beta = np.radians(beta_deg)
        delta = np.radians(self.oblique_shock.deflection)

        # Pre-shock: incoming flow is horizontal (along x)
        U1 = self.oblique_shock.mach * np.sqrt(gamma * R * T1)
        pre_shock = np.array([
            rho1,
            rho1 * U1,  # momentum in x
            0,  # momentum in y
            0,  # momentum in z
            rho1 * (R * T1 / (gamma - 1) + 0.5 * U1 ** 2)
        ])

        # Post-shock: properties computed using shock ratios
        rho2 = rho1 * dens_ratio
        T2 = T1 * temp_ratio
        U2 = self.oblique_shock.mach * mach_ratio * np.sqrt(gamma * R * T2)
        # Flow deflected by δ (assumed downward in y)
        post_shock = np.array([
            rho2,
            rho2 * U2 * np.cos(delta),  # x-component
            rho2 * U2 * np.sin(delta),  # y-component
            0,  # z-component remains 0
            rho2 * (R * T2 / (gamma - 1) + 0.5 * U2 ** 2)
        ])

        grd = self.grid.grd  # shape (ni, nj, nk, 3)
        x_coord = grd[..., 0, 0]
        y_coord = grd[..., 1, 0]
        # Compute signed distance from the shock plane in the x-y plane.
        s = x_coord * np.sin(beta) - (y_coord - self.ny_max / 2) * np.cos(beta)

        # compute viscosity using sutherland's law
        _c1 = 1.716e-5 * (273.15 + 110.4) / 273.15 ** 1.5
        viscosity = _c1 * self.inlet_temperature ** 1.5 * 0.4 / (self.inlet_temperature + 110.4)

        self.flow.nb = 1
        self.flow.ni = self.grid.ni
        self.flow.nj = self.grid.nj
        self.flow.nk = self.grid.nk
        self.flow.mach = self.oblique_shock.mach
        self.flow.alpha = 0.0
        dx = self.grid.grd[1, 0, 0, 0, 0] - self.grid.grd[0, 0, 0, 0, 0]
        self.flow.rey = (rho1 * U1 * dx) / viscosity
        self.flow.time = 1.0

        ni = self.flow.ni[0]
        nj = self.flow.nj[0]
        nk = self.flow.nk[0]
        self.flow.q = np.zeros((ni, nj, nk, 5, self.flow.nb), dtype='f8')
        # Partition the 3D grid based on s: pre_shock for s < 0, post_shock otherwise.
        mask = s  < 0
        for var in range(5):
            self.flow.q[..., var, 0][mask] = pre_shock[var]
            self.flow.q[..., var, 0][~mask] = post_shock[var]
        return

