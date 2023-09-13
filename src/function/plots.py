# Create plots for the data saved from streamlines file

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import seaborn as sns
import re
import pandas as pd
from src.streamlines import Search, Interpolation
from src.function import Variables


class Plots:
    """
    Module to plot data
    """
    def __init__(self, file, grid=None, flow=None):
        self.file = file
        self.grid = grid
        self.flow = flow
        self.data = None

    # Extract and organize data from the flow and particle files
    def sort_data(self):
        """
        Extract and organize data from the files
        """
        # load the file
        _data = np.load(self.file)
        # Extract the data from the file
        x_p = _data[:, 0]
        y_p = _data[:, 1]
        z_p = _data[:, 2]
        v_x = _data[:, 3]
        v_y = _data[:, 4]
        v_z = _data[:, 5]
        u_x = _data[:, 6]
        u_y = _data[:, 7]
        u_z = _data[:, 8]
        time_p = _data[:, 9]
        x_f = _data[:, 10]
        y_f = _data[:, 11]
        z_f = _data[:, 12]
        # create a dataframe from _data variable
        self.data = pd.DataFrame({'x_p': x_p, 'y_p': y_p, 'z_p': z_p, 'v_x': v_x, 'v_y': v_y, 'v_z': v_z,
                                  'u_x': u_x, 'u_y': u_y, 'u_z': u_z, 'time_p': time_p, 'x_f': x_f,
                                  'y_f': y_f, 'z_f': z_f})

        return

    # plot particle paths
    def plot_paths(self, ax=None, **kwargs):
        """
        Plot the particle paths
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.data['x_p'], self.data['y_p'], **kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Particle paths')
        return ax

    # plot particle velocity
    def plot_velocity(self, ax=None, **kwargs):
        """
        Plot the particle velocity
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.data['x_p'], self.data['v_x'], **kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('v_x')
        return ax

    # plot fluid velocity on top of particle path
    def plot_fluid_velocity(self, ax=None, **kwargs):
        """
        Plot the fluid velocity
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.data['x_p'], self.data['u_x'], **kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('u_x')
        return ax

    # compute required variables as a static method and add it to the dataframe
    @staticmethod
    def compute_variables(self):
        """
        Compute the required variables
        Args:
            self:

        Returns:

        """
        mach, pressure, temperature, density, velocity_magnitude = [], [], [], [], []
        viscosity = []
        for i in range(len(self.data['x_p'])):
            idx = Search(self.grid, [self.data['x_p'][i], self.data['y_p'][i], self.data['z_p'][i]])
            interp = Interpolation(self.flow, idx)
            interp.adaptive = 'shock'
            idx.compute(method='p-space')
            interp.compute(method='p-space')
            var = Variables(interp)
            var.compute()
            mach.append(var.mach)
            pressure.append(var.pressure)
            temperature.append(var.temperature)
            density.append(var.density)
            velocity_magnitude.append(var.velocity_magnitude)
            viscosity.append(var.viscosity)
        self.data['mach'] = mach
        self.data['pressure'] = pressure
        self.data['temperature'] = temperature
        self.data['density'] = density
        self.data['velocity_magnitude'] = velocity_magnitude
        self.data['viscosity'] = viscosity

        return

    # plot relative mach number
    def plot_relative_mach(self, ax=None, **kwargs):
        """
        Plot the relative mach number
        """
        if ax is None:
            fig, ax = plt.subplots()

        # compute relative mach and add it to the dataframe
        self.compute_variables(self)
        self.data['relative_mach'] = (((self.data['v_x'] - self.data['u_x'])**2 +
                                      (self.data['v_y'] - self.data['u_y'])**2 +
                                      (self.data['v_z'] - self.data['u_z'])**2)**0.5 * self.data['mach']
                                      / self.data['velocity_magnitude'])

        ax.plot(self.data['x_p'], self.data['relative_mach'], **kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('Relative Mach')
        return ax

    # plot relative reynolds number
    def plot_relative_reynolds(self, ax=None, **kwargs):
        """
        Plot the relative reynolds number
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Extract diameter from the file name
        matches = re.findall(r'([-+]?\d*\.\d*|\d+)e([-+]?\d+)', self.file)
        # Create the diameter from the matches
        diameter = float(matches[0][0])*10**int(matches[0][1])

        # if plot_relative_mach is not called before, run compute_variables
        if 'relative_mach' not in self.data.columns:
            self.compute_variables(self)

        # compute relative mach and add it to the dataframe
        self.data['relative_reynolds'] = (((self.data['v_x'] - self.data['u_x'])**2 +
                                           (self.data['v_y'] - self.data['u_y'])**2 +
                                           (self.data['v_z'] - self.data['u_z'])**2)**0.5 *
                                          self.data['density'] * diameter
                                          / (self.data['viscosity']) * self.data['velocity_magnitude'])

        ax.plot(self.data['x_p'], self.data['relative_reynolds'], **kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('Relative Reynolds')
        return ax

