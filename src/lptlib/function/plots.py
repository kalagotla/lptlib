# Create plots for the data saved from streamlines file

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import re
import pandas as pd
from ..streamlines import Search, Interpolation
from .variables import Variables


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
        d_p = _data[:, 13]
        rho_p = _data[:, 14]
        # create a dataframe from _data variable
        self.data = pd.DataFrame({'x_p': x_p, 'y_p': y_p, 'z_p': z_p, 'v_x': v_x, 'v_y': v_y, 'v_z': v_z,
                                  'u_x': u_x, 'u_y': u_y, 'u_z': u_z, 'time_p': time_p, 'x_f': x_f,
                                  'y_f': y_f, 'z_f': z_f, 'd_p': d_p, 'rho_p': rho_p})

        return

    # compute required variables as a method and add it to the dataframe
    def compute_variables(self):
        """
        Compute the required variables
        Args:
            self:

        Returns:

        """
        mach, pressure, temperature, density, velocity_magnitude = [], [], [], [], []
        viscosity, gamma = [], 1.4

        # loop over the data and compute the variables
        for i in range(len(self.data['x_p'])):
            idx = Search(self.grid, [self.data['x_p'][i], self.data['y_p'][i], self.data['z_p'][i]])
            interp = Interpolation(self.flow, idx)
            # interp.adaptive = 'shock'
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
            gamma = var.gamma

        # add those variables to the dataframe
        self.data['mach'] = mach
        self.data['pressure'] = pressure
        self.data['temperature'] = temperature
        self.data['density'] = density
        self.data['velocity_magnitude'] = velocity_magnitude
        self.data['viscosity'] = viscosity
        self.data['relative_velocity'] = ((self.data['v_x'] - self.data['u_x'])**2 +
                                          (self.data['v_y'] - self.data['u_y'])**2 +
                                          (self.data['v_z'] - self.data['u_z'])**2)**0.5
        self.data['relative_mach'] = (self.data['relative_velocity'] * self.data['mach']
                                      / self.data['velocity_magnitude'])
        # Extract diameter from the file name
        # Find the values that surround the exponential notation
        try:
            matches = re.findall(r'([-+]?\d*\.\d*|\d+)e([-+]?\d+)', self.file)
            # Create the diameter from the matches
            diameter = float(matches[-1][0])*10**int(matches[-1][1])
        except:
            diameter = self.data['d_p'][0]
        self.data['relative_reynolds'] = (self.data['relative_velocity'] * self.data['density'] * diameter
                                          / self.data['viscosity'])
        self.data['knudsen_number'] = ((self.data['relative_mach'] / self.data['relative_reynolds'])
                                       * np.sqrt(np.pi * gamma / 2))
        # apply map to convert the elements in the dataframe to floats only if the data is an array or list
        self.data = self.data.map(lambda x: x.reshape(-1)[0] if isinstance(x, np.ndarray) else x)
        # convert all NaN values to zero
        self.data = self.data.fillna(0)

        return

    # get color code for the plots
    def get_color_code(self, **kwargs):
        """
        Get the color code for the plots
        Args:
            kwargs: arguments
                c: color code (variable name)
                cmap: color map

        Returns:
            color code
        """
        # Checks for the default color code
        # This is the case when c is not used to plot
        # This gives color_by precedence over c
        if kwargs.get("c") is not None:
            sm = None
            new_cmap = None
            return new_cmap, sm
        # This is the case when plot is used without any kwargs
        if kwargs.get('c') is None and kwargs.get('color_by') is None:
            sm = None
            new_cmap = None
            return sm, new_cmap

        # Compute the variables if not already computed
        if kwargs.get('color_by') not in self.data.columns:
            self.compute_variables()

        # get the color map -- This is basically setting default values if nothing is provided
        if kwargs.get('cmap') is None:
            cmap = plt.get_cmap('viridis')
        else:
            cmap = plt.get_cmap(kwargs.get('cmap'))

        # Create new color map based on the color_by variable
        norm = colors.Normalize(vmin=self.data[kwargs.get('color_by')].min(),
                                vmax=self.data[kwargs.get('color_by')].max())
        new_cmap = []
        for i in range(len(self.data['x_p']) - 1):
            new_cmap.append(cmap(norm(self.data[kwargs.get('color_by')][i])))
        # create a map for colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        return new_cmap, sm

    def plots(self, x, y, ax=None, **kwargs):
        """
        Plot the data
        Args:
            self: class
            x: x data
            y: y data
            ax: axis
            kwargs: arguments

        Returns:
            axis
        """
        if ax is None:
            fig, ax = plt.subplots()

        new_cmap, sm = self.get_color_code(**kwargs)
        if sm is None:
            ax.plot(x, y, **kwargs)
        else:
            for i in range(len(x) - 1):
                ax.plot(x[i:i+2], y[i:i+2], c=new_cmap[i])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(kwargs.get('color_by'))
        return ax

    # plot particle paths
    def plot_paths(self, ax=None, **kwargs):
        """
        Plot the particle paths
        """
        ax = self.plots(self.data['x_p'], self.data['y_p'], ax=ax, **kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Particle paths')
        return ax

    # plot particle velocity
    def plot_velocity(self, ax=None, **kwargs):
        """
        Plot the particle velocity
        """
        ax = self.plots(self.data['x_p'], self.data['v_x'], ax=ax, **kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('v_x')
        return ax

    # plot fluid velocity on top of particle path
    def plot_fluid_velocity(self, ax=None, **kwargs):
        """
        Plot the fluid velocity
        """
        ax = self.plots(self.data['x_p'], self.data['u_x'], ax=ax, **kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('u_x')
        return ax

    # plot relative mach number
    def plot_relative_mach(self, ax=None, **kwargs):
        """
        Plot the relative mach number
        """
        if 'relative_mach' not in self.data.columns:
            self.compute_variables()
        ax = self.plots(self.data['x_p'], self.data['relative_mach'], ax=ax, **kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('Relative Mach')
        return ax

    # plot relative reynolds number
    def plot_relative_reynolds(self, ax=None, **kwargs):
        """
        Plot the relative reynolds number
        """
        if 'relative_reynolds' not in self.data.columns:
            self.compute_variables()
        ax = self.plots(self.data['x_p'], self.data['relative_reynolds'], ax=ax, **kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('Relative Reynolds')
        return ax

    def plot_drag_coefficient(self, ax=None, **kwargs):
        """
        Plot the drag coefficient
        """
        # added if in case drag is called before drag coefficient
        if 'drag_coefficient' not in self.data.columns:
            if 'relative_reynolds' not in self.data.columns:
                self.compute_variables()
            # Need to compute it here to get the label tag for setting the model
            var = Variables(self.flow)
            drag_coefficient = []
            if kwargs.get('model') is None:
                for _re, _mach in zip(self.data['relative_reynolds'], self.data['relative_mach']):
                    drag_coefficient.append(var.compute_drag_coefficient(_re=_re, _mach=_mach, _model=kwargs.get('label')))
            else:
                for _re, _mach in zip(self.data['relative_reynolds'], self.data['relative_mach']):
                    drag_coefficient.append(var.compute_drag_coefficient(_re=_re, _mach=_mach, _model=kwargs.get('model')))
                kwargs.pop('model')  # Do this to remove model; so that it doesn't get passed to plots
            self.data['drag_coefficient'] = drag_coefficient
            # convert all NaN values to zero
            self.data = self.data.fillna(0)

        # Need to make sure that the model is not passed to plots
        if kwargs.get("model") is not None:
            kwargs.pop("model")

        # do the actual plotting
        ax = self.plots(self.data['x_p'], self.data['drag_coefficient'], ax=ax, **kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('Drag Coefficient')
        return ax

    def plot_drag(self, particle_density=4230, ax=None, **kwargs):
        """
        Plot the drag
        Args:
            particle_density:
            ax:
            **kwargs:

        Returns:

        """
        if 'drag_coefficient' not in self.data.columns:
            if 'relative_reynolds' not in self.data.columns:
                self.compute_variables()
            # Need to compute it here to get the label tag for setting the model
            var = Variables(self.flow)
            drag_coefficient = []
            if kwargs.get("model") is None:
                for _re, _mach in zip(self.data["relative_reynolds"], self.data["relative_mach"]):
                    drag_coefficient.append(
                        var.compute_drag_coefficient(_re=_re, _mach=_mach, _model=kwargs.get("label"))
                    )
            else:
                for _re, _mach in zip(self.data["relative_reynolds"], self.data["relative_mach"]):
                    drag_coefficient.append(
                        var.compute_drag_coefficient(_re=_re, _mach=_mach, _model=kwargs.get("model"))
                    )
                kwargs.pop("model")  # Do this to remove model; so that it doesn't get passed to plots
            self.data["drag_coefficient"] = drag_coefficient
            # convert all NaN values to zero
            self.data = self.data.fillna(0)

        # Need to make sure that the model is not passed to plots
        if kwargs.get("model") is not None:
            kwargs.pop("model")

        # Extract diameter from the file name
        # Find the values that surround the exponential notation
        matches = re.findall(r'([-+]?\d*\.\d*|\d+)e([-+]?\d+)', self.file)
        # Create the diameter from the matches
        diameter = float(matches[-1][0])*10**int(matches[-1][1])
        self.data["drag"] = (0.125 * np.pi * diameter**2 * particle_density
                             * self.data["relative_velocity"] ** 2 * self.data["drag_coefficient"])
        ax = self.plots(self.data["x_p"], self.data["drag"], ax=ax, **kwargs)
        ax.set_xlabel("x")
        ax.set_ylabel("Drag")
        return ax

