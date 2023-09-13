import unittest
from src import Plots
import matplotlib.pyplot as plt
path = ('/Users/kal/Library/CloudStorage/OneDrive-UniversityofCincinnati/Desktop/University of Cincinnati/'
        'DoctoralWork/Codes/project-arrakis/data/shocks/new_start/ragni_data/315e-09/')


def get_grid_and_flow():
    from src.test_cases.oblique_shock_data import ObliqueShock, ObliqueShockData

    # Create oblique shock
    os = ObliqueShock()
    os.mach = 2
    os.deflection = 9
    os.compute()

    # Create grid and flow files
    osd = ObliqueShockData()
    osd.nx_max = 15e-3
    osd.ny_max = 15e-3
    osd.nz_max = 1e-4
    osd.inlet_temperature = 152.778
    osd.inlet_density = 1.2663
    osd.xpoints = 100
    osd.ypoints = 100
    osd.zpoints = 5
    osd.oblique_shock = os
    osd.shock_strength = 'weak'
    osd.create_grid()
    osd.create_flow()

    return osd.grid, osd.flow


class TestPlots(unittest.TestCase):

    def test_plots(self):
        # Create grid and flow files
        grid, flow = get_grid_and_flow()
        p = Plots(file=path + 'loth_weak_dia3.15e-07_ppath_0.npy', grid=grid, flow=flow)
        f = Plots(file=path + 'fluid_weak_dia3.15e-07_ppath_0.npy', grid=grid, flow=flow)
        p.sort_data()
        f.sort_data()

        # plot paths
        ax = p.plot_paths(label='Particle')
        ax = f.plot_paths(ax=ax, label='Fluid')
        ax.set_title('Paths')
        ax.legend()

        # plot velocity
        ax = p.plot_velocity(label='Particle')
        ax = p.plot_fluid_velocity(ax=ax, label='Fluid')
        ax = f.plot_velocity(ax=ax, label='Fluid from p-space')
        ax.set_title('Velocity')
        ax.legend()

        # plot relative mach
        ax = p.plot_relative_mach()
        ax.set_title('Relative Mach')

        # plot relative Reynolds
        ax = p.plot_relative_reynolds()
        ax.set_title('Relative Reynolds')
        plt.show()
        return


if __name__ == "__main__":
    unittest.main()
