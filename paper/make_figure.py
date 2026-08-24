"""Generate the figure used in the JOSS paper.

Releases tracer particles of three diameters into the synthetic oblique-shock
case that lptlib builds in memory, integrates them across the shock, and plots
their speed against streamwise position alongside the fluid itself.

Deterministic and self-contained. No external data files are required.

Run from the repository root:

    python paper/make_figure.py

Parameters
----------
MACH, DEFLECTION   free-stream Mach number and wedge deflection, degrees
DIAMETERS          tracer diameters, meters
DENSITY            tracer material density, kg/m^3
DRAG_MODEL         drag closure used for every inertial particle
SEED               release point, meters, upstream of the shock at x = 0
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lptlib import ObliqueShock, ObliqueShockData, Streamlines  # noqa: E402

MACH = 7.6
DEFLECTION = 20.0
DIAMETERS = (0.5e-6, 1.94e-6, 5.0e-6)
DENSITY = 950.0
DRAG_MODEL = 'loth'
SEED = [-5e-3, 1e-3, 5e-5]
TIME_STEP = 1e-8
MAX_STEPS = 30000
X_WINDOW = (-1.0, 0.75)   # plotted streamwise window, mm

# Sequential single-hue ramp: tracer diameter is an ordered magnitude, so the
# series are stepped light to dark rather than given unrelated hues. Lightness
# is monotonic, which keeps the figure readable in greyscale and in print.
PARTICLE_COLORS = ('#7FB3D8', '#2E76B4', '#10375C')
FLUID_COLOR = '#2B2B2B'


def build_case():
    shock = ObliqueShock()
    shock.mach = MACH
    shock.deflection = DEFLECTION
    shock.compute()

    osd = ObliqueShockData()
    osd.oblique_shock = shock
    osd.nx_max, osd.ny_max, osd.nz_max = 5e-3, 30e-3, 1e-4
    osd.inlet_temperature = 48.20
    osd.inlet_density = 0.07747
    osd.xpoints, osd.ypoints, osd.zpoints = 20, 60, 5
    osd.shock_strength = 'weak'
    osd.create_grid()
    osd.create_flow()
    return osd


def track(osd, diameter, drag_model):
    sl = Streamlines(point=list(SEED),
                     diameter=diameter,
                     density=DENSITY,
                     drag_model=drag_model,
                     time_step=TIME_STEP,
                     interpolation='simple_oblique_shock')
    sl.max_steps = MAX_STEPS
    sl.compute(method='adaptive-ppath', grid=osd.grid, flow=osd.flow)
    x = np.array(sl.streamline)[:, 0]
    speed = np.linalg.norm(np.array(sl.svelocity), axis=1)
    fluid = np.linalg.norm(np.array(sl.fvelocity), axis=1)
    return x, speed, fluid


def _worker(args):
    return track(build_case(), *args)


def main():
    from multiprocessing import Pool
    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(here, '.tracks.npz')
    if os.path.exists(cache) and '--recompute' not in sys.argv:
        z = np.load(cache)
        tracks = [(z[f'x{i}'], z[f'v{i}'], z[f'f{i}']) for i in range(len(DIAMETERS))]
    else:
        with Pool(len(DIAMETERS)) as pool:
            tracks = pool.map(_worker, [(d, DRAG_MODEL) for d in DIAMETERS])
        np.savez_compressed(cache, **{f'{k}{i}': a
                                      for i, t in enumerate(tracks)
                                      for k, a in zip('xvf', t)})
    x_fluid, _, fluid_speed = tracks[0]

    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8.5,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth': 0.8,
        'font.family': 'sans-serif',
    })

    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    ax.axvline(0.0, color='#B0B0B0', lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.annotate('shock', xy=(0.0, 1071), xytext=(3, 0),
                textcoords='offset points', color='#6B6B6B',
                fontsize=8.5, va='top', ha='left')

    ax.plot(x_fluid * 1e3, fluid_speed, color=FLUID_COLOR, lw=2.0,
            ls=(0, (5, 2)), zorder=3, label='fluid')

    for (x, speed, _), d, c in zip(tracks, DIAMETERS, PARTICLE_COLORS):
        ax.plot(x * 1e3, speed, color=c, lw=2.0, zorder=4,
                label=f'{d * 1e6:.2f} ' + r'$\mu$m tracer')

    ax.set_xlabel('streamwise position, mm  (shock at 0)')
    ax.set_ylabel('speed, m s$^{-1}$')
    ax.set_xlim(*X_WINDOW)
    ax.set_ylim(938, 1075)
    ax.grid(True, color='#E4E7EA', lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#9AA3AC')

    # Label the quantity the library exists to measure.
    xb = 0.62
    j = int(np.argmin(np.abs(tracks[-1][0] * 1e3 - xb)))
    vb = tracks[-1][1][j]
    vf = fluid_speed[-1]
    ax.annotate('', xy=(xb, vb), xytext=(xb, vf),
                arrowprops=dict(arrowstyle='<->', color='#6B6B6B', lw=1.0,
                                shrinkA=0, shrinkB=0))
    ax.annotate('velocity bias reported\nby a %.0f ' % (DIAMETERS[-1] * 1e6)
                + r'$\mu$m tracer' + '\n%d m s$^{-1}$' % round(vb - vf),
                xy=(0.27, 968), color='#4A4A4A', fontsize=8.5,
                ha='center', va='center', linespacing=1.4)

    leg = ax.legend(loc='lower left', frameon=True, framealpha=1.0,
                    edgecolor='#DDE1E5', borderpad=0.6, handlelength=2.4)
    leg.get_frame().set_linewidth(0.7)

    fig.tight_layout(pad=0.6)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(here, f'particle_lag.{ext}'), dpi=300,
                    bbox_inches='tight')

    for (x, speed, _), d in zip(tracks, DIAMETERS):
        print(f'{d * 1e6:5.2f} um : {len(x):5d} pts, '
              f'x {x[0] * 1e3:+.2f} to {x[-1] * 1e3:+.2f} mm, '
              f'exit speed {speed[-1]:7.1f} m/s')
    print(f'fluid      : exit speed {fluid_speed[-1]:7.1f} m/s')


if __name__ == '__main__':
    main()
