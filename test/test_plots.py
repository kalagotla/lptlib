"""Smoke tests for the ``Plots`` helper.

``Plots`` is exported from ``lptlib`` and named in ``docs/index.md`` as one of
the ``lptlib.function`` helpers, but every ``plot_*`` method was unreferenced
anywhere in the repository and the module sat at 13 per cent coverage. These
tests drive each public method once, under the Agg backend that ``conftest.py``
forces, and check it returns an axis carrying drawn data. They are deliberately
shallow: the point is that the plotting API runs at all against a real particle
path file, not that any particular pixel is right.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lptlib import Plots

# Column layout written by ``Streamlines._save_data``:
#   0:3   particle position      3:6   particle velocity   6:9  fluid velocity
#   9     time step             10:13  fluid position      13   diameter
#   14    particle density
N_COLUMNS = 15
DIAMETER = 1.94e-06
PARTICLE_DENSITY = 950.0


@pytest.fixture()
def ppath_file(oblique_case, tmp_path):
    """A small, in-domain particle path written in the on-disk ppath layout.

    The file name carries the diameter in exponential notation because
    ``Plots.plot_drag`` parses it back out of the name.
    """
    grid = oblique_case.grid
    lo = grid.grd_min[0]
    hi = grid.grd_max[0]

    n = 6
    # Walk across the shock, staying well inside the block on every axis.
    x = np.linspace(lo[0] * 0.8, hi[0] * 0.8, n)
    y = np.full(n, 0.5 * (lo[1] + hi[1]))
    z = np.full(n, 0.5 * (lo[2] + hi[2]))

    records = np.zeros((n, N_COLUMNS))
    records[:, 0], records[:, 1], records[:, 2] = x, y, z
    # Particle lags the fluid slightly, so relative velocity is non-zero and the
    # drag correlations get a real Reynolds number to work with.
    records[:, 3] = 500.0
    records[:, 6] = 520.0
    records[:, 9] = 1e-9
    records[:, 10], records[:, 11], records[:, 12] = x, y, z
    records[:, 13] = DIAMETER
    records[:, 14] = PARTICLE_DENSITY

    path = tmp_path / f"ppath_{DIAMETER:.2e}.npy"
    np.save(path, records)
    return str(path)


@pytest.fixture()
def plots(ppath_file, oblique_case):
    """A ``Plots`` object with its data frame already built."""
    obj = Plots(ppath_file, grid=oblique_case.grid, flow=oblique_case.flow)
    obj.sort_data()
    yield obj
    plt.close("all")


def _drawn(ax):
    """True when something was actually drawn on the axis."""
    return len(ax.lines) > 0 or len(ax.collections) > 0


def test_sort_data_builds_the_expected_frame(plots):
    """``sort_data`` unpacks the 15 saved columns into named series."""
    assert plots.data is not None
    assert len(plots.data) == 6
    for column in ("x_p", "y_p", "z_p", "v_x", "v_y", "v_z",
                   "u_x", "u_y", "u_z", "time_p", "x_f", "y_f", "z_f",
                   "d_p", "rho_p"):
        assert column in plots.data.columns
    np.testing.assert_allclose(plots.data["d_p"], DIAMETER)
    np.testing.assert_allclose(plots.data["rho_p"], PARTICLE_DENSITY)


def test_compute_variables_adds_derived_columns(plots):
    """``compute_variables`` interpolates the flow onto the path.

    This is the method every ``plot_*`` below depends on, so it is checked once
    on its own: the derived columns must exist, be finite, and be physical.
    """
    plots.compute_variables()

    for column in ("mach", "pressure", "temperature", "density",
                   "velocity_magnitude", "viscosity", "relative_velocity",
                   "relative_mach", "relative_reynolds", "knudsen_number"):
        assert column in plots.data.columns
        assert np.all(np.isfinite(plots.data[column])), column

    assert np.all(plots.data["density"] > 0)
    assert np.all(plots.data["temperature"] > 0)
    assert np.all(plots.data["relative_velocity"] > 0)


def test_plots_draws_on_a_supplied_axis(plots):
    """``plots`` reuses an axis handed to it rather than making a new figure."""
    fig, ax = plt.subplots()
    returned = plots.plots(plots.data["x_p"], plots.data["v_x"], ax=ax)
    assert returned is ax
    assert _drawn(ax)
    assert len(fig.get_axes()) == 1


def test_plots_creates_its_own_axis_when_none_is_given(plots):
    """With no axis, ``plots`` opens one and returns it."""
    ax = plots.plots(plots.data["x_p"], plots.data["v_x"])
    assert ax is not None
    assert _drawn(ax)


@pytest.mark.parametrize("method", ["plot_paths", "plot_velocity",
                                    "plot_fluid_velocity", "plot_relative_mach",
                                    "plot_relative_reynolds"])
def test_plot_methods_run_and_draw(plots, method):
    """Each single-series plot method returns a labeled axis with data on it."""
    ax = getattr(plots, method)()
    assert _drawn(ax), method
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() != ""


@pytest.mark.parametrize("method", ["plot_drag_coefficient", "plot_drag"])
@pytest.mark.parametrize("kwargs", [{"model": "stokes"}, {"label": "stokes"}])
def test_drag_plot_methods_run_and_draw(plots, method, kwargs):
    """The two drag plots run through both ways of naming the drag model.

    ``model=`` is popped before the kwargs reach matplotlib; ``label=`` is
    forwarded to matplotlib *and* doubles as the model name. Both paths are
    exercised because they take different branches.
    """
    ax = getattr(plots, method)(**kwargs)
    assert _drawn(ax), method
    assert "drag_coefficient" in plots.data.columns
    assert np.all(np.isfinite(plots.data["drag_coefficient"]))
    assert np.all(plots.data["drag_coefficient"] >= 0)


def test_plot_drag_uses_the_diameter_from_the_file_name(plots):
    """``plot_drag`` adds a drag column scaled by the parsed diameter."""
    plots.plot_drag(model="stokes")
    assert "drag" in plots.data.columns
    assert np.all(np.isfinite(plots.data["drag"]))

    expected = (0.125 * np.pi * DIAMETER ** 2 * 4230
                * plots.data["relative_velocity"] ** 2
                * plots.data["drag_coefficient"])
    np.testing.assert_allclose(plots.data["drag"], expected, rtol=1e-12)


def test_plot_drag_honours_particle_density(plots):
    """The drag scales linearly with the particle density argument."""
    plots.plot_drag(model="stokes")
    reference = np.array(plots.data["drag"], copy=True)

    plots.data = plots.data.drop(columns=["drag"])
    plots.plot_drag(particle_density=8460, model="stokes")
    np.testing.assert_allclose(plots.data["drag"], 2 * reference, rtol=1e-12)


def test_color_by_builds_a_colour_mapped_line(plots):
    """``color_by`` switches ``plots`` onto the per-segment colored path."""
    ax = plots.plot_paths(color_by="relative_mach")
    assert _drawn(ax)
    # One line per segment, plus the colorbar axis on the figure.
    assert len(ax.lines) == len(plots.data) - 1
    assert len(ax.figure.get_axes()) == 2


def test_get_color_code_returns_no_mapping_without_kwargs(plots):
    """Plain calls take the fast path: no color map, no scalar mappable."""
    first, second = plots.get_color_code()
    assert first is None
    assert second is None


def test_get_color_code_ignores_color_by_when_c_is_given(plots):
    """An explicit ``c`` wins over ``color_by``, and disables the mapping."""
    new_cmap, sm = plots.get_color_code(c="red", color_by="relative_mach")
    assert new_cmap is None
    assert sm is None
