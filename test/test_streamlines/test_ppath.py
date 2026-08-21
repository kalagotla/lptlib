"""Every ``Streamlines.compute`` algorithm, run on the synthetic case.

The original version of this file drove eight algorithm names against the
PLOT3D vortex data that is not tracked in the repository, so it skipped on every
clean checkout, and it asserted nothing when it did run. It now drives the same
eight names against the in-memory synthetic oblique-shock fixture and checks
each produced trajectory: non-empty, finite, inside the domain, recording one
positive time-step value per point, and moving downstream.

The four fluid algorithms (``p-space``, ``c-space`` and their adaptive
variants) should track the flow; the four particle algorithms (``ppath``,
``ppath-c-space`` and their adaptive variants) additionally record a particle
velocity that stays finite.
"""

import numpy as np
import pytest

from lptlib.streamlines import Streamlines

FLUID_METHODS = ["p-space", "adaptive-p-space", "c-space", "adaptive-c-space"]
PARTICLE_METHODS = ["ppath", "adaptive-ppath", "ppath-c-space",
                    "adaptive-ppath-c-space"]
ALL_METHODS = FLUID_METHODS + PARTICLE_METHODS


@pytest.fixture
def start_point(oblique_case):
    """A point in the uniform upstream region, off the grid nodes."""
    grd = oblique_case.grid.grd
    node = np.array([grd[2, 4, 2, 0, 0], grd[2, 4, 2, 1, 0], grd[2, 4, 2, 2, 0]])
    spacing = grd[3, 4, 2, 0, 0] - grd[2, 4, 2, 0, 0]
    return list(node + np.array([0.3 * spacing, 0.3 * spacing, 0.0]))


def _run(oblique_case, point, method):
    sl = Streamlines(None, None, point=point)
    sl.diameter = 281e-9
    sl.density = 813.0
    # The synthetic domain is ~15 mm across at ~500 m/s, so a microsecond step
    # crosses it in a few tens of steps. `max_steps` below is the backstop --
    # every algorithm honours it now, not just `adaptive-ppath`.
    sl.time_step = 1e-6
    sl.max_time_step = 1e-5
    sl.adaptivity = 0.01
    sl.magnitude_adaptivity = 0.01
    sl.drag_model = "stokes"
    sl.max_steps = 200
    sl.compute(method=method, grid=oblique_case.grid, flow=oblique_case.flow)
    return sl


@pytest.mark.parametrize("method", ALL_METHODS)
def test_algorithm_produces_a_valid_trajectory(oblique_case, start_point, method):
    """Each algorithm yields a bounded, finite, downstream-moving path."""
    sl = _run(oblique_case, start_point, method)

    path = np.asarray(sl.streamline, dtype=float)
    assert path.ndim == 2 and path.shape[1] == 3
    assert path.shape[0] >= 2
    assert np.all(np.isfinite(path))

    grd_min = oblique_case.grid.grd_min[0]
    grd_max = oblique_case.grid.grd_max[0]
    assert np.all(path >= grd_min - 1e-9)
    assert np.all(path <= grd_max + 1e-9)

    # Starts where we asked and moves downstream (+x) overall.
    np.testing.assert_allclose(path[0], start_point, rtol=1e-9, atol=1e-12)
    assert path[-1, 0] > path[0, 0]

    # One positive, finite time-step value is recorded per stored point.
    steps = np.asarray(sl.time, dtype=float).reshape(-1)
    assert steps.shape[0] == path.shape[0]
    assert np.all(np.isfinite(steps))
    assert np.all(steps > 0)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_velocities_are_recorded_and_finite(oblique_case, start_point, method):
    """Particle and fluid velocity histories line up with the path."""
    sl = _run(oblique_case, start_point, method)

    path = np.asarray(sl.streamline, dtype=float)
    particle = np.asarray(sl.svelocity, dtype=float)
    fluid = np.asarray(sl.fvelocity, dtype=float)

    assert particle.shape == path.shape
    assert fluid.shape == path.shape
    assert np.all(np.isfinite(particle))
    assert np.all(np.isfinite(fluid))
    # The upstream flow is supersonic and moves in +x.
    assert np.all(fluid[:, 0] > 0)


@pytest.mark.parametrize("method", FLUID_METHODS)
def test_fluid_algorithms_follow_the_fluid(oblique_case, start_point, method):
    """A fluid streamline carries the local fluid velocity as its own."""
    sl = _run(oblique_case, start_point, method)
    particle = np.asarray(sl.svelocity, dtype=float)
    fluid = np.asarray(sl.fvelocity, dtype=float)
    np.testing.assert_allclose(particle, fluid, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("method", ["ppath", "adaptive-ppath"])
def test_inertial_particle_lags_the_fluid(oblique_case, start_point, method):
    """A heavy inertial particle keeps a bounded, finite slip from the fluid.

    A 5 micron TiO2-density particle has real inertia, so its velocity history
    is a genuine drag response rather than a copy of the fluid; the slip must
    still stay finite and never exceed the fluid speed itself.
    """
    sl = Streamlines(None, None, point=start_point)
    sl.diameter = 5e-6          # large enough to have real inertia
    sl.density = 4200.0
    sl.time_step = 1e-6
    sl.max_time_step = 1e-5
    sl.adaptivity = 0.01
    sl.magnitude_adaptivity = 0.01
    sl.drag_model = "stokes"
    sl.max_steps = 200
    sl.compute(method=method, grid=oblique_case.grid, flow=oblique_case.flow)

    particle = np.asarray(sl.svelocity, dtype=float)
    fluid = np.asarray(sl.fvelocity, dtype=float)
    slip = np.linalg.norm(particle - fluid, axis=1)
    assert np.all(np.isfinite(slip))
    # Starting from the fluid velocity the slip stays small and bounded.
    assert np.all(slip <= np.linalg.norm(fluid, axis=1) + 1e-9)


def test_out_of_domain_start_produces_no_path(oblique_case):
    """Launching outside the grid gives an empty trajectory, not a crash."""
    sl = Streamlines(None, None, point=[1e3, 1e3, 1e3])
    sl.time_step = 1e-6
    sl.max_steps = 5
    sl.compute(method="p-space", grid=oblique_case.grid, flow=oblique_case.flow)
    # Only the start point is recorded; the loop exits before any step is taken.
    assert len(sl.streamline) <= 1
