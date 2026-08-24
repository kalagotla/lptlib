"""Deterministic tests for the search, interpolation, integration and
particle-tracking stack on the synthetic oblique-shock case.

Every test here runs against the in-memory synthetic grid and flow, so the full
Lagrangian pipeline (cell search, curvilinear coordinate transforms, tri-linear
interpolation, streamline and particle-path integration, and the stochastic
ensemble driver) is exercised in CI without any external data files. Where the
flow is piecewise constant the interpolated answers are known exactly, so the
assertions check real numbers rather than mere shapes.
"""

import os
import tempfile

import numpy as np
import pytest

from lptlib.streamlines import (Search, Interpolation, Integration,
                                StochasticModel, Particle, SpawnLocations)


# ----------------------------- Search ----------------------------------------

def test_search_finds_cell_for_interior_point(synthetic_grid, upstream_point):
    """An interior point resolves to a hexahedral cell in block 0."""
    idx = Search(synthetic_grid, list(upstream_point))
    idx.compute(method="p-space")
    assert idx.block == 0
    assert idx.cell.shape == (8, 3)


def test_search_reports_out_of_domain(synthetic_grid):
    """A point well outside the grid is flagged and returns no cell."""
    idx = Search(synthetic_grid, [1e3, 1e3, 1e3])
    idx.compute(method="p-space")
    assert idx.cell is None
    assert "not in the domain" in idx.info


def test_search_recognizes_a_node(synthetic_grid):
    """Querying an exact grid node is reported as a node hit."""
    node = synthetic_grid.grd[4, 4, 2, :, 0]
    idx = Search(synthetic_grid, node)
    idx.compute(method="distance")
    assert "is a node in the domain" in idx.info


def test_search_methods_agree_on_cell(synthetic_grid):
    """The p-space and distance searches select the same cell for a point.

    A point at the center of a cell (rather than on a shared node or face) has
    an unambiguous owning cell, so the two independent search strategies must
    agree on it.
    """
    grd = synthetic_grid.grd
    # Midpoint of the hexahedral cell with corner node (3, 5, 2).
    interior = 0.5 * (grd[3, 5, 2, :, 0] + grd[4, 6, 3, :, 0])
    a = Search(synthetic_grid, list(interior))
    a.compute(method="p-space")
    b = Search(synthetic_grid, list(interior))
    b.compute(method="distance")
    assert (a.cell == b.cell).all()


def test_c2p_returns_node_coordinates_exactly(synthetic_grid):
    """The c-space to p-space map sends integer indices to the grid node.

    For an integer computational coordinate the tri-linear weights collapse to
    a single node, so ``c2p`` must return that node's physical coordinates
    exactly.
    """
    idx = Search(synthetic_grid, [0.0, 0.0, 0.0])
    idx.block = 0
    physical = idx.c2p(np.array([3.0, 5.0, 2.0]))
    assert np.allclose(physical, synthetic_grid.grd[3, 5, 2, :, 0])


def test_p2c_c2p_round_trip(synthetic_grid):
    """Physical to computational and back recovers the original point.

    Starting from a known grid node, ``p2c`` recovers its computational index
    and ``c2p`` maps it back to the same physical location.
    """
    node = synthetic_grid.grd[5, 6, 2, :, 0].copy()
    idx = Search(synthetic_grid, node)
    idx.block = 0
    cpoint = idx.p2c(node)
    assert cpoint is not None
    back = idx.c2p(cpoint)
    assert np.allclose(back, node, atol=1e-8)


# -------------------------- Interpolation ------------------------------------

def test_interpolation_recovers_constant_upstream_state(synthetic_flow,
                                                        synthetic_grid,
                                                        upstream_point,
                                                        oblique_case):
    """Interpolation in the uniform upstream region returns the exact state.

    The flow is piecewise constant, so interpolating anywhere in the pre-shock
    region must return the pre-shock conserved-variable vector exactly.
    """
    idx = Search(synthetic_grid, list(upstream_point))
    idx.compute(method="p-space")
    interp = Interpolation(synthetic_flow, idx)
    interp.compute(method="p-space")
    assert interp.q.shape == (1, 1, 1, 5, 1)
    pre_shock = oblique_case.flow.q[0, 0, 0, :, 0]
    assert np.allclose(interp.q.reshape(-1), pre_shock)


@pytest.mark.parametrize("interp_method,search_method", [
    ("p-space", "p-space"),
    ("c-space", "c-space"),
    ("rbf-p-space", "p-space"),
    ("rbf-c-space", "c-space"),
    ("simple_oblique_shock", "distance"),
])
def test_all_interpolation_methods_recover_upstream_state(
        synthetic_flow, synthetic_grid, upstream_point, oblique_case,
        interp_method, search_method):
    """Every interpolation scheme returns the exact upstream state.

    Physical-space, computational-space, both radial-basis variants, and the
    nearest-neighbor oblique-shock scheme must all reproduce the constant
    pre-shock conserved-variable vector in the uniform upstream region.
    """
    idx = Search(synthetic_grid, list(upstream_point))
    idx.compute(method=search_method)
    interp = Interpolation(synthetic_flow, idx)
    interp.compute(method=interp_method)
    pre_shock = oblique_case.flow.q[0, 0, 0, :, 0]
    assert np.allclose(np.asarray(interp.q).reshape(-1), pre_shock)


def test_interpolation_recovers_constant_downstream_state(synthetic_flow,
                                                         synthetic_grid,
                                                        oblique_case):
    """Interpolation deep in the post-shock region returns the exact state."""
    grd = synthetic_grid.grd
    point = [grd[-3, 5, 2, 0, 0], grd[-3, 5, 2, 1, 0], grd[-3, 5, 2, 2, 0]]
    idx = Search(synthetic_grid, point)
    idx.compute(method="p-space")
    interp = Interpolation(synthetic_flow, idx)
    interp.compute(method="p-space")
    post_shock = oblique_case.flow.q[-1, 0, 0, :, 0]
    assert np.allclose(interp.q.reshape(-1), post_shock)


def test_variables_velocity_matches_momentum_over_density(synthetic_flow,
                                                         synthetic_grid,
                                                        upstream_point,
                                                        oblique_case):
    """Point velocity equals momentum over density from the interpolated state."""
    from lptlib.function import Variables

    idx = Search(synthetic_grid, list(upstream_point))
    idx.compute(method="p-space")
    interp = Interpolation(synthetic_flow, idx)
    interp.compute(method="p-space")
    variables = Variables(interp)
    variables.compute_velocity()

    pre_shock = oblique_case.flow.q[0, 0, 0, :, 0]
    expected = pre_shock[1:4] / pre_shock[0]
    assert np.allclose(variables.velocity.reshape(3), expected)


# --------------------------- Integration -------------------------------------

def test_integration_step_moves_along_the_flow(synthetic_flow, synthetic_grid,
                                                upstream_point, oblique_case):
    """A fluid-streamline step advances the point along the local velocity.

    In the upstream region the velocity has positive x and y components, so a
    single forward step must increase both coordinates while staying in-domain.
    """
    idx = Search(synthetic_grid, list(upstream_point))
    idx.compute(method="p-space")
    interp = Interpolation(synthetic_flow, idx)
    interp.compute(method="p-space")
    intg = Integration(interp)
    new_point = intg.compute(method="p-space", time_step=1e-8)

    assert new_point is not None
    displacement = np.asarray(new_point) - np.asarray(upstream_point)
    pre_shock = oblique_case.flow.q[0, 0, 0, :, 0]
    velocity = pre_shock[1:4] / pre_shock[0]
    assert displacement[0] > 0 and velocity[0] > 0
    assert displacement[1] > 0 and velocity[1] > 0
    x_min = synthetic_grid.grd_min[0]
    x_max = synthetic_grid.grd_max[0]
    assert np.all(np.asarray(new_point) >= x_min - 1e-9)
    assert np.all(np.asarray(new_point) <= x_max + 1e-9)


def test_particle_relaxes_toward_the_fluid(synthetic_flow, synthetic_grid,
                                           upstream_point):
    """Drag pulls a slipping particle toward the local fluid velocity.

    Launching a particle with a velocity that differs from the fluid, one drag
    step reduces the slip magnitude and increases the particle speed toward the
    fluid speed, the expected behavior of a relaxing inertial tracer.
    """
    idx = Search(synthetic_grid, list(upstream_point))
    idx.compute(method="p-space")
    interp = Interpolation(synthetic_flow, idx)
    interp.compute(method="p-space")
    intg = Integration(interp)

    v_start = np.array([100.0, 0.0, 0.0])
    _x_new, v_new, u_f = intg.compute_ppath(
        diameter=281e-9, density=813.0, velocity=v_start.copy(),
        method="pRK4", time_step=1e-8, drag_model="stokes")

    slip_before = np.linalg.norm(v_start - u_f)
    slip_after = np.linalg.norm(np.asarray(v_new) - u_f)
    assert slip_after < slip_before
    # The particle speeds up toward the fluid but has not overtaken it.
    assert np.linalg.norm(v_start) < np.linalg.norm(v_new) < np.linalg.norm(u_f)


# ----------------------- Stochastic ensemble ---------------------------------

def test_particle_distribution_is_deterministic_with_zero_spread():
    """A zero-spread Gaussian yields every particle at the mean diameter."""
    particle = Particle()
    particle.min_dia = particle.max_dia = particle.mean_dia = 281e-9
    particle.std_dia = 0.0
    particle.density = 813.0
    particle.n_concentration = 25
    particle.compute_distribution()
    assert particle.particle_field.shape == (25,)
    assert np.allclose(particle.particle_field, 281e-9)


def test_spawn_locations_form_a_vertical_line():
    """Spawn locations trace the requested straight line between end points."""
    particle = Particle()
    particle.n_concentration = 5
    spawn = SpawnLocations(particle)
    spawn.x_min = -1e-3
    spawn.z_min = 5e-5
    spawn.y_min, spawn.y_max = 2e-3, 13e-3
    spawn.compute()
    assert spawn.locations.shape == (5, 3)
    # x and z are held constant; y spans the requested range.
    assert np.allclose(spawn.locations[:, 0], -1e-3)
    assert np.allclose(spawn.locations[:, 2], 5e-5)
    assert spawn.locations[0, 1] == pytest.approx(2e-3)
    assert spawn.locations[-1, 1] == pytest.approx(13e-3)


def test_fluid_streamline_integrates_to_domain_exit(oblique_case):
    """A fluid streamline advances through the field and terminates at the exit.

    Launched from a point near the outflow, the massless streamline is
    integrated step by step until it leaves the grid, producing a multi-step
    trajectory. This drives the ``p-space`` streamline branch end to end.
    """
    particle = Particle()
    particle.min_dia = particle.max_dia = particle.mean_dia = 281e-9
    particle.std_dia = 0.0
    particle.density = 813.0
    particle.n_concentration = 1
    particle.compute_distribution()

    spawn = SpawnLocations(particle)
    spawn.x_min = 0.013
    spawn.z_min = 5e-5
    spawn.y_min, spawn.y_max = 6e-3, 6e-3
    spawn.compute()

    filepath = tempfile.mkdtemp() + "/"
    model = StochasticModel(particle, spawn, grid=oblique_case.grid,
                            flow=oblique_case.flow)
    model.method = "p-space"
    model.drag_model = "stokes"
    model.search = "p-space"
    model.interpolation = "p-space"
    model.time_step = 1e-7
    model.max_time_step = 1e-7
    model.adaptivity = 0.01
    model.filepath = filepath

    model.serial()
    trajectory = np.load(filepath + "ppath_0.npy")
    # The streamline records several steps before exiting the domain.
    assert trajectory.shape[0] > 5
    assert trajectory.shape[1] == 15
    assert np.all(np.isfinite(trajectory))
    # Every recorded position stays within the grid bounds.
    x_min = oblique_case.grid.grd_min[0]
    x_max = oblique_case.grid.grd_max[0]
    positions = trajectory[:, :3]
    assert np.all(positions[:, 0] >= x_min[0] - 1e-9)
    assert np.all(positions[:, 0] <= x_max[0] + 1e-9)


def test_stochastic_serial_run_tracks_particles(oblique_case):
    """A bounded serial ensemble tracks every particle and writes its path.

    Two tracers are launched into the synthetic field with a hard step cap so
    the run stays fast. Each produces a saved trajectory array whose columns are
    the recorded per-step state, confirming the whole particle-tracking loop
    executes end to end.
    """
    particle = Particle()
    particle.min_dia = particle.max_dia = particle.mean_dia = 281e-9
    particle.std_dia = 0.0
    particle.density = 813.0
    particle.n_concentration = 2
    particle.compute_distribution()

    spawn = SpawnLocations(particle)
    spawn.x_min = -1e-3
    spawn.z_min = 5e-5
    spawn.y_min, spawn.y_max = 2e-3, 13e-3
    spawn.compute()

    filepath = tempfile.mkdtemp() + "/"
    model = StochasticModel(particle, spawn, grid=oblique_case.grid,
                            flow=oblique_case.flow)
    model.method = "adaptive-ppath"
    model.drag_model = "stokes"
    model.search = "p-space"
    model.time_step = 1e-9
    model.max_time_step = 1e-7
    model.adaptivity = 0.01
    model.max_steps = 15
    model.filepath = filepath

    result = model.serial()
    assert len(result) == 2

    saved = sorted(f for f in os.listdir(filepath) if f.endswith(".npy"))
    assert saved == ["ppath_0.npy", "ppath_1.npy"]
    trajectory = np.load(filepath + saved[0])
    # Bounded by max_steps, 15 recorded state columns per step, all finite.
    assert trajectory.ndim == 2
    assert 2 <= trajectory.shape[0] <= 15
    assert trajectory.shape[1] == 15
    assert np.all(np.isfinite(trajectory))
