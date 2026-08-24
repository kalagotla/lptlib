"""Integration-scheme tests on the synthetic oblique-shock case.

This file used to drive six schemes against PLOT3D plate data that is not
tracked in the repository, so it skipped everywhere, and it never inspected
``new_point`` even when it did run. It now runs on the in-memory synthetic
fixture from ``conftest.py`` and asserts on the result: the point moves along
the local velocity, stays finite and inside the domain, the displacement scales
with the time step, and the three physical-space schemes agree with each other
in the uniform upstream region.

Note the differing return signatures of ``Integration.compute``, which the
helpers below normalize: ``p-space`` and ``c-space`` return a bare point,
``pRK2``/``pRK4`` return ``(point, fluid velocity)``, and ``cRK2``/``cRK4``
return a computational-space point plus two velocities.

``Integration.compute_ppath`` is uniform by contrast: every method returns
``(point, particle velocity, fluid velocity)``. It did not use to be --
``cRK4`` returned the two velocities the other way round from ``pRK4``, and
its own blow-up early-returns disagreed with its normal return -- so
``test_compute_ppath_return_order_is_uniform`` pins it.
"""

import numpy as np
import pytest

from lptlib.function import Variables
from lptlib.streamlines import Search, Interpolation, Integration

P_SPACE_SCHEMES = ["p-space", "pRK2", "pRK4"]
C_SPACE_SCHEMES = ["c-space", "cRK2", "cRK4"]
ALL_SCHEMES = P_SPACE_SCHEMES + C_SPACE_SCHEMES

TIME_STEP = 1e-9


@pytest.fixture(scope="module")
def interior_point(oblique_case):
    """An upstream point inside a cell, away from its nodes.

    This exercises the general interpolation path, where the tri-linear
    weights actually do work; ``node_point`` below covers the exact-node
    shortcut. (The offset used to be a workaround: on a node the c-space
    interpolation left ``J_inv`` unset and the c-space RK schemes raised from
    ``np.matmul``. That is fixed, so both cases are now tested on purpose.)
    """
    grd = oblique_case.grid.grd
    node = np.array([grd[3, 5, 2, 0, 0], grd[3, 5, 2, 1, 0], grd[3, 5, 2, 2, 0]])
    spacing = grd[4, 5, 2, 0, 0] - grd[3, 5, 2, 0, 0]
    return node + np.array([0.25 * spacing, 0.25 * spacing, 0.0])


@pytest.fixture(scope="module")
def node_point(oblique_case):
    """An upstream point sitting exactly on a grid node."""
    grd = oblique_case.grid.grd
    return np.array([grd[3, 5, 2, 0, 0], grd[3, 5, 2, 1, 0], grd[3, 5, 2, 2, 0]])


def _stack(grid, flow, point, space):
    """Run search plus interpolation and hand back a ready Integration object."""
    idx = Search(grid, list(point))
    idx.compute(method=space)
    interp = Interpolation(flow, idx)
    interp.compute(method=space)
    return Integration(interp), interp, idx


def _local_velocity(interp):
    variables = Variables(interp)
    variables.compute_velocity()
    return variables.velocity.reshape(3)


def _step(grid, flow, point, method, time_step=TIME_STEP):
    """Advance one step and return (physical new point, local fluid velocity)."""
    space = "p-space" if method in P_SPACE_SCHEMES else "c-space"
    intg, interp, _idx = _stack(grid, flow, point, space)
    velocity = _local_velocity(interp)
    result = intg.compute(method=method, time_step=time_step)

    if method in ("p-space", "c-space"):
        raw = result
    else:
        raw = result[0]
    if raw is None:
        return None, velocity

    raw = np.asarray(raw, dtype=float).reshape(-1)
    if space == "c-space":
        # Map the computational-space answer back to physical space.
        back = Search(grid, list(point))
        back.compute(method="c-space")
        return np.asarray(back.c2p(raw.copy()), dtype=float), velocity
    return raw, velocity


@pytest.mark.parametrize("method", ALL_SCHEMES)
def test_scheme_advances_point_along_the_flow(synthetic_grid, synthetic_flow,
                                              interior_point, method):
    """Every scheme returns a finite point that moved along the local velocity.

    The synthetic upstream state is uniform with positive x and y velocity, so
    a forward step must increase both coordinates, by very nearly ``u * dt``.
    """
    new_point, velocity = _step(synthetic_grid, synthetic_flow, interior_point,
                                method)
    assert velocity[0] > 0 and velocity[1] > 0

    assert new_point is not None
    assert new_point.shape == (3,)
    assert np.all(np.isfinite(new_point))

    displacement = new_point - np.asarray(interior_point, dtype=float)
    assert displacement[0] > 0
    assert displacement[1] > 0
    np.testing.assert_allclose(displacement, velocity * TIME_STEP, rtol=1e-3,
                               atol=1e-18)

    # Still inside the grid bounds.
    assert np.all(new_point >= synthetic_grid.grd_min[0] - 1e-12)
    assert np.all(new_point <= synthetic_grid.grd_max[0] + 1e-12)


@pytest.mark.parametrize("method", ALL_SCHEMES)
def test_step_scales_with_time_step(synthetic_grid, synthetic_flow,
                                    interior_point, method):
    """Doubling the time step doubles the displacement in a uniform flow."""
    start = np.asarray(interior_point, dtype=float)
    far, _ = _step(synthetic_grid, synthetic_flow, interior_point, method, 2e-9)
    near, _ = _step(synthetic_grid, synthetic_flow, interior_point, method, 1e-9)

    ratio = np.linalg.norm(far - start) / np.linalg.norm(near - start)
    assert ratio == pytest.approx(2.0, rel=1e-3)


def test_physical_space_schemes_agree_in_a_uniform_flow(synthetic_grid,
                                                        synthetic_flow,
                                                        interior_point):
    """Euler, RK2 and RK4 coincide where the velocity field is constant.

    The upstream region of the synthetic case is piecewise constant, so all the
    intermediate RK stages sample the same velocity and every scheme reduces to
    ``x + u dt``. Any disagreement means a stage is being weighted wrongly.
    """
    results = {method: _step(synthetic_grid, synthetic_flow, interior_point,
                             method)[0]
               for method in P_SPACE_SCHEMES}

    np.testing.assert_allclose(results["pRK2"], results["p-space"], rtol=1e-12)
    np.testing.assert_allclose(results["pRK4"], results["p-space"], rtol=1e-12)


def test_c_space_and_p_space_agree(synthetic_grid, synthetic_flow,
                                   interior_point):
    """Integrating in computational space lands in the same physical place."""
    p_point, _ = _step(synthetic_grid, synthetic_flow, interior_point, "pRK4")
    c_point, _ = _step(synthetic_grid, synthetic_flow, interior_point, "cRK4")
    np.testing.assert_allclose(c_point, p_point, rtol=1e-9)


def test_out_of_domain_start_returns_none(synthetic_grid, synthetic_flow):
    """A start point outside the grid yields no new point rather than garbage."""
    idx = Search(synthetic_grid, [1e3, 1e3, 1e3])
    idx.compute(method="p-space")
    interp = Interpolation(synthetic_flow, idx)
    interp.compute(method="p-space")
    intg = Integration(interp)

    assert intg.compute(method="p-space", time_step=TIME_STEP) is None
    assert intg.compute(method="pRK4", time_step=TIME_STEP)[0] is None


@pytest.mark.parametrize("method", ["pRK4", "cRK4"])
def test_particle_path_step_relaxes_toward_the_fluid(synthetic_grid,
                                                     synthetic_flow,
                                                     interior_point, method):
    """compute_ppath returns finite state and drag reduces the slip velocity."""
    space = "p-space" if method == "pRK4" else "c-space"
    intg, _, _ = _stack(synthetic_grid, synthetic_flow, interior_point, space)

    v_start = np.array([100.0, 0.0, 0.0])
    result = intg.compute_ppath(diameter=281e-9, density=813.0,
                                velocity=v_start.copy(), method=method,
                                time_step=1e-9, drag_model="stokes")
    assert result is not None
    point, v_new, u_f = (np.asarray(item, dtype=float) for item in result)

    assert np.all(np.isfinite(point))
    assert np.all(np.isfinite(v_new))
    assert np.all(np.isfinite(u_f))
    assert np.linalg.norm(v_new - u_f) < np.linalg.norm(v_start - u_f)


@pytest.mark.parametrize("method", ["pRK4", "cRK4", "unsteady-pRK4"])
def test_compute_ppath_return_order_is_uniform(synthetic_grid, synthetic_flow,
                                               interior_point, method):
    """Every ``compute_ppath`` branch returns (point, particle, fluid).

    ``cRK4`` used to return the velocities the other way round from ``pRK4``,
    and its three blow-up early-returns disagreed with its own normal return,
    so a caller that unpacked one correctly mislabelled the other. The
    identifying property here is physical: the particle is launched at rest
    while the fluid is supersonic in +x, so after a single 1 ns step the
    particle velocity is still far below the fluid velocity. Swap the two and
    this fails.
    """
    space = "c-space" if method == "cRK4" else "p-space"
    intg, interp, _ = _stack(synthetic_grid, synthetic_flow, interior_point,
                             space)
    if method == "unsteady-pRK4":
        # The unsteady branch falls back to the steady state when there is no
        # previous flow object to blend with.
        interp.flow_old = None
        interp.time = [0.0]

    u_fluid = _local_velocity(interp)
    v_start = np.zeros(3)

    point, v_new, u_f = intg.compute_ppath(diameter=5e-6, density=4200.0,
                                           velocity=v_start.copy(),
                                           method=method, time_step=1e-9,
                                           drag_model="stokes")
    point = np.asarray(point, dtype=float)
    v_new = np.asarray(v_new, dtype=float)
    u_f = np.asarray(u_f, dtype=float)

    # The third value is the fluid velocity: it matches the field the search
    # and interpolation already reported at this point.
    np.testing.assert_allclose(u_f, u_fluid, rtol=1e-6)
    # The second value is the particle's own velocity: a heavy particle
    # released from rest has barely started to accelerate.
    assert np.linalg.norm(v_new) < 0.5 * np.linalg.norm(u_fluid)
    assert point.shape == (3,)


def test_blowup_return_keeps_the_same_order(synthetic_grid, synthetic_flow,
                                            interior_point):
    """The blow-up early-return matches the normal return, position by position.

    Forcing the guard to fire (``blowup_factor = 0``) must hand back the
    unchanged input state in the same ``(point, particle, fluid)`` order, with
    the step-failed sentinel set so callers can tell no step was taken.
    """
    for method, space in (("pRK4", "p-space"), ("cRK4", "c-space")):
        intg, interp, idx = _stack(synthetic_grid, synthetic_flow,
                                   interior_point, space)
        u_fluid = _local_velocity(interp)
        intg.blowup_factor = 0.0

        point, v_new, u_f = intg.compute_ppath(diameter=5e-6, density=4200.0,
                                               velocity=np.zeros(3),
                                               method=method, time_step=1e-6,
                                               drag_model="stokes")

        assert intg.rk4_bool is True, method
        # No step taken: the point came straight back.
        expected = idx.ppoint if space == "p-space" else idx.cpoint
        np.testing.assert_allclose(np.asarray(point, dtype=float),
                                   np.asarray(expected, dtype=float),
                                   rtol=1e-12, err_msg=method)
        # Third slot is still the fluid velocity, second is still the particle.
        np.testing.assert_allclose(np.asarray(u_f, dtype=float), u_fluid,
                                   rtol=1e-6, err_msg=method)
        assert np.linalg.norm(np.asarray(v_new, dtype=float)) < \
            0.5 * np.linalg.norm(u_fluid), method


@pytest.mark.parametrize("method", ALL_SCHEMES)
def test_schemes_run_at_an_exact_grid_node(synthetic_grid, synthetic_flow,
                                           node_point, method):
    """Every scheme steps forward from a point sitting exactly on a node.

    The c-space interpolation used to take a node shortcut that skipped
    assigning ``J_inv``, so ``cRK2``/``cRK4`` then raised ``ValueError`` out of
    ``np.matmul``. Tests worked around it by probing off-node.
    """
    new_point, velocity = _step(synthetic_grid, synthetic_flow, node_point,
                                method)

    assert new_point is not None
    assert np.all(np.isfinite(new_point))
    displacement = new_point - np.asarray(node_point, dtype=float)
    np.testing.assert_allclose(displacement, velocity * TIME_STEP, rtol=1e-3,
                               atol=1e-18)


def test_c_space_interpolation_sets_metrics_at_a_node(synthetic_grid,
                                                      synthetic_flow,
                                                      node_point):
    """``J`` and ``J_inv`` are populated on the exact-node path too."""
    idx = Search(synthetic_grid, list(node_point))
    idx.compute(method="c-space")
    interp = Interpolation(synthetic_flow, idx)
    interp.compute(method="c-space")

    assert idx.info is not None      # the node shortcut really was taken
    assert interp.J is not None
    assert interp.J_inv is not None
    assert np.asarray(interp.J).shape == (3, 3)
    assert np.asarray(interp.J_inv).shape == (3, 3)
    np.testing.assert_allclose(np.asarray(interp.J) @ np.asarray(interp.J_inv),
                               np.eye(3), atol=1e-9)
