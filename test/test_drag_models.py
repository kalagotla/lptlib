"""Unit tests for the spherical-particle drag suite.

Two layers are covered. First, the ten closures exposed directly through
``Variables.compute_drag_coefficient`` are checked against their analytical
limits and monotonic trends. Second, the full twelve-model suite reachable
through ``Integration.compute_ppath`` (which adds ``melling-2`` and
``tedeschi``) is exercised end to end on the synthetic grid so that every drag
branch actually advances a particle to a finite state.
"""

import numpy as np
import pytest

from lptlib.function import Variables
from lptlib.streamlines import Search, Interpolation, Integration


# Models available on the public Variables.compute_drag_coefficient method.
VARIABLES_MODELS = [
    "zero-drag", "sphere", "stokes", "melling", "oseen", "schiller-nauman",
    "cunningham", "henderson", "subramaniam-balachandar", "loth",
]

# The full particle-path suite adds two more closures.
PPATH_MODELS = VARIABLES_MODELS + ["melling-2", "tedeschi"]


@pytest.fixture
def drag():
    """A Variables instance whose drag method can be called on scalar inputs.

    Variables only needs a flow object with a ``q`` array to construct; the
    drag coefficient itself depends on Reynolds number, Mach number, and gamma.
    """
    class _Flow:
        q = np.ones((1, 1, 1, 5, 1))

    variables = Variables(_Flow())
    return lambda re, mach=0.1, model="stokes": variables.compute_drag_coefficient(
        _re=re, _mach=mach, _model=model)


def test_stokes_equals_analytical(drag):
    """Stokes drag is exactly 24/Re in the creeping-flow regime."""
    for re in [1e-6, 1e-3, 0.1, 1.0, 5.0]:
        assert drag(re, model="stokes") == pytest.approx(24.0 / re, rel=1e-12)


def test_schiller_naumann_reduces_to_stokes(drag):
    """Schiller-Naumann approaches Stokes as Re goes to zero.

    The correction factor is ``1 + 0.15 Re**0.687``, which tends to 1 as Re
    tends to 0, so the ratio to Stokes drag tends to 1.
    """
    ratios = [drag(re, model="schiller-nauman") / drag(re, model="stokes")
              for re in [1e-2, 1e-4, 1e-6]]
    # Ratio marches monotonically toward 1 as Re shrinks.
    assert ratios[0] > ratios[1] > ratios[2] > 1.0
    assert ratios[-1] == pytest.approx(1.0, abs=1e-3)


def test_oseen_reduces_to_stokes(drag):
    """Oseen drag approaches Stokes as Re goes to zero and exceeds it above."""
    assert drag(1e-6, model="oseen") / drag(1e-6, model="stokes") == pytest.approx(
        1.0, abs=1e-4)
    # Oseen adds a positive 3/16 Re term, so it is larger than Stokes for Re > 0.
    assert drag(0.5, model="oseen") > drag(0.5, model="stokes")


def test_sphere_standard_drag_plateaus(drag):
    """The rigid-sphere curve matches its documented piecewise values."""
    # Creeping regime collapses onto Stokes.
    assert drag(1e-4, model="sphere") == pytest.approx(24.0 / 1e-4, rel=1e-9)
    # Newton regime: constant drag coefficient of 0.44.
    assert drag(1000.0, model="sphere") == pytest.approx(0.44)
    assert drag(1e5, model="sphere") == pytest.approx(0.44)
    # Post-critical drop.
    assert drag(5e5, model="sphere") == pytest.approx(0.07)


@pytest.mark.parametrize("model", VARIABLES_MODELS)
def test_zero_reynolds_gives_zero_drag(drag, model):
    """Every closure returns zero drag at Re = 0 (the massless-slip limit)."""
    assert drag(0.0, mach=0.2, model=model) == 0


@pytest.mark.parametrize("model", VARIABLES_MODELS)
def test_drag_is_finite_and_nonnegative(drag, model):
    """Across each closure's defined Re range it stays finite and non-negative.

    The sweep stops below 3e5 because the ``subramaniam-balachandar`` closure is
    only defined up to that Reynolds number; the high-Re behaviour of the models
    that do extend further is checked in
    ``test_high_reynolds_defined_models``.
    """
    for re in [1e-3, 1.0, 20.0, 200.0, 1e4]:
        cd = drag(re, mach=0.3, model=model)
        assert np.isfinite(cd)
        assert cd >= 0.0


@pytest.mark.parametrize("model", ["sphere", "henderson", "loth"])
def test_high_reynolds_defined_models(drag, model):
    """Closures defined for all regimes stay finite and non-negative at high Re."""
    for re in [1e5, 5e5, 1e6]:
        cd = drag(re, mach=0.3, model=model)
        assert np.isfinite(cd)
        assert cd >= 0.0


@pytest.mark.parametrize("model", ["stokes", "oseen", "schiller-nauman", "melling"])
def test_drag_decreases_with_reynolds(drag, model):
    """Continuum closures are monotonically decreasing in Re at low Re.

    All of these carry the ``24/Re`` prefactor, so drag coefficient falls as Re
    rises across the low-Reynolds range.
    """
    values = [drag(re, mach=0.2, model=model) for re in [0.1, 1.0, 10.0, 50.0]]
    assert all(earlier > later for earlier, later in zip(values, values[1:]))


def test_zero_drag_is_identically_zero(drag):
    """The zero-drag tracer model returns zero for any input."""
    for re in [1e-3, 1.0, 100.0, 1e5]:
        assert drag(re, mach=0.5, model="zero-drag") == 0


@pytest.mark.parametrize("model", PPATH_MODELS)
def test_all_twelve_models_advance_particle(synthetic_grid, synthetic_flow,
                                             upstream_point, model):
    """Each of the twelve drag models advances a particle to a finite state.

    This drives the actual ``compute_ppath`` code path, which builds the drag
    coefficient, the relative Reynolds and Mach numbers, and the RK4 update, so
    every drag branch is executed on real interpolated flow data rather than in
    isolation.
    """
    idx = Search(synthetic_grid, list(upstream_point))
    idx.compute(method="p-space")
    interp = Interpolation(synthetic_flow, idx)
    interp.compute(method="p-space")
    intg = Integration(interp)

    result = intg.compute_ppath(diameter=281e-9, density=813,
                                velocity=np.array([100.0, 0.0, 0.0]),
                                method="pRK4", time_step=1e-9, drag_model=model)
    assert result is not None
    x_new, v_new, u_f = result
    assert np.all(np.isfinite(x_new))
    assert np.all(np.isfinite(v_new))
    # The particle stays inside the physical domain after one small step.
    assert synthetic_grid.grd_min[0][0] <= x_new[0] <= synthetic_grid.grd_max[0][0]


def test_zero_drag_particle_follows_fluid(synthetic_grid, synthetic_flow,
                                          upstream_point):
    """A zero-drag tracer takes the local fluid velocity as its own.

    With no drag the particle is massless, so ``compute_ppath`` returns a
    particle velocity equal to the interpolated fluid velocity.
    """
    idx = Search(synthetic_grid, list(upstream_point))
    idx.compute(method="p-space")
    interp = Interpolation(synthetic_flow, idx)
    interp.compute(method="p-space")
    intg = Integration(interp)

    x_new, v_new, u_f = intg.compute_ppath(
        diameter=281e-9, density=813,
        velocity=np.array([10.0, 0.0, 0.0]), method="pRK4",
        time_step=1e-9, drag_model="zero-drag")
    assert np.allclose(v_new, u_f)
