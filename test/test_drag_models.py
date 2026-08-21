"""Unit tests for the spherical-particle drag suite.

Two layers are covered. First, the twelve closures exposed through
``Variables.compute_drag_coefficient`` are checked against their analytical
limits and monotonic trends, over a Mach sweep that includes the incompressible
limit. Second, the same twelve models are exercised end to end through
``Integration.compute_ppath`` on the synthetic grid so that every drag branch
actually advances a particle to a finite state.

``Integration`` no longer carries its own private copy of the drag suite; it
calls ``Variables.compute_drag_coefficient`` directly, and
``test_single_drag_implementation`` pins that.
"""

import numpy as np
import pytest

from lptlib.function import Variables
from lptlib.streamlines import Search, Interpolation, Integration


# Every model available on the public Variables.compute_drag_coefficient method.
VARIABLES_MODELS = [
    "zero-drag", "sphere", "stokes", "melling", "melling-2", "oseen",
    "schiller-nauman", "cunningham", "henderson", "subramaniam-balachandar",
    "loth", "tedeschi",
]

# The particle-path integrator reaches exactly the same set.
PPATH_MODELS = VARIABLES_MODELS

# Mach numbers spanning incompressible, subsonic, transonic and supersonic.
MACH_SWEEP = [0.0, 0.05, 0.2, 0.3, 0.8, 1.5, 3.0]


def _scalar(cd):
    """Collapse a drag coefficient to a float.

    Most closures return a scalar; ``tedeschi`` solves for its correction with
    ``fsolve`` and so returns a length-1 array.
    """
    return float(np.asarray(cd, dtype=float).ravel()[0])


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


@pytest.mark.parametrize("model", VARIABLES_MODELS)
@pytest.mark.parametrize("mach", MACH_SWEEP)
def test_drag_finite_across_mach_sweep(drag, model, mach):
    """Every closure stays finite and non-negative across the Mach sweep.

    The older tests pinned mach=0.3 only, which is why a sign error in the
    Loth compressibility factor (``ln(M - 0.1)`` instead of ``ln(M + 0.1)``)
    went unnoticed: it only produces NaN below M = 0.1.
    """
    for re in [1e-3, 1.0, 20.0, 200.0, 1e4]:
        cd = _scalar(drag(re, mach=mach, model=model))
        assert np.isfinite(cd), f"{model} gave {cd} at Re={re}, M={mach}"
        assert cd >= 0.0, f"{model} gave {cd} at Re={re}, M={mach}"


@pytest.mark.parametrize("mach", [0.0, 0.05, 0.1])
def test_loth_is_finite_below_mach_point_one(drag, mach):
    """Regression test for the Loth compressibility-factor sign.

    With the published ``tanh(3 ln(M + 0.1))`` the compression-dominated branch
    (Re > 45) is well defined all the way down to M = 0. With the erroneous
    ``M - 0.1`` the logarithm takes a negative argument and returns NaN.
    """
    for re in [45.1, 100.0, 1e4]:
        cd = drag(re, mach=mach, model="loth")
        assert np.isfinite(cd)
        assert cd > 0.0


@pytest.mark.parametrize("mach", [0.05, 0.2, 0.3, 0.5, 0.8])
def test_loth_compressibility_factor_matches_published_form(drag, mach):
    """Recover C_M from the returned Cd and compare to Loth (2008).

    For Re > 45 the model is

        Cd = A(Re) * H(M) + 0.42 * C_M / (1 + 42000 G(M) / Re**1.16)

    with A(Re) = 24/Re (1 + 0.15 Re**0.687), H(M) = 1 - 0.258 C_M / (1 + 514 G),
    and G(M) = 1 - 1.525 M**4 for M <= 0.89. That is linear in C_M, so C_M can
    be solved for from Cd without touching the library's own expression, and
    compared against the published closed form

        C_M = 5/3 + 2/3 tanh(3 ln(M + 0.1)).
    """
    re = 100.0
    cd = drag(re, mach=mach, model="loth")

    a = 24.0 / re * (1 + 0.15 * re ** 0.687)
    g = 1 - 1.525 * mach ** 4            # M <= 0.89 branch
    b = 1 + 42000 * g / re ** 1.16
    recovered_cm = (cd - a) / (0.42 / b - 0.258 * a / (1 + 514 * g))

    expected_cm = 5 / 3 + 2 / 3 * np.tanh(3 * np.log(mach + 0.1))
    assert recovered_cm == pytest.approx(expected_cm, rel=1e-9)


def test_unknown_model_raises_value_error(drag):
    """An unrecognised model name is an error, not a silent None."""
    with pytest.raises(ValueError, match="unknown drag model"):
        drag(10.0, mach=0.3, model="not-a-real-model")


def test_out_of_range_reynolds_raises_value_error(drag):
    """subramaniam-balachandar is undefined above Re = 3e5 and says so."""
    with pytest.raises(ValueError) as excinfo:
        drag(4e5, mach=0.3, model="subramaniam-balachandar")
    message = str(excinfo.value)
    assert "subramaniam-balachandar" in message
    assert "3e5" in message
    # Just inside the range it still returns a number.
    assert np.isfinite(drag(2.9e5, mach=0.3, model="subramaniam-balachandar"))


def test_single_drag_implementation():
    """The integrator must not carry a second, drift-prone copy of the suite.

    ``Integration.compute_ppath`` used to define a nested ``_drag_constant``
    that duplicated ``Variables.compute_drag_coefficient`` and had already
    diverged from it (two extra models on one side, a sign error on the other).
    """
    import inspect

    from lptlib.streamlines import integration

    source = inspect.getsource(integration)
    assert "_drag_constant" not in source
    assert "compute_drag_coefficient" in source


# ---------------------------------------------------------------------------
# Continuity of Cd across the internal Re = 1 branch point.
# ---------------------------------------------------------------------------

# Cd is a physical closure: two models that differ by an epsilon in Reynolds
# number must not differ by a finite amount in drag. Several models in this
# suite are piecewise in Re, and a piecewise definition is only legitimate if
# the arms meet. This sweep measures the jump at Re = 1, which is where the
# suite's Knudsen-number definition changes, and pins it for every model.
#
# The tolerance is loose by the standards of a closure that should simply be
# continuous. It is set at 3 per cent, which passes 'sphere' -- whose
# standard-drag-curve arms meet at Re = 1 with a measured 1.75 per cent step, a
# real but modest artefact of a piecewise fit whose branch points were tuned by
# hand, documented as such in the in-line note on that case -- and fails
# anything that changes the meaning of a variable across the branch. The two
# models below the line jump by 4.4 to 33 per cent, so nothing here turns on
# where in that gap the threshold is put.
RE_CONTINUITY_TOL = 0.03

# Models whose Re = 1 branch is known to be discontinuous, with the measured
# relative jump at M = 0.1 and M = 0.5. Both come from the same line of code in
# the same place: the branch redefines the Knudsen number as ``M/sqrt(Re)``
# above Re = 1, where the arm below Re = 1 uses ``Kn = M/Re * sqrt(gamma*pi/2)``
# -- which is also the definition every other model in the suite uses, and the
# one the module docstring states. ``M/sqrt(Re)`` is not a Knudsen number in
# any usual sense, and using two different ones on either side of Re = 1 makes
# Cd step there.
#
#   'cunningham'  src/lptlib/function/variables.py, the ``case 'cunningham'``
#                 branch -- ``if _re > 1: _kn = _mach / np.sqrt(_re)``.
#                 Measured jump: 15.0 per cent at M = 0.1, 33.4 per cent at
#                 M = 0.5. The in-line note on that case already flags it as a
#                 defect a maintainer must resolve; this marks it in the test
#                 suite so it is tracked rather than invisible.
#   'tedeschi'    same file, the ``case 'tedeschi'`` branch -- the identical
#                 ``if _re <= 1: ... else: _kn = _mach / np.sqrt(_re)`` split,
#                 feeding the rarefaction factor epsilon(Kn). Measured jump:
#                 4.4 per cent at M = 0.1, 13.4 per cent at M = 0.5. Not
#                 previously recorded anywhere; it is the same defect, copied.
#
# Neither is changed here: the fix is a numerics decision about which Knudsen
# definition is correct, which belongs to the maintainer, not to a test.
DISCONTINUOUS_AT_RE_1 = {
    "cunningham": "Kn redefined as M/sqrt(Re) above Re = 1; "
                  "measured jump 15.0% at M = 0.1, 33.4% at M = 0.5",
    "tedeschi": "same Kn switch as cunningham; "
                "measured jump 4.4% at M = 0.1, 13.4% at M = 0.5",
}


def _relative_jump_across_re_one(drag, model, mach, epsilon=1e-9):
    """Relative step in Cd between Re just below and just above 1."""
    below = _scalar(drag(1.0 - epsilon, mach=mach, model=model))
    above = _scalar(drag(1.0 + epsilon, mach=mach, model=model))
    return abs(above - below) / max(abs(below), 1e-30)


@pytest.mark.parametrize("mach", [0.1, 0.5])
@pytest.mark.parametrize("model", VARIABLES_MODELS)
def test_drag_is_continuous_across_reynolds_one(drag, model, mach, request):
    """Cd does not step at the internal Re = 1 branch.

    Applied to every model in the suite, because a discontinuity is a property
    of the implementation rather than of any one closure, and because the two
    models that fail here fail for the same copied reason -- which is only
    visible when the whole suite is measured the same way.

    A particle crossing Re = 1 during a trajectory sees the jump as an
    instantaneous change in the force acting on it, so this is not cosmetic:
    it makes the integrated path depend on which side of the branch the time
    step happened to land.
    """
    if model in DISCONTINUOUS_AT_RE_1:
        request.node.add_marker(
            pytest.mark.xfail(strict=True,
                              reason=f"{model}: {DISCONTINUOUS_AT_RE_1[model]}"))

    jump = _relative_jump_across_re_one(drag, model, mach)
    assert jump < RE_CONTINUITY_TOL, (
        f"{model} at M = {mach}: Cd jumps by {100 * jump:.1f} per cent across "
        f"Re = 1")


@pytest.mark.parametrize("mach", [0.1, 0.5])
@pytest.mark.parametrize("model", sorted(DISCONTINUOUS_AT_RE_1))
def test_known_reynolds_one_discontinuity_has_not_grown(drag, model, mach):
    """Pin the size of the two known jumps so a change is noticed.

    ``xfail`` above says only that the models are discontinuous. This says by
    how much, so that a maintainer who changes the Knudsen definition sees the
    number move, and so that the figures quoted in the note above stay
    verifiable rather than becoming folklore.
    """
    expected = {("cunningham", 0.1): 0.150, ("cunningham", 0.5): 0.334,
                ("tedeschi", 0.1): 0.044, ("tedeschi", 0.5): 0.134}
    jump = _relative_jump_across_re_one(drag, model, mach)
    assert jump == pytest.approx(expected[(model, mach)], abs=2e-3)
