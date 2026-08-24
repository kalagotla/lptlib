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
    only defined up to that Reynolds number; the high-Re behavior of the models
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
    x_new, v_new, _u_f = result
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

    _x_new, v_new, u_f = intg.compute_ppath(
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
# the arms meet. This sweep measures the jump at Re = 1 and pins it for every
# model.
#
# HISTORY. Re = 1 used to be where the suite's Knudsen-number definition
# changed. The 'cunningham' and 'tedeschi' cases each redefined Kn as
# ``M/sqrt(Re)`` above Re = 1 while using ``Kn = M/Re * sqrt(gamma*pi/2)``
# below it, and so stepped by a finite amount there: 15.0 per cent at M = 0.1
# and 33.4 per cent at M = 0.5 for 'cunningham', 4.4 and 13.4 per cent for
# 'tedeschi'. Those two tests were strict xfails pinning exactly those numbers.
#
# Both were transcription bugs, not seams between two source correlations, and
# both are now fixed in src/lptlib/function/variables.py -- each case uses the
# single kinetic-theory Knudsen number at all Re. The evidence is recorded in
# the in-line note on the 'cunningham' case; in outline:
#   * ``Kn = sqrt(pi*gamma/2) * M/Re`` (Kn on the particle diameter) is the
#     standard hard-sphere relation, confirmed against four independent
#     sources, and is what the 'loth' case and the method docstring already
#     used.
#   * ``M/sqrt(Re)`` is Tsien's rarefaction parameter, proportional to
#     sqrt(M*Kn). It is a real group -- it is what sits in Henderson's
#     ``exp(-0.5*M/sqrt(Re))`` factor -- but it is not a Knudsen number, and it
#     cannot be substituted into a slip correction of the form (1 + A*Kn)^-1.
#   * In 'cunningham' both arms carried the same functional form and the same
#     prefactor, so unifying Kn makes the branch vanish outright. A genuine
#     seam between two correlations would leave two distinct formulas behind.
#   * In 'tedeschi' the group ``(s*sqrt(pi)/Kn)**0.687`` inside the implicit
#     equation for k is exactly ``Re**0.687`` under the correct Kn and is not a
#     Reynolds number at all under ``M/sqrt(Re)``, so the Re > 1 arm was
#     internally inconsistent with its own derivation.
#
# The xfails are gone and the tolerance is now tight: every model except
# 'sphere' is continuous at Re = 1 to within 1e-6 relative, which is itself
# only the finite-difference residual of a continuous function sampled 1e-9
# apart in Re (measured: at most 2e-9 relative for every model in the suite).
RE_CONTINUITY_TOL = 1e-6

# 'sphere' is the one model with a real, documented step at Re = 1. Its
# standard-drag-curve arms are ``24/Re (1 + 3Re/16)`` below and
# ``24/Re (1 + Re**(2/3)/6)`` above, two different published fits whose branch
# point was tuned by hand against the VISUAL3 code (see the in-line note on
# that case). The arms do not meet: the measured step is 1.75 per cent,
# independent of Mach number because the model is incompressible. That is a
# genuine modeling seam between two correlations rather than a variable
# changing meaning, so it is documented and bounded here instead of being
# fixed -- closing it would mean re-tuning a fit this library did not author.
SEAM_AT_RE_1 = {
    "sphere": 0.0175439,
}
SEAM_TOL = 1e-5


def _relative_jump_across_re_one(drag, model, mach, epsilon=1e-9):
    """Relative step in Cd between Re just below and just above 1."""
    below = _scalar(drag(1.0 - epsilon, mach=mach, model=model))
    above = _scalar(drag(1.0 + epsilon, mach=mach, model=model))
    return abs(above - below) / max(abs(below), 1e-30)


@pytest.mark.parametrize("mach", [0.1, 0.5])
@pytest.mark.parametrize("model", [m for m in VARIABLES_MODELS
                                   if m not in SEAM_AT_RE_1])
def test_drag_is_continuous_across_reynolds_one(drag, model, mach):
    """Cd does not step at Re = 1.

    Applied to every model in the suite bar 'sphere', because a discontinuity
    is a property of the implementation rather than of any one closure, and
    because the two models that used to fail here failed for the same copied
    reason -- which is only visible when the whole suite is measured the same
    way.

    A particle crossing Re = 1 during a trajectory sees a jump as an
    instantaneous change in the force acting on it, so this is not cosmetic:
    it would make the integrated path depend on which side of the branch the
    time step happened to land.
    """
    jump = _relative_jump_across_re_one(drag, model, mach)
    assert jump < RE_CONTINUITY_TOL, (
        f"{model} at M = {mach}: Cd jumps by {100 * jump:.4g} per cent across "
        f"Re = 1")


@pytest.mark.parametrize("mach", [0.1, 0.5])
@pytest.mark.parametrize("model", sorted(SEAM_AT_RE_1))
def test_known_reynolds_one_seam_has_not_moved(drag, model, mach):
    """Pin the size of the one remaining Re = 1 step.

    'sphere' joins two different published fits at Re = 1 and they do not
    meet. The step is real and is left in place; this records how big it is so
    that the figure quoted above stays verifiable rather than becoming
    folklore, and so that a maintainer who re-tunes the branch points sees the
    number move.
    """
    jump = _relative_jump_across_re_one(drag, model, mach)
    assert jump == pytest.approx(SEAM_AT_RE_1[model], abs=SEAM_TOL)


# ---------------------------------------------------------------------------
# The suite uses one Knudsen number.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mach", [0.05, 0.2, 0.5, 1.5])
@pytest.mark.parametrize(
    "re", [1e-3, 0.1, 0.5, 1.0 - 1e-9, 1.0, 1.0 + 1e-9, 1.1, 2.0, 10.0, 1e3])
def test_slip_models_share_one_knudsen_number(drag, re, mach):
    """The three ``(1 + A*Kn)^-1`` closures must agree on what Kn is.

    'melling', 'melling-2' and 'cunningham' are the same slip correction on
    Stokes drag with A = 1, 2.7 and 4.5 respectively, so

        Cd = 24/Re * (1 + A*Kn)^-1   =>   Kn = (24/(Re*Cd) - 1) / A

    recovers Kn from each without touching the library's own expression. All
    three must return the suite's single definition,
    ``Kn = M/Re * sqrt(pi*gamma/2)``, at every Reynolds number.

    This is the direct regression test for the defect the continuity tests
    above only detect indirectly: 'cunningham' used to return a different Kn
    above Re = 1, which this would catch at every sampled Re > 1 rather than
    only at the branch point.
    """
    gamma = 1.4  # the Variables default the drag fixture is built with
    expected_kn = mach / re * np.sqrt(np.pi * gamma / 2)

    for model, a in [("melling", 1.0), ("melling-2", 2.7), ("cunningham", 4.5)]:
        cd = _scalar(drag(re, mach=mach, model=model))
        recovered = (24.0 / (re * cd) - 1.0) / a
        assert recovered == pytest.approx(expected_kn, rel=1e-12), (
            f"{model} at Re = {re}, M = {mach} implies Kn = {recovered:g}, "
            f"expected {expected_kn:g}")


@pytest.mark.parametrize("mach", [0.05, 0.2, 0.5, 1.5])
@pytest.mark.parametrize("re", [0.5, 1.0 - 1e-9, 1.0 + 1e-9, 2.0, 100.0])
def test_tedeschi_knudsen_group_reduces_to_reynolds(mach, re):
    """The identity that fixes which Kn ``tedeschi`` was written for.

    ``_solve_k`` inside the 'tedeschi' case contains the group
    ``(s*sqrt(pi)/Kn)**0.687`` with ``s = M*sqrt(gamma/2)``. That exponent is
    Schiller-Naumann's, so the base has to be a Reynolds number. It is, and
    only under the suite's Knudsen definition:

        s*sqrt(pi)/Kn = M*sqrt(gamma/2)*sqrt(pi) * Re / (M*sqrt(pi*gamma/2))
                      = Re,  for every M.

    Under the ``M/sqrt(Re)`` the case used to apply above Re = 1 the same group
    collapses to ``sqrt(pi*gamma/2 * Re)`` -- Mach-independent, and not a
    Reynolds number -- so that arm was feeding ``fsolve`` an equation
    inconsistent with its own derivation. This is the evidence that the Re = 1
    split there was a transcription bug and not a modeling seam, so it is
    checked rather than left in a comment.
    """
    gamma = 1.4
    s = mach * np.sqrt(gamma / 2)

    kn = mach / re * np.sqrt(np.pi * gamma / 2)
    assert s * np.sqrt(np.pi) / kn == pytest.approx(re, rel=1e-12)

    kn_tsien = mach / np.sqrt(re)
    assert s * np.sqrt(np.pi) / kn_tsien == pytest.approx(
        np.sqrt(np.pi * gamma / 2 * re), rel=1e-12)
