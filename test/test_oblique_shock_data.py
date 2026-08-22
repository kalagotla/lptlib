"""Verification of ``ObliqueShock`` against analytic oblique-shock theory.

``ObliqueShock.compute`` solves the theta-beta-Mach relation as a cubic in
cot(beta) and then applies the Rankine-Hugoniot jump conditions. Nothing in the
suite used to check any of that: the only assertion compared
``shock_angle.all()`` with ``np.array([...]).all()``, i.e. ``True == True``.

The reference values below are produced two ways, and both are independent of
the library's own code path.

1. ``_reference_solution`` below solves the theta-beta-Mach relation in its
   standard trigonometric form,

       tan(theta) = 2 cot(beta) (M1^2 sin^2(beta) - 1)
                    / (M1^2 (gamma + cos(2 beta)) + 2)

   with a bracketed root find (``brentq``) on the weak branch
   ``[asin(1/M1), beta_max]`` and on the strong branch ``[beta_max, pi/2]``,
   then applies the Rankine-Hugoniot relations for a normal shock at
   ``Mn1 = M1 sin(beta)``:

       p2/p1   = 1 + 2 gamma/(gamma+1) (Mn1^2 - 1)
       rho2/rho1 = (gamma+1) Mn1^2 / ((gamma-1) Mn1^2 + 2)
       T2/T1   = (p2/p1) / (rho2/rho1)
       Mn2^2   = (1 + (gamma-1)/2 Mn1^2) / (gamma Mn1^2 - (gamma-1)/2)
       M2      = Mn2 / sin(beta - theta)

   That is a different formulation (trig root find rather than the cubic in
   cot(beta)) written out here from the standard relations, so agreement to
   machine precision is a genuine cross-check of the library's algebra.

2. ``PUBLISHED`` pins the library against an externally published oblique-shock
   table, entry by entry, at the precision that table prints. This catches an
   error that both formulations above might share, because the numbers come
   from outside this repository entirely.

   The table used is the "Oblique Shock Wave Table" (gamma = 7/5) from
   *Aerodynamics for Students*, AMME, University of Sydney (copyright
   1996-2006), as mirrored by the Cambridge University Engineering Department:
   https://www-mdp.eng.cam.ac.uk/web/library/enginfo/aerothermal_dvd_only/aero/oblique/obtable.html
   Its columns are Delta (deflection, deg), M1, Theta (shock angle, deg), M2
   and P2/P1; shock angle is printed to 2 dp, P2/P1 to 3 dp and M2 to 2 dp.

   Note that NACA Report 1135 (Ames Research Staff, 1953) is *not* usable for
   this purpose and is deliberately not cited here: it presents oblique-shock
   results only as charts 2-4, and its one numerical table (table II) is a
   normal-shock table. An earlier version of this docstring cited it, and the
   worked examples in Anderson, anyway; neither had been checked.
"""

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar

from lptlib.test_cases import ObliqueShock, ObliqueShockData

GAMMA = 1.4


def _theta_of_beta(mach, beta, gamma=GAMMA):
    """theta-beta-Mach relation in its standard trigonometric form (radians)."""
    return np.arctan(2 / np.tan(beta) * (mach ** 2 * np.sin(beta) ** 2 - 1) /
                     (mach ** 2 * (gamma + np.cos(2 * beta)) + 2))


def _beta_at_max_deflection(mach, gamma=GAMMA):
    """Shock angle at which the deflection angle is maximum, found numerically."""
    lo = np.arcsin(1.0 / mach) + 1e-12
    hi = np.pi / 2 - 1e-12
    result = minimize_scalar(lambda b: -_theta_of_beta(mach, b, gamma),
                             bounds=(lo, hi), method="bounded",
                             options={"xatol": 1e-14})
    return result.x


def _reference_solution(mach, deflection_deg, branch="weak", gamma=GAMMA):
    """Independent oblique-shock solution: shock angle plus the R-H ratios.

    Returns a dict with the shock angle in degrees and the pressure, density,
    temperature, downstream-Mach and Mach-ratio values across the shock.
    """
    theta = np.radians(deflection_deg)
    beta_max = _beta_at_max_deflection(mach, gamma)
    bracket = ((np.arcsin(1.0 / mach) + 1e-14, beta_max) if branch == "weak"
               else (beta_max, np.pi / 2 - 1e-14))
    beta = brentq(lambda b: _theta_of_beta(mach, b, gamma) - theta,
                  *bracket, xtol=1e-15, rtol=8.9e-16)

    mn1 = mach * np.sin(beta)
    pressure = 1 + 2 * gamma / (gamma + 1) * (mn1 ** 2 - 1)
    density = (gamma + 1) * mn1 ** 2 / ((gamma - 1) * mn1 ** 2 + 2)
    temperature = pressure / density
    mn2 = np.sqrt((1 + (gamma - 1) / 2 * mn1 ** 2) /
                  (gamma * mn1 ** 2 - (gamma - 1) / 2))
    mach2 = mn2 / np.sin(beta - theta)

    return {"beta": np.degrees(beta), "pressure": pressure, "density": density,
            "temperature": temperature, "mach2": mach2, "mach_ratio": mach2 / mach}


# Every attached-shock row of the published table named in the module docstring,
# transcribed exactly as printed (shock angle 2 dp, p2/p1 3 dp, M2 2 dp). The
# tolerances below are just over half a unit in the last printed digit of each
# column, which is what agreement with a rounded table can mean.
PUBLISHED = [
    # (mach, deflection deg, beta deg, p2/p1, M2)
    (2.0, 5.0, 34.30, 1.315, 1.82),
    (2.0, 10.0, 39.31, 1.707, 1.64),
    (2.0, 15.0, 45.34, 2.195, 1.45),
    (2.0, 20.0, 53.42, 2.843, 1.21),
    (3.0, 5.0, 23.13, 1.454, 2.75),
    (3.0, 10.0, 27.38, 2.054, 2.51),
    (3.0, 15.0, 32.24, 2.822, 2.25),
    (3.0, 20.0, 37.76, 3.771, 1.99),
    (3.0, 25.0, 44.14, 4.925, 1.72),
    (3.0, 30.0, 52.01, 6.356, 1.41),
]

CASES = [(2.0, 10.0), (2.0, 20.0), (3.0, 15.0), (2.3, 10.0), (5.0, 25.0)]


def _solved(mach, deflection):
    shock = ObliqueShock()
    shock.mach = mach
    shock.deflection = deflection
    shock.compute()
    return shock


def test_oblique_shock_docstring_example():
    """The angles quoted in the ObliqueShock docstring are the real answer.

    This replaces an assertion that compared ``shock_angle.all()`` against
    ``np.array([...]).all()``, i.e. ``True`` against ``True``.
    """
    shock = _solved(2.3, 10)
    np.testing.assert_allclose(shock.shock_angle,
                               np.array([34.32642717, 85.02615188]), rtol=1e-8)
    reference = _reference_solution(2.3, 10, "weak")
    assert shock.shock_angle[0] == pytest.approx(reference["beta"], rel=1e-10)


@pytest.mark.parametrize("mach, deflection", CASES)
def test_weak_solution_matches_independent_theta_beta_mach(mach, deflection):
    """Weak-branch shock angle and all four jump ratios, against section 1."""
    shock = _solved(mach, deflection)
    reference = _reference_solution(mach, deflection, "weak")

    assert shock.shock_angle[0] == pytest.approx(reference["beta"], rel=1e-10)
    assert shock.pressure_ratio[0] == pytest.approx(reference["pressure"], rel=1e-10)
    assert shock.density_ratio[0] == pytest.approx(reference["density"], rel=1e-10)
    assert shock.temperature_ratio[0] == pytest.approx(reference["temperature"],
                                                       rel=1e-10)
    # mach_ratio is M2/M1.
    assert shock.mach_ratio[0] == pytest.approx(reference["mach_ratio"], rel=1e-10)


@pytest.mark.parametrize("mach, deflection", CASES)
def test_strong_solution_is_the_second_root(mach, deflection):
    """The strong solution is returned second, and it really is the strong one."""
    shock = _solved(mach, deflection)
    reference = _reference_solution(mach, deflection, "strong")

    assert shock.shock_angle[1] > shock.shock_angle[0]
    assert shock.shock_angle[1] == pytest.approx(reference["beta"], rel=1e-10)
    assert shock.pressure_ratio[1] == pytest.approx(reference["pressure"], rel=1e-10)
    assert shock.density_ratio[1] == pytest.approx(reference["density"], rel=1e-10)
    assert shock.temperature_ratio[1] == pytest.approx(reference["temperature"],
                                                       rel=1e-10)
    assert shock.mach_ratio[1] == pytest.approx(reference["mach_ratio"], rel=1e-10)

    # Physical sanity: the strong solution is stronger and subsonic behind it.
    assert shock.pressure_ratio[1] > shock.pressure_ratio[0]
    assert shock.density_ratio[1] > shock.density_ratio[0]
    assert shock.mach_ratio[1] * mach < 1.0


@pytest.mark.parametrize("mach, deflection, beta, pressure, mach2", PUBLISHED)
def test_matches_published_gas_dynamics_tables(mach, deflection, beta, pressure,
                                               mach2):
    """Every attached-shock row of the published table, at its printed precision.

    Source: "Oblique Shock Wave Table" (gamma = 7/5), Aerodynamics for Students,
    AMME, University of Sydney, mirrored at
    https://www-mdp.eng.cam.ac.uk/web/library/enginfo/aerothermal_dvd_only/aero/oblique/obtable.html
    """
    shock = _solved(mach, deflection)
    assert shock.shock_angle[0] == pytest.approx(beta, abs=0.006)
    assert shock.pressure_ratio[0] == pytest.approx(pressure, abs=0.0006)
    assert shock.mach_ratio[0] * mach == pytest.approx(mach2, abs=0.006)


@pytest.mark.parametrize("mach, deflection", CASES)
def test_rankine_hugoniot_internal_consistency(mach, deflection):
    """T2/T1 = (p2/p1)/(rho2/rho1) and every ratio exceeds one."""
    shock = _solved(mach, deflection)
    np.testing.assert_allclose(shock.temperature_ratio,
                               shock.pressure_ratio / shock.density_ratio,
                               rtol=1e-12)
    assert np.all(shock.pressure_ratio > 1.0)
    assert np.all(shock.density_ratio > 1.0)
    assert np.all(shock.temperature_ratio > 1.0)
    # Density ratio can never exceed the (gamma+1)/(gamma-1) = 6 limit for air.
    assert np.all(shock.density_ratio < (GAMMA + 1) / (GAMMA - 1))
    # Entropy rises, so the flow slows: M2 < M1 on both branches.
    assert np.all(shock.mach_ratio < 1.0)


def test_weak_solution_is_supersonic_downstream():
    """For deflections well below detachment the weak shock leaves M2 > 1."""
    for mach, deflection in [(2.0, 10.0), (3.0, 15.0), (5.0, 25.0)]:
        shock = _solved(mach, deflection)
        assert shock.mach_ratio[0] * mach > 1.0


@pytest.mark.parametrize("mach, expected_max", [(1.5, 12.1127), (2.0, 22.9735),
                                                (3.0, 34.0734), (5.0, 41.1177)])
def test_max_deflection_matches_numerical_maximum(mach, expected_max):
    """The closed-form detachment angle equals a direct numerical maximisation."""
    shock = ObliqueShock(mach=mach)
    numerical = np.degrees(_theta_of_beta(mach, _beta_at_max_deflection(mach)))
    assert shock.max_deflection() == pytest.approx(numerical, rel=1e-9)
    assert shock.max_deflection() == pytest.approx(expected_max, abs=1e-3)


def test_deflection_beyond_detachment_raises():
    """Past the detachment angle there is no attached shock, and it says so."""
    shock = ObliqueShock()
    shock.mach = 2.0
    shock.deflection = 30.0  # theta_max at M = 2 is 22.97 deg
    with pytest.raises(ValueError) as excinfo:
        shock.compute()
    message = str(excinfo.value)
    assert "22.97" in message
    assert "detach" in message
    # The input is left untouched so the caller can retry.
    assert shock.deflection == 30.0


def test_just_inside_detachment_still_solves():
    """Just below the detachment angle the two roots survive and nearly meet."""
    shock = _solved(2.0, 22.9)
    assert shock.shock_angle[1] > shock.shock_angle[0]
    # Approaching detachment the weak and strong roots converge.
    assert shock.shock_angle[1] - shock.shock_angle[0] < 5.0


def test_subsonic_free_stream_raises():
    """No oblique shock exists below Mach 1."""
    shock = ObliqueShock()
    shock.mach = 0.8
    shock.deflection = 5.0
    with pytest.raises(ValueError, match="supersonic"):
        shock.compute()


def test_oblique_shock_data_flow_matches_shock_ratios():
    """The generated flow field carries exactly the computed jump ratios.

    ``ObliqueShockData`` writes a piecewise-constant PLOT3D-style q array with
    the pre-shock state for x < 0 and the post-shock state for x >= 0. The
    density jump across that interface must equal ``density_ratio``, and the
    static temperature recovered from the conservative variables must jump by
    ``temperature_ratio``.
    """
    shock = _solved(2.3, 10)
    weak_density_ratio = shock.density_ratio[0]
    weak_temperature_ratio = shock.temperature_ratio[0]

    osd = ObliqueShockData()
    osd.nx_max, osd.ny_max, osd.nz_max = 10, 10, 10
    osd.inlet_density = 1.273
    osd.inlet_temperature = 300
    osd.xpoints, osd.ypoints, osd.zpoints = 20, 20, 4
    osd.oblique_shock = shock
    osd.create_grid()
    osd.create_flow()

    q = osd.flow.q
    pre, post = q[0, 0, 0, :, 0], q[-1, 0, 0, :, 0]

    assert pre[0] == pytest.approx(osd.inlet_density, rel=1e-12)
    assert post[0] / pre[0] == pytest.approx(weak_density_ratio, rel=1e-10)

    def static_temperature(state):
        rho = state[0]
        vel_sq = (state[1:4] / rho) ** 2
        return ((GAMMA - 1) * (state[4] / rho - vel_sq.sum() / 2)
                / shock.gas_constant)

    assert static_temperature(pre) == pytest.approx(osd.inlet_temperature, rel=1e-10)
    assert (static_temperature(post) / static_temperature(pre) ==
            pytest.approx(weak_temperature_ratio, rel=1e-10))

    # The grid spans the requested box and the shock sits at the midpoint.
    np.testing.assert_allclose(osd.grid.grd_min, [[-10, 0, 0]])
    np.testing.assert_allclose(osd.grid.grd_max, [[10, 10, 10]])
    assert osd.grid.grd.shape[:3] == (40, 20, 4)


def test_instances_do_not_share_an_oblique_shock():
    """Each ObliqueShockData gets its own ObliqueShock.

    The constructors used to default to ``oblique_shock=ObliqueShock()``, a
    single object built once at import time and shared by every instance --
    and ``create_flow`` mutates it in place, replacing the two-element ratio
    arrays with scalars.
    """
    from lptlib.test_cases import ObliqueShockAlignedData

    for cls in (ObliqueShockData, ObliqueShockAlignedData):
        first, second = cls(), cls()
        assert first.oblique_shock is not second.oblique_shock

        first.oblique_shock.mach = 2.0
        first.oblique_shock.deflection = 8.0
        first.oblique_shock.compute()

        assert second.oblique_shock.mach is None
        assert second.oblique_shock.shock_angle is None


def test_default_constructor_can_build_a_case():
    """The defaults are a working case, not a pile of Nones.

    ``create_grid`` used to fail with ``TypeError: bad operand type for unary -:
    'NoneType'`` on a default-constructed object.
    """
    osd = ObliqueShockData(xpoints=8, ypoints=8, zpoints=3)
    osd.oblique_shock.mach = 2.0
    osd.oblique_shock.deflection = 8.0
    osd.oblique_shock.compute()
    osd.create_grid()
    osd.create_flow()

    assert osd.grid.grd.shape == (16, 8, 3, 3, 1)
    assert osd.flow.q.shape == (16, 8, 3, 5, 1)
    assert np.all(np.isfinite(osd.flow.q))


@pytest.mark.parametrize("attribute", ["nx_max", "xpoints", "zpoints"])
def test_missing_grid_input_raises_named_error(attribute):
    """Clearing a geometry attribute gives a message naming it."""
    osd = ObliqueShockData(xpoints=8, ypoints=8, zpoints=3)
    setattr(osd, attribute, None)
    with pytest.raises(ValueError, match=attribute):
        osd.create_grid()


def test_missing_inlet_temperature_raises_named_error():
    osd = ObliqueShockData(xpoints=8, ypoints=8, zpoints=3)
    osd.oblique_shock.mach = 2.0
    osd.oblique_shock.deflection = 8.0
    osd.oblique_shock.compute()
    osd.create_grid()
    osd.inlet_temperature = None
    with pytest.raises(ValueError, match="inlet_temperature"):
        osd.create_flow()


# ---------------------------------------------------------------------------
# ObliqueShockAlignedData
#
# The shock-aligned generator is exported from ``lptlib.test_cases``, re-exported
# at the package top level, and named in the README and docs alongside
# ``ObliqueShockData``, but nothing exercised it: ``create_grid`` and
# ``create_flow`` were entirely uncovered. Unlike ``ObliqueShockData``, which
# puts a shock-normal flow either side of the plane x = 0, this one keeps the
# incoming flow horizontal and tilts the shock plane to the computed shock
# angle beta, so the two are checked against different expectations.
# ---------------------------------------------------------------------------

ALIGNED_KWARGS = dict(nx_max=10e-3, ny_max=10e-3, nz_max=1e-4,
                      xpoints=8, ypoints=9, zpoints=3,
                      inlet_temperature=300.0, inlet_density=1.273)


def _aligned_case(shock_strength="weak", mach=2.3, deflection=10.0, **overrides):
    """A built ``ObliqueShockAlignedData`` plus the ``ObliqueShock`` behind it."""
    from lptlib.test_cases import ObliqueShockAlignedData

    shock = _solved(mach, deflection)
    kwargs = dict(ALIGNED_KWARGS)
    kwargs.update(overrides)
    osd = ObliqueShockAlignedData(oblique_shock=shock, **kwargs)
    osd.shock_strength = shock_strength
    osd.create_grid()
    osd.create_flow()
    return osd, shock


def _static_temperature(state, gas_constant):
    """Static temperature from a conservative ``[rho, rho*u, rho*v, rho*w, e]``."""
    rho = state[0]
    vel_sq = ((state[1:4] / rho) ** 2).sum()
    return (GAMMA - 1) * (state[4] / rho - vel_sq / 2) / gas_constant


def _pre_post_states(osd):
    """Two q states straddling the tilted shock plane, taken from grid nodes.

    The plane is ``s = sin(beta) x - cos(beta) (y - ny_max/2)``, so the node at
    the far-left of the mid-height row has ``s < 0`` (upstream) and the node at
    the far-right of the same row has ``s > 0`` (downstream), for any attached
    shock angle.
    """
    j_mid = osd.flow.nj[0] // 2
    return osd.flow.q[0, j_mid, 0, :, 0], osd.flow.q[-1, j_mid, 0, :, 0]


def test_aligned_data_builds_a_grid():
    """``create_grid`` fills in a usable GridIO, metrics included."""
    osd, _ = _aligned_case()

    assert osd.grid.nb == 1
    assert osd.grid.grd.shape == (2 * ALIGNED_KWARGS["xpoints"],
                                  ALIGNED_KWARGS["ypoints"],
                                  ALIGNED_KWARGS["zpoints"], 3, 1)
    np.testing.assert_allclose(osd.grid.grd_min, [[-10e-3, 0, 0]])
    np.testing.assert_allclose(osd.grid.grd_max, [[10e-3, 10e-3, 1e-4]])
    # The domain really spans the requested box.
    assert osd.grid.grd[..., 0, 0].min() == pytest.approx(-10e-3)
    assert osd.grid.grd[..., 0, 0].max() == pytest.approx(10e-3)
    # compute_metrics ran, so the grid is usable by Search/Interpolation.
    assert osd.grid.m1 is not None
    assert osd.grid.m2 is not None
    assert osd.grid.J is not None
    assert np.all(np.isfinite(osd.grid.grd))


def test_aligned_data_builds_a_flow():
    """``create_flow`` fills in a usable FlowIO matching the grid."""
    osd, shock = _aligned_case()

    assert osd.flow.nb == 1
    assert osd.flow.q.shape == (2 * ALIGNED_KWARGS["xpoints"],
                                ALIGNED_KWARGS["ypoints"],
                                ALIGNED_KWARGS["zpoints"], 5, 1)
    np.testing.assert_array_equal(osd.flow.ni, osd.grid.ni)
    np.testing.assert_array_equal(osd.flow.nj, osd.grid.nj)
    np.testing.assert_array_equal(osd.flow.nk, osd.grid.nk)
    assert osd.flow.mach == shock.mach
    assert np.all(np.isfinite(osd.flow.q))
    # The field is genuinely two-valued, not a single constant state.
    assert len(np.unique(osd.flow.q[..., 0, 0])) == 2


def test_aligned_flow_carries_the_computed_shock_ratios():
    """The two states either side of the tilted plane are the R-H jump."""
    osd, shock = _aligned_case()
    pre, post = _pre_post_states(osd)

    # Upstream state is exactly the free stream, flowing along +x.
    assert pre[0] == pytest.approx(osd.inlet_density, rel=1e-12)
    assert _static_temperature(pre, shock.gas_constant) == pytest.approx(
        osd.inlet_temperature, rel=1e-10)
    assert pre[2] == pytest.approx(0.0, abs=1e-12)  # no y-momentum upstream
    assert pre[3] == pytest.approx(0.0, abs=1e-12)  # no z-momentum anywhere

    # Downstream state carries the weak-branch density and temperature ratios.
    assert post[0] / pre[0] == pytest.approx(shock.density_ratio[0], rel=1e-10)
    assert (_static_temperature(post, shock.gas_constant)
            / _static_temperature(pre, shock.gas_constant)
            == pytest.approx(shock.temperature_ratio[0], rel=1e-10))

    # Downstream velocity is turned by exactly the deflection angle.
    turn = np.degrees(np.arctan2(post[2], post[1]))
    assert turn == pytest.approx(shock.deflection, rel=1e-10)


def test_aligned_flow_strong_branch_uses_the_second_root():
    """``shock_strength='strong'`` picks the strong solution, not the weak one."""
    weak, shock = _aligned_case("weak")
    strong, _ = _aligned_case("strong")

    _, weak_post = _pre_post_states(weak)
    weak_pre, _ = _pre_post_states(weak)
    _, strong_post = _pre_post_states(strong)
    strong_pre, _ = _pre_post_states(strong)

    assert weak_post[0] / weak_pre[0] == pytest.approx(shock.density_ratio[0],
                                                       rel=1e-10)
    assert strong_post[0] / strong_pre[0] == pytest.approx(shock.density_ratio[1],
                                                           rel=1e-10)
    # The strong shock really is the stronger compression.
    assert strong_post[0] > weak_post[0]


def test_aligned_flow_rejects_an_unknown_shock_strength():
    from lptlib.test_cases import ObliqueShockAlignedData

    shock = _solved(2.3, 10)
    osd = ObliqueShockAlignedData(oblique_shock=shock, **ALIGNED_KWARGS)
    osd.shock_strength = 'oblique'
    osd.create_grid()
    with pytest.raises(ValueError, match="weak"):
        osd.create_flow()


def test_aligned_create_flow_leaves_the_shock_object_intact():
    """Building a case must not consume the ObliqueShock it was handed.

    ``ObliqueShockData.create_flow`` replaces each two-element ratio array on
    the shock with the single branch it selected, so the same ``ObliqueShock``
    cannot be reused. The aligned generator reads the branch into locals
    instead, and this pins that difference: two instances can share one shock
    and both get the same answer.
    """
    from lptlib.test_cases import ObliqueShockAlignedData

    shock = _solved(2.3, 10)
    before = (np.array(shock.shock_angle, copy=True),
              np.array(shock.density_ratio, copy=True),
              np.array(shock.temperature_ratio, copy=True),
              np.array(shock.mach_ratio, copy=True))

    first = ObliqueShockAlignedData(oblique_shock=shock, **ALIGNED_KWARGS)
    first.create_grid()
    first.create_flow()

    for expected, name in zip(before, ('shock_angle', 'density_ratio',
                                       'temperature_ratio', 'mach_ratio')):
        np.testing.assert_allclose(getattr(shock, name), expected,
                                   err_msg=f'create_flow mutated {name}')

    second = ObliqueShockAlignedData(oblique_shock=shock, **ALIGNED_KWARGS)
    second.create_grid()
    second.create_flow()
    np.testing.assert_allclose(second.flow.q, first.flow.q)


def test_aligned_instances_do_not_share_grid_or_flow_objects():
    """Two instances hold their own GridIO/FlowIO, so one cannot clobber the other."""
    from lptlib.test_cases import ObliqueShockAlignedData

    first = ObliqueShockAlignedData(**ALIGNED_KWARGS)
    second = ObliqueShockAlignedData(**ALIGNED_KWARGS)

    assert first.grid is not second.grid
    assert first.flow is not second.flow
    assert first.oblique_shock is not second.oblique_shock

    first.oblique_shock.mach = 2.0
    first.oblique_shock.deflection = 8.0
    first.oblique_shock.compute()
    first.create_grid()
    first.create_flow()

    assert second.grid.grd is None
    assert second.flow.q is None
    assert second.oblique_shock.shock_angle is None


@pytest.mark.parametrize("attribute", ["nx_max", "ypoints", "nz_max"])
def test_aligned_missing_grid_input_raises_named_error(attribute):
    """The shared input checks are wired up on the aligned generator too."""
    from lptlib.test_cases import ObliqueShockAlignedData

    osd = ObliqueShockAlignedData(**ALIGNED_KWARGS)
    setattr(osd, attribute, None)
    with pytest.raises(ValueError, match=attribute):
        osd.create_grid()


def test_aligned_missing_inlet_temperature_raises_named_error():
    from lptlib.test_cases import ObliqueShockAlignedData

    osd = ObliqueShockAlignedData(oblique_shock=_solved(2.3, 10), **ALIGNED_KWARGS)
    osd.create_grid()
    osd.inlet_temperature = None
    with pytest.raises(ValueError, match="inlet_temperature"):
        osd.create_flow()


# ---------------------------------------------------------------------------
# create_flow must not consume the ObliqueShock it is given
#
# It used to select the weak or strong branch by overwriting the two-element
# ratio arrays on the shock object in place. That made the object single-use: a
# second create_flow call indexed a scalar and raised, and a shock shared
# between two instances silently handed the second one the branch the first had
# already selected.
# ---------------------------------------------------------------------------


def _reusable_shock():
    shock = ObliqueShock()
    shock.mach = 7.6
    shock.deflection = 20
    shock.compute()
    return shock


def _build_case(shock, strength):
    osd = ObliqueShockData()
    osd.oblique_shock = shock
    osd.nx_max, osd.ny_max, osd.nz_max = 5e-3, 30e-3, 1e-4
    osd.inlet_temperature, osd.inlet_density = 48.20, 0.07747
    osd.xpoints, osd.ypoints, osd.zpoints = 12, 20, 3
    osd.shock_strength = strength
    osd.create_grid()
    osd.create_flow()
    return osd


@pytest.mark.parametrize('name', ['shock_angle', 'density_ratio', 'pressure_ratio',
                                  'temperature_ratio', 'mach_ratio'])
def test_shock_object_survives_create_flow(name):
    shock = _reusable_shock()
    _build_case(shock, 'weak')
    assert np.asarray(getattr(shock, name)).shape == (2,), \
        f'{name} was consumed by create_flow'


def test_create_flow_is_repeatable_on_one_shock():
    shock = _reusable_shock()
    first = _build_case(shock, 'weak')
    second = _build_case(shock, 'weak')
    np.testing.assert_array_equal(first.flow.q, second.flow.q)


def test_strong_branch_still_differs_from_weak():
    shock = _reusable_shock()
    weak = _build_case(shock, 'weak')
    strong = _build_case(shock, 'strong')
    assert not np.array_equal(weak.flow.q, strong.flow.q)


def test_unknown_shock_strength_raises():
    with pytest.raises(ValueError, match='weak'):
        _build_case(_reusable_shock(), 'medium')
