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

2. ``PUBLISHED`` pins three classic textbook cases to the values tabulated in
   the standard gas-dynamics references (NACA Report 1135 charts and the
   worked examples in Anderson, *Modern Compressible Flow*), quoted to the
   precision those sources give. These catch an error that both formulations
   might share.
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


# Textbook cases. beta, p2/p1 and M2 are the values given by the standard
# gas-dynamics references (NACA Report 1135 theta-beta-M chart and normal-shock
# tables; the worked examples in Anderson, Modern Compressible Flow), quoted to
# the precision those sources carry -- hence the loose tolerances here.
PUBLISHED = [
    # (mach, deflection deg, beta deg, p2/p1, M2)
    (2.0, 10.0, 39.3, 1.707, 1.641),
    (2.0, 20.0, 53.4, 2.843, 1.210),
    (3.0, 15.0, 32.2, 2.822, 2.255),
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
    """Three classic cases against tabulated textbook values.

    M1 = 2, theta = 10 deg -> beta = 39.3 deg, p2/p1 = 1.707, M2 = 1.641
    M1 = 2, theta = 20 deg -> beta = 53.4 deg, p2/p1 = 2.843, M2 = 1.210
    M1 = 3, theta = 15 deg -> beta = 32.2 deg, p2/p1 = 2.822, M2 = 2.255
    """
    shock = _solved(mach, deflection)
    assert shock.shock_angle[0] == pytest.approx(beta, abs=0.05)
    assert shock.pressure_ratio[0] == pytest.approx(pressure, rel=2e-3)
    assert shock.mach_ratio[0] * mach == pytest.approx(mach2, rel=2e-3)


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
