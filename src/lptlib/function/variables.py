# Calculates different variables from plot3d data
# Equations can be found here: https://www.grc.nasa.gov/WWW/winddocs/towne/plotc/plotc_p3d.html
# TODO: Add more variables
#  Enthalpies, Vorticity, Entropy, Turbulence Parameters, Gradients, Move metrics here?
#  Total quantities
import numpy as np
from scipy.optimize import fsolve
from scipy.special import erf


class Variables:
    """
    Module to compute flow variables

    ...

    Attributes
    ----------
    Input:
        flow : ..streamlines.io.plot3dio.FlowIO or src.streamlines.interpolation.Interpolation or similar
            object with q --> flow data
        gamma : float
            default is 1.4; can specify
    Output:
        velocity : ndarray
            velocity of the flow at nodes; shape of the flow.q (ni x nj x nk x 3 x nb)
        velocity_magnitude : ndarray
            magnitude of velocity at nodes; shape is (ni x nj x nk x nb)
        temperature : ndarray
            temperature at nodes; shape is (ni x nj x nk x nb)
        pressure : ndarray
            pressure at nodes; shape is (ni x nj x nk x nb)

    Methods
    -------
    compute_velocity()
        computes the velocity and velocity_magnitude
    compute_temperature()
        computes the temperature
    compute_pressure()
        computes the pressure
    compute_viscosity(law='keyes')
        computes the dynamic viscosity of air
    compute_drag_coefficient(_re, _mach, _model='stokes')
        coefficient of drag for a spherical particle
    compute()
        computes all the variables available

    Viscosity laws
    --------------
    Selected by the ``law`` argument of :meth:`compute_viscosity`; that method's
    docstring carries the regime of validity and the full reference for each.

    ==============  ==========================================================
    ``law``         Reference
    ==============  ==========================================================
    'sutherland'    Sutherland (1893), doi:10.1080/14786449308620508
    'keyes'         Keyes' correlation for air -- source UNVERIFIED, see
                    :meth:`compute_viscosity` and paper.bib
    ==============  ==========================================================

    Drag models
    -----------
    Selected by the ``_model`` argument of :meth:`compute_drag_coefficient`;
    that method's docstring carries the regime of validity in Reynolds and Mach
    number and the full reference for each.

    ==========================  ==============================================
    ``_model``                  Reference
    ==========================  ==============================================
    'zero-drag'                 none -- passive tracer, Cd = 0
    'sphere'                    White, "Fluid Mechanics" (+ VISUAL3 tuning)
    'stokes'                    Stokes (1851), pre-DOI
    'melling'                   Melling (1997), doi:10.1088/0957-0233/8/12/005
    'melling-2'                 Melling (1997), doi:10.1088/0957-0233/8/12/005
    'oseen'                     Oseen (1927), pre-DOI monograph
    'schiller-nauman'           Schiller and Naumann (1933), no DOI
    'cunningham'                Cunningham (1910), doi:10.1098/rspa.1910.0024
    'henderson'                 Henderson (1976), doi:10.2514/3.61409
    'subramaniam-balachandar'   Subramaniam and Balachandar, eds. (2022),
                                ISBN 978-0-323-90133-8
    'loth'                      Loth (2008), doi:10.2514/1.28943
    'tedeschi'                  Tedeschi, Gouin and Elena (1999),
                                doi:10.1007/s003480050291
    ==========================  ==============================================

    Full bibliographic detail for every entry above lives in ``paper.bib`` at
    the repository root.

    ...

    Example:
    -------
        variables = Variables(flow)  # Assume flow object is pre-defined
        variables.compute_velocity()  # returns velocity attribute
        variables.compute_temperature()  # fills up the temperature attribute
        variables.compute()  # computes all the attributes available

    """

    def __init__(self, flow, gamma=1.4, gas_constant=287.052874):
        self.flow = flow
        self.gamma = gamma
        self.gas_constant = gas_constant
        self.density = flow.q[..., 0, :]  # q0
        self.velocity = None
        self.mach = None
        self.velocity_magnitude = None
        self.temperature = None
        self.pressure = None
        self.viscosity = None

    def compute_velocity(self):
        """
        Function to compute velocity and velocity magnitude
        :return: None
        """
        # velocity = [q1, q2, q3] / q0
        self.velocity = self.flow.q[..., 1:4, :] / (self.flow.q[..., 0, :, None])
        self.velocity_magnitude = (self.velocity[..., 0, :]**2 + self.velocity[..., 1, :]**2 + self.velocity[..., 2, :]**2)**0.5

        return

    def compute_temperature(self):
        """
        Function to compute temperature.
        This computes velocity first
        :return: None
        """
        self.compute_velocity()
        _q4 = self.flow.q[..., 4, :]
        self.temperature = (self.gamma - 1) * (_q4/self.density - self.velocity_magnitude**2/2) / self.gas_constant
        # Guard against negative temperature from interpolation at cell boundaries
        self.temperature = np.maximum(self.temperature, 1e-30)

        return

    def compute_mach(self):
        """
        Function to compute local mach number
        Returns: None
        """
        self.compute_temperature()
        self.mach = self.velocity_magnitude / np.sqrt(self.gamma * self.gas_constant * self.temperature)

        return

    def compute_pressure(self):
        """
        Function to compute pressure.
        This computes velocity and temperature first
        :return: None
        """
        self.compute_mach()
        self.pressure = self.density * self.temperature * self.gas_constant

        return

    def compute_viscosity(self, law='keyes'):
        """
        Dynamic viscosity of air as a function of temperature.

        Temperatures must be in kelvin; ``self.temperature`` is in kelvin by
        default from :meth:`compute_temperature`. The result is in Pa.s and is
        written to ``self.viscosity``.

        Args:
            law: name of the correlation to use -- defaults to ``'keyes'``.
        :return: None

        Available laws
        --------------
        ``'sutherland'``
            Sutherland's two-constant law,
            ``mu = C1 * T**1.5 / (T + S)`` with ``S = 110.4 K`` and ``C1`` fixed
            by the reference value ``mu = 1.716e-5 Pa.s`` at ``T = 273.15 K``.
            Regime: air in the continuum regime, roughly 170--1900 K; the error
            grows below about 100 K and the law is not valid near liquefaction.
            Reference: Sutherland, W. (1893), "LII. The viscosity of gases and
            molecular force", The London, Edinburgh, and Dublin Philosophical
            Magazine and Journal of Science 36(223), 507-531,
            doi:10.1080/14786449308620508.

        ``'keyes'``
            Keyes' correlation for air,
            ``mu = a0 * sqrt(T) * 1e-6 / (1 + (a / T) * 10**(-a1 / T))``
            with ``a0 = 1.488``, ``a = 122.1``, ``a1 = 5.0``.
            Regime: air in the continuum regime; this is the library default
            because it holds up better than Sutherland's law at the low static
            temperatures reached in supersonic expansions.
            Provenance: the three constants were verified against NASA Glenn's
            Wind-US 3.0 documentation, which states the same correlation in
            imperial units and names it Keyes' law; see the in-line note on the
            ``'keyes'`` case below. Cited as ``keyes1951viscosity``.

        """
        match law:
            case 'sutherland':
                # Sutherland's viscosity law.
                # ref: Sutherland, W. (1893), "LII. The viscosity of gases and
                #   molecular force", Phil. Mag. 36(223), 507-531,
                #   doi:10.1080/14786449308620508.
                # All temperatures must be in kelvin.
                # Maintainer's note, kept from the original implementation: the
                # formula as coded here is the cfd-online form. The two
                # reference constants below (mu = 1.716e-5 Pa.s at T = 273.15 K
                # and Sutherland temperature S = 110.4 K) are the standard
                # tabulated air values reproduced there, not values read out of
                # Sutherland's 1893 paper.
                _c1 = 1.716e-5 * (273.15 + 110.4) / 273.15**1.5
                self.viscosity = _c1 * self.temperature**1.5 / (self.temperature + 110.4)
            case 'keyes':
                # Keyes' viscosity correlation for air, in the form
                #   mu = a0 * sqrt(T) * 1e-6 / (1 + (a/T) * 10**(-a1/T))   [Pa.s]
                #
                # PROVENANCE. The three coefficients below were checked against
                # NASA Glenn Research Center's Wind-US 3.0 user guide (VISCOSITY
                # keyword), which gives "Keyes' law" as
                #   mu = 2.32e-8 * T**0.5 / (1 + (220/T) * 10**(-9/T))
                # with mu in slug/(ft.s) and T in degrees Rankine:
                #   https://www.grc.nasa.gov/WWW/winddocs/windus3.0/user/keywords/viscosity.html
                # Converting that to SI (Pa.s, kelvin) gives a1 = 9 R = 5.0 K
                # exactly, a = 220 R = 122.22 K against the 122.1 coded here,
                # and a0 = 1.490e-6 against the 1.488e-6 coded here. So this is
                # the same correlation, each source carrying its own rounding:
                # the two forms agree to within 0.13% over 50-1000 K.
                # The bibliographic record for the underlying paper (F. G.
                # Keyes, Trans. Am. Soc. Mech. Engrs. 73, 589-596, 1951) was
                # confirmed from an indexed bibliography; it is now cited in
                # paper.bib as `keyes1951viscosity`. No DOI exists for it that
                # could be found, and the ASME Digital Collection page itself
                # refuses automated retrieval, so the constants are traced to
                # Keyes only through the NASA source above and NOT read out of
                # the 1951 paper. Note also that Wind-US applies Keyes' law only
                # below 160 R (89 K) and blends into Sutherland's law above it,
                # whereas lptlib uses it at all temperatures.
                a0, a, a1 = 1.488, 122.1, 5.0
                _tau = 1/self.temperature
                self.viscosity = a0 * self.temperature**0.5 * 10**-6 / (1 + a * _tau / 10 ** (a1 * _tau))
            case _:
                raise ValueError('Viscosity law not supported')

        return

    def compute_drag_coefficient(self, _re=None, _mach=None, _model='stokes'):
        """
        Coefficient of drag for a spherical particle

        This is the single implementation of the drag suite in the library; the
        particle-path integrator in ``streamlines.integration`` calls straight into
        it rather than keeping a second copy.

        Args:
            _re : Relative Reynolds Number
            _mach : Relative Mach Number
            _model : Drag Model Name -- one of the strings tabulated below

        Returns:
            coefficient of drag based on local flow/particle properties

        Raises:
            ValueError: if the model name is unknown, or if the Reynolds number
                lies outside the range the requested model is defined over.

        Available models
        ----------------
        Re is the relative Reynolds number, M the relative Mach number and
        Kn the Knudsen number formed from them as ``Kn = M/Re * sqrt(pi*gamma/2)``.
        Every model returns 0 for ``Re <= 1e-9`` to keep the creeping-flow limit
        finite. The stated regimes are the range over which the source
        correlation was fitted or claimed valid, NOT a range this function
        enforces: apart from 'subramaniam-balachandar', which raises above
        Re = 3e5, the closures extrapolate silently outside their regime.

        ``'zero-drag'``
            Cd = 0. Not a physical closure -- it makes the particle a passive
            tracer that follows the fluid exactly, and exists to isolate
            inertia effects in a response study. No reference.
        ``'sphere'``
            Piecewise standard drag curve for a rigid sphere, incompressible
            (no M dependence), covering Re < 1e-3 up to Re >= 4e5.
            Reference: White, F. M., "Fluid Mechanics" (standard drag curve for
            a sphere). Not in paper.bib -- no edition or equation number was
            recorded with the original implementation, and the branch points
            were tuned by hand (see the in-line note on the case).
        ``'stokes'``
            Cd = 24/Re. Regime: creeping flow, Re << 1, M << 1, continuum
            (Kn << 1). Reference: Stokes, G. G. (1851), Trans. Cambridge Phil.
            Soc. 9, 8-106. Pre-DOI; not in paper.bib.
        ``'melling'``
            Slip-corrected Stokes drag, Cd = 24/Re * (1 + Kn)^-1. Regime:
            Re << 1, small but non-zero Kn (slip flow, Kn <~ 0.1); intended for
            PIV seed particles. Reference: Melling, A. (1997), "Tracer particles
            and seeding for particle image velocimetry", Meas. Sci. Technol.
            8(12), 1406-1416, doi:10.1088/0957-0233/8/12/005.
        ``'melling-2'``
            As ``'melling'`` but with the 2.7 prefactor on the Knudsen term,
            Cd = 24/Re * (1 + 2.7*Kn)^-1. Same regime and same reference
            (doi:10.1088/0957-0233/8/12/005).
        ``'oseen'``
            Oseen's correction to Stokes drag, Cd = 24/Re * (1 + 3*Re/16).
            Regime: creeping to low Reynolds flow, Re < 1, M << 1, continuum.
            Reference: Oseen, C. W. (1927), "Neuere Methoden und Ergebnisse in
            der Hydrodynamik", Akademische Verlagsgesellschaft, Leipzig. Pre-DOI
            monograph; no DOI exists.
        ``'schiller-nauman'``
            Cd = 24/Re * (1 + 0.15*Re^0.687). Regime: the correlation is
            classically quoted for Re <= 800; this implementation is used here
            for Re <~ 200 and M <~ 0.25, where compressibility is negligible.
            Reference: Schiller, L. and Naumann, A. (1933), Z. Ver. Dtsch. Ing.
            77, 318-320. No DOI exists. NOTE the year: this work is very widely
            miscited as 1935, and the paper.bib key ``schiller1933drag`` keeps
            that mis-citation as its key while carrying the correct 1933 date.
        ``'cunningham'``
            Cunningham slip correction on Stokes drag,
            Cd = 24/Re * (1 + 4.5*Kn)^-1, with the suite's single Knudsen
            definition at all Re. Regime: Re << 1, M << 1, Kn <~ 0.1.
            Reference: Cunningham, E. (1910), "On the velocity of steady fall of
            spherical particles through fluid medium", Proc. R. Soc. Lond. A
            83(563), 357-365, doi:10.1098/rspa.1910.0024. NOTE that the
            prefactor A = 4.5 is UNSOURCED -- it matches no published slip
            coefficient and could not be traced to Cunningham; see the in-line
            note on the case. The reference covers the form only.
        ``'henderson'``
            Henderson's correlation, valid across continuum, slip, transitional
            and free-molecular flow and across subsonic and supersonic speeds.
            Three branches: M < 1, M >= 1.75, and a linear blend between them.
            Regime: all Re and all M in the source; the sphere-temperature
            dependence of the original is dropped here (see the in-line note).
            Reference: Henderson, C. B. (1976), "Drag Coefficients of Spheres in
            Continuum and Rarefied Flows", AIAA Journal 14(6), 707-708,
            doi:10.2514/3.61409.
        ``'subramaniam-balachandar'``
            Piecewise standard drag curve assembled from four sub-correlations
            (Stokes, Clift, Schiller-Naumann, Clift-Gauvin). Regime:
            incompressible (no M dependence), Re < 3e5; raises ValueError above
            that rather than extrapolating.
            Reference: Subramaniam, S. and Balachandar, S. (eds.) (2022),
            "Modeling Approaches and Computational Methods for Particle-Laden
            Turbulent Flows", 1st ed., Elsevier, ISBN 978-0-323-90133-8. Note
            that Subramaniam and Balachandar are the volume's EDITORS, not the
            sole authors of the chapter this correlation comes from. Elsevier
            lists no DOI for the book.
        ``'loth'``
            Loth's compressibility- and rarefaction-corrected drag, split at
            Re = 45 into a rarefaction-dominated branch (Re < 45, blending a
            free-molecular limit with slip-corrected Schiller-Naumann) and a
            compression-dominated branch (Re > 45). Regime: all Re and all M
            in the source.
            Reference: Loth, E. (2008), "Compressibility and Rarefaction Effects
            on Drag of a Spherical Particle", AIAA Journal 46(9), 2219-2228,
            doi:10.2514/1.28943, as corrected by Harrison, A. K. (2021),
            doi:10.2514/1.J060681, which reports two errata in Loth's printed
            equations. This implementation follows the corrected equations; see
            the in-line note on the case.
        ``'tedeschi'``
            Tedeschi's correlation for tracer particles in supersonic flow,
            which solves implicitly for a velocity-lag factor k. Regime: all Re
            and all M in the source; developed for PIV tracers in supersonic
            flow.
            Reference: Tedeschi, G., Gouin, H. and Elena, M. (1999), "Motion of
            tracer particles in supersonic flows", Experiments in Fluids 26(4),
            288-296, doi:10.1007/s003480050291.

        Notes
        -----
        Equation numbers are given below only where the equation could actually
        be located in the source. Most of these papers are paywalled, and an
        equation number that could not be checked is deliberately omitted
        rather than guessed.

        """
        match _model:
            case 'zero-drag':
                # zero drag model to simulate fluid
                # Not a published closure: Cd = 0 makes the particle a massless
                # tracer that follows the fluid exactly. No reference applies.
                return 0

            case 'sphere':
                # Piecewise standard drag curve for a rigid sphere; incompressible.
                # ref: Fluid Mechanics, Frank M. White
                # This was decided by trail-and-error from VISUAL3 code
                #   (maintainer's note, kept: the branch points below were tuned
                #   by hand against the VISUAL3 code, so they are not lifted
                #   verbatim from White. No edition or equation number was
                #   recorded, and White is not in paper.bib for that reason.)
                # The individual arms are recognisable published forms:
                #   Re < 1e-3        -- Stokes drag, Cd = 24/Re.
                #   1e-3 <= Re < 1   -- Oseen's correction, Cd = 24/Re (1 + 3Re/16);
                #                       see Oseen (1927), the 'oseen' case below.
                #   1 <= Re < 800    -- Cd = 24/Re (1 + Re^(2/3)/6); the source of
                #                       this particular form is unrecorded here.
                #   800 <= Re < 4e5  -- Newton regime, Cd = 0.44.
                #   Re >= 4e5        -- post-drag-crisis value, Cd = 0.07.
                if _re <= 1e-9:
                    return 0
                if _re < 1e-3:
                    return 24 / _re
                if 1e-3 <= _re < 0.45:
                    return 24 / _re * (1 + 3 * _re / 16)
                if 0.45 <= _re < 1:
                    # Same as above due to lack of data
                    return 24 / _re * (1 + 3 * _re / 16)
                if 1 <= _re < 800:
                    return 24 / _re * (1 + _re ** (2 / 3) / 6)
                if 800 <= _re < 3e5:
                    return 0.44
                if 3e5 <= _re < 4e5:
                    # Same as above due to lack of data
                    return 0.44
                if _re >= 4e5:
                    return 0.07

            case 'stokes':
                # Stokes Drag; for creeping flow regime; Re << 1
                # Cd = 24/Re, the continuum creeping-flow limit (M << 1, Kn << 1).
                # ref: Stokes, G. G. (1851), "On the effect of the internal
                #   friction of fluids on the motion of pendulums", Trans.
                #   Cambridge Phil. Soc. 9, 8-106. Pre-DOI; no DOI exists and the
                #   work is not in paper.bib. Every other closure in this suite
                #   reduces to this expression as Re -> 0.
                if _re <= 1e-9:
                    return 0
                else:
                    return 24/_re

            case 'melling':
                # The popular melling correction
                # Slip-corrected Stokes drag for PIV seed particles,
                #   Cd = 24/Re * (1 + Kn)^-1,  Kn = M/Re * sqrt(pi*gamma/2).
                # Regime: Re << 1, slip flow (Kn <~ 0.1).
                # ref: Melling, A. (1997), "Tracer particles and seeding for
                #   particle image velocimetry", Measurement Science and
                #   Technology 8(12), 1406-1416,
                #   doi:10.1088/0957-0233/8/12/005.
                #   Equation number not verified -- the article is paywalled.
                if _re <= 1e-9:
                    return 0
                else:
                    knd = _mach / _re * np.sqrt(np.pi*self.gamma/2)
                    return 24/_re * (1 + knd)**-1

            case 'melling-2':
                # The melling correction with the 2.7 pre-factor on the Knudsen term
                #   Cd = 24/Re * (1 + 2.7*Kn)^-1,  Kn = M/Re * sqrt(pi*gamma/2).
                # Same regime and same source as the 'melling' case above.
                # ref: Melling, A. (1997), Measurement Science and Technology
                #   8(12), 1406-1416, doi:10.1088/0957-0233/8/12/005.
                #   Equation number not verified -- the article is paywalled.
                if _re <= 1e-9:
                    return 0
                else:
                    knd = _mach / _re * np.sqrt(np.pi*self.gamma/2)
                    return 24/_re * (1 + 2.7*knd)**-1

            case 'oseen':
                # Oseen's model; for creeping flow regime; Re < 1
                # Cd = 24/Re * (1 + 3/16 * Re); incompressible, continuum.
                # ref: Oseen, C. W. (1927), "Neuere Methoden und Ergebnisse in
                #   der Hydrodynamik", Akademische Verlagsgesellschaft, Leipzig
                #   (in German). Pre-DOI monograph; no DOI exists. Equation
                #   number not verified -- no accessible copy.
                if _re <= 1e-9:
                    return 0
                else:
                    return 24/_re * (1 + 3/16 * _re)

            case 'schiller-nauman':
                # Schiller and Nauman's model; for Re <~ 200 & M <~ 0.25
                # Cd = 24/Re * (1 + 0.15 * Re^0.687). Classically quoted as
                # valid to Re <= 800; the tighter Re <~ 200 above is this
                # library's working limit. Incompressible, continuum.
                # ref: Schiller, L. and Naumann, A. (1933), "Ueber die
                #   grundlegenden Berechnungen bei der Schwerkraftaufbereitung",
                #   Zeitschrift des Vereines Deutscher Ingenieure 77, 318-320
                #   (in German). No DOI exists. The year is very widely miscited
                #   as 1935 -- the paper.bib key is 'schiller1933drag' for that
                #   historical reason, but the entry itself carries 1933.
                #   Equation number not verified -- no accessible copy.
                if _re <= 1e-9:
                    return 0
                else:
                    return 24/_re * (1 + 0.15 * _re**0.687)

            case 'cunningham':
                # Cunningham model; for Re << 1; M << 1; Kn <~ 0.1
                # Cunningham slip correction on Stokes drag,
                #   Cd = 24/Re * (1 + A*Kn)^-1  with A = 4.5 as coded here,
                #   Kn = M/Re * sqrt(pi*gamma/2)  -- one definition at all Re.
                # ref: Cunningham, E. (1910), "On the velocity of steady fall of
                #   spherical particles through fluid medium", Proceedings of the
                #   Royal Society of London A 83(563), 357-365,
                #   doi:10.1098/rspa.1910.0024.
                #   The reference covers the FORM of the slip correction only.
                #
                # FIXED: the Re > 1 branch used to redefine Kn as M/sqrt(Re),
                # which made Cd step by 15.0% at M = 0.1 and 33.4% at M = 0.5
                # across Re = 1. That was a defect, not a modeling seam:
                #   (a) Kn = sqrt(pi*gamma/2) * M/Re is THE kinetic-theory
                #       relation for a hard-sphere gas with Kn built on the
                #       particle diameter. Verified against four independent
                #       sources: Harrison (2021), doi:10.2514/1.J060681, eq. 10
                #       (read from LA-UR-21-21429, https://www.osti.gov/servlets/purl/1812671),
                #       cited as `harrison2021comment`; Singh et al. (2022),
                #       doi:10.2514/1.J060648, which gives
                #       Kn_inf = M_inf/Re_inf * sqrt(gamma*pi/2), cited as
                #       `singh2022general` and read from the authors' open
                #       preprint; the review Capecelatro and Wagner (2024),
                #       doi:10.1146/annurev-fluid-121021-015818, same
                #       expression, cited as `capecelatro2024gasparticle` and
                #       read from arXiv:2303.00825; and NASA/NTRS 20220018430
                #       (AIAA 2023), eq. 4, at
                #       https://ntrs.nasa.gov/api/citations/20220018430/downloads/AIAA_2023_Palmer_Drag_Model_Paper_v2.pdf
                #       -- not in paper.bib, its full author list was not
                #       confirmed. It is also the
                #       definition the 'loth' case below already uses, the one
                #       this method's docstring states, and the one every other
                #       model in this suite uses.
                #   (b) M/sqrt(Re) is a real dimensionless group -- it is
                #       Tsien's rarefaction parameter, proportional to
                #       sqrt(M*Kn), i.e. to the SQUARE ROOT of a Knudsen number,
                #       and it is what appears in Henderson's exp(-0.5*M/sqrt(Re))
                #       factor in the 'henderson' case below. It is not Kn, and
                #       substituting it into a slip correction of the form
                #       (1 + A*Kn)^-1 mixes two different groups.
                #   (c) Both arms carried the SAME functional form and the SAME
                #       prefactor; only the meaning of the variable changed.
                #       Restoring one Kn therefore makes the two arms literally
                #       the same expression and the branch disappears, which a
                #       genuine seam between two source correlations could not
                #       do. Cunningham's slip correction is a single creeping-
                #       flow result; nothing in the literature branches it at
                #       Re = 1.
                # Cd is now continuous at Re = 1 to machine precision.
                #
                # A = 4.5 IS STILL UNSOURCED -- a third attempt to trace it
                # failed. Neither Cunningham (1910) nor Melling (1997) could be
                # retrieved (both paywalled, no open copy found), so neither was
                # read directly; no leading Knudsen coefficient of 4.5 appears
                # in any reachable secondary literature on slip-corrected sphere
                # drag. Nearby published values are 2.514 (Cunningham-Millikan-
                # Davies, as used by Loth's f(Kn) below), 1.257 (the standard
                # aerosol form of the same) and 2.7 (the 'melling-2' case
                # above). A radius-versus-diameter convention does not close the
                # gap: 2x2.514 = 5.028 and 2x2.7 = 5.4, neither is 4.5.
                #
                # TWO UNPROVEN LEADS, recorded so the next attempt need not
                # repeat the search. NEITHER IS AN ATTRIBUTION; do not cite
                # either as the source of 4.5 without reading the paper.
                #   1. The 'tedeschi' case below contains the group
                #      9/4 * 2*Kn = 4.5*Kn inside its implicit equation for k
                #      (the a2 coefficient), i.e. 9/4 acting on a radius-based
                #      Knudsen number 2*Kn. Tedeschi, Gouin and Elena (1999)
                #      describe their model as "extending Cunningham's method to
                #      higher velocities and Knudsen numbers" (abstract, via
                #      doi:10.1007/s003480050291), so 4.5 plausibly came from
                #      there rather than from Cunningham (1910) directly. The
                #      full text is paywalled and was not read.
                #   2. 4.5 is also the leading numerator constant of Henderson's
                #      subsonic bridging term, (4.5 + 0.38*(0.03*Re +
                #      0.48*sqrt(Re)))/(1 + 0.03*Re + 0.48*sqrt(Re)) -- see the
                #      'henderson' case below, and Appendix B of
                #      `singh2022general`, which reproduces it. That term is
                #      multiplied by
                #      exp(-0.5*M/sqrt(Re)), so Henderson's equation contains
                #      BOTH constants this case used to carry. Circumstantial
                #      only, and the git history runs the wrong way (the
                #      'cunningham' case predates the 'henderson' case by a day),
                #      so this is a coincidence worth checking, not evidence.
                # Until A is traced, treat the magnitude of this model's slip
                # correction as unvalidated and do not present its output as
                # "the Cunningham correction". The Re = 1 discontinuity is
                # resolved; the prefactor is not.
                if _re <= 1e-9:
                    return 0
                _kn = _mach / _re * np.sqrt(self.gamma * np.pi/2)
                return 24/_re * (1 + 4.5*_kn)**-1

            case 'henderson':
                # Henderson model; for all flow regimes
                # Simplified by ignoring sphere temperature
                #   (maintainer's note, kept: Henderson's correlation carries a
                #   sphere-to-gas temperature ratio; this implementation drops it,
                #   i.e. it assumes the sphere is in thermal equilibrium with the
                #   gas. That is a deliberate simplification of the source.)
                # Covers continuum through free-molecular flow, subsonic and
                # supersonic. Three branches, in the order coded below:
                #   M < 1        -- subsonic correlation, _f1 + _f2 + _f3
                #   M >= 1.75    -- supersonic correlation, (_g1 + _g2)/_g3
                #   1 <= M < 1.75-- linear interpolation between the two
                # ref: Henderson, C. B. (1976), "Drag Coefficients of Spheres in
                #   Continuum and Rarefied Flows", AIAA Journal 14(6), 707-708,
                #   doi:10.2514/3.61409.
                #   Equation numbers not verified -- the article is paywalled and
                #   could not be retrieved to confirm which numbered equation each
                #   branch corresponds to.
                if _re < 1e-9:
                    return 0

                # For Mach < 1
                _s = _mach * np.sqrt(self.gamma/2)
                if np.all(_s <= 1e-12):
                    # Incompressible limit: s * exp(-0.247 Re/s) -> 0 as s -> 0.
                    # Evaluating it directly at M = 0 divides by s = 0.
                    _f1 = 24 / _re
                else:
                    _f1 = 24 * (_re + _s * (5.89688 * np.exp(-0.247 * _re/_s)))**-1
                _f2 = np.exp(-0.5*_mach/np.sqrt(_re)) * \
                      ((4.5 + 0.38*(0.03*_re + 0.48*np.sqrt(_re))) / (1 + 0.03*_re + 0.48*np.sqrt(_re)) +
                       0.1*_mach**2 + 0.2*_mach**8)
                _f3 = (1 - np.exp(-_mach/_re))*0.6*_s
                _cd1 = _f1 + _f2 + _f3
                if _mach < 1:
                    return _cd1

                # For Mach >= 1.75
                _mach_inf = _mach
                _re_inf = _re
                _s_inf = _mach_inf * np.sqrt(self.gamma/2)
                _g1 = 0.9 + 0.34/_mach_inf**2
                _g2 = 1.86 * np.sqrt(_mach_inf/_re_inf) * (2 + 2/_s_inf**2 + 1.058/_s_inf - 1/_s_inf**4)
                _g3 = 1 + 1.86 * np.sqrt(_mach_inf/_re_inf)
                _cd2 = (_g1 + _g2) / _g3
                if _mach >= 1.75:
                    return _cd2

                # For 1 <= Mach < 1.75; linear interpolation
                if 1 <= _mach < 1.75:
                    return _cd1 + 4/3 * (_mach_inf - 1) * (_cd2 - _cd1)

            case 'subramaniam-balachandar':
                # Piecewise standard drag curve; incompressible (no M dependence);
                # defined for Re < 3e5 and raises above that.
                # ref: Subramaniam, S. and Balachandar, S. (eds.) (2022),
                #   "Modeling Approaches and Computational Methods for
                #   Particle-Laden Turbulent Flows", 1st ed., Elsevier,
                #   ISBN 978-0-323-90133-8. Elsevier lists no DOI for the book.
                #   Subramaniam and Balachandar are the volume's EDITORS, not the
                #   sole authors of the chapter this correlation is taken from;
                #   the model string keeps their names for backwards
                #   compatibility. Chapter, page and equation numbers were not
                #   verified -- no accessible copy.
                # The four arms are named published correlations; each is
                # labeled at its branch below.
                if _re < 1e-9:
                    return 0

                if _re < 0.5:
                    # Stokes drag -- see the 'stokes' case above for the reference.
                    return 24/_re

                if _re < 20:
                    # Clift's correlation. Attributed in the source volume to
                    # Clift, R., Grace, J. R. and Weber, M. E. (1978), "Bubbles,
                    # Drops and Particles", Academic Press. Not in paper.bib and
                    # not independently verified.
                    return 24/_re * (1 + 0.1315 * _re**(0.82-0.05*np.log10(_re)))

                if _re < 800:
                    # Schiller-Naumann -- see the 'schiller-nauman' case above for
                    # the reference (Schiller and Naumann, 1933).
                    return 24/_re * (1 + 0.15 * _re**0.687)

                if _re < 3e5:
                    # Clift-Gauvin correlation. Attributed in the source volume to
                    # Clift, R. and Gauvin, W. H. (1971). Not in paper.bib and not
                    # independently verified.
                    return 24/_re * (1 + 0.15 * _re**0.687 + 0.42/24 * _re * (1 + 4.25e4 * _re**(-1.16))**-1)

                raise ValueError(
                    f"drag model 'subramaniam-balachandar' is only defined for "
                    f"Reynolds numbers below 3e5; got _re={float(np.max(_re)):g}")

            case 'loth':
                # Loth's model; for all flow regimes
                # Compressibility- and rarefaction-corrected sphere drag, split
                # at Re = 45 into a rarefaction-dominated branch (Re < 45) and a
                # compression-dominated branch (Re > 45).
                # ref: Loth, E. (2008), "Compressibility and Rarefaction Effects
                #   on Drag of a Spherical Particle", AIAA Journal 46(9),
                #   2219-2228, doi:10.2514/1.28943.
                # ERRATA IN THE SOURCE -- ALREADY ACCOUNTED FOR BELOW.
                #   Harrison, A. K. (2021), "Comment on 'Compressibility and
                #   Rarefaction Effects on Drag of a Spherical Particle'", AIAA
                #   Journal 59(8), 3288-3289, doi:10.2514/1.J060681, identifies
                #   two errors in Loth (2008). Read from the accepted manuscript
                #   (LA-UR-21-21429) at https://www.osti.gov/servlets/purl/1812671
                #   since the AIAA page is not retrievable:
                #     (i) Loth's eq. (25b), the free-molecular drag at particle-
                #         gas temperature equilibrium, is printed without the
                #         2*sqrt(pi)/(3*s) term that eq. (25a) carries. The
                #         omitted term is positive and large -- dropping it cuts
                #         C_D,fm by 17-27% over M = 0.5-3.
                #     (ii) Loth's eq. (26) carries a stray prime: the drag
                #         coefficient in the denominator should be the same
                #         (unprimed) C_D,fm as in the numerator, so that the
                #         correlation passes through the C_D = 1.63 nexus at
                #         Re_p = 45.
                #   The code below already matches Harrison's corrected forms on
                #   both counts: _cd_fm keeps the 2*sqrt(pi)/(3*s) term, and
                #   _cd_fm_re uses _cd_fm in both numerator and denominator,
                #   which yields exactly 1.63 at _re = 45 for every Mach number
                #   (checked numerically) -- consistent with the `_re == 45`
                #   branch below. So no change is required here; this note
                #   exists so that a reader comparing the code line-by-line
                #   against the printed paper does not "fix" it back to the
                #   erroneous published equations.
                #   Loth published a reply, doi:10.2514/1.J060850; its content
                #   has NOT been read (AIAA blocks retrieval and no open copy
                #   was found), so whether he accepts both corrections is
                #   unverified here.
                if _re < 1e-9:
                    return 0

                if _re < 45:
                    # Rarefraction dominated domain
                    _s = _mach * np.sqrt(self.gamma/2)
                    _kn = (np.pi * self.gamma / 2)**0.5 * _mach / _re
                    if np.all(_mach <= 1e-12):
                        # Incompressible limit: the free-molecular contribution is
                        # weighted by mach**4 and vanishes, while f(Kn) -> 1, so the
                        # expression below reduces analytically to Schiller-Naumann.
                        # Evaluating it directly at M = 0 divides by _s = 0.
                        return 24/_re * (1 + 0.15 * _re**0.687)
                    _cd_fm = (1 + 2 * _s**2) * np.exp(-_s**2) / (_s**3 * np.pi**0.5) + \
                             (4*_s**4 + 4*_s**2 - 1) * erf(_s) / (2*_s**4) + 2 * np.pi**0.5 / (3 * _s)
                    _cd_fm_re = _cd_fm / (1 + (_cd_fm/1.63 - 1) * (_re/45)**0.5)
                    _f_kn = (1 + _kn * (2.514 + 0.8 * np.exp(-0.55/_kn)))**-1
                    _cd_kn_re = 24/_re * (1 + 0.15 * _re**0.687) * _f_kn
                    _cd = (_cd_kn_re + _mach**4 * _cd_fm_re) / (1 + _mach**4)
                    return _cd

                if _re == 45:
                    return 1.63

                if _re > 45:
                    # compression-dominated regime
                    # C_M = 5/3 + 2/3 tanh(3 ln(M + 0.1)) -- Loth (2008), eq. 12
                    if _mach <= 1.45:
                        _cm = 5/3 + 2/3 * np.tanh(3 * np.log(_mach + 0.1))
                    else:
                        _cm = 2.044 + 0.2 * np.exp(-1.8 * (np.log(_mach/1.5))**2)
                    if _mach <= 0.89:
                        _gm = 1 - 1.525 * _mach**4
                    else:
                        _gm = 2e-4 + 8e-4 * np.tanh(12.77 * (_mach - 2.02))
                    _hm = 1 - 0.258 * _cm / (1 + 514 * _gm)
                    _cd = 24/_re * (1 + 0.15 * _re**0.687) * _hm + 0.42 * _cm / (1 + 42000 * _gm / _re**1.16)
                    return _cd

            case 'tedeschi':
                # Tedeschi's model; for all flow regimes
                # Correlation for tracer particles in supersonic flow. Solves
                # implicitly (via fsolve) for the velocity-lag factor k, then
                # applies a compressibility factor c and a rarefaction factor
                # epsilon(Kn) to Schiller-Naumann drag.
                # ref: Tedeschi, G., Gouin, H. and Elena, M. (1999), "Motion of
                #   tracer particles in supersonic flows", Experiments in Fluids
                #   26(4), 288-296, doi:10.1007/s003480050291.
                #   Equation numbers not verified -- the article is paywalled.
                #
                # FIXED: this case used to carry the same split as the
                # 'cunningham' case above -- Kn = M/Re * sqrt(gamma*pi/2) for
                # Re <= 1 and Kn = M/sqrt(Re) above it -- which made Cd step by
                # 4.4% at M = 0.1 and 13.4% at M = 0.5 across Re = 1. It was
                # copied: the two lines were added verbatim in commit 41538de
                # (Mar 2024), seventeen months after the 'cunningham' case they
                # came from (commit 6f99987, Oct 2022). See the note on that
                # case for the sources establishing the correct relation.
                #
                # Beyond that shared argument, this case carries its own
                # internal proof that M/Re * sqrt(pi*gamma/2) is the definition
                # its equations were written for. Inside _solve_k below, a1
                # contains the group (s * sqrt(pi) / Kn)**0.687. With
                # s = M*sqrt(gamma/2) and Kn = M/Re * sqrt(pi*gamma/2),
                #     s * sqrt(pi) / Kn = Re   exactly, for every M,
                # so that group is the Schiller-Naumann Re**0.687 factor, which
                # is what the surrounding correlation needs it to be. Under
                # Kn = M/sqrt(Re) the same group collapses to
                # sqrt(pi*gamma/2 * Re) -- Mach-independent, and not a Reynolds
                # number -- so the Re > 1 arm was feeding fsolve an equation
                # inconsistent with its own derivation. That settles it as a
                # transcription bug rather than a seam between two source
                # correlations, which is consistent with the paper's abstract:
                # it states a single expression valid "from continuum to free
                # molecule conditions, for Re <~ 200 and M <~ 1", with no
                # branch at Re = 1.
                # Cd is now continuous at Re = 1 to machine precision.
                if _re < 1e-9:
                    return 0
                if np.all(_mach <= 1e-12):
                    # Continuum limit: Kn -> 0 gives k -> 1, c -> 1 and
                    # epsilon(Kn) -> 1, leaving Schiller-Naumann. Evaluating the
                    # general expression at M = 0 divides by Kn = 0.
                    return 24/_re * (1 + 0.15 * _re**0.687)
                _kn = _mach / _re * np.sqrt(self.gamma * np.pi/2)

                s = _mach * np.sqrt(self.gamma/2)

                def _solve_k(_k):
                    s_prime = (1 - _k) * s
                    epsilon_prime = 3/8 * (np.pi**2 / s_prime) * (1 + s_prime**2) * s_prime + np.exp(-s_prime**2) /4
                    a1 = 9/4 * 0.15 * 2 * _kn / epsilon_prime * (s * np.pi**0.5 / _kn)**0.687
                    a2 = 1 + 9/4 * 2 * _kn / epsilon_prime
                    return a1 * _k**1.687 + a2 * _k - 1

                # solve the equation
                k = fsolve(_solve_k, np.array([0.5]))

                c = 1 + _re**2 / (_re**2 + 100) * np.e**(-0.225/_mach**2.5)
                _epsilon_kn = 1.177 + 0.177 * (0.851 * _kn**1.16 - 1) / (0.851 * _kn**1.16 + 1)

                return 24/_re * k * (1 + 0.15 * (k*_re)**0.687) * c * _epsilon_kn

            case _:
                raise ValueError(f"unknown drag model {_model!r}")

    def compute(self):
        # implicitly runs compute_velocity() and compute_temperature()
        """
        Function to compute all the attributes in the class
        :return: None
        """
        self.compute_pressure()
        # This computes viscosity using the keyes formula
        # To compute using sutherland's law, call compute_viscosity() separately, which will update the attribute
        self.compute_viscosity()
