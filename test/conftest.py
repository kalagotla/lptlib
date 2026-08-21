"""pytest configuration and shared fixtures for the lptlib test suite.

Forces the non-interactive matplotlib backend, adds the ``test/`` directory to
``sys.path`` so the shared ``testdata`` and ``synthetic`` helper modules can be
imported from test modules in this
directory and its subdirectories, and exposes session-scoped synthetic grid
and flow fixtures so the streamline stack can be exercised without any external
data files.
"""

import os
import sys

import matplotlib
import numpy as np
import pytest

# Never open a GUI window during the test run: some tests build figures, and a
# blocking backend would hang CI. Must be set before pyplot is first imported.
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))

from synthetic import (make_analytic_flow,  # noqa: E402
                       make_coordinate_flow,
                       make_curvilinear_annulus_grid,
                       make_oblique_shock_case)


@pytest.fixture(scope="session")
def oblique_case():
    """A small, deterministic synthetic oblique-shock grid and flow field.

    Session scoped because building it runs the grid-metric computation, which
    is pure and deterministic, so a single shared instance is safe for the
    read-only search, interpolation, and integration tests.
    """
    return make_oblique_shock_case()


@pytest.fixture(scope="session")
def synthetic_grid(oblique_case):
    """The ``GridIO`` object from the synthetic oblique-shock case."""
    return oblique_case.grid


@pytest.fixture(scope="session")
def synthetic_flow(oblique_case):
    """The ``FlowIO`` object from the synthetic oblique-shock case."""
    return oblique_case.flow


@pytest.fixture(scope="session")
def upstream_point(oblique_case):
    """A grid node that sits well upstream of the shock (x < 0).

    Returned as physical-space ``[x, y, z]`` coordinates taken directly from a
    grid node, so interpolation there should recover the constant pre-shock
    state exactly.
    """
    grd = oblique_case.grid.grd
    return np.array([grd[3, 5, 2, 0, 0], grd[3, 5, 2, 1, 0], grd[3, 5, 2, 2, 0]])


@pytest.fixture(scope="session")
def curvilinear_grid():
    """A curvilinear quarter-annulus ``GridIO`` with metrics computed.

    Session scoped for the same reason as ``oblique_case``: building it runs
    the deterministic grid-metric computation and the search tests only read
    from it.

    Unlike every Cartesian fixture here, this block does not fill its own
    bounding box, so points can lie inside the bounding box while being
    outside the block. That is what ``test_search_curvilinear.py`` exercises.
    """
    return make_curvilinear_annulus_grid()


@pytest.fixture(scope="session")
def curvilinear_stretched_grid():
    """A quarter-annulus with the radial spacing stretched.

    ``curvilinear_grid`` above is curved, but its radial node lines are
    straight rays along which ``r`` is linear in ``i``, so the mapping is
    exactly linear in the i direction. That accident hides cell-indexing
    errors: interpolating along i from the wrong cell, at a local fraction
    outside ``[0, 1]``, still reproduces a linear field exactly. Stretching the
    radius removes it, so a wrong cell shows up as a real error in the
    interpolated value. See ``test_curvilinear_cell_indexing.py``.
    """
    return make_curvilinear_annulus_grid(radial_stretch=2.5)


@pytest.fixture(scope="session")
def curvilinear_analytic_flow(curvilinear_stretched_grid):
    """The closed-form vortex field sampled on the stretched annulus."""
    return make_analytic_flow(curvilinear_stretched_grid)


@pytest.fixture(scope="session")
def curvilinear_coordinate_flow(curvilinear_stretched_grid):
    """``q = (x, y, z, 0, 0)`` on the stretched annulus.

    Interpolating it recovers the query point when the cell is right and a
    displaced point when it is wrong.
    """
    return make_coordinate_flow(curvilinear_stretched_grid)
