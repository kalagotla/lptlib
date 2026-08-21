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

from synthetic import make_oblique_shock_case  # noqa: E402


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
