"""pytest configuration and shared fixtures for the lptlib test suite.

Adds the ``test/`` directory to ``sys.path`` so the shared ``testdata`` and
``synthetic`` helper modules can be imported from test modules in this
directory and its subdirectories, and exposes session-scoped synthetic grid
and flow fixtures so the streamline stack can be exercised without any external
data files.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from synthetic import make_oblique_shock_case  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_search_warm_start():
    """Clear the module-global Newton-Raphson warm start before every test.

    ``Search.p2c`` caches the previous computational-space guess in a module
    global to speed up successive particle-tracking steps. Left in place across
    tests it makes cell-search results depend on execution order. Clearing it
    before each test keeps every test deterministic and independent.
    """
    import sys
    from lptlib.streamlines import Search
    search = sys.modules[Search.__module__]
    if hasattr(search, "_cpoint"):
        del search._cpoint
    yield


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
