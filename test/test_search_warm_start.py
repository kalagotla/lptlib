"""The Newton-Raphson warm start in ``Search.p2c`` must be per-instance.

It used to live in a module global, which made results depend on which point
had been searched last -- across particles, across threads, and across tests.
These tests pin the fixed behaviour: the cache is an instance attribute, two
instances never see each other's guess, and concurrent searches agree with
serial ones.
"""

import sys
from multiprocessing.pool import ThreadPool

import numpy as np

from lptlib.streamlines import Search


def _sample_points(grid, n=8):
    """A handful of interior physical-space points spread through the block."""
    ni, nj, nk = grid.ni[0], grid.nj[0], grid.nk[0]
    idx = np.linspace(1, min(ni, nj) - 3, n).astype(int)
    return [np.array([grid.grd[i, j, nk // 2, 0, 0],
                      grid.grd[i, j, nk // 2, 1, 0],
                      grid.grd[i, j, nk // 2, 2, 0]]) + 1e-6
            for i, j in zip(idx, idx[::-1])]


def test_no_module_global_warm_start():
    """The module must not grow a ``_cpoint`` global when p2c runs."""
    module = sys.modules[Search.__module__]
    assert not hasattr(module, "_cpoint")


def test_warm_start_is_instance_attribute(synthetic_grid):
    point = _sample_points(synthetic_grid, 1)[0]
    search = Search(synthetic_grid, point)
    assert search._cpoint is None
    search.compute(method='p-space')
    assert search._cpoint is not None
    # a brand new instance starts cold again
    assert Search(synthetic_grid, point)._cpoint is None


def test_results_independent_of_instance_history(synthetic_grid):
    """A reused instance and fresh instances must land in the same cell."""
    points = _sample_points(synthetic_grid)

    fresh = []
    for point in points:
        search = Search(synthetic_grid, point)
        search.compute(method='p-space')
        fresh.append(search.cpoint)

    # Same points, but walked with a single warm-started instance, and in a
    # different order first to make any order dependence show up.
    reused = Search(synthetic_grid, points[-1])
    reused.compute(method='p-space')
    warm = []
    for point in points:
        reused.ppoint = point
        warm.append(reused.p2c(point).copy())

    for a, b in zip(fresh, warm):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_concurrent_searches_match_serial(synthetic_grid):
    """Threads each get their own Search, so no warm start is raced on."""
    points = _sample_points(synthetic_grid)

    def run(point):
        search = Search(synthetic_grid, point)
        search.compute(method='p-space')
        return search.cpoint

    serial = [run(point) for point in points]
    with ThreadPool(4) as pool:
        threaded = pool.map(run, points)

    for a, b in zip(serial, threaded):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_warm_start_argument_seeds_the_guess(synthetic_grid):
    """An explicit warm start is copied in and reaches the same answer.

    Tracking loops build a fresh Search per step and pass the previous step's
    converged c-space point, which skips the nearest-node scan. The seed is
    copied, so the caller's array is never mutated, and a good seed must not
    change where Newton-Raphson lands.
    """
    points = _sample_points(synthetic_grid, 4)

    cold = Search(synthetic_grid, points[0])
    cold.compute(method='p-space')
    seed = cold._cpoint.copy()

    warm = Search(synthetic_grid, points[1], warm_start=seed)
    assert warm._cpoint is not seed          # copied, not aliased
    np.testing.assert_allclose(warm._cpoint, seed)
    warm.compute(method='p-space')

    reference = Search(synthetic_grid, points[1])
    reference.compute(method='p-space')

    np.testing.assert_allclose(warm.cpoint, reference.cpoint, atol=1e-6)
    np.testing.assert_allclose(seed, cold.cpoint)  # caller's array untouched


def test_streamlines_warm_start_is_per_instance(synthetic_grid, synthetic_flow):
    """Streamlines carries its own warm start and starts each run cold."""
    from lptlib.streamlines import Streamlines

    grd = synthetic_grid.grd
    node = np.array([grd[2, 4, 2, 0, 0], grd[2, 4, 2, 1, 0], grd[2, 4, 2, 2, 0]])
    spacing = grd[3, 4, 2, 0, 0] - grd[2, 4, 2, 0, 0]
    point = list(node + np.array([0.3 * spacing, 0.3 * spacing, 0.0]))

    first = Streamlines(None, None, point=point)
    first.time_step = 1e-6
    assert first._last_search is None
    first.compute(method='p-space', grid=synthetic_grid, flow=synthetic_flow)
    assert first._last_search is not None

    # A second object shares nothing with the first.
    second = Streamlines(None, None, point=point)
    assert second._last_search is None
    second.time_step = 1e-6
    second.compute(method='p-space', grid=synthetic_grid, flow=synthetic_flow)

    np.testing.assert_allclose(np.asarray(second.streamline),
                               np.asarray(first.streamline), rtol=0, atol=0)
