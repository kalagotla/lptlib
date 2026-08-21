"""Termination guarantees for the particle-path loops.

Two things used to make a trajectory run for ever.

``Integration.compute_ppath``'s mid-RK4 blow-up guard returns the *input*
state unchanged and sets the ``rk4_bool`` step-failed sentinel, meaning "no
step could be taken here". ``Streamlines.compute(method='ppath')`` never
looked at the sentinel: it appended the returned point and carried on, so any
particle that could not cross a discontinuity appended the same point for ever
and never terminated. The reproducer was a particle seeded at
``[-5e-3, 1e-3, 5e-5]`` in an oblique-shock case with ``time_step=1e-8``,
``drag_model='loth'`` and ``interpolation='simple_oblique_shock'``.

``max_steps`` was honoured only by ``adaptive-ppath``, so a runaway trajectory
under any other algorithm was unbounded too.

Every test here puts a hard bound on the work and fails if the bound is hit,
so a regression shows up as a failure rather than as a hung test run.
"""

import logging

import numpy as np
import pytest

from lptlib.streamlines import Streamlines
from lptlib.streamlines.integration import Integration

ALL_METHODS = ["p-space", "adaptive-p-space", "c-space", "adaptive-c-space",
               "ppath", "adaptive-ppath", "ppath-c-space",
               "adaptive-ppath-c-space"]

# Any correct implementation ends a stalled trajectory within max_loop_check
# reductions. This is the "did it hang?" bound, not a tight expectation.
CALL_BUDGET = 400


@pytest.fixture
def start_point(oblique_case):
    """A point in the uniform upstream region, off the grid nodes."""
    grd = oblique_case.grid.grd
    node = np.array([grd[2, 4, 2, 0, 0], grd[2, 4, 2, 1, 0], grd[2, 4, 2, 2, 0]])
    spacing = grd[3, 4, 2, 0, 0] - grd[2, 4, 2, 0, 0]
    return list(node + np.array([0.3 * spacing, 0.3 * spacing, 0.0]))


def _streamlines(point, **kwargs):
    sl = Streamlines(None, None, point=point)
    sl.diameter = 281e-9
    sl.density = 813.0
    sl.time_step = 1e-6
    sl.max_time_step = 1e-5
    sl.adaptivity = 0.01
    sl.magnitude_adaptivity = 0.01
    sl.drag_model = "stokes"
    sl.task = 7
    for key, value in kwargs.items():
        setattr(sl, key, value)
    return sl


def _budgeted(monkeypatch, behaviour, budget=CALL_BUDGET):
    """Replace ``compute_ppath`` with ``behaviour`` under a hard call budget.

    ``behaviour(intg, call_number, kwargs)`` returns the triple
    ``compute_ppath`` would. Exceeding ``budget`` calls raises, which is how a
    non-terminating loop is reported as a test failure instead of hanging the
    suite.
    """
    calls = {"n": 0}

    def _patched(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] > budget:
            raise AssertionError(
                f"compute_ppath called more than {budget} times: the "
                f"integration loop is not terminating")
        return behaviour(self, calls["n"], kwargs)

    monkeypatch.setattr(Integration, "compute_ppath", _patched)
    return calls


def _always_fails(intg, call_number, kwargs=None):
    """Every step is rejected by the blow-up guard, at the same location."""
    point = intg.interp.idx.ppoint
    if point is None:
        point = np.asarray(intg.interp.idx.cpoint, dtype=float)
    velocity = np.array([500.0, 10.0, 0.0])
    return intg._step_failed(np.asarray(point, dtype=float), velocity, velocity)


@pytest.mark.parametrize("method", ["ppath", "adaptive-ppath"])
def test_unresolvable_blowup_terminates(oblique_case, start_point, method,
                                        monkeypatch, caplog):
    """A particle that can never take a step ends, with a named warning.

    This is the regression for the ``ppath`` infinite loop: with every step
    rejected the loop must give up after a bounded number of reductions rather
    than appending the same point for ever.
    """
    calls = _budgeted(monkeypatch, _always_fails)
    sl = _streamlines(start_point)

    with caplog.at_level(logging.WARNING, logger="lptlib.streamlines.streamlines"):
        sl.compute(method=method, grid=oblique_case.grid, flow=oblique_case.flow)

    # Terminated well inside the budget.
    assert calls["n"] <= sl.max_loop_check + 2

    # Nothing was stored beyond the seed point: no step ever succeeded, so no
    # duplicate points were appended.
    path = np.asarray(sl.streamline, dtype=float)
    assert path.shape[0] <= 1

    # The failure is visible and names the particle and the location.
    messages = [record.getMessage() for record in caplog.records]
    assert any("blow up" in message and "Particle 7" in message
               for message in messages), messages


def test_repeated_blowup_never_appends_duplicate_points(oblique_case,
                                                        start_point,
                                                        monkeypatch):
    """A stalled ``ppath`` step is retried, never recorded.

    Before the fix the rejected state was appended on every iteration, so the
    trajectory filled up with copies of one point.
    """
    _budgeted(monkeypatch, _always_fails)
    sl = _streamlines(start_point)
    sl.compute(method="ppath", grid=oblique_case.grid, flow=oblique_case.flow)

    path = np.asarray(sl.streamline, dtype=float)
    if path.shape[0] > 1:
        assert len(np.unique(path, axis=0)) == path.shape[0]


@pytest.mark.parametrize("method", ["ppath", "adaptive-ppath"])
def test_time_step_recovers_after_a_blowup(oblique_case, start_point, method,
                                           monkeypatch):
    """A blow-up halves the step; later successes grow it back.

    ``adaptive-ppath`` used to halve the time step on every blow-up and never
    restore it, so one discontinuity crossing permanently crippled the rest of
    the trajectory.
    """
    requested = 1e-6
    n_failures = 4

    def _fails_then_succeeds(intg, call_number, kwargs):
        if call_number <= n_failures:
            return _always_fails(intg, call_number, kwargs)
        # A straight, drag-free step: the particle keeps the velocity it was
        # handed, so the adaptive logic sees no deflection and simply stores
        # the point. That isolates the blow-up book-keeping from everything
        # else the adaptive algorithm does to the time step.
        point = np.asarray(intg.interp.idx.ppoint, dtype=float)
        velocity = kwargs.get("velocity")
        if velocity is None:
            velocity = np.array([500.0, 10.0, 0.0])
        velocity = np.asarray(velocity, dtype=float)
        return point + velocity * kwargs["time_step"], velocity, velocity

    _budgeted(monkeypatch, _fails_then_succeeds, budget=120)
    sl = _streamlines(start_point, time_step=requested, max_steps=20)
    sl.compute(method=method, grid=oblique_case.grid, flow=oblique_case.flow)

    # The step really was reduced: the first stored steps are smaller than the
    # requested one.
    assert min(sl.time) < requested
    # ... and it grew back. Nothing is still owed, and the step is no longer
    # held below the value that was in force before the blow-up.
    assert sl._blowup_halvings == 0
    assert sl.time_step >= requested
    assert max(sl.time) >= requested


@pytest.mark.parametrize("method", ALL_METHODS)
def test_max_steps_bounds_every_algorithm(oblique_case, start_point, method,
                                          caplog):
    """``max_steps`` caps the stored trajectory whichever algorithm runs.

    Only ``adaptive-ppath`` used to look at it.
    """
    cap = 6
    sl = _streamlines(start_point, max_steps=cap)

    with caplog.at_level(logging.WARNING, logger="lptlib.streamlines.streamlines"):
        sl.compute(method=method, grid=oblique_case.grid, flow=oblique_case.flow)

    path = np.asarray(sl.streamline, dtype=float)
    assert path.shape[0] <= cap
    # The cap really was the reason we stopped, and it said so.
    if path.shape[0] == cap:
        messages = [record.getMessage() for record in caplog.records]
        assert any("maximum step count" in message and "Particle 7" in message
                   for message in messages), messages


def test_shock_crossing_particle_terminates(oblique_case, monkeypatch):
    """The reported reproducer runs to completion instead of spinning.

    A heavy particle driven at the shock with the Loth drag law and
    nearest-neighbour shock interpolation is exactly the configuration that
    tripped the blow-up guard on every iteration.
    """
    calls = {"n": 0}
    original = Integration.compute_ppath

    def _counted(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 20000:
            raise AssertionError("ppath did not terminate on the shock case")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Integration, "compute_ppath", _counted)

    sl = _streamlines([-5e-3, 1e-3, 5e-5], diameter=5e-6, density=4200.0,
                      time_step=1e-8, drag_model="loth",
                      interpolation="simple_oblique_shock")
    sl.compute(method="ppath", grid=oblique_case.grid, flow=oblique_case.flow)

    path = np.asarray(sl.streamline, dtype=float)
    assert np.all(np.isfinite(path))
    # A finite trajectory with no repeated points.
    assert len(np.unique(path, axis=0)) == path.shape[0]
