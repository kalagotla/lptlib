"""Stochastic ensemble tests on the synthetic oblique-shock case.

This file used to launch 1300 particles through ``multi_process`` against
PLOT3D shock data that is not tracked in the repository, and asserted nothing.
It now runs a handful of particles on the in-memory synthetic fixture and
checks the parts that matter: the diameter distributions honour their inputs,
spawn locations lie on the requested line, and -- most importantly -- the
threaded driver produces exactly the same trajectories as the serial one.

That last check is the regression test for the Newton-Raphson warm start, which
used to live in a module global and was therefore raced on by every thread in
``multi_thread``.
"""

import os
import tempfile

import numpy as np
import pytest

from lptlib.streamlines import StochasticModel, Particle, SpawnLocations

N_PARTICLES = 3


def _particles(distribution="gaussian", n=N_PARTICLES, seed=11, **kwargs):
    particle = Particle(seed=seed)
    particle.distribution = distribution
    particle.min_dia = kwargs.get("min_dia", 200e-9)
    particle.max_dia = kwargs.get("max_dia", 400e-9)
    particle.mean_dia = kwargs.get("mean_dia", 281e-9)
    particle.std_dia = kwargs.get("std_dia", 20e-9)
    particle.density = 813.0
    particle.n_concentration = n
    particle.distribution_parameter = kwargs.get("distribution_parameter", 2.0)
    particle.compute_distribution()
    return particle


def _spawn(particle):
    spawn = SpawnLocations(particle)
    spawn.x_min = -1e-3
    spawn.z_min = 5e-5
    spawn.y_min, spawn.y_max = 2e-3, 13e-3
    spawn.compute()
    return spawn


def _model(oblique_case, particle, spawn, filepath):
    model = StochasticModel(particle, spawn, grid=oblique_case.grid,
                            flow=oblique_case.flow)
    model.method = "adaptive-ppath"
    model.drag_model = "stokes"
    model.search = "p-space"
    model.time_step = 1e-9
    model.max_time_step = 1e-7
    model.adaptivity = 0.01
    model.max_steps = 12
    model.filepath = filepath
    return model


@pytest.mark.parametrize("distribution",
                         ["gaussian", "uniform", "skewnorm", "lognorm"])
def test_distribution_respects_bounds_and_size(distribution):
    """Every distribution yields n diameters clipped to [min_dia, max_dia]."""
    particle = _particles(distribution, n=64)
    field = np.asarray(particle.particle_field)

    assert field.shape == (64,)
    assert np.all(np.isfinite(field))
    assert np.all(field >= particle.min_dia)
    assert np.all(field <= particle.max_dia)


def test_zero_spread_gaussian_is_monodisperse():
    """A zero standard deviation puts every particle at the mean diameter."""
    particle = _particles("gaussian", n=16, min_dia=281e-9, max_dia=281e-9,
                          mean_dia=281e-9, std_dia=0.0)
    np.testing.assert_allclose(particle.particle_field, 281e-9, rtol=0, atol=0)


def test_spawn_locations_lie_on_the_requested_line():
    """A vertical spawn line holds x and z fixed and spans y end to end."""
    particle = _particles(n=5)
    spawn = _spawn(particle)

    assert spawn.locations.shape == (5, 3)
    np.testing.assert_allclose(spawn.locations[:, 0], -1e-3)
    np.testing.assert_allclose(spawn.locations[:, 2], 5e-5)
    np.testing.assert_allclose(spawn.locations[0, 1], 2e-3)
    np.testing.assert_allclose(spawn.locations[-1, 1], 13e-3)
    assert np.all(np.diff(spawn.locations[:, 1]) > 0)


def test_serial_run_tracks_every_particle(oblique_case):
    """The serial driver writes one bounded, finite trajectory per particle.

    ``Streamlines._save_data`` writes each path to ``filepath`` and then clears
    the in-memory lists, so the saved arrays -- not the returned objects -- are
    where the trajectories live.
    """
    particle = _particles(n=N_PARTICLES)
    spawn = _spawn(particle)
    filepath = tempfile.mkdtemp() + "/"
    model = _model(oblique_case, particle, spawn, filepath)

    result = model.serial()
    assert len(result) == N_PARTICLES

    saved = sorted(f for f in os.listdir(filepath) if f.endswith(".npy"))
    assert saved == [f"ppath_{i}.npy" for i in range(N_PARTICLES)]

    x_min = oblique_case.grid.grd_min[0]
    x_max = oblique_case.grid.grd_max[0]
    for name in saved:
        trajectory = np.load(filepath + name)
        assert 1 <= trajectory.shape[0] <= model.max_steps
        assert trajectory.shape[1] == 15
        assert np.all(np.isfinite(trajectory))
        positions = trajectory[:, :3]
        assert np.all(positions >= x_min - 1e-9)
        assert np.all(positions <= x_max + 1e-9)


def test_multi_thread_matches_serial(oblique_case):
    """Threaded and serial ensembles produce identical trajectories.

    ``multi_thread`` runs the particles through a ThreadPool. Before the
    Newton-Raphson warm start was moved onto the Search instance, the threads
    shared a module global and the answers depended on interleaving.
    """
    def run(driver):
        particle = _particles(n=N_PARTICLES, std_dia=0.0, min_dia=281e-9,
                              max_dia=281e-9, mean_dia=281e-9)
        spawn = _spawn(particle)
        filepath = tempfile.mkdtemp() + "/"
        model = _model(oblique_case, particle, spawn, filepath)
        getattr(model, driver)()
        return [np.load(f"{filepath}ppath_{i}.npy") for i in range(N_PARTICLES)]

    serial = run("serial")
    threaded = run("multi_thread")

    assert len(threaded) == len(serial) == N_PARTICLES
    for a, b in zip(serial, threaded):
        assert a.shape == b.shape
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=0)


DISTRIBUTIONS = ["gaussian", "uniform", "skewnorm", "lognorm"]


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_same_seed_gives_an_identical_particle_field(distribution):
    """Two identically seeded Particles produce the same field, values and order.

    Before ``seed`` existed the gaussian and uniform draws came from a
    module-level generator with a hardcoded seed that advanced between calls,
    so two identically configured Particles disagreed; skewnorm and lognorm
    hardcoded ``random_state=7`` so they were frozen; and the final ordering
    came from the unseeded ``numpy.random`` global, so it was never
    reproducible at all.
    """
    first = _particles(distribution, n=128, seed=1234)
    second = _particles(distribution, n=128, seed=1234)

    np.testing.assert_array_equal(second.particle_field, first.particle_field)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_different_seeds_give_different_particle_fields(distribution):
    """A different seed is a genuinely different draw, not a reshuffle."""
    first = _particles(distribution, n=128, seed=1234)
    second = _particles(distribution, n=128, seed=4321)

    assert not np.array_equal(second.particle_field, first.particle_field)
    # Different values, not merely a different ordering of the same values.
    assert not np.array_equal(np.sort(second.particle_field),
                              np.sort(first.particle_field))


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_recomputing_a_seeded_distribution_is_stable(distribution):
    """Calling compute_distribution twice on one Particle repeats the field."""
    particle = _particles(distribution, n=64, seed=99)
    first = np.array(particle.particle_field, copy=True)
    particle.compute_distribution()

    np.testing.assert_array_equal(particle.particle_field, first)


def test_ordering_is_seeded_not_just_the_values():
    """The shuffle draws from the seeded generator, not the numpy global.

    ``compute_distribution`` used to finish with ``np.random.shuffle``, whose
    global state no seed argument could reach; seeding the legacy global and
    getting the same answer anyway is what proves the shuffle has moved onto
    the Particle's own generator.
    """
    reference = _particles("gaussian", n=256, seed=7).particle_field

    np.random.seed(0)
    first = _particles("gaussian", n=256, seed=7).particle_field
    np.random.seed(12345)
    second = _particles("gaussian", n=256, seed=7).particle_field

    np.testing.assert_array_equal(first, reference)
    np.testing.assert_array_equal(second, reference)
    # And the shuffle actually happened: the field is not left sorted.
    assert not np.array_equal(first, np.sort(first))


def test_unseeded_particles_differ():
    """The default is nondeterministic: no hidden hardcoded seed."""
    first = _particles("gaussian", n=256, seed=None)
    second = _particles("gaussian", n=256, seed=None)

    assert not np.array_equal(second.particle_field, first.particle_field)


def test_seeded_particle_field_reproduces_the_whole_ensemble(oblique_case):
    """The same seed drives identical trajectories for the whole ensemble."""
    def run(seed):
        particle = _particles(n=N_PARTICLES, seed=seed)
        spawn = _spawn(particle)
        filepath = tempfile.mkdtemp() + "/"
        _model(oblique_case, particle, spawn, filepath).serial()
        return [np.load(f"{filepath}ppath_{i}.npy") for i in range(N_PARTICLES)]

    for a, b in zip(run(2024), run(2024)):
        assert a.shape == b.shape
        np.testing.assert_array_equal(a, b)


def test_skewnorm_without_parameter_raises():
    """A skewnorm distribution needs its shape parameter and says so."""
    particle = Particle()
    particle.distribution = "skewnorm"
    particle.min_dia, particle.max_dia = 200e-9, 400e-9
    particle.mean_dia, particle.std_dia = 281e-9, 20e-9
    particle.n_concentration = 8
    particle.distribution_parameter = None
    with pytest.raises((ValueError, TypeError)):
        particle.compute_distribution()
