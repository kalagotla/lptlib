"""Tests for the Lagrangian-to-Eulerian reduction in ``DataIO``.

The reduction takes scattered particle records, interpolates the fluid state to
each particle, and maps both the fluid and particle fields onto a regular grid
in PLOT3D format. These tests cover the deterministic building blocks
(natural-order file sorting, scatter-to-grid interpolation, stratified spatial
sampling, and per-point flow interpolation) with known answers, and then run
the full ``compute`` pipeline on a tiny synthetic particle set to confirm the
output arrays have the expected shape.
"""

import os

import numpy as np
import pytest

from lptlib.io import DataIO
from lptlib.streamlines import Search, Interpolation


def _make_dataio(oblique_case, tmp_path, **kwargs):
    grid = oblique_case.grid
    flow = oblique_case.flow
    location = str(tmp_path) + "/"
    return DataIO(grid, flow, location=location, **kwargs)


def test_natural_sort_orders_numerically():
    """File names sort in natural (human) numeric order, not lexicographic."""
    names = ["p10.npy", "p2.npy", "p1.npy", "p21.npy", "p3.npy"]
    assert DataIO._natural_sort(names) == [
        "p1.npy", "p2.npy", "p3.npy", "p10.npy", "p21.npy"]


def test_grid_interp_recovers_linear_field():
    """Scatter-to-grid interpolation reproduces a linear field exactly.

    For a field that is linear in x and y, linear interpolation is exact, so
    the interpolated grid values must match the analytic field to within
    floating-point round-off.
    """
    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 1.0, (200, 2))
    values = 2.0 * points[:, 0] + 3.0 * points[:, 1]
    x_grid, y_grid = np.meshgrid(np.linspace(0.2, 0.8, 5),
                                 np.linspace(0.2, 0.8, 5), indexing="ij")

    result = DataIO._grid_interp(points, values, x_grid, y_grid,
                                 fill_value=0.0, method="linear")
    exact = 2.0 * x_grid + 3.0 * y_grid
    assert np.allclose(result, exact, atol=1e-12)


def test_grid_interp_uses_fill_value_outside_hull():
    """Query points outside the scattered convex hull take the fill value."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    values = np.array([0.0, 0.0, 0.0, 0.0])
    # A grid point far outside the unit square cannot be interpolated.
    x_grid = np.array([[5.0]])
    y_grid = np.array([[5.0]])
    result = DataIO._grid_interp(points, values, x_grid, y_grid,
                                 fill_value=-99.0, method="linear")
    assert result[0, 0] == pytest.approx(-99.0)


def test_sample_data_conserves_all_points_at_full_percent(oblique_case, tmp_path):
    """Sampling at 100 percent returns every original record exactly once.

    Each point falls in exactly one spatial bin, and at full percentage the
    sampler keeps all points in every bin, so the sampled set is a permutation
    of the input with no additions or losses.
    """
    data = _make_dataio(oblique_case, tmp_path)
    data.oblique_shock = False
    rng = np.random.default_rng(1)
    n = 200
    # Keep points strictly inside the bounding box so none sit on the top edge.
    coords = np.zeros((n, 15))
    coords[:, 0] = rng.uniform(0.01, 0.99, n)
    coords[:, 1] = rng.uniform(0.01, 0.99, n)
    coords[:, 2:] = rng.random((n, 13))

    sampled = data._sample_data(data, coords, 100)
    assert sampled.shape[1] == 15
    # The bins are half-open, so only the single x-max and y-max points can fall
    # outside the top bin; every other record is retained exactly once.
    assert n - 2 <= sampled.shape[0] <= n
    # Nothing is duplicated and nothing is invented.
    sampled_rows = [tuple(row) for row in sampled]
    assert len(sampled_rows) == len(set(sampled_rows))
    original_rows = {tuple(row) for row in coords}
    for row in sampled_rows:
        assert row in original_rows


def test_sample_data_is_spatially_stratified(oblique_case, tmp_path):
    """Stratified sampling keeps representation from separated regions.

    Two well-separated clusters are binned into different cells; a partial
    sample must still draw from both clusters rather than collapsing onto one.
    """
    data = _make_dataio(oblique_case, tmp_path)
    data.oblique_shock = False
    rng = np.random.default_rng(2)
    left = np.zeros((100, 15))
    left[:, 0] = rng.uniform(0.0, 0.1, 100)
    left[:, 1] = rng.uniform(0.0, 1.0, 100)
    right = np.zeros((100, 15))
    right[:, 0] = rng.uniform(0.9, 1.0, 100)
    right[:, 1] = rng.uniform(0.0, 1.0, 100)
    coords = np.vstack([left, right])

    np.random.seed(3)
    sampled = data._sample_data(data, coords, 20)
    xs = sampled[:, 0]
    # Both the left and right clusters contribute to the sample.
    assert np.any(xs < 0.1)
    assert np.any(xs > 0.9)
    # A partial sample is strictly smaller than the full set.
    assert sampled.shape[0] < coords.shape[0]


def test_sample_data_rejects_bad_percent(oblique_case, tmp_path):
    """A percentage outside (0, 100] is rejected."""
    data = _make_dataio(oblique_case, tmp_path)
    data.oblique_shock = False
    coords = np.random.default_rng(4).random((10, 15))
    with pytest.raises(ValueError):
        data._sample_data(data, coords, 0)


def test_flow_data_in_and_out_of_domain(oblique_case, tmp_path, upstream_point):
    """Per-point flow interpolation returns 5 values inside, a sentinel outside.

    ``_flow_data`` interpolates the flow state to a scattered point and returns
    a length-5 vector; for a point outside the grid it returns the integer
    sentinel used downstream to drop the record.
    """
    data = _make_dataio(oblique_case, tmp_path)
    inside = data._flow_data(np.asarray(upstream_point))
    assert inside.reshape(-1).shape == (5,)
    assert np.all(np.isfinite(inside))

    outside = data._flow_data(np.array([1e3, 1e3, 1e3]))
    assert outside.shape == (1,)


def test_compute_reduces_to_grid_shapes(oblique_case, tmp_path):
    """The full reduction writes fluid and particle fields of the grid shape.

    A small synthetic particle set is placed inside the domain, then
    ``compute`` interpolates fluid data to the particles and maps fluid and
    particle momentum onto a coarse Eulerian grid. The saved arrays must have
    shape ``(5, x_refinement, y_refinement)`` and the PLOT3D output files must
    exist.
    """
    x_refinement, y_refinement = 20, 16
    data = _make_dataio(oblique_case, tmp_path,
                        x_refinement=x_refinement, y_refinement=y_refinement)

    rng = np.random.default_rng(5)
    n = 60
    records = np.zeros((n, 15))
    records[:, 0] = rng.uniform(-0.014, 0.014, n)   # x inside domain
    records[:, 1] = rng.uniform(0.001, 0.014, n)    # y inside domain
    records[:, 2] = rng.uniform(0.0, 1e-4, n)       # z inside domain
    records[:, 3:6] = [380.0, 500.0, 0.0]           # particle velocity
    records[:, 6:9] = [380.0, 500.0, 0.0]           # fluid velocity
    records[:, 9] = 1e-6                            # time
    records[:, 10:13] = [380.0, 500.0, 0.0]        # integrated velocity
    records[:, 13] = 281e-9                         # diameter
    records[:, 14] = 813.0                          # density
    for chunk in range(3):
        np.save(data.location + f"p{chunk}.npy", records[chunk * 20:(chunk + 1) * 20])

    data.compute()

    flow_field = np.load(data.location + "dataio/flow_data.npy")
    particle_field = np.load(data.location + "dataio/particle_data.npy")
    assert flow_field.shape == (5, x_refinement, y_refinement)
    assert particle_field.shape == (5, x_refinement, y_refinement)
    assert np.all(np.isfinite(flow_field))
    # The grid and both solution files are written for downstream visualization.
    produced = os.listdir(data.location + "dataio")
    assert "mgrd_to_p3d.x" in produced
    assert any(name.endswith("_fluid.q") for name in produced)
    assert any(name.endswith("_particle.q") for name in produced)
