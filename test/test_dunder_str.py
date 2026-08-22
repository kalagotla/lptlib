"""``__str__`` smoke tests for the public classes.

``Search.__str__`` and ``Interpolation.__str__`` used to raise ``TypeError``
because they concatenated a tuple (``grid.grd.shape``) and a NumPy array
(``ppoint``) onto a string. Both are public dunders on the most-used classes in
the library, so a bare ``print(obj)`` blew up. These tests pin all five
``__str__`` implementations in ``src`` to "returns a non-empty string".
"""

import numpy as np
import pytest

from lptlib import Integration, Interpolation, Search


@pytest.fixture()
def located_search(synthetic_grid, upstream_point):
    """A ``Search`` that has already run ``compute`` on a point in the domain."""
    idx = Search(synthetic_grid, upstream_point)
    idx.compute(method="p-space")
    return idx


def test_grid_str(synthetic_grid):
    doc = str(synthetic_grid)
    assert isinstance(doc, str)
    assert "grd" in doc


def test_flow_str(synthetic_flow):
    doc = str(synthetic_flow)
    assert isinstance(doc, str)
    assert "q attribute" in doc


def test_search_str_does_not_raise(synthetic_grid, upstream_point):
    """``str(Search(...))`` must not raise, and must mention the grid shape."""
    idx = Search(synthetic_grid, upstream_point)
    doc = str(idx)
    assert isinstance(doc, str)
    assert str(synthetic_grid.grd.shape) in doc
    assert "compute" in doc


def test_search_str_works_before_and_after_compute(synthetic_grid, upstream_point):
    """The dunder must be safe in both lifecycle states of the object."""
    idx = Search(synthetic_grid, upstream_point)
    before = str(idx)
    idx.compute(method="p-space")
    after = str(idx)
    assert before == after


def test_search_str_with_list_point(synthetic_grid, upstream_point):
    """A plain Python list point formats too -- ppoint is not always an array."""
    idx = Search(synthetic_grid, list(np.asarray(upstream_point)))
    assert isinstance(str(idx), str)


def test_interpolation_str_does_not_raise(synthetic_flow, located_search):
    """``str(Interpolation(...))`` must not raise, and must name the flow file."""
    interp = Interpolation(synthetic_flow, located_search)
    doc = str(interp)
    assert isinstance(doc, str)
    assert str(synthetic_flow.filename) in doc


def test_integration_str_does_not_raise(synthetic_flow, located_search):
    interp = Interpolation(synthetic_flow, located_search)
    interp.compute()
    doc = str(Integration(interp))
    assert isinstance(doc, str)
    assert "integrates" in doc
