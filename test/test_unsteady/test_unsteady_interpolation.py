"""Unsteady interpolation on the committed cylinder case.

The seven-snapshot cylinder solution under ``cylinder_data/`` is tracked in the
repository, so this test really runs. It used to loop over the snapshots
without asserting anything; it now checks the interpolated state at each one.
"""

import unittest
from pathlib import Path

import numpy as np

from lptlib import GridIO, FlowIO
from lptlib import Search
from lptlib import Interpolation

DATA = str(Path(__file__).resolve().parent / 'cylinder_data') + '/'
PROBE = [1.0, 2.5, 0.5]


class TestUnsteadyInterpolation(unittest.TestCase):
    def setUp(self):
        self.grid = GridIO(DATA + "cylinder.sp.x")
        self.grid.read_grid(data_type="f4")
        self.grid.compute_metrics()
        self.flow = FlowIO(DATA + "sol-0000010.q")
        self.flow.read_unsteady_flow(data_type="f4")
        self.idx = Search(self.grid, PROBE)
        self.idx.method = "p-space"
        self.idx.compute()

    def test_probe_point_is_located_in_the_grid(self):
        """The probe resolves to a single hexahedral cell in block 0."""
        self.assertEqual(self.idx.block, 0)
        self.assertEqual(self.idx.cell.shape, (8, 3))

    def test_unsteady_interpolation(self):
        """Every snapshot interpolates to a physically valid state.

        The interpolated conservative variables must be finite, the density
        positive, and the recovered state must lie inside the range spanned by
        the eight surrounding cell nodes -- a tri-linear interpolant cannot
        overshoot its own stencil.
        """
        cell = self.idx.cell
        block = self.idx.block
        results = []

        for i, snapshot in enumerate(self.flow.unsteady_flow):
            interp = Interpolation(snapshot, self.idx)
            interp.flow_old = None if i == 0 else self.flow.unsteady_flow[i - 1]
            interp.time.append(1e-6)
            interp.compute(method='unsteady-rbf-p-space')

            q = np.asarray(interp.q, dtype=float).reshape(-1)
            self.assertEqual(q.shape, (5,))
            self.assertTrue(np.all(np.isfinite(q)))
            self.assertGreater(q[0], 0.0)   # density
            self.assertGreater(q[4], 0.0)   # total energy per unit volume

            node_q = snapshot.q[cell[:, 0], cell[:, 1], cell[:, 2], :, block]
            lo, hi = node_q.min(axis=0), node_q.max(axis=0)
            span = np.maximum(hi - lo, 1e-12)
            # Allow a small relative slack for the RBF blend across snapshots.
            self.assertTrue(np.all(q >= lo - 0.05 * span))
            self.assertTrue(np.all(q <= hi + 0.05 * span))

            results.append(q)

        self.assertEqual(len(results), len(self.flow.unsteady_flow))
        # The wake is unsteady, so the state is not identical at every snapshot.
        stacked = np.vstack(results)
        self.assertGreater(np.ptp(stacked[:, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
