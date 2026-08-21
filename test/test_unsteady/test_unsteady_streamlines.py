"""Unsteady particle tracking on the committed cylinder case.

The seven-snapshot cylinder solution under ``cylinder_data/`` is tracked in the
repository, so unlike most of the historical tests this one really runs. It
used to end in ``plt.show()`` and assert nothing; it now asserts on the
trajectory instead.
"""

import unittest
from pathlib import Path

import numpy as np

from lptlib import GridIO, FlowIO
from lptlib import Streamlines

DATA = str(Path(__file__).resolve().parent / 'cylinder_data') + '/'
START_POINT = [1.0, 2.5, 0.5]


class TestUnsteadyStreamlines(unittest.TestCase):
    def setUp(self):
        self.grid = GridIO(DATA + "cylinder.sp.x")
        self.grid.read_grid(data_type="f4")
        self.grid.compute_metrics()
        self.flow = FlowIO(DATA + "sol-0000010.q")
        self.flow.read_unsteady_flow(data_type="f4")

    def _track(self, drag_model='loth', diameter=281e-9, density=813):
        sl = Streamlines(point=list(START_POINT))
        sl.diameter = diameter
        sl.density = density
        sl.time_step = 0.1
        sl.drag_model = drag_model
        sl.compute(grid=self.grid, flow=self.flow, method='unsteady-ppath')
        return sl

    def test_unsteady_flow_snapshots_are_loaded(self):
        """All seven committed snapshots are read with a consistent shape."""
        self.assertEqual(len(self.flow.unsteady_flow), 7)
        expected = (self.grid.ni[0], self.grid.nj[0], self.grid.nk[0], 5, 1)
        for snapshot in self.flow.unsteady_flow:
            self.assertEqual(snapshot.q.shape, expected)
            self.assertTrue(np.all(np.isfinite(snapshot.q)))

    def test_unsteady_streamlines(self):
        """The particle path is finite, in-domain, and advances downstream."""
        sl = self._track()

        path = np.asarray(sl.streamline, dtype=float)
        particle = np.asarray(sl.svelocity, dtype=float)
        fluid = np.asarray(sl.fvelocity, dtype=float)
        steps = np.asarray(sl.time, dtype=float).reshape(-1)

        # Shapes line up: one position, particle velocity, fluid velocity and
        # time-step value per recorded step.
        self.assertGreaterEqual(path.shape[0], 2)
        self.assertEqual(path.shape[1], 3)
        self.assertEqual(particle.shape, path.shape)
        self.assertEqual(fluid.shape, path.shape)
        self.assertEqual(steps.shape[0], path.shape[0])

        self.assertTrue(np.all(np.isfinite(path)))
        self.assertTrue(np.all(np.isfinite(particle)))
        self.assertTrue(np.all(np.isfinite(fluid)))
        self.assertTrue(np.all(steps > 0))

        # Starts where asked.
        np.testing.assert_allclose(path[0], START_POINT, rtol=0, atol=1e-12)

        # The free stream runs in +x, so the particle moves downstream and
        # barely deviates in y or z over this short track.
        self.assertGreater(path[-1, 0], path[0, 0])
        self.assertLess(abs(path[-1, 1] - path[0, 1]), 0.1)
        self.assertLess(abs(path[-1, 2] - path[0, 2]), 0.1)

        # Every point stays inside the grid bounds.
        self.assertTrue(np.all(path >= self.grid.grd_min[0] - 1e-9))
        self.assertTrue(np.all(path <= self.grid.grd_max[0] + 1e-9))

        # A sub-micron tracer follows the flow closely: the slip stays a small
        # fraction of the local fluid speed.
        slip = np.linalg.norm(particle - fluid, axis=1)
        speed = np.linalg.norm(fluid, axis=1)
        self.assertTrue(np.all(slip <= 0.1 * speed + 1e-12))

    def test_zero_drag_tracer_follows_the_fluid(self):
        """With no drag the particle velocity is the fluid velocity.

        The two are not bit-identical: the recorded fluid velocity comes from a
        slightly different stage of the RK4 update than the assigned particle
        velocity, which leaves a relative difference of order 1e-5 on the small
        transverse components.
        """
        sl = self._track(drag_model='zero-drag')
        particle = np.asarray(sl.svelocity, dtype=float)
        fluid = np.asarray(sl.fvelocity, dtype=float)
        np.testing.assert_allclose(particle, fluid, rtol=1e-4, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
