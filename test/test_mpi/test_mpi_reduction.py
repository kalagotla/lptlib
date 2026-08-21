"""Gated, small-memory MPI test for the parallel DataIO reduction.

The ``DataIO.compute`` pipeline is genuinely parallel: it scatters particle
records across ranks, gathers interpolated flow state, and splits the target
grid into per-rank chunks with ``Scatterv``/``Gatherv``/``bcast``. This test
drives that path on two ranks with a small synthetic grid and a handful of
particles, so it exercises the collective communication without the large
memory footprint of the full research cases.

It is skipped by default. Set ``LPTLIB_RUN_MPI=1`` in an environment with an
MPI launcher on ``PATH`` to run it. The test relaunches this module under
``mpiexec -np 2``; the child process builds the case, runs the reduction, and
on rank 0 asserts the reduced fields have the expected grid shape.
"""

import os
import shutil
import subprocess
import sys
import unittest


X_REFINEMENT = 20
Y_REFINEMENT = 16


def _run_reduction(location):
    """Build a tiny synthetic case and run the parallel reduction (all ranks).

    Only rank 0 writes the particle files; every rank then participates in the
    collective ``compute`` call. Rank 0 checks the reduced output shapes and
    raises on any mismatch so the process exits non-zero on failure.
    """
    import numpy as np
    from mpi4py import MPI
    from lptlib.io import DataIO
    from lptlib.test_cases import ObliqueShock, ObliqueShockData

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    shock = ObliqueShock()
    shock.mach = 2.0
    shock.deflection = 8.0
    shock.compute()
    osd = ObliqueShockData()
    osd.nx_max = 15e-3
    osd.ny_max = 15e-3
    osd.nz_max = 1e-4
    osd.inlet_temperature = 152.778
    osd.inlet_density = 1.2663
    osd.xpoints = 12
    osd.ypoints = 12
    osd.zpoints = 5
    osd.oblique_shock = shock
    osd.create_grid()
    osd.create_flow()

    if rank == 0:
        rng = np.random.default_rng(11)
        n = 80
        records = np.zeros((n, 15))
        records[:, 0] = rng.uniform(-0.014, 0.014, n)
        records[:, 1] = rng.uniform(0.001, 0.014, n)
        records[:, 2] = rng.uniform(0.0, 1e-4, n)
        records[:, 3:6] = [380.0, 500.0, 0.0]
        records[:, 6:9] = [380.0, 500.0, 0.0]
        records[:, 9] = 1e-6
        records[:, 10:13] = [380.0, 500.0, 0.0]
        records[:, 13] = 281e-9
        records[:, 14] = 813.0
        for chunk in range(4):
            np.save(location + f"p{chunk}.npy", records[chunk * 20:(chunk + 1) * 20])
    comm.Barrier()

    data = DataIO(osd.grid, osd.flow, location=location,
                  x_refinement=X_REFINEMENT, y_refinement=Y_REFINEMENT)
    data.compute()

    if rank == 0:
        flow_field = np.load(location + "dataio/flow_data.npy")
        particle_field = np.load(location + "dataio/particle_data.npy")
        assert flow_field.shape == (5, X_REFINEMENT, Y_REFINEMENT), flow_field.shape
        assert particle_field.shape == (5, X_REFINEMENT, Y_REFINEMENT), particle_field.shape
        assert np.all(np.isfinite(flow_field))
        print("MPI reduction produced correct output shapes on rank 0")


class TestMPIReduction(unittest.TestCase):
    def test_mpi_reduction_two_ranks(self):
        """Run the parallel reduction on two ranks and check it succeeds."""
        if not os.environ.get("LPTLIB_RUN_MPI"):
            self.skipTest("MPI reduction test disabled: set the environment variable "
                          "LPTLIB_RUN_MPI=1 to enable it")
        if shutil.which("mpiexec") is None:
            self.skipTest("MPI reduction test requires an mpiexec launcher on PATH "
                          "(LPTLIB_RUN_MPI=1 is set but mpiexec was not found)")

        import tempfile
        location = tempfile.mkdtemp() + "/"
        script = os.path.abspath(__file__)
        command = ["mpiexec", "-np", "2", sys.executable, script, "--mpi", location]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print("MPI stdout:\n", result.stdout)
            print("MPI stderr:\n", result.stderr)
        self.assertEqual(result.returncode, 0, "MPI reduction script failed")

        import numpy as np
        flow_field = np.load(location + "dataio/flow_data.npy")
        self.assertEqual(flow_field.shape, (5, X_REFINEMENT, Y_REFINEMENT))


if __name__ == "__main__":
    if "--mpi" in sys.argv:
        _run_reduction(sys.argv[sys.argv.index("--mpi") + 1])
    else:
        unittest.main()
