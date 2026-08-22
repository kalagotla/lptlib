"""Package-level checks: version metadata and the optional-MPI import guard."""

import importlib
import subprocess
import sys

import pytest


def _mpi4py_available():
    """True when ``mpi4py.MPI`` can be imported on this machine."""
    from lptlib.io import dataio

    return dataio.MPI is not None


def test_version_is_non_empty_string():
    import lptlib

    assert isinstance(lptlib.__version__, str)
    assert lptlib.__version__ != ""


def test_import_does_not_require_mpi():
    """``import lptlib`` must not blow up when mpi4py cannot be imported.

    The modules that use MPI import it defensively, so ``MPI`` is either the
    real module or ``None`` -- never an import-time failure.
    """
    from lptlib.io import dataio
    from lptlib.streamlines import stochastic_model

    for module in (dataio, stochastic_model):
        assert hasattr(module, "MPI")
        assert hasattr(module, "_require_mpi")


@pytest.mark.parametrize("module_name", ["lptlib.io.dataio",
                                         "lptlib.streamlines.stochastic_model"])
def test_require_mpi_raises_actionable_error(module_name, monkeypatch):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "MPI", None)
    with pytest.raises(ImportError) as excinfo:
        module._require_mpi()
    message = str(excinfo.value)
    assert "mpi4py" in message
    assert "MPI" in message


def test_dataio_compute_raises_without_mpi(monkeypatch):
    from lptlib.io import dataio

    monkeypatch.setattr(dataio, "MPI", None)
    obj = dataio.DataIO(grid=None, flow=None)
    with pytest.raises(ImportError, match="mpi4py"):
        obj.compute()


def test_stochastic_model_mpi_run_raises_without_mpi(monkeypatch):
    from lptlib.streamlines import stochastic_model

    monkeypatch.setattr(stochastic_model, "MPI", None)
    obj = stochastic_model.StochasticModel(particles=None, spawn_locations=None)
    with pytest.raises(ImportError, match="mpi4py"):
        obj.mpi_run()


def test_require_mpi_returns_module_when_available():
    """When MPI is importable the helper returns it unchanged."""
    from lptlib.io import dataio

    if dataio.MPI is None:
        pytest.skip("mpi4py/MPI runtime is not available on this machine")
    assert dataio._require_mpi() is dataio.MPI


def test_import_does_not_initialise_mpi():
    """``import lptlib`` must not call ``MPI_Init``.

    mpi4py initialises MPI as a side effect of ``from mpi4py import MPI``
    unless ``mpi4py.rc.initialize`` is turned off first, which is what the
    MPI-using modules do. If that regressed, any process that merely imports
    lptlib would become an MPI singleton, and a nested ``mpiexec`` launched
    from it (as the gated MPI tests do) would fail immediately.

    Runs in a fresh subprocess: MPI state is per-process and another test in
    this session may legitimately have initialised it already.
    """
    if not _mpi4py_available():
        pytest.skip("mpi4py/MPI runtime is not available on this machine")

    script = (
        "import lptlib\n"
        "from mpi4py import MPI\n"
        "print(MPI.Is_initialized())\n"
    )
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        "importing lptlib initialised MPI; check that mpi4py.rc.initialize is "
        "set to False before 'from mpi4py import MPI' in lptlib.io.dataio and "
        "lptlib.streamlines.stochastic_model"
    )


def test_init_mpi_initialises_and_returns_module():
    """``_init_mpi`` initialises MPI on demand and hands back the module.

    Runs in a subprocess so the pytest process itself is not turned into an
    MPI singleton, which would break the gated nested-``mpiexec`` tests.
    """
    if not _mpi4py_available():
        pytest.skip("mpi4py/MPI runtime is not available on this machine")

    script = (
        "from lptlib.io import dataio\n"
        "assert not dataio.MPI.Is_initialized()\n"
        "assert dataio._init_mpi() is dataio.MPI\n"
        "print(dataio.MPI.Is_initialized())\n"
    )
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


@pytest.mark.parametrize("module_name", ["lptlib.io.dataio",
                                         "lptlib.streamlines.stochastic_model"])
def test_init_mpi_raises_actionable_error(module_name, monkeypatch):
    """``_init_mpi`` keeps ``_require_mpi``'s actionable error when MPI is absent."""
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "MPI", None)
    with pytest.raises(ImportError, match="mpi4py"):
        module._init_mpi()
