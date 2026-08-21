"""Package-level checks: version metadata and the optional-MPI import guard."""

import importlib

import pytest


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
