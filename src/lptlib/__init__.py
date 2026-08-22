# set package-level access
from importlib.metadata import PackageNotFoundError, version as _version

# Re-exported explicitly rather than with `import *`: the star imports used to
# pull the submodule names in alongside the classes, and `lptlib.streamlines`
# then resolved to the streamlines *module* instead of the subpackage the
# README documents.
from .function import Plots, Timer, Variables
from .io import DataIO, FlowIO, GridIO
from .streamlines import (Integration, Interpolation, Particle, Search,
                          SpawnLocations, StochasticModel, Streamlines)
from .test_cases import ObliqueShock, ObliqueShockAlignedData, ObliqueShockData

__all__ = [
    "DataIO",
    "FlowIO",
    "GridIO",
    "Integration",
    "Interpolation",
    "ObliqueShock",
    "ObliqueShockAlignedData",
    "ObliqueShockData",
    "Particle",
    "Plots",
    "Search",
    "SpawnLocations",
    "StochasticModel",
    "Streamlines",
    "Timer",
    "Variables",
    "__version__",
]

try:
    __version__ = _version("lptlib")
except PackageNotFoundError:  # package is not installed (e.g. run from a source tree)
    __version__ = "0.0.0+unknown"
