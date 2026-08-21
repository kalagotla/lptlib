# set package-level access
from importlib.metadata import PackageNotFoundError, version as _version

from .function import *
from .io import *
from .streamlines import *
from .test_cases import *

try:
    __version__ = _version("lptlib")
except PackageNotFoundError:  # package is not installed (e.g. run from a source tree)
    __version__ = "0.0.0+unknown"
