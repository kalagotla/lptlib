"""pytest configuration for the lptlib test suite.

Adds the ``test/`` directory to ``sys.path`` so the shared ``testdata`` helper
module can be imported from test modules in this directory and its
subdirectories.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
