"""Helpers for locating test data robustly.

Test data lives under the repository ``data/`` directory. Most of the large
PLOT3D grid and solution files are not committed to the repository (they are
ignored via ``.gitignore``), so any test that needs them must skip cleanly when
the files are absent, for example on a fresh CI checkout. Paths are resolved
relative to this file rather than the current working directory so the tests
behave the same no matter where pytest is launched from.
"""

from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
DATA_DIR = REPO_ROOT / "data"


def data_path(*parts):
    """Return the absolute path to a file or directory under ``data/``."""
    return str(DATA_DIR.joinpath(*parts))


def require_data(*parts):
    """Return an absolute data path, or skip the test if it is not present.

    Use this for any test that depends on a data file which is not tracked in
    the repository. On a clean checkout the file is missing and the test is
    skipped with a clear message instead of failing.
    """
    target = DATA_DIR.joinpath(*parts)
    if not target.exists():
        rel = target.relative_to(REPO_ROOT)
        pytest.skip(f"data file not available in checkout: {rel}")
    return str(target)
