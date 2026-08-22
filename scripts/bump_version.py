#!/usr/bin/env python3
"""
Version bumping script for lptlib
Usage: python scripts/bump_version.py [major|minor|patch|alpha|beta|rc]

The version is carried in four files that have to agree: ``pyproject.toml``
(the source of truth that the build backend reads), ``CITATION.cff``,
``uv.lock``, and the release heading in ``CHANGELOG.md``. A fifth site,
``__version__`` in ``src/lptlib/__init__.py``, normally derives from installed
package metadata and needs no bumping, but it is rewritten if it is ever
changed to carry a literal release number. This script updates all of them in
one pass so they cannot drift apart between releases.
"""

import datetime
import re
import sys
from pathlib import Path

PYPROJECT_PATH = Path("pyproject.toml")
CITATION_PATH = Path("CITATION.cff")
UV_LOCK_PATH = Path("uv.lock")
CHANGELOG_PATH = Path("CHANGELOG.md")
INIT_PATH = Path("src") / "lptlib" / "__init__.py"

REPO_URL = "https://github.com/kalagotla/lptlib"

# A literal fallback version in the package __init__ looks like a plain PEP 440
# release, e.g. __version__ = "0.2.0". The sentinel the package currently ships,
# "0.0.0+unknown", is deliberately not a release number and is left alone.
_INIT_VERSION_RE = re.compile(
    r'(__version__\s*=\s*")(\d+\.\d+\.\d+(?:[a-z]+\d+)?)(")'
)

# The version in pyproject.toml sits in the [project] table. Anchor on the
# first `version = "..."` line so a version pin that later appears in another
# table is never rewritten by accident.
_PYPROJECT_VERSION_RE = re.compile(r'^version = "([^"]+)"$', re.MULTILINE)

# CITATION.cff is YAML; the release version is a top-level `version:` key.
_CITATION_VERSION_RE = re.compile(r'^version: *(\S+) *$', re.MULTILINE)

# uv.lock holds one [[package]] table per dependency. Only the table whose
# name is lptlib carries this project's version.
_UV_LOCK_VERSION_RE = re.compile(
    r'(\[\[package\]\]\nname = "lptlib"\nversion = ")([^"]+)(")'
)


def get_current_version():
    """Get current version from pyproject.toml"""
    if not PYPROJECT_PATH.exists():
        raise FileNotFoundError("pyproject.toml not found")

    content = PYPROJECT_PATH.read_text()

    match = _PYPROJECT_VERSION_RE.search(content)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")

    return match.group(1)


def bump_version(current_version, bump_type):
    """Bump version based on type"""
    # Parse version (e.g., "0.0.5a4" -> (0, 0, 5, 'a', 4))
    pattern = r'(\d+)\.(\d+)\.(\d+)([a-z]+)?(\d+)?'
    match = re.match(pattern, current_version)

    if not match:
        raise ValueError(f"Invalid version format: {current_version}")

    major, minor, patch, prerelease_type, prerelease_num = match.groups()
    major, minor, patch = int(major), int(minor), int(patch)

    if prerelease_num:
        prerelease_num = int(prerelease_num)

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
        prerelease_type = None
        prerelease_num = None
    elif bump_type == "minor":
        minor += 1
        patch = 0
        prerelease_type = None
        prerelease_num = None
    elif bump_type == "patch":
        patch += 1
        prerelease_type = None
        prerelease_num = None
    elif bump_type in ["alpha", "beta", "rc"]:
        # Convert full names to short forms for comparison
        current_prerelease_type = prerelease_type
        if current_prerelease_type == "alpha":
            current_prerelease_type = "a"
        elif current_prerelease_type == "beta":
            current_prerelease_type = "b"

        target_type = bump_type
        if target_type == "alpha":
            target_type = "a"
        elif target_type == "beta":
            target_type = "b"

        if current_prerelease_type == target_type:
            # Same prerelease type, increment number
            prerelease_num = (prerelease_num or 0) + 1
        else:
            # Different prerelease type, start from 0
            prerelease_num = 0

        prerelease_type = target_type
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")

    # Build new version string
    new_version = f"{major}.{minor}.{patch}"
    if prerelease_type and prerelease_num is not None:
        new_version += f"{prerelease_type}{prerelease_num}"

    return new_version


def update_pyproject_version(new_version):
    """Rewrite the [project] version in pyproject.toml.

    Returns True when the file was rewritten.
    """
    content = PYPROJECT_PATH.read_text()
    new_content, count = _PYPROJECT_VERSION_RE.subn(
        f'version = "{new_version}"', content, count=1
    )
    if not count:
        raise ValueError("Could not find version in pyproject.toml")
    PYPROJECT_PATH.write_text(new_content)
    return True


def update_citation_version(new_version):
    """Rewrite the top-level ``version:`` key in CITATION.cff.

    Returns True when the file was rewritten.
    """
    if not CITATION_PATH.exists():
        return False

    content = CITATION_PATH.read_text()
    new_content, count = _CITATION_VERSION_RE.subn(
        f'version: {new_version}', content, count=1
    )
    if not count:
        raise ValueError("Could not find a top-level 'version:' key in CITATION.cff")
    CITATION_PATH.write_text(new_content)
    return True


def update_uv_lock_version(new_version):
    """Rewrite the lptlib entry's version in uv.lock.

    The lockfile is generated, so this only keeps it consistent until the next
    ``uv lock``. Returns True when the file was rewritten.
    """
    if not UV_LOCK_PATH.exists():
        return False

    content = UV_LOCK_PATH.read_text()
    new_content, count = _UV_LOCK_VERSION_RE.subn(
        lambda m: f'{m.group(1)}{new_version}{m.group(3)}', content, count=1
    )
    if not count:
        raise ValueError("Could not find the lptlib package entry in uv.lock")
    UV_LOCK_PATH.write_text(new_content)
    return True


def update_changelog(old_version, new_version, release_date=None):
    """Turn the ``[Unreleased]`` section into the new release section.

    The entries collected under ``## [Unreleased]`` become
    ``## [new_version] - YYYY-MM-DD``, a fresh empty ``[Unreleased]`` heading is
    written above it, and the reference links at the foot of the file are
    rewritten so ``[Unreleased]`` compares against the new tag and the new
    version compares against the previous one.

    Returns True when the file was rewritten.
    """
    if not CHANGELOG_PATH.exists():
        return False

    content = CHANGELOG_PATH.read_text()
    if "## [Unreleased]" not in content:
        raise ValueError("Could not find an '## [Unreleased]' heading in CHANGELOG.md")

    if release_date is None:
        release_date = datetime.date.today().isoformat()

    content = content.replace(
        "## [Unreleased]",
        "## [Unreleased]\n"
        "\n"
        "Nothing yet.\n"
        "\n"
        f"## [{new_version}] - {release_date}",
        1,
    )

    # Reference links at the foot of the file.
    content = re.sub(
        r'^\[Unreleased\]: .*$',
        f'[Unreleased]: {REPO_URL}/compare/v{new_version}...HEAD\n'
        f'[{new_version}]: {REPO_URL}/compare/v{old_version}...v{new_version}',
        content,
        count=1,
        flags=re.MULTILINE,
    )

    CHANGELOG_PATH.write_text(content)
    return True


def update_init_version(new_version):
    """Update a literal fallback ``__version__`` in src/lptlib/__init__.py.

    The package normally derives ``__version__`` from installed package
    metadata via ``importlib.metadata``, with a non-release sentinel as the
    fallback for source-tree use. Nothing needs bumping in that case. If the
    file is ever changed to carry a literal release version instead, this
    keeps it in step with pyproject.toml.

    Returns True when the file was rewritten.
    """
    if not INIT_PATH.exists():
        return False

    content = INIT_PATH.read_text()

    new_content, count = _INIT_VERSION_RE.subn(
        lambda m: f'{m.group(1)}{new_version}{m.group(3)}', content
    )
    if not count:
        return False

    INIT_PATH.write_text(new_content)
    return True


def update_version_files(old_version, new_version):
    """Update every file that carries the version."""
    updated = []
    for path, updater in (
        (PYPROJECT_PATH, lambda: update_pyproject_version(new_version)),
        (CITATION_PATH, lambda: update_citation_version(new_version)),
        (UV_LOCK_PATH, lambda: update_uv_lock_version(new_version)),
        (CHANGELOG_PATH, lambda: update_changelog(old_version, new_version)),
        (INIT_PATH, lambda: update_init_version(new_version)),
    ):
        if updater():
            updated.append(str(path))

    print(f"Updated version to {new_version} in {', '.join(updated)}")
    if str(INIT_PATH) not in updated:
        print(
            f"  ({INIT_PATH} carries no literal release version; "
            "__version__ is read from package metadata.)"
        )
    print(
        "  (uv.lock is generated; rerun 'uv lock' if the dependency set also changed.)"
    )
    return updated


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py [major|minor|patch|alpha|beta|rc]")
        sys.exit(1)

    bump_type = sys.argv[1].lower()
    valid_types = ["major", "minor", "patch", "alpha", "beta", "rc"]

    if bump_type not in valid_types:
        print(f"Invalid bump type. Must be one of: {', '.join(valid_types)}")
        sys.exit(1)

    try:
        current_version = get_current_version()
        new_version = bump_version(current_version, bump_type)
        update_version_files(current_version, new_version)
        print(f"Version bumped from {current_version} to {new_version}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
