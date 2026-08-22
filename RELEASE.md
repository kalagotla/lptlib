# Release Process

This document describes how to release new versions of lptlib to PyPI.

## Automated Release Options

### Option 1: GitHub Actions (Recommended)

The easiest way to release is using GitHub Actions:

1. **Set up PyPI Trusted Publishing** (once):
   - Go to [https://pypi.org/manage/project/lptlib/settings/publishing/](https://pypi.org/manage/project/lptlib/settings/publishing/)
   - Add a GitHub publisher with owner `kalagotla`, repository `lptlib`,
     workflow `publish.yml`, and environment `pypi`
   - No API token is needed. `.github/workflows/publish.yml` authenticates with
     a short-lived OIDC token. The header comment in that file explains how to
     revert to the old `PYPI_API_TOKEN` flow if the publisher is not set up yet

2. **Create a release**:
   - Go to your GitHub repository
   - Click "Releases" → "Create a new release"
   - Create a new tag matching the version in `pyproject.toml` (e.g., `v0.2.0`)
   - Add release notes
   - Click "Publish release"

3. **Automatic upload**:
   - GitHub Actions will automatically build and upload to PyPI
   - Check the Actions tab to monitor progress

### Option 2: Local Scripts

For local releases, use the provided scripts:

#### Quick Release (TestPyPI)
```bash
# Bump version and upload to TestPyPI
python scripts/release.py alpha --test
```

#### Production Release
```bash
# Bump version and upload to production PyPI
python scripts/release.py patch --prod
```

#### Manual Steps
```bash
# 1. Bump version
python scripts/bump_version.py patch

# 2. Build and upload (requires: pip install build twine)
python -m build
twine check dist/*
twine upload dist/*
```

## Version Bumping

Use the version bumping script to automatically update version numbers:

```bash
# Major version (1.0.0 -> 2.0.0)
python scripts/bump_version.py major

# Minor version (1.0.0 -> 1.1.0)
python scripts/bump_version.py minor

# Patch version (1.0.0 -> 1.0.1)
python scripts/bump_version.py patch

# Alpha release (1.0.0 -> 1.0.0a0, then 1.0.0a0 -> 1.0.0a1)
python scripts/bump_version.py alpha

# Beta release (1.0.0 -> 1.0.0b0, then 1.0.0b0 -> 1.0.0b1)
python scripts/bump_version.py beta

# Release candidate (1.0.0 -> 1.0.0rc0, then 1.0.0rc0 -> 1.0.0rc1)
python scripts/bump_version.py rc
```

A prerelease bump attaches the prerelease segment to the current release number
rather than opening the next one, and it restarts the counter at 0 whenever the
prerelease kind changes. Bump `patch` first if the prerelease is meant to lead
to a new release number.

The script rewrites every file that carries the version, so the four version
sites cannot drift apart between releases:

| File | What is updated |
|---|---|
| `pyproject.toml` | the `version` field in `[project]`, which the build backend reads |
| `CITATION.cff` | the top-level `version:` key |
| `uv.lock` | the `version` in the `lptlib` package entry |
| `CHANGELOG.md` | the `[Unreleased]` entries become a dated section for the new version, a fresh empty `[Unreleased]` heading is written above it, and the comparison links at the foot of the file are rewritten |

`__version__` in `src/lptlib/__init__.py` is not a fifth site to maintain by
hand. It is read from the installed package metadata with
`importlib.metadata.version`, so it follows `pyproject.toml` automatically. The
script rewrites it only if that file is ever changed to carry a literal release
number instead.

`uv.lock` is a generated file. The script keeps its version consistent, but run
`uv lock` as well if the dependency set changed in the same release.

## Release Checklist

Before releasing:

- [ ] Write the release's entries under `[Unreleased]` in `CHANGELOG.md`; the bump script moves them under the new version heading and adds the date and the comparison link
- [ ] Run tests: `pytest test`
- [ ] Run the linter: `ruff check src test`
- [ ] Update documentation if needed
- [ ] Choose appropriate version bump type and run the bump script
- [ ] Review the diff across all four version sites, then commit all changes

## Testing Releases

Always test on TestPyPI first:

```bash
# Upload to TestPyPI
python scripts/release.py alpha --test

# Test installation
pip install --index-url https://test.pypi.org/simple/ lptlib

# If everything works, upload to production
python scripts/release.py alpha --prod
```

## Troubleshooting

### Common Issues

1. **"Package already exists"**: Increment the version number
2. **"Invalid metadata"**: Check pyproject.toml format
3. **"Trusted publishing exchange failure" or "invalid-publisher"**: The PyPI publisher is not configured yet. Add it in the project's publishing settings, or fall back to the token flow documented at the top of `.github/workflows/publish.yml`
4. **"Authentication failed"** on a manual upload: Verify your PyPI API token

### Manual Cleanup

If builds get corrupted:
```bash
rm -rf dist/ build/ src/*.egg-info/
python -m build
```

## Environment Variables

Only needed for **manual** uploads from a laptop. The GitHub Actions workflow
uses Trusted Publishing and needs no secrets.

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=your_pypi_api_token
```

## File Structure

```
lptlib/
├── .github/workflows/publish.yml  # GitHub Actions workflow
├── scripts/
│   ├── bump_version.py            # Version bumping script
│   └── release.py                 # Complete release script
├── pyproject.toml                 # Package configuration and the source of truth for the version
├── CITATION.cff                   # Citation metadata, carries the version
├── uv.lock                        # Generated lockfile, carries the version
├── CHANGELOG.md                   # Release notes, carries the version headings
└── RELEASE.md                     # This file
```

There is no `setup.py`. All package metadata lives in `pyproject.toml`, which is
also the source of truth for the version, and `scripts/bump_version.py` is the
supported way to change it. Editing any version site by hand is what lets the
four of them drift apart.
