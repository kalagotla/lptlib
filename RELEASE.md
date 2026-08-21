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
   - Create a new tag (e.g., `v0.0.6`)
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

# Alpha release (1.0.0 -> 1.0.1a1)
python scripts/bump_version.py alpha

# Beta release (1.0.0 -> 1.0.1b1)
python scripts/bump_version.py beta

# Release candidate (1.0.0 -> 1.0.1rc1)
python scripts/bump_version.py rc
```

## Release Checklist

Before releasing:

- [ ] Move the `[Unreleased]` entries in `CHANGELOG.md` under a heading for the new version, with the date and the comparison link
- [ ] Run tests: `pytest`
- [ ] Update documentation if needed
- [ ] Commit all changes
- [ ] Choose appropriate version bump type

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
├── pyproject.toml                 # Package configuration and version
├── CHANGELOG.md                   # Release notes
└── RELEASE.md                     # This file
```

There is no `setup.py`. All package metadata, including the version, lives in
`pyproject.toml`, and `scripts/bump_version.py` is the supported way to change
the version.
