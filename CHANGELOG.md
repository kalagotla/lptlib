# Changelog

All notable changes to lptlib are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries for the released versions were reconstructed from the git history
between tags, so they summarize what changed rather than reproducing a
contemporaneous release note.

## [Unreleased]


### Fixed

- `ObliqueShockData.create_flow` no longer consumes the `ObliqueShock` it is
  given. It selected the weak or strong branch by overwriting the two-element
  ratio arrays in place, so the shock object was single-use and a shock shared
  between two instances silently gave the second one the first one's branch.
  Output is bit-identical for a single call.
- `test_dataio_reduction.py::test_compute_reduces_to_grid_shapes` now skips
  rather than errors when `mpi4py` is absent. `DataIO.compute` runs its whole
  body through a communicator, and `mpi4py` lives in the optional `mpi` extra,
  so a checkout installed with `[test]` alone reported a failure on healthy code.

### Removed

- `DataIO.__init__`'s `read_file` parameter, which was accepted and never
  assigned. Passing it now raises `TypeError` rather than being ignored.
Nothing yet.

## [0.2.0] - 2026-08-22

### Added

- A PLOT3D I/O benchmark under `benchmarks/` comparing the vectorized strided
  reader against a naive Python reader and a compiled Fortran reader, with the
  measured results recorded in the repository.
- A synthetic test-fixture layer (`test/conftest.py`, `test/synthetic.py`) so the
  search, interpolation, integration, and stochastic-model tests run without any
  external data files, plus new tests for the drag models, PLOT3D round trips,
  the `DataIO` Lagrangian-to-Eulerian reduction, the plotting helpers, and a
  gated two-rank MPI reduction test.
- An optional `store_type` argument to `GridIO.read_grid` that keeps the
  reconstructed grid in single precision, roughly halving the read time on large
  grids.
- Coverage reporting in CI with a minimum coverage floor.
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `CITATION.cff`, `.mailmap`,
  and this changelog.
- Community support guidance in `README.md` and `CONTRIBUTING.md`.
- `mpi`, `examples`, and `test` dependency extras.
- `macos-latest` and Python 3.13 in the CI matrix, a job that verifies the
  package imports without the `mpi` extra, and a Ruff lint step.

### Changed

- `mpi4py` is now an optional dependency behind the `mpi` extra, and the MPI
  imports in `DataIO` and `StochasticModel` are deferred, so `import lptlib` and
  every non-MPI code path work without a system MPI runtime.
- The README quickstart was rewritten around the synthetic `ObliqueShock` test
  case so every example runs on a fresh clone with no external data.
- The license declaration migrated to the PEP 639 form (`license = "MIT"` with
  `license-files`), replacing the deprecated MIT license classifier.
- Package discovery is declared explicitly in `pyproject.toml` instead of
  relying on setuptools auto-discovery.
- Releases are built with `python -m build` instead of the deprecated
  `python setup.py sdist bdist_wheel`.
- `.gitignore` no longer blanket-ignores `*.png`, `*.csv`, `*.dat`, `*.txt`, and
  similar extensions across the whole repository; those patterns are scoped to
  run-output directories.
- The Ruff rule set is pinned in `pyproject.toml` (`select = ["E4", "E7", "E9",
  "F"]`) instead of inheriting Ruff's default selection, which widens between
  releases, and the CI lint job is now blocking rather than advisory.
- `scripts/bump_version.py` now updates every file that carries the version:
  `pyproject.toml`, `CITATION.cff`, `uv.lock`, and the release heading and
  comparison links in `CHANGELOG.md`.
- The top level of the package re-exports its public classes explicitly with an
  `__all__` instead of using star imports, so `lptlib.streamlines` resolves to
  the subpackage rather than to the module of the same name.

### Removed

- The `gpu` optional extra and the PyTorch CUDA index it resolved from. Neither
  `torch` nor `gpytorch` is imported anywhere in the package, so the extra pulled
  a multi-gigabyte CUDA wheel for code that never used it, and its custom index
  made `uv lock` fail on any machine without access to `download.pytorch.org`.
- `setup.py`. Package metadata comes from `pyproject.toml`, and the file had
  drifted out of sync with it.
- The three private git submodules under `external/` and the `[tool.uv.workspace]`
  entry that pointed at one of them. The repositories are not public, so
  `git clone --recursive` and `uv sync` failed for anyone outside the group.
- `scikit-learn` and `h5py` from the runtime dependencies. Neither is imported
  anywhere in `src/`.
- `seaborn` from the runtime dependencies; it is used only by `main.py` and moved
  to the `examples` extra.
- `JOSS_SUBMISSION_CHECKLIST.md`, an internal working document.

## [0.1.0] - 2026-08-19

First release prepared for public review.

### Added

- A GitHub Actions CI workflow that runs the test suite on Python 3.10, 3.11,
  and 3.12.
- `CONTRIBUTING.md` and a `docs/index.md` overview covering the concepts, the
  package layout, the public API, and the drag models.
- A class docstring for `Integration`.

### Changed

- `README.md` was rewritten as a public front page with badges, a feature
  overview, quickstart snippets, and citation information.
- The package license metadata was corrected to MIT so that it matches the
  `LICENSE` file.
- Bumped Pillow from 12.1.1 to 12.2.0.

## [0.0.6] - 2026-05-19

### Added

- Reading of Fortran-record PLOT3D grid and flow planes, and conversion of
  two-dimensional data to three dimensions.
- Adaptive particle-path integration in computational space, with debug and
  live-plot visualization of trajectories and domain bounds.
- Support for external case submodules and static inlet-isolator case values.
- Computational-grid visualization.

### Changed

- The PLOT3D reader was reworked for speed and correctness.
- Adaptive particle-path integration was made robust for expansion fans as well
  as shocks, and no longer oscillates; integration now terminates when a
  trajectory exceeds a step limit inside a recirculation zone.
- The point search was made more robust.
- The code structure was refactored around the tracking pipeline.

### Fixed

- A viscosity bug in `Variables`.
- The Keyes viscosity formula.
- The Loth drag correlation, which did not work with current SciPy.

## [0.0.5a6] - 2025-10-08

Earlier alpha releases predate this changelog. See the git history for details.

[Unreleased]: https://github.com/kalagotla/lptlib/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/kalagotla/lptlib/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kalagotla/lptlib/compare/v0.0.6...v0.1.0
[0.0.6]: https://github.com/kalagotla/lptlib/compare/v0.0.5a6...v0.0.6
[0.0.5a6]: https://github.com/kalagotla/lptlib/releases/tag/v0.0.5a6
