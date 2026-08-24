# Contributing to lptlib

Thanks for your interest in improving lptlib. Bug reports, feature requests, documentation fixes, and pull requests are all welcome.

## Ways to contribute

You can open an issue to report a bug or request a feature, improve the documentation, add a test case, or submit a pull request with a code change. If you are planning a large change, please open an issue first so the design can be discussed before you invest time in the implementation.

## Reporting bugs

Good bug reports make fixes much faster. When you open an issue at https://github.com/kalagotla/lptlib/issues, please include the lptlib version and Python version, your operating system, a minimal example that reproduces the problem, the full traceback, and what you expected to happen instead. A small PLOT3D grid or flow snippet that triggers the issue is especially helpful.

## Getting help

If you are stuck, please ask rather than guess.

- **Questions about how to use lptlib.** Open an issue at https://github.com/kalagotla/lptlib/issues and apply the `question` label, or start a thread in [GitHub Discussions](https://github.com/kalagotla/lptlib/discussions). Include what you are trying to do, the snippet you ran, and the full output.
- **Questions about a change you are working on.** Open an issue describing the change before you invest time in it, and a maintainer will comment on the approach.
- **Anything that does not belong in public.** Email the maintainer at dilipkalagotla@gmail.com.

Public issues are preferred, because the answer then helps the next person with the same question.

## Development setup

Clone the repository and install the package in editable mode together with the test extra.

```bash
git clone https://github.com/kalagotla/lptlib.git
cd lptlib
python -m pip install -e ".[test,mpi]"
```

The `test` extra pulls in `pytest`, `parameterized`, and `pytest-cov`. Installing bare `pytest` is not enough: several test modules import `parameterized`, and collection fails without it.

The MPI-parallel execution backends are optional and live behind the separate `mpi` extra. They require a system MPI implementation. On Debian or Ubuntu you can install one with `sudo apt-get install libopenmpi-dev openmpi-bin`, and on macOS with `brew install open-mpi`. Everything except those backends works without MPI, and a change must not make `import lptlib` depend on it.

Install `mpi` alongside `test` all the same, because one test drives `DataIO.compute()`, which runs over MPI even on a single rank, and it errors rather than skipping when `mpi4py` is absent. With `pip install -e ".[test]"` alone, `pytest test` reports one failure in `test/test_dataio_reduction.py` on an otherwise healthy checkout. The continuous integration workflow installs `".[test,mpi]"` for the same reason.

## Running the tests

The test suite lives under `test/` and is run with pytest from the repository root.

```bash
pytest test -v
```

The suite covers search, interpolation for steady and unsteady flow, integration, the drag models, streamlines, the stochastic model, the PLOT3D readers and round trips, the `DataIO` Lagrangian-to-Eulerian reduction, the `Plots` helpers, and the MPI helpers.

Two groups of tests do not run by default. Tests that need the large PLOT3D datasets from the original research cases skip with a message naming the missing path, because those datasets are unpublished and are not part of this repository. Tests that launch MPI processes are opt-in and are enabled with an environment variable.

```bash
LPTLIB_RUN_MPI=1 pytest test -v
```

Those tests launch `mpiexec -np 2`, so they also need `mpiexec` on `PATH`, and they take a few minutes. Open MPI refuses to launch as root, so in a container running as root add `OMPI_ALLOW_RUN_AS_ROOT=1` and `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` to the environment as well. GitHub-hosted runners execute as an unprivileged user and need neither variable.

Matplotlib is forced to the headless `Agg` backend in `test/conftest.py`, so you do not need to set `MPLBACKEND` yourself.

Please make sure the full suite passes before you open a pull request. If you add a feature or fix a bug, add a test that covers it. The continuous integration workflow runs the same suite on Ubuntu and macOS for Python 3.10 through 3.13 for every push and pull request, under a coverage floor of 60 percent.

## Coding style

Match the style of the surrounding code. Keep public classes and functions documented with clear docstrings that state what the method does, its arguments, and what it returns. Prefer vectorized NumPy operations over Python-level loops in the performance-sensitive I/O and interpolation paths.

The project lints with Ruff, and `ruff check src test` should report no findings before you submit. That is the exact command the CI lint job runs, and it is a blocking gate rather than an advisory one. The rule set is pinned in `pyproject.toml` under `[tool.ruff.lint]` as `select = ["E4", "E7", "E9", "F"]`: the pycodestyle error groups for imports, statements, and syntax or IO errors, plus Pyflakes for undefined names, unused imports, and unused variables. It is pinned rather than left at Ruff's default because that default widens between Ruff releases, which would fail the gate on rules the project never opted into. Widening the set is a deliberate change to `pyproject.toml`, made together with the fixes it demands.

## Pull request process

Fork the repository and create a feature branch from `main`. Keep each pull request focused on a single change so it is easy to review. Write a clear description of what the change does and why, and reference any related issue. Confirm that the test suite passes and that new public behavior is covered by a test. A maintainer will review the pull request and may request changes before merging.

## Code of conduct

Everyone taking part in this project, in issues, discussions, and pull requests alike, is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report unacceptable behavior to dilipkalagotla@gmail.com.

## License

By contributing to lptlib you agree that your contributions will be licensed under the MIT License that covers the project. See [LICENSE](LICENSE) for the full text.
