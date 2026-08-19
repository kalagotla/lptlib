# Contributing to lptlib

Thanks for your interest in improving lptlib. Bug reports, feature requests, documentation fixes, and pull requests are all welcome.

## Ways to contribute

You can open an issue to report a bug or request a feature, improve the documentation, add a test case, or submit a pull request with a code change. If you are planning a large change, please open an issue first so the design can be discussed before you invest time in the implementation.

## Reporting bugs

Good bug reports make fixes much faster. When you open an issue at https://github.com/kalagotla/lptlib/issues, please include the lptlib version and Python version, your operating system, a minimal example that reproduces the problem, the full traceback, and what you expected to happen instead. A small PLOT3D grid or flow snippet that triggers the issue is especially helpful.

## Development setup

Clone the repository and install the package in editable mode with its dependencies.

```bash
git clone https://github.com/kalagotla/lptlib.git
cd lptlib
python -m pip install -e .
pip install pytest
```

The MPI-parallel components require a system MPI implementation. On Debian or Ubuntu you can install one with `sudo apt-get install libopenmpi-dev openmpi-bin`, and on macOS with `brew install open-mpi`.

## Running the tests

The test suite lives under `test/` and is run with pytest from the repository root.

```bash
pytest test -v
```

Please make sure the full suite passes before you open a pull request. If you add a feature or fix a bug, add a test that covers it. The continuous integration workflow runs the same suite on Python 3.10, 3.11, and 3.12 for every push and pull request.

## Coding style

Match the style of the surrounding code. Keep public classes and functions documented with clear docstrings that state what the method does, its arguments, and what it returns. Prefer vectorized NumPy operations over Python-level loops in the performance-sensitive I/O and interpolation paths. The project is configured for Ruff, so `ruff check src` should pass before you submit.

## Pull request process

Fork the repository and create a feature branch from `main`. Keep each pull request focused on a single change so it is easy to review. Write a clear description of what the change does and why, and reference any related issue. Confirm that the test suite passes and that new public behavior is covered by a test. A maintainer will review the pull request and may request changes before merging.

## License

By contributing to lptlib you agree that your contributions will be licensed under the MIT License that covers the project. See [LICENSE](LICENSE) for the full text.
