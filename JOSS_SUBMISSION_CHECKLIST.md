# JOSS submission checklist for lptlib

Prepared 18 August 2026. This file tracks what is ready for a Journal of Open Source Software
(JOSS) submission and the exact steps that remain. It is written to be honest about the current
state of the software rather than to oversell it.

## Status update (19 August 2026)

Several items below are now complete. The README was rewritten as a public front page with badges,
a `docs/index.md` overview was added, a `CONTRIBUTING.md` was added, a GitHub Actions CI workflow
that runs the test suite on Python 3.10, 3.11, and 3.12 was added, the license metadata was
corrected to MIT to match the `LICENSE` file, and `paper.md` and `paper.bib` are now in the
repository root. A GitHub release `v0.1.0` was published and archived on Zenodo, minting the
concept DOI 10.5281/zenodo.22006302 (this v0.1.0 version DOI is 10.5281/zenodo.22006303). The
README and `paper.md` now carry the DOI. The remaining open items are author-facing: confirm the
co-author ORCID for Paul D. Orkwis, confirm the co-authorship and acknowledgement decisions, verify
the institutional affiliation, and supply the numpy-strides-versus-Fortran benchmark numbers for the
Performance section.

## Overall assessment

lptlib is a legitimate JOSS candidate. It is a real, installable, version-controlled research
library of roughly 7,000 lines across sixteen modules, it is published on PyPI, it backs several
peer-reviewed conference papers, and it carries a test suite of about 37 test functions. It clears
the JOSS bar of "substantial scholarly effort." The remaining work is mostly administrative rather
than fundamental, which is consistent with the pipeline audit rating this as the fastest item to
publish. The main substantive gap is the missing performance benchmark described below.

## What is already in place

- Open-source license. A permissive MIT `LICENSE` file is present with a 2025 copyright to Dilip
  Kalagotla. See the license note below for one inconsistency to resolve.
- Public repository. The code lives at https://github.com/kalagotla/lptlib and is distributed on
  PyPI as `lptlib` (version 0.0.6, released May 2026).
- Documentation. `README.md` gives installation, a feature overview, quickstart snippets for the
  I/O, search, interpolation, integration, streamline, and stochastic APIs, a description of the
  DataIO pipeline, a core API reference, and testing instructions.
- Tests. About 37 test functions across 23 files cover search, interpolation (steady and
  unsteady), integration, drag models, streamlines, DataIO, plotting, PLOT3D I/O, and MPI helpers.
- Automated example. `main.py` runs an end-to-end oblique-shock case, seeds a particle
  distribution, and launches an adaptive parallel simulation.
- Packaging and release automation. `pyproject.toml`, `setup.py`, version-bump and release scripts,
  and a GitHub Actions workflow that publishes to PyPI on release.
- The paper. `paper.md` and `paper.bib` are now in the repository root in JOSS format.

## What still needs to be done before or during submission

### Author-facing decisions (do these first)

1. Confirm authorship and ORCIDs in `paper.md`.
   - Dilip Kalagotla is the primary author. He is the sole PyPI maintainer, the license copyright
     holder, and the author of 321 of 324 git commits.
   - Paul D. Orkwis is listed as the second author. This is inferred from his role as advisor, the
     `credit: Paul Orkwis` notes in the source, and his co-authorship on all four related AIAA
     papers. Confirm he agrees to be a JOSS author and is not merely acknowledged.
   - Harpreet Chhabra (git email chhabrhh@mail.uc.edu) made 2 commits and contributed an early
     vortex test case. This is currently an acknowledgement, not authorship. JOSS requires authors
     to have made a significant contribution, so decide whether this level of contribution meets
     that bar or stays in the acknowledgements. It is currently in the acknowledgements.
   - Replace both `0000-0000-0000-0000` ORCID placeholders with real ORCID iDs, or delete the
     `orcid` line for any author who does not have one.

2. Verify the affiliation. The paper lists the University of Cincinnati for both authors. Your files
   sit under a Florida State University OneDrive, so confirm which affiliation should appear, and
   add a second affiliation block if the work spanned both institutions.

3. Supply or regenerate the PLOT3D I/O benchmark. You describe a vectorized, numpy-strided PLOT3D
   reader benchmarked against a Fortran PLOT3D reader, and this performance angle materially
   strengthens the submission. A search of the Scripts folder and the University of Cincinnati
   OneDrive backup did not surface the actual numpy-versus-Fortran timing numbers (the arrakis
   notebooks that were reachable contain visualization and integration-algorithm comparisons, not
   I/O timings, and most of that tree is cloud-only). The Performance section of `paper.md`
   therefore describes the method but leaves the quantitative result as a planned insertion. Please
   either point to the benchmark data if it exists, or regenerate a small, reproducible benchmark
   (a script that times the vectorized reader against a reference Fortran reader on a representative
   grid and flow file) and paste the numbers into the Performance section. Do not ship a fabricated
   figure.

### Repository hygiene JOSS reviewers will check

4. Add community guidelines. JOSS requires clear guidance for third parties on how to contribute,
   report issues, and seek support. Add a `CONTRIBUTING.md` and, ideally, a `CODE_OF_CONDUCT.md`,
   and confirm the GitHub issue tracker is enabled (the `pyproject.toml` already points to it).

5. Add continuous integration that runs the tests. The current GitHub Actions workflow only
   publishes to PyPI. Add a workflow that installs the package and runs `pytest` on pushes and pull
   requests. A visible passing test badge substantially helps the review. Note that some tests pull
   in `mpi4py` and MPI, so the CI image needs an MPI runtime, or those tests need a skip marker when
   MPI is unavailable.

6. Confirm the test suite runs from a clean checkout. Several test files use relative data paths
   (for example `../../data/shocks/...`) and `from src.lptlib import ...` rather than the installed
   package. Verify that `pytest` passes from the repository root on a fresh clone with the declared
   dependencies, and fix any path or import assumptions so a reviewer can reproduce the run.

7. Resolve the license inconsistency. `pyproject.toml` declares
   `MIT AND (Apache-2.0 OR BSD-2-Clause)` and the README repeats it, while the `LICENSE` file is
   pure MIT. This compound expression appears to reflect vendored or submodule code under
   `external/`. Either add the corresponding third-party license texts and a short NOTICE
   explaining the compound license, or simplify the declaration to MIT if no third-party code is
   actually redistributed in the package.

8. Consider the repository size. The `external/forebody_articulation` submodule and the `data/`
   tree contain tens of thousands of `.dat` files. This is not a JOSS blocker, but a leaner repo or
   a clearly separated data submodule makes review easier and keeps the archived release small.

### Submission mechanics

9. Tag a release on GitHub that matches the version in `pyproject.toml` (currently 0.0.6), or bump
   to a clean submission version and tag that.

10. Archive the tagged release to obtain a DOI. Push the release to Zenodo (or figshare), confirm
    the archived metadata title and author list match `paper.md`, and record the DOI. JOSS requires
    an archived DOI at acceptance.

11. Submit at https://joss.theoj.org. Provide the repository URL and the version, and point the
    submission at the branch or tag that contains `paper.md`. The JOSS bot builds the paper PDF from
    `paper.md` and `paper.bib`, so confirm the PDF compiles and the references resolve before and
    during review.

## Gaps summary

- Missing: test-running CI, `CONTRIBUTING.md`, community and support guidelines, a DOI-bearing
  archived release.
- Missing data: the numpy-strides versus Fortran I/O benchmark numbers referenced in the paper.
- To confirm: co-author status and ORCIDs for Orkwis and Chhabra, the correct institutional
  affiliation, and the compound-versus-MIT license question.
- Present and adequate: license file, public repo, PyPI distribution, README documentation, a
  substantive test suite, a working end-to-end example, and a clear statement of need.

## Note on scope relative to other pipeline items

The paper is intentionally scoped as a software and tools contribution. The drag-model comparison
capability is described as a feature of the library, and the paper does not attempt the full
research-grade drag-model study that the audit flagged as a separate 3 to 6 month paper. The
Lagrangian-to-Eulerian reduction is described as producing the Eulerian PIV-like fields, while
synthetic image generation is left to the separate syPIV tool, so the two software papers do not
double-count the same capability.
