---
title: 'lptlib: A parallel Lagrangian particle tracking library for compressible CFD data and tracer-response analysis in optical velocimetry'
tags:
  - Python
  - computational fluid dynamics
  - Lagrangian particle tracking
  - particle image velocimetry
  - compressible flow
  - shock waves
  - uncertainty quantification
authors:
  - name: Dilip Kalagotla
    orcid: 0000-0002-5453-2585
    corresponding: true
    affiliation: 1
  - name: Paul D. Orkwis
    orcid: 0000-0000-0000-0000  # TODO: confirm co-author ORCID
    affiliation: 1
affiliations:
  - name: Department of Aerospace Engineering and Engineering Mechanics, University of Cincinnati, Cincinnati, OH, USA
    index: 1
date: 18 August 2026
bibliography: paper.bib
---

# Summary

`lptlib` is a Python library for one-way coupled Lagrangian particle tracking (LPT) in steady and unsteady computational fluid dynamics (CFD) data. The library reads structured, multi-block, curvilinear grid and flow fields in the PLOT3D format, locates arbitrary points inside the curvilinear domain, interpolates the flow state to those points, and integrates fluid streamlines and inertial particle trajectories through the field. Particle motion is advanced with a spherical-particle drag law selected from a broad set of compressible and rarefied models, so a single particle definition can be tracked under different drag closures and the results compared directly. Large ensembles of particles are tracked in parallel using multiprocessing, thread pools, or MPI, with adaptive time stepping for stiff particle dynamics near shocks.

Two capabilities distinguish the library. First, the PLOT3D reader is fully vectorized. Each grid or flow file is read in a single buffered `numpy.fromfile` call and reconstructed into multi-block arrays through strided slicing and Fortran-order reshaping, which avoids Python-level loops over grid points [@harris2020numpy]. Second, `lptlib` reduces scattered Lagrangian particle tracks back onto an Eulerian mesh to produce particle-image-velocimetry-like fields. The reduction is parallelized with MPI and exports PLOT3D grid, fluid, and particle files for visualization and for downstream synthetic-image generation. Together these features let a user move from a CFD solution to a seeded, tracked, and Eulerian-reduced particle field within one framework.

An archived snapshot of `lptlib` is deposited on Zenodo (DOI: [10.5281/zenodo.22006302](https://doi.org/10.5281/zenodo.22006302)).

# Statement of need

Particle image velocimetry (PIV) and particle tracking velocimetry (PTV) infer fluid velocity from the motion of seeded tracer particles [@raffel2018piv]. In high-speed and supersonic flows the tracers do not follow the fluid exactly. Their inertia makes them lag across shocks and through steep gradients, which biases the measured velocity jumps and gradient magnitudes [@melling1997tracer]. Quantifying this bias requires tracking particles of known size and density through a resolved flow field with a drag law valid across the compressible and rarefied regimes that a tracer experiences as it crosses a shock.

Existing open-source LPT tools do not fill this need. Most target incompressible, atmospheric, or oceanographic transport and assume simple Stokes drag, and general CFD post-processing of the PLOT3D format is dominated by Fortran and closed commercial visualization tools. `lptlib` was built to close this gap for the optical-velocimetry community. It reads the structured curvilinear PLOT3D files produced by common compressible solvers, implements a range of drag models spanning creeping, continuum, compressible, and rarefied regimes, and exposes them through one interface so that tracer response can be studied as a function of particle size, density, and drag closure. The library underlies a series of studies on particle response, drag modeling, and bias correction for supersonic PIV [@kalagotla2023oop; @kalagotla2023api; @kalagotla2024drag; @kalagotla2025bias].

The scope of `lptlib` is deliberately broad. The same machinery that quantifies PIV tracer lag applies to any one-way coupled particle-laden flow problem where inertial particles are advected through a resolved compressible field, so the drag-model breadth and the comparison capability are useful well beyond the velocimetry cases explored so far.

# Key features

- PLOT3D input and output for multi-block, structured, curvilinear grids and flow solutions, including single and double precision, unsteady flow sequences, and two-dimensional Fortran-record planes. The reader is vectorized for throughput.
- Point location in curvilinear grids with physical-space and computational-space search, and conversion between the two spaces.
- Interpolation of flow variables to arbitrary points, with physical-space, computational-space, radial-basis-function, and regular-grid options, plus an analytic oblique-shock interpolant for controlled test cases.
- Streamline and particle-path integration with second- and fourth-order Runge-Kutta schemes in physical and computational space, unsteady variants, and adaptive time stepping.
- A broad set of spherical-particle drag models, including Stokes, Oseen, Schiller-Naumann, Melling, Cunningham, Henderson, Loth, Subramaniam-Balachandar, and Tedeschi closures, together with a zero-drag mode for fluid tracers, all selectable through a single argument for direct comparison [@henderson1976drag; @loth2008compressibility]. Air viscosity is evaluated with Sutherland or Keyes laws.
- Stochastic seeding of particle-size distributions and spawn locations, with parallel execution over many particles through multiprocessing, thread pools, and MPI.
- Derived-variable computation for velocity, temperature, pressure, Mach number, and viscosity.
- An MPI-parallel Lagrangian-to-Eulerian reduction that bins and interpolates scattered particle tracks onto a user-defined structured mesh and writes PLOT3D fluid and particle fields for visualization and for synthetic PIV imaging.

The Eulerian reduction produces the gridded fields that a separate synthetic-imaging tool consumes to render PIV frames, so `lptlib` covers the tracking and field-reduction stages while image synthesis remains outside its scope.

# Performance

The vectorized PLOT3D reader is the main performance contribution. Reading a file in one buffered call and reshaping with Fortran-order strides removes the per-point Python overhead that dominates naive readers, and the approach was designed to be competitive with a compiled Fortran PLOT3D reader on the same files. A quantitative benchmark of the vectorized reader against a reference Fortran implementation is planned for inclusion here.

# Acknowledgements

The authors thank Harpreet Chhabra for contributions to early test cases. This work was carried out in the Department of Aerospace Engineering and Engineering Mechanics at the University of Cincinnati.

# References
