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
    orcid: 0000-0003-4090-4031
    affiliation: 1
affiliations:
  - name: Department of Aerospace Engineering and Engineering Mechanics, University of Cincinnati, Cincinnati, OH, USA
    index: 1
date: 21 August 2026
bibliography: paper.bib
---

# Summary

Laser-based velocity measurement never observes the fluid directly. Particle image velocimetry (PIV) seeds a flow with microscopic tracer particles, photographs them twice in quick succession, and infers the velocity field from how far they travelled in between [@raffel2018piv]. The inference rests on an assumption that the particles go where the fluid goes. In high-speed flow that assumption fails. A tracer crossing a shock wave carries too much inertia to decelerate as abruptly as the gas around it, so the measured velocity jump is smeared and the measured gradients are weaker than the real ones [@melling1997tracer]. The measurement is biased, and the size of the bias depends on the particle, not on the instrument.

`lptlib` quantifies that bias. Given a computed flow field, the library releases particles of a specified size and density into it and integrates their trajectories, so the motion a camera would have recorded can be compared against the motion of the fluid itself. The difference between the two is the measurement error attributable to the tracer.

Working from a simulated flow rather than an experiment makes the comparison possible, because the true fluid velocity is known everywhere. The library reads structured, multi-block, curvilinear grids and flow solutions in the PLOT3D format [@walatka1990plot3d] produced by common compressible solvers, locates arbitrary points inside the curvilinear domain, interpolates the flow state to them, and advances both fluid streamlines and inertial particle paths. Particle motion is advanced under a spherical-particle drag law chosen from a set spanning creeping, continuum, compressible and rarefied regimes, so one particle can be tracked under different closures and the answers compared directly. Large ensembles are tracked in parallel through multiprocessing, thread pools, or MPI [@dalcin2021mpi4py], and the resulting scattered tracks are reduced back onto a user-defined Eulerian mesh to produce PIV-like fields.

# Statement of need

Correcting inertial bias in supersonic PIV requires tracking particles of known size and density through a resolved flow field, under a drag law that stays valid across every regime a tracer visits while crossing a shock. A 1 micron alumina particle in a Mach 5 flow passes from continuum to slip to near-free-molecular conditions within a few particle diameters, and no single classical closure covers that path. The practical questions the community asks are consequently comparative. How much does the answer change if the tracer is 300 nanometres instead of 1 micron? How much does it change under Henderson's closure [@henderson1976drag] rather than Loth's [@loth2008compressibility]? Answering either requires a tool where particle properties and the drag closure are both free parameters over the same flow field and the same integrator.

Such a tool has not been openly available. Researchers in this area have generally written their own single-purpose trackers, which are rarely released, rarely tested, and impossible to compare against one another. `lptlib` was built to close that gap. It exposes nine drag closures through a single argument, spanning Stokes and Oseen [@oseen1927hydrodynamik] creeping flow, the Schiller-Naumann [@schiller1935drag] and Melling [@melling1997tracer] continuum corrections, Cunningham slip [@cunningham1910slip], and the compressible and rarefied models of Henderson, Loth, Tedeschi [@tedeschi1999motion], and Subramaniam and Balachandar [@subramaniam2022particle], with viscosity from Sutherland's law [@sutherland1893viscosity]. All nine track against a common integrator, and the results reduce to a gridded field that downstream synthetic-imaging tools can consume. The intended users are experimentalists interpreting high-speed PIV data, and computational researchers studying one-way coupled particle-laden compressible flow, where the same machinery applies unchanged.

# State of the field

Open-source Lagrangian particle tracking is mature, but it is partitioned into communities whose assumptions do not transfer to compressible aerodynamics.

Geophysical transport frameworks are the largest group. Parcels [@delandmeter2019parcels], OpenDrift [@dagestad2018opendrift], and MPTRAC [@hoffmann2025mptrac] advect particles through ocean or atmosphere velocity fields with great sophistication in interpolation, kernels, and scale. Their particles are effectively passive, and the compressible and rarefied drag physics that dominates a tracer crossing a shock has no counterpart in them.

Experimental particle tracking velocimetry tools such as MyPTV [@shnapp2022myptv] sit on the other side of the measurement. They consume camera images and produce trajectories, whereas `lptlib` consumes a CFD field and produces trajectories, then reduces them back to a grid. The two are complementary, and using them together is how a synthetic PIV experiment is closed against a real one.

Closest in intent is ppiclF [@zwick2019ppiclf], a parallel particle-in-cell library for particle-laden flow that scales to very large processor counts. However, ppiclF is written in Fortran and designed to be linked into a host solver, so it tracks particles during a simulation rather than after one. Solver-embedded Lagrangian cloud libraries such as those in OpenFOAM [@weller1998openfoam] share that constraint, along with a dependence on the solver's own mesh format. `lptlib` instead operates on a written solution file, which lets a user study tracer response in an existing archived CFD result without rerunning it, and lets the drag closure be varied without touching the solver.

General-purpose visualization also reaches part of this space. VTK and ParaView [@ahrens2005paraview] read PLOT3D files and offer particle tracers. Those tracers integrate massless streamlines, so they answer where the fluid goes rather than where a tracer of a given size and density goes, which is the entire quantity of interest here. Beyond visualization, open PLOT3D post-processing is dominated by legacy Fortran utilities and closed commercial tools.

# Software design

Three decisions shape the library.

The first is adaptive integration keyed to trajectory curvature. Particle response near a shock is stiff, and a globally small time step is wasteful everywhere else. `lptlib` therefore adapts the step to the angle between successive displacements, refining where a path deflects sharply and coarsening again in smooth regions. Deflection was chosen over a conventional local error estimate because the quantity of interest is the tracer's spatial lag through a discontinuity, and deflection responds to the discontinuity directly rather than to the smoothness of the interpolated field. A blow-up detector guards the stiff branch, reduces the step, retries, and terminates the trajectory with a diagnostic if the step cannot be taken. The step is restored once integration succeeds again, so one shock crossing does not cripple the remainder of a path.

The second is one-way coupling. Particles do not act back on the fluid, which is accurate at the low seeding densities used in PIV and which makes every trajectory independent. Independence is what allows the same tracking code to be driven by multiprocessing, a thread pool, or MPI without changing the numerics, and it is why the library can post-process an archived solution at all.

The third concerns input throughput, since the flow field is read repeatedly across an ensemble. The PLOT3D reader loads each file in one buffered `numpy.fromfile` call and reconstructs multi-block arrays by strided slicing and Fortran-order reshaping, avoiding Python-level iteration over grid points [@harris2020numpy; @virtanen2020scipy]. A reproducible benchmark under `benchmarks/` measures this, comparing readers only when they perform matched work, since a bare stream read and a read followed by a full array reorder are not the same task. On a 2.04 million-point grid, with page-cache state verified by measurement rather than assumed, the strided reader is faster than an otherwise identical naive Python reader by more than an order of magnitude, at a ratio of 27.8 with a 95 percent bootstrap confidence interval of 26.1 to 28.8. Against a compiled Fortran reader performing the identical transpose it is about 1.4 times slower, and the full `read_grid`, which additionally builds the padded double-precision array and the per-block coordinate bounds, is about 1.75 times faster than Fortran doing that same work. Reducing the bounds over the single-precision block views rather than over the padded array makes the full read about 2.5 to 3 times faster, with the bounds verified bit-identical either way.

These ratios come from one machine so far and are reported as ratios deliberately, because absolute read times on this benchmark move by more than an order of magnitude with page-cache state and memory bandwidth while the matched-work ratios largely do not. Every result file, its machine metadata, and the per-run uncertainty are recorded under `benchmarks/results/`, and `benchmarks/README.md` explains how to add a machine.

# Research impact

`lptlib` grew out of, and now underpins, a line of work on particle response and bias correction in supersonic PIV. The library provided the object-oriented tracking framework described in @kalagotla2023oop and the PLOT3D post-processing interface of @kalagotla2023api, supplied the drag-model comparison and particle-size estimation of @kalagotla2024drag, and generated the synthetic training data behind the bias-corrector work of @kalagotla2025bias.

That last line of work has since appeared in peer-reviewed form. The neural bias corrector reported in @kalagotla2026bicsnet was trained on 600 synthetic cases generated by this library and reduces inertial bias by up to 87 percent for conditions inside its training distribution. The solver described there as in-house is `lptlib`, under the name it now carries.

Beyond this group, the library is installable from PyPI, documented, and covered by a test suite that runs on Linux and macOS across Python 3.10 to 3.13. Its verification tests check the implemented oblique-shock relations against an independently coded solution of the theta-beta-Mach relation and against published gas-dynamics tables, so a user can confirm the analytic test cases reproduce before trusting a computed field.

# AI usage disclosure

Generative AI coding assistants were used during the development of `lptlib`, in parts of the implementation, the test suite, the documentation, and the preparation of this paper. All AI-assisted output was reviewed, corrected where necessary, and validated by the authors, who take full responsibility for the correctness of the software and of the claims made here. The scientific formulation, the choice of models, the design decisions described above, and the interpretation of all results are the authors' own.

# Acknowledgements

The authors thank Harpreet Chhabra for contributions to early test cases, and Daniel Cuppoletti for discussions on the experimental context. This work was carried out in the Department of Aerospace Engineering and Engineering Mechanics at the University of Cincinnati.

# References
