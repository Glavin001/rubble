# A fast soft-body tier that looks like sheet metal: research notes (2026-11-02)

Survey for a faster, calibrated alternative to the explicit 2 kHz node–beam solver
(`docs/explicit-beams.md`), such as Rubble's AVBD soft-body path. Only four citations
were checked by web search (AVBD, VBD, Kenwright, Jones et al.); the others are
recalled: confirm them before quoting.

## Why an implicit solver at 120–480 Hz looks like jello

1. Stopped iterations: block Gauss–Seidel (VBD, PBD, projective dynamics) removes
   short-wavelength error fast and long-wavelength error slowly, so stiff beams act
   softer the fewer the iterations and the finer the mesh.
2. Implicit Euler damping grows with frequency × step: panels lose the ring of a
   stiff shell and settle slowly.
3. Large steps spread an impact over few steps; yield checked once after the solve
   sees smaller forces, so less crush.
4. A node–beam lattice without bending energy and plastic bending rest states
   buckles into smooth bulges, not creases.

## Techniques, most directly applicable first

1. **Substeps over iterations, yield every substep.** Equal cost, less softness and
   damping (Macklin et al., *Small Steps in Physics Simulation*, SCA 2019; *XPBD*,
   MIG 2016). First experiment: 480 Hz × 4 against 120 Hz × 16.
2. **Stiff beams as augmented-Lagrangian constraints** with a persistent,
   warm-started λ (AVBD's core: Giles, Diaz, Yuksel, *Augmented Vertex Block
   Descent*, ACM TOG 44(4), 2025, doi:10.1145/3731195; Chen et al., *Vertex Block
   Descent*, 2024, arXiv:2403.06321). λ is a beam-force estimate to test yield and
   break against; reset it when a beam yields or breaks.
3. **Bending (hinge) energy with plastic rest angles** for sharp creases (Grinspun et
   al., *Discrete Shells*, SCA 2003; Bergou et al., quadratic bending, SGP 2006;
   Narain, Pfaff, O'Brien, *Folding and Crumpling Adaptive Sheets*, SIGGRAPH 2013).
   Interim: yield on the existing cross and diagonal beams.
4. **Return mapping inside the iterations**: clamp a beam's force at yield and move
   its rest length while iterating, so plastic flow absorbs the energy (less
   springback, more crush) (Simo & Hughes 1998; Bargteil et al., SIGGRAPH 2007).
5. **Strain limiting** of in-plane stretch, so sheets deform by bending (Provot, GI
   1995; Goldenthal et al., SIGGRAPH 2007).
6. **Less dissipative integration**: BDF2 (Choi & Ko, SIGGRAPH 2002), energy
   projection (Dinev et al., FEPR, 2018). Mostly affects ringing and rebound; fix
   stiffness and plasticity first.
7. **Acceleration** for the slow long-wavelength modes: Chebyshev (Wang, SIGGRAPH Asia
   2015), Galerkin multigrid (Xian, Tong, Liu, 2019), a per-vehicle coarse
   correction (speculative).
8. **Calibration against the explicit reference**: matched barrier, pole and tap
   tests; fit a few global, physically explainable knobs (yield scale, hinge yield
   moment, substeps), each an explicit engine-wide setting.
9. **Looks**: a detailed render mesh skinned to the physics nodes (RoR and BeamNG
   flexbodies), damage normal maps keyed to plastic strain; Jones et al.,
   *Example-based Plastic Deformation of Rigid Bodies*, SIGGRAPH 2016; Kenwright,
   *Scalable Real-Time Vehicle Deformation*, arXiv:2304.05045.

## Metrics for jello against sheet metal

Modal frequencies and damping (tap test, lowest 5–10 modes); kinetic energy left
after removing the best-fit rigid motion, and its decay after impact; springback
(peak over final crush); plastic energy absorbed against impact energy; crease
sharpness (share of total dihedral change in the top 5 % of hinges); constraint
residual against iteration count.

## Constraint

Damping, strain-limit bands, fewer substeps and calibrated yield scales change
results: per AGENTS.md they are explicit engine-wide settings adopted only by the
user's decision, each reported against the explicit reference.
