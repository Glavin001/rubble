# What engineering and GPU simulation have learned about fast, faithful stress solving

Written 2026-10-16, after the AVBD study (`avbd-stress.md`). A literature pass over
structural, impact and geomechanics engineering, GPU simulation codes and real-time
graphics, asking what they add to or change in `REALTIME_ROADMAP.md` (sections B–D). Each
claim links its source; numbers are the sources' own, not re-measured here. Recommendations
are proposals: the roadmap is unchanged until reviewed.

## Summary

| # | Takeaway | What it means here |
|---|---|---|
| 1 | Rigid chunks + softening spring bonds is the engineering-standard model; explicit for fast loading, implicit for slow | Our model and tiers are standard practice; nothing to change |
| 2 | Explicit multirate (subcycling, AVI) is established; GPU local time stepping pays only when rate clusters fill the GPU | Promote multirate to an oracle switch; batch rate clusters across buildings |
| 3 | Explicit–implicit multi-time-step domain decomposition (Gravouil–Combescure) is the proven "hot zone" | Explicit hot zone + nonlinear Newmark exterior, ahead of a Craig–Bampton exterior |
| 4 | Reduced linear exteriors: cost set by interface size, contact forces off by up to 22 % | Craig–Bampton only as the cheap variant, measured |
| 5 | Skin (Verlet) neighbour lists and decoupled contact detection | Lossless contact cost cut in impact windows; do first |
| 6 | Multigrid is the real-time GPU solver; Chebyshev avoids reductions; mixed-precision refinement keeps f64 accuracy | Concrete design for the T1/T2 implicit solver; dynamic relaxation for small islands |
| 7 | Single precision is fine for short impact runs, test double for long ones | Keep state relative, accumulate energies and history in f64 |
| 8 | SDOF/P–I, empirical impact formulas and limit analysis are screens | Router features and certificates, scored against the oracle |
| 9 | Production destruction: sleeping islands; unconverged bond-graph solvers fail | Confirms T0 and the oracle's role |

## 1. Our model is the engineering-standard one

The chunk–bond model belongs to an established family:
- rigid-body-spring networks (RBSN; Bolander and coworkers), regularised by crack-band
  theory so cracking does not depend on the network
  ([FraMCoS-4](https://framcos.org/paper/rigid-body-springnetworkmodelingofcement-basedcomposites/));
- the Applied Element Method (Tagel-Din and Meguro), used commercially by Extreme Loading
  for Structures for progressive collapse, validated against a shake-table frame
  ([Wikipedia](https://en.wikipedia.org/wiki/Applied_element_method),
  [WCEE 2012](https://www.iitk.ac.in/nicee/wcee/article/WCEE2012_0453.pdf));
- the bonded-particle model of DEM codes (parallel bonds).

Practice in that community matches our tiers: explicit Verlet integration for impact on RC
beams with a rate-dependent unit
([Hwang, Bolander, Lim, FraMCoS-9](https://framcos.org/FraMCoS-9/Full-Papers/216.pdf)),
implicit for slower loading, where the step is limited by accuracy, not stability
([Eliáš](https://engmechx.it.cas.cz/improc/2008/Elias_FT.pdf)). Our `G_f A` softening is
the crack-band regularisation. No change.

## 2. Multirate explicit is established, not research

- **Asynchronous variational integrators** (Lew, Marsden, Ortiz, West, 2003) give each
  element its own step (not integer multiples), with exact local momentum conservation and
  near energy conservation, and substantial speed-ups over explicit Newmark at equal
  accuracy, with a convergence proof
  ([ARMA 2003](https://lagrange.mechse.illinois.edu/pubs/LeMaOrWe2003/),
  [IJNME 2004](https://lagrange.mechse.illinois.edu/pubs/LeMaOrWe2004b/)).
- Extended with domain decomposition and local Lagrange multipliers; accuracy, not
  stability, usually limits the step
  ([Gates, Matouš, Heath](https://www3.nd.edu/~kmatous/Papers/IJNME_AMVI.pdf)); variational
  multirate on macro/micro grids ([arXiv 2406.12991](https://arxiv.org/html/2406.12991v1)).
- **On GPUs**, clustered local time stepping in SeisSol (elastic waves) gained only 1.3×
  on V100 clusters: each rate cluster updates in sequence and holds too little work to fill
  the GPU. On one A100 node with CUDA graphs: 9.75×
  ([arXiv 2108.10565](https://arxiv.org/pdf/2108.10565v2),
  [TUM](https://mediatum.ub.tum.de/doc/1730250/1730250.pdf)).

**For us:** A1 sized multirate at ×1.9–4.3 (×18 on minas-tirith). The roadmap's caution
("IMEX interface is research") applies to implicit–explicit coupling; explicit–explicit
subcycling is the safe variant. The GPU lesson is to batch each rate cluster across all
awake buildings, so a hundred houses fill the GPU where one house cannot, and to launch the
per-rate sequence as a graph or persistent kernel.

## 3. The established "hot zone": explicit–implicit multi-time-step decomposition

The Gravouil–Combescure method (2001–02) splits a structure into subdomains, each with its
own Newmark-family integrator and step, coupled by velocity continuity at the interfaces.
Each subdomain is solved free, then interface corrections give the global solution.
Stability is proven for explicit/implicit, implicit/implicit and explicit/explicit pairs
([arXiv 1305.6355](https://arxiv.org/pdf/1305.6355),
[dual Schur, arXiv 0807.2108](https://arxiv.org/pdf/0807.2108),
[EJCM](https://journals.riverpublishers.com/index.php/EJCM/article/view/2047)). Successors:
- GC+ (energy conserving, free and link problems in parallel);
- a monolithic heterogeneous integrator (Mahjoubi et al., 2011);
- FETI-based coupling for Newmark and α-schemes (Brun et al., 2015);
- an asynchronous explicit–implicit integrator for non-smooth dynamics (Fekak et al., 2017)

([M2AN reference list](https://www.esaim-m2an.org/articles/m2an/ref/2023/01/m2an210239/m2an210239.html)).
Coupling an implicit code (Z-set) with an explicit one (Europlexus) for impacts on
composite panels reached step ratios up to 5,000, with the explicit domain still the
bottleneck ([EJCM](https://journals.riverpublishers.com/index.php/EJCM/article/view/2047)).

**For us:** explicit (the oracle's integrator) in a hot zone around the impact, Newmark
for the rest of the building at a few ms, solved by multigrid (§6). The exterior stays
nonlinear and can crack, which `avbd-stress.md` showed matters: b5's regime at 15–30 m/s is
decided by bending and cracking at the wall's base, far from the impact. Keep the hot zone
small; escalate chunks by utilisation with a trigger sound against leverage (influence
bounds, README "force at the tip of a slab").

## 4. Reduced linear exteriors

- Craig–Bampton reduction with the nonlinear contacts kept at full resolution (a car door,
  squeak and rattle): about 98 % less time, contact forces off by 2.5–22 % on average; the
  run time is dominated by interface DOFs
  ([Chalmers](https://research.chalmers.se/publication/525648)). Interface reduction by
  characteristic constraint modes
  ([Kuether et al., Sandia](https://www.sandia.gov/research/publications/details/on-hurty-craig-bampton-substructuring-with-interface-reduction-on-contactin-2017-08-01/));
  substructuring with localised nonlinearities ([arXiv 2006.16612](https://arxiv.org/pdf/2006.16612)).
- Multi-domain reduced models coupled by inertia, with large rigid motion, in real time at a
  few hundred DOFs ([Barbič and Zhao 2011](https://history.siggraph.org/?p=114678)).
- Fracture from modal analysis: contact force and duration estimated, a damped modal wave
  starts the crack ([Glondu et al. 2013](https://hal.archives-ouvertes.fr/hal-00752372));
  precomputed "fracture modes" ([Breaking Good](https://www.dgp.toronto.edu/projects/breaking-good/)).

**For us:** a linear reduced exterior is the cheaper variant of §3, not the primary one: it
cannot crack, and its interface forces carry errors of the size reported. The graphics
methods are fracture *predictors* (router features), not truth.

## 5. Contact detection: a lossless lever we don't use

DEM and molecular dynamics build contact lists with a skin distance and rebuild them only
when a body has moved more than half the skin: forces are unchanged while the bound holds.
Chrono DEM-Engine decouples detection from force computation onto asynchronous GPU streams
and scales linearly to 150 M elements on two A100s
([arXiv 2311.04648](https://arxiv.org/pdf/2311.04648v1)). GPU FDEM needed a contact
detection redesign to reach 128× over its sequential code
([Fukuda et al. 2019](https://cir.nii.ac.jp/crid/1050566774778391552)).

**For us:** the oracle (and stress-gpu's islands) re-detect contacts every substep inside
impact windows. A skin list gives identical forces with far less detection work: the kind
of lossless optimisation AGENTS.md asks for first.

## 6. Solvers for the implicit and quasi-static tiers on the GPU

- **Multigrid is the proven real-time GPU solver** for deformables: Galerkin multigrid
  with skinning-space coarse spaces and piecewise-constant weights, hundreds of thousands
  of DOFs in real time ([Xian, Tong, Liu 2019](https://unpaywall.org/10.1145%2F3355089.3356486));
  an unstructured-mesh successor up to 6.9× faster
  ([TOG 2025](https://cg.cs.tsinghua.edu.cn/papers/TOG-2025-Fast-Galerkin.pdf)); hexahedral
  GPU multigrid ([Dick, Georgii, Westermann](https://www.cs.cit.tum.de/cg/research/projects/projects/multigrid-fem-simulation-using-cuda/)).
- **Our coarse space is free:** aggregate chunks and use each aggregate's six rigid-body
  modes, the standard near-nullspace of smoothed aggregation for elasticity. This makes
  the roadmap's open "smoothed-aggregation multigrid" item concrete.
- **Latency:** Chebyshev iteration and smoothing need no inner products
  ([Templates](https://netlib.org/utk/papers/templates/node45.html)); s-step CG with a
  Chebyshev basis cuts synchronisations on GPUs
  ([arXiv 2603.09790](https://arxiv.org/html/2603.09790v2), preprint).
- **Precision:** mixed-precision iterative refinement (low-precision solve, f64 residual)
  keeps f64 accuracy at 4–5× the speed on Volta
  ([Haidar et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7735315/)). This answers
  the roadmap's "f32 residual about 6 %" concern for HHT-α.
- **Dynamic relaxation** (kinetic damping: zero velocities at kinetic-energy peaks;
  Cundall's local damping) reaches equilibrium with only the explicit kernel and no global
  reductions; it breaks momentum conservation, so it is for statics only
  ([LAMMPS](https://doc.lammps.org/fix_damping_cundall.html),
  [Springer 2021](https://link.springer.com/article/10.1007/s00366-021-01410-x)). A
  candidate T1 for small islands where PCG is latency-bound.
- AVBD and Gauss–Seidel are not on this list: `avbd-stress.md` measured why.

## 7. Precision over long runs

LS-DYNA's guidance: test single against double precision on the model; long explicit runs
(over about 200,000 steps) and very small steps are where double is needed; double costs
about 30 % more ([Ansys](https://lsdyna.ansys.com/double-precision/)). Our windows run
thousands of substeps per frame over many frames. Keep state relative (the hidden DOFs
already are), accumulate energies and long-lived history (damage, fatigue) in f64 or with
compensated sums, and check f32 against f64 per asset class.

## 8. Engineering's fast assessments are routers and certificates

- **Equivalent SDOF models and P–I diagrams** (Biggs; UFC 3-340-02): standard for blast
  design, but with significant errors on two-way RC panels against explicit FE
  ([UTS](https://opus.lib.uts.edu.au/bitstream/10453/117527/1/2041-4196.4.2.143.pdf)); for
  impact, SDOF is valid when global bending dominates, localised damage needs a two-DOF model
  ([West Point](https://athena.westpoint.edu/items/55620042-4d4f-4e5e-94bd-604ecc031049)).
- **Empirical local-impact formulas:** modified NDRC (penetration, scabbing), CEA-EDF (best
  for perforation in a 145-test assessment), Li and Reid's critical energies against
  thickness-to-diameter ratio. Rigid missiles only, and wide scatter: most fail to predict
  perforation velocity over 95 tests
  ([FFI review](https://www.ffi.no/publikasjoner/arkiv/a-review-of-empirical-equations-for-missile-impact-effects-on-concrete),
  [Carleton](https://carleton.ca/sadeghian/publications/prediction-of-perforation-velocity-of-hard-missile-impacts-on-reinforced-concrete-wall-panels),
  [BIT](https://pure.bit.edu.cn/en/publications/critical-impact-energies-for-scabbing-and-perforation-of-concrete/)).
- **Rigid-block limit analysis by linear programming** (Livesley 1978; Gilbert and
  Melbourne 1994) gives a collapse load factor quickly
  ([IStructE](https://www.istructe.org/journal/volumes/volume-72-(published-in-1994)/issue-21/rigid-block-analysis-of-masonry-structures)):
  a "stands under these loads" certificate for T0.

**For us:** candidate features for the impact router (Phase 7) and certificates (Phase 3),
scored against the oracle's measured critical speeds (`examples/wall_variants`), never the
damage result. This matches the roadmap's verdict on P–I/SDOF.

## 9. Production and game destruction

- Corotational FEM with islands live, asleep or kinematic, shipped in *Star Wars: The
  Force Unleashed* ([Parker and O'Brien 2009](https://escholarship.org/uc/item/2kj9p4bw)):
  sleeping is the main lever (our T0).
- NVIDIA Blast's stress solver: a fixed number of bond-graph iterations per frame, cost
  linear in it, plus "graph reduction" to graphs 2^n times smaller for better propagation
  at coarser damage ([Blast](https://docs.nvidia.com/gameworks/content/gameworkslibrary/blast/1.1/api_docs/files/pageextstress.html),
  [Omniverse](https://docs.omniverse.nvidia.com/extensions/latest/ext_blast.html)). A crude
  two-level scheme with unconverged stresses: the failure mode the oracle exists to avoid.
- Industrial and research GPU impact codes are explicit, full-law and GPU-resident: IMPETUS
  (high-order elements, node-splitting fracture,
  [Teratec 2011](https://teratec.eu/library/pdf/forum/2011/presentations/A4_03_FTeratec_2011_Lacome_Impetus.pdf));
  GPU bonded-particle DEM, atomic-free, 1.86 M particles on a P100
  ([Pacevič et al. 2019](https://am.ippt.pan.pl/am/article/view/v71p459)); GPU
  peridynamics ([PeriPy](https://arxiv.org/pdf/2105.04150)); multi-GPU MPM with a fused
  G2P2G kernel to cut launches ([Wang et al. 2020](https://yzhu.io/publication/mpmgpu2020siggraph/));
  real-time explicit FE on GPUs for surgery (TLED,
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2783832)). Their engineering effort goes
  into contact detection, kernel fusion and avoiding atomics. Older GPU FEA studies found
  graph colouring beat atomics in assembly
  ([IIT Guwahati](https://www.iitg.ac.in/dsharma/papers/Conference/2017_two_kernel.pdf));
  modern atomics narrow the gap, so measure before choosing.
- Graph neural network surrogates are emerging (progressive collapse: F1 0.885,
  [BJUT](https://ir.bjut.edu.cn/irpui/item/410855); review
  [arXiv 2407.11060](https://arxiv.org/pdf/2407.11060)) but are not truth-grade: at most a
  router hint, and adopting one would be a lossy change for the user to decide.
- Selective mass scaling (scale only the highest frequencies) exists but needs non-diagonal
  mass matrices, conditions badly and is little adopted
  ([arXiv 2410.23816](https://arxiv.org/html/2410.23816v2),
  [Springer 2025](https://link.springer.com/article/10.1007/s00466-025-02611-7)); on our
  slivers it would be a lossy model change, measured only.

## Proposed roadmap changes (not applied)

| Roadmap item | Proposal |
|---|---|
| Multirate ("IMEX interface is research") | Explicit–explicit subcycling (AVI or clustered LTS) as an oracle switch next to sliver elimination; GPU version batches rate clusters across buildings |
| Craig–Bampton "deferred" | Gravouil–Combescure explicit hot zone + nonlinear Newmark exterior first; Craig–Bampton as the cheap variant |
| Smoothed-aggregation multigrid "not done" | Chunk aggregates, rigid-body near-nullspace, Chebyshev smoother, mixed-precision refinement |
| New | Contact skin lists in impact windows (lossless) |
| New | Dynamic relaxation for small T1 islands, against GPU PCG |
| Router (Phase 7) | Empirical impact formulas and SDOF/P–I as candidate features, scored against the oracle's critical speeds |

## Recommended experiments (oracle first)

| # | Experiment | Class | Gate |
|---|---|---|---|
| 1 | Contact skin lists in impact windows (`world.rs` `apply_contacts`, `chunk_pairs`) | lossless | observations byte-identical (`scripts/verify_identical.py`); contact time per substep |
| 2 | Multirate explicit (clustered LTS or AVI) on house-1story and the packs | integrator switch | `fidelity.py` against single-rate explicit; substep work (A1: ×1.9–4.3) |
| 3 | Gravouil–Combescure hot zone: explicit hot zone + Newmark exterior; Craig–Bampton variant | integrator/model switch | b5 10–40 m/s, `ram_into_free_wall`, `house_1story_ram` inside the truth's spread (as in `avbd-stress.md`); hot-zone size and work |
| 4 | Smoothed aggregation + Chebyshev for the implicit/static solve, f32 inner and f64 refinement | exact at convergence | iterations and synchronisations against block-Jacobi PCG (house gravity: 769, A3) |
| 5 | Router features: NDRC, CEA-EDF, Li–Reid, SDOF/P–I against `wall_variants` critical speeds | screen | whether they predict the regime boundary |
| 6 | Dynamic relaxation as T1 on small islands | exact at convergence | iterations, wall time, agreement with the static solve |

Already proposed elsewhere: implicit contacts inside explicit windows (the house's window
step is the contact law's stability bound, `avbd-stress.md`), and a test of rubble's AVBD
contacts for approach-normal artifacts.
