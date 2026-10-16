# stress-oracle — reference destruction stress solver and its oracles

> **In rubble (2026-10-16).** This workspace was copied without history from the PhysX
> fork (`integration/oracle-stress-gpu` at `ec8118341`), so rubble no longer depends on
> that repository. It keeps `stress-ref` (the CPU f64 reference), `stress-gpu` (the Slang
> GPU solver and its wgpu host) and `stress-reduce` (static condensation), with the
> goldens, oracle pipelines and docs. The scene packs the tests load are in `packs/`.
> Left behind: `stress-physx`, the CUDA shader output, the PhysX build glue and
> `scripts/realtime_verify.sh` (they link PhysX). Rubble's AVBD engine takes over the
> rigid-body role `stress-physx` played. PhysX runs described below and in the other
> documents are historical measurements. `stress-physx`'s product-path video tool (its
> offscreen `render.rs` and the `destruction_video` example) was ported to
> `crates/stress-rubble` (`render` module, `destruction_video` example, `gpu` feature;
> see "Destruction videos" below).
>
> `stress-viz` (the debug renderer and the interactive viewer) followed later from the
> fork's `feat/oracle-physx-quality` (`905d53209`: the GPU rasterizer, `stress-viz view`
> and its `--solver-mode`), with `scripts/render_videos.sh`; its `physx` feature became
> `rubble` (`--engine rubble`: `stress_rubble::RubbleEngine` through `World::with_engine`).

This workspace is the ground truth for the destruction stress solver described in
the spec (rigid clusters of chunks, a small-strain elastodynamic bond network per
cluster, fracture by dynamic stress). It has three layers, validated in order:

1. **External oracles** (`oracles/`): analytic solutions plus open tools, each run
   as a separate process from the *same* scene file through an exporter. Their
   results are committed as golden observations (`golden/`).
2. **Reference solver** (`crates/stress-ref`): slow, double-precision, explicit CPU
   Rust implementation of the full model, with true or scaled stiffness. It must
   pass every benchmark; the optimized CUDA solver is later diffed against it.
3. **Engine coupling** (`crates/stress-physx`): the same world with PhysX (CPU) as its
   rigid-body engine instead of its own rigid-body code, plus the engine-facing trait
   that a CUDA solver will also implement. This is an *additional* comparison, not a
   replacement: every PhysX run is gated against the analytic values, the oracle
   goldens and the standalone world's run of the same scene.

Nothing is "is this right?": every quantity is compared, by one shared metric
implementation, against an analytic value or an oracle run of the identical setup.

Taking over this work? Start with [`HANDOFF.md`](HANDOFF.md).

## Layout

| path | what |
|---|---|
| `crates/stress-ref/src/scene.rs` | scene format `stress-scene/1` (the single input every solver reads) |
| `crates/stress-ref/src/builders.rs` | the benchmark scene catalogue (single source of every scene) |
| `crates/stress-ref/src/material.rs`, `bond.rs`, `joint.rs` | materials, bond geometry/stiffness, the damageable joint |
| `crates/stress-ref/src/solver.rs`, `statics.rs`, `implicit.rs`, `refine.rs` | clusters, explicit substeps, quasi-static solve, implicit Newmark step, fracture/splitting, refinement |
| `crates/stress-ref/src/world.rs`, `contact.rs`, `blast.rs` | standalone world: impactors, ground, contact, scripted loads, blast, events, probes |
| `crates/stress-ref/src/engine.rs`, `world_coupling.rs` | `RigidEngine` trait and `World::with_engine`: an external engine owns rigid motion and contact except in stress-owned impact islands |
| `crates/stress-ref/src/cli.rs` | the command line, shared by `stress-ref` (standalone) and `stress-physx` (PhysX, gated against standalone) |
| `crates/stress-ref/src/observation.rs`, `metrics.rs` | `stress-observation/1` and the metrics computed identically for every solver |
| `crates/stress-ref/src/api.rs` | engine-facing trait (`StressSolverApi`), contact-impulse filter (pulse or velocity condition), `EngineCoupledSolver` |
| `crates/stress-ref/src/snapshot.rs` | `World::snapshot()`: world-frame chunk boxes, Love-Weber chunk stress, bond damage, for renderers and debugging |
| `crates/stress-ref/src/showcases.rs` | the spec's showcase scenes (overhang, supports, car, floor drop, arch, house) |
| `crates/stress-ref/tests/` | the test suite (analytic, conservation, failure laws, dynamics, solve modes, features, determinism, refinement, engine API, snapshots, showcases, oracle goldens); `tests/shared/showcases.rs` is also run through PhysX |
| `crates/stress-viz` | debug renderer: runs a scene and writes an MP4 of view panels (utilization, stress, damage, fragments, speed, deformation) and variant comparisons; rasterizes on the GPU through wgpu (Metal, Vulkan, DX12) or in software (`--renderer auto\|gpu\|cpu`); `stress-viz view` is the interactive window (see "Viewer"); `--features rubble` adds rubble's engine (`--engine rubble`) |
| `scripts/render_videos.sh` | renders every benchmark, showcase and feature comparison, and the oracle-vs-rubble comparisons, into `videos/` (not committed) |
| `crates/stress-rubble` | `RubbleEngine` (rubble's AVBD GPU engine behind `RigidEngine`), its tests and benchmarks; with `gpu`, the offscreen video renderer (`render`) and the `destruction_video` example |
| `crates/stress-physx` | `PhysxEngine` (PhysX CPU behind `RigidEngine`), the `stress-physx` command, the older `PhysxDestruction` adapter over `StressSolverApi`, and their tests |
| `oracles/CONTRACT.md` | what an oracle reads and writes |
| `oracles/<tool>/` | `install.sh`, `PINNED.md`, `export.py`, `run.py`, `observe.py`, `pipeline.py`, `README.md` |
| `golden/<scene>/<tool>[_seedN].json` | oracle observations, with `provenance_<tool>*.json` |
| `scenes/` | generated (not committed): `stress-ref gen-scenes scenes` |

## Quick start

```sh
cd destruction/stress-oracle
cargo test --release                         # whole reference test suite (~15 min, mostly one refinement study)
cargo run --release -- gen-scenes scenes     # write every benchmark scene (with derived bond data)
cargo run --release -- run scenes/b2_cantilever.json --out /tmp/ours.json
cargo run --release -- compare scenes/b2_cantilever.json /tmp/ours.json golden/b2_cantilever/opensees.json
cargo run --release -- check scenes golden   # run every scene, compare with analytic + every golden (~7 min)
cargo run --release -- seeds scenes/b9_panel_high_weibull.json --out /tmp/seeds --count 20

# PhysX as the rigid-body engine (needs a PhysX SDK build; see "PhysX" below)
PHYSX_ROOT=/path/to/physx-install cargo test --release -p stress-physx
PHYSX_ROOT=/path/to/physx-install cargo run --release -p stress-physx -- check scenes golden
```

### Destruction videos

`crates/stress-rubble/examples/destruction_video.rs` renders the product path, the GPU
stress solver coupled to rubble on the solver's device (`GpuWorld::with_engine` +
`RubbleEngine::with_device`), one 1/60 s frame at a time at the realtime_peak settings.
Every chunk hull or box is drawn at its solver pose (WGSL, 4x MSAA, headless), with an
overlay, and piped into ffmpeg (`FFMPEG` overrides the binary); a per-frame CSV (step
time, substeps, broken bonds, clusters) is written next to the video.

```sh
cargo build --release -p stress-rubble --features gpu --example destruction_video
# four packs (house-1story, brick-building, house-2story, villa-savoye) struck by lobbed balls
target/release/examples/destruction_video --scene town --seconds 6 --out videos/demo/town.mp4
# free walls rammed in turn in zero gravity, with the measured step time per frame
target/release/examples/destruction_video --scene live --seconds 8 --encode-after --live-readout --out videos/demo/live.mp4
```

Other options: `--width`/`--height` (default 1920x1080), `--no-render` (timing only),
`--info` (scene summary), `--scene live-gravity`, a comma-separated subset of the town's
packs, and `--walls`, `--wall NXxNYxNZ`, `--chunk`, `--ram-mass`, `--ram-half`, `--houses`.
The packs load unchanged: their hull colliders (at most 42 points) are within the
64-corner limit stress-ref and rubble share, and a pack beyond it fails at import. The
fork's version-1 Voronoi packs (fractured-tower, -bridge, -wall), which it reduced to 64
corners at load time, are not in `packs/`, so the town here has four structures, not seven.

`stress-ref run` options: `--seed N`, `--mode explicit|implicit|adaptive|quasi_static`,
`--stiffness-scale S`, `--max-substep DT`, `--frame-dt DT`, `--no-fracture`,
`--feature NAME=on|off` (model switches, see below), `--set dotted.path=JSON` (any scene
field, e.g. `--set sim.implicit_dt=2e-4`, `--set impactors.0.velocity=[0,2,0]`).

```sh
# Videos (needs ffmpeg): one scene, a comparison, or the whole set
cargo run --release -p stress-viz -- render b5_wall_impact_v40 --views utilization,stress,damage,fragments -o v.mp4
cargo run --release -p stress-viz -- compare s_blast_two_walls \
    --variant "shadowing on:" --variant "shadowing off:feature blast_shadowing=off" -o cmp.mp4
scripts/render_videos.sh            # or a subset: scripts/render_videos.sh b5 s_house feature_
PARALLEL=4 RAYON_NUM_THREADS=4 scripts/render_videos.sh   # several videos at once (many-core machines)

# Interactive window: live run, replay/scrub, free camera, click to fire steel spheres
cargo run --release -p stress-viz -- view s_house_car --views damage,fragments,utilization --arrows
# rubble's AVBD engine moving the bodies (in render, still, compare and view)
cargo run --release -p stress-viz --features rubble -- view s_house_car --engine rubble
```

```sh
# rubble's production pipeline in a window (stress-gpu + rubble's AVBD engine on one
# device, no frame rewinds), drawn from the GPU state: click fires, R resets, 1-9 scenes
cargo run --release -p stress-rubble --bin destruction-view -- --scene wall_ram
cargo run --release -p stress-rubble --bin destruction-view -- --list
```

### Oracles

Each oracle installs into its own pinned environment and runs from the scene file:

```sh
oracles/opensees/install.sh      && python -I oracles/opensees/pipeline.py scenes/b8_frame_sudden.json
oracles/opencourant/install.sh   && python -I oracles/opencourant/pipeline.py scenes/b6_spall.json
oracles/lmgc90/install.sh        && python -I oracles/lmgc90/pipeline.py scenes/b7_masonry_v15.json
oracles/kratos/install.sh        && python -I oracles/kratos/pipeline.py scenes/b7_masonry_v15.json
```

| oracle | version (pinned) | licence | covers |
|---|---|---|---|
| Analytic | closed forms in `builders.rs` | — | benchmarks 1-4, b6 1-D spall estimate |
| [OpenCourant](https://opencourant.org) (OpenRadioss fork) | `latest-20261006`, commit `33e6851`, zip sha256 `9d67531d…` | AGPL v3 (run as a separate process only) | b1, b3, b5, b6, b7, b9 |
| [OpenSeesPy](https://openseespydoc.readthedocs.io/) | 3.8.0.0 (core 3.8.0), wheel hashes in `PINNED.md` | free, **not OSI**; research/internal use only | b2, b8 |
| [LMGC90](https://git-xen.lmgc.univ-montp2.fr/lmgc90/lmgc90_user) (pylmgc90) | `lmgc90_user_2026.rc1`, sha256 `a3190240…`, built locally | CeCILL v1.1 | b7 |
| [Kratos DEM](https://github.com/KratosMultiphysics/Kratos) | 10.4.4 (KratosMultiphysics + DEMApplication) | BSD-4 | b7 |

`compas_lmgc90` was evaluated and rejected (one contact law for every pair, no initial
velocity, no sphere contactor); `KratosMultiphysics-all` is not installable (missing
wheel), so the individual Kratos applications are pinned. Details per tool in
`oracles/<tool>/README.md`.

### PhysX

`stress-physx` links this repository's PhysX adapter (`blast/blast-stress-solver-rs`,
feature `physx`) against a CPU-only PhysX build. One that works:

```sh
cmake -S physx/compiler/public -B <build> -G Ninja -DCMAKE_BUILD_TYPE=release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS=-UDISABLE_CUDA_PHYSX -DCMAKE_C_FLAGS=-UDISABLE_CUDA_PHYSX \
  -DPX_GPU_BACKEND=CUDA -DPHYSX_ROOT_DIR=$PWD/physx -DTARGET_BUILD_PLATFORM=linux \
  -DPX_OUTPUT_LIB_DIR=<build>/artifacts -DPX_OUTPUT_BIN_DIR=<build>/artifacts \
  -DNV_FORCE_64BIT_SUFFIX=TRUE -DPX_OUTPUT_ARCH=x86 -DPX_GENERATE_STATIC_LIBRARIES=TRUE \
  -DPX_BUILDSNIPPETS=FALSE -DPX_BUILDPVDRUNTIME=FALSE -DPX_GENERATE_GPU_PROJECTS=FALSE \
  -DPX_GENERATED_INCLUDE_DIR=<build>/include
cmake --build <build> --target PhysX PhysXCommon PhysXFoundation PhysXExtensions PhysXPvdSDK PhysXCooking
# install: physx/include + <build>/include/PxConfig.h -> $PHYSX_ROOT/include, the .a files -> $PHYSX_ROOT/lib
```

`-UDISABLE_CUDA_PHYSX` keeps `PxCreateCudaContextManager` linkable (the adapter calls
it unconditionally); without a GPU it returns null and the CPU scene is used.

## The model (what the reference implements)

See the module docs for the equations. In brief:

* **Chunks** (`structure.rs`, `hull.rs`): boxes or convex hulls (up to 64 vertices;
  exact mass, centroid and inertia by Mirtich's integrals). Contact samples a hull as
  it does a box (corners pulled 10% in, face centroids) against the other's face
  planes. A scene given as hulls of its boxes reproduces the box scene (statics and
  waves to 1e-14; impact to rounding, `tests/hulls.rs`).
* **Bonds** (`bond.rs`): beam-type joints at the contact patch with axial `EA/L`,
  shear `GA/L`, bending `EI/L` (two axes) and torsion `GJ/L`; `L` is the chunk spacing
  along the normal. E and G may be scaled (`sim.stiffness_scale`); strengths and
  fracture energies never are.
* **Joint law** (`joint.rs`): tension cutoff on the extreme fibre, Mohr-Coulomb shear
  (`c + mu sigma_c`, capped), crushing, Euler buckling cap for slender members, dynamic
  increase factor (fib Model Code 2010 tensile law for concrete), Weibull strength
  factor per bond, and **static fatigue** (delayed failure under sustained load):
  subcritical crack growth `da/dt = A K^n` (Charles; Evans & Wiederhorn) integrated to
  a consumed life `omega' = (n + 1) s^n / t_test` and a residual strength
  `(1 - omega)^(1/(n-2))`. For concrete `n` and `t_test` follow from the same fib
  low-rate DIF exponent and reference rate (`n = 54.6`, `t_test = 100 s`): rate
  dependence and delayed failure are one mechanism. This is strength loss with time,
  not creep (no viscous deformation). Damage `D` (tension/shear) and `Dc` (crushing)
  soften with the fracture energy so a bond dissipates exactly `G_f A` (resolution
  independent); brittle = linear softening, ductile (steel) = plateau then snap. The
  cracked share of a joint becomes a no-tension multi-spring patch with Coulomb
  friction (rocking about the compressed edge, arching), so a cracked joint still
  carries compression inside its cluster. Rebar acts in parallel until its plastic
  work reaches its rupture energy.
* **Clusters** (`solver.rs`): connected components of intact bonds, one rigid body
  each. Chunks carry hidden 6-DOF displacements in the cluster's floating (mean-axis)
  frame; chunk loads subtract the rigid motion (linear, angular, centrifugal, Coriolis
  and gyroscopic terms). Splits give each child the exact momentum of its chunks
  (`v + w x r` plus hidden velocities); the energy still stored in the separating
  bonds is booked (`split_release`), not injected into the fragments.
* **Solve modes** (`sim.solve_mode`):
  * `explicit` — central differences with stiffness-proportional dashpots, the
    substep a safety fraction of the smallest per-chunk damped limit
    `2/w (sqrt(1 + zeta^2) - zeta)`, with `w` and `zeta` from Gershgorin bounds on
    each chunk's own stiffness and dashpot rows. **The ground truth.**
  * `implicit` (`implicit.rs`) — Newmark average acceleration at `sim.implicit_dt`
    (default the frame): `(K + gamma/(beta dt) C + M/(beta dt^2)) du = r`, i.e. the
    static solve plus one diagonal term; modified Newton with the joints' exact 6x6
    tangent (active contact springs, friction return map), secant fallback and line
    search; damage committed after convergence, so cascades spread over steps.
  * `quasi_static` (`statics.rs`) — equilibrium every frame (block-Jacobi PCG, inertia
    relief for free clusters) with the same-step cascade. A cluster with no static
    equilibrium (a mechanism, e.g. a joint yielding through) keeps its last
    equilibrium and the frame is counted (`static_unconverged_frames`): damage is never
    committed from a non-converged iterate (it used to be, which collapsed the villa
    pack in one frame while the explicit solve breaks 15 of its 2464 bonds in 0.1 s).
    `adaptive` keeps such a cluster explicit.
  * `adaptive` — explicit while recently loaded, quasi-static once quiet, no work
    while loads are steady.
* **Stiffness vs mass scaling**: `sim.stiffness_scale` lowers E (the original
  real-time compromise; strengths stay physical); `sim.mass_scaling_dt` instead adds
  deformation inertia only to chunks whose own stable step is below the target (as
  explicit FE codes do) and reports the added mass (`added_mass_fraction`).
* **Two scales** (`refine.rs`): coarse chunks with pre-fractured children refine where
  a bond's utilization passes `sim.refine_utilization` or a contact hits them; the
  children are relaxed to equilibrium and inherit crack history.
* **Loads** (`world.rs`, `blast.rs`, `api.rs`): gravity, supports with reactions,
  substep penalty contact (impactors, ground, debris landing on structures), impactor
  crush laws (the impactor's own rigid-plastic force-deformation curve, acting in its
  contact with the structure), Friedlander face pressures, Kinney-Graham blasts with
  angle of incidence, shadowing and clearing/venting, member and support removal by
  force replacement (sudden or gradual). Engine contact impulses are split per
  touching pair into the impact that stopped the approach (at most `2 m v`, counted
  above the engine's resting band `|g| dt`) and a force sustained over the frame; the
  impact is never `impulse / dt` but a pulse of Hertz (or crush-plateau) duration or,
  with `ImpactModel::VelocityCondition`, an instantaneous velocity change of the
  struck chunk (momentum exact, no assumed duration).
* **Model switches** (`sim.features`, `Scene::with_feature`, `--feature`): every
  mechanism can be switched off to see what it contributes: `rate_effects`,
  `static_fatigue`, `weibull`, `softening` (off = threshold model), `crack_contact`,
  `crushing`, `buckling`, `rebar`, `rigid_motion_loads`, `blast_shadowing`,
  `blast_clearing`, `damping`. `tests/features.rs` checks that each removes exactly its mechanism.
* **Outputs**: per-bond damage/utilization/mode (`api::BondReport`), debug snapshots
  (`World::snapshot`: Love-Weber chunk stress, von Mises, principal stress, bond
  status), events (`Cracked`, `Creaked`, `Broken` with position and normal, `Split`,
  `Refined`), an energy ledger (fracture, friction, dashpots, contact, softening
  overshoot, energy left in split bonds).

## Review decisions (2026-10-09)

A review of the spec raised these points; how the reference resolves each:

| point | resolution |
|---|---|
| "Stored elastic energy of broken bonds released to fragments" would mean writing velocities into engine bodies | Not done. Splits are momentum exact from the chunks' actual velocities; the energy left in separating bonds is booked as `split_release`. Engine bodies only ever receive the children's exact momentum. |
| Crush caps vs PhysX contacts | The crush law is the impactor's own force-deformation curve in its contact with the structure (`CrushDesc`), never a cap on rigid-body contacts. With `ImpactModel::VelocityCondition` the engine's contact impulse is used as is. |
| Same-step cascade vs one correction per tick | Only `quasi_static` cascades within a step. `explicit` and `implicit` cascade over steps (a cascade takes time). |
| Sustained-load damage needs a cited law (and creep is out of scope) | Replaced by static fatigue from subcritical crack growth (Charles 1958; Evans & Wiederhorn 1974; Ritter 1978), with `n` and `t_test` derived from the material's fib MC2010 rate law. It is strength loss, not creep. |
| Lowering E shifts the impulsive/quasi-static boundary | Kept as an option and measured. Added implicit Newmark at true E and selective mass scaling; `scripts/render_videos.sh` renders the b5 punch-through with true E, `E x 0.01` and mass scaling side by side. |
| Impact as an initial velocity condition | `ImpactModel::VelocityCondition` (engine path), and the implicit mode's predictor delivers substep contact impulses the same way. |
| Analytic tolerances of 5% hide bugs | Benchmarks 1-4 gate at 1%; b2 also gates the exact discrete model at 1e-6; oracle-only comparisons keep the spec's 20-30%. |
| Bit-identical runs; explicit convergence criteria | `tests/determinism.rs`: repeated fracturing runs are bit-identical (explicit and implicit); static solves stop on a relative residual of 1e-10 and match the exact discrete solution in every load direction. |

## Swapping implementations

`api::StressSolverApi` is the contract a stress solver (this reference, the CUDA one)
implements: `step(FrameInput{dt, motion, contacts}) -> FrameOutput{fractures, events}`,
`clusters()`, `bond_reports(structure)`. Applying fractures to the engine
(`stress-physx`) is shared code on the other side of that trait.

## PhysX as the rigid-body engine

The standalone world is the oracle: its own stress solve *and* its own rigid-body
motion and contact (chunk-level penalty contact on the stress substep). As an
additional comparison, `World::with_engine(scene, PhysxEngine::boxed(gravity))` runs
the same scene with PhysX CPU integrating the rigid bodies and resolving their
contacts. `stress-physx` is the same command line as `stress-ref`; its `run` and
`check` also run each scene in the standalone world and gate every metric against
it, next to the analytic and oracle-golden rows.

**Where a rigid engine's assumption fails, and what happens then** (`engine.rs`,
`world_coupling.rs`). A rigid-body engine treats bodies as infinitely stiff: a contact
instantly involves the whole body and the body cannot change during it. That holds
for resting, stacking, sliding and slow contact, and fails for a fast impact on
something breakable. Before each frame the world tests every pair whose swept bounds
touch: the rigid assumption fails if the Hertz contact time at the closing speed is
shorter than a wave round trip across the struck body, or if the peak contact force
exceeds its weakest bond (the closing speed is taken along the normal from each
body's centre to the nearest point of the other's bounds, so bodies flying apart are
not impacts). Such pairs, and every body touching them during the frame (union-find
over swept bounds), form an **impact island**. The prediction is then **verified**:
if a body PhysX moved fractures during the frame (a force at the tip of a slab breaks
it far below its weakest bond's capacity, through leverage), PhysX's assumption was
wrong for it; the world and PhysX roll back to the start of the frame, the body joins
an island, and the frame is solved again. PhysX holds island bodies
kinematic for the frame; the world simulates them with the validated reference
physics (chunk contact on the stress substep, fracture during the contact) and hands
PhysX their end states. Every other body is **driven**: PhysX integrates it with all
its contacts and with the impulses of the scripted loads; the stress solve follows
its motion (interpolated across the substeps) and takes its contact impulses. After
fractures the bodies are rebuilt in PhysX (one body per cluster, one box shape per
chunk, so contacts are reported on the chunk they touch).

**Engine contacts into stress** (`api::ContactLoadFilter`). A frame's contact points
between one chunk and one body are one manifold (summed impulse; point and closing
speed at its centre of pressure). An impact at closing speed `v` between bodies of
reduced mass `m` delivers at most `2 m v`, so `min(J, 2 m v)` is an impact (a Hertz
pulse, or a velocity condition) and the rest of the frame's impulse is a force
sustained over the frame; only the closing speed above the engine's resting cycle
`|g| dt` counts. No tuning constants. PhysX bodies are kept awake: a sleeping body
reports no contacts, and a resting body's reactions are loads.

| check (`cargo test --release -p stress-physx`) | standalone | PhysX |
|---|---|---|
| box resting on the ground: reported impulse per step | — | 16.351 N s up (m g dt = 16.350) |
| free column on the ground, base joint (4 m g = 753.4 N) | 753.4 N | 753.7 N (PhysX integrates it every frame, no island) |
| 40 m/s ram through a free wall: ram after, fragments, momentum | 35.03 m/s, 6 | 35.05 m/s, 6, conserved to 1e-3 |
| showcases (`tests/shared/showcases.rs`, all 7 behaviours asserted) | pass | pass |

**Oracle check through PhysX** (`stress-physx check scenes golden`). Every gated
metric gives the same verdict as the standalone run against the analytic values and
the goldens (including the 20-seed Weibull distribution); the only failures are b5 at
40 m/s against OpenCourant, the documented gap that the standalone world fails
identically. Against the standalone run all 76 gated metrics pass, the largest
difference 0.03% (b3 wave speed 0.01%, b9 fragment speed and peak displacement
0.02-0.03%; all others 0.00%). How much of each scene PhysX actually moved matters for
reading this ("engine share": movable body-frames integrated by PhysX vs simulated in
impact islands):

| benchmarks | engine share | what the agreement shows |
|---|---|---|
| b2, b4, b8 gradual / sudden elastic, b9 low | no movable bodies | nothing for an engine to do (anchored structures) |
| b1 (broken pieces), b3 (free bar) | 100% PhysX | PhysX motion with the stress solve following it: failure loads, wave speed and reflections as standalone |
| b6 (spall) | 75% PhysX | the spalled layer flies under PhysX: same spall speed and back-face velocity |
| b5, b7, b8 sudden, b9 high (impacts, collapse, blast) | 0-0.3% PhysX | islands own the impacts, and collapse debris that fractures as it falls is re-solved as islands; shows the hand-off and the verification are clean, not PhysX's contact dynamics |

**Showcases, oracle vs PhysX** (bonds broken / fragments; "PhysX" is the share of
movable body-frames PhysX integrated; "redone" counts frames solved again because a
body PhysX moved fractured, see `engine.rs`). All 7 asserted behaviours also pass
through PhysX (`showcases_physx`).

| showcase | oracle | PhysX path | PhysX | redone |
|---|---|---|---|---|
| arch standing / keystone removed | 0 / 1, 14 / 16 | 0 / 1, 14 / 16 | —, 69% | 0, 1 |
| overhang, thin neck (creak 3.6 s, crack 6.8 s, snap 8.8 s) | 1 / 2 | 1 / 2, same times | 93% | 3 |
| supports removed one by one | 1 / 2 | 1 / 2 | 100% | 1 |
| car into brick / ductile wall | 158 / 75, 10 / 2 | 158 / 75, 10 / 2 | 0% | 0 |
| explosion beside two walls | 319 / 14 | 319 / 14 | 0% | 2 |
| brick house: car / blast / settlement | 234 / 64, 413 / 75, 633 / 243 | identical | 0%, 0%, 6% | 0, 3, 3 |
| floor dropped onto floor | 122 / 44 | 113 / 33 | 5% | 0 |

The floor drop is the one difference, and it is inside the oracle's own spread: the
slab breaks in a fracture cascade, and moving the falling slab by 1 mm or changing
the frame step changes the oracle's own result to 108-122 bonds and 32-44 fragments
(PhysX runs at three engine steps: 113-118 bonds, 33-37 fragments).

What PhysX moves: free flight, resting, stacking, slow contact and debris settling;
impacts between stiff breakable bodies are usually islands (a concrete-on-concrete
Hertz contact is shorter than a wave round trip), which is why the impact-heavy
scenes run at 0%. Sphere impactors are 42-vertex hulls of the sphere's volume while
PhysX drives them (the backend has no sphere shape); their impacts on breakable
bodies are islands, resolved against the true sphere.

**Known gap: friction.** The PhysX backend gives every shape one material (friction
0.25, no restitution) instead of the scene's. Fixing it needs a per-shape material in
the backend's `CreateShape`, which the native GPU pipeline shares.

## Status

All results below are from this revision (`cargo test --release`, `stress-ref check scenes golden`).

### Validation benchmarks

| # | benchmark | reference | result |
|---|---|---|---|
| 1 | single bond, tension and shear | analytic (1%), OpenCourant (5%) | failure force 0.01% / 0.00%, fracture energy 0.00%; OpenCourant 3.6% / 0.5% |
| 2 | chunked cantilever, N = 10/20/40 | Euler-Bernoulli (1%), exact discrete model (1e-6), OpenSees | tip 0.09-0.14%, root moment exact, frequency 0.07-0.15%; discrete model exact; OpenSees within 0.15% |
| 3 | bar impact, true and scaled E | analytic (1%), OpenCourant | wave speed 0.57%, reflection 0.46-0.66%, free-end doubling 0.55%; OpenCourant within 1.6% |
| 4 | sudden / gradual support loss | damped SDOF closed form (1%) | 1.9433 vs 1.9391 (0.22%); gradual 1.0014 |
| 5 | ram speed sweep 2/10/40 m/s | OpenCourant | 2 and 10 m/s: push-over as OpenCourant; speed lost at 10 m/s within 1%, at 2 m/s within 21% at a converged step but **bimodal** (41% at the default step, see gaps). **40 m/s: open** — ours punches a hole; OpenCourant pushes over at every mesh but that verdict rests on unconverged far-field cracking (see gaps) |
| 6 | planar spall | 1-D analytic, OpenCourant | spall occurs in both (central back layer separates), layer speed 3.61 vs 3.02 m/s (19%), back-face peak 4.48 vs 4.83 m/s (7%; 1-D 4.94), front face intact in both |
| 7 | masonry wall, 4 and 15 m/s | LMGC90, Kratos DEM, OpenCourant (ensemble) | breach in all; ball speed lost and debris speed inside the oracles' spread. The oracles disagree among themselves (4 m/s debris speed 0.07-3.7 m/s) |
| 8 | frame column removal, sudden / gradual | OpenSees | redistribution within 0.01-0.3% (gradual), sudden elastic peaks within 2.3%; first failure matches |
| 9 | pressure pulse on a panel, low / high / 20 Weibull seeds | OpenCourant | breach matches; fragment speed 3.46 vs 3.45 m/s; 20-seed mean 3.13 vs 2.88 m/s (8%), peak displacement within 24% |
| 10 | refinement study | itself | timestep halving: < 1% on waves, < 10% on fracture metrics; two chunk sizes: same breach (fragment speed differs, see gaps) |
| 11 | optimized vs reference | — | out of scope here (no CUDA solver yet); the reference provides `StressSolverApi`, snapshots and bit-identical runs to diff against |

### Showcases (`tests/showcases.rs`)

| scenario | ours (asserted) |
|---|---|
| same car, fast vs slow | b5: 40 m/s punches a hole, 2 m/s pushes the wall over |
| overhang on a thin connection | creak 3.6 s, crack 6.8 s, snap 8.8 s, slab swings down; a thick neck holds |
| sudden vs gradual column loss | b8: sudden collapses the frame that gradual removal leaves standing |
| supports removed one by one | first removal carried; after the second: creak 4.3 s, crack 6.6 s, give way 8.9 s |
| thick-wall hit (spalling) | b6: back face separates first, front face intact |
| explosion beside building | facing wall breaches, shadowed wall behind survives; the same wall alone breaches |
| steel vs brick, same car | ductile wall dissipates ~100x more and takes more car speed |
| floor falls onto floor below | dropped slab breaks the lower slab; the same weight as a static load does not |
| masonry arch, keystone removed | stands under self-weight; collapses without its keystone |
| same wall, two chunk sizes | same breach outcome (`dynamics::same_panel_two_chunk_sizes_same_outcome`) |
| brick house | stands at 25% utilization with stress paths around the openings; car breaches it; corner settlement brings the corner down |

### Solve-mode findings (for the optimized solver)

* **Implicit Newmark at the frame step** reproduces structural redistribution (b8
  within 5% of explicit, b2 statics exact) and is unconditionally stable, but cannot
  show dynamics faster than its step: b4's amplification (period 0.89 ms) is 1.26 at
  one step per period and converges as the step resolves it (1.73 at T/4.5, 1.92 at
  T/18, 1.936 at T/45). It also needs the joint's exact tangent: with a diagonal secant
  the Newton iteration diverged on cracked joints (contact patch, friction).
* **Penalty contact must not be split across solve rates**: substep penalty contacts
  with an implicit frame step (predictor/corrector) inject energy on impact (b5 at
  10 m/s). In the engine path the contact is PhysX's and enters as an impulse
  (`ImpactModel::VelocityCondition`), which is momentum exact in both modes; the
  struck beam breaks into 3 (pulse), 4 (velocity condition, explicit) or 6 pieces
  (velocity condition, implicit at the frame step).
* **A rigid-body engine's impulse is wrong for punch-through; impact islands fix it.**
  A 200 kg ram at 40 m/s into a free 528 kg concrete wall: the reference world, which
  resolves the contact on individual chunks while they fail, leaves the ram at
  35.0 m/s (~1000 N s handed over) and 6 fragments. PhysX resolves the rigid contact
  against the whole wall within one step: the ram ends near 11 m/s — the perfectly
  inelastic whole-wall collision, 200 x 40 / 728 = 11.0 — so the stress solver
  receives ~5.8x the real impulse. Contact attribution is right (each contact is
  reported on its chunk shape with its point and normal); the magnitude is wrong
  (`physx_impact_impulse_vs_reference_world`, run with `--ignored`, and
  `engine_world::ram_through_free_wall_matches_the_reference_world`):

  | engine path | ram after (m/s) | bodies |
  |---|---|---|
  | reference world (truth) | 35.03 | 6 |
  | PhysX 1/60 s, contacts fed to the stress solve | 10.93 | 22 |
  | PhysX 1/60 s, 4 correction passes | 10.93 | 25 |
  | PhysX 1/960 s | 11.52 | 42 |
  | PhysX 1/960 s, 4 correction passes | 13.26 | 63 |
  | **PhysX with stress-owned impact islands** (`World::with_engine`) | **35.05** | **6** |

  Correction passes (`PhysxDestruction::correction_limit`, the native
  `internalCorrectionLimit`: rewind, accept the trial's first failure front, re-solve)
  are momentum exact but cannot change the impulse: a rigid contact spends it in zero
  time, before the plug is a separate body. (An earlier version of this table showed
  corrections reducing fragmentation; that came from a reversed contact-normal sign,
  since fixed.) What works is letting the stress solve own the contact whenever the
  rigid assumption fails for it — see "PhysX as the rigid-body engine".
* **Mass scaling** keeps statics exact and cuts substeps, but on uniform chunks every
  chunk is critical: 4x the step on the 40-chunk cantilever costs 15x the mass.

### Performance (single thread, 4-core container)

| scene | before the speedups | convex-hull commit | bit-identical speedups | + step bound |
|---|---|---|---|---|
| b5 wall impact 10 m/s | 75.3 s | 77.8 s | 55.1 s | 57.7 s |
| b9 panel, high pressure | 14.3 s | 15.2 s | 11.5 s | 11.1 s |
| house hit by a car | 56.6 s | 48.6 s | 45.6 s | 32.0 s |
| floor drop | 28.2 s | 31.6 s | 24.2 s | 21.6 s |
| car into brick wall | 1.23 s | 1.83 s | 1.46 s | 1.47 s |

The bit-identical speedups reproduce every benchmark output bit for bit: substep
phases evaluated from the start-of-substep state then applied in order, allocation-free
contact sampling, bounding-sphere and separating-axis rejection of pairs that cannot
touch, pair-contact memory moved instead of cloned and rehashed. A deterministic
rayon pool is in place (results are identical for any thread count) but only takes
loops above 4096 items: on 4 cores every benchmark scene ran as fast or faster
serially. The per-chunk damped step bound (see "Solve modes") then gives house_car 30%
fewer substeps and the imported house and villa 25-28x larger substeps; other scenes
move by 0-2%.

A second round of bit-identical work (2026-10-10, `IMPROVEMENTS.md`, "Runtime")
reproduces every observation byte for byte over all 182 cells of
`scripts/verify_identical.py`. That covers every scene, both method sets, one thread
and all threads, and forced solve modes. CPU seconds at one thread on a 16-core
machine, base 0c71ffe13 -> 5161fb0aa:

| scene | default methods | `layer_contact,scaled_step_bound` |
|---|---|---|
| house settlement | 115.5 -> 111.8 s | 372.7 -> 165.7 s |
| house hit by a car | 16.0 -> 15.4 s | 252.3 -> 88.0 s |
| b9 panel, high pressure | 5.3 -> 5.1 s | 227.1 -> 69.6 s |
| b9 panel, quasi-static | 70.8 -> 47.6 s | 69.2 -> 46.7 s |
| floor drop | 11.6 -> 12.0 s | 203.5 -> 65.1 s |
| b5 wall impact 2 m/s | 80.8 -> 62.5 s | 161.4 -> 73.3 s |
| all one-thread cells | | 2353 -> 1178 s (2.0x) |

The gains come from:
- polytope clipping and field cells in reused buffers;
- a broadphase that indexes each cluster in two dimensions instead of testing every
  chunk pair of overlapping clusters;
- a dense index in the static solve;
- bond responses applied as they are computed.

After the accuracy fixes of e6fc3ade9 (a heavier layer contact), the same work against
that commit, one thread (`verify_identical.py --bench`): house hit by a car with
layer contact 312.6 -> 94.3 s, b9 panel 205.8 -> 59.3 s, floor drop 196.4 -> 64.9 s,
and 830.6 -> 296.6 s (2.8x) over the 22 bench cells.

Against cd112933c (proven methods default), which it reproduces bit for bit: 531.6 ->
209.8 s (2.5x) over the 22 bench cells; house hit by a car 298.8 -> 97.0 s, floor drop
81.5 -> 23.2 s.

`stress-ref run SCENE --profile` prints where a run spends its time.

### Scene packs (`scene_pack.rs`, `examples/import_pack.rs`)

The repository's authored destruction assets (`packs/`, copied from `blast/blast-stress-demo-rs/assets/scenes`)
import as oracle scenes: hull chunks (up to 64 vertices; the Voronoi-fractured v1 packs
have more and are refused), supports, and bonds whose patches are recomputed from the
touching faces (the asset's area where no shared face is found: 17 of 1652 in the
house). Physics comes from the calibrated materials, not the assets' gameplay limits
(their "reinforced concrete" breaks in tension at 11.5 MPa). Three assumptions, each
from the assets' own conventions or a design code:

* the ground is the assets' grade (`y = 0`, the demo's ground plane; footings are
  embedded below it) and chunks reaching grade bear on it as supports;
* reinforced-concrete joints carry the EN 1992-1-1 minimum reinforcement
  (`max(0.26 fctm/fyk, 0.0013)`, 0.22% with the calibrated concrete and steel);
* masonry units are joined by mortar, and a joint takes the asset's bond material.

Under self-weight (`tests/hulls.rs`): rig-portal, comp-wall-bay and the 2-storey house
stand; in the house the four mortar joints holding its solid steel rails
(470 kg each) to the brick crack and the rails stay on their steel and concrete
supports. The villa (1045 chunks) has no static equilibrium: its cantilevered stairs
and roof parapets exceed plain/minimum-reinforced capacity, and an explicit run cracks
them progressively (1, 1, 2, 10, 13, 15 broken bonds over the first 0.1 s, one piece).

### Known gaps

* **b5 at 2 m/s: ours and OpenCourant disagree on the speed the ram loses.** With the
  proven default methods (`layer_contact` and the rest) the result is converged:
  1.441, 1.423, 1.425, 1.418 m/s at Courant safety 0.5 to 1/16 (with the established
  contact it was not, the ram striking twice at times that depended on the step). It
  agrees with the momentum balance of the wall as a rigid body pivoting on its base
  (1920 kg, 2 m high, `I = m h^2 / 3`, struck at 1 m): a perfectly plastic impact
  leaves the ram at `1000 * 2 / (1000 + 2560) = 0.562` m/s, 1.438 lost; ours loses a
  little less (the wall cracks). OpenCourant loses 0.951 (the ram keeps 1.05 m/s while
  the wall's angular momentum allows its mid-height 0.37 m/s): only possible if the
  wall breaks up under the ram, and it does, with 805 broken cohesive elements at 2 m/s.
  Its elastic elements have no crushing (as at 40 m/s below), so the contact stress at
  the ram face is not capped. Failure mode (push-over) agrees. The gate fails; closing
  it needs the oracle rerun with a crushing law.
* **b5 at 40 m/s**: ours makes a local hole; OpenCourant pushes the wall over at 2, 3
  and 4 elements per chunk edge (a hole only at 1). The oracle mesh study
  (`golden_mesh/b5_wall_impact_v40/`) shows why this is not yet a usable verdict: the
  impact zone is fully shattered and converged at every mesh (ours detaches it too),
  but the far-field bending cracks that make it a push-over keep growing with
  refinement (9, 128, 348, 552 broken bonds) — the oracle's elastic chunks have no
  crushing or erosion, so the ram face sees ~190 MPa (concrete crushes at 30 MPa) and
  a pulse ~5x sharper than ours. The converged oracle quantity, ram speed lost
  ~6.4 m/s, is 37% above ours (4.0; 4.2 with `--feature crushing=off`), just outside
  the 30% gate. Closing this needs a crushing law in the oracle (pressure-capped
  chunks) and a contact-pulse comparison; it stays open.
* **Lateral release / Poisson coupling**: the chunk network has no Poisson effect, so
  spall breaks whole planes where the continuum keeps corner ligaments (b6), and peak
  panel displacement is 20-24% above OpenCourant (b9).
* **Fragment speed vs chunk size**: the breached panel at 0.05 m chunks throws
  fragments at 2.3 m/s vs 3.5 m/s at 0.1 m (two layers through the thickness resolve
  hinge crushing). Breach agrees; speed is reported, not gated.
* **Angular momentum** is conserved to 0.1-0.2% in a 40 m/s fracturing impact (the
  small-strain bond model's moments balance about undeformed centres); linear
  momentum to round-off.
* **Implicit mode in the standalone world** is for structural dynamics, blasts and
  debris; impactor impacts need the explicit solve or the engine path.
* **Debris management** (lifetimes, merging settled rubble) is an engine concern and
  not in the reference.

### Spec checklist → implementation → test

| spec item | where | verified by |
|---|---|---|
| Clusters with a bond graph, split into rigid bodies | `solver.rs` (`split_cluster`) | `conservation::momentum_is_conserved_through_impact_and_fracture`, `stress-physx` tests |
| Fragments inherit `v + w x r`; momentum through splits | `solver.rs` | `conservation::fragments_inherit_parent_rigid_velocity_field` |
| Bond stiffness from E, G, area, spacing | `bond.rs`, `scene.rs` (`with_derived`) | b2 (OpenSees, Euler-Bernoulli), b3 (wave speed) |
| Axial, shear, bending, torsion | `bond.rs` | `bond.rs` unit tests (torsion constant), b2, b8 |
| Per-material parameters | `material.rs` | `failure_laws` |
| Brittle, steel (ductile) and rebar bond types | `joint.rs` | `failure_laws::steel_absorbs_…`, `failure_laws::rebar_holds_…`, showcase car |
| Multi-level pre-fracture | `refine.rs`, `world.rs` | `refinement` (3 tests) |
| Gravity prestress | `statics.rs` | `dynamics::gravity_prestress_starts_structures_at_rest` |
| Anchors and supports with reactions | `world.rs` | b4, b8 (OpenSees reactions), `dynamics::debris_landing_…` |
| Rigid motion subtracted from chunk loads | `solver.rs` (`frame_loads`) | `dynamics::rigid_motion_is_subtracted_from_chunk_loads` |
| Contact: resting load + impact pulse of physical duration | `api.rs` (`ContactLoadFilter`) | `engine_api` (Hertz duration, resting load), `stress-physx` |
| Crush-capped impactors | `world.rs`, `api.rs` | `engine_api::crush_capped_…`, showcase car |
| Blast: pulse, falloff, incidence, shadowing, venting | `blast.rs`, `world.rs` | `blast.rs` unit tests, b9 (OpenCourant), showcase two walls |
| Debris landing loads structure dynamically | `world.rs` contact | `dynamics::debris_landing_…`, showcase floor drop |
| Chunk inertia in the stress solve | explicit dynamics | b3, b4, b6 |
| Wave speed `sqrt(E_eff/rho)` | | b3 (true and scaled stiffness) |
| Dynamic amplification on sudden release | | b4 (analytic 2x), b8 sudden vs gradual (OpenSees) |
| Reflection at free surfaces (spall) | | b3 reflection, b6 spall |
| Light damping | stiffness-proportional dashpots | energy ledger tests |
| Tension cutoff, Mohr-Coulomb, crushing | `joint.rs` | `failure_laws`, b1 |
| Buckling cap | `joint.rs` | `failure_laws::buckling_caps_…` |
| Dynamic increase factor | `material.rs` (fib MC2010), `joint.rs` | `failure_laws::dynamic_increase_factor_…`, `features::rate_effects_off_…` |
| Fracture-energy softening (resolution independent) | `joint.rs` (`damage_increment`) | b1 energy, `failure_laws::fracture_energy_per_area_…`, `joint.rs` unit tests |
| Time-dependent damage under sustained overload (creak, crack, give way) | static fatigue, `material.rs` / `joint.rs` | `failure_laws::sustained_overload_…` (closed-form lifetime within 1%), showcases overhang and supports |
| Weibull strengths | `material.rs` | `failure_laws::weibull_…`, b9 20-seed distribution |
| Same-step cascade | `statics.rs` | `failure_laws::same_step_cascade_…` |
| Stored energy of broken bonds released | ledger `split_release`, `softening_overshoot` | `conservation::fracture_never_gains_energy_…` |
| Per-bond damage exposed | `api.rs` (`BondReport`), `SolverEvent` | `engine_api::engine_coupled_fracture_reports_…` |
| Converges with timestep and chunk size | | `dynamics::converges_as_the_timestep_shrinks`, `dynamics::same_panel_two_chunk_sizes_…`, `refinement` |
| No energy gain; dissipation accounted | `EnergyLedger`, `ContactLedger` | `conservation` |
| Momentum through fracture | | `conservation`, `stress-physx` |
| Same outcome at different frame rates | | `dynamics::same_outcome_at_different_frame_rates` |
| Stiffness scaling compromise | `sim.stiffness_scale`, `min_stiffness_scale`, or `sim.mass_scaling_dt` | b3 scaled, `dynamics::stiffness_scaling_…`, `solve_modes::mass_scaling_…` |
| Activity-based solving | `SolveMode::Adaptive`, `SolveMode::Implicit` | `dynamics::solve_modes_agree`, `solve_modes` |
| Model switches (runtime) | `sim.features` | `features` (9 tests) |
| Two-scale structure | `refine.rs` | `refinement` |
| Debris management (lifetimes, merging rubble) | not in the reference: an engine concern | — |

### Videos

`scripts/render_videos.sh` renders every benchmark, showcase and comparison into
`videos/` (47 MP4s, ~90 MB, not committed; about 1.5 h on 2 cores). Comparisons:

| video | what it shows |
|---|---|
| `lead_fast_vs_slow` | same ram at 40 vs 2 m/s: local failure vs the wall pushed over |
| `lead_sudden_vs_gradual` | sudden column loss collapses the frame that gradual removal leaves standing |
| `feature_inertia` | explicit dynamics vs quasi-static (no inertia) on the 40 m/s ram |
| `feature_stiffness_scaling` | true E, `E x 0.01` and mass scaling to 4x the step on the 40 m/s ram: 114, 63 and 3 pieces — both compromises change the failure (mass scaling turns it into a punched plug; on uniform chunks it is not selective) |
| `feature_implicit` | explicit vs implicit at the frame step on sudden column loss |
| `feature_shadowing` | blast beside two walls: with shadowing the back wall survives, without it breaks up |
| `feature_softening` | fracture-energy softening vs a threshold model |
| `feature_crack_contact` | arch without its keystone, with and without the cracked joints' contact patch |
| `feature_weibull` | Weibull vs uniform bond strengths on the breached panel |
| `s_overhang_thin_fatigue` | overhang with and without static fatigue (snaps at ~9 s vs holds) |
| `b5_wall_impact_v10_rate`, `b7_masonry_v15_modes` | rate effects on/off; explicit vs adaptive |
| `rubble_<scene>` | the standalone world (left) vs the same scene with rubble's AVBD engine moving the bodies (right), the stress solve owning impact islands; `RUBBLE=0` skips them. A scene the coupled world keeps entirely in stress-owned islands renders identically in both (`s_arch_keystone_removed` up to 1 s does) |

### Viewer

`stress-viz view <scene> [options]` opens a window on a live run (cargo feature
`viewer`, on by default). The simulation runs on its own thread (`--sim-threads`,
default 4) as fast as it can or capped at real time, and records every frame; the
window shows the live head or any recorded frame. Projectiles are steel spheres fired
where you click; firing on a past frame branches the run there (restored from the
nearest checkpoint and re-simulated deterministically with the shots logged so far;
with an external engine there are no checkpoints and the branch re-simulates from the
start). `--solver-mode explicit|soft|implicit|mass-scaling|adaptive` (M in the window)
trades fidelity for speed; the rubble coupling runs only explicit and adaptive.
`--renderer auto|gpu|cpu` picks the rasterizer, as for videos.

| input | does |
|---|---|
| drag / right- or shift-drag / wheel | orbit / pan / zoom |
| click | fire a sphere (`[` `]` speed, `-` `=` size) |
| drag the timeline | scrub |
| Space, arrows, L | play/pause the replay, step a frame, back to the live head |
| `,` `.` | replay speed |
| P, T, R | pause the simulation, real-time cap, restart |
| M, E | solver mode, engine (standalone / rubble; restarts) |
| V, A, K, C | view mode, velocity arrows, cracks, fit camera |
| H, Q | help, quit |
