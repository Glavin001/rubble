# Cross-engine rigid-body benchmark

Compares rubble (`engine/`) with **three-avbd** (GPU, WebGPU), **rapier** (CPU) and
**PhysX** (CPU and GPU) on the same scenes, timed by the same protocol, and judged for
quality by one engine-neutral judge. On macOS, PhysX's GPU path is NVIDIA's CUDA GPU
module running on Apple Metal through CuMetal (sibling repos `../PhysX` and
`../cuda-metal`). A time counts only if the run is physically
right: an engine that is fast because it dropped contacts, slept the scene, or let the pile
fall through the floor does not win.

```sh
# from the repo root; needs Rust stable, Node ≥ 22.6 (24 recommended), slangc for rubble
python3 bench/run.py --list                # scenes, engines, what is available here
python3 bench/run.py --smoke               # ~5 steps per run: proves everything runs (seconds, ~no GPU)
python3 bench/run.py --dry-run --group webgpu --scenes small,steady,paper   # the plan, run nothing
python3 bench/run.py --group webgpu        # rubble WGSL vs three-avbd, small + steady tiers
python3 bench/run.py --group cpu --scenes small,steady,paper,large --repeat 3
python3 bench/run.py --sweep               # also each engine's iteration sweep (quality vs time)
python3 bench/run.py report bench/results/<dir>   # rebuild report.md from existing runs
```

Results land in `bench/results/<UTC time>/`. Each run gets a result JSON, a pose dump and a
judge JSON. `report.md` ranks the engines per scene and has a headline table of the
fastest *passing* configuration of each engine.

## Layout

| Path | Role |
|---|---|
| `scenes/suite.ts` | The scenes, how long each runs, groups and quality gates |
| `scenes/export.ts` | Builds each scene with the pinned three-avbd builders, writes `scenes/out/<id>.scene.{json,bin}` |
| `tools/fracture-export` | Shardwright fracture GLBs → one scene per hierarchy level, through rubble's own importer (`engine/crates/engine-fracture`) |
| `crates/bench-core` | Scene reader, the `Engine` trait, the timing protocol, result/pose writers, shared CLI |
| `crates/bench-judge` | Quality metrics from pose dumps (parry3d for contacts), gate evaluation |
| `adapters/rubble` | rubble on native wgpu (Slang → SPIR-V → WGSL) or native Metal |
| `adapters/three-avbd` | three-avbd's `GpuSolver3D` headless through Dawn (Node) |
| `adapters/rapier` | rapier3d 0.36 with rayon |
| `adapters/box3d`, `adapters/jolt` | Box3D and Jolt, built from pinned sources (`third_party/fetch.sh`) through small C/C++ shims |
| `adapters/physx` | PhysX 5 through a C++ shim: the CuMetal 5.10 package (macOS CPU + GPU), the official SDK 5.x (CPU + CUDA GPU), or vendored 5.1 (CPU); see its README |
| `run.py` | Builds, runs every engine × scene serially, judges, writes the report |

## Engines

| run name | engine | where it runs | timed modes |
|---|---|---|---|
| `rubble-wgpu` | rubble, Slang → WGSL on wgpu | GPU (Metal/Vulkan/DX12) | batch, sync |
| `rubble-metal` | rubble, Slang → MSL on Metal (macOS) | GPU | batch, sync |
| `three-avbd` | three-avbd `GpuSolver3D`, WGSL on Dawn | GPU | batch, sync |
| `rapier-1t` / `rapier-mt` | rapier3d, one thread / many threads | CPU | batch |
| `box3d-1t` / `box3d-mt` | Box3D (Erin Catto), soft-step solver, one thread / many (internal scheduler) | CPU | batch |
| `jolt-1t` / `jolt-mt` | Jolt Physics 5.6, one thread / many (job system) | CPU | batch |
| `physx-cpu-1t` / `physx-cpu-mt` | PhysX TGS, 1 / many worker threads | CPU | batch |
| `physx-gpu` | PhysX with GPU dynamics and GPU broadphase | CUDA GPU on NVIDIA; on macOS the same CUDA module on Metal via CuMetal | batch, sync |
| `rubble-cuda` | rubble's CUDA path (`engine/crates/engine-cuda`) | *pending: engine-cuda emits CUDA source but has no host yet; on macOS it would run through CuMetal like PhysX* | — |

### Comparisons

`--group` picks one; the report has a headline table per group, comparing rubble with
the best passing configuration of each other engine.

| group | rubble | against | what it answers |
|---|---|---|---|
| `webgpu` | `rubble-wgpu` | `three-avbd` | Is our WGSL path faster than the best open WebGPU AVBD? |
| `metal` | `rubble-metal` | `three-avbd` (Dawn on Metal), `physx-gpu` (CUDA on Metal) | Our native GPU path vs every GPU engine on this Mac |
| `cuda` | `rubble-cuda` (pending) | `physx-gpu` | Like-for-like CUDA, once rubble has a CUDA host |
| `cpu` | — | `rapier`, `box3d`, `jolt`, `physx-cpu` (1 thread and many) | CPU baselines; rubble's GPU rows are read against these |

### Scheduling GPU time

Other people may be using the GPU, so run the real thing in a reserved window. Two modes
make that safe:

- `--smoke` runs every adapter for about 5 steps per scene. It proves the builds work,
  the scene loads bit-identically (checkpoint 0) and stepping stays finite, and it makes no
  timing or quality claim. All engines × the small and steady tiers take about 20 s.
- `--dry-run` prints every command a run would execute and how many touch the GPU. Paste
  them into a scheduler if you want per-run control.

Runs are strictly serial, so a benchmark never competes with itself. The 1-minute load
average before each run is recorded, and the report flags anything that started above 2.0.

"Many threads" is the performance-core count on Apple silicon (12 on an M3 Max), because
its efficiency cores make a fork-join step slower. Elsewhere it is every logical core. The
result records the count.

The ring-110k row uses rubble's own production default for that scene (persistent
contacts, `engine/crates/engine-webgpu/README.md`). It is the only per-scene setting
anywhere, and the report shows it.

## Protocol

Every adapter does exactly this (`crates/bench-core/src/protocol.rs`; the three-avbd adapter
mirrors it line for line in TypeScript):

1. **Build** the scene from the scene file. Build time is reported, not compared.
2. **Checkpoint 0:** read the state back. The judge compares it with the scene file, so an
   adapter that mis-loads a body fails before any timing is trusted.
3. **Lead-in:** `lead` untimed steps (three-avbd's own lead per scene: 30–180).
4. **Timed window** (three-avbd's `compare`): 2 rounds × (3 untimed steps, drain, then 5
   batches × 10 steps). A batch runs from the first step's submission to the last step's
   completion, divided by 10.
   - **batch** mode queues all 10 steps and then drains, so a GPU engine may overlap CPU
     encoding with GPU work, as a game loop does.
   - **sync** mode drains after every step and records each step's latency.

   **wall** = the smaller of the two round medians (three-avbd's statistic). The report
   also keeps every batch, the mean, the max and per-step p50/p95.
5. **Tail:** untimed steps on to the scene's last checkpoint, reading the state back at
   each one. Checkpoints never fall inside the timed window; the exporter enforces this.

Runs execute one at a time, one process each, in scene-major order with engines
interleaved, so slow drift (thermals, background work) is shared. The 1-minute load
average before each run is recorded, and the report flags runs that started above 2.0.
Use `--repeat N` to get a median and spread.

## Fairness: what is held equal, and what cannot be

**Held equal**

- **Initial state, bit for bit.** Bodies come from the pinned three-avbd builders (the
  functions its own bench calls), rotated to Y-up by a proper rotation, stored as f32
  with a sha256. Every adapter loads the same file, and checkpoint 0 is verified per run.
- **Mass properties** are set explicitly from the file (uniform-density solids, three-avbd's
  formulas), never recomputed by an engine.
- **Physics inputs:** dt 1/60, gravity −10, friction per body, restitution 0, no linear or
  angular damping (PhysX's default 0.05 angular damping is turned off), no velocity
  clamping (PhysX max angular velocity raised), CCD off everywhere (AVBD has none).
- **Joints:** hard ball joints at the scene's anchors; joined bodies don't collide (as in AVBD).
- **Sleeping is off for every engine.** three-avbd cannot sleep, and a sleeping scene
  measures nothing. rubble's sleep timer is pushed out of reach; rapier and PhysX bodies
  are created unable to sleep. `--sleep on` exists for a separate "native behaviour"
  comparison and is never mixed into the headline.
- **The measurement** is the same code (or a line-by-line port), the same windows and the
  same statistic. GPU time includes completion. Nothing is read back inside the window.

**Not comparable, so not equalised: solver iterations.** Four AVBD iterations, four TGS
position iterations and four rapier substeps are different amounts of work and buy
different accuracy. Equal knob values would be a false equality. So:

- each engine runs at its **default** (AVBD engines at the scene's reference count, as
  three-avbd benchmarks them; rapier `num_solver_iterations = 4`; PhysX TGS 4 position /
  1 velocity);
- `--sweep` also runs each engine across its own iteration range (rapier/PhysX 1–16, AVBD
  ½× and 2× the reference), plus named production variants: PhysX `depen10` caps
  `maxDepenetrationVelocity` at 10 m/s. PhysX leaves it unbounded by default and then
  launches bricks out of deep impacts; rapier bounds its corrective velocity by default;
- the **headline counts only runs that pass every quality gate**, and takes each engine's
  fastest passing configuration. An engine is compared at the cost of getting the scene
  right, which is the question a user actually has.

**Known asymmetries** (also recorded in each result's `engine.config`)

- *Contact margins differ by design:* AVBD collision margin 1 cm, PhysX `contactOffset`
  2 cm (default), rapier prediction distance 2 cm (normalized default). The judge measures
  penetration on exact shapes, so it can see the effect.
- *Friction combine:* three-avbd and rubble use sqrt(a·b). rapier and PhysX have no
  geometric mean and use the average. The suite's pairs are 0.5/0.5 and 0.5/0.6
  (0.548 vs 0.55).
- *rubble's native hosts read a 64-byte counter block after every step*, so batch mode
  cannot pipeline them; their batch and sync numbers are the same measurement. Its browser
  host does pipeline (`engine/crates/engine-webgpu/demo.html?bench=only`), and so does
  three-avbd here; compare their *sync* numbers for latency parity.
- *Dawn vs the browser:* three-avbd runs headless on Dawn from the `webgpu` npm package.
  The newest build that loads on macOS 14 is 0.2.12 (three-avbd pins 0.6.1, which needs
  macOS 26; `BENCH_DAWN=pinned` uses it). On this M3 Max its wall times match the Chrome
  numbers recorded in `engine/docs/LOCAL_ORACLE_BASELINE.md` (wall smash 2.95 vs 2.86 ms,
  ring 110k 10.8 vs ~10 ms).
- *PhysX versions:* on macOS the CPU and GPU runs both use PhysX **5.10** from the CuMetal
  package (`../PhysX/out/install/macos-cumetal/release`, profile `rigid-demo-experimental`;
  override with `PHYSX_CUMETAL_DIR`). It is one build, switched by `--set gpu=on|off`.
  Without that package, the adapter falls back to PhysX 5.1 vendored by `physx-sys` (CPU
  only). Linux/Windows use an official 5.x SDK (`PHYSX_SDK_DIR`).
- *PhysX GPU on Metal is a translation:* CuMetal compiles PhysX's CUDA kernels to Metal,
  with raw device addresses (`CUMETAL_USE_METAL_DEVICE_ADDRESSES=1`, which removes
  cross-stream concurrency) and the package's restricted rigid-body profile. It measures
  PhysX's GPU algorithms on this GPU, not what PhysX achieves on an NVIDIA card. The `cuda`
  group on Linux answers that. Under GPU dynamics PhysX does not publish its contact-pair
  statistics (`pairStatsValid: false`); the judge measures contacts from poses anyway.
- *Convex hulls are rebuilt per engine from the same vertices* (fracture tier):
  - **rubble** uses Shardwright's baked topology (≤ 64 vertices).
  - **rapier** builds `convex_hull`s.
  - **PhysX** cooking drops near-coplanar and non-extreme points. For example, house L1 goes
    from 43,192 to 27,728 vertices, with no input point more than 1.8 mm outside its cooked
    hull. On the GPU, 39 of house L1's 2,253 hulls are rejected for GPU cooking ("oblong"), and
    PhysX collides those on the CPU.
  - **three-avbd** caps hulls at 32 vertices, and replaces the plane with its usual 1 m ground
    box, top face on the plane.

  Each result records what changed in `engine.config` (`hullCooking*`, `hullResult`).
- *Joints in rubble* exist only on the Metal backend so far. `rubble-wgpu` reports
  joint scenes as n/a. The adapter makes them hard (stiffness unbounded, capped at `kMaxN`),
  as three-avbd's are; `--set joint_stiffness=1e5` gives rubble's demo setting.

## Scenes

From three-avbd's `bench-cases.ts` (same builders, iterations and lead), so every
number lines up with its published and local results.

| id | tier | bodies | joints | ref. iters | steps | gates (all scenes: no NaN, nothing escapes) |
|---|---|---:|---:|---:|---:|---|
| `wall_smash_2k` | small | 2,002 | 0 | 10 | 240 | ball breaks through; penetration p99 ≤ 3 cm |
| `chain_mail_1.6k` | small | 1,602 | 3,120 | 10 | 300 | joint gap p99 ≤ 5 cm; the net holds the 335 kg ball off the ground; penetration p99 ≤ 5 cm |
| `brick_ring_9k` | small | 9,132 | 0 | 4 | 240 | ball crosses the ring and stays on the floor; no body above 60 m/s |
| `brick_gables_6.7k` | small | 6,723 | 0 | 3 | 240 | unhit walls stand (mean \|Δy\| ≤ 0.2 m, max ≤ 0.6 m); both balls roll ≥ 30 m |
| `jointed_drop_6k` | small | 6,025 | 12,484 | 10 | 300 | joint gap p99 ≤ 5 cm; penetration p99 ≤ 5 cm; the plates really fall (mean drift ≥ 2 m) |
| `brick_pyramid_12` | steady | 651 | 0 | 10 | 300 | pyramid stands: horizontal drift ≤ 25 cm, mean \|Δy\| ≤ 15 cm, penetration p99 ≤ 2 cm |
| `brick_pyramid_30` | steady | 9,456 | 0 | 10 | 300 | as above |
| `settled_pile_32k` | steady | 32,001 | 0 | 4 | 600 | pile at rest (mean speed ≤ 0.2 m/s) at 10 s; penetration p99 ≤ 3 cm |
| `box_columns_100k` | steady | 100,001 | 0 | 4 | 300 | no column leans/slides (≤ 10 cm), settles ≤ 25 cm, penetration p99 ≤ 2 cm |
| `brick_ring_110k` | paper | 110,442 | 0 | 4 | 480 | flanks stand (mean \|Δy\| ≤ 0.3/0.5/1.6 m at 3/4/8 s); ball crosses both walls on the floor |
| `jointed_drop_34k` | paper | 34,097 | 71,064 | 10 | 300 | as jointed drop 6k |
| `brick_gables_506k` | large | 505,923 | 0 | 3 | 240 | as gables 6.7k |

Gate bounds are physical envelopes, not "matches three-avbd". Where rubble's own
regressions already define one (ring, gables), the gate uses that. They were then checked
against three-avbd as the reference implementation. It passes all of them except
`settled_pile_32k`, where about 5,700 boxes are pushed through the 1 m ground slab at 4
iterations. three-avbd's own builder does this too, without our scene round-trip.

## Fracture scenes (Shardwright)

The `fracture` tier is generated from the Shardwright assets bundled with rubble's fracture
demo (`engine/crates/engine-webgpu/fracture/assets`): a masonry arch and a veneer house,
one scene per pre-fractured detail level.

```sh
cd bench && cargo run --release -p bench-fracture-export   # run.py --export also does this
```

| id | fragments | hulls | compound bodies | three-avbd | rubble-wgpu |
|---|---:|---:|---:|---|---|
| `shardwright_arch_l0` | 23 | 23 | 0 | runs | n/a |
| `shardwright_arch_l1` | 165 | 392 | 106 | n/a | n/a |
| `shardwright_arch_l2` | 171 | 392 | 109 | n/a | n/a |
| `shardwright_arch_l3` | 392 | 392 | 0 | runs | n/a |
| `shardwright_house_l0` | 131 | 163 | 6 | n/a | n/a |
| `shardwright_house_l1` | 567 | 2,253 | 544 | n/a | n/a |

House levels 2 and 3 (3,158 and 5,905 fragments) are skipped. Their source hulls include
zero-volume ones that rubble's importer rejects ("recook the source collision asset"). The
exporter records the reason in `scenes/out/fracture-index.json`. Recook in Shardwright and
re-export to add them.

- **Same geometry everywhere.** The exporter calls rubble's `engine_fracture::build_scene`,
  the importer rubble's GPU uses. Each fragment is a body at its centre of mass, turned to its
  principal axes, with the asset's mass and principal inertia, and its baked convex hulls'
  vertices (≤ 64 each) in that frame. rubble rebuilds the same level from the GLB to keep its
  baked hull topology. Every other engine builds hulls from the same vertices: rapier
  `convex_hull`, PhysX convex cooking, three-avbd `convexHull`. Checkpoint 0 is verified for
  all of them.
- **The scene:** the object stands on a ground plane at y = 0 with fragments **unbonded** at
  their baked poses. Bonds are inspection metadata in rubble too, not constraints. One scripted
  wrecking ball (6 % of the extent across, half the object's mass) is launched along −z at the
  mass-weighted height of the fragments in its path (an arch's crown, not its opening), timed
  to arrive just as the 1 s lead-in ends. dt 1/120 s, gravity −9.81, 8 AVBD iterations and
  α 0.95 are rubble's fracture defaults. Friction is 0.5 everywhere, as in rubble's fracture
  path, which does not read the Shardwright materials yet.
- **Gates (provisional):** no NaN, nothing through the floor, no fragment at 3× the ball's
  launch speed, penetration p99 ≤ 3 cm, and the ball travels at least a quarter of the
  object's extent. There is no "stands before impact" gate. Unbonded pieces that only bonds
  held up rightly fall, so the 0.4 s checkpoint records how much each engine settles rather
  than judging it.
- **Engines:** rubble runs these on its native Metal fracture host
  (`engine-viewer` `FractureSim`, the same `fracture_step.slang` kernels as the browser demo;
  `rubble-metal` picks it automatically for hull scenes). Its native wgpu host has no
  fracture kernels, so `rubble-wgpu` is n/a. Build without `engine-viewer` with
  `cargo build -p bench-rubble --no-default-features`; hull scenes then fail with a clear error.
  three-avbd has no multi-hull bodies, so it runs only the single-hull levels.

## The judge

`bench-judge --result run.json` reads only the pose dumps and the scene file:

- `initial`: checkpoint 0 vs the scene (position ≤ 0.1 mm, rotation ≤ 1 mrad, velocity ≤ 1 mm/s).
- `nonfinite`, `escaped`: NaN/Inf; dynamic bodies through the floor (under the ground top
  − 1 m while over the ground) or 200 m above the scene. `off_edge` counts bodies pushed past
  the ground's edge, which is legitimate motion, so it is reported but not gated.
- `pen_max/p99/mean`, `touching_pairs`: exact shape-shape depth (parry3d) for every pair
  within 5 mm, joined pairs excluded. A compound counts once per body pair, at its deepest
  hull pair; the ground plane is a half-space.
- `joint_gap_*`: world distance between each joint's two anchors.
- per group (`dynamic`, `ball`, `flank`, `unhit`, ...): drift, \|Δy\|, horizontal drift,
  mean/min/max position, speed, displacement; plus total kinetic energy.

## File formats

- **Scene** `<id>.scene.json`: metadata, protocol, groups, gates, sha256 of the bin, and
  `source` (builder, or asset file + level).
  `<id>.scene.bin`, little-endian: `u32 'RBSC', version 1, bodies, joints`, then per body
  24 words (`shape` 0 box / 1 sphere, `flags` bit 0 static, half extents ×3, mass,
  principal inertia ×3, friction, position ×3, quaternion xyzw, linear velocity ×3, angular
  velocity ×3, restitution), then per joint 12 words (`a`, `b`, kind 0 = ball, flags bit 0 =
  collide, anchor A ×3, anchor B ×3 in body frames, linear stiffness = ∞, angular = 0).
  Y-up, SI units. **Version 2** (fracture scenes) has a 6-word header (`…, hulls, vertices`).
  It adds shape 2, a compound of convex hulls (words 2–4: first hull, hull count, bounding
  radius), and shape 3, a static ground plane (through the body position, normal = body +y).
  After the joints come a hull table (first vertex, vertex count) and the vertices (f32 ×3,
  in the body frame).
- **Result** `<run>.json` (`rubble-bench-result` v1): machine, scene id + sha, engine
  (version, backend, device, threads, solver, iterations and their meaning, every
  non-default setting), protocol, build time, timing (rounds, wall, step stats, trace of
  untimed chunks), counters at each checkpoint.
- **Poses** `<run>.poses.bin`: per checkpoint `u32 'RBPS', step, bodies, 13`, then per body
  position ×3, quaternion ×4, linear velocity ×3, angular velocity ×3 (f32, scene order).
- **Judge** `<run>.judge.json`: metrics per checkpoint, each gate's value and status, `passed`.

## Adding an engine

Implement `bench_core::Engine` (`info`, `step`, `drain`, `read_state`, `counters`) and
call `bench_core::main_with(build)`. Then everything else is shared: CLI, protocol and
output. Load mass properties from the scene, turn off damping/CCD/sleep, record every
non-default setting in `EngineInfo::config`, and add the engine to `ENGINES` in `run.py`.
A non-Rust engine can mirror `adapters/three-avbd/run.ts`. Check that checkpoint 0
passes the judge's `initial` test before trusting any timing.

## Cross-check in the browser

Same-browser numbers remain the fairest rubble-vs-three-avbd comparison for the WebGPU
product: `engine/crates/engine-webgpu/demo.html?scene=ring110k&bench=only` and
three-avbd's `bench3d.html` run the same 2×5×10 protocol in Chrome. This suite adds rapier
and PhysX, quality gates, and native hosts.
