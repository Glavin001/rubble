# bench-jolt: Jolt Physics 5.6 adapter (CPU)

`bench-jolt` runs a benchmark scene on [Jolt Physics](https://github.com/jrouwe/JoltPhysics).
The Rust binary (`src/main.rs`) implements `bench_core::Engine`. A small C++ shim
(`src/jolt_shim.cpp`) does the Jolt work through an `extern "C"` API. It creates the world from
flat body/joint/hull arrays plus a settings struct, steps it, reads the state in scene order,
reads counters, and destroys the world. This mirrors `adapters/physx`.

## Build

```sh
cd bench
third_party/fetch.sh                      # clones JoltPhysics at v5.6.0 (e77f175595e6) into third_party/JoltPhysics
cargo build --release -p bench-jolt -j6   # needs cmake (Homebrew) and the Xcode command line tools
./target/release/bench-jolt --scene scenes/out/wall_smash_2k.scene.json --out results/dev/jolt-wall.json --threads 1
./target/release/bench-judge --result results/dev/jolt-wall.json --scene scenes/out/wall_smash_2k.scene.json
```

`build.rs` does three things:

1. **Configures and builds Jolt with CMake.** It configures `third_party/JoltPhysics/Build`
   (override the source with `JOLT_DIR`) in `$OUT_DIR/jolt-build`, then builds only the `Jolt`
   target. The build uses `JOLT_BUILD_JOBS` jobs, defaulting to cargo's `-j`. The options are:
   - `CMAKE_BUILD_TYPE=Distribution`, `INTERPROCEDURAL_OPTIMIZATION=ON`
   - `DOUBLE_PRECISION=OFF`, `OBJECT_LAYER_BITS=16`, `CROSS_PLATFORM_DETERMINISTIC=OFF`, `USE_ASSERTS=OFF`
   - debug renderer and profiler off in every configuration
   - every `TARGET_*` off, `GENERATE_DEBUG_SYMBOLS=OFF`, `ENABLE_ALL_WARNINGS=OFF` (no `-Werror` surprises from newer clang)
   - the 5.5+ GPU compute back-ends off: `JPH_USE_MTL/VK/DX12/CPU_COMPUTE=OFF`
   - static library

   The result is `libJolt.a`, which holds LLVM bitcode objects because of IPO. Apple's `ld64`
   LTO-links them into the Rust binary.
2. **Compiles the shim with the library's own compile flags.** It reads the compile line CMake
   used for `PhysicsSystem.cpp` from `compile_commands.json`. It copies every `-D` and the
   code-generation flags that matter (`-std`, `-fno-rtti`, `-fno-exceptions`, `-ffp-contract`,
   `-faligned-allocation`) onto the shim's `cc` build. On arm64 macOS the defines are
   `JPH_OBJECT_LAYER_BITS=16 JPH_OBJECT_STREAM NDEBUG`. NEON is detected from `__aarch64__` in
   `Jolt/Core/Core.h`, and there are no instruction-set defines on ARM.
3. **Checks at runtime.** `VerifyJoltVersionID()` runs before anything else, so a feature-define
   mismatch fails world creation instead of crashing. Every result records the defines and the
   CMake options in `config.defines` / `config.build`.

## Fairness settings

Every value below is recorded in the result's `engine.config`.

- **Solver**: Jolt's sequential-impulse solver. `--iterations` sets
  `PhysicsSettings::mNumVelocitySteps` (default 10, Jolt's default). `mNumPositionSteps`
  defaults to 2, also Jolt's default (`--set pos_steps=N`). `Update(dt, collisionSteps = 1)`
  with the scene dt.
- **Other `PhysicsSettings`** are Jolt's defaults and are recorded:
  - baumgarte 0.2
  - penetration slop 0.02
  - speculative contact distance 0.02
  - max penetration distance 0.2
  - manifold tolerance 1e-3
  - deterministic simulation, warm start, body-pair cache, manifold reduction, large-island splitter and active-edge checks all on
- **Threads**: `threads` = `--threads N` (0 = all logical cores) is the **total** number of
  threads stepping. That is the calling thread plus
  `JobSystemThreadPool(cMaxPhysicsJobs = 2048, cMaxPhysicsBarriers = 8, N - 1)` workers. Jolt's
  calling thread executes jobs while it waits on a barrier, so `--threads 1` means zero workers
  and everything runs on the main thread (`--set jobs=`, `barriers=`).
- **Sizing** (from the scene's body count n):
  - max bodies: n + 64
  - max body pairs: clamp(16n, 65536, 2^26) (`--set max_pairs=N`)
  - max contact constraints: clamp(16n, 65536, Jolt's 32-bit limit) (`--set max_contacts=N`)
  - temp allocator: `TempAllocatorImpl`, max_contacts × (cMaxConstraintSize + 4) + 4 KiB·n + 256 MiB
    (`--set temp_mb=N`). Jolt takes the whole contact-constraint buffer from it every step. Untouched pages are never committed.

  There is no malloc fallback in the hot path; if the temp allocator runs out, Jolt aborts. If
  `Update` returns any `EPhysicsUpdateError` (`ManifoldCacheFull`, `BodyPairCacheFull`,
  `ContactConstraintsFull`, which means contacts were dropped), `step()` fails the run and
  names the knob to raise. `counters.updateErrors` / `overflow` record the sticky bits.
- **Layers**: two object layers (non-moving, moving) mapped to two broadphase layers, with
  the standard HelloWorld filters (static–static never tested). `OptimizeBroadPhase()` runs
  once after all bodies are added (batched `AddBodiesPrepare/Finalize`, dynamics activated).
- **Bodies**:
  - **Mass properties**: `EOverrideMassProperties::MassAndInertiaProvided` with the scene mass
    and `Mat44::sScale(principal inertia)`. The body frame is the principal frame.
  - **Damping**: linear and angular damping 0. Jolt's defaults are 0.05 each.
  - **Velocity limits**: `mMaxLinearVelocity` = `mMaxAngularVelocity` = 1e4. Jolt's defaults
    are 500 m/s and 0.25·π·60 ≈ 47 rad/s (`--set max_lin_vel=`, `max_ang_vel=`).
  - **Other body settings**: `EMotionQuality::Discrete` (no CCD), gravity factor 1, initial
    linear and angular velocity from the scene.
  - **Statics**: scene bodies with `is_static` are `EMotionType::Static` in the non-moving layer.
- **Friction / restitution**: per body, from the scene. Jolt's default combine for friction is
  the geometric mean sqrt(a·b), the same as three-avbd and rubble (PhysX and rapier average).
  Restitution combines by max and is 0 in every scene.
- **Sleeping**: off by default (`PhysicsSettings::mAllowSleeping = false` and per body
  `mAllowSleeping = false`). `--sleep on` uses Jolt's defaults (0.5 s, 0.03 m/s).
  `counters.activeBodies` confirms that every dynamic body stays active.

## Shapes, and how they differ from the scene's exact shapes

- **Boxes**: `BoxShape(half extents, r)` with convex radius
  r = min(0.05, 0.1 × smallest half extent). The 0.05 is Jolt's `cDefaultConvexRadius`
  (`--set convex_radius=X`, `convex_radius_frac=F`). A Jolt box keeps its outer size; the
  radius only rounds edges and corners (by up to r). A radius > 0 lets GJK resolve shallow
  contacts without EPA, which is how Jolt is meant to be used. The clamp keeps small
  boxes (and thin hull shards) from being rounded away. Identical boxes share one shape.
- **Spheres**: `SphereShape(r)`, exact.
- **Convex hulls** (format v2): `ConvexHullShapeSettings(points, r)` with r chosen by the same
  rule from the hull's AABB. Jolt may shrink it further for thin hulls, and the radius used is
  recorded. Other hull settings are Jolt's defaults (hull tolerance 1e-3, max convex-radius
  error 0.05).
  - Jolt's quickhull drops input points within 1 mm of the hull. For shardwright_house_l1 that
    took 43 192 input vertices to 27 426 and changed 2 125 of 2 253 hulls. The farthest input
    point left outside its hull is ≤ 1e-3 m (recorded in `config.hullCookingResult`).
  - Several hulls per body become a `StaticCompoundShape`, with identity sub-shape transforms
    because the vertices are already in the body frame.
- **Centre of mass**: Jolt places a body at its shape's volume centroid. For hull bodies the
  compound or hull is therefore wrapped in `OffsetCenterOfMassShape(-shape COM)`. Jolt's COM
  is then exactly the body origin, which is the scene's COM. Checkpoint 0 matches the scene and
  the scene's inertia applies about the right point. The largest centroid shift compensated is
  recorded (up to 4.4 cm on shardwright_house_l1 compounds: the scene's COM is not the union
  of its hulls' volume centroid).
- **Ground plane** (v2): a static `PlaneShape(Plane(+y, 0))` in the body frame at the scene
  pose (the plane passes through the body position, normal = body +y, solid below). Its
  bounding half extent is max(1000, 4 × scene extent), so it is effectively infinite here.

## Joints

- **Frames**: every joint is built in `EConstraintSpace::LocalToBodyCOM` with the scene anchors.
  The body origin is the COM (above), so these are exactly the scene's body-local anchors.
- **Ball joints** become a `PointConstraint`.
- **D6 joints** (format v3) use Jolt's specialised constraints where the mapping is exact.
  `--set joints=sixdof` forces everything to `SixDOFConstraint`; on hinge_chains_10k it gave
  the same joint gap.
  - **Hinge**: translations locked, swings locked, twist free or limited with
    min ≤ 0 ≤ max. This becomes a `HingeConstraint` (hinge axis = frame X, normal axis =
    frame Y, limits = twist limits). A twist drive is the hinge motor:
    - velocity drive: `EMotorState::Velocity`, target = drive velocity, torque limit ± force limit
    - servo: `EMotorState::Position` with a `StiffnessAndDamping` spring
  - **Swing-twist**: translations locked, at least one swing not locked, cone (or both
    swings free), no drive. This becomes a `SwingTwistConstraint`:
    - twist axis = frame X, plane axis = frame Y
    - plane half-cone (swing about Y) = swing1 limit, normal half-cone (swing about Z) = swing2 limit
    - free = π, which Jolt treats as free above 179.5°
    - twist range = twist limits
  - **Everything else** becomes a `SixDOFConstraint`:
    - locked axes are fixed, free axes free
    - limited: X translation uses `linear_x`, twist uses the twist limits, swings use ± their half-angles
    - swing type: cone or pyramid, from the scene
    - motors on the drive axis, with force/torque limits
  - **Drive differences**:
    - PhysX implements a velocity drive as a stiff damper (damping 1e6) capped at the force
      limit. Jolt's velocity motor is a hard velocity constraint with the same torque cap.
    - A servo's target velocity is ignored, because Jolt's position motor has none. No suite
      scene has a servo.
- **Joined bodies never collide** (`collide = false` in every scene). Each body that is in a
  joint gets `CollisionGroup(filter, GroupID = scene index)`. The `GroupFilter` subclass looks
  the pair up in a CSR adjacency list of joined partners (a few entries per body), and the
  broadphase pair filter rejects joined pairs before the narrowphase. Bodies without joints
  carry no filter. A `GroupFilterTable` per joint-connected component was rejected: its
  O(k²) bit table explodes on the big jointed components (chain mail, jointed_drop_34k).

## Counters

Read at checkpoints and never timed:

- `bodies`, `activeBodies` (`GetNumActiveBodies(RigidBody)`)
- `bodyPairs`: body pairs in Jolt's contact cache after the last step, which includes
  AABB-overlapping pairs without contact points
- `contactManifolds`: cached contact manifolds, about the number of contact constraints
- `joints`
- `updateErrors` / `overflow`: sticky `EPhysicsUpdateError` bits

Jolt has no public getter for the contact cache, so the shim reaches
`PhysicsSystem::mContactManager` read-only, through the legal explicit-instantiation access
idiom, and calls the cache's public `GetAllBodyPairsSorted` / `GetAllManifoldsSorted`. That
code is untimed. All counters read 0 at checkpoint 0.

## Known differences / caveats

- **Rounded edges**: the convex radius rounds box and hull edges by ≤ min(5 cm, 10 % of the
  smallest half extent). Mass properties are still the scene's sharp-cornered ones.
- **Hull cooking** drops near-interior points (1 mm tolerance), so hulls can be up to 1 mm
  smaller than the input.
- **Ground plane** is a true half-space (`PlaneShape`) with a finite bounding box of ±1000 m.
- **Collision filtering** is by explicit joint pairs, not by islands or components, the same
  as PhysX's per-joint `eCOLLISION_ENABLED` off.
- **chain_mail_1.6k** fails its gates at default settings: joint gap p99 0.30 m, ball sinks
  to y ≈ 2.48, pen p99 0.85. PhysX, rapier and box3d fail this scene too. A heavy ball on a
  net of light links is a mass-ratio problem that iterative impulse solvers do not hold at
  their default iteration counts.
- **hinge_chains_10k** misses the 5 cm joint gate: gap p99 0.078 at 10 velocity / 2 position
  steps, and 0.059 at 20/4. Long 25-link chains converge slowly in sequential impulses.
