# rubble as the oracle's rigid-body engine (`stress-rubble`)

Written 2026-10-16. `crates/stress-rubble` implements `stress_ref::engine::RigidEngine`
on rubble's own AVBD GPU engine (`engine/crates/engine-webgpu`, `WgpuSim`), in the role
PhysX played in `stress-physx` (which stays in the PhysX repository). The coupled frame
(`World::with_engine`, `GpuWorld::with_engine`) is unchanged.

## What was built

**Engine host APIs** (`engine-webgpu/src/gpu_host_state.rs`, additive):

| API | What |
|---|---|
| `write_bodies(first, &[BodyState])` | overwrite body state (pose, velocity, mass, flags), current and previous rows, host mirror; the next step runs even if the last left nothing awake |
| `last_contacts()` | the last step's contact rows with their final AVBD forces (`lambda_n/t1/t2` along normal/tangents, anchors, collider ids); default and compact storage (not persistent) |
| `save_state()` / `save_state_reusing(old)` / `restore_state(&snap)` | GPU copy of every buffer plus the host mirror (counters, contact ping-pong, warm start): a restored run repeats bit for bit (hull arena excepted: the host's) |
| `body_count()`, `collider_count()`, `params()`, `last_counters()` | accessors |

The only changes to the step itself: a `wake_pending` flag (set only by `write_bodies`
/ `restore_state`) that defeats the "nothing awake" early return, and the recorded
offset of the step's contact rows. Unused, the step is unchanged (see Verification).
No kernel was changed.

**`RubbleEngine`** (`crates/stress-rubble/src/lib.rs`):

* One rubble body per world body. Cluster shapes become box colliders (box chunks) or
  convex-hull colliders (hull and compound chunks, through `ConvexHull::from_points`
  into the shared hull arena); sphere impactors are true spheres (PhysX used a
  42-vertex hull). Contacts are reported per chunk through a collider → (structure,
  chunk) table.
* The rubble body frame is the centre of mass with principal axes (the ABI stores a
  diagonal inverse inertia; `FLAG_WORLD_INERTIA` applies it as `R diag(I) R^T`);
  the oracle body frame is recovered from the stored COM and principal rotation.
* **Z-up / Y-up**: every world vector crosses the boundary through the proper rotation
  `(x, y, z) -> (x, z, -y)` (−90° about x); orientations as `q_axes * q`. Gravity must
  be along z (rubble has `gravity_y` only; anything else is an error).
* Params: `FLAG_CONTACT_GAP` (zero-gap rest), `FLAG_WORLD_INERTIA`, sleep disabled
  (`sleep_lin = sleep_ang = -1`: a sleeping body reports no support reactions; PhysX
  set sleep thresholds 0), AVBD defaults otherwise (10 iterations, α 0.99).
* Ground: a static plane collider at the given height.
* Supported (fixed) clusters: static bodies (`inv_mass = 0`).
* **Body removal**: rubble's arrays are append-only. A body the world drops or
  rebuilds (signature or support change) is retired: static where it stands, its
  colliders given a collision filter no pair admits. Capacity
  (`default_capacity()`: 16k bodies, 64k colliders) is consumed by every rebuild.

## Semantics

* **Freeze** (island bodies): emulated with static + restore, not a kinematic flag
  in the kernels. For the step the body is written with `inv_mass = 0`, zero inverse
  inertia and zero velocity; the solver then neither predicts nor integrates it and
  resolves contacts against it for the other bodies as against any static body. This
  is PhysX's kinematic body without a target (held still). `set_motion` writes the
  world's end state with the mass restored; a frozen body not handed back is released
  where it stands with zero velocity.
* **Impulses** (scripted loads) change driven bodies' velocities before the step:
  `dv = J / m`, `dw = I_world^-1 L` about the COM.
* **CCD**: rubble has none. A `ccd` body makes the engine step subdivide so it moves at
  most half its thinnest half-extent per substep (refines, never coarsens). Every
  substep's contacts are reported with that substep's impulse.
* **Contacts**: each live row's force `f = λn n + λt1 t1 + λt2 t2` gives impulse
  `-f h` on collider B's body and `+f h` on A's (verified: a resting box's impulses
  sum to its weight × dt within 2 %), at the midpoint of the two anchors. Approach
  speed: closing speed along the normal from the bodies' velocities at the start of
  the engine step (as PhysX).
* **Save / restore / discard**: a full GPU snapshot (buffers reused frame to frame);
  restore also returns the body table and releases anything still frozen.
* **Rest overlaps** (`set_rest_overlaps`): not implemented (returns false: the world
  keeps them). PhysX offset pair separations in a contact-modify callback; rubble
  would need a per-pair offset of the contact's `C0` in the kernels.

## Results

Ram (200 kg steel, 40 m/s) through a free 1 × 0.2 × 1.1 m concrete wall, established
methods, default step (tests `engine_world.rs`, `engine_world_gpu.rs`, M3 Max):

| Run | Ram after | Fragments | Broken bonds | Notes |
|---|---|---|---|---|
| standalone reference | 35.05 m/s | 6 | 24 | 19.8 s |
| reference + PhysX CPU (A11) | 35.05 m/s | 6 | 24 | |
| **reference + rubble** | **35.05 m/s** | **6** | **24** | 21.3 s; 1 island frame (2 bodies), 0 redone, momentum conserved < 1e-3 |
| GPU + PhysX CPU (A11) | 35.02 m/s | 6 | 26 | |
| **GPU + rubble** | **34.99 m/s** | **6** | **25** | 8.6 s; 1 island frame, 0 redone |

Other ported tests:

| Test | Result |
|---|---|
| resting box: impulses on its chunk carry its weight upward (±2 %) and rest at z = 0.5 ± 0.01 | pass |
| free 5-chunk column on the ground: bottom joint carries 4 m g | pass: −753.4 N expected, standalone −753.4, rubble −753.4; 60 engine body-frames, 0 island |
| frozen body holds still, others collide with it, released with the handed state | pass |
| impulses: `J / m`, `I^-1 L` | pass |
| save / restore repeats 30 steps bit for bit (engine and `WgpuSim`) | pass |
| handover energy, cube (established / proven) | pass / pass |
| handover energy, pair (established / proven) | pass (+0.088 J vs 0.190 J) / **fail** (+0.087 J vs 0.048 J) |
| handover energy, split (established / proven) | **fail** (+0.513 J vs 0.460 J) / **fail** (+0.507 J vs 0.106 J); `#[ignore]` with this note |
| showcases (`stress-ref/tests/shared/showcases.rs`, 7 cases: overhang, supports removed, ductile vs brittle wall, dropped floor, arch keystone, blast two walls, masonry house) | all 7 pass (109 s) |

## Gaps

1. **Rest overlaps.** The world's held overlaps (crushed contact, a split joint's
   compression) are pushed out by rubble at `1 − α` (1 %) of the overlap per step.
   In `split` that lifts the top cube 17 µm in the handover frame (+0.51 J; 11 % over
   the established allowance). With layer contact (`Methods::proven()`) the world
   hands these overlaps to the engine, and rubble cannot hold them: pair and split
   fail. Closing it: a per-pair `C0` offset in narrowphase emit (`contactC0N`), fed by
   `set_rest_overlaps`, mirroring PhysX's `set_pair_offsets`.
2. **Contact warm-up on creation.** AVBD starts each new contact at `k_start` (1e3)
   and ramps it; a 2.4 t cube placed just touching sinks 1.6 mm in its first frame
   (−19 J). The handover test does not credit the engine's own losses, so this does
   not fail it, but newly created fragments settle into the ground by about g dt².
3. **Append-only bodies.** Retired bodies keep their slots and stay in the broadphase
   (filtered out of every pair). Long fracturing runs need capacity for every rebuild;
   overflow panics with rubble's capacity message.
4. **Freeze is static emulation**, not `BODY_KINEMATIC` in the kernels. Equivalent for
   the coupling (held still, no target velocity), but a kinematic body with a target
   motion would need the kernels.
5. **CCD by substepping**: rubble steps with `dt / n` for fast CCD bodies (up to 256
   substeps per frame). Island-owned impacts never reach it (the ram's impact frame is
   an island); a driven fast impactor would.
6. **Gravity along z only** (rubble's `gravity_y`).
7. Per-contact friction is rubble's material 0 (0.5) everywhere; the scene's ground
   friction is not passed (PhysX used its default material too).

## Verification

* `cargo test --release -p stress-rubble` (fast tests), `-- --ignored` (ram, column),
  `--features gpu --test engine_world_gpu -- --ignored`, `--features showcases
  --test showcases_rubble`.
* `engine-webgpu` (`cargo test -p engine-webgpu --release`, GPU tests included): 58 pass, among them the new `host_state_round_trips_and_contact_forces_carry_the_weight`; 1 fails, `count_only_kernels_do_not_load_unrelated_atomic_counters` (a WGSL text check on `extract_velocity_sleep`, generated from Slang this work does not touch, so it predates it). No Slang kernel changed.
