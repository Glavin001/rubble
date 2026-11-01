# One host for every target: the web pages on `World`

The web pages used to drive the GPU through their own JavaScript host. Native and
Unreal use the Rust one: `engine-webgpu`'s `WgpuSim` under `engine-world`'s `World`.
Host-side work therefore never reached the browser. That included the large-maps
branch (`feat/large-maps`, PR #92): the shared threshold fracture, the piece pool,
sleep with archived contacts, change-only readback and the cheaper per-step host.

The target is `engine-world-wasm`, a thin wasm-bindgen layer over `World`. JavaScript
keeps only rendering, input and UI.

## How the browser runs the Rust host

- **Device.** `WebWorld.create(def)` opens the browser's WebGPU through wgpu
  (`Backends::BROWSER_WEBGPU`) and builds a `World` on it (`GpuContext::Shared`).
  The page renders on the same `GPUDevice` (`WebWorld.device()`, wgpu 30's
  `Device::as_webgpu`). Its render pipelines bind the world's buffers in place
  (`Buffer::as_webgpu`), with no copy and no readback, as the JS host's pages did.
- **Kernels.** These are the WGSL `engine-webgpu` embeds at build time
  (`direct_shaders`), the same text the JS host fetched from `generated/wgsl/`.
- **No blocking on the frame path.** `WebWorld.step()` steps only while a readback
  slot is free (`World::can_step_without_waiting`). Otherwise it returns `false`
  and the page polls and tries again next frame. Steps are deferred, never dropped
  or changed. `WebWorld.poll()` is `World::poll_coalesced`, which never waits.
- **Rare blocking reads.** Examples are setup preloads and `read_transforms`. They go
  through `engine-gpu-wait`, the JSPI suspension (and Asyncify on Safari) that
  stress-web already uses. They are not on the per-frame path.
- **Native switches.** The `RUBBLE_*` / `ENGINE_*` environment variables read as
  unset on wasm. A page passes configuration through `WorldDef` instead.

## Inventory of the JavaScript host (before)

Paths are under `engine/crates/engine-webgpu/`.

| Page | JS host | Step | Wasm | Renders from |
|---|---|---|---|---|
| `demo.html` | its own (dispatch list `K`, per-URL kernel routing, RUBSNAP1 scenes) | hand-written | vehicle-wasm (constants) | b0, b2 |
| `vehicle.html` | `world-host.mjs`, `vehicle.mjs`, `soft.mjs`, guidance | replays `RigidStep` | vehicle-wasm | b0, b2, b31; buggy renderer |
| `softbody.html` | WorldHost, SoftHost, VehicleHost | `RigidStep` | vehicle-wasm | b0, b2, b31, b32; GPU skinning |
| `motion.html` | WorldHost (characters, projectiles, kinematics) | `RigidStep` | vehicle-wasm | b0, b2 |
| `terrain.html` | WorldHost plus heightfield | `RigidStep` | vehicle-wasm | b0, b2, b18 |
| `impact.html` | WorldHost, VehicleHost, `terrain-host.mjs` | `RigidStep` plus `TerrainStep` | vehicle-wasm | b0, b2, b18, b31, terrain buffers, `drawIndirect` |
| `showcase.html` | as impact, plus buggy renderer | same | vehicle-wasm | same |
| `fracture/` (`main.mjs` → `simulation.mjs`) | fracture `Simulation` | replays engine-fracture `Plan` | fracture-wasm | a b0 copy, b5 |
| `fracture/` stress mode | none (stress-web: Rust host and renderer) | Rust | stress-web | Rust |

The JavaScript host's readbacks are all asynchronous `mapAsync`:
- counters (b7)
- the joint, mount and beam break rings (b21)
- motion (b33)
- soft beams, particles and modes
- guidance states

The tests are:
- `tests/*.test.mjs` and `fracture/*.test.mjs`, run with `node --test` in CI. Three of
  them slice `demo.html`'s source text, and `fracture/simulation.test.mjs` mocks a
  `GPUDevice`.
- The CDP drivers `motion-check.mjs` and `softbody-check.mjs`, and the recorders. These
  are run by hand.

`web/` (the Vite app) is not part of this. It is the legacy root-workspace engine
(`crates/rubble-wasm`, wgpu 29), and its host is already Rust.

### JavaScript host → `World`

| JS host | `World` |
|---|---|
| `createWorld` | `World::new` |
| `addBody` / `addCollider` | `create_body(s)` / `create_shape` |
| `setMaterial` | `shape_set_surface_material` |
| heightfield arena | `create_height_field_shape`, `set_height_field_region` |
| ball / D6 joints | `create_joint`, `joint_set_drive` |
| `encodeStep` + submit | `step` |
| counters, pose reads | `Frame.counters`, `Frame.body_events` |
| `applyImpulse`, kinematic velocity | `body_apply_linear_impulse`, `body_set_velocity` / `body_set_target_transform` |
| characters, projectiles, `readMotion` | `create_character`, `character_set_input`, `spawn_projectile`, `Frame.characters` / `character_events` |
| break rings | `Frame.destruction_events` |
| vehicles | `add_vehicle`, `set_vehicle_input`, `Frame.vehicles` |
| soft bodies | `add_soft_vehicle`, `soft_bind_skin`, `soft_damage`, `Frame.soft` |
| guidance | `set_guidance(_zones)`, `body_set_guided(_tags)`, `body_guidance` |
| terrain | `enable_terrain`, `terrain_*`, `Frame.terrain*` |
| fracture `Simulation` | `create_destructible_asset` (GLB, pack, cuboid), `create_destructible` |

### What `World` lacked

- Scene save/restore (every page's R key). Added for vehicle.html:
  `World::save_state` / `restore_state` (bodies, shapes, joints, vehicles and soft
  bodies; not destructibles or characters). Terrain was added for impact.html
  and showcase.html: saved with no area open, restored with the pool started
  over.
- Vehicle part control (detach or disable wheels and parts, functional state,
  mount loads) and soft settings and obstacles were already in `World`
  (`vehicle_damage`, `rig.rs`); the wasm layer now exposes them.
- GPU buffer handles for renderers. These are reachable only through `World::sim()`.
- `demo.html`'s benchmark harness: kernel variants, per-kernel timestamps, RUBSNAP1
  scenes.
- The fracture lab's sliced and overlapped stepping and its contact inspection.

## Progress

Each page moves in its own commit.

- **terrain.html.** On `World` with a heightfield shape, region edits and
  `heightFieldSurface`. Mass and inertia are set as before (`bodySetMassData`).
  Contacts are canonical, as the JS host's were. Reset and rain are body
  teleports plus velocities, where the JS host restored its CPU mirrors.
  `__check()` passes: no body sinks into the terrain before or after a crater.
  The off-field counts (17 / 42) fall within the old page's own run-to-run spread
  (16-21 / 37-49): a 2,000-body pile is chaotic, and GPU float order already
  varied between runs.

- **motion.html.** Characters (`createCharacter`, `characterSetInput`), projectiles
  (`spawnProjectile`, drawn from the world's motion pool in place) and kinematic
  platforms (`bodySetVelocity`). Parked balls, debris and rockets used to be static
  bodies switched to dynamic by rewriting their records. They are now asleep
  dynamic bodies, woken by `bodySetTransform` + `bodySetVelocity`. Projectiles keep
  the JS host's 8 hits per step (`maxHitsPerStep: 8`); `World`'s default is 4.
  `motion-check.mjs` passes every check: CCD at 60, 140 and 250 m/s, the character
  stairs and jump, and bullets through five walls. Reset (R) teleports bodies and
  characters back to their starts. It does not clear projectiles in flight, which
  the JS host's full-buffer restore did. The HUD no longer shows "stepped up" (the
  character's output flags), which `World`'s `CharacterOutput` does not carry.
- **vehicle.html.** Every scene (playground, destruction, traffic, the labs, buggies
  and vibeland rigid or `?build=hybrid`, gauntlet, meteors, run) runs on `World`.
  The page keeps rendering, input, UI and its scene scripts. Vehicles, soft
  vehicles, damage, functional state and mount loads go through `World`; the
  page renders wheels and the buggy meshes from the world's vehicle pool
  (`vehiclePoolBuffer()`) in place. The street, meteor-range and run builders
  stay in engine-vehicle-wasm (Rust that only describes launches); the page
  applies them through `World`. `World` gained:
  - `save_state` / `restore_state`: the R key, and every take's reset.
  - `body_set_type`: parked static blocks, meteors and cannonballs launched or
    put back. The JS host rewrote their records.
  - `set_material_slot`: the JS host's 64-slot table pinned exactly (slot 0
    friction 1.0, 7 the rims at 0.9, 9 soft nodes at 0.8, the rest 0.6),
    because vehicle colliders name slots directly.
  - `reserve_moving_radius`: the facade blocks and meteors, which the JS host
    counted as moving colliders while parked, size the grid the same way.
  - `joint_set_linear_penalty`: the bridge bolts start at 1e8 N/m, as before.
  - `body_at`: guidance zones name anchors by body slot.
  - A soft body's bolted part joints are registered, so their breaks report.

  Masses and inertias are set with `bodySetMassData`, the solver as before
  (8 iterations, 20 on the bridge, the hybrid's 240 Hz settings), contacts
  canonical. Driving the same input from the same state gives the same
  chassis positions to 2 decimals over 360 steps in every scene
  (`?stopAt=1`, then `__step(60)` idle and 6 × `__step(60, forward)`):
  playground, destruction, buggies, vibeland, bridge, gauntlet, run, meteors,
  slab, seesaw, carry and sidewall. Hybrid is within 0.01 m with its first
  step one substep apart. Traffic (48 AI cars), tanklab (loose rubble) and the
  last sample of destruction diverge after collisions, as chaotic piles do.
  `__gauntletTake` and `__meteorTake` give the old page's numbers exactly.
  `__runTake` (1,200 steps) differs slightly: 85 near misses and 7 hits
  against 87 and 6. Mount breaks (buggies at `?strength=0.1`) and soft
  break groups (hybrid) report the same part at the same force.
  `motion-check.mjs vehicle` passes (CCD cannonballs, no overflow). Differences:
  - `maxCommands` is 2,048 (JS 64). Teleports and velocity sets are
    commands in `World`. This is capacity only.
  - Parked meteors carry `BODY_CCD` while parked, so the CCD passes run from
    the start. The flag is inert on a static body.
  - Reset restores every buffer, contacts and warm starts included. The JS
    host kept the last contact state.
  - Results after a reset can vary between runs. The warm-start hash is sized
    from the latest polled contact count (`set_plan_hints`), and polls now
    land between steps at wall-clock-dependent times. One of three reset runs
    matched the JS host bit for bit. This hint timing was already in `World`;
    the JS host's pages varied too, through their frame loops.
  - The status line's counters come from frames; the HUD no longer reads the
    vehicle pool itself.
- **softbody.html.** Every scene (the yard, the crash tests and presets,
  crates, import, the traffic derby, cruise and parked, limbo, sandwich,
  crusher, and `?car=buggy`) runs on `World`. The page keeps rendering, input,
  UI and its scene scripts: the traffic AI, the side ram, the crusher's winch,
  the cannon and the imports. The `.truck` and JBeam importers and the
  procedural car, crate and buggy stay in engine-vehicle-wasm (`soft.mjs`);
  they only describe vehicles, and the page spawns them with `addSoftVehicle`.
  `World` itself is unchanged. engine-world-wasm gained (`soft.rs`):
  - `softAddObstacle`, `softCount`, and in `softEntry` the triangle range,
    mass and importer warnings.
  - `softSkinTriangles` / `softSkinRest` for the collision skin.
  - `softBindMesh` (`SoftHost::bind_mesh`) and `softFlexbody`. The flexbody
    word builder moved into `engine_softbody::flexbody::flexbody_words`, so
    both wasm crates share it.
  - `softBuffer` / `softLayout` for the beam view, bound in place.
  - `colliderMaterials`, so the page draws only drawable colliders.
  - `readSoftModes` and `readSoftDamage` (`damage_summary`), Promises like
    `readBodies`.

  Skinning stays on the page. The car mesh is skinned by the engine's
  `soft_skin` kernel, dispatched by `soft-skin.mjs` and bound with
  `softBindMesh`. That code only builds the vertex-to-triangle lists and
  binds buffers; it is rendering, not physics. `World::soft_bind_skin` is not
  used, because it is built for CPU consumers:
  - one skinner per soft body,
  - encoded on every step that has a readback slot,
  - every vertex read back for `Frame::skins`.
  In 100-car traffic it would mean 100 skinners × 3 dispatches × 4 steps per
  frame, plus a full readback each step. The page skins every car in one
  dispatch per drawn frame. A follow-up would be a render-only `World` skin:
  several soft bodies, no readback, encoded on demand.

  Masses and inertias are set with `bodySetMassData`. The solver is as
  before: 240 Hz, 10 iterations, `betaLin` 1e7, gravity −9.81. Contacts are
  canonical and soft settings `calibrated_for(240)`. The material table is
  pinned slot for slot: 0 → 1.0, 7 → 0.9, 9 → 0.8, the rest 0.6.
  The initial GPU state is bit-identical to the JS host's. This covers
  bodies, colliders, the soft arena, the vehicle pool, materials and the
  host words. The exceptions are the parked cannonballs, which are static
  with no mass (the JS host kept an unused `invMass`), and collider density,
  which only vehicle belts read.

  With the same input from the same state (`?stopAt=0`, then 6 × `__step(60)`):
  - The yard and buggy drives and the parked traffic match the old page
    exactly.
  - The crash scenes diverge after 60–120 steps. The cause is in the old
    page, not in `World`. The JS host's plan read its counters only when a
    `__step` call ended, so the second contact step built its contact rows
    in another order (the same rows as a set) and the solve took them in
    another order. Given counters after every step (a copy of the old page
    that reads back each step), the JS host gives `World`'s results exactly
    over 360 steps in the wall, tbone, sandwich, drop, crates, yard, buggy
    and parked-traffic scenes.
  - Pole, head-on, limbo and crusher match until 180–300 steps into the
    crush. There `World` varies from run to run, because its plan hints come
    from frames polled at wall-clock-dependent times (as for vehicle.html).

  `softbody-check.mjs` passes all 19 default checks: drive, crash tests, tear,
  limbo roof, traffic, beams, crates, close-ups and perf. `PAGE=` checks
  another copy of the page. Its opt-in BUDGET check now times `World.step` on
  the CPU. The JS host's per-kernel GPU split (`?gputime`, `profileStep`) and
  its dispatch histogram are gone, because they were that host's internals.
  Every automation hook is kept (`__step`, `__particles`, `__spawnImported`,
  `__crashImported`, `__setView`, `__launchRam`, `__setCaption`, `__live`,
  `__skinBench`, `__renderBench`, `__tornOff`, `__world`). Other
  differences:
  - Records the JS host rewrote are now `World` calls. The cannonballs and
    the ram launch with `bodySetType`. The crusher's winch sets 1e7 kg mass
    data once, then `bodySetType` each tick, with the same pose and velocity
    as before. The cannonballs are parked static, so
    `reserveMovingRadius(0.5)` sizes the grid as the JS host's count did.
  - `readbackFrames` is 10. At `World`'s default of 3, a 240 Hz frame takes
    only 3 of its 4 steps, which is 75 % real time.
  - Breaks and torn-off parts come from each frame's destruction events. The
    JS host polled them every 30 steps with the damage; the totals are the
    same. Poses for the HUD, camera, AI and crusher come from `readBodies`,
    as the JS host's `mapAsync` copies did.
  - `?stopAt=N` stops the page's own loop at step N.
- **impact.html.** The meteor impact (impact and buried charge, every view,
  the cut, the tracked parcel, Bake now, the timed bake, the drive-in truck and
  Replay) runs on `World`. The page keeps rendering, input, UI and its
  scripting. The terrain is `enableTerrain` with the `TerrainDef` the JS
  host's `TerrainStep` had: the game profile (or `?profile=physical`), the
  browser limits, the margin, the settle speed and 30 frames, and the live
  surface without patches or debris copies. The strike is `terrainImpact` /
  `terrainExplode`; a settled area bakes inside `World::step`, Bake now and
  the timed bake are `terrainClose`. The page binds the particle, parcel,
  counter, surface, body, collider, hull and vehicle buffers in place and
  draws the engine's debris list with `drawIndirect`. Counters, the live
  surface and bakes come from frames. The crater is measured from World's
  heightfield mirror (`heightFieldHeights`) once the bake's frame arrives.
  The prediction comes from `terrainPredictImpact` before the world exists,
  because it sizes the heightfield. The solver is as before: 120 Hz, 8
  iterations, g = 9.81, `SPEC_VELOCITY`, canonical contacts, friction 0.9 on
  slot 0 and the rims and 0.6 elsewhere.

  With the same input from the same state (launch, then frames polled one by
  one), World gives the JS host's results exactly. Over 600 coupled frames the
  max speeds, parcel counts and substeps are equal. Each frame's substeps
  follow the speed in the last report that arrived (`RegionClock`), so they
  depend on when reports arrive. With pipelined `__step(60)` both hosts
  diverge from that reference, by different amounts:
  - the crater depth is 93.28 m (JS) and 93.62 m (World), the rim radius
    236.1 m in both, the rim height 24.10 m and 24.21 m;
  - the drive-in truck follows the same path to within 0.4 m;
  - the particle counts and window are identical.

  `World` gained (each change is in World and engine-webgpu, shared with
  native and Unreal):
  - Save/restore with terrain (`save_state` refuses an open area). A restore
    drops open areas without baking them, as the JS host's `terrain.reset()`
    did, and starts the pool over (`WgpuSim::terrain_reset`). It also puts
    every heightfield's samples back: the hull arena is not part of a
    snapshot, so it kept what the bakes wrote. Snapshots skip the terrain
    pool's buffers, about 200 MB for impact.html.
  - `TerrainDef::predict_impact` / `predict_explosion` (and
    `vertical_impact`, `tnt_charge`): the crater, the region's cell and the
    source's peak speed, without a world.
  - `terrain_last_substeps` for the HUD and the frame log.
  - Two `WgpuSim` fixes:
    - With a live surface and no patches, the last bake now clears the
      surface's valid word, so renderers stop drawing a stale surface. The
      JS host did this itself.
    - The surface buffer can be used for indirect draws.
  - Read-only particle and parcel buffers for renderers.

  engine-world-wasm gained `terrain.rs`: `enableTerrain` (a camelCase
  `TerrainDef`), the events, `terrainAreas`, the buffers, Promise reads of
  particles, parcels and the surface, and world-less `terrainSettings`,
  `terrainPredictImpact`, `terrainPredictExplosion` and `terrainLayout`.
  Frames carry `terrain()`, `terrainBakes()` and `terrainSurface()`.

  Differences:
  - The HUD and `__impact.timings()` no longer have the terrain pass's GPU
    time. The JS host timed its own pass with timestamp queries, and World
    encodes that pass itself.
  - The bake shows (phase "baked") when its frame arrives, a readback lag
    after the step that baked. The JS host knew when it encoded the bake.
  - The switch error compares the last frame's live surface with the baked
    mirror (`switchError` in terrain-render.mjs). The JS host read both back
    from the GPU.
  - `__step(n)` is new; `__impact.frames(n)` is kept.
- **showcase.html.** Every scene (drive, drive1, pass, bts and play) runs on
  `World`. The terrain settings are those of the JS host, as a `TerrainDef`:
  the game profile, 6 cells per radius, the margin, 60 settle frames,
  `maxRegions` per scene, the local settle rule in play, and a surface of
  2²¹ cells with one patch of `PN` samples per area slot (`patchSamples`).
  The dune buggy is `addVehicle`, drawn by buggy-render.mjs through a view of
  the world's buffers, as in vehicle.html. The page keeps the schedule, the
  director, the autopilot, the camera, the recording and the HUD.
  `liveHeightAt` reads the surface with `readTerrainSurface`. Reset is
  `World::restore_state`, and a run after it repeats the first run bit for
  bit.

  With the same input from the same state (`?manual=1`, `__showcase.frames(n)`,
  each frame waited for, as the JS host's), the terrain gives the JS host's
  results exactly: substeps, particles and parcels in flight agree over 2,400
  frames, and the strike times and points are identical. The buggy's path is
  bit-identical until it couples to open sand (6.7 s into the drive scene).
  From there it diverges and ends 20 m apart after 20 s.

  The cause is the order of the colliders. The JS page made the 8 surface
  patches before the car; World makes them at the first strike, after it. A
  copy of the old page that creates the car first gives World's result
  exactly. The coupling bins colliders by index, so float sums run in
  another order, and a buggy on fresh sand amplifies that. Settled bakes
  happen at the same times (4.783 s and 7.683 s, `scene=pass&settle=3`).

  Other differences:
  - The JS page's switch error was 0 with patches. Its read came after the
    surface retire had cleared the view table. The new measurement gives
    0.09 m rms.
  - The terrain and surface GPU times are gone from the HUD and
    `timings()`, as in impact.html.
  - The page's own loop never waits. A frame with no readback slot free
    only renders. The JS loop waited for the GPU every frame.
  - Frames carry the open areas' live heights every coupled step
    (`TerrainDef::surface`). The page draws from the buffer and uses the
    copy only for the switch error.
- **world-host.mjs, terrain-host.mjs.** No page imports them any more. They
  stay, as do `RigidStep` and `TerrainStep` in engine-vehicle-wasm, which
  `tests/terrain_plan.test.mjs` still checks against engine-terrain's plan;
  removing them is a follow-up.
- **Non-blocking reads.** `readBodies(slots)` returns a Promise (a copy, then
  `mapAsync`) for checks that need velocities.

## Step time, before and after

`node engine/crates/engine-webgpu/tools/page-bench.mjs '<page>'`. This is headless
Chrome on the Apple M-series GPU, with no other GPU job running. The figure is ms
per step in frame-locked automation (`__step(300)`), best of 3 after 60 warm-up
steps. It covers the host's encode, submit and readback as well as the GPU work.

| Page | Before (JS host) | After (Rust host) |
|---|---|---|
| terrain.html?n=2000 | 2.43 | 2.36 |
| terrain.html?n=10000 | 3.45 | 3.09 |
| motion.html (CCD gallery, idle) | 0.72 | 0.008 |
| motion.html?scene=character | 1.04 | 1.08 |
| vehicle.html (playground, idle) | 0.61 | 0.019 |
| vehicle.html?scene=destruction (idle) | 0.90 | 0.018 |
| vehicle.html?scene=traffic | 2.40 | 2.68 |
| vehicle.html?scene=bridge | 1.94 | 2.46 |
| vehicle.html?scene=buggies | 1.64 | 1.75 |
| vehicle.html?scene=gauntlet (idle) | 0.63 | 0.020 |
| vehicle.html?scene=run (idle) | 0.61 | 0.029 |
| vehicle.html?scene=meteors | 1.12 | 0.94 |
| vehicle.html, driving | 1.20 | 1.30 |
| vehicle.html?scene=destruction, driving | 1.50 | 1.68 |
| vehicle.html?scene=traffic, driving | 2.25 | 2.52 |
| vehicle.html?scene=bridge, driving | 2.65 | 3.00 |
| vehicle.html?scene=buggies, driving | 1.73 | 1.65 |
| vehicle.html?scene=buggies&build=hybrid, driving (240 Hz substeps) | 2.30 | 2.26 |
| vehicle.html?scene=gauntlet, driving | 1.04 | 1.14 |
| vehicle.html?scene=run, driving | 1.05 | 1.09 |
| vehicle.html?scene=meteors, driving | 1.21 | 1.31 |
| softbody.html (yard: 3 cars, 3 crates) | 0.98 / 1.01 | 1.10 |
| softbody.html?scene=wall | 1.64 / 1.66 | 1.69 |
| softbody.html?scene=crates | 1.31 / 1.33 | 1.32 |
| softbody.html?scene=traffic&cars=20 (AI) | 3.58 / 3.52 | 2.39 |
| softbody.html?scene=traffic&cars=100 (AI) | 3.95 / 3.96 | 3.32 |
| softbody.html?car=buggy | 1.01 / 1.04 | 1.12 |
| softbody.html, driving | 1.72 / 1.74 | 1.77 |
| softbody.html?car=buggy, driving | 1.87 / 1.88 | 1.87 |
| softbody.html?rigid=0, driving | 1.80 / 1.83 | 1.88 |
| impact.html (ready, idle) | 0.48 | 0.006 |
| impact.html?autostart=1 (impact open, game profile) | 1.95 | 1.60 |
| impact.html?autostart=1&profile=physical | 5.59 | 5.01 |
| showcase.html?lead=100 (driving, no strike) | 1.63 | 1.66 |
| showcase.html (drive: the barrage) | 2.29 / 2.06 / 2.08 | 2.82 / 2.33 / 2.29 |
| showcase.html?scene=drive1 | 2.30 | 2.81 |
| showcase.html?scene=bts (300 Hz) | 1.55 | 1.61 |

`motion.html`'s CCD gallery is at rest until a volley: `World` skips a step when
nothing is awake and nothing was written, and the JS host never did.

vehicle.html's plain `__step(300)` sets no input, so the player car comes to rest and
most scenes go idle in the same way (the earlier 0.564 / 0.825 ms baselines were
this measurement on the JS host). The "driving" rows time `__step(300, { forward:
1, steer: ±0.3 })` after `__step(60, { forward: 1 })`, best of 3, so the car and
what it hits stay awake. Runs vary by about ±10 % from one invocation to the next.
On the Rust host, driving costs 0-13 % more per step. `World` copies every
vehicle's record and wheels with each step (`Frame::vehicles`), which is 98 copies
in traffic; the JS page read only the player's.

softbody.html steps at 240 Hz, and "driving" times `__step(300, { forward: 1,
steer: ±0.3 })` there as well. Its "before" column gives two runs of the old
page, taken at the start and the end of the port. On the Rust host, the
single-car scenes cost 0-10 % more per step: `World` submits and reads back
every step, where the JS host batched 8. Traffic is faster because of how
each 8-step chunk waits for the AI's pose read. The old page polled with 1 ms
timers; the new one awaits the `readBodies` Promise.

impact.html and showcase.html are timed with a `__step(n)` added to both
pages: n frames, pipelined, with no render. The old copy waited only when all
8 counter readbacks were in flight. Once a strike lands, each run of 300
frames covers another phase of it (other strikes in showcase.html, the
crater forming in impact.html), so best of 3 is the cheapest phase.

showcase.html's drive row gives three invocations. Their means over the 3
runs are 3.05 (JS) and 2.95 (World), and the best runs differ by 10-20 %. The
terrain work is identical. On top of it, World copies the open areas' live
heights back every step, and the page reads the buggy's chassis with
`readBodies` every step.
