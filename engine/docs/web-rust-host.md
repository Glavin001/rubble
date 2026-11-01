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
  bodies; not destructibles, terrain or characters).
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
