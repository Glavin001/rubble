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

- Scene save/restore (every page's R key). `WgpuSim::save_state` exists.
- Vehicle part control: detach or disable wheels and parts, functional state, mount
  loads.
- Soft-body settings, obstacles and steering.
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

## Step time, before and after

`node engine/crates/engine-webgpu/tools/page-bench.mjs '<page>'`. This is headless
Chrome on the Apple M-series GPU, with no other GPU job running. The figure is ms
per step in frame-locked automation (`__step(300)`), best of 3 after 60 warm-up
steps. It covers the host's encode, submit and readback as well as the GPU work.

| Page | Before (JS host) | After (Rust host) |
|---|---|---|
| terrain.html?n=2000 | 2.43 | 2.36 |
| terrain.html?n=10000 | 3.45 | 3.09 |
