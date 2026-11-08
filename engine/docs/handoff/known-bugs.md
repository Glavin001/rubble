# Known bugs and open issues: soft-body vehicles in the World

As of 2026-11-07, branch `feat/soft-rigid-world` (PR #105). Each entry says how sure we are:
- **Confirmed:** reproduced or measured.
- **Reported:** seen by the user, not yet reproduced.
- **Suspected:** a hypothesis with a reason.

Context is in [soft-rigid-world.md](soft-rigid-world.md).

## Behaviour bugs (wrong physics or wrong results)

### B1. Car and bricks interpenetrate on a slow push (Reported, high severity)
- **Where:** `brick-ring-drive.html`, deployed preview, authored sedan. Pushing slowly into the wall, bricks end up inside the car body, under its floor and through its wheels. Solids must never overlap. The user's screenshots are in the PR #105 thread.
- **Suspected causes:**
  1. **The ring drive uses node-only coupling contact.** Only car nodes collide with brick boxes. Nodes are 0.3–0.5 m apart and bricks are 1 × 0.5 × 0.5 m, so a brick corner or edge can pass between nodes, and nothing pushes it back out of a car triangle. Triangle-against-shape surface contact exists, opt-in, and is not enabled in this scene.
  2. **Raycast wheels ignore rigid bodies** (F7), so wheels drive through loose bricks.
- **Status:** diagnosis running on branch `diag/ring-phase-through`, with a reproduction at 2, 5 and 10 km/h, surface contact off and on, and a rigid-only control. Its report belongs here.

### B2. Authored sedan: about 41 km/h top speed and jittery, draggy driving (partly Confirmed)
- **Confirmed:** with the explicit solver's air drag on, the sedan tops out near 41 km/h, on the CPU reference too. With `?drag=0` it passes 33 m/s and keeps accelerating. The fast demo videos use `?drag=0`.
- **Suspected:** the drag is too large for the car's frontal area (to be checked against ½ρC_dAv²). The jitter is unexplained; physical and frame-pacing causes still need separating.
- **Note:** the soft car does **not** use the smooth rigid raycast vehicle from `vehicle.html`. It has its own raycast wheels on hub nodes inside the 2 kHz solver (`engine-softbody/src/explicit_drive.rs`).

### F1. Coupling uses the wrong contact rule for BeamNG-profile cars (Confirmed, small effect)
- **The difference:** contact with World rigid bodies always uses the plain Rigs of Rods rule, `ContactSurface::BOX`. Static boxes and shapes use the vehicle's profile, which for JBeam cars is BeamNG's: exact friction, no positive work.
- **Code:** CPU `explicit_rigid.rs` `node_contact_with`; GPU `explicit_beams_rigid.slang` (around lines 384/397 and 441/454, against `surfaceV(boxSurface(), …)` around line 844).
- **Effect:** "an immovable proxy equals the static shape bit for bit" holds only for Rigs of Rods-profile cars. For the ETK at 50 km/h, momentum differs by about 5 %.
- **Decision:** the user's.

### F7. Raycast wheels don't see rigid bodies (Confirmed)
- **Cause:** `explicit_drive.rs` `ray_cast` sees the car's own ground plane, boxes and static shapes only.
- **Effect:** cars can't drive up onto debris or rigid ramps; wheels pass through bricks (part of B1).
- **Decision:** the user's.

### R1. AVBD rigid solver loses about 5.5 % of momentum in multi-body impacts (Confirmed, on `main` too)
- **Measured:** a rigid-only World, 1,453 kg block into floating bricks, no car: 20,197 → 19,080 N·s.
- **Coupling impact:** the soft car into floating bricks shows ±7 %, attributed to this.
- **Missing tests:** no AVBD momentum test across iteration count or step rate (verification plan T3.6/T3.1).

### N1. ETK goes NaN at house strength 0.5 (Confirmed earlier, `one-world.html`, not re-checked)

### N2. The converted ETK has no tyres (Confirmed)
The BeamNG mod's shared tyre content isn't converted. On pressure wheels it rides on its rims, so the ring drive uses raycast wheels.

## Capacity and performance

### C1. Coupling shape room: 1,024 shapes per car can be exceeded (Confirmed)
The sedan pressed into the brick pile once hit "explicit vehicles: more colliders than the shapes room (… shapes: 1024)" in a recording replay. Earlier sampling never saw more than 122. It fails loudly, which is correct per `AGENTS.md`. Raise `ExplicitVehiclesDef::capacity` or make it an explicit engine setting; the CCF also needs more than 1,024.

### C2. Two touching cars are not real time (Confirmed)
A contact group runs on one workgroup: two touching ETKs cost 21.3 ms a step in Chrome (26–57 steps/s), and natively 24–28 ms. The fix is lossless: split the group across workgroups and exchange contacts.

### C3. First 2–3 s of a fresh ring page aren't real time (Confirmed)
The 110k ring settles into sleep (110k → 30k awake, rigid 20–30 ms a step). This is the ring's own cost, not the vehicles'.

### C4. Native timing shows `cpl_share` at 3.86 ms (Suspected mis-attribution)
That pass does trivial work, so the timestamps are probably attributed to the wrong pass.

### P1. Metal and CUDA lack the 2 kHz speed-up and surface modules (Confirmed)
`engine_exec::explicit_plan::MODULES`, the table Metal and CUDA compile, is missing the `*_seg` and `explicit_surface_*` modules that the WebGPU build ships. Test `engine-webgpu explicit_plan_modules` fails. Those paths don't run on Metal or CUDA yet.

## Tests and CI

| Item | Status |
|---|---|
| `engine-webgpu explicit_plan_modules` | Fails (P1) |
| Metal `explicit_surface_proxy_parity` | Hits the 15-minute kill; open in `explicit-vehicles-in-world.md` |
| `engine-softbody showcase_car`, 4 tests | `#[ignore = "known failure"]`: 120 km/h ledger 13.03 J, latch 78.1 kN, restitution 0.2018, T-bone 13 of 1,165 nodes. Numbers in `vehicle-authoring.md` |
| `engine-world threshold cuboid_house_breaks_and_splits_as_in_the_lab` | Fails on `main` too |
| `engine-ffi soft_car_visual a_wall_crash_reports_strain_tears_and_break_points` | Fails on `main` too |
| `engine-viewer fracture_regression every_scene_menu_entry_runs_on_metal` | Fails on `main` too |
| `engine-viewer fracture_tower_ccd` (60 Hz tower) | Fails on `main` too |
| `engine-oracles convex_slang_cpu`, 3 tests | Fail on `main` too |
| `engine-world prewarm` | Flaky under load |
| `cargo check -p engine-world --no-default-features` | Broken: `showcase_scene` uses the stress-gated `explicit_scene`. Nothing in CI builds it |
| CI "Build, then test on Modal" (CUDA) | Runner lost about 18 min into the build. Suspected nvcc out of memory: the first CUDA build of the vehicle kernels, one nvcc per core on a 16 GB runner. `2eea1dea` sets `RUBBLE_BUILD_JOBS=2` and logs memory. **Unconfirmed until CI re-runs** |
| CI "C library (macos)" | GitHub dropped `macos-14`; moved to `macos-15` in `2eea1dea`. Unconfirmed |
| Stress-oracle workspace tests, CUDA GPU tests | Not run for PR #105 |

## Platforms

- **iPhone / Safari: untested.** Safari has WebGPU on iOS 26. Risks:
  - workgroup memory (the vehicle kernels use about 30 of 32 KB);
  - the 10-storage-buffer limit;
  - phone GPU throughput.
- **Determinism:** native wgpu and Chrome are not bit-reproducible run to run (sleep on, crash outcomes). Compare kernels on identical state; don't compare two live runs.

## Content and licensing

- **ETK screenshots are committed:** about 25 Rubble renders of the BeamNG ETK, in `engine/docs/media/brick-ring-drive/etk*.jpg`, `engine/docs/media/crash-orbit/crash-orbit-etk-*.jpg` and `engine/docs/media/gen3-r3`, `gen3-r4`, `gen3-r5` (`*etk*`). No mod files are committed, but the renders show the mod's design. Keeping them or replacing them is the user's decision.

## Paused or abandoned

- **AVBD vehicles** (`feat/avbd-vehicles`): rejected. They look like jello at affordable settings and cost 4–9× at good ones.
- **Generated gen3 showcase car:** paused by the user (`vehicle-gen3.md`, `engine/tools/vehicle-gen`).
