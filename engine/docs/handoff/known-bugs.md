# Known bugs and open issues: soft-body vehicles in the World

As of 2026-11-07, branch `feat/soft-rigid-world` (PR #105). Each entry says how sure we are:
- **Confirmed:** reproduced or measured.
- **Reported:** seen by the user, not yet reproduced.
- **Suspected:** a hypothesis with a reason.

Context is in [soft-rigid-world.md](soft-rigid-world.md).

## Behaviour bugs (wrong physics or wrong results)

### B1. Car and bricks interpenetrate on a slow push (Reported, high severity)
- **Where:** `brick-ring-drive.html`, deployed preview, authored sedan. Pushing slowly into the wall, bricks end up inside the car body, under its floor and through its wheels. Solids must never overlap. The user's screenshots are in the PR #105 thread.
- **Root causes: Confirmed and reproduced** (branch `diag/ring-phase-through` 1a302062; probe `engine-world/examples/ring_phase_probe.rs`; batch `engine/tools/ring-phase/gpu-batch.sh`; logs in `~/Development/rubble-artifacts/ring-phase-through/`). There are two independent mechanisms.
  1. **No penetration correction against moving rigid bodies** (the main one). `node_contact_with` (CPU `explicit_rigid.rs`, GPU `explicit_beams_rigid.slang`, bit-identical) calls the contact force with penetration 0 and the plain BOX surface. The only push is `−0.8·vn` while a node approaches, and nothing pushes a node back out once it's inside. A brick's proxy moves as a free body inside the coupling window (no gravity, ground or neighbours); then the World's rigid step moves it differently, and each mismatch becomes penetration that's never undone.
     - **Measured** (sedan, GPU World, wall of 25 kg bricks): nodes reach 0.25 m deep (a brick's centre), bricks 0.64 m inside the car's hull, the same at 2, 5 and 10 km/h, with sleep on or off. The bricks are then carried inside the car: 5–6 bricks still inside at z = 57 m.
     - **Static-brick control:** 0.003–0.009 m. A 1,300 kg rigid box pushed into the same wall overlaps at most 3 cm, so the rigid side is fine.
     - **Partial evidence about the window:** at a 240 Hz World, depth falls only from 0.21 to 0.13 m.
  2. **Node-only contact.** The ring drive never turns on surface contact (`ExplicitBeams::surface_contact` defaults to false). A 45°-turned static brick pushed into the nose ends 21 cm inside the car with surface contact off, 0.000 with it on (CPU). Against dynamic bricks surface contact helps but isn't enough (hull 0.64 → 0.15–0.22 m), because mechanism 1 remains.
  - **Wheels:** F7 (raycast wheels don't see rigid bodies). Bricks reached up to 11 cm inside the wheel cylinders; only hub nodes collide.
- **Fixes, ranked; each changes physics and needs the user's decision:**
  1. Turn on surface contact in the ring drive (exists and is tested; cost unmeasured here).
  2. Give proxy node contact the penetration response ground and soft↔soft contact already have (the §37 position step or `0.2·p/h`), CPU and Slang. This is the root fix for mechanism 1, and ties in with F1.
  3. Make the proxy's motion inside a window match the World's (ground and contact-connected neighbours in the window, or the World adopting the window's prediction). This is a larger design change.
  4. F7: wheels that query the World broadphase once per window.

### B2. Authored sedan: about 41 km/h top speed and "dragging" (Confirmed, root cause found)
- **Cause:** the deployed `?car=sedan` (the two-tier showcase sedan) isn't in the BeamNG profile. Its truck text has no `rubble_beamng_profile` or `rubble_drag_coef`, so it runs Rigs of Rods' per-node air drag: F = −0.05·|v|·v **on every node**. 987 nodes give 49.4·v² N in total, against a physical ½ρC_dAv² = 0.39·v² (C_d 0.30, A 2.16 m²): **127× too much**.
  - At 40 km/h that's 6.1 kN against 48 N, a 0.43 g deceleration.
  - Coast-down on the CPU confirms it: 5.63 kN measured against 5.64 kN from the formula.
  - Top speed is where the traction limit (about 6.9 kN) meets this drag: √(6.9 kN / 49.4) = 42.6 km/h, and 41.1 km/h was measured. With drag off it reaches 229 km/h.
- **Not the cause:** per-triangle drag (`dragCoef` 100). That path applies only to the one-tier sedan.
- **Fix:** at the source (authoring), not at load time. Give the two-tier sedan the BeamNG-profile triangle aero, or C_d·A-calibrated drag. This changes the car's physics, so it's the user's decision. `?drag=0` is the workaround.
- **No break-through at 40 km/h** follows from this: the drag caps impact speed, and bricks absorbed by B1 never load the front.

### B3. Driving looks jittery (partly measured)
- **Measured: no physical shake.** On flat ground the car's vertical rms is 0.15–0.25 mm.
- **Proven, small:** the solver steps 33 or 34 times per 1/60 s frame, carrying the remainder. That gives a ±3–6 mm, 20 Hz sawtooth in position per drawn frame at 41 km/h. It's probably too small to be what the user sees.
- **Hypothesis** (not measured in Chrome): the page's camera follows the car from an asynchronous node readback, not extrapolated by the readback's age, while the car is drawn at the current step. Skipped readbacks make the car move unevenly against the camera, and so do frames under 60 World steps/s (the ring settling, slower GPUs).
  - **Next:** measure rAF dt, World steps per rAF and readback age in Chrome.
  - **The fix would be lossless and render-only:** extrapolate the camera target by the readback's age, or interpolate by simulated time.
- **Wheel model:** the soft car does **not** use `main`'s rigid raycast vehicle (`engine-vehicle`, `vehicle_step.slang`; implicit wheel rows inside the AVBD solve of a rigid chassis, with rays against the World broadphase including dynamic bodies). It uses the explicit solver's own raycast wheels on hub nodes, in direct drive (`explicit_drive.rs`, `explicit_beams_drive.slang`).
  - **Minimal path to `main`'s behaviour:** cast wheel rays once per World window against the World broadphase, and use the hits as moving contact planes for that window's 33 steps.
  - **Full parity:** a fitted rigid frame carrying AVBD wheel rows. That's a larger design.

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
