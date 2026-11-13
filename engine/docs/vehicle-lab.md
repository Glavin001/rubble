# The vehicle test lab

One soft-body car (the 2 kHz explicit solver) in a GPU World, and the things a tester does to it: shoot it, crush it, crash it, drive it into obstacles, throw it down a mountain. Every action is measured.

- Page: `engine/crates/engine-webgpu/vehicle-lab.html`
- Logic (Rust): `engine/crates/engine-world/src/vehicle_lab/` (`mod.rs` the lab, `monitor.rs` the measurements, `mountain.rs` the terrain, `cpu.rs` the CPU reference)
- Wasm host: `engine/crates/engine-world-wasm/src/vehicle_lab.rs`
- Offline: `engine/crates/engine-world/examples/vehicle_lab.rs`
- Recorder: `engine/crates/engine-webgpu/vehicle-lab-record.mjs`, run through `engine/tools/vehicle-lab/run.sh`
- Scenarios: `engine/crates/engine-webgpu/vehicle-lab-scripts/*.json`

The lab changes no solver or kernel behaviour. It places bodies, reads state and counts. The two engine additions are a state reset (`ExplicitBeamsGpu::reset_state`, `World::reset_explicit_vehicles`) and an option of the scene renderer; neither changes what a step computes.

## Using the page

Build the World wasm (CONTRIBUTING.md), serve `engine/crates/engine-webgpu` (or the built site) and open `vehicle-lab.html`.

**Cars.** The generated sedan when a local export is at `__mods/sedan_new` (default), the BeamNG community ETK when it is at `__mods/etkk_tex` (local only, never committed), and the repo's authored sedan always. A car that is not there is greyed out in the menu.

**Modes.**

| Mode | What it does |
|---|---|
| Shoot | The launcher is the camera. Click the scene (or `F`, or Fire, through the reticle): the lab picks what the view ray meets (the car's collision skin as last read, else the ground) and launches from the camera at the elevation that carries the shot, falling under gravity, through that point. A ring marks the aimed point; a speed that cannot reach it is said so. Shape: cuboid with its three sizes, sphere, cylinder (end on or side on). Material: steel 7850, aluminium 2700, concrete 2400, wood 600, rubber 1100 kg/m³, or a mass in kg. Speed 1–100 m/s, spin, bursts of up to 32. Projectiles are World rigid bodies. |
| Compactor | Kinematic plates close on the car: top down, the two sides, or front and rear. Speed 0.02–0.5 m/s, a stop gap, pause, reverse, remove. Force against travel is plotted per plate, with the work done. |
| Crash test | Offset wall (overlap slider, default 40 %), full wall, pole (offset slider), side pole on a sled. Speed, overlap, offset and wall angle are sliders; each change restarts the test in place. |
| Mountain | The car leaves a cliff and falls down an endless rocky slope. Seed, steepness, roughness, launch speed, loose rocks, a start depth. |
| Drive | `W A S D`, `Space` handbrake. Obstacles appear ahead of the car: wall, pole, kerb, ramp, a stack of 25 kg bricks. A start speed. A second, parked car is an option. |

**Common controls.** Slow motion ×1, ×¼, ×1⁄16; pause and single step; restart (`R`); camera (orbit by dragging, chase, tumble, side, top); cutaway (hide the car's material groups; drawing only).

**Every option is a control, and the URL mirrors the controls** (a control's id is its query key), so a run can be shared. A control is one of three kinds:

| Kind | Effect | Which |
|---|---|---|
| live | Applies as the lab runs | Launcher, compactor, camera, slow motion, measurement rate, clean layout |
| restart | Restarts the test in place: the car back to its rest shape at the new start, the scene laid out again. No device, pipeline, download or page load | Mode, scenario, crash sliders, mountain settings, start speed |
| build | Rebuilds the simulation in place, with progress shown; parsed cars are kept in memory | Car, surface contact, second car, air drag, centred damping, rigid sleep, GPU timing, the coupling's room |

The build options need other kernels or another World definition, which is why they rebuild. Surface contact defaults to on for the generated sedan and off otherwise. `?record=1` is the one query key without a control: it is the recorder's frame-locked harness, not an option of the lab.

## What is shown

All numbers come from Rust (`Bench::stats`), read from the car's node, beam and coupling records.

- **Car:** speed, kinetic energy, mechanical energy lost (kinetic plus gravitational, against the first reading), acceleration of the centre of mass now and at its peak (g, over one World step; free fall reads 0), beams yielded and broken, per-part counts and a part-detached log when the car names parts.
- **Deformation:** the largest node displacement from the rest shape's best rigid fit, and the nodes more than 10 cm from it. Only the largest piece still joined by unbroken beams is fitted, and its share of the car's mass is shown.
- **Penetration monitor:** the deepest car node inside any rigid body the coupling holds (now and at its peak), how many nodes are inside, and how many sample points of rigid bodies (corners, centre) are inside the car's collision hull, with their depth. This is known bug B1 as a number.
- **Shots:** per shot its mass, impact speed, the momentum it gave the car (N·s and as a share of its own), the largest one-step force and an outcome (stopped, bounced, slowed, glanced, passed through, embedded).
- **Plates:** travel, force now and at its peak, work, the deepest node inside the plate.
- **Crash:** the stated speed and the speed at first contact.
- **Run:** World steps per second and the share of real time, FPS with frame p50/p95, GPU ms per step and its rigid/vehicle split (GPU timing on), the coupling's use against its room, the share of steps read back, and the last restart's time.
- **Energy ledger:** not exposed by the GPU World. The CPU reference has it (`tests/energy_ledger.rs`); the page says so.

## Scenarios and the recorder

A scenario is JSON: `{ title, mode, scene, mountain, duration, events: [{ t, …command }] }`. Commands are the lab's own (`Bench::command`): `fire`, `plates`, `plateControl`, `obstacle`, `clear`, `drive`. The page also reads `cam` and `slow` events. Pick one in the Scenario menu or with `?script=<name>`.

```bash
engine/tools/vehicle-lab/run.sh shoot-box compactor-top mountain     # clips + stats JSON
engine/tools/vehicle-lab/run.sh --probe shoot-sphere                 # real-time check; console errors fail
engine/tools/vehicle-lab/run.sh --restarts crash:mode=crash          # restart and rebuild timings
engine/tools/vehicle-lab/probe-modes.sh                              # every mode in turn
```

The recorder is frame-locked: one World step per 1/60 s video frame, each step read back. It shows what the physics does, not how fast the page runs; the probe reports that. Output goes to `~/Development/rubble-artifacts/vehicle-lab/`.

Offline, the same scenarios run without a browser:

```bash
cargo run --release -p engine-world --example vehicle_lab -- --script engine/crates/engine-webgpu/vehicle-lab-scripts/shoot-box.json            # CPU reference
cargo run --release -p engine-world --example vehicle_lab -- --mode gpu --script …                                                           # GPU World
cargo run --release -p engine-world --example vehicle_lab -- --measure plates|spheres|mountain [--mode gpu]
cargo run --release -p engine-world --example vehicle_lab -- --mode gpu --crash offset --kmh 56 --overlap 0.4
```

The CPU reference is the explicit solver with `explicit_rigid`'s proxies and **no rigid solver**: between their contacts with the car, bodies fly free, fall and stop on the flat ground. That is the coupling exactly for one projectile, one plate or static rock. It is not the World for piles: a brick stack does not stack there.

## The in-place restart

`Lab::restart` puts the same fleet back at a new start without making a device, buffer or pipeline:

- `ExplicitBeamsGpu::reset_state` rewrites the state buffers (statics, nodes, beams, accumulators, groups, misc, grid) from a new upload of the same vehicles, applies the parameter words that changed (the ground's height, the statics' places), puts the contact plan back to its first frame and rewrites the host settings that live in the rewritten records (the coupling header, contact margins). A fleet of another layout is refused and nothing changes.
- `World::reset_explicit_vehicles` resets the coupling's clock, pending impulses, sleep bits and broadphase proxies with it.
- The lab removes its bodies, moves the scene's own (the ground slab and fixtures go far away for the mountain) and lays the scene out again.

So that every lab car of one template has one layout, the arena's ramp and kerb are in the car's solver in every scene (far overhead on the mountain).

`tests/vehicle_lab_reset.rs` holds it: a lab that took a shot, ran a 40 km/h wall and was restarted steps **bit for bit** as a lab just built (nodes, beams and origin after 40 steps), and the side-pole sled delivers the car within 1 km/h of the stated speed.

| Measured | Result |
|---|---|
| Restart, native (authored sedan, 987 nodes) | 2.6–7.2 ms |
| Restart in Chrome, a crash slider change | 1.3–1.8 ms in `restart`; stepping again 2–25 ms after the change |
| Flat arena to mountain, in place, Chrome | 3.6–5 ms; stepping again after 61–126 ms (11 chunks made) |
| Page loads during 5 restarts, a rebuild and a scene change | 0 |
| First load, generated sedan (local server, headless) | 3.0–3.6 s |
| First load, authored sedan | 29 s with two paints built up front; one paint is now built and the second only for a second car |
| In-place rebuild (surface contact toggled) | 21 s: the surface kernels compile on a new device |

**The generated sedan's model is 139 MB** (`sedan.rskin` 122.7 MB, `sedan.glb` 16.3 MB). The page downloads and parses it once and keeps it. Half-float weights and 16-bit node indices would cut the skin about 2.5×; that format lives in the skin writer and reader (`engine-softbody` `flexskin`), which this task did not touch. It is a follow-up, with a rest-fidelity check of 1e-4 m.

**Why a rebuild is slow.** A World on the web makes its own device (`web_device`), so a rebuilt World compiles its pipelines again. A World built on an existing device would make the rebuild a few seconds.

## The mountain

`mountain.rs` makes the slope in chunks along the fall line. A chunk is a pure function of (seed, index), 24 m long:

- **Ground:** 3 m thick slabs along a piecewise-linear profile. A chunk is a slope with a knee, a shelf with a near-vertical drop, or a gentler run with a 65–75° cliff face; every profile ends on the fall line, so chunks join.
- **Canyon walls** keep the car on the mountain.
- **Rocks** stand on the slabs: spikes (convex hulls from a palette of 12, so the hull arena stays bounded), boulders (spheres) and blocks (boxes). Roughness sets how many, and how often ledges and cliffs come.
- **Loose rocks** (dynamic bodies) are optional and off by default: contact with moving bodies has no penetration push-back (B1).

Chunks are made 8 ahead of the car's chunk (192 m, beyond the fog) and removed 2 behind, at most one new chunk per step.

**Everything is static**, held by the coupling as immovable proxies, which use the solver's own static-shape contact rule. The explicit solver has no heightfield and its own static shapes are fixed at upload, so World static bodies are what it can collide with at runtime.

**Decisions and their reasons:**

- **Default steepness is 50°.** The rock's contact surface has kinetic friction 0.75 (37°) and static friction 1.2 (50°). On the CPU reference the car stopped after 32 m at 42° and after 36 m at 46°; at 50° it kept going (129 m in 25 s).
- **Rockslide.** A car at rest for 2 s gets a 3.4 t boulder thrown at it from 8 m up the slope (a dynamic body, on by default, a checkbox). This keeps the fall going by physical means; the run's log counts them.
- **The car starts just past the lip, in the air, at the launch speed.** Raycast wheels do not see rigid bodies (F7), so the car cannot drive on the plateau; its own ground plane is put 100 km down.
- **Precision.** A chunk's shapes are placed about the chunk's own origin (f64), each as one static body at an f32 World position. The coupling subtracts the car's origin from a body's position, and the car's nodes are relative to that origin (rebased every 100 m), so contact is computed in small numbers. A static body's f32 position rounds once (0.5 mm at 5 km) and stays there: an offset, not jitter. The "start" control begins a run kilometres down the fall line to test this directly.

## Engine measurements

Authored sedan (987 nodes, 6,865 beams, 1,450 kg) unless said, M3 Max. `examples/vehicle_lab.rs --measure …`.

### (a) Kinematic plates

A kinematic body is an infinite-mass proxy. The coupling reads its velocity from the body record, and node contact uses the static-shape rule on the node's velocity relative to the plate (`explicit_rigid.rs` `node_contact_with`, the `is_static` branch). The World moves the plate by its velocity each step; the lab's travel is that integral.

**A slow press does not sink into the car.** Deepest node inside a plate, 0.15 m of clearance then 0.30 m of crush:

| Plate speed | Top, CPU | Top, GPU World | Sides, CPU | Sides, GPU World |
|---|---|---|---|---|
| 0.02 m/s | 0 mm | 2 mm | 2–10 mm | 6–9 mm |
| 0.05 m/s | 0 mm | 3 mm | 3 mm | 3–4 mm |
| 0.10 m/s | 0 mm | 0 mm | 5–7 mm | 12–15 mm |
| 0.20 m/s | 0 mm | 1 mm | 12 mm | 6–10 mm |
| 0.50 m/s | 4 mm | 3 mm | 10–15 mm | 2–10 mm |

Depth does not grow with speed and stays under 15 mm. B1's 25 cm is a dynamic body's problem: a kinematic plate has no impulse to wait for and its motion inside a window is the motion the World gives it.

Forces, for the record (GPU World): the top plate peaks at 159–171 kN over 0.30 m of roof crush (6.4–9.0 kJ); the side plates at 1.51–1.67 MN each (226–262 kJ, 201–212 beams broken). The side figure is the car's authoring (the plates meet the sills and hubs), not a lab artefact; real quasi-static side crush is an order lower.

**The alternative, plates as the solver's own static shapes, does not exist as such.** The solver's static shapes have no velocity and are fixed at upload on the GPU. Teleported each 2 kHz step on the CPU, a top plate's nodes sink 118–150 mm (the plate's 150 mm half thickness, that is, through it) at every speed from 0.02 to 0.5 m/s. Kinematic World bodies are the right plates.

### (b) Projectiles

A 0.1 m steel sphere (32.9 kg) into the left side at door height, two aim points:

| Surface contact | Speed | CPU reference | GPU World |
|---|---|---|---|
| off | 20 m/s | passes through, gives 21–45 % of its momentum | passes through, 12–38 % |
| off | 40 m/s | passes through, 7–24 % | passes through, 7–31 % |
| off | 60 m/s | passes through, 6–16 % | passes through, 5–25 % |
| on | 20 m/s | bounces (−3 to −5 m/s), never inside the hull | bounces (−2 m/s), never inside |
| on | 40 m/s | stopped, 13–20 beams broken | stopped or bounced, 15–21 broken |
| on | 60 m/s | stopped, 25–55 broken, peak 371–424 kN | stopped or bounced, 27–70 broken, peak 371–1,069 kN |

- **Node-only contact lets a small projectile through at every speed.** The sphere is smaller than the gaps between nodes; what it gives the car is what the few nodes it meets take.
- **Surface contact stops it at every speed tested**, with no node inside the sphere and the sphere's centre never inside the car's hull.
- **No tunnelling at 60 m/s** (1 m per World window, 3 cm per 2 kHz step): the proxy integrates inside the window at 2 kHz, and surface contact casts fast bodies (spec §38.4). `examples/ccd_cannonball.rs` covers 30–600 m/s.
- **Generated sedan, surface on, Chrome:** 20 m/s bounced, 40 m/s stopped (peak 322 kN), nothing inside.
- **A large, heavy projectile still sinks in with node-only contact:** the 188 kg steel cuboid at 30 m/s puts its corners 0.44 m (GPU) to 0.58 m inside the hull.
- **The World and the CPU reference differ on what a stopped projectile does next.** The same cuboid gives the car 98 % of its momentum on the CPU reference and 70 % in the GPU World, leaving at 9 m/s. On the CPU reference a proxy carries on from where its window ended; in the World the rigid step moves the body from its own pose with the impulse applied a step later (B1's first mechanism).

Cost: surface contact with the generated sedan is 3.7–4.0 ms of GPU per step idle and 6.5–8.9 ms in contact (headless Chrome, per-phase timing on, which itself adds).

### (c) Budgets and capacities

- **The coupling's room** per car is a build option (default 256 bodies and shapes and 4,096 hull planes; 1,024 shapes at most). More in one window fails the step by name ("more colliders than the shapes room"), and the page shows the error. The HUD shows shapes now, the peak and the room, and turns red past 80 %.
- **Peak use seen:** 1–2 shapes in Shoot and Compactor, 7 on the mountain's start (28 on the CPU reference's 40 s run), where only shapes near the car's box are proxies.
- **The lab's own limits** are refused with a message, not degraded: 48 live projectiles (the oldest is removed for a new one), 64 queued shots, bursts of 32, 240 obstacle bodies, brick stacks of 200, 1,000 script events, shot sizes of 2 cm to 4 m, speeds to 100 m/s.
- **Kernel loops** take the engine's per-frame pass budgets (`explicit_loop_cap`); a stop fails `World::step`, which the page shows. The lab adds no kernel loop.
- **Device loss** shows a message with a Rebuild button.

## What the engine must change for the lab to be physically right

Each is the user's decision; none was made here.

1. **Penetration response against moving bodies (B1).** Needed for heavy projectiles, loose rocks and brick stacks: the cuboid's corners end 0.44–0.58 m inside the car.
2. **The window mismatch between proxy and World body** (B1's first mechanism): momentum given differs by 28 points between the CPU reference and the World for the same shot.
3. **Surface contact as the default for cars that meet small or sharp bodies.** Off, a 0.1 m sphere passes through at 20–60 m/s. On, it costs 2–5 ms more GPU per step in contact.
4. **Raycast wheels that see rigid bodies (F7).** The mountain's car cannot roll, ramps and kerbs spawned at runtime stop the body instead of lifting the wheels, and the plateau cannot be driven on. The arena's fixed ramp and kerb are in the car's own solver for this reason.
5. **Static shapes that can change at runtime** in the car's solver (or the F7 fix), so terrain and obstacles get the wheel contact too.
6. **A World on an existing device from the wasm host**, so a rebuild does not compile every pipeline again (21 s).
7. **An energy ledger on the GPU**, to close plate work against energy absorbed in the page.
8. **The authored sedan's drag (B2)** takes 1.3–2.0 km/h off a crash test's stated speed over a 1 m run-up; the generated sedan arrives at 56.0 of 56 km/h.

## Pacing and what the page does each frame

- **The car is drawn from the solver's GPU buffers**; drawing never waits on a readback. Only the HUD's numbers, the click's pick and the camera's target use readbacks.
- **Readbacks ("Measurements")**: 15 times a second by default, and every step while a projectile is in its first 4 s, a plate exists, a crash preset has not yet touched, or a scenario runs (exact impulses when they matter). "Every step", "15 Hz" and "off" are options. The HUD's text is rebuilt 4 times a second.
- **The camera's target is carried forward by the reading's age in simulated time** and the frame's lead, so it moves with the car drawn (the cause of the brick ring page's camera jitter, B3).
- **Steps are owed, not lost.** `World::step` is refused while every readback slot is in flight (3 by default). On a late frame the page lost that step: 42 steps/s at 75 FPS and 29 at 48 FPS were seen on a real display. The lab's World now allows 8 steps' results in flight and a refused step is owed to the next frame (up to 3 beyond the pacer's 4 a frame). On the user's display afterwards: 60 steps/s (100 % real time) at 50 FPS.
- **`FramePacer::advance` takes how many times slower than real time** (4 for ×¼), not the time scale.

Headless Chrome, M3 Max, generated sedan (554 nodes, 4,228 beams, 1,492 kg), surface contact on, per-phase GPU timing on (which adds cost), **before** the 8-slot gate:

| Mode | World steps/s | FPS | Frame p50 / p95 | GPU per step (rigid · vehicle) |
|---|---|---|---|---|
| Shoot, idle | 60 | 60 | 16.7 / 16.8 ms | 3.7–4.0 ms (0.2–0.8 · 3.4–3.6) |
| Compactor, top plate pressing | 33 (×1) | 46–58 | 16.7 / 33.4 ms | 7.4–7.6 ms (1.0 · 6.3–6.7) |
| Crash, 56 km/h offset wall, after the strike | 26–29 | 43–44 | 16.7 / 33–50 ms | 8.7–8.9 ms (0.7–0.9 · 7.5–7.9) |
| Mountain | 27–33 | 43–50 | 16.7 / 50 ms | 6.5–8.3 ms (0.9–1.1 · 6.2–6.9) |
| Drive into a wall | 54 before, 30 after the strike | 51–60 | 16.7 / 16.8–33 ms | 3.8 → 6.9 ms |

In contact the car's frame costs 6–8 ms of GPU with surface contact on, which with drawing left the old 3-slot gate closed every other frame. These rows must be measured again with the 8-slot gate; that run was held back while the user tested live.

## Status

Verified: the Rust unit tests (`cargo test -p engine-world --lib vehicle_lab`), the restart tests on the GPU (`--test vehicle_lab_reset`), the offline scenarios and measurements on the CPU reference and the GPU World, and in headless Chrome: Shoot (clicks fire and log), the scenario runs, the restart and rebuild timings, flat to mountain in place, 60 steps/s idle, no console errors.

Not yet verified (held while the machine was in use): click aim in Chrome (verified on the CPU reference: three picks at 20, 20 and 8 m/s struck the car); every mode's pace with the 8-slot gate; the coupling-room error shown on the page; a tiny loop budget ending a step with its error; the mountain for 5 minutes and from 8 km down on the GPU (on the CPU reference a 20 s run from 8 km down had no non-finite state); the clips of each mode; the full `build-web.sh` (this worktree's fracture assets are Git LFS pointers, so only the World wasm was built).

## Verified

See the report in the pull request or commit log for the clips and the per-mode real-time numbers; they are kept in `~/Development/rubble-artifacts/vehicle-lab/` (`*.mp4`, `*.stats.json`, `stills/`).
