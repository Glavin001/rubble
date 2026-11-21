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

**Cars.** The generated sedan at `__mods/sedan_new` (default; `build-web.sh` makes it in every build by running the generator, so the deployed site has it), the BeamNG community ETK when it is at `__mods/etkk_tex` (a community mod: local only, never committed or deployed), and the repo's authored sedan always. A car that is not there is greyed out in the menu with the reason ("not generated: run engine/scripts/build-web.sh", "community mod: local only").

**Modes.**

| Mode | What it does |
|---|---|
| Shoot | The launcher is the camera. Click the scene (or `F`, or Fire, through the reticle): the lab picks what the view ray meets (the car's collision skin as last read, else the ground) and launches from the camera at the elevation that carries the shot, falling under gravity, through that point. A ring marks the aimed point; a speed that cannot reach it is said so. Shape: cuboid with its three sizes, sphere, cylinder (end on or side on). Material: steel 7850, aluminium 2700, concrete 2400, wood 600, rubber 1100 kg/m³, or a mass in kg. Speed 1–100 m/s, spin, bursts of up to 32. Projectiles are World rigid bodies. |
| Compactor | Kinematic plates close on the car: top down, the two sides, or front and rear. Speed 0.02–0.5 m/s, a stop gap, pause, reverse, remove. Force against travel is plotted per plate, with the work done. |
| Crash test | Offset wall (overlap slider, default 40 %), full wall, pole (offset slider), side pole on a sled. Speed, overlap, offset and wall angle are sliders; each change restarts the test in place. |
| Mountain | The car leaves a cliff and falls down an endless rocky slope. Terrain (rolling gully, the default, or ledges and cliffs), seed, steepness, roughness, launch speed, loose rocks, a start depth. The launcher works here too: click the scene (or `F`) to shoot, for example to knock a stuck car loose. |
| Drive | `W A S D`, `Space` handbrake. Obstacles appear ahead of the car: wall, pole, kerb, ramp, a stack of 25 kg bricks. A start speed. A second, parked car is an option. |

**Common controls.** Slow motion ×1, ×¼, ×1⁄16; pause and single step; restart (`R`); camera (overhead chase, orbit by dragging, chase, tumble, side, top); with more than one car, the car the camera follows (`V` cycles; the readouts stay the first car's); cutaway (hide the car's material groups; drawing only).

**The camera.**

- **It follows its own reading.** Every drawn frame copies the followed car's nodes and origin record back (tens of kB, its own three buffers), and the lab (`Lab::ingest_follow`) makes its centre of mass, velocity and heading; the camera carries that forward by the reading's age. It used to follow the measurements' reading, which is taken 15 times a second and skipped while the GPU is busy: the camera stopped (its forward carry was capped at 0.25 s) and then jumped to catch up. The reading's age is in `__state().camAge` and in the recorder's `--probe` lines.
- **Nothing drawn hides the car.** Each frame the lab casts the view ray from the car to the eye (`Bench::clear_view`) against the mountain's rock (floor, banks, walls, humps, boulders: the live chunks' shapes, kept as they are made) and the arena's ground, and the eye comes in 0.6 m in front of what it meets. It comes in at once and goes back out eased over 0.5 s.
- **Overhead chase** (the mountain's default): in the fall line's frame, 9 m over the slope and 10 m up it, looking a little down the slope; the frame is fixed, so a tumbling car does not swing the view. On the flat arena: 9 m up and 7 m behind the car's travel.

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

`mountain.rs` makes the slope in chunks along the fall line. A chunk is a pure function of (seed, index), 24 m long. There are two terrains (`mountain.terrain`, the page's "terrain" control):

**Rolling** (the page's default, added 2026-11-16 because cars stopped on the ledges' shelves): a gully down a mountainside with no shelf or cliff, so the cars keep falling and bouncing.

- **Ground:** the fall line swells along its normal by three seeded waves (71, 29 and 13 m), a function of the arc alone, so chunks join exactly; 4 m slabs follow it. The local slope stays within 4–14° of the mean (roughness) and never under 38° (over the rock's 37° kinetic friction angle), faded in over the first 30 m so every run leaves the lip the same way. A test holds both over 200 chunks, 8 seeds, 3 roughnesses and 3 steepnesses.
- **Cross-section:** a flat bed half the width, and two banks rising 18° into the canyon walls (so nothing falls between bank and wall).
- **Rocks, all rounded:** humps (spheres of 5–12 m buried to 0.2–1.1 m: faces at most 25° off the ground, a ramp that throws the car up and aside, never a step), part-buried boulders (0.35–1.1 m), and crags on the banks only (they turn a car back into the gully).
- A car can still come to rest (in the crease behind a boulder, say): the rockslide, or a shot from the launcher, sets it going again.

**Ledges** (the first mountain; `MountainSpec::default()` keeps it so the offline tools, the fixtures and the measurements below keep their scene; on the page `?terrain=ledges`):

- **Ground:** 3 m thick slabs along a piecewise-linear profile. A chunk is a slope with a knee, a shelf with a near-vertical drop, or a gentler run with a 65–75° cliff face; every profile ends on the fall line, so chunks join.
- **Canyon walls** keep the car on the mountain.
- **Rocks** stand on the slabs: spikes (convex hulls from a palette of 12, so the hull arena stays bounded), boulders (spheres) and blocks (boxes). Roughness sets how many, and how often ledges and cliffs come.
- **Loose rocks** (dynamic bodies) are optional and off by default: contact with moving bodies has no penetration push-back (B1).

Both terrains:

- **Shooting on the mountain.** The launcher panel shows in Mountain mode, and a click (or `F`) fires as in Shoot. The pick meets the car's skin, else the mountain's drawn rock (its boxes and spheres, `MountainSpec::ray`), else the air beside the car. The canyon has no ceiling; its walls go on undrawn above their drawn rock to 500 m over the fall line (`WALL_TOP`), so nothing thrown up leaves it, and a shot from a camera beyond a wall starts where the view ray leaves the wall. A shot is removed 30 s after it is fired (`MOUNTAIN_SHOT_TTL`): one that missed rolls on down without end. The page's stability stop keeps only its non-finite check in Mountain mode, since shots put energy into the cars.

Chunks are made 8 ahead of the car's chunk (192 m, beyond the fog) and removed 2 behind, at most one new chunk per step. A chunk is also kept, with one either side, while any node of any car lies on it at the last reading (`Bench::held`: the first car's nodes at every reading, the other cars' with their beams, about 10 a second, `ReadPlan::fleet_nodes`), so a part that comes to rest far behind its car keeps its ground; before, its chunk was removed under it and it fell without end. More than `MAX_HELD_CHUNKS` (48) such chunks outside the cars' own windows stops the run with an error rather than drop ground. Each car's nodes are put in up to 16 coupling groups by where they lie (`coupling_groups`, 4 m cells; `ExplicitBeamsGpu::set_coupling_groups`), so the coupling's window per group stays about its own nodes: a piece's box spanned every node of the piece, and the loose parts' label spanned from a part hundreds of metres back to the car, which put every collider between them in the car's window (more than its 1,024 shapes).

**Everything is static**, held by the coupling as immovable proxies, which use the solver's own static-shape contact rule. The explicit solver has no heightfield and its own static shapes are fixed at upload, so World static bodies are what it can collide with at runtime.

**Decisions and their reasons:**

- **Default steepness is 50°.** The rock's contact surface has kinetic friction 0.75 (37°) and static friction 1.2 (50°). On the CPU reference the car stopped after 32 m at 42° and after 36 m at 46°; at 50° it kept going (129 m in 25 s).
- **Rockslide.** A car at rest for 2 s gets a 3.4 t boulder thrown at it from 8 m up the slope (a dynamic body, on by default, a checkbox). This keeps the fall going by physical means; the run's log counts them.
- **The car starts just past the lip, in the air, at the launch speed.** Its own ground plane is put 100 km down. (Written when raycast wheels did not see rigid bodies, F7; they do now, `wheels_hit_bodies`, so the plateau could be driven on: the start is unchanged.)
- **Precision.** A chunk's shapes are placed about the chunk's own origin (f64), each as one static body at an f32 World position. The coupling subtracts the car's origin from a body's position, and the car's nodes are relative to that origin (rebased every 100 m), so contact is computed in small numbers. A static body's f32 position rounds once (0.5 mm at 5 km) and stays there: an offset, not jitter. The "start" control begins a run kilometres down the fall line to test this directly.

## The derby

`vehicle-lab.html?mode=derby&cars=4` (the Derby mode button): a demolition derby of the generated sedan in a walled arena, one car driven by the player and the rest by an AI. The scene, the AI and the round are Rust (`vehicle_lab/derby.rs`, `engine_softbody::drive_ai`) so another host can reuse them; the page forwards keys and draws.

- **Arena** (`derby::arena_fixtures`): a 70 m floor (the flat arena's ground) closed by eight 5 m walls in an octagon, and two ramps pointing at the middle. Walls and ramps are the cars' own static shapes (the raycast wheels see them, the nodes get the solver's static contact) and fixture bodies in the World (the props collide with them, the renderer draws them). No ceiling. The Derby is a build of its own: its cars' static shapes are other ones than the flat arena's, so switching to or from it rebuilds in place.
- **Obstacles** (`derby::arena_props`, the "obstacles" control, a restart): two brick walls (6 × 5 blocks of 1 × 0.5 × 0.5 m) that break apart when rammed, tyre stacks, barrels, crates and two 2.3 t concrete blocks, all ordinary dynamic rigid bodies. Off, the props are not made and the ramps go far under the floor (the same layout, so it is a restart, not a rebuild).
- **Cars** start round the edge, 27 m from the middle, facing it; up to 32 (the cars control: 4, 6, 8, 12, 16, 32).
- **Poses.** Every drawn frame and every frame that took a step the page copies each car's origin record and raycast-wheel records (a few kB) and hands them to the lab (`Lab::derby_plan`, `ingest_derby`): centre, forward, up and speed from the wheels on the main body.
- **AI** (`DerbyAi`): each car hunts a target, picked again every 4 to 8 s by distance with a penalty for cars already hunted by others (so they spread) and a preference for the player that grows with aggression; it leads the target by its motion, rams flat out inside 14 m, backs off a run-up's length (6 to 12 m) after a hit and rams again, reverses out when stuck (and goes forward on full lock when stuck both ways), turns back from the walls, steers round the brick walls, concrete blocks and the ramps' sides, and now and then (12 to 30 % of retargets) takes a ramp. A flipped or knocked-out car gets no input. Deterministic from the seed (the "AI seed" control); it sets drive input as keys do and never touches the solver.
- **Round** (`DerbyGame`): a car is out on its roof or side for 3 s, with two wheels off its main body, thrown out of the arena, or not having moved 2 m in 12 s. The last car in wins; at 3 minutes the least damaged car still in. If every car left would go out on the same step (a pile that stopped together), the least damaged stays in. Damage is the car's beams bent plus three times its beams broken, over half its beam count. After the round every car brakes; "New round" (or `R`) restarts in place.
- **Controls** (all on the page, applied as it runs unless noted): cars (rebuild), "you drive car 1" (off: spectator, every car the AI's), AI aggression, obstacles (restart), AI seed (restart). Car 1 drives with the Drive mode's keys (`W` `A` `S` `D` or the arrows, `Space` handbrake); the chase camera follows it, and `V` cycles the followed car.
- **Surface contact** is off by default in the Derby, as on the mountain: the walls and ramps are the solver's static shapes, and without it the fleet-wide passes keep many cars faster.
- **Wheels on debris and on other cars** (known-bugs F7, fixed 2026-11-18; the "wheels on bodies" checkbox, on by default, `?wheelsHitBodies=0` off): the raycast wheels stand on the props (bricks, crates, barrels, blocks, tyre stacks) and on the other cars' bodies, and push them back. Before, a car whose body came to rest on fallen bricks, or whose crushed nose rode up onto another car, had its driven wheels in the air and stayed put whatever its input; the round then knocked it out ("stopped moving"). The probe now reports each car's wheels touching anything (`grounded`), and `tools/vehicle-lab/derby-beached.py` sums the time a car was beached. The AI still aims at its target's flank and staggers its start: kept because they make a better round (side hits, cars arriving one at a time), not only because of the limit.

**Measured** (2026-11-18, M3 Max, headless Chrome, `vehicle-lab-record.mjs --probe --secs 30 --eval tools/vehicle-lab/derby-probe.js`, spectator, the page's own real-time loop, GPU timing per step on, which itself costs a few ms a step). A World step is 1/60 s of physics (33 steps of the 2 kHz solver); real time is 60 steps a second.

| Cars | GPU ms a World step, median (p95), main | Steps/s reached, main | With PR 128 (contact levels hit once) merged |
| --- | --- | --- | --- |
| 4 | 18.4 (21.6) | 30–34 | 14.4 (17.4) ms, ~36/s |
| 8 | 22.3 (27.9) | 21–30 | 17.8 (22.1) ms, ~29/s |
| 12 | 21.6 (29.0) | 20–24 | not measured |
| 16 | 25.8 (34.4) | 16–20 | 21.8 (26.8) ms, ~18/s |
| 32 | 26.2 (37.0) | 11–13 | 24.3 (29.8) ms, ~11/s |

**Wheels on bodies** (2026-11-19, the same probe, 90 s at 4 cars and 60 s at 8, the scripted player on; `tools/vehicle-lab/derby-beached.py` counts a car beached while it is in the round, asked to move, under 2 km/h and with at most one wheel touching anything):

| Cars | Wheels on bodies | GPU ms a World step, median (p95) | Simulated | Beached | Out |
| --- | --- | --- | --- | --- | --- |
| 4 | on | 13.6 (18.3) | 53 s | 0 car-s | 1 (the player, "stopped moving" against the wall: W held into it) |
| 4 | off | 12.1 (15.8) | 58 s | 0.6 car-s | 1 (the same) |
| 8 | on | 25.9 (31.2) | 24 s | 0 car-s | 0 |
| 8 | off | 18.4 (22.0) | 30 s | 0 car-s | 4 ("stopped moving", all wheels down: pinned, not beached) |

In these rounds no car was left beached either way: the flank aim and the staggered start keep the 4-way head-on pile from forming, and the cars that stop are pinned against walls or each other on their wheels. The setting costs 12 % of a World step at 4 cars and 41 % at 8 (the wheels' rays against the other cars in a contact group, and the wheels-ahead schedules it turns off; explicit-vehicles-in-world.md §13). Clip and stills of a car driving up a rigid ramp body onto a layer of bricks (`vehicle-lab-scripts/drive-ramp-bricks.json`): `~/Development/rubble-artifacts/derby-wheels/`.

With GPU timing off, 4 cars ran at 59 steps a second until they met and about 30 once in contact: car-to-car contact is what costs. No count holds 60 (or 120) steps a second once cars are in contact; 32 cars are also limited by the page (readbacks and drawing 32 cars). The props (about 100 rigid bodies) cost 1–2 ms of rigid solver a step.

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
4. **Raycast wheels that see rigid bodies (F7): done 2026-11-18** (`wheels_hit_bodies`, explicit-vehicles-in-world.md §13). Ramps and kerbs spawned at runtime now lift the wheels; the arena's fixed ramp and kerb stay in the car's own solver.
5. **Static shapes that can change at runtime** in the car's solver (the F7 fix now gives World bodies the wheel contact).
6. **A World on an existing device from the wasm host**, so a rebuild does not compile every pipeline again (21 s).
7. **An energy ledger on the GPU**, to close plate work against energy absorbed in the page.
8. **The authored sedan's drag (B2)** takes 1.3–2.0 km/h off a crash test's stated speed over a 1 m run-up; the generated sedan arrives at 56.0 of 56 km/h.

## Pacing and what the page does each frame

- **The car is drawn from the solver's GPU buffers**; drawing never waits on a readback. Only the HUD's numbers, the click's pick and the camera's target use readbacks.
- **Readbacks ("Measurements")**: 15 times a second by default, and every step while a projectile is in its first 4 s, a plate exists, a crash preset has not yet touched, or a scenario runs (exact impulses when they matter). "Every step", "15 Hz" and "off" are options. The HUD's text is rebuilt 4 times a second.
- **The camera's target is carried forward by the reading's age in simulated time** and the frame's lead, so it moves with the car drawn (the cause of the brick ring page's camera jitter, B3).
- **Steps are owed, not lost.** `World::step` is refused while every readback slot is in flight (3 by default). On a late frame the page lost that step: 42 steps/s at 75 FPS and 29 at 48 FPS were seen on a real display. The lab's World now allows 8 steps' results in flight and a refused step is owed to the next frame (up to 3 beyond the pacer's 4 a frame). On the user's display afterwards: 60 steps/s (100 % real time) at 50 FPS.
- **`FramePacer::advance` takes how many times slower than real time** (4 for ×¼), not the time scale.
- **Physics first: the drawing gives way** (`rubble::DrawPacer`, `draw_pacer.rs`; the controls "drawing gives way to physics", "at least N pictures a second" and the render scale, live). A picture is never queued behind one still on the GPU or in a frame whose step was refused (down to the floor, 30 pictures a second by default); while steps are refused the drawing drops a level at a time (every display frame, 60, 30 a second, then 75 % and 50 % of the pixels) and comes back when they are not. A frame not drawn still reads back what its steps left (measurements, the derby's AI). The status bar shows the pictures drawn a second and the level.
- **Skin cars of one model together** (a control, live: together (wide, the default), together (loop), one by one). The cars of one model share the skin's bind buffer (157 MB for the generated car); together, their skin is one dispatch (`showcase-car-render.mjs` `skinBatchWgsl`: the same skin code, each car's buffers a range of a pooled one), so the binds are read from memory once for every car instead of once a car. The skinned vertices are the same bit for bit (`tools/vehicle-lab/skin-batch.sh` compares them with the base commit's kernel on the same paused state). Measurements: `docs/measurements/vehicle-perf-first-principles.md` §8.41.
- **Steps in flight** (a control, 3 by default; `setStepsInFlight`, pacing only): with explicit vehicles the World takes a step only while fewer than this many are still waiting for their readback. Chrome lands a readback well after the GPU is done, and with 2 the GPU sat idle. Measurements of both (derby at 4 and 8 cars, the mountain at 8, before and after): `vehicle-perf-first-principles.md` §8.38.

Headless Chrome, M3 Max, generated sedan (554 nodes, 4,228 beams, 1,492 kg), surface contact on, per-phase GPU timing on (which adds cost), **before** the 8-slot gate:

| Mode | World steps/s | FPS | Frame p50 / p95 | GPU per step (rigid · vehicle) |
|---|---|---|---|---|
| Shoot, idle | 60 | 60 | 16.7 / 16.8 ms | 3.7–4.0 ms (0.2–0.8 · 3.4–3.6) |
| Compactor, top plate pressing | 33 (×1) | 46–58 | 16.7 / 33.4 ms | 7.4–7.6 ms (1.0 · 6.3–6.7) |
| Crash, 56 km/h offset wall, after the strike | 26–29 | 43–44 | 16.7 / 33–50 ms | 8.7–8.9 ms (0.7–0.9 · 7.5–7.9) |
| Mountain | 27–33 | 43–50 | 16.7 / 50 ms | 6.5–8.3 ms (0.9–1.1 · 6.2–6.9) |
| Drive into a wall | 54 before, 30 after the strike | 51–60 | 16.7 / 16.8–33 ms | 3.8 → 6.9 ms |

The mountain's cost per World step, natively, and the separation cull that took the surface pass's share down (the pair path's listing overflowed on turned slabs): `measurements/lab-mountain-perf.md`. Native timings of the lab need `RUBBLE_SURFACE_SPLIT=0` (§1a there).

In contact the car's frame costs 6–8 ms of GPU with surface contact on, which with drawing left the old 3-slot gate closed every other frame. These rows must be measured again with the 8-slot gate; that run was held back while the user tested live.

## Status

Verified: the Rust unit tests (`cargo test -p engine-world --lib vehicle_lab`), the restart tests on the GPU (`--test vehicle_lab_reset`), the offline scenarios and measurements on the CPU reference and the GPU World, and in headless Chrome: Shoot (clicks fire and log), the scenario runs, the restart and rebuild timings, flat to mountain in place, 60 steps/s idle, no console errors.

Not yet verified (held while the machine was in use): click aim in Chrome (verified on the CPU reference: three picks at 20, 20 and 8 m/s struck the car); every mode's pace with the 8-slot gate; the coupling-room error shown on the page; a tiny loop budget ending a step with its error; the mountain for 5 minutes and from 8 km down on the GPU (on the CPU reference a 20 s run from 8 km down had no non-finite state); the clips of each mode; the full `build-web.sh` (this worktree's fracture assets are Git LFS pointers, so only the World wasm was built).

## Verified

See the report in the pull request or commit log for the clips and the per-mode real-time numbers; they are kept in `~/Development/rubble-artifacts/vehicle-lab/` (`*.mp4`, `*.stats.json`, `stills/`).
