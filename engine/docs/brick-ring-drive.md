# Brick Ring Drive

Page: `engine/crates/engine-webgpu/brick-ring-drive.html`. Scene: `rubble::brick_ring_drive`
(`engine-world/src/brick_ring_drive.rs`). Wasm host: `engine-world-wasm/src/ring_drive.rs`.
Probe: `engine-world/examples/brick_ring_drive_probe.rs`. Recorder:
`engine-webgpu/brick-ring-drive-record.mjs`.

A soft-body car driven by the keys through the 110k brick ring of the rigid demos, in one
GPU World: the 2 kHz node–beam car (the explicit solver, BeamNG-style deformation) on
raycast wheels, and 110,440 AVBD rigid bricks, coupled both ways
(`explicit-vehicles-in-world.md`).

## Decisions

- **The ring is the demo's.** `RingSpec::demo()` reproduces `engine-scenes`
  `brick_ring_110k` / demo.html `?scene=ring110k` brick for brick (110,440 bricks of
  1 × 0.5 × 0.5 m, radius 80 m, 40 courses, a row less every 4 courses from 10). The World
  is the demo's (`ring_world_def`: 1/60 s, 4 iterations, persistent contacts, sparse colour
  start 10; `ring_world_def_is_the_demos` holds the values demo-config.mjs gives). A ring
  scaled to half size (0.5 × 0.25 × 0.25 m bricks) did not settle under the same solver
  settings (107k of 110k bodies still awake after 2.5 s, then a warm-hash overflow), so the
  scale stays 1: the bricks are big blocks next to the car.
- **Brick mass.** The demo's bricks weigh 0.25 kg (three-avbd's density 1 kg/m³): a 1.3 t car
  does not notice them. The drive's bricks are 100 kg/m³ (25 kg a block, a hay bale's
  density), stated in `drive_ring()`. Measured (native, ETK from 22 m/s): the car punches
  through the 5.2 m base, slowing from 27 to 7 m/s, 18 beams broken.
- **Sleep on.** The demo runs the ring without sleep for three-avbd parity; the drive turns the
  World's sleep on (0.5 s, demo.html's `?sleep=on`; `?sleep=0` on the page restores the
  demo's). The ring settles in about 3 s and only what the car disturbs is awake.
- **Raycast wheels.** The cars drive on the explicit solver's raycast wheels on their hubs
  (`engine_softbody::explicit_drive`, as softbody.html's drive scenes): the converted ETK has
  no tyres (BeamNG's shared content). The rays meet the vehicles' own ground, not the
  bricks: a car drives over loose bricks on its body's nodes.
- **The ground slab** is the demo's static box, in its own collision category that the
  vehicles' mask leaves out (the cars keep their own ground plane at its top).

## Results (2026-11-07, M3 Max, other agents' jobs on the machine)

**Baseline: the ring alone.**

| Run | Per World step |
|---|---|
| Chrome, demo.html `?scene=ring110k` (no sleep, the ball), frame-locked wall time | 10.6 ms |
| Chrome, the same with `&sleep=on` | 4.8 ms |
| Native, the demo's ring (no sleep, the ball), GPU | 13.0 ms (16–17 under load) |
| Native, the drive's ring alone (sleep on, no ball, 25 kg blocks), GPU, two runs | 4.5, 4.7 ms |

**With a car, in Chrome** (`brick-ring-drive-record.mjs --probe`, `?auto=1&timing=phases`,
headless, 60 Hz):

| Car | World steps/s | GPU per step (rigid · vehicle) |
|---|---|---|
| Soft-body sedan, cruising, ring asleep | 60 (100 %) | 4.2 ms (1.4 · 2.8) |
| Soft-body sedan, the first 3 s (the ring settling, 110k → 27k awake) | 56–60 | 7.7–12.1 ms |
| Soft-body sedan, pressed into the wall | 60 (100 %) | 9.7–10.7 ms (1.9 · 8.9) |
| ETK K-series, cruising | 47–54 (78–90 %) | 16.4–19 ms (2.3–7 · 13.5–14) |
| ETK K-series, through the wall | 39–53 | 16.7–19.7 ms (2.4 · 15–17.4) |

- The sedan drives, strikes and pushes bricks in real time. The rigid side's cost with the
  car (1.4 ms asleep, 1.9 ms with the struck bricks awake) is the ring's own.
- The ETK's own 2 kHz frame is 13.5 ms a World step in Chrome (457 nodes and 3,800 beams on
  one workgroup), so it is not real time on its own. That waits on `perf/explicit-2khz`.

**What the cars do.**
- The ETK at 111 km/h goes through the 5.2 m base and comes out at 40 km/h, with 43–72
  beams broken. A tunnel of blocks bursts out of the far face, and the wall above stands.
- The authored sedan on raycast wheels tops out at about 41 km/h with the explicit solver's
  air drag and turbulence on, on the CPU reference as on the GPU. So it is the drag model,
  not the World. Without drag (`?drag=0`) it reaches 33 m/s and still climbs. At 41 km/h it
  stops in the wall (31–35 beams broken). With `?drag=0` it hits at 99 km/h, crumples (158
  beams broken), breaks out to z = 90 m, and stays real time: 54–60 World steps/s, at most
  9.7 ms GPU a step (vehicle 8.8). The drag's effect on that car is a follow-up; the default
  keeps drag on.
- Coupling room: at most 122 proxy shapes per car were seen (sampled every 15 steps, the
  ETK at 28 m/s into the wall), of the 1,024 it has. So the room was not raised.

**Correctness.**
- `engine-world/tests/explicit_drive_world.rs`: the sedan on raycast wheels, driven through
  the World's explicit vehicles, matches the CPU reference bit for bit over 170 World steps.
  The run crosses the origin rebase.
- The origin rebase of a coupled vehicle (100 m from the origin) hung the GPU on this
  branch. The fix was `feat/vehicle-sleep`'s `cplRebase` header fix, now merged. The
  repro, the ETK driven past 100 m, now runs to 180 m.

**Videos** (`~/Development/rubble-artifacts/brick-ring-drive/`, 1280 × 720, 60 fps,
frame-locked, recorded with `brick-ring-drive-record.mjs`):
- `brick-ring-drive-etk.mp4`: the ETK's 100 m run-up and the strike, from the chase, the
  wall's side and outside the ring, then the strike again ×4 slower from the wall's side and
  from outside;
- `brick-ring-drive-etk2.mp4`: the same with two ETKs.

Stills: `docs/media/brick-ring-drive/`.
