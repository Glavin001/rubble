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

## Results

(see below)
