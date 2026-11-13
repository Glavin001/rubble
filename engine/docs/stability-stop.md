# The pages' stop on an unstable simulation

**This is detection and a stop. It is not a fix.** When it trips, the simulation has a bug (or content the solver cannot integrate at its step), and that bug still has to be found and fixed in the solver or the content. The stop only keeps a broken state from taking the GPU away from the machine while that work is done.

## Why

An explicit vehicle that has gone numerically unstable makes every later step slower on the GPU: its contact and self-collision lists grow with its bounding box. A page that keeps submitting steps then queues more GPU work than the GPU finishes, the browser loses the device, and on macOS the desktop can go with it (`gpu-overload.md` on the branch `fix/gpu-overload-guards`, failure mode (c)). Submitted GPU work cannot be taken back, so the page has to stop submitting as soon as the state is seen to be impossible.

Seen on 2026-11-13: `softbody.html`, side pole at 186 km/h with surface contact, generated sedan. The car warped, then moved wildly; the HUD read 4,861 km/h with 373 broken and 2,017 bent of 4,228 beams; then the device was lost.

## What it does and does not do

- It **reads** the vehicles' node records back once per displayed frame (the first 8 vehicles; a copy and a map, never waited for) and judges them in Rust (`engine-world-wasm/src/stability.rs`, `StabilityGuard`).
- It **never** clamps, damps, caps, rescales or writes any simulation state, parameter or input. There is no speed limit and no energy limiter in the solver. Sliders keep their ranges. A run with the guard is the same run bit for bit as one without it (test `the_guard_does_not_change_a_run`).
- When it trips, the page stops submitting steps for good, keeps its last frame on screen and shows an **engine error** panel with the evidence, the scene link and a copyable report. The same report goes to the console. It never resumes by itself; the button reloads the scene.
- A lost device ends the same way: a message, no rebuild, no further calls into the device (wgpu panics on a lost device's errors, so `ExplicitSim.frame`, `WebWorld.step` and the kernel warm-up return an error instead of reaching it).

## What trips it

Only states correct physics cannot reach. The thresholds are named constants at the top of `stability.rs`.

| Check | Trips when | Constants |
|---|---|---|
| Non-finite | a node's position, velocity or mass is NaN or infinite | none |
| Energy | the read vehicles' kinetic energy is past 3 × what the scene can have supplied + 50 kJ, for 3 readbacks in a row | `ENERGY_FACTOR`, `ENERGY_FLOOR`, `ENERGY_SUSTAIN` |
| Box | a vehicle's bounding box is past 10 × its spawn size + 50 m while its kinetic energy is above 50 kJ and 1.5 × its lowest since the last launch | `BOX_FACTOR`, `BOX_MARGIN`, `BOX_ENERGY_RISE` |

What the scene "can have supplied" is the vehicles' kinetic energy at the first readback, the launches the page reports (`launched(speed)`), the work of gravity over each vehicle's fall from its start to its lowest point, and a drive allowance of 1 MW per vehicle for the simulated time (`DRIVE_POWER_MAX`).

A 200 km/h pole crash, a spawn at 3,000 km/h, a launch at 1,000 km/h, a 500 m fall and a car rammed from rest do not trip it (tests in `stability.rs`).

## Limits of the detection

- Kinetic energy only. Energy stored in stretched beams is not read, so the check is late by however long an instability stays elastic. The factor of 3 covers preloaded beams.
- The drive allowance grows with time, so the energy check gets weaker in a long run (after 60 s the budget of one car is 180 MJ above its start). Non-finite values and the box check do not weaken.
- With more than 8 vehicles only the non-finite check runs: an unread vehicle can hand a read one its energy in a collision, which would look like energy from nowhere.
- Rigid bodies are not read. A heavy body that falls on a vehicle adds energy the budget does not count; the factor of 3 and the floor cover the scenes shipped.
- Readbacks arrive a frame or two late, so a few more steps of the broken state are already submitted when the page stops. For that reason these pages also keep at most 2 steps in flight (`EXPLICIT_STEPS_IN_FLIGHT` for a World with explicit vehicles; `softbody.html` already did) and never catch up on steps not taken.

## Where it is wired

`stability-stop.mjs` (the panel; no logic) with `ExplicitSim.health()` in `softbody.html` and `WebWorld.explicitHealth()` in `one-world.html`, `explicit-world.html` and `brick-ring-drive.html`. `vehicle-lab.html` is on another branch and takes the same two lines when it merges.

## Kernel compiles left behind (a related cause of slow loads)

On macOS the Metal compiler service (`MTLCompilerService`) goes on compiling a pipeline after the page that asked for it is closed, reloaded or has lost its device. It was seen at 100 % CPU for 9.5 and 28 minutes after the page was gone, and such leftovers slow every later load. A cancelled or lost load cannot stop its compile. The pages therefore compile kernels one at a time, the smallest first (`warm_pipelines`), so an abandoned load leaves at most one compile behind instead of all of them.
