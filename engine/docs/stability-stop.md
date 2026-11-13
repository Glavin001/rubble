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
| Rebound | over the last second of readbacks: the kinetic energy is past its lowest there + 0.5 × what was lost from the peak before it + the sources since (the fall's work, what the tyres can have delivered, launches, reported inputs) + 50 kJ, and past 2 × that low, for 2 readbacks in a row | `REBOUND_SHARE`, `REBOUND_RISE`, `REBOUND_FLOOR`, `REBOUND_SUSTAIN`, `REBOUND_WINDOW` |
| Box | a vehicle's bounding box is past 10 × its spawn size + 50 m while its kinetic energy is above 50 kJ and 1.5 × its lowest since the last launch | `BOX_FACTOR`, `BOX_MARGIN`, `BOX_ENERGY_RISE` |

What the scene "can have supplied" is the vehicles' kinetic energy at the first readback, the launches the page reports (`launched(speed)`), the work of gravity over each vehicle's fall from its start to its lowest point, and a drive allowance: a traction force of 2 × the vehicle's weight over the distance its centre of mass moved (`TRACTION_MAX`; a car at rest gets none). Inputs the guard cannot see are reported by the page: a shot's energy (`supplied`), a hand's grab (`disturbed`, each frame it acts), a pulling rig (`setUnbounded`, which turns the rebound check off).

A 200 km/h pole crash, a spawn at 3,000 km/h, a launch at 1,000 km/h, a 500 m fall and a car rammed from rest do not trip it (tests in `stability.rs`). Nor do a springy rebound off a wall, a car bouncing after a 10 m drop, rolling 60 m downhill after a stop, driving off after a crash, a part thrown 80 m, or a launch from rest.

The rebound check was added after the guard's first real trip on 2026-11-13 (a 31 m drop at 189 km/h on the user's GPU). The kinetic energy fell to 681 kJ at the impact and then rose to 4,413 kJ in 0.4 s. The whole-run energy budget (about 22,000 kJ from the scene's height and speed) was far above that, so the box check stopped it, at 112 m across. The rebound check stops the same run within its first doublings.

## Limits of the detection

- Kinetic energy only. Energy stored in stretched beams is not read, so the check is late by however long an instability stays elastic. The factor of 3 covers preloaded beams.
- The whole-run energy budget grows with the distance driven, so it is weak in a long drive. The rebound check looks at the last second only and does not weaken (the 2026-11-13 yard run, 155 kJ to 7,569 kJ in 0.65 s after 35 s of driving, trips at 760 kJ; the first version's fixed 1 MW allowance had put the budget at 315,000 kJ, and only the box check caught it, at 161 m).
- With more than 8 vehicles only the non-finite check runs: an unread vehicle can hand a read one its energy in a collision, which would look like energy from nowhere.
- Rigid bodies are not read. A heavy body that falls on a vehicle adds energy the budget does not count; the factor of 3 and the floor cover the scenes shipped.
- Readbacks arrive a frame or two late, so a few more steps of the broken state are already submitted when the page stops. For that reason these pages also keep at most 2 steps in flight (`EXPLICIT_STEPS_IN_FLIGHT` for a World with explicit vehicles; `softbody.html` already did) and never catch up on steps not taken.

## Where it is wired

`stability-stop.mjs` (the panel; no logic) with `ExplicitSim.health()` in `softbody.html` and `WebWorld.explicitHealth()` in `one-world.html`, `explicit-world.html` and `brick-ring-drive.html`. `vehicle-lab.html` is on another branch and takes the same two lines when it merges.

## Kernel compiles left behind (a related cause of slow loads)

On macOS the Metal compiler service (`MTLCompilerService`) goes on compiling a pipeline after the page that asked for it is closed, reloaded or has lost its device. It was seen at 100 % CPU for 9.5 and 28 minutes after the page was gone, and such leftovers slow every later load. A cancelled or lost load cannot stop its compile. The pages therefore compile kernels one at a time, the smallest first (`warm_pipelines`), so an abandoned load leaves at most one compile behind instead of all of them.
