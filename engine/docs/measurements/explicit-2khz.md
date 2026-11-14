# The explicit 2 kHz solver on the GPU: lossless cost work (perf/explicit-2khz)

M3 Max, shared with other agents (every run takes the machine's GPU lock). Tools:
`engine/tools/explicit-2khz/`:
- `bounded.sh`: the lock and a kill timer;
- `ab.sh`: interleaved A/B of `explicit_frame_dump time` with a bitwise check;
- `bench_ab.sh`: interleaved A/B of `explicit_vehicles_bench`;
- `crash-town-chrome.mjs`: the crash town's HUD in headless Chrome;
- `etk_probe.sh`, `verify.sh` (the parity suites), `wg_mem.py` (workgroup memory per
  module), `barrier_loops.py` (loops that reach a barrier).

Base: `fix/gpu-guards` 46b0a3ba (`fix/contact-ub` with the vehicle-sleep fixes and the
GPU guards), with this branch's measurement examples (`EV_ETK`, `EV_PAGE`,
`EV_NO_CONTACT`). The ETK K-series: `__mods/etkk_tex`, `etkk/kc6.pc`, JBeam mode with
pressure wheels (585 nodes, 4,304 beams, 508 triangles), in the crash town with the
coupling off.

## Results

**The explicit frame alone** (`explicit_frame_dump time` from one saved state, 33-step
frames, median of 9 a run, runs interleaved base/branch; every state word equal to the
base's over 10 frames, `ab.sh`):

| Vehicle | Base | Branch | Change |
|---|---|---|---|
| ETK K-series (6 rounds) | 14.25–14.29 ms | 8.53–8.55 ms | −40 % |
| Page car, 1 in the crash town, 100 proxy shapes (5 rounds) | 2.97 ms | 2.86 ms | −4 % |

**GPU ms per World step, native** (`explicit_vehicles_bench`, 1/60 s steps, 2 s):

| Case | Base | Branch | Share of the step saved |
|---|---|---|---|
| ETK K-series, n = 1 (`EV_PHASES`, 3 rounds) | 15.75–16.00 (frame 14.04–14.08) | 10.39–10.46 (frame 8.51–8.55) | 34 % |
| Page car, n = 1 (`EV_SPAN`: submit to completion, 3 rounds) | 5.30–5.50 | 5.28–5.36 | ~2 % |
| Page car, n = 4 (one contact group of four in the house) | 18.39–18.57 | 18.87–18.88 | −2 % (slower) |
| Page car, n = 16 | 8.65–8.76 | 8.63–8.66 | ~1 % |

The group of four, its frame alone from one saved state (`ab.sh`, 5 rounds, bitwise equal):
17.97 → 18.05 ms (+0.5 %). Turning the coupling's step frames off makes it slower
(18.73 ms), so the cost is elsewhere in the shared code the groups compile (the
tryContact and coupling edits change the contact modules' codegen); not found.

**Chrome** (headless, `explicit-world.html?n=…&timing=phases`, GPU ms per World step,
medians over 4 s after 6 s; base, branch, branch again):

| n | Base | Branch (run 1) | Branch (run 2) |
|---|---|---|---|
| 1 | 7.6–8.8 (frame 5.5–6.2), 60 steps/s | 6.4–6.8 (frame 4.7–5.1), 60 steps/s | 8.2–8.8 (frame 5.8–6.3), 60 steps/s |
| 4 | 25.0–25.2 (groups 23.4–23.7), 54 % real time | 25.4–25.6 (groups 23.8–24.1) | 25.4–25.6 (groups 23.8–24.1) |
| 16 | 26.9–28.3 (groups 20.4–26.0), 49 % | 27.5–28.6 (groups 20.9–26.5) | 27.5–28.7 (groups 20.3–26.5) |

The crash town's page car has few self-collision triangles and spends its frame in the
coupling (n = 1) and the contact groups (n = 4, 16): in Chrome the branch is within the
runs' spread at every n. The gains are the self-collision's (the ETK, and any car with
many collision triangles).

## The changes (each lossless: every state word equal to the base's)

1. **The triangles' step geometry once a step** (`triGeoFill`, `triGeoLoad`). The grid walk
   spread a triangle over 8 lanes, each computing its geometry, and with anti-clip each
   computed the triangle's frame a step back for every contacter in its box. The beam
   phase (positions final) now writes each tested triangle's geometry and previous frame
   once (gBeams, 9 records a triangle); the walk loads it, its liveness applied at the
   load as before.
2. **Walk micro-changes**: tryContact issues its tests' loads together; the duplicate
   bucket check steps its cell offsets instead of dividing.
3. **The walk lists, the lanes test** (`triPairs`): the walk lists the (triangle,
   contacter) pairs it would test (gNext, 8,192 a vehicle; past that it walks again with
   its tests), and after a barrier the lanes test the list a pair each, so no lane waits
   on the triangle with the most contacters. The hits are applied in the specification's
   order whatever order they are found in (selfApply).
4. **The list walk's box alone** (3 of the 9 records), 2 lanes a triangle (1 and 8 were
   slower), and no second load of a bucket entry's position.
5. **Coupling**: the bodies' sums pick their shapes from workgroup memory (the shape's
   body slot beside its terms); each proxy shape's step frame (centre, rotation, extents)
   computed once a step beside its sphere and read by `cplFace` (with the cull on; off,
   computed as before, so `the_coupling_shape_cull_is_bit_exact` compares the two).

ETK, in order (explicit frame): 14.27 → 9.16 ms (1–3) → 8.85 (4, the box and 2 lanes) →
8.54 (4, no second load).

## Not done

- **Fewer barriers (step 3).** Not attempted: in the solo step each barrier orders a real
  dependency (node phase → beams and contact → node phase; the coupling's integration →
  spheres → count → node phase). `sync()` is two barriers (storage, then workgroup); WGSL
  has no combined form.
- **A contact group across workgroups (step 4 for groups).** The crash town's n = 4 and 16
  cost is the contact groups (a group of four: 18 ms natively, 24 ms in Chrome, one
  workgroup stepping four cars at 128 lanes each with their coupling). Splitting a group's
  members over workgroups needs a dispatch per step between the members' steps and the
  vehicle contact pass (about 66 dispatches a frame); not built.
- **The coupling's count pass** (`cplCount`, about half of the page car's coupling: every
  node against every proxy shape's sphere, then the face test) is unchanged.

## The runaway guards (built on this branch, ported to `fix/gpu-guards`)

During this work the GPU wedged twice (a coupled vehicle crossing 100 m on a branch
without the `cplRebase` fix). Every explicit kernel loop now draws from a
per-invocation pass budget (`explicit_loop_cap.slang`, `EB_LOOP_BUDGET`): the host's bound
on one invocation's legitimate passes in a frame (`explicit_plan::loop_budget`: twice a
per-step bound for the whole fleet times the steps, 2^22..2^31). Loops that reach a
barrier have caps from their real maxima instead (steps, members, levels, chunks),
computed from uniforms only (Tint's uniformity analysis). The coupling passes
(`vehicle_coupling.slang`) have the same caps and budget (`ST_LOOP`). The World reads the
frame's loop-cap word and the coupling's status with every step's readback and fails that
step. Tests: `the_pass_budget_ends_a_frame_and_reports_it`,
`a_world_step_past_the_pass_budget_fails_fast`. A legitimate frame of the test fleets
stays under 2^16 passes a lane (`explicit_beams_gpu` passes with `RUBBLE_LOOP_BUDGET=65536`).

## Verification (branch head, each suite under a kill timer)

engine-webgpu `explicit_loop_cap` 5, `explicit_beams_bitwise` 12, `explicit_contact_gpu`
12, `explicit_beams_gpu` 5, `explicit_rigid_gpu` 6, `explicit_sleep_gpu` 3,
`explicit_centred_damping_gpu` 4, `explicit_jbeam_ext_gpu` 8; engine-world
`explicit_vehicles` 8, `explicit_loop_budget` 1; engine-metal `explicit_parity` 7,
`vehicle_parity`, `explicit_drive_parity`: all pass. Chrome: the crash town compiles and
runs at n = 1, 4 and 16 with no page error.

## Where the frame's time goes (before)

The explicit frame alone (`explicit_frame_probe`, 33 steps, after 60 World steps at
60 Hz, median of 9):

| Vehicle | Frame | Ablation |
|---|---|---|
| ETK K-series (585 nodes, 4,304 beams, 508 triangles; coupling off) | 13.5–14.1 ms | self-collision tests off (`EB_ABLATE` 256): 3.0 ms; walk stopped after the triangle geometry: 3.5 ms; no hit test past the box: 8.7 ms; no duplicate-bucket check: 12.3 ms; no anti-clip crossing test: 12.1 ms |
| Page car (crash town, 100 proxy shapes after the strike) | 2.94 ms | coupling off: 0.72 ms |

So the ETK's frame is two thirds self-collision (the grid walk and its hit tests), and
the page car's three quarters coupling. Neither is the beams' force accumulation.

## Workgroup memory (the brief's step 1)

`wg_mem.py` on the built WGSL: the frame modules declare 29.9–31.0 KB of workgroup
memory, the contact modules up to 31.5 KB, against Metal's 32 KB. A 470–585-node car's
positions and velocities are 15–19 KB. They do not fit beside the grid counts and
offsets (8 KB), the tested-triangle list (8 KB), the pair list (4 KB) and the coupling's
copies (8.3 KB), all live within a step.

## Gather instead of scatter (the brief's step 2): already the design

The beams' forces are not accumulated by atomics. The beam phase writes each beam's
force on each end into that end's incidence entry (`gBeams[incBase + slot]`, the CSR of
beam ends built at upload), and the node phase sums a node's entries in beam order
(`beamSum`): a fixed order, the CPU's, no atomics. The fixed-point atomics (`gAcc`) carry
only contact forces (self-collision hits, vehicle contact, the coupling's reactions),
whose integer sums are order-free by construction. Nothing to change.

## Packing small vehicles (step 5): no idle lanes

Fleets of up to 32 vehicles run 512 lanes a vehicle, larger ones 256. The cars measured
have 470–585 nodes and 1,500–4,300 beams: every phase spreads over all lanes (the beam
phase 3–9 passes a lane), so no lanes are idle to pack another vehicle into.

## Spreading a vehicle over workgroups (step 4): exists as the wide passes

`explicit_beams_wide_*` already run one vehicle across the GPU, a dispatch per phase per
step (6 a step with centred damping, about 200 a frame). They do not run the coupling,
aero, raycast wheels or surface contact, so the town's cars (coupled) cannot use them.
