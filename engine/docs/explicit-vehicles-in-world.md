# Explicit soft-body vehicles in the Rubble World (E2)

Status: design (2026-11-05), branch `feat/vehicles-in-world`. Item E2 of
`vehicle-acceptance-criteria.md`, with E1 (scale) and E3 (parity) as constraints.

Goal: the explicit node–beam vehicles (`engine_softbody::explicit_beams`, 2 kHz, the GPU
kernel `slang/kernels/explicit_beams.slang`) collide **two-way** with the World's rigid
bodies and with prefractured, stress-solved destructibles, in one World, on the GPU. Today
they meet only their own static boxes and shapes, and each other.

## Constraints this design keeps

- The explicit solver keeps its fixed 0.5 ms step, its integrator and its node-only contact
  rule (`contact_force`: cancel the normal force, remove 80 % of an approaching normal
  velocity, static or Stribeck sliding friction, the "concrete" box surface, no
  penetration term for box-like shapes). Nothing about tolerances or the rule changes; a
  rigid partner is a *moving, finite-mass* surface for the same rule (below), and a
  partner of infinite mass at rest is bit-identical to a static shape.
- The World keeps its own rate (`AvbdParams::dt`, 1/60 or 1/120 s) and its AVBD solve.
- Destruction is forward-only: no rewind, no correction pass, no save-and-restore.
- One library: the rules are Rust (CPU reference) and Slang (GPU), single source; the JS
  page only asks for a scene.

## 1. How vehicle nodes meet rigid bodies and chunks

### Proxies

For every World step each vehicle receives a short list of **proxies**: the rigid bodies
whose colliders overlap the vehicle's swept bounds for the window (node bounds at the
window's start, grown by the fastest node's travel over the window, the collision range
and the fastest body's travel). A proxy is the body's state at the window's start —
centre of mass `c`, orientation `q`, linear and angular velocity `v`, `ω`, inverse mass
`1/M` and inverse principal inertia (0 for static, anchored or kinematic bodies) — plus
its colliders that overlap the vehicle (a **proxy shape** each: kind, pose in the body's
frame, parameters, and the GPU chunk it belongs to, if any).

Limits are explicit and loud: at most `PROXY_BODIES` bodies and `PROXY_SHAPES` shapes per
vehicle per window (engine-wide configuration); more fails the step with an error naming
the limit. Nothing is dropped.

### Shapes

The World's collider kinds map onto the explicit solver's box rule ("a node strictly inside
is pushed out along the normal of the nearest surface"):

| World collider | Explicit rule |
|---|---|
| box | the oriented-box rule (`StaticShape::OrientedBox`, unchanged) |
| sphere | the sphere rule |
| capsule (local Y) | the capsule rule |
| convex hull (chunks) | new, the box rule over the hull's face planes: inside every plane, pushed along the plane nearest the node. A box is the hull with six planes. |
| plane, heightfield | not proxies (the vehicles keep their own ground); listed by filter |

Contact is **node-only**, as for static shapes: nodes against shapes. Rigid edges against
vehicle collision triangles are not tested (as for static boxes today). This is the
explicit solver's own semantics, kept.

A collider takes part only if its collision filter accepts the vehicles' filter (the
World's ground is normally left out: the explicit ground is the vehicle's).

### The two-body form of the contact rule

For node `i` (mass `m`, force `f`, velocity `u`) inside proxy shape `s` of body `b` at
world point `x` with outward normal `n`, at step `t`:

```
r      = x − c(t)                      lever arm from the body's centre of mass
v_b    = v(t) + ω(t) × r               the body's material velocity at x
w_b    = k · (1/M + (r × n) · I⁻¹(t) (r × n))   body's inverse mass at x along n,
                                        split k ways (k: nodes of this vehicle touching
                                        b this step) and V ways (V: vehicles whose
                                        proxies hold b this window, see below)
w_n    = 1/m
m_red  = 1 / (w_n + w_b)               reduced mass
s      = m_red / m  (= w_n/(w_n + w_b)) share of the node's force the contact cancels
f_c    = contact_force(s·f, u − v_b, m_red, n, h, BOX, 0, friction)
node:  f  += f_c
body:  F_b −= f_c,  τ_b −= r × f_c
```

With `1/M = 0` and `I⁻¹ = 0` (static or anchored), `s = 1`, `m_red = m` and `v_b` is the
body's (zero) velocity: exactly `contact_force(f, u, m, n, h, BOX, 0, friction)`, the
static shape rule, bit for bit. With finite mass the same rule acts in the two-body frame:
the normal force it cancels is the share that makes the *relative* normal acceleration
zero, and the approaching *relative* velocity it removes is removed with the reduced mass.
(The vehicle–vehicle triangle contact already uses a reduced mass; the static shapes use
the node mass. This is the same rule for a partner of finite mass.)

**Mass splitting.** A rigid body meets many nodes in one step; computing each node's
contact against the whole body (Jacobi) over-pushes the body `k` times. Each contacting
node therefore sees `1/k` of the body (Tonge, Benevolenski and Voroshilov, *Mass splitting
for jitter-free parallel rigid body simulation*, SIGGRAPH 2012): the body is split into `k`
equal sub-bodies, one per contact, and recombined by averaging, which is the body
receiving the sum of the impulses at its full mass. The count `k` is this step's (a count
pass before the node pass, on the same positions the node pass tests), never the last
step's. Bodies held by `V` vehicles' proxies in one window are split `V` ways across the
vehicles in the same way.

Friction uses the same scalar effective mass along the normal (the static rule's
convention). The angular inertia's tangential coupling is not modelled in the scalar rule;
it is exact for the normal response and for bodies of infinite mass.

### The proxy inside the frame

Each step, after the vehicle's node pass, the proxy integrates its accumulated contact
reaction the way the nodes integrate theirs (symplectic Euler, `h = 0.5 ms`):

```
v ← v + h F_b / M          ω ← ω + h I⁻¹(q) τ_b
c ← c + h v                q ← normalize(q + ½ h (ω, 0) q)
```

No gravity, no other contact and no joint act on the proxy inside the frame: those are the
World's, applied in its own step. The proxy carries the body's start-of-window motion plus
the vehicle's push, so a body the vehicle shoves moves away inside the frame (substep
coupling), rather than standing as a wall for the window.

Accumulated per proxy shape over the window: the impulse `J_s = −Σ h f_c`, its moment
about the shape's start-of-window reference point `p_s`, `M_s = −Σ h (x − p_s) × f_c`,
and the largest single-step force on it (diagnostics).

## 2. Time coupling

Explicit steps per World step: `k = ` the window's 2 kHz steps with the remainder carried
(the `FrameClock` rule generalised: 1/60 s → 33, 33, 34; 1/120 s → 17, 17, 16), so explicit
time never drifts from World time by more than one 0.5 ms step.

Order of one `World::step` (window N, `[t_N, t_N + Δ]`), all in one submission:

1. host commands (impulse records, …) are uploaded;
2. **emit** (GPU): window N's per-shape impulses become impulse records, each on the body
   its shape belongs to *now* (after the last fracture's split), `J_s` through the body's
   centre and angular impulse `M_s + (p_s − c_body) × J_s`, appended after the host's
   records (as the terrain coupling appends its reactions);
3. the rigid step N applies them at its start, then solves as always;
4. **explicit frame N+1** (GPU): proxies planned from the bodies after step N, the vehicles
   stepped over window N+1, impulses and chunk loads accumulated;
5. **fracture N** (GPU): the rigid contacts of step N (ingest slot 0) and the vehicles'
   chunk loads of window N+1 (ingest slot 1) load the stress solver; bonds break and bodies
   split;
6. readback.

The first step primes window 0 (step 4 runs once before step 2).

Impulse exchange is therefore **per World step**, with the vehicle side resolved at 2 kHz
inside it. The rigid body feels the vehicle's whole impulse for a window at the start of
the rigid step covering that window, the same staggered (explicit) coupling the terrain
uses. A body resting *on* a vehicle is held by these impulses: it sinks by at most
`g Δ²` between frames (0.7 mm at 120 Hz) and is pushed back by the next frame.

## 3. Momentum and energy

**Momentum.** Every contact force `f_c` on a node is matched by `−f_c` on its proxy in the
same step, and the proxy's accumulated impulse is delivered to the body (step 2). Linear
momentum closes to rounding; angular momentum about any point too (the moment is carried
with its reference point). On the GPU the per-step body sums are integer fixed point
(1/32 N, as the vehicle's own self-collision forces): integer atomics are order-free, so
runs are deterministic; the rounding is below 1/64 N per contact per step.

Contacts with static or anchored bodies send their impulse into the ground, as a wall's
contact does today; such tests measure the vehicle alone.

**Energy.** The rule removes relative normal velocity (80 % of the approach) with the
reduced mass, cancels relative normal acceleration, and friction opposes sliding: in
the two-body system none of these adds energy, and mass splitting keeps the many-node
contact from over-pushing the body. Two sources of energy error remain and are measured
in the tests, not hidden:

- the World's own solve of the body after the impulse (AVBD, its restitution and
  contacts), as for any impulse;
- the one-window latency of fracture on the vehicle's side (below).

## 4. Strikes on prefractured buildings: forward-only

Destructibles are the World's: `StrengthModel::Bonds` (every chunk a body, AVBD bond joints)
or `StrengthModel::Threshold` (D1b: bonded clusters are single bodies, the GPU fracture
breaks bonds whose capacity a chunk load beats and splits bodies in place).

- **Bonds.** Chunks are bodies, so they are proxies like any body; the vehicle's impulses
  reach the chunk bodies at the rigid step's start and the AVBD bonds carry, and break
  under, the load in that step. Nothing else is needed.
- **Threshold.** The stress solver loads chunks from contact manifolds (stress-gpu
  `EngineIngest`, coupling.slang). The vehicle's chunk loads enter the *same* ingest as a
  second substep slot: one record per (proxy shape, window) carrying the chunk, the
  impulse `J_s` (stress frame, Z up), its point, and a closing speed of 0. The threshold
  rule then takes them as a sustained load, `J_s / Δ`: the window's mean contact force,
  which the 2 kHz frame resolved exactly (no Hertz impact estimate is needed for a contact
  the explicit solver integrated). Records sum with the rigid contacts' in the chunk's
  manifolds; the break rule, the split, the support loss and every event are unchanged.

The vehicle frame for window N+1 runs before the rigid step of that window (§2, step 4),
so **its strike is predicted at the end of the frame before it and split before the
engine steps**: the routing design of AGENTS.md, with the explicit frame itself as the
predictor (it is the exact soft-body response over the window, not an estimate). Nothing
is re-simulated: the frame is computed once, its loads applied once.

The impulses of window N+1 are emitted after the split, consistently with what the vehicle
met in the window (`vehicle_coupling.slang` `emitPair`):
- a proxy body of infinite mass in the window (anchored, static): nothing; the vehicle met an
  immovable body and its reaction went into the ground, as for a wall;
- a proxy body still whole: its impulse and angular impulse;
- a proxy body that split (or whose shapes now belong to other bodies): each piece takes the
  change of motion the whole body's impulse gives at the piece's centre of mass,
  `m (Δv + Δω × r)` and `I Δω`, as split pieces move with their parent's motion. Momentum
  closes and no piece moves faster than the whole body would have.

The first version gave each freed chunk its own share of the impulse the vehicle exchanged
with the *unbroken* structure: a 1 kg chunk took 1,888 m/s in the house test. That is the
energy error of the one-window latency, and the rule above removes it.

Latency, stated: inside window N+1 the vehicle meets the structure as it stood at the
window's start (unbroken, or anchored). A piece that breaks free in that window starts at
its parent's motion and meets the vehicle as a free body from the next window on. Over one
window (≤ 16.7 ms at 60 Hz) a car's crush force is still rising. The exact alternative,
releasing a chunk proxy inside the frame at the substep its load passes the capacity, is a
follow-up: it puts the threshold rule into the explicit kernel.

## 5. GPU data flow (every backend)

Single source: the coupling kernels are Slang (`slang/kernels/explicit_coupling.slang`),
the proxy contact is Slang inside the explicit kernel, the host orchestration is Rust
(`engine-world`), the CPU reference is Rust (`engine_softbody::explicit_rigid`).

Per World step, no host readback and no host data upload beyond the existing commands:

```
 World buffers (bodies, colliders, hull arena)      explicit buffers (nodes, misc)
            │                                                  │
            └──► couple_bounds   (one group per vehicle: node bounds, fastest node)
                 couple_plan     (per vehicle × collider: AABB overlap, filter, limits;
                                  per body its vehicle count V)
                 → proxy table   (bodies, shapes, chunk ids)       [coupling buffer]
                       │
                 explicit_beams_frame (k steps; per step: count pass, node pass with the
                                  two-body rule, lane-parallel proxy integration)
                 → per-shape J, M, peak                            [coupling buffer]
                       │
                 couple_loads    → ingest records, slot 1 (threshold destructibles)
                 ...fracture N (break, split, reparent colliders)...
                 couple_emit     (next step) → impulse records after the host's
                                  (gScratchAtomics IMPULSE_*; overflow flagged)
                 rigid step      apply_impulses
```

The proxy table and accumulators live in one coupling buffer the explicit kernel binds;
its per-step sums are workgroup integer atomics. A vehicle with no proxy this window takes
the kernel's existing path (bit-identical; the proxy path is a separate function behind a
uniform early return, as the static shapes are, so Metal's fast-math codegen of the shipped
paths is untouched).

Backends: the Slang compiles to WGSL (wgpu: browser WebGPU and native, Metal through wgpu
on macOS), MSL and CUDA from the same source. The explicit solver itself runs on wgpu today
(`ExplicitBeamsGpu`); the coupling runs wherever it runs, with no backend-specific code.

## 6. Cost at scale

- Plan: vehicles × colliders AABB tests per window (brute force in v1; a uniform grid over
  colliders is the planned replacement when collider counts make it matter).
- Frame: per step, for a vehicle with proxies, two extra barriers, a pass writing each proxy
  shape's bounding sphere (grown past rounding) once, and a count pass over its nodes against
  its proxy shapes; a node skips a shape outside its grown sphere (the shape cull, lossless:
  cplFace would reject it) and tests the rest in shape order. Vehicles with none pay nothing.
- Emit and loads: one thread per proxy shape.

Measured numbers (1, 16, 64 vehicles) are in §8 when the implementation lands.

## 7. Tests

- CPU: a static proxy is bit-identical to the same static shape; momentum closes across
  the coupling (a free vehicle into a free box, no ground: horizontal momentum); energy
  does not rise; a vehicle into a box of known mass gives a box speed between the
  perfectly plastic and the elastic results and the exact momentum balance.
- GPU: CPU↔GPU parity of the coupled frame; the World test of a vehicle into a dynamic box
  (momentum through the World's impulse path); a vehicle into a prefractured threshold
  wall breaks bonds, forward-only, and the freed pieces carry the vehicle's impulse.

## 8. Results (2026-11-05, M3 Max, other agents on the GPU)

Code: `engine_softbody::explicit_rigid` (CPU reference), `slang/kernels/explicit_beams_rigid.slang`
(included by the explicit kernel), `slang/kernels/vehicle_coupling.slang` and
`engine-world/src/explicit_vehicles.rs` (World), `engine-world/src/explicit_scene.rs` (the crash
town), `engine-webgpu/explicit-world.html` (the page). The coupling's records also run on Metal
and CUDA through `engine_exec::explicit_plan` (`engine-metal/tests/explicit_parity.rs`).

Tests:

| Test | Result |
|---|---|
| CPU: a static proxy is the static shape (`engine-softbody/tests/explicit_rigid.rs`) | bit for bit through a 15 m/s crash |
| CPU: momentum across the coupling (free flight into a free box) | horizontal ΔP 1e-6 of P |
| CPU: energy | rises no more than the vehicle's own free-flight oscillation |
| CPU: box of known mass (300, 1200, 4000 kg) | between the plastic and elastic speeds, exact momentum balance, restitution 0.07 |
| GPU vs CPU (`engine-webgpu/tests/explicit_rigid_gpu.rs`) | first contact frames 2.5e-7 m, outcome 1e-5, momentum 7e-8; coupling built but off: bit for bit |
| World: vehicle into a free 1200 kg box (`engine-world/tests/explicit_vehicles.rs`) | ΔP 1.4e-6 of P through the World's impulse path; box 13.1 m/s in [11.37, 22.75]; kinetic energy lost |
| World: vehicle at 20 m/s into the prefractured cuboid house (826 chunks, threshold) | bonds break in the step of first contact, the house splits, freed pieces move, none faster than 2 × 20 m/s (fastest 0.72 m/s) |

Frame times (`engine-world/examples/explicit_vehicles_bench.rs`): the crash town, repo sedans at
15–18 m/s, one house per four vehicles, World steps of 1/60 s (33–34 explicit steps each), GPU ms
per World step from the step's own timestamps (median / p95):

| Vehicles | Houses | With vehicles | Houses alone | Explicit frame alone |
|---|---|---|---|---|
| 1 | 1 | 4.20 / 5.30 | 4.31 / 5.20 | 0.46 |
| 16 | 4 | 7.61 / 15.5 | 2.61 / 3.07 | 0.53 |
| 64 | 16 | 13.44 / 18.0 | 2.53 / 4.25 | 1.14 |

(After the emit rule, the 1024-shape room and the per-window contact replan; measured with
the load average back up to 20 from other agents, so the 1-vehicle pair is noise-limited.
An earlier run on a quieter machine, before those changes: 3.08, 6.35 and 11.75 ms.)

At 64 with the coupling filtered out (vehicles meet only each other, four at each house's centre)
the step is 7.6 ms: the fleet with its vehicle–vehicle contact groups is about 4.8 ms, and the
strikes (proxies, and the fracture's breaks and splits of 1700 bonds in 2 s) about 4 ms. The
coupling's own barriers cost 3–5 % of the explicit frame. The wide passes are not used with the
coupling on; for the repo sedan the per-vehicle frame is faster anyway (0.44 against 0.63 ms
alone), and with vehicle contact on (any fleet) the wide passes are off already.

### 8.1 Scale baseline (E1), 2026-11-05

`explicit_vehicles_bench` (now: `EV_HZ` the World step rate, `EV_RAYCAST=1` the
procedural sedan on raycast wheels, `EV_ALONE=0` skips the frame-alone runs, wall
time per step beside the GPU time; at most 32 houses, the fracture's chunk capacity,
the rest of a fleet meeting head on in open lanes, `crash_town_at`). M3 Max, load 5–8
from other agents, one run at a time. GPU ms per World step, median / p95 / p99:

| Cars | Beam tyres, 60 Hz | Beam tyres, 120 Hz | Raycast, 60 Hz | Raycast, 120 Hz |
|---|---|---|---|---|
| 16 | 7.37 / 15.71 / 15.86 | 6.98 / 8.11 / 8.31 | 18.74 / 20.48 / 20.94 | 11.33 / 14.76 / 16.95 |
| 64 | 13.84 / 18.75 / 19.31 | 8.51 / 12.70 / 13.67 | 25.15 / 40.44 / 44.93 | 14.46 / 23.70 / 25.69 |
| 128 | 21.18 / 28.13 / 30.01 | 12.52 / 17.53 / 18.43 | 33.12 / 47.07 / 48.43 | 18.66 / 27.57 / 31.67 |
| 256 | 28.26 / 39.90 / 46.29 | 16.20 / 22.97 / 26.15 | 44.54 / 64.64 / 72.81 | 25.21 / 38.53 / 40.50 |

Wall time (step + the bench's blocking poll) adds 2–5 ms at the median. Where the
GPU time goes (`RUBBLE_GPU_TIMING_PRINT=1`, mean per step over 240 steps at 120 Hz):
the fracture phase (the houses' threshold loads, breaks and splits) 4.5 ms of 7.0 at
64 beam-tyre cars and 7.4 of 10.3 at 128 (raycast: 4.9 of 7.3, 8.5 of 11.4); the rigid
solve about 1.5 ms; the explicit vehicles' marked span 0.14–0.27 ms. The fracture's
cost follows the bonds broken (1,776 at 64 cars and 120 Hz, 2,806 at 128; raycast
3,265 and 4,761). Against E1 (≥ 100 cars at p99 ≤ 8.33 ms per 120 Hz step; ≥ 256 at
60 Hz): not met. (This phase split came from timestamp marks that Metal can take early;
§8.2 measures it again: the explicit frame is the largest phase.)

### 8.2 Where the time goes, and the lossless changes (E1), 2026-11-05

**The attribution above was wrong.** A mark is an empty compute pass that writes a
timestamp as it begins; on Metal a pass with no dependencies can start before the
passes encoded ahead of it finish, so the marks put the explicit frame's time into
the fracture's segment (and the marked total can run short). Three measures replace
it (branch `feat/vehicle-scale`):

- `EV_SPAN=1` (`explicit_vehicles_bench`): each step's span from its submit to its
  completion. The bench waits for every step, so the GPU is idle at each submit; the
  span is the GPU's time plus the submit latency and the readback decode (an upper
  bound). It is the column the tables below report.
- `RUBBLE_GPU_TIMING_SYNC=1`: every mark ends the command buffer, submits it and
  waits; a segment's time is the host's wall time of its wait. Immune to the overlap,
  slower in total (the GPU idles and clocks down between segments), so it is for the
  split, not the total. `RUBBLE_GPU_TIMING_EXPLICIT=1` adds a mark after each of the
  frame's dispatches.
- `explicit_frame_probe`: the vehicles' frame alone on one crash-town state (a
  snapshot), timed by its own pass timestamps, under switches (shape cull, vehicle
  contact, coupling, the raycast wheels' pass).

64 beam-tyre cars, 120 Hz, submit-and-wait medians: the frame's contact-group
dispatch 6.1 ms, rigid broadphase 1.6, primal 1.7, dual 0.7, the threshold fracture
1.7, the coupling plan 0.6 (the in-mode total is 15.6, inflated by the waits). The
explicit frame, not the fracture, is the largest phase. The probe on the state after
the strikes (17 steps, 64 cars): 0.66 ms with coupling and vehicle contact off, 2.7
with vehicle contact on (head-on pairs are contact groups of two from before they
meet), 3.7 with the coupling on and contact off (cars inside broken houses meet 30 to
117 chunk shapes each), 5.8 with both. A frame's time is its slowest workgroup's:
64 to 256 cars fill far less than the GPU, so it is the latency of the busiest vehicle
or group, not the fleet's total work.

Lossless changes (each bit for bit, or results unchanged by construction):

| Change | Check | Effect |
|---|---|---|
| Coupling shape cull: each step every proxy shape's bounding sphere, grown past rounding, written once (`cplSpheres`); a node skips the shapes outside it | `the_coupling_shape_cull_is_bit_exact` (engine-world): frames with the cull on and off from the same snapshot, every state word equal, cars among 20 to 79 chunk shapes | frame 6.6 → 5.8 ms (64 cars, after the strikes) |
| Raycast wheels a wheel per lane (rays, suspension, tyre and spin; the anti-roll bars on one lane between; one lane when two wheels share a hub node) | `explicit_drive_gpu` (bit for bit against the CPU), Metal `explicit_drive_parity` | the wheels' pass about 0.5 ms of a 7.9 ms raycast frame (64 cars) |
| Crash town capacity sized per house (`capacity_for`'s rule per house, not its one-scene floor of 16,384 bodies times the houses) | same bonds broken (1,776 and 2,806 at 64 and 128 cars) | 32 houses alone 3.79 → 2.91 ms; 64 cars 10.19 → 9.39 ms; 128 cars 14.43 → 13.09 (span medians, 120 Hz) |

**The 2^20 limit.** It is genuine: the fracture's split ranks chunks and body slots
with a two-level scan (one group of 1024 blocks of 1024). It was reached by an
artefact. The crash town multiplied a one-scene capacity floor (16,384 body slots for
an 826-chunk house) by the houses, so 32 houses had 2^20 body slots, and every
per-slot pass of each step ran over a million slots. 64 houses went past the bound.
Sized per house, 32 houses take 2^17 slots and 64 fit (256 cars with 64 houses step:
20.8 / p99 34.9 ms at 120 Hz). The bound is now checked when the fracture is created,
named by its field (`WorldCapacity::max_bodies` past 2^20 fails with that message).
The bench keeps 32 houses by default (`EV_MAX_HOUSES`) to compare with the baseline.

**Before and after** (`explicit_vehicles_bench`, `EV_SPAN=1`: GPU ms per World step, submit
to completion, 2 s of steps, at most 32 houses; before is eb0b389a with only the timing
added, after is this branch; M3 Max, load 6 to 12 from other agents, one run at a time,
23:00 to 23:45). The strikes are the same in both (the same bonds broken in every row):

| Cars | Wheels | Hz | Before p50 / p95 / p99 | After p50 / p95 / p99 | After wall p50 |
|---|---|---|---|---|---|
| 16 | beam tyres | 60 | 14.44 / 16.64 / 16.80 | 13.07 / 14.62 / 14.77 | 13.76 |
| 64 | beam tyres | 60 | 15.97 / 19.39 / 20.34 | 14.57 / 18.93 / 19.24 | 15.44 |
| 128 | beam tyres | 60 | 24.25 / 31.69 / 32.29 | 21.31 / 26.32 / 28.84 | 22.23 |
| 256 | beam tyres | 60 | 30.95 / 43.22 / 45.86 | 29.88 / 39.51 / 48.33 | 30.78 |
| 16 | beam tyres | 120 | 8.09 / 9.08 / 9.47 | 7.36 / 8.21 / 8.49 | 7.93 |
| 64 | beam tyres | 120 | 10.06 / 14.06 / 15.52 | 9.18 / 12.48 / 14.15 | 9.90 |
| 128 | beam tyres | 120 | 14.69 / 19.37 / 21.46 | 13.03 / 16.56 / 17.80 | 13.78 |
| 256 | beam tyres | 120 | 18.72 / 25.11 / 28.17 | 18.14 / 23.04 / 25.98 | 18.81 |
| 16 | raycast | 60 | 19.94 / 21.42 / 21.60 | 16.57 / 17.58 / 17.86 | 17.01 |
| 64 | raycast | 60 | 27.41 / 47.44 / 50.12 | 22.40 / 32.32 / 36.41 | 23.20 |
| 128 | raycast | 60 | 38.98 / 52.18 / 53.91 | 30.65 / 40.00 / 40.49 | 31.57 |
| 256 | raycast | 60 | 49.82 / 67.29 / 74.91 | 41.71 / 55.05 / 61.23 | 42.64 |
| 16 | raycast | 120 | 13.37 / 15.73 / 17.69 | 10.99 / 13.31 / 14.15 | 11.63 |
| 64 | raycast | 120 | 15.73 / 26.30 / 28.11 | 13.08 / 20.41 / 22.32 | 13.85 |
| 128 | raycast | 120 | 23.02 / 32.56 / 35.74 | 17.54 / 25.14 / 27.31 | 18.38 |
| 256 | raycast | 120 | 28.82 / 39.76 / 44.55 | 23.63 / 32.13 / 37.44 | 24.53 |

256 beam-tyre cars at 60 Hz: the p99 is within the run-to-run spread (one long step).
512 lanes a vehicle (`RUBBLE_EXPLICIT_LANES=512`) for these fleets: within noise (64 cars,
120 Hz: p99 13.5 against 14.2), so the lanes policy is unchanged.

**Against E1: not met.** 128 beam-tyre cars at 120 Hz: p99 17.8 ms against 8.33. 256 cars at
60 Hz: p99 48.3 ms against 16.7. 16 cars at 120 Hz are at 8.5 ms. What bounds further scaling:

- **The explicit frame is latency-bound.** It is one dispatch; its time is its slowest
  workgroup's 17 or 34 serial steps of about ten barriers each. That workgroup is a contact
  group, or a car among a broken house's chunks. Splitting a group into partitions (two cars:
  128 lanes each) and the vehicle–vehicle pass cost a pair about four times a solo car
  (0.66 against 2.7 ms a frame for 64 cars with the coupling off). The proxy shapes add
  about 3 ms after the strikes. More cars barely raise it, until they fill the GPU.
- **The rigid step.** About 2.9 ms at 32 houses alone, plus the debris the strikes free.
  Broadphase and primal are 1.6 to 1.7 ms each at 64 cars (submit-and-wait split).
- **The threshold fracture.** About 1.7 ms, following the bonds broken.
- The coupling plan is brute force: 0.6 ms at 64 cars and 16 houses. It grows with cars ×
  colliders.

Lossless paths not yet taken:
- a per-step grid of a vehicle's proxy shapes (only the cell's shapes tested; the bitmask
  version measured here was slower, through atomic loads of global memory);
- contact groups' partitions sized by the members' node counts;
- the plan's colliders binned per window;
- the contact module's cold Metal compile (about 62 s for the 428 KB module, past Chrome's
  GPU watchdog): split into smaller modules.

## 9. Remaining

- In-frame release: a chunk proxy freed the substep its load passes its capacity (today the
  vehicle meets the unbroken structure for one window, §4).
- The proxy plan is brute force (vehicles × colliders); a collider grid when collider counts grow.
- Per node, every proxy shape's grown sphere is tested (the shape cull, §8.2); a vehicle inside
  a broken house meets up to about a hundred pieces: a per-vehicle shape grid would cut that.
- E1 is not met (§8.2): the frame's slowest workgroup, the rigid step and the fracture bound it.
- Threshold destructibles with their own fracture (crushing, blast loading) are refused
  loudly; the shared fracture's are supported.
- Wide passes with the coupling (single very large vehicles without vehicle contact).
- Rigid bodies resting *on* vehicles are held by staggered impulses (§2), a few mm of sink.
