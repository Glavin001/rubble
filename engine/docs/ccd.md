# Continuous collision (CCD), GPU impulses and projectiles

How Rubble keeps fast bodies from passing through thin geometry between
steps, all inside the GPU step (`slang/kernels/ccd_step.slang`, scheduled by
`engine_exec::rigid_plan::encode_step` and by the fracture lab's
`engine-fracture/src/plan.rs`). It implements the CCD audit's rows 1–7.

## What a host does

| Want | Call | Notes |
|---|---|---|
| Look-ahead contacts for every awake body | `AvbdParams::flags \|= FLAG_SPEC_VELOCITY` | Off in the engine default, on in the fracture lab (see row 6). A body's contacts reach as far as it moves this step, capped at 0.5 m per collider unless host word 31 says otherwise |
| A body that must never tunnel | `BodyState::flags \|= BODY_CCD` (256) | Uncapped reach, the fast list, a time-of-impact (TOI) contact and a TOI start pose. For projectiles, rockets, thrown debris, vehicles at speed |
| An impulse at a point, a set velocity, or linear + angular | `WgpuSim::apply_impulse`, `set_kinematic_velocity`, `apply_linear_angular_impulse`, `queue_impulses` | Applied on the GPU at step start (`apply_impulses`), summed per body in record order, so the result does not depend on thread timing |
| A kinematic body (platform, elevator, door) | `flags & 3 == 2` and a velocity | Moved by its velocity in `predict`; dynamic bodies ride and are pushed; it wakes what it touches |
| Bullets that pierce | `WgpuSim::spawn_projectile(&ProjectileDesc)` | Segment casts, not bodies: energy cost per metre of collider, impulses on the bodies hit, hit and stop events (`poll_motion_events`) |
| The fracture lab's balls | `Planner::set_projectile_ccd(true)`, the lab's "CCD balls" checkbox | Shots carry `BODY_CCD` from then on |

## The step

```
apply_impulses (host records)
build_aabbs                 swept boxes; CCD and kinematic reach uncapped
broadphase_clear
ccd_register                one slot per BODY_CCD body (scratch, cap 1024)
broadphase_insert           sweeps longer than a cell go to the fast list
broadphase_query            grid entries only
broadphase_fast             each fast collider walks the cells along its sweep
projectile_step             segment casts; apply_impulses (late pass) for hits
write_work_args / carry_rest_pairs
ccd_toi                     GJK ray cast of each CCD pair along its relative motion
ccd_pick                    per body: the earliest TOI, then the lowest candidate
ccd_mark                    tags the winning pair; the narrowphase skips it
narrowphase / convex
ccd_emit                    the winner's manifold at the TOI pose
... graph, colouring ...
predict_bodies              a CCD body starts from its TOI pose
ccd_trust                   bodies a CCD body can strike: trust region for its reach
solve
ccd_validate / ccd_clamp    post-solve sweep check: no CCD body ends past a surface
extract_velocity_sleep
... sleep ...
ccd_respond / _apply        inelastic momentum exchange with a partner held still
```

### Row 1: look-ahead that scales

`colliderReach` caps a body's reach (0.5 m per collider) so one fast body
cannot inflate every neighbour's pairs. CCD and kinematic bodies are exempt.
The bound used to place a box in the hierarchical grid includes its sweep
(`_pad0 = boundR + max(reach, |sweep|)`), so a fast box sits at a level whose
27-cell stencil covers it.

A sweep longer than a grid cell would still miss pairs in the 27-cell stencil,
so `broadphase_insert` puts such a collider on the **fast list** instead of the
grid. `broadphase_fast` gives each one a workgroup: it walks every cell within
reach of its sweep segment (a tube, not the whole swept box) on every level and
table (uniform, split, hierarchical, the lab's rest and awake tables), plus the
plane, large and other-fast lists. It owns every pair it is part of, so nothing
is found twice. `tests/broadphase_oracle.rs` checks the candidate pairs against
a CPU brute force: 0 missed, 0 extra, 0 duplicate on both grids, while the
control (fast list off) misses pairs on the uniform grid and overflows the
hierarchical grid's oversized list.

On the uniform grid an oversized awake box (larger than a cell) is owned by the
larger box, which walks its own extent; the smaller one skips it.

### Rows 2–3: TOI start pose and TOI contacts

`ccd_toi` casts each CCD body's collider along its motion relative to each
candidate (start rotations, as Jolt's `LinearCast`): a GJK ray cast (van den
Bergen) in the pair's local frame. A cast starts one step back, so a pair
already touching and closing gets TOI 0. It ends when the distance stops
shrinking: a hit if the gap is within 5 mm (`GJK_TOUCH`), a miss otherwise.

Each body keeps the earliest TOI by an atomic minimum on its bits, then the
lowest candidate index among equal TOIs, so the winner is deterministic.
`ccd_mark` tags the winner, the narrowphase leaves it alone, and `ccd_emit`
builds its manifold at the TOI pose. The rows hold the gap along the path, so
AVBD solves it as an ordinary contact: real two-way forces, visible to
`last_contacts` and the stress solver. `predict_bodies` starts the body at its
TOI fraction of the motion, so it arrives at the surface rather than inside it.
A TOI that would not matter (the body stops short anyway) is skipped.

Against a body that cannot move this step (static, kinematic, or asleep with its
wake deferred) the TOI row starts rigid (`k_max`): the usual cold start
(m/h², ramping up over the iterations) let a 576 m/s ball through a sleeping
wall piece within the step. Against a free dynamic body it starts as usual (a
rigid row made a 21 t ball yield to a 0.25 kg brick). If `ccd_emit`'s own cast
misses a pair `ccd_toi` gave it (a touching pair at TOI 0 is borderline in f32
across two kernels), it still emits a stopper contact at the stored TOI, normal
along the motion: a pair the narrowphase left to it always gets a contact.

**Momentum through the solve (`ccd_trust`, the trust region).** Against a
free dynamic body the impact goes through the solve, as any contact does. Two
things used to eat its momentum:

- The primal trust region (trustLin · 0.5 = 0.175 m per iteration, 1.75 m a
  step at ten iterations) held both bodies short of the motion the impact
  needs: a CCD body starts at its TOI pose, and a struck box going from rest
  to 214 m/s must move 3.6 m. A 300 m/s ball into a free 2 kg box kept 85% of
  the momentum. A CCD body's steps along its velocity may now reach its whole
  step's travel over the iterations (its position along the sweep stays in the
  swept segment, padded); across it the ordinary trust holds, so its
  broadphase pad stays a hard bound. `ccd_trust` gives every awake body paired
  with a CCD body this step a trust region covering the largest CCD travel this
  step (one value for all, so concurrent writes agree).
- Warm start: a TOI row's force and rigid stiffness carried into the next step
  held the striker dead against the struck body while the body moved off (an
  edge hit lost 84%). Contacts no longer warm-start from a TOI row (features
  `0x0CCD`–`0x0CCF`): `ccd_respond` has already resolved that impact.

**Impact response (`ccd_respond`, `ccd_respond_apply`).** A partner held still
this step (asleep, its wake deferred) took no momentum in the solve: the
striker stopped against it as against a wall. After the sleep pass, each CCD
body whose TOI partner was held gets a perfectly inelastic, frictionless,
momentum-conserving exchange along the TOI normal from both step-start
velocities, with both bodies' rotational effective mass (the partner spins
from the contact point), and the partner wakes with its velocity, under the
per-step rotation limit when `FLAG_MAX_ROTATION` is on. A free partner gets
none: the solve already exchanged momentum (adjusting one side's speed there
broke conservation). The lab's 277 kg ball at 300 m/s sends the 123 kg arch
crown off at the inelastic speed. `ccd_respond` computes each slot's velocity
changes (one thread per slot); `ccd_respond_apply` has one owner thread per
body sum every change addressed to it in slot order: deterministic. (One
thread over the slots cost 3 ms at 512 strikers.)

**Post-solve sweep check (`ccd_validate`, `ccd_clamp`).** In dense rubble the
solve can still push a CCD body into a piece (wedged among contacts, or carried
by a struck piece that spins). After the solve, each CCD pair (listed by
`ccd_pick`; the candidate buffer is the constraint graph's by then) is cast
along the body's solved motion relative to the other, from both step-start
poses, and the body is pulled back along its path to the first contact:
against a body that cannot move this step, any crossing deeper than a quarter of
the thinner body's smallest half extent; against an awake body that stayed
still, the CCD body's centre entering it (a heavy ball's overlap with bricks the
solve has not pushed yet is left to the contacts). The centre test is exact
(`segmentEnters`: slabs for boxes, hull planes, spheres; a GJK point cast
grazing a box edge could stall and miss). Stopped by a dynamic body it keeps
its velocity (it meets that body awake next step); by static geometry it
stops. A CCD body's bounds are padded all round by the farthest the solve can
deflect it off its sweep (the ordinary trust region × iterations, at most its
sweep) so every body it can reach is paired; pairs only the pad found get no
contact row. Spin counts in a body's reach (`spinSweep`).

### Row 4: GPU impulses

The scratch buffer holds up to 1024 impulse records (8 words: body, mode and
key, impulse, point). `apply_impulses` runs one thread per body that has
records and sums them in key order. Host records come first in a step;
projectile hits append records, applied in a late pass in the same step.
Queuing more than the cap panics on the host.

### Row 5: projectiles

`projectile_step` advances each projectile along its arc in up to four
segments. Each segment cast walks the grid (`scene_query.slang` `sceneWalk`)
and orders hits along the path. A hit costs `penetration cost × thickness`
of kinetic energy (the exit point is a second cast from beyond the collider),
pushes the body with an impulse, and logs an event. A projectile stops when its
energy runs out and continues from its exit point otherwise. A character's
inner capsule is a collider like any other, so its hit event names the
character's collider. (A `BODY_CCD` body, such as a rocket, never passes a
character: `tests/character.rs`.)

### Row 6: on by default (evaluated, not adopted)

The fracture lab has look-ahead on (`LAB_FLAGS`). With `AvbdParams::default()`
switched to `FLAG_SPEC_VELOCITY`, every engine-webgpu suite passes (lib 63,
vehicles 32, columns, gables, ring, joints, CCD, characters, queries). The
default is still off because:

- `MetalWorld`'s legacy schedule then diverges from the shared schedule by
  8.4 m in `joints_wgpu_parity` (0 m with look-ahead off), so parity would
  need work on that host first.
- stress-rubble's look-ahead (`STRESS_RUBBLE_SPECULATIVE_CCD=1`) roughly
  doubles broken bonds in the bombard and is an open decision
  (docs/stress-qs-baseline-plan.md).

The tunnelling suite (`tests/ccd_tunnel.rs`) is the other half of row 6: each
case runs with look-ahead off, look-ahead on, and `BODY_CCD` (which turns
look-ahead on for the scene by itself).

### Row 7: content rules

- Thin pieces: a body thinner than its speed × dt tunnels without CCD. Flag
  fast movers `BODY_CCD` rather than raising the step rate.
- 120 Hz halves the distance per step but is no substitute for CCD at
  projectile speeds (the 250 m/s cases tunnel at 120 Hz without CCD).
- `stress-rubble` keeps its whole-world CCD substeps by default; TOI CCD for its
  impactors is opt-in (`set_toi_ccd`, see "In the demos").

## Limits

| Limit | Value | Over it |
|---|---|---|
| CCD bodies per step | 1024 | `OVERFLOW_CCD_BODIES` (4096) |
| Fast list | 16,384 colliders (one workgroup each), 262,144 cells per level walked | `OVERFLOW_FAST_LIST` (1024) |
| Impulse records | 1024 | host panic / `OVERFLOW_IMPULSES` (8192) |
| GJK iterations | 32 | not an overflow: counted in `ccd_fallbacks()`, and the pair keeps its ordinary look-ahead contact |
| Large dynamics on the uniform grid | 16 | `OVERFLOW_LARGE_DYNAMICS` (2048) |
| Candidate pairs of CCD bodies (post-solve check) | 131,072 | `OVERFLOW_CCD_PAIRS` (32768) |

Manifold reuse (`step_reuse_manifolds`, the lab's reuse mode) is off while CCD
bodies exist: a reused step would keep last step's TOI contacts.

The fracture lab's colouring-round limit is 64 (was 32): a 576 m/s ball through
`reinforcedTower`'s dense cubes needs more than 32 rounds (the lab failed loudly
with "choose a coarser level"); rounds past last step's need run only when a
body is blocked.

An off-centre hit near a box's edge at 300 m/s and more spins the box up to a
turn a step; when the ball meets the spinning box again, the ten-iteration
solve loses part of that re-contact's momentum (up to half at 600 m/s into an
awake box; a plain ball does the same at the speeds it does not tunnel at).
Momentum and energy never grow and the ball never ends inside the box
(`tests/ccd_unit.rs`).

## At scale

Measured on an Apple M-series GPU (release, single queue):

| Scene | Plain balls | CCD balls | Phase-throughs (plain → CCD) |
|---|---|---|---|
| 16 brick walls (3,200 bricks), 128 balls at 200 m/s (`ccd_scale`) | 4.8 ms/step | 5.0 ms/step (+5%) | 296 → 0 |
| 64 walls (12,800 bricks), 512 balls at 200 m/s (`ccd_scale`, ignored) | 6.3 ms/step | 6.9 ms/step (+10%) | 2,377 → 0 |
| 25 vibe-city fractured towers (92,775 hull pieces, all awake), a ball every 15 steps at 16× (`fracture_scale_ccd`, 3 seeds) | 18.3–18.9 ms/step | 20.3–20.9 ms/step (+10–11%) | 0 with CCD |

### The towers showcase (real time at 120 Hz)

`fracture_viewer towers --grid 2x2 --bonds --bond-strength 0.2 --volley --speed 16
--ball 1.0,30000`: four vibe-city fractured towers (14,844 pieces, 43,040
breakable welds at 0.2× the pack's limits: a weaker masonry, so the towers can
fall) under a scripted volley of 30 t, 1 m balls at 576 m/s: eight raking
shots at 20–45° off a face, aimed off the axis so they clip a tower and carry
on into the next, then sixteen at the lower floors until the towers fall into
each other. Ball trails: cyan with CCD, red without (`X` toggles, `V` replays
the volley). The title shows TPS, × real time, the physics batch's GPU time per
tick (its command buffer's GPUStartTime..GPUEndTime) against the 8.33 ms
budget, and the draw's GPU time per frame. `towers_capture` renders the same
volley offscreen to an MP4 (identical camera, CCD on or off).

On an M3 Max, live: 0.98–1.00× real time throughout, physics 7.0–7.8 ms/tick
on the GPU, drawing 0.7–0.8 ms/frame. Headless (`tests/towers_volley.rs`,
`--ignored`, every ball checked every step): CCD 0 phase-throughs at 6.9
ms/tick (worst second 7.3), plain 6–11 phase-throughs.

| Towers | Pieces | Physics GPU at the peak (ms/tick) | Real time |
|---|---|---|---|
| 3 × 1 | 11,133 | 5.8–6.3 | 1.00× |
| 2 × 2 | 14,844 | 7.0–7.9 | 0.98–1.00× |
| 3 × 2 | 22,266 | 8.0–8.6 | 0.91–0.98× |
| 4 × 2 | 29,688 | 8.1–9.4 (headless) | below |

The solve (bonds and contacts) is about 70% of a tick at these sizes
(`FractureSim::profile_step`, `VOLLEY_PROFILE=1`); the CCD passes are a few
percent. The viewer sizes its tick batch from that GPU time, not the wall time
(which includes the wait for the next drawable), so a late frame catches up the
next one instead of losing the tick.

**In the browser lab** (`fracture/`, Scene → Vibe City presets → "Fractured
towers · 2×2 city block", then **Fire the volley**): the same block, ball and
volley (`engine_fracture::towers`, through `Scene.towers` / `Scene.volley` in the
lab's wasm), Bonds: breakable D6 at 0.2× strength, Ball speed 16×, CCD balls on.
Ball trails (cyan CCD, red plain) are drawn in every lab scene. Headless Chrome
on the M3 Max: 120 TPS, 1.00× real time after the first second (GPU 5.6–6.5
ms/tick), 0 phase-throughs with CCD, 7 without (`motion-check.mjs towers`). The
lab sizes its tick batch from the GPU timestamp time too: at 9 ms wall (the
counter readback included) for a 6 ms GPU tick it ran one tick per frame, so a
60 Hz display held physics at 60 TPS.

## Tests

| Test | Shows |
|---|---|
| `engine-webgpu/tests/ccd_tunnel.rs` | 60 cases (60/140/250 m/s; 0.05/0.2/1 m slabs and brick walls; 60/120 Hz; both grids) hold; the controls tunnel. A debris shower through a slab: 0 through with CCD vs 51 (60 Hz) and 25 (120 Hz) without |
| `tests/broadphase_oracle.rs` | Candidate pairs equal a CPU brute force for sweeps of 0.5–10 cells |
| `tests/kinematic.rs` | Kinematic platforms carry and push; sleeping bodies wake |
| `tests/queries.rs` | Ray and shape casts match a full scan; bullets pierce five walls with exact energy accounting; impulses are exact and replay bit-identically |
| `stress-rubble/tests/engine_world.rs` | With TOI CCD on, a 250 m/s CCD impactor stops at a 0.1 m wall with one step per frame |
| `tests/static_pairs.rs` | A box lands on a lower-index static pedestal (on main it fell through: the grid query dropped dynamic–static pairs whose static had the lower index) |
| `engine-viewer/tests/fracture_regression.rs` | A 300 m/s lab ball tunnels through the arch without CCD and knocks the crown out with it |
| **`engine-viewer/tests/fracture_tower_ccd.rs`** (acceptance) | vibe-city `fracturedTower` (3,711 hull pieces) and `reinforcedTower` (6,308 box pieces) in the lab's own plan: balls from four sides at three heights, four corners and two plunging roof shots at 1×–64× the shot speed (to ~2,300 m/s), bonds off at 120 and 60 Hz and D6-welded: 234 shots, no phase-through (the ball's centre path in a step meets a piece that did not move that step; exact GJK segment–hull distance). The CCD-off controls phase through dozens of times |
| `tests/ccd_gallery.rs` | `motion.html`'s gallery natively, native and browser contact modes: 0 of 49 CCD cubes through the slab (40 without) |
| `tests/ccd_plough.rs` | The ring benchmark's 21 t ball ploughs through a brick wall with CCD as without, and does not ride up |
| **`tests/ccd_unit.rs`** | The TOI is exact (30–2,000 m/s, a static wall at 0.5–2.5 steps); momentum and energy (above) into a sleeping and an awake box, head on and off centre, 30–600 m/s; a sleeping slab is never passed and takes the hit; resting and rolling CCD balls behave as plain ones; more CCD bodies than the cap fail loudly. Native and browser contact modes |
| **`tests/ccd_scale.rs`** | 128 CCD balls through 16 brick walls: 0 phase-throughs (a plain run: hundreds), cost reported; the 512-ball, 64-wall run is `--ignored` |
| `tests/ccd_ground_skim.rs` | A CCD ball skimming the ground never passes the bottom course |
| **`engine-viewer/tests/fracture_scale_ccd.rs`** (`--ignored`) | `SCALE_GRID`×`SCALE_GRID` fractured towers under a stream of balls in the lab's plan on Metal: timing (no shots, plain, CCD) and every ball checked every step for a phase-through |
| **`engine-viewer/tests/fracture_tower_ccd.rs` `…cuboid_house`** | The lab's procedural cuboid house (715 boxes, and the same as hulls), built in code so it runs everywhere: the tower shots at 4–64×, 0 phase-throughs; the CCD-off control phases through |
| **`engine-viewer/tests/towers_volley.rs`** (`--ignored`) | The showcase's volley headless: ms/tick against the 120 Hz budget, and every ball every step checked for a phase-through, CCD on (0) and off (the control) |
| `motion-check.mjs live` (`fracture/regression.mjs runLiveShotRegression`) | The lab as a person uses it: the frame loop running (heavy steps sliced across frames), 24 shots through the page's click path while earlier balls fly into an already broken tower, every step captured and every ball checked: both towers 0 phase-throughs with CCD, 8 and 32 without. `LIVE_CASES=all` sweeps the panel (60 Hz, D6 bonds, ball radius and mass, the house, cuboid, grid and arch scenes) |
| `motion-check.mjs towers` | The towers block's volley in the browser: phase-throughs (0 with CCD) and the page's rates |
| `engine-webgpu/motion-check.mjs` | The same in headless Chrome: `motion.html`, `demo.html` (wall, ring check with `?ccd=1`), `vehicle.html`, and the fracture lab (arch; `tower`: both towers at 16× and 64×, 72 shots, with sensitive controls) |

## In the demos

| Page | CCD | Switch |
|---|---|---|
| `motion.html` | CCD gallery, character rockets, bullets | per lane, per body |
| `demo.html` | the smash balls of the built scenes and the snapshot scenes' F projectile carry `BODY_CCD`; `demo.html`'s own schedule runs the CCD passes only in a scene that has one | `?ccd=0` off. The `?check=ring\|gables` regressions keep it off unless `?ccd=1` (their native baselines run without it); the ring check passes with it too |
| `vehicle.html` | cannonballs (B) | `?ccd=0` off |
| `fracture/` | Rubble bond modes: the "CCD balls" checkbox (on). Stress-solver modes: whole-world substeps for their balls (below) | checkbox |

stress-rubble keeps whole-world CCD substeps by default: `BODY_CCD` turns on
look-ahead contact semantics for the whole scene, and the stress coupling's breakage
was measured against substeps (with `BODY_CCD`, `punch_gpu`'s ram left the wall at
6.5 m/s instead of 28, reference 35.1). `RubbleEngine::set_toi_ccd(true)` or
`STRESS_RUBBLE_TOI_CCD=1` opts in.

## Browser demo

`crates/engine-webgpu/motion.html`: `?scene=ccd` (ball gallery, CCD on/off
lanes, a debris shower), `?scene=character`, `?scene=bullets`. The fracture
lab has a "CCD balls" checkbox next to Ball speed.
