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
predictor (it is the exact soft-body response over the window, not an estimate). The
impulses of window N+1 are emitted after the split, to each chunk's new body: a piece that
breaks free receives the vehicle's push on its chunks, so momentum closes across the
fracture. Nothing is re-simulated: the frame is computed once, its loads applied once.

Latency, stated: inside window N+1 the *vehicle* meets the structure as it stood at the
window's start (unbroken, or anchored). A piece that breaks off in that window takes the
impulse the vehicle exchanged with the unbroken structure. Over one window (≤ 8.3 ms at
120 Hz) a car's crush force is still rising, so this is a fraction of the strike; the
test reports the energy of the freed pieces against the vehicle's loss. The exact
alternative, releasing a chunk proxy inside the frame the substep its load passes the
capacity, is a follow-up (it puts the threshold rule into the explicit kernel).

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
- Frame: per step, for a vehicle with proxies, one extra barrier and a count pass over its
  nodes against its proxy shapes (bounding-box culled); vehicles with none pay nothing.
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

## 8. Results

(filled in as the work lands)
