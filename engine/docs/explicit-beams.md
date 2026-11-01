# Explicit beams: Rubble's fixed-step node–beam solver

Specification for `engine_softbody::explicit_beams` (CPU reference) and
`engine_webgpu::ExplicitBeamsGpu` (`slang/kernels/explicit_beams.slang`). It describes
behaviour: the equations, rules, constants and order of operations. It is the only
input for the implementation, which is written in Rubble's style from this document.

## Purpose

A soft-body vehicle is nodes (point masses) joined by beams (spring–dampers that yield,
harden, weaken and break). This solver advances them with small fixed explicit steps,
which is what makes node–beam vehicles crash convincingly. Rubble's goal is to match
the established node–beam simulators in accuracy and exceed them in performance by
running every vehicle on the GPU, every step of a frame in one dispatch. This is the
foundation: later work adds wheel torque, vehicle–vehicle contact, coupling with
Rubble's rigid world, BeamNG-style elements and multi-workgroup vehicles.

Accuracy is measured against Rigs of Rods (RoR) as a black box: a local tool runs
RoR's own program on the same `.truck` file and scene and compares results
(`soft_calibrate REF=ror`, `ror_compare`). The behaviours below that exist to reproduce
RoR's results (its approximations, its yield force, its break exception) form the
**RoR-compatible profile**; each is a switch in `ExplicitBeamsSettings` so Rubble can
later choose better physics where it wants to.

## Units and constants

| name | value |
|---|---|
| step h | 0.0005 s |
| gravity | −9.807 m/s² (y up) |
| air drag coefficient | 0.05 |
| shortest rest length after yielding | 0.1 m |
| default support break factor | 4.0 (× rest length) |
| frame | 1/60 s; steps per frame = ⌊(1/60)/h⌋ = 33, the remainder carried to the next frame |
| origin rebase radius | 100 m |

Contact surfaces (two are used: ground and static boxes):

| surface | adhesion speed va | static μs | sliding μc | hydrodynamic t2 | Stribeck speed vs | exponent α | strength |
|---|---|---|---|---|---|---|---|
| ground ("gravel") | 3.0 | 0.85 | 0.6 | 0.006 | 3.0 | 2.0 | 1.0 |
| boxes and self-collision ("concrete") | 3.0 | 1.2 | 0.75 | 0.01 | 6.0 | 2.0 | 1.0 |

## Numerical primitives (RoR-compatible profile)

All arithmetic is IEEE single precision unless stated otherwise. These are well-known
published approximations:

* **Inverse square root** (the Quake III method): bits(y) = 0x5f3759df − (bits(x) >> 1)
  (arithmetic shift of the signed integer view), then one Newton iteration
  y ← y·(1.5 − 0.5·x·y·y). Used for beam lengths (switch `fast_inv_sqrt`; off = exact).
* **Square root**: bits = ((bits(x) − 0x3F800000) >> 1) + 0x3F800000 (signed integers).
  Used for the drag speed.
* **Exponential** (Schraudolph 1999): for x < −15 the result is 0, for x > 88 it is
  1e38; otherwise bits = trunc(12102203.0·x + 1064652319.0) evaluated in single
  precision (the constant itself rounds to the nearest float).
* **Power** xʸ: bits = trunc(y·float(bits(x) − 1065353216) + 1065353216.0), single
  precision.
* **Turbulence noise**: a 32-bit signed state starting at 1; each draw multiplies it by
  16807 (wrapping); the value is the float whose bits are 0x40000000 OR (state AND
  0x7FFFFF), minus 3, in [−1, 1). Draws happen in node order, three per node (x, y, z)
  per step, one sequence per vehicle. (The GPU may use any well-mixed per-node,
  per-step hash instead; then GPU and CPU differ only in noise.)

## State

Per node: position relative to the vehicle's origin, velocity, accumulated force,
mass, friction coefficient, and flags: *no ground contact*, *immovable*, *on a
collision triangle*, *contacter*, *tyre*.

Per beam: end nodes a, b; stiffness k; damping c; rest length L and spawn length L₀;
compression threshold Tc, tension threshold Tt (stored as a positive magnitude here;
note the tension yield acts on negative forces); a gate value G; strength S; plastic
coefficient p (0..1); kind; type; limit stiffness kℓ and limit damping cℓ (limited
springs); short and long bounds sb, lb (fractions of L); detacher group g (0 none,
> 0 a group master, < 0 a follower of group |g|); broken; last force.

Kinds: **plain**, **limited spring** (shocks and some tyre spokes), **support**
(no stiffness when stretched), **rope** (no stiffness when compressed).
Types: **deformable** (may yield), **actuator** (shocks and hydros; never yield),
**virtual** (never yields; tyre rigidity beams).

Per collision triangle: its three nodes in file order (n₀, n₁, n₂) and a throttle
(two integers: *wait*, *idle*, both 0 at spawn).

Per vehicle: origin (world position the node positions are relative to), static boxes
(axis-aligned, lo and hi), ground height (flat), collision range r (from the file,
default 0.02 m).

## Spawning from a `.truck` file

Use the existing loader (`truck::load_truck`, `parse_truck`); its node masses,
cinecam nodes and beam parameters are already exact. Then:

* **Nodes**: the desc's nodes in order (truck nodes, then cinecam nodes), then tyre
  nodes (below). Position = spawn position + node offset, added in single precision.
  Velocity = spawn velocity (zero for immovable nodes). Flags: no ground contact when
  the desc node has collision off; immovable when fixed; on a collision triangle when
  it belongs to a triangle with collision on; contacter exactly when listed in the
  file's `contacters` section (the desc marks every colliding node a contacter when
  that section is absent — do not use that here); tyre for tyre nodes.
* **Beam order** (it matters for results): cinecam beams, tyre beams, then the file's
  `beams` (plain, supports, ropes, in file order), then shocks, then hydros.
* **Rest length** L = L₀ = |xa − xb| from the spawned single-precision positions (so a
  vehicle spawned far from the origin carries that rounding); shocks multiply by their
  precompression.
* **Deformable beams**: Tc = Tt = desc `deform`; G = Tc; S = desc strength.
  **Actuators and virtual beams**: no yield; G = S.
  **Limited springs from shocks**: kℓ = desc `deform` (the limit spring), cℓ = desc
  `limit_damp` (12000 if 0), bounds from the desc.
* **Tyres** (`wheels` section only; other wheel sections are an error): for each
  wheel, with axle nodes A1, A2 (spawned positions), radius R, ray count n, mass M,
  tyre spring and damping, the beam defaults at its line (unscaled break strength,
  deform threshold, plastic coefficient) and the node friction default:
  * axis = normalize(A2 − A1); first spoke s = normalize(axis × (1,0,0)) or, if that
    has squared length < 1e-12, normalize(axis × (0,1,0)); s ← R·s;
  * rotation angle θ = −360°/(2n), as a single-precision quaternion
    (w, v) = (cos θ/2, sin θ/2·axis); rotate a vector u by u + 2w(v × u) + 2 v × (v × u);
  * for i in 0..n: outer node Oi at A1 + s, then rotate s; inner node Ii at A2 + s,
    then rotate s. Each tyre node: mass M/(2n), contacter, tyre, ground contact on;
  * beams per ray i (next = (i+1) mod n), all with the tyre spring and damping:
    (A1, Oi) and (A2, Ii) limited springs with sb = 0.66, lb = 0, kℓ = 9 000 000,
    cℓ = 12 000, type deformable but never yields (limited springs do not yield);
    (A2, Oi), (A1, Ii) plain; rim: (Oi, Ii), (Oi, Onext), (Ii, Inext), (Ii, Onext)
    plain; if the wheel names a rigidity node P: one virtual beam (P, Oi) when P is
    nearer A1 than A2, else (P, Ii).

## One frame

Once per frame, before its steps: if node 0's relative position is farther than 100 m
from the origin, move the origin there and subtract that offset from every node.
Then the frame's steps.

## One step

Three passes, in this order. Forces accumulate across passes and across the step
boundary: the node pass consumes what the previous step's beam and self-collision
passes produced. At spawn every node's force is zero.

### 1. Nodes (each node, in order)

1. Unless the node has no ground contact: with the world position x = origin + rel:
   * if the ground height is above x.y: add the contact force (below) with normal +y,
     penetration = ground height − x.y, the ground surface;
   * then for every static box with x strictly inside on all three axes: pick the
     face: start with the low-z face (normal −z, distance x.z − lo.z); replace it,
     only when strictly nearer, by the high-z face (+z), then low-x (−x), high-x (+x),
     low-y (−y), high-y (+y). Add the contact force with that normal, penetration 0,
     the box surface. Each contact sees the node's force including contacts added
     before it in this pass.
2. Unless immovable: v ← v + F/m·h, then rel ← rel + v·h.
3. Restart the force: F ← (0, m·gravity, 0). With drag on: s = approximate √(v·v),
   F += −0.05·s·v; with turbulence on also F += (0.05·s)·s·0.005·(three noise draws).

### 2. Beams (each live beam, in spawn order)

1. d = rel_a − rel_b; q = d·d; ι = inverse square root of q; ℓ = q·ι; e = ℓ − L;
   relative speed u = (v_a − v_b)·d·ι. Start with k and c from the beam.
2. Kind rules:
   * limited spring: overshoot o = e − lb·L if e > lb·L, else −e − sb·L if e < −sb·L,
     else 0. If o ≠ 0: k ← k + (kℓ − k)·o and c ← c + (cℓ − c)·o (o in metres,
     not clamped).
   * support: if e > 0: k ← 0, c ← 0.1·c; and if e > L·f (f = lb when lb > 0, else 4)
     the beam breaks (its force this step is still applied, with k = 0).
   * rope: if e < 0: k ← 0, c ← 0.1·c.
3. Force (positive = compression, pushing a away from b): F = −k·e − c·u. Store it as
   the beam's last force. Let τ = |F|.
4. If τ > G:
   * if the beam is deformable, not a limited spring, and k ≠ 0:
     * **compression yield** if F > Tc and e < 0: δ = e + (Tc/k)·(1 − p);
       L_old = L; L ← max(L + δ, 0.1); F ← F − (F − Tc)/2; τ ← F; if L < L_old:
       Tc ← Tc·L_old/L and G ← min(Tc, Tt, S);
     * **tension yield** else if F < −Tt and e > 0: δ = e − (Tt/k)·(1 − p);
       L_old = L; L ← L + δ; F ← F − (F + Tt)/2; τ ← −F; if L > L_old:
       Tt ← Tt·L/L_old and G ← min(Tc, Tt, S); and in any case S ← S − δ·k
       (the beam weakens by its plastic stretch);
   * **break** if τ > S: unless either end is on a collision triangle and has fewer
     than 3 live plain beams (live = not broken; plain = kind plain, any type), the
     beam breaks: F ← 0. If it is a group master (g > 0), every beam whose |g| equals
     g breaks too, at once (beams later in this pass are then skipped this step;
     earlier ones keep the force they applied). If an end saved it, S ← 2·G instead.
5. Apply: node a gets d·(F·ι), node b gets −d·(F·ι).

### 3. Self-collision (vehicles with contacters and collision triangles)

For each collision triangle in file order:

1. Throttle: if wait > 0: idle += 1, wait −= 1, skip the triangle. Else if idle > 0:
   wait = min(idle, 12), idle = 0.
2. With p₀, p₁, p₂ the world positions of n₀, n₁, n₂: u = p₁ − p₀, w = p₂ − p₀,
   n = normalize(u × w) (left as is when zero). Solve p − p₀ = α·u + β·w + δ·n for
   (α, β, δ) with the inverse of the matrix with columns (u, w, n); if its determinant
   has magnitude ≤ 1e-6 every (α, β, δ) is (0, 0, 0). γ = 1 − α − β.
3. Candidates: contacter nodes that are not tyre nodes, not n₀/n₁/n₂, and inside the
   triangle's bounding box grown by r on every side (inclusive). A candidate h hits
   when α ≥ 0, β ≥ 0, γ ≥ 0 and |δ| ≤ r. On a hit: if δ < 0, flip n and δ; penetration
   = r − δ; relative velocity v = v_h − (α·v₁ + β·v₂ + γ·v₀); triangle mass
   m_t = α·m₁ + β·m₂ + γ·m₀; reduced mass m = m_h·m_t/(m_h + m_t); f = contact force
   with node h's current force, v, m, n, the box surface, that penetration; add f to
   h and −α·f, −β·f, −γ·f to n₁, n₂, n₀.
4. After the candidates: wait = −20000 if anything hit, else wait += 1.

### Contact force

Inputs: the node's current force F, a velocity v, a mass m, a unit normal n, the step
h, a surface, a penetration p. vn = v·n, fn = F·n. If p < 0 the force is zero.
Otherwise:

* reaction R = −fn; if vn < 0: R −= (0.8·vn − 0.2·p/h)·m/h.
* If R ≤ 0 the force is zero. Otherwise: slip force sf = F − fn·n, slip velocity
  s = v − vn·n, slip speed σ = |s| (s normalised when σ > 0), Gr = R·strength·friction
  (friction = node h's coefficient), Ms = μs·Gr:
  * **static** when σ < va and Gr > 0 and |sf|² ≤ Ms²: force = R·n − Ms·(1 −
    exp≈(−σ/va))·s − sf;
  * **sliding** otherwise: g = μc + (μs − μc)·exp≈(−pow≈(σ/vs, α)); force =
    R·n − (g + min(t2·σ, 5))·Gr·s.

(exp≈ and pow≈ are the approximations above.)

## GPU design (`ExplicitBeamsGpu`)

* One 256-lane workgroup per vehicle; one dispatch per frame runs the origin rebase
  and all the frame's steps; lanes stride over nodes, beams and triangles.
* Barriers between passes, all in uniform control flow (Chrome's WebGPU rejects
  barriers under data-dependent branches: loop bounds from a uniform buffer, no early
  returns before a barrier; values decided by one lane pass through workgroup memory
  and a barrier).
* No float atomics. Beam forces go to a per-beam slot; the node pass sums its beams
  through a per-node incidence list (beam index plus an "is end b" bit). The
  self-collision pass accumulates into per-node integer atomics in fixed point
  (1/32 N).
* Detacher groups: a breaking master records its beam index into its group slot with
  an atomic minimum; after a barrier every live beam of a broken group breaks, and
  beams after the master's index drop this step's force (earlier ones keep it).
* The live-plain-beam count for the break exception reads the incidence list at the
  time of the test.
* Every vehicle is independent; the dispatch scales with the vehicle count.
* Targets (M3 Max): the Agora S bus (195 nodes, 862 beams with tyres) at ≤ 0.6 ms per
  60 fps frame for 1–40 vehicles, ≤ 9 ms for 1000.

## Required API

`engine_softbody::explicit_beams`:
`ExplicitBeamsSettings { fast_inv_sqrt, drag, turbulence }` (all true by default =
RoR-compatible), `ContactSurface { va, ms, mc, t2, vs, alpha, strength }` with
`GROUND` and `BOX`, `ExplicitBeams` with public node, beam, triangle and box state,
`from_truck(text, position, velocity, settings)`, `new(desc, position, velocity,
settings)` (no tyres), `step()`, `rebase_origin()`, `run(seconds)` (rebase every
frame), `positions()` (world), `STEP` = 0.0005.

`engine_webgpu::ExplicitBeamsGpu`: `new(device, queue, &[ExplicitBeams])`,
`frame(steps)`, `wait()`, `positions()`, `velocities()`, `beams()` (broken, L),
`set_velocity(vehicle, node, v)`; WGSL via `build.rs` like `soft_skin`
(`EXPLICIT_BEAMS_WGSL`).

## Tests

From this document's rules: the approximations' defining properties, compression
yield and hardening, the halfway force and breaking above strength, the break
exception, detacher groups, tension weakening, supports and ropes, limited-spring
bounds, the contact force stopping a node against a box without bounce, tyre
geometry (2n nodes on circles of radius R around each axle node, 8–9 beams per ray),
spawn order; and GPU against CPU on the procedural sedan (`car_truck_bare` with
`minimass` 10): rest within millimetres, a 56 km/h wall crash with matching broken
and bent counts.
