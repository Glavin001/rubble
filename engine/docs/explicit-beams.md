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

Kinds: **plain**, **limited spring** (shocks and some tyre spokes), **shock2**
(`shocks2`, `explicit-beams-wheels2-shocks2-spec.md` §6), **support**
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
  `beams` (plain, supports, ropes, in file order, including lines whose ends are
  nodes a `wheels` line generated, which legacy numbering lets a file name), then
  shocks, then `shocks2`, then `shocks3`, then commands (`commands`, `commands2`),
  then hydros, each section in file order. Tyre beams are those of every `wheels`
  line, then every `meshwheels` line, then every `meshwheels2` line, whatever the
  file order (legacy node numbers still follow the file order).
* **Rest length** L = L₀ = |xa − xb| from the spawned single-precision positions (so a
  vehicle spawned far from the origin carries that rounding); shocks multiply by their
  precompression.
* **Deformable beams**: Tc = Tt = desc `deform`; G = Tc; S = desc strength.
  **Hydros and commands**: actuators (no yield) with G = the deform threshold at
  their line and S = the scaled break threshold, so they break only when
  |F| > max(G, S) (checked against the reference: a hydro under a load above S but
  below G holds). **Other actuators and virtual beams**: no yield; G = S.
  **Limited springs from shocks**: kℓ = desc `deform` (the limit spring), cℓ = desc
  `limit_damp` (12000 if 0), bounds from the desc.
* **Tyres** (`wheels`, `meshwheels`, `meshwheels2`; `wheels2` and
  `flexbodywheels` are an error). `meshwheels` is `wheels` with another argument
  layout; `meshwheels2` differs in its rim beams' rates and its spokes' long bound
  (`explicit-beams-wheels2-shocks2-spec.md` §2). For each
  wheel, with axle nodes A1, A2 (spawned positions), radius R, ray count n, mass M,
  tyre spring and damping, the beam defaults at its line (unscaled break strength,
  deform threshold, plastic coefficient) and the node friction default:
  * A1 is the axle node of lower z (the file's first on a tie), A2 the other;
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

## Wings, fuselage drag and commands

`wings`, `fusedrag` and `commands`/`commands2` with no input follow
`explicit-beams-aero-commands-spec.md` (`engine_softbody::explicit_aero`): per frame
the wing break check runs before the origin rebase; per step, after the node pass,
the wings add their forces, the fuselage drag D is computed for the next step's force
restart (a vehicle with a `fusedrag` line gets D there instead of air drag and
turbulence), and self-centering command beams move their rest lengths; then the
beams. Airfoil tables are read at run time (`RUBBLE_AIRFOIL_DIR`, or a loader given
to `from_truck_with_airfoils`); Rubble ships none.

## Sections, triggers, slide nodes, animators

`section` / `end_section` configurations, `triggers` (with the command motion they
drive), `slidenodes` and `tacho` animators with no engine follow
`explicit-beams-sections-triggers-slidenodes-spec.md`
(`engine_softbody::explicit_triggers`):

* the loader spawns the root module plus one configuration
  (`truck::parse_truck_config`, `ExplicitBeams::from_truck_config`; default the
  first `section` name in byte order), every list's root lines before the module's;
  `TruckFile::section_configs` and `section_config` report the names and the choice;
* beam order after hydros: triggers (zero-force beams of kind trigger), then
  animators (constant-length actuators); slide nodes add no beams;
* per step: with triggers every command line moves by its keys in the command pass,
  reading the trigger inputs the last beam pass set (one step of latency); the beam
  pass runs each trigger's bounds test; after self-collision the slide nodes add their
  spring forces in spawn order;
* unsupported options (trigger `x b B A s h H t E` or long action −1, slide-node
  `R D C G`, `railgroups`, `slidenode_connect_instantly`, animator sources other
  than `tacho`, a trigger-held command with key inertia or `o`/`p`) are refused:
  `from_truck` fails with the parser's `TruckFile::errors`.

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

## Vehicle contact (`engine_softbody::explicit_contact`)

Contact between vehicles follows `explicit-beams-vehicle-contact-spec.md`;
`ExplicitBeamsWorld` steps several vehicles together. Spawn data (built by
`ExplicitBeams::finish`): `contact_points` (contacters, plus the nodes with ground
contact that lie on a collision triangle, or every such node when the vehicle has no
collision triangles), `neighbours` (per node the far ends of its beams in beam
order, duplicates and broken beams kept) and a contact throttle (cwait, cidle) per
collision triangle. A file naming a contact surface other than `concrete`
(`submesh_groundmodel`) or with more than 3000 collision triangles fails to spawn; a
`cab` line with a solid and a buoyant option is two triangles.

One world step: every vehicle's own step, in vehicle order; each vehicle's box (its
node world positions, every node, grown by 0.05 m); then for each owner in vehicle
order its partner scan (every other vehicle whose box overlaps; first nodes more
than 4 m/s apart reset both vehicles' contact and self-collision throttle waits to
0) and, when its partners offer a contact point, its triangle loop: per collision
triangle in list order the contact throttle (wait 0 → wait min(idle, 12), tested;
no candidate → +1; a hit → 0), the partners' contact points inside the triangle's
box grown by the owner's range, the self-collision hit test, the side vote (3·sign δ
plus, with more than 3 neighbours, the sign of each neighbour's height over the
plane), and the contact force with the reduced mass, the owner's range and the
concrete surface, its reaction on the triangle's nodes by their weights. The forces
are read by the next step's node pass. `world.contact = false` (or vehicles whose
boxes never overlap) gives each vehicle's results bit for bit as stepped alone.

Spawning turned: `ExplicitBeams::from_truck_config_yaw(text, config, position,
yaw_degrees, velocity, settings)` turns every node offset by the yaw about +y
(x' = c·x + s·z, z' = −s·x + c·z) before adding the position; tyres are built in
the file's axes and turned with the rest (Rigs of Rods' convention, checked against
its spawned positions). The velocity is in world axes.

## GPU design (`ExplicitBeamsGpu`)

* One 256-lane workgroup per vehicle; one dispatch per frame runs the origin rebase
  and all the frame's steps; lanes stride over nodes, beams and triangles.
* Barriers between passes, all in uniform control flow (Chrome's WebGPU rejects
  barriers under data-dependent branches: loop bounds from a uniform buffer, no early
  returns before a barrier; values decided by one lane pass through workgroup memory
  and a barrier).
* No float atomics. Each node has an incidence list (beam index plus an "is end b"
  bit); the beam pass writes its force on each end into that end's incidence entry,
  so the node pass sums a contiguous run of independent loads, in list order. The
  self-collision pass accumulates into per-node integer atomics in fixed point
  (1/32 N), spreading (triangle, contacter) pairs over the lanes (the sums do not
  depend on the order); a triangle's throttle is finished in the next node pass.
* Detacher groups: a breaking master records its beam index into its group slot with
  an atomic minimum; after a barrier every live beam of a broken group breaks, and
  beams after the master's index drop this step's force (earlier ones keep it).
* The live-plain-beam count for the break exception reads the incidence list at the
  time of the test.
* Every vehicle is independent; the dispatch scales with the vehicle count.
* Fleets with wings, a `fusedrag` line or self-centering commands run a separately
  compiled module (`EB_AERO`, 256 or 512 lanes) so every other fleet keeps the exact
  old code: after the node pass a lane per wing computes its forces, a lane the
  fuselage drag D, a lane per command its rest length; a barrier; then one lane adds
  the wing forces in file order while the others run the beams. The specification's
  double-precision sub-expressions run in strict single precision there. The wide
  passes are off for such fleets.
* Triggers, key-driven commands and slide nodes also run in the `EB_AERO` modules
  with no added barrier. A trigger beam stores 1 into its key's slot of a
  per-vehicle, per-step-parity table (plain stores); the next step's command lanes
  read the other parity, and its beam phase clears it. One lane per vehicle runs the
  slide nodes in spawn order during the self-collision phase, adding into float
  slots of the slide nodes and rail ends (flagged in their node info) that the next
  node phase adds and clears. Slide-node loops are bounded by static counts and
  every index is clamped.
* Targets (M3 Max): the Agora S bus (195 nodes, 862 beams with tyres) at ≤ 0.6 ms per
  60 fps frame for 1–40 vehicles, ≤ 9 ms for 1000.

### Vehicle contact on the GPU

On by default for a fleet of two or more vehicles (`set_vehicle_contact(false)`:
the one-dispatch frame above, unchanged). A frame is then, in one compute pass:

1. `explicit_beams_vv_bounds` (a workgroup per vehicle; planning frames only): the
   box of its node world positions (checked against the last plan's frame bounds)
   and per axis the extreme node velocities; its *frame bounds*, each face of the
   box moved out by 2·(the fastest node velocity that way)·(planned time) + a
   margin (`set_contact_margin`, default 1 m), where a plan covers the next frames
   of at least 34 steps (every frame at 60 fps, every other frame at 120; a new
   plan when a frame needs more steps than are left, after `set_velocity`, or when
   contact or the margin changes); its *padded frame box*, the bounds grown by
   0.05 m, into a hashed vehicle grid (a box over more than 27 cells, or a full
   bucket, goes on a list tested against every vehicle).
2. `explicit_beams_vv_group` (one workgroup, planning frames only; with more than
   2048 vehicles the pair search is its own pass, `explicit_beams_vv_pairs`, a
   thread per vehicle): the pairs of overlapping padded frame boxes, their
   connected components (label propagation with pointer jumping), each vehicle's
   grouped flag, each group's members, and the groups' indirect dispatch
   arguments. With no pair it returns at once (all solo).
3. The solo frame, a workgroup per vehicle: `explicit_beams_frame` compiled with
   `EB_VV_SOLO` (a separate module, so the plain frame's code is untouched): a
   grouped vehicle's workgroup returns at once (a uniform decision through
   workgroup memory); every other runs the frame, checking at the start of each
   node update that the position the last step left is inside its frame bounds,
   and the last step's positions at the frame's end.
4. `explicit_beams_contact`, indirect, one workgroup per group: the members sorted
   by vehicle index; each member's frame start (wing check, rebase); then per step
   each member's step (a copy of the frame's step, every accumulator read, member
   by member with the whole workgroup; its node phase also adds the member's step
   box into fixed-point min/max keys and checks it against the frame bounds)
   followed by the contact pass; then each member's frame end.

Why the solo vehicles need no contact pass: two vehicles whose padded frame boxes do
not overlap, each with every node inside its frame bounds at every step, have
step boxes that cannot overlap, so they are never partners. A node leaving its
frame bounds (a node accelerated past the bound) is reported: the host reads a
status word every 16 frames without waiting and panics ("vehicle N left its frame
bounds … raise set_contact_margin"); `contact_status()` reads it at once. The
margin trades grouping (time) for headroom, never results.

The contact pass of a group, per step (the members' boxes are already in the keys),
for each owner **in vehicle order**, three barriers each: (S) the partner scan
over the other members in parallel (box overlap; a fast approach sets both
vehicles' contact throttle waits to 0 and marks their self-collision throttles for
a reset, applied right after the next throttle update, which the GPU makes at the
next node pass), with the previous owner's throttle updates; (P) the contact
throttle resets, the owner's throttle step (only when its partners offer points),
listing its tested triangles whose box grown by the range meets a partner's box
(the others have no candidate: +1 at once), and the partners' contact points
inside the owner's box grown by its range compacted into a candidate list; (T)
the listed triangles against the candidates, `GRID_SUBS` lanes per triangle (box
test, hit test, vote, force). The throttle update: a hit → 0, no candidate → +1.

What the GPU does differently from the specification's exact order (§8.3):

* forces from the contact pass go to the same per-node integer atomics as
  self-collision, in 1/32 N fixed point; a candidate's "current force" is its force
  slot, its beams' incidence entries, its accumulator (and slide-node slot) at the
  time of the read, so contributions are rounded to 1/32 N;
* owners are processed strictly in vehicle order (exact), but an owner's triangles
  are tested in parallel: a candidate hit by two triangles of the same owner in one
  step reads its force with none, or part, of the other's contribution (the
  self-collision precedent); candidates of one triangle in parallel (order changes
  only rounding);
* the partner scan runs over partners in parallel (the resets commute: exact).

Costs and limits: a fleet where no padded frame boxes overlap pays, every other
frame at 120 fps, the bounds pass (a read of every node) and the grouping
workgroup, and every frame the frame bounds checks and an empty indirect dispatch;
the wide passes are not used while contact is active. A group's members are stepped
one after another by one workgroup, so a group of k vehicles takes about k times a
vehicle's frame (plus its contact passes). Limits, each failing loudly (the status
above) rather than silently: a group of more than 64 vehicles runs alone that frame
(no contact); more overlapping pairs than 16 per vehicle (at least 1024); a grouping
that has not converged in 64 rounds. A vehicle whose frame box is not finite (a node
at infinity or NaN) partners no one on the GPU, a deviation from the
specification's rule (which makes it a partner of every vehicle and would group the
whole fleet into one workgroup); the CPU follows the rule.

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
`set_velocity(vehicle, node, v)`, `set_vehicle_contact(on)`, `contact_active()`,
`set_contact_margin(m)`, `contact_status()`, `contact_throttles()`; WGSL via
`build.rs` like `soft_skin` (`EXPLICIT_BEAMS_WGSL`, `EXPLICIT_BEAMS_VV_PASSES`,
`EXPLICIT_BEAMS_CONTACT_*`).

`engine_softbody::explicit_contact`: `ExplicitBeamsWorld::new(vehicles)`, `step()`,
`frame(steps)`, `frame_60()`, `contact` (on), `throttle` (on), statistics `last`,
`total`, `first_hit_step`.

## Tests

From this document's rules: the approximations' defining properties, compression
yield and hardening, the halfway force and breaking above strength, the break
exception, detacher groups, tension weakening, supports and ropes, limited-spring
bounds, the contact force stopping a node against a box without bounce, tyre
geometry (2n nodes on circles of radius R around each axle node, 8–9 beams per ray),
spawn order; and GPU against CPU on the procedural sedan (`car_truck_bare` with
`minimass` 10): rest within millimetres, a 56 km/h wall crash with matching broken
and bent counts.
