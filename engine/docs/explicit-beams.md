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
(axis-aligned, lo and hi), static shapes (Rubble only, below), ground height (flat),
collision range r (from the file, default 0.02 m).

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
   * then for every static shape, in order (Rubble only, see *Static shapes*): if x
     is strictly inside the shape's culling box and strictly inside the shape, add
     the contact force with the shape's normal at x, penetration 0, the box surface.
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

### Static shapes (Rubble only)

RoR's reference (and the RoR harness the accuracy runs compare against) has only flat
ground and axis-aligned boxes; these shapes are Rubble's, for ramps, angled walls,
poles and bollards. Contact stays node-only, as RoR's and BeamNG's static collision
is (BeamNG's collision triangles collide with nodes only, not with static world
objects). Shapes (`StaticShape`, in `ExplicitBeams::shapes`) are tested after the
axis-aligned boxes, in order, with the box rule's terms: a node strictly inside is
pushed out along the outward normal of the nearest surface.

* **Oriented box** (centre c, half extents e, unit rotation q; built by
  `StaticShape::oriented_box`, which stores q's matrix columns a₀, a₁, a₂, the box's
  local axes in world). With d = x − c, the local point l = (a₀·d, a₁·d, a₂·d) (each
  dot (x + y) + z). Strictly inside: −eᵢ < lᵢ < eᵢ on every axis. The face is the
  axis-aligned rule's in local terms (distances l_z + e_z for −a₂, then e_z − l_z for
  +a₂, l_x + e_x for −a₀, e_x − l_x for +a₀, l_y + e_y for −a₁, e_y − l_y for +a₁,
  replaced only when strictly nearer); the normal is that ±aᵢ. A box whose rotation
  is exactly the identity quaternion is the axis-aligned box lo = c − e, hi = c + e
  (the same test, bit for bit; the GPU host gives it an axis-aligned slot).
* **Vertical cylinder** (centre c, radius R, half height H): d = x − c,
  ρ = √(d.x·d.x + d.z·d.z). Strictly inside: ρ < R and −H < d.y < H. The side's
  normal (d.x/ρ, 0, d.z/ρ) (+x when ρ = 0) at distance R − ρ, replaced only when
  strictly nearer by the top cap (+y, H − d.y), then the bottom cap (−y, d.y + H).
* **Capsule** (ends a, b, radius R): t = clamp((x − a)·(b − a) / |b − a|², 0, 1)
  (0 when a = b), p = a + (b − a)·t component by component; the round rule on
  d = x − p.
* **Sphere** (centre c, radius R): the round rule on d = x − c.
* Round rule: ℓ = √(d·d); strictly inside when ℓ < R, normal d/ℓ component by
  component (+y when ℓ = 0).
* Culling box (`StaticShape::bounds`, the same on CPU and GPU): the shape's world
  box grown by 0.01 m + 10⁻⁵·(the largest coordinate of its corners), far beyond
  the tests' rounding, so the cull never changes a result.

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
  (1/32 N); a triangle's throttle is finished in the next node pass.
* Static shapes: a vehicle's box slots are its boxes then its shapes, two records
  each: (lo, kind) and (hi, parameter record); kind 0 is an axis-aligned box, else
  lo, hi are the shape's culling box and the parameter records (after every slot)
  hold an oriented box's centre, axes and half extents, a cylinder's centre, radius
  and half height, a capsule's ends and radius, a sphere's centre and radius. A
  vehicle with shapes (vehicle flag 64) steps its nodes in `nodeUpdateShapes`
  behind an early return at the top of `nodeUpdate`/`nodeUpdateSolo`, so their code
  (and the frozen kernels' results for every other vehicle) is unchanged; the shape
  tests use the strict arithmetic in the CPU's order.
* Self-collision in the specification's order. The lanes find the hits in parallel
  ((triangle, contacter) pairs spread over the lanes, the grid or the grid skip
  below) and record each in the vehicle's hit list: its key, triangle (file order)
  << 16 | contacter (node order), the triangle's nodes, the contact's weights,
  normal, penetration, relative velocity and reduced mass, and (unless beams of
  this step are pending, below) its force as it is when no earlier hit touches the
  contacter. Then `selfApply`, after a barrier: in rounds, every hit whose earlier
  hits (smaller keys) touching its contacter node have their force takes the
  shares of exactly those hits, in fixed point, as the accumulator would hold them
  had the hits been added one by one in key order; with none it keeps the force
  the finding lane computed, else it computes it again; then it adds its own
  shares (integer sums: any order; the accumulators hold nothing else in this
  phase). A round is one barrier (the flag of the last round with open hits only
  grows within a step, so every lane reads the same decision); a step usually
  needs one (a chain of hits on one node needs one each; past 32 rounds one lane
  finishes in key order). So the result is that of the sequential pass whatever
  order the lanes found the hits in, and runs are deterministic. The hit force is
  computed in strict arithmetic in the CPU's order. A step with more than 256 hits
  (`HIT_CAP`) runs the whole self-collision of that vehicle and step on one lane
  in the specification's order instead (exact, slow; never seen in the test cars,
  at most 37 hits in an E36 wall crash step). The wide passes find hits in
  `explicit_beams_wide_tris` and apply them in `explicit_beams_wide_hits`, a
  workgroup per vehicle.
* The break exception reads the beams in spawn order. A beam whose force passes its
  strength with an end on a collision triangle is deferred in the beam phase
  (marked pending, nothing else written); after the beam phase's barrier one lane
  re-runs the vehicle's pending beams in spawn order, each counting the live plain
  beams at its ends as the sequential pass sees them: broken before the step, or
  by an earlier beam this step (a beam that breaks carries a mark until its next
  beam phase), or by an earlier master's group break this step, is dead; a later
  beam's break this step is not yet. The counts no longer depend on lane timing.
* Detacher groups: a breaking master records its beam index into its group slot with
  an atomic minimum; after a barrier every live beam of a broken group breaks, and
  beams after the master's index drop this step's force (earlier ones keep it).
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

### Lower step latency (feat/explicit-gpu-parallel)

Each of these keeps the results bit for bit (see *Bit-identity tests* below):

* **Lanes.** Fleets of up to 32 vehicles (about one workgroup per GPU core, where a
  vehicle's step latency sets the frame) run 512 lanes per vehicle; larger fleets
  keep 256 (512 for vehicles over 2048 beams). 1024 lanes was slower on an M3 Max
  (wider barriers, a longer serial grid scan) and is not shipped.
* **Grid skip.** A step with few (triangle, contacter) pairs (at most 16 per lane)
  skips the grid's counting sort (its scan costs five barriers) and tests every
  candidate, keeping exactly the hits the grid walk finds. It was switched off
  while hits were applied in lane order: it resolved the same-contacter race the
  other way (a wave's lanes read before any adds, so two triangles hitting one
  contacter both cancelled its whole normal force) and the E36 drive ensemble
  against RoR failed (348 mm against a 136 mm limit, ratio 3.85). With the hits in
  the specification's order the hit set alone decides the result: on again, and
  the bit-identity tests below run with it on against the baseline's grid.
* **Early pairs.** For a vehicle without a grid, the beam phase lists the (triangle, contacter)
  pairs that pass those tests and tryContact's box test, on the lanes the beams
  leave idle (lanes in reverse); the triangle phase runs tryContact on the listed
  pairs only. More than 1024 pairs: the usual walk.
* **Slide nodes in parallel.** A lane per slide node computes its spring and
  registers its three contributions with their slots (a count and a list whose
  capacity, from the rails its slide nodes can reach, is every contribution the
  slot can get); the next node phase (or the frame's end) adds a slot's
  contributions in spawn order, as the one-lane pass did. Contact groups keep the
  one-lane pass (the contact pass reads the slots).
* **Wing forces per node.** A lane per wing node adds its entries in the order the
  one-lane wingApply reached it.
* **Small contact groups side by side.** A group of up to four vehicles splits the
  workgroup's lanes into a power-of-two partition per member; partition m runs
  member m's step exactly as the member-by-member path, with its tested list and
  grid counts and offsets in the per-vehicle global regions the wide passes use.
  Its contact pass runs every owner's scan, throttle step and candidates, and
  triangle tests together, then applies the hits in levels: an owner joins the
  level after every earlier owner with hits whose vehicles (itself and its
  partners) it may touch, so each owner's hits read exactly the earlier owners'
  forces. The throttle resets need no order (an owner with a fast partner resets
  itself; a later owner's reset of an earlier owner is applied after that owner's
  update). Larger groups keep the member-by-member path.

Measured and not adopted: one barrier instead of two per `sync()` (Apple:
about 0); the hit test in the beam phase as well (Viper slower: it hits every step,
so the geometry ran twice); 1024 lanes.

### Latency and throughput experiments (feat/explicit-latency)

Every candidate below kept the state bit for bit (the frozen reference
`explicit_beams_ref.slang` in the bitwise tests) and none was adopted: each one was
slower or no faster. Before/after timings ran both kernels through the shipped
pipelines (`RUBBLE_EB_SRC` / `RUBBLE_EB_REF` builds in their own target directory;
the variant pipelines of `set_kernel_variant` run 2–4 µs per step slower, whatever
the kernel). M3 Max, 120 fps, `speed=27.78`, `drag=0`, N=1 GPU µs per step (slope of
17-step against 1-step frames), the reference first.

* Where a step goes (ablation, N=1, drive). Agora S 12.7 µs, Viper 22.0 µs, E36
  37.8 µs, Mustang 23.5 µs. Skipping the triangle phase saves 7, 8 and 16 µs on Agora
  S, Viper and E36. Mustang (no self-collision hits) is mostly nodes (8.0 µs, of
  which 2.9 µs is the second node of the 15 lanes past 512) and beams (9.4 µs, about
  1.8 µs per beam a lane runs). Four extra barriers per step cost under 0.3 µs.
  Plain instead of strict arithmetic (results differ; timing only) saves 1.9, 1.6 and
  0.4 µs on Agora S, E36 and Mustang.
* Hits per step (GPU count): Agora S 1.9 in 1 round, Viper 1.8 in 1 round (6.1 in
  1.6 rounds in a wall crash), E36 6.0 in 2.0 rounds at rest and driving (16.5 in a
  crash), Mustang and Audi 0 driving.
* Hit records and the hit and pending counters in workgroup memory (the one-vehicle
  frame): Agora S −0.2 µs, Viper −0.2, E36 +0.9, Mustang +0.6.
* Round 1 of the ordered hits sharing the step-end barrier: E36 +0.6, Mustang +2.1
  (its count and list loads on a step without hits; guarded, still no gain).
* Both, plus the self-collision accumulators in workgroup memory for the frame:
  Agora S −0.4, Viper +0.2 (+1.7 crash), E36 +2.1 (+3.0 crash), Mustang +0.7.
* Node positions and velocities resident in workgroup memory for the frame (the
  plan's A4; written through by the node phase, read by beams, triangles, wings,
  slides; 13.8 KB for up to 576 nodes): Mustang +1.0 µs at N=1, and at large N far
  worse, `mixed` Viper 2048: 16.0 → 24.8 ms per frame (+55 %).
* Smaller workgroup lists (TRI_LIST 512, PAIR_LIST 256 or 512, a fallback past them):
  N=1 within ±0.3 µs, Viper `mixed` 2048 +2 %.
* 128 lanes per vehicle for fleets over 32: Viper `mixed` +13 %, E36 the same.
* Removing code that never runs for the car (the one-lane overflow pass, the grid
  walks) moves a step by ±0.5 µs: the size of the Metal compiler's code layout noise.

So on Apple's GPU, with its on-chip memory shared between registers, threadgroup
memory and cache, moving storage data into workgroup memory does not shorten the
chains that set a step, and costs occupancy at large fleets.

### Bit-identity tests

`slang/kernels/explicit_beams_baseline.slang` is the kernel before this work,
frozen (a test oracle, read from the build's WGSL directory at run time, never
shipped). `tests/explicit_beams_bitwise.rs` runs it and the current kernel from one
saved GPU state each frame (`ExplicitBeamsGpu::snapshot`/`restore`,
`set_kernel_variant`). The baseline applies hits in lane order; its `EB_DET_HITS`
build applies the hits its lanes found on one lane, sorted by (triangle, node): the
specification's order. The current kernel compiled with `EB_DET_HITS` (tests only)
differs from the shipped one only in the hit force's arithmetic (the baseline's
expressions instead of strict). Per frame: the current kernel run twice leaves the
same state (deterministic); its `EB_DET_HITS` build leaves exactly the baseline
`EB_DET_HITS` state (nodes, beams, incidence forces, accumulators, group slots,
throttles, per-vehicle records) on every frame where the baseline's race probe
(`EB_RACE_PROBE`: bit 0 a contact read raced with an add, bit 2 a break-exception
count raced with a break or a group break) finds no break-exception race; those
frames may differ (the baseline reads whatever the other lanes wrote). The grid
skip is on in the current kernel and off in the baseline, so this also checks that
the skip changes nothing. `cpu_reference_per_frame` runs the shipped kernel and the
baseline one frame at a time from the CPU reference's state against the CPU. Real
cars come from `RUBBLE_INSPIRATION` when set (Viper, E36, Mustang, Agora S); the
repository's sedan always. `explicit-wgsl-check.mjs` compiles every shipped module
in Chrome.

### Self-collision order (feat/explicit-selfcol-order)

Measured on an M3 Max, 120 fps (17 steps a frame), `speed=27.78`, `drag=0`,
single-car ensembles of 24 nudged runs per engine (`explicit_beams_scaling
nudges=24`, 2.5 s; pairs 3.5 s) against RoR. `max_load=8 max_gpu=10` throughout: a
macOS background service holds the load average at 4–6; every run started under
both limits.

Why the grid skip failed (E36 drive, M_XR against a limit of 1.5 × M_RR = 135.7 mm):

| Rubble | M_XR (mm) | ratio [95 %] |
|---|---|---|
| CPU reference (`engine=cpu`, the specification's order) | 85.7 | 0.95 [0.73, 1.23] |
| GPU, hits in lane order, grid always | 96.5 | 1.07 [0.79, 1.45] |
| GPU, hits in lane order, grid skip | 348.3 (fail) | 3.85 [2.34, 5.67] |
| GPU `EB_DET_HITS` (one lane, sorted), grid always | 80.8 | 0.89 [0.68, 1.20] |
| GPU `EB_DET_HITS`, grid skip | 80.8 (the same runs, bit for bit) | 0.89 [0.68, 1.20] |
| GPU, ordered hits (this kernel), grid skip | 91.3 | 1.01 [0.74, 1.33] |

The race probe (baseline kernel) on the E36: a contact race in 300 of 300 frames
idle, drive and impact; a break-exception race in 0 frames idle and drive, 4 of 300
in the wall crash. So the same-contacter race drives it: in the skip's brute-force
walk the lanes of a wave read a contacter's force before any of them adds, so two
triangles hitting one contacter both cancel its whole normal force.

Per frame from the CPU's state (`cpu_reference_per_frame`, 40 frames, node RMS to
the CPU after one frame): E36 at rest median 0.213 mm (max 0.396) before, 0.001 mm
(max 0.006) now; E36 into a wall 0.381 mm (max 3.92) before, 0.001 mm (max 0.0014)
now; Viper into a wall max 0.68 mm before, 0.0006 mm now. Over whole runs the
kernel is deterministic (two worlds in lockstep stay bit-identical, `diff=1`).

Accuracy ensembles before (lane-order hits, grid always) and after (ordered hits,
grid skip): every one of the 21 single-car and 6 two-car cases passes both times;
RoR's runs are shared (`ror_reuse=1`). M_XR with its ratio to M_RR and the
bootstrap interval; broken and bent are Rubble medians against RoR's range.

| car | scenario | M_RR (mm) | before M_XR (ratio) | after M_XR (ratio) | broken before/after/RoR | bent before/after/RoR | verdict before | verdict after |
|---|---|---|---|---|---|---|---|---|
| e36 | idle | 4.76 | 3.83 (0.80 [0.66, 0.94]) | 4.03 (0.85 [0.69, 0.99]) | 0.0 / 0.0 / 0.0–0.0 | 13.0 / 13.5 / 12.0–15.0 | PASS | PASS |
| e36 | drive | 90.44 | 77.67 (0.86 [0.68, 1.07]) | 91.34 (1.01 [0.74, 1.33]) | 0.0 / 0.0 / 0.0–0.0 | 11.0 / 11.0 / 10.0–13.0 | PASS | PASS |
| e36 | impact | 554.07 | 663.46 (1.20 [1.01, 1.35]) | 546.82 (0.99 [0.87, 1.06]) | 90.5 / 63.0 / 54.0–121.0 | 532.5 / 542.0 / 511.0–557.0 | PASS | PASS |
| viper | idle | 4.06 | 3.47 (0.85 [0.56, 1.29]) | 2.80 (0.69 [0.47, 1.00]) | 0.0 / 0.0 / 0.0–0.0 | 0.0 / 0.0 / 0.0–0.0 | PASS | PASS |
| viper | drive | 16.81 | 16.43 (0.98 [0.77, 1.09]) | 16.36 (0.97 [0.78, 1.09]) | 0.0 / 0.0 / 0.0–0.0 | 0.0 / 0.0 / 0.0–0.0 | PASS | PASS |
| viper | impact | 366.19 | 379.66 (1.04 [0.90, 1.29]) | 371.95 (1.02 [0.88, 1.24]) | 0.0 / 0.0 / 0.0–0.0 | 134.5 / 134.5 / 129.0–138.0 | PASS | PASS |
| mustang | idle | 0.76 | 0.77 (1.02 [0.72, 1.23]) | 0.77 (1.02 [0.72, 1.23]) | 0.0 / 0.0 / 0.0–0.0 | 0.0 / 0.0 / 0.0–0.0 | PASS | PASS |
| mustang | drive | 16.94 | 16.65 (0.98 [0.81, 1.12]) | 16.65 (0.98 [0.81, 1.12]) | 0.0 / 0.0 / 0.0–0.0 | 0.0 / 0.0 / 0.0–0.0 | PASS | PASS |
| mustang | impact | 290.72 | 285.08 (0.98 [0.85, 1.13]) | 282.83 (0.97 [0.82, 1.21]) | 0.0 / 0.0 / 0.0–0.0 | 805.5 / 805.5 / 794.0–815.0 | PASS | PASS |
| audi | idle | 2.58 | 2.75 (1.06 [0.80, 1.49]) | 2.75 (1.06 [0.80, 1.49]) | 0.0 / 0.0 / 0.0–0.0 | 0.0 / 0.0 / 0.0–0.0 | PASS | PASS |
| audi | drive | 237.58 | 265.84 (1.12 [0.80, 1.90]) | 265.84 (1.12 [0.80, 1.90]) | 0.0 / 0.0 / 0.0–0.0 | 0.0 / 0.0 / 0.0–0.0 | PASS | PASS |
| audi | impact | 638.54 | 545.47 (0.85 [0.61, 1.03]) | 557.68 (0.87 [0.62, 1.06]) | 167.0 / 167.5 / 157.0–178.0 | 445.0 / 444.5 / 423.0–458.0 | PASS | PASS |
| agora_s | idle | 1.25 | 1.51 (1.20 [0.91, 1.45]) | 1.46 (1.16 [0.92, 1.51]) | 0.0 / 0.0 / 0.0–0.0 | 0.0 / 0.0 / 0.0–0.0 | PASS | PASS |
| agora_s | drive | 10.43 | 11.01 (1.06 [0.93, 1.26]) | 10.81 (1.04 [0.91, 1.23]) | 0.0 / 0.0 / 0.0–0.0 | 0.0 / 0.0 / 0.0–0.0 | PASS | PASS |
| agora_s | impact | 602.00 | 595.99 (0.99 [0.83, 1.94]) | 565.92 (0.94 [0.60, 1.47]) | 19.0 / 19.0 / 19.0–21.0 | 50.0 / 51.0 / 48.0–52.0 | PASS | PASS |
| agora_l | idle | 6.98 | 7.02 (1.01 [0.85, 1.19]) | 6.59 (0.94 [0.81, 1.11]) | 0.0 / 0.0 / 0.0–0.0 | 0.0 / 0.0 / 0.0–0.0 | PASS | PASS |
| agora_l | drive | 11.09 | 11.31 (1.02 [0.92, 1.15]) | 10.41 (0.94 [0.86, 1.06]) | 0.0 / 0.0 / 0.0–0.0 | 0.0 / 0.0 / 0.0–0.0 | PASS | PASS |
| agora_l | impact | 465.43 | 455.47 (0.98 [0.88, 1.19]) | 459.24 (0.99 [0.89, 1.21]) | 30.0 / 30.5 / 28.0–31.0 | 67.0 / 67.0 / 65.0–70.0 | PASS | PASS |
| sedan | idle | 0.20 | 0.26 (0.38 [0.31, 0.48]) | 0.26 (0.38 [0.31, 0.48]) | 0.0 / 0.0 / 0.0–0.0 | 0.0 / 0.0 / 0.0–0.0 | PASS | PASS |
| sedan | drive | 1.69 | 1.66 (0.98 [0.78, 1.20]) | 1.66 (0.98 [0.78, 1.20]) | 0.0 / 0.0 / 0.0–0.0 | 11.0 / 11.0 / 10.0–11.0 | PASS | PASS |
| sedan | impact | 259.68 | 254.25 (0.98 [0.91, 1.05]) | 254.25 (0.98 [0.91, 1.05]) | 344.0 / 344.0 / 344.0–345.0 | 543.5 / 543.5 / 530.0–556.0 | PASS | PASS |
| viper | headon | 1236.76 | 1056.06 (0.85 [0.65, 1.46]) | 1065.43 (0.86 [0.69, 1.51]) | 0.0 / 0.0 / 0.0–0.0 | 339.0 / 304.0 / 294.0–330.0 | PASS | PASS |
| viper | tbone | 702.60 | 653.19 (0.93 [0.76, 1.05]) | 644.23 (0.92 [0.81, 1.05]) | 2.0 / 2.0 / 0.0–2.0 | 79.0 / 69.0 / 62.0–104.0 | PASS | PASS |
| e36 | headon | 745.29 | 685.23 (0.92 [0.81, 1.04]) | 731.82 (0.98 [0.84, 1.12]) | 264.0 / 278.0 / 232.0–356.0 | 1911.0 / 1859.0 / 1730.0–1942.0 | PASS | PASS |
| e36 | tbone | 534.49 | 438.49 (0.82 [0.72, 1.11]) | 452.56 (0.85 [0.69, 1.16]) | 24.0 / 24.0 / 24.0–42.0 | 1464.0 / 1458.0 / 1422.0–1532.0 | PASS | PASS |
| mustang | headon | 211.74 | 213.96 (1.01 [0.95, 1.10]) | 216.88 (1.02 [0.96, 1.11]) | 0.0 / 0.0 / 0.0–0.0 | 2317.0 / 2315.0 / 2198.0–2396.0 | PASS | PASS |
| mustang | tbone | 234.89 | 246.28 (1.05 [0.85, 1.40]) | 233.29 (0.99 [0.83, 1.31]) | 0.0 / 0.0 / 0.0–0.0 | 1152.0 / 1145.0 / 1100.0–1202.0 | PASS | PASS |

N=1 GPU ms per frame (3 trials, mean [95 %]), before → after: E36 idle 0.710 →
0.689, drive 0.698 → 0.688, wall 0.754 → 0.999; Viper 0.533 → 0.426, 0.531 →
0.424, 0.581 → 0.512; Mustang 0.436 → 0.455, 0.436 → 0.452, 0.516 → 0.588; Audi
0.425 → 0.436, 0.420 → 0.432, 0.723 → 0.629; Agora S 0.224 → 0.282, 0.217 →
0.285, 0.228 → 0.279; Agora L 0.252 → 0.311, 0.238 → 0.315, 0.255 → 0.306; sedan
(wide passes) 0.252 → 0.311, 0.258 → 0.307, 0.258 → 0.323. The grid skip gains on
grid vehicles; a step with hits pays one or two barriers and the force chain after
them (about 3 µs on a small vehicle), and a crash step with many dependent hits a
few rounds. RoR (thread pool) in the same session: E36 0.49 / 0.51 / 0.63, Viper
0.25 / 0.25 / 0.31, Mustang 0.36 / 0.36 / 0.40, Audi 0.37 / 0.37 / 0.46, Agora S
0.15 / 0.16 / 0.18, Agora L 0.21 / 0.22 / 0.25, sedan 0.27 / 0.26 / 0.27.

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
4. The contact groups, one workgroup each, run in the same dispatch as the solo
   vehicles (workgroups past the vehicles), so a crash pair and the other vehicles
   step side by side: the plan writes two indirect argument sets, one of them
   empty, and a frame with no group runs the module of step 3, a frame with groups
   the same frame entry compiled also with `EB_VV_GROUPS`. A group's workgroup: the members sorted
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
next node pass), with the previous owner's throttle updates and its hits' forces;
(P) the contact throttle resets, the owner's throttle step (only when its partners
offer points), listing its tested triangles whose box grown by the range meets the
partners' boxes' union (the others have no candidate: +1 at once), and the
partners' contact points inside the owner's box grown by its range compacted into
a candidate list; (T) the listed triangles against the candidates, all lanes
spread over them (box test, hit test), each hit appended to a list whose forces
(the side vote over the node's neighbour far ends, kept as a flat list at upload,
and the contact force) are applied one lane per hit in the next phase (S of the
next owner, or after the last). The throttle update: a hit → 0, no candidate → +1.
A tested triangle's two marks for it (had a candidate, hit) are one plain word each
in `gNext` (`vvMarkAt`, after the tested list), set by the triangle's lanes in (T)
and read and cleared by the update. They were the `.z` and `.w` of the triangle's
throttle record in `gMisc`; a store to one component of a `uint4` can compile to a
read-modify-write of the whole record (Metal), so a lane marking only a candidate
could undo another lane's hit mark: a rare nondeterministic throttle (the sedan
head-on in `explicit_beams_bitwise`, about 1 run in 10). With one word per mark every
lane stores the same value, so the marks no longer race; the frozen baseline and
reference kernels keep the record marks (and the race).

What the GPU does differently from the specification's exact order (§8.3):

* forces from the contact pass go to the same per-node integer atomics as
  self-collision, in 1/32 N fixed point; a candidate's "current force" is its force
  slot, its beams' incidence entries, its accumulator (and slide-node slot) at the
  time of the read, so contributions are rounded to 1/32 N;
* owners are processed strictly in vehicle order (exact); an owner's triangles are
  tested in parallel and its hits listed, then applied one lane per contact point,
  each point's hits in triangle order (an owner's hits only read their contact
  point, and only hits on the same point add to it), so a point hit by two
  triangles reads the earlier one's force, as specified. A list fuller than twice
  the group's contact points applies the rest while testing (lane order: the old
  race, kept for that overflow);
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
`GROUND` and `BOX`, `ExplicitBeams` with public node, beam, triangle, box and shape
state (`StaticShape::{oriented_box, cylinder, capsule, sphere}`, `shape_face`),
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
spawn order; static shapes (`engine-softbody/tests/explicit_shapes.rs`: a node on a
20° oriented ramp holds with friction 1.0 and slides with 0.2, a frictionless node
driven into a cylinder exits radially, faces and caps picked nearest first, an
identity-rotation box equal to the axis-aligned box bit for bit;
`engine-webgpu/tests/explicit_shapes_gpu.rs`: the Viper at 100 km/h up a 10° ramp
into a pole within float noise of the fixed-point CPU per frame, and the identity
box equal to the plain box on the GPU bit for bit); and GPU against CPU on the procedural sedan (`car_truck_bare` with
`minimass` 10): rest within millimetres, a 56 km/h wall crash with matching broken
and bent counts.
