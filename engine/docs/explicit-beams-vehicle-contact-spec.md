# Explicit beams: vehicle–vehicle contact

This is a behavioural specification for contact between two or more vehicles in
Rubble's explicit node–beam solver: a node of one vehicle meeting a collision
triangle of another.

It extends `explicit-beams.md` (the solver, the step, ground and box contact,
self-collision and the contact force). Terms not defined here mean what they mean
there. That includes:

* the step h = 0.0005 s and the frame;
* the node pass, the beam pass and the self-collision pass;
* collision triangles, their nodes n₀, n₁, n₂ in file order, and their throttle
  (*wait*, *idle*);
* the contacter, tyre, no-ground-contact and on-a-collision-triangle flags;
* the collision range r;
* the **contact force** of `explicit-beams.md` ("Contact force") and the
  "concrete" surface.

It also uses the f32 notation of `explicit-beams-aero-commands-spec.md`.

The reference is Rigs of Rods (RoR), judged as a black box. The reference harness
(`inspiration/ror-harness`, `scale` mode) runs RoR's own inter-vehicle contact code
on several vehicles, in RoR's order, one vehicle at a time (see §8.3). The goal is
agreement to float tolerance in trajectories. Behaviours that are artefacts of the
reference rather than physics are marked **(quirk)**; the RoR-compatible profile
reproduces them.

Notation:

* All arithmetic is IEEE single precision (f32). An expression `a + b·c` is
  evaluated left to right as written, each product rounded before the sum, no fused
  multiply–add.
* `min(a, b)` means "b if b < a, else a".
* World position of a node: the vehicle's origin + the node's relative position,
  component-wise f32 sum, as the node pass recomputes it after integration (an
  immovable node keeps the world position it had at spawn).
* sign(x) = +1 if x ≥ 0 (including −0), else −1 (a NaN gives −1).

---

## 0. Summary

* After **every** vehicle has finished the step's node pass, beam pass,
  self-collision pass and slide-node forces, a new **vehicle-contact pass** runs, one
  vehicle at a time in vehicle order. Its forces are added to the nodes' force
  accumulators and are consumed by the next step's node pass (ground and box contact
  first, then integration), like the beam forces.
* Vehicle A's collision triangles are tested against the **contact points** of every
  other vehicle B whose padded bounding box overlaps A's. Both directions happen,
  because B's own turn tests B's triangles against A's points.
* The contact points of a vehicle are its contacters **and** its *contactable* nodes.
  These are the nodes of its collision triangles, or every node if it has no collision
  triangles, minus nodes with ground contact off (§2.2). This is a wider set than
  self-collision uses.
* The point-in-triangle test and the force are the self-collision ones, except for two
  rules. Which side of the triangle the node is on is decided by a vote of the node
  and its beam neighbours, not by the sign of its distance alone. The force uses the
  triangle owner's collision range and contact surface.
* Each triangle has a second throttle, separate from its self-collision one, with
  different rules (§3.3). When two overlapping vehicles' first nodes differ in speed
  by more than 4 m/s, every throttle of both vehicles is reset each step (§3.4).

---

## 1. Scope and terms

**Vehicles.** A world holds vehicles V₀, V₁, … in **vehicle order** (spawn order).
Every vehicle is stepped with the same h and the same number of steps per frame;
each keeps its own origin and does its own per-frame origin rebase, as now.

**Owner and partner.** In vehicle A's turn, A is the *owner*. Its collision triangles
are the surfaces. Every other vehicle B that qualifies (§4.1) is a *partner*. Its
contact points are the candidates. A vehicle is never its own partner; self-contact
stays in the self-collision pass.

**Both directions.** For every unordered pair {A, B} whose boxes overlap, A's
triangles are tested against B's points in A's turn, and B's triangles against A's
points in B's turn. The two are independent tests with independent state. A node of
A that is also a vertex of one of A's triangles can take part in both: as a triangle
vertex in A's turn and as a candidate point in B's turn.

**Not in scope:** vehicles joined by hooks, ties or ropes ("linked" vehicles; see
§2.2 for the rule if they are ever added), networked or remote vehicles, and paused
vehicles (a vehicle that is not stepped neither tests nor is tested, §4.1).

---

## 2. Data fixed at spawn

### 2.1 Collision triangles

These are the ones self-collision already uses. A `cab` line is a collision triangle
when its options contain any of `c`, `p`, `u` (contact; 10× tougher;
invulnerable) or `D`, `F`, `S` (the buoyant variants of the same three). The
triangle's nodes are the line's three nodes in order: n₀, n₁, n₂.

* **(quirk)** A line whose options contain one letter from each group (for example
  `cD`) is entered **twice**: two collision triangles with the same nodes, each with
  its own throttles, both tested, both applying force. The list order is file order,
  with such a line's two entries adjacent.
* At most 3000 collision triangles per vehicle. A file with more is an error
  (`from_truck` fails).
* Triangles are never removed. A triangle whose beams have all broken still collides.

### 2.2 Contact points

Each node gets one more flag, **contactable**, computed once after spawn (after all
nodes, including tyre nodes, exist):

1. Mark *on a collision triangle* for the three nodes of every collision triangle
   (already done for self-collision and for the break exception).
2. For each node in order:
   * if it is a **contacter**, it is a contact point (not marked contactable);
   * else it is **contactable** if ground contact is not off for it (node option
     `c`) **and** at least one of these holds:
     * it is on a collision triangle;
     * it is a rim node (only `wheels2` and `flexbodywheels` make rim nodes; Rubble
       refuses those sections, so this case does not arise yet);
     * the vehicle has **no** collision triangles at all.

The vehicle's **contact points** are its contacters plus its contactable nodes, in
node order.

Which nodes are contacters (unchanged from `explicit-beams.md`):

* every node listed in the file's `contacters` section;
* every tyre node of `wheels`, `meshwheels` and `meshwheels2` (in RoR every tyre
  node of every wheel kind is a contacter).

A file **without** a `contacters` section has no contacters except its tyre nodes.
There is **no** "every node is a contacter" default in RoR. The loader's desc marks
every colliding node a contacter when the section is absent; do not use that. The
wide default comes from the contactable rule instead. A vehicle with no collision
triangles offers every node except `c` nodes, cinecam nodes included (they are `c`,
so excluded).

Consequences worth testing:

* A contacter is a contact point even when it has ground contact off.
* Tyre nodes **are** contact points for other vehicles. Self-collision excludes them;
  vehicle contact does not.
* Body nodes that are not on a collision triangle and not listed as contacters are
  **not** contact points when the vehicle has triangles. They pass through other
  vehicles' triangles.
* A vehicle with no collision triangles can push its nodes into other vehicles but
  presents no surface. The other vehicles' nodes pass through it.

*Linked vehicles (not in Rubble today):* if vehicles are ever joined, a partner that
is linked (directly or indirectly) to the owner offers only its contacters, not its
contactable nodes.

### 2.3 Collision range r

r is per vehicle. It is the value of the vehicle's `set_collision_range` line (the
last one, if several). A negative value means the default. With no line r = 0.02 m.
In A's turn the **owner's** r is used throughout: the candidate box margin, the
distance band and the penetration. The partner's r plays no part.

### 2.4 Contact surface

The surface passed to the contact force is the **owner's** submesh surface. That is
"concrete" (adhesion speed 3.0, static μs 1.2, sliding μc 0.75, hydrodynamic t2 0.01,
Stribeck speed 6.0, exponent 2.0, strength 1.0, solid ground level 0), the surface
self-collision already uses. A file may name another surface with
`submesh_groundmodel`; RoR then uses that named surface, falling back to concrete when
the name is unknown. Rubble's loader currently ignores that line. Until named
surfaces exist, a file whose `submesh_groundmodel` names anything other than
`concrete` must be refused with a clear error rather than silently run on concrete.
The same applies to self-collision.

The friction coefficient in the contact force is the **candidate node's** own
friction coefficient. That is the node of the partner vehicle; tyre nodes have their
wheel's value.

### 2.5 Neighbour lists

For the side vote (§5.3), each node of each vehicle has a **neighbour list**, built
once at spawn:

* go through the vehicle's beam table in beam order (every beam of every kind and
  type: body, tyre, shock, hydro, command, trigger, animator, cinecam);
* append the far end to both ends' lists.

It is never rebuilt:

* a beam that breaks later stays in the lists;
* two beams joining the same pair of nodes put that neighbour in twice;
* the list length is the node's beam count at spawn.

### 2.6 State

Per collision triangle, a second throttle, the **contact throttle** (*cwait*,
*cidle*), both 0 at spawn. It is independent of the self-collision throttle (*wait*,
*idle*). Neither is reset when a vehicle is reset, only when it is spawned.

---

## 3. When it runs

### 3.1 Position in the step

One step of the world:

1. For each vehicle in vehicle order, its whole existing step: node pass (ground and
   box contact, integration, force restart), wings and fuselage drag, commands,
   beam pass, self-collision pass, slide-node forces. Vehicles do not read each
   other's data in this phase, so they may run in parallel.
2. **Bounding boxes** of every vehicle (§4.1), from the world positions after this
   step's integration.
3. **Vehicle-contact pass**: for each vehicle A in vehicle order:
   a. A's **partner scan** (§4.1), which may reset throttles (§3.4);
   b. if A has at least one partner offering at least one contact point, A's
      **triangle loop** (§3.3, §5, §6). Otherwise A's triangle loop does not run
      at all, and A's contact throttles do not advance (they keep their values).

Nothing moves during passes 2–3. Positions and velocities are the post-integration
values of step 1 for every vehicle. Only force accumulators change. The forces
added in pass 3 are part of each node's force when the next step's node pass runs:

* the ground and box contact forces there read the node's force, inter-vehicle
  contributions included;
* integration then uses it.

The node force read by the contact force in pass 3 (its "current force") is the
node's accumulator at that moment. That is gravity, drag and turbulence, beam forces,
self-collision and slide-node forces of this step, plus any vehicle-contact forces
already added to that node earlier in pass 3. See §8.3 for why the order matters.

Every step runs the full pass. There is no per-frame part apart from the existing
origin rebase. Bounding boxes and partner sets are recomputed every step.

### 3.2 First step

At spawn every node's force is zero, as now. Vehicles spawned overlapping are not
separated; the contact force acts from the first step.

### 3.3 The contact throttle

In A's triangle loop, for each collision triangle T of A, in list order (§2.1):

1. If cwait > 0: cidle ← cidle + 1; cwait ← cwait − 1; **skip T** this step.
2. Otherwise (cwait ≤ 0): cwait ← min(cidle, 12); cidle ← 0. This step is
   **unconditional**. Self-collision does it only when idle > 0.
3. Find T's candidates (§4.2) among all partners' contact points.
   * If there are **none**: cwait ← cwait + 1.
   * If there are some: test each one (§5). Every candidate that **hits** sets
     cwait ← 0. If candidates exist but none hits, cwait stays at the value step 2
     gave it. There is no increment.

Resulting rhythm (a test, figures from spawn, no resets):

* a triangle with no candidates is tested at steps 1, 3, 6, 10, 15, … and settles
  at once every 14 steps (13 skipped);
* a triangle with candidates but no hit keeps its current interval;
* a triangle with a hit is tested every step from then on until it has no candidates.

cwait is never negative. There is **no** −20000 "stay alert" value, unlike
self-collision.

### 3.4 Fast-approach reset **(quirk)**

During A's partner scan (§4.1), for every partner B, in vehicle order, compute
Δ = v_A,0 − v_B,0. Here v_X,0 is the velocity of vehicle X's node 0 (its first node
in node order). Compute q = Δx·Δx + Δy·Δy + Δz·Δz, summed left to right. If q > 16
(strictly, so |Δ| > 4 m/s), then for **every** collision triangle of A **and every**
collision triangle of B:

* cwait ← 0 (the contact throttle);
* wait ← 0 (the **self-collision** throttle).

idle and cidle are not touched.

Effects that must be reproduced:

* The rule is symmetric in A and B. It fires in A's scan and again in B's scan of the
  same step. For the pair's earlier vehicle (in vehicle order), the reset from the
  later vehicle's scan comes after the earlier vehicle's triangle loop. It overwrites
  what that loop left in cwait, so the next step starts from cwait = 0 even if the
  condition no longer holds then.
* While the condition holds, every triangle of both vehicles is tested every step,
  because the reset comes before each triangle loop.
* The self-collision throttle reset takes effect at the next step's self-collision
  pass (the self-collision pass of this step has already run). It also **erases** the
  −20000 value a recent self-hit left. After the episode such a triangle goes back to
  the normal growing rhythm instead of staying alert. A fast pass by another vehicle
  can therefore change self-collision results without any vehicle–vehicle contact.
* Rubble's GPU self-collision finishes its throttle update in the next node pass. The
  reset must land after that update for the step that produced it and before the
  next self-collision test.

---

## 4. Broad phase

Only the candidate set matters. Any faster structure (a grid, a BVH, a sort along an
axis, a GPU pair list) is acceptable if it finds exactly the same set for every
triangle in every step it is tested.

### 4.1 Bounding boxes and partners

Each vehicle's **box** is the axis-aligned box of the world positions of **all** its
nodes after this step's integration, every node included (`c` nodes, cinecam nodes,
tyre nodes, nodes of parts that have broken off). Each face is then moved outward by
0.05 m (min − 0.05, max + 0.05 on each axis).

Two boxes **overlap** unless, on some axis, one's max < the other's min. The test is
inclusive: touching boxes overlap.

A's partner scan considers every other vehicle B in vehicle order. B is a partner if
B is stepped this step and the two boxes overlap. For each partner the scan applies
§3.4. A's triangle loop runs only if the partners together offer at least one contact
point (§2.2).

A detached part far from its vehicle stretches that vehicle's box. That changes only
which vehicles become partners, and so only the throttles and run time, not the
narrow-phase result for a given candidate.

### 4.2 Candidates of a triangle

For triangle T of A, with p₀, p₁, p₂ the world positions of n₀, n₁, n₂:

* lo = component-wise min(p₀, p₁, p₂) − r; hi = component-wise max(p₀, p₁, p₂) + r
  (r = A's range; the min/max are taken first, then r subtracted or added);
* the candidates are **all** contact points of **all** of A's partners whose world
  position x satisfies lo ≤ x ≤ hi on all three axes (inclusive).

A point is a candidate of T at most once. A node that is a contacter and on a
triangle is still one point. The candidate set does not depend on the order in which
partners or points are visited.

---

## 5. Narrow phase

For each triangle T of A tested this step, compute its local frame once, from the
positions at the time of the test:

* u = p₁ − p₀, w = p₂ − p₀;
* n = normalize(u × w): length ℓ = √(nx² + ny² + nz²) (exact square root); if
  ℓ > 1e-8, multiply each component by 1/ℓ, otherwise leave n unchanged (possibly
  zero);
* M = the inverse of the 3×3 matrix whose columns are (u, w, n), computed exactly as
  self-collision does. If its determinant's magnitude ≤ 1e-6, M is the zero matrix.

### 5.1 Local coordinates

For a candidate h with world position x:

* (α, β, δ) = M·(x − p₀); row i of the product is
  M[i][0]·dx + M[i][1]·dy + M[i][2]·dz;
* γ = (1 − α) − β.

α weights n₁, β weights n₂, γ weights n₀. δ is the signed distance from the plane
through p₀, positive on the side n points to.

### 5.2 Hit test

h **hits** T when α ≥ 0, β ≥ 0, γ ≥ 0 and |δ| ≤ r. This is the self-collision test,
with A's r.

* There is no exclusion of tyre nodes and no exclusion of T's own vertices. They
  belong to another vehicle, so they cannot coincide.
* A degenerate T (zero matrix) gives α = β = δ = 0 and γ = 1, so every candidate
  "hits". n is then zero or near zero, so the force is zero (R = 0, §6). The hit
  still sets cwait ← 0 **(quirk)**.

### 5.3 Side of contact (the vote)

On a hit, decide whether h is in front of T (on n's side) or behind:

1. score = 3·sign(δ).
2. If h's neighbour list (§2.5, in h's own vehicle) has **more than 3** entries: for
   each entry j, in list order and duplicates included, with y_j the current world
   position of node j, add sign(n·(y_j − p₀)). The dot product is
   nx·ex + ny·ey + nz·ez with e = y_j − p₀.
3. If score < 0, h is **behind**: replace n by −n and δ by −δ. Otherwise (score ≥ 0,
   ties included) keep them.

This differs from self-collision, which flips exactly when δ < 0. Here a node that
has crossed to the back of the plane, but whose neighbours are mostly in front, is
still pushed to the front. A node with ≤ 3 neighbour entries is decided by sign(δ)
alone, so δ = 0 counts as front.

### 5.4 Penetration

p = r − δ, with δ as possibly flipped in §5.3. p lies in [0, 2r] for an unflipped
hit. It can exceed r when the vote overrules the sign of δ (δ is then negative). It is
never negative, because |δ| ≤ r.

### 5.5 Several triangles, several points

Every hit is resolved: all candidates of a triangle, all triangles, all owners. There
is no "nearest only" or "first only" rule. A node inside the bands of two triangles of
the same owner gets two forces. A node near triangles of two vehicles gets a force in
each owner's turn.

---

## 6. Response

For each hit (h of partner B, triangle T of owner A, normal n and penetration p after
§5.3–5.4):

* **Relative velocity**: v = v_h − (v₁·α + v₂·β + v₀·γ). Here v₀, v₁, v₂ are the
  velocities of n₀, n₁, n₂. The bracket is the sum of the three scaled vectors, left
  to right, each component separately.
* **Triangle mass**: m_t = m₁·α + m₂·β + m₀·γ (left to right).
* **Effective mass**: m = (m_h·m_t) / (m_h + m_t). This is the reduced mass, the same
  as self-collision. A remote vehicle would use m_h instead; not applicable.
* **Force**: f = the contact force of `explicit-beams.md` with:
  * the node's current force F = h's force accumulator now;
  * velocity v and mass m;
  * normal n and penetration p;
  * the owner's surface (§2.4), with friction = h's friction coefficient;
  * step h = 0.0005 s.

  For reference, that force is:
  * vn = v·n, fn = F·n;
  * R = −fn;
  * if vn < 0: R ← R − ((0.8·vn + (0.2·(−p))/h)·m)/h, evaluated in that order
    (N = kg·(m/s)/s; the −p is the surface level 0 minus p);
  * if R ≤ 0, f = 0;
  * otherwise the static or sliding friction branch, with the slip force F − fn·n and
    the slip velocity v − vn·n.

  Units: forces in N, masses in kg, velocities in m/s, p in m, h in s.
* **Apply**: h's force += f. The triangle nodes' forces: n₁ −= f·α, n₂ −= f·β,
  n₀ −= f·γ (each f·weight rounded, then subtracted).

Only h's force is read by the contact force. The triangle nodes' forces are only
written. The steady part R = −fn cancels whatever pushes h into the surface, the
impact part removes 80 % of the approach speed of the reduced mass per step, and the
0.2·p/h part pushes h back towards δ = r. Because p is measured from the far edge of
the band, an unflipped node at rest settles near δ ≈ r, one range off the surface.

No clamping beyond what the contact force already does: R ≤ 0 means no force, and the
hydrodynamic term is capped at 5. There is no maximum force, no restitution parameter
and no positional correction.

Side effects in RoR that have no physical effect in Rubble's scope (skid and particle
bookkeeping, a "mesh contact" flag read only by wheel torque code, which Rubble does
not run) need not be reproduced.

---

## 7. Differences from self-collision (do not assume they are the same)

| | self-collision | vehicle contact |
|---|---|---|
| when | inside the vehicle's step, after its beam pass | after **all** vehicles' steps, one owner at a time |
| candidate points | contacters only, minus tyre nodes, minus the triangle's own nodes | contacters **and** contactable nodes of other vehicles; tyre nodes included |
| side | flip when δ < 0 | the weighted vote of §5.3 |
| throttle state | *wait*, *idle* | separate *cwait*, *cidle* |
| throttle, before testing | wait ← min(idle, 12) only if idle > 0 | cwait ← min(cidle, 12) always |
| throttle, after testing | −20000 if anything hit, else +1 | 0 on any hit; +1 only if there were no candidates; unchanged if candidates but no hit |
| throttle reset by others | yes: §3.4 sets wait ← 0 | yes: §3.4 sets cwait ← 0 |
| runs at all | always, for a vehicle with triangles and contacters | only if some other vehicle's box overlaps and offers a point; otherwise the throttle freezes |
| range, surface | own | the owner's (the triangle's vehicle) |
| force formula, weights, reduced mass | — | identical |

---

## 8. Edge cases, precision, order

### 8.1 Broken beams and detached parts

Beams breaking changes nothing in the data of §2:

* triangles stay collision triangles;
* contact points stay contact points;
* neighbour lists keep their spawn entries.

A part that has broken off still collides with other vehicles. It stretches its
vehicle's box (§4.1). Its nodes vote with their spawn-time neighbours, even those no
longer attached.

### 8.2 Fast approach and tunnelling

Contact exists only inside the band |δ| ≤ r (4 cm thick at the default range) and only
on steps the triangle is tested. RoR tunnels in these cases, and so must Rubble:

* a point whose motion relative to the triangle exceeds about 2r per step (> 80 m/s
  at the default range) can cross the band between two tests and get no force;
* a point that enters the band after crossing its middle may still be pushed the right
  way if its neighbours vote it to the front (§5.3).

Below 4 m/s difference between the vehicles' first nodes, the throttle can skip up to
13 steps. Points then move up to about 13·h·v between tests, and contact can start late
or be missed. This is part of RoR's behaviour. In the RoR-compatible profile the
throttle is always on. A switch to test every triangle every step may exist, like the
other profile switches, but it must default to on, the RoR behaviour.

### 8.3 Order dependence (exact profile)

The contact force reads h's current force. Results therefore depend on the order of
contributions to h. Rounding aside, they depend on exactly these two orders:

* **Vehicle order.** In B's turn, a node x of A that is a candidate reads x's force
  including what A's turn added to x as a vertex of A's triangles. A's turn ran
  first. Swapping the vehicles changes the result. This is common: nodes on collision
  triangles are both vertices and contact points. Required: owners processed strictly
  in vehicle order. Every owner's contributions are in the accumulators before the
  next owner reads.
* **Triangle order within an owner.** A node hit by two triangles of the same owner
  reads, at the second, the force the first added. Required: within an owner,
  triangles in list order (§2.1) as far as any one candidate node is concerned.

These orders affect results only through float rounding:

* the order of candidates within one triangle (each candidate is a different node, and
  its force does not read the triangle's forces);
* the order of partners;
* the order of the vertex reactions summed into a triangle node.

Summing those in any order matches RoR to rounding. For the GPU, a dependence-
respecting schedule therefore needs sequential phases per owner (or per owner and
triangle where one node is hit twice), not one unordered scatter. Any relaxation, such
as all owners in one phase reading forces from before the pass or fixed-point
accumulation that rounds the read force, is a change of physics. It must be a named,
default-off setting and reported, not adopted silently.

**(quirk)** RoR proper runs the owners' turns on several threads at once. Its own
results are then racy in exactly these places: the read of h's force, and the
§3.4 resets of another vehicle's throttles. The reference is the single-threaded order
the harness uses, vehicles in spawn order. That order is the one specified here.

### 8.4 Precision

* Everything in f32. Positions compared are world positions (origin + relative,
  §Notation), which differ in rounding from relative positions. Use world positions,
  as RoR does, for the boxes, candidates, local frame and vote.
* The local frame uses an exact square root for the normal and the cofactor inverse
  for M. Use the same routines as self-collision. Do not use the fast approximations,
  which are for beams and drag only.
* The vote's dot products and the frame's δ are computed differently (dot product
  versus matrix row), so they can disagree in sign for a point within rounding of the
  plane. Reproduce both as specified, not one from the other.

### 8.5 Vehicles that never come near

If two vehicles' boxes never overlap, the pass does nothing for them. Their results
are **bit-identical** to running each vehicle alone.

If boxes overlap without contact, no forces arise. The throttles still change, and a
fast pass (§3.4) can change a vehicle's self-collision timing. Results are then
bit-identical to separate runs **unless** a self-collision hit falls on a step that
the separate run's throttle would have skipped, or vice versa.

---

## 9. Acceptance against the reference harness

Run Rubble and the harness (`scale` mode with inter-vehicle contact) on the same
`.truck` files, spawn positions and velocities. Compare node world positions and
velocities per step or per frame, broken-beam sets and plastic lengths.

1. **No contact.** Two cars driven in parallel lanes far enough apart that the padded
   boxes never overlap. Each car's trajectory is bit-identical (CPU) to a single-car
   run, and agrees with the harness as the single-car runs do now.
2. **Head-on.** Two copies of one car (a real RoR car with collision triangles, for
   example the Agora S bus or a car with `cab` lines), facing each other, 56 km/h
   each. Required:
   * contact starts on the same step in both;
   * node trajectories agree to the same tolerance as the existing single-car wall
     crash, through and after the impact;
   * broken and bent beam counts match;
   * the total linear momentum of the pair is conserved by the contact itself, apart
     from ground contact (f on h and −f·(α+β+γ) = −f on the triangle);
   * swapping the spawn order changes results in both, in the same way.
3. **T-bone.** Car A at 50 km/h into the side of a stationary car B. Same checks.
   This also exercises tyre nodes as contact points (A's tyres against B's side
   triangles) and the side vote for B's door nodes pushed past A's front triangles.
4. **Slow touch.** One car rolling into a parked one at 2 m/s (below the 4 m/s reset).
   This exercises the contact throttle's growing interval. Contact must start on the
   same step as in the harness.
5. **No-triangle vehicle.** A vehicle with no `cab` collision triangles against one
   with triangles. Every non-`c` node of the first is a contact point; the second's
   nodes pass through the first.

Unit tests, written from this document:

* the contactable rule (§2.2) on small files (with and without triangles, with `c`
  nodes, tyre nodes, a `contacters` section);
* the throttle sequences of §3.3 and the reset of §3.4, including its effect on the
  self-collision throttle;
* one node above one triangle: force and the three reactions, each with weights α,
  β, γ;
* the vote of §5.3 (a node with 3 neighbours versus 4 or more, a tie, a node behind
  the plane voted to the front);
* the order dependence of §8.3: two owners whose order swaps changes the result;
  candidate order within a triangle changes it by rounding only.
