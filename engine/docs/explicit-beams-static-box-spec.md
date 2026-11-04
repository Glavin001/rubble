# Explicit beams: node vs static box (addendum)

This addendum pins down what Rigs of Rods (RoR) does, per step, between a vehicle and
a static collision box, as the reference harness registers it, and answers one
question: why RoR stops the E36 at a 0.2 m pole that sits between the car's node
lines, where Rubble lets it through.

It extends `explicit-beams.md` ("One step", "1. Nodes", "Contact force", the
surfaces table). Terms mean what they mean there: h = 0.0005 s, the node pass, the
contact force, the "concrete" (box) surface, rel / origin / world position.

**Short answer.** RoR's box contact is the per-node rule that `explicit-beams.md`
already states, with nothing else involved. There are no beam, triangle, cab or
swept tests against the box, no previous position and no per-node collision memory.
RoR stops the car at the 0.233 m pole in the test scene because, in that scene, the
car reaches the pole yawed and rolled, so that some heavy node *is* inside the
pole's lane. The car is airborne because it spawns with its tyre nodes 18 cm below
the ground. In a scene where the car starts on the ground, or with a thinner pole,
RoR also slices past the pole (evidence in §3). The Rubble/RoR difference therefore
comes from the car's state *before* the impact, not from the box rule.

## 1. How the harness registers `box=`

`ror_ref … free … box=cx,cy,cz,hx,hy,hz` (harness `src/ror_ref.cpp`, scene `free`)
makes one call to RoR's collision-box registration (`Collisions::addCollisionBox`)
before spawn. The box is:

* not rotating (no self-rotation), with zero global rotation, so it is an
  **unrefined** (axis-aligned) box: lo = c − e, hi = c + e, with e the half extents
  (scale 1);
* not virtual: it produces contact forces, not only events;
* without an event name, so it raises no events and has no script callback;
* with no forced camera and no reverb (graphics and audio only);
* registered in RoR's static spatial hash (see §2.1). No collision triangles are
  generated for it, and no collision mesh is added. The harness adds no terrain
  mesh: the ground is the flat terrain height.

Nothing else in the harness's step touches static geometry. The step is: node pass
(`Actor::CalcNodes`), wings, fuselage drag, hydros, commands, beams, intra-vehicle
cab collisions (`Actor::CalcCabCollisions`), slide nodes. Cab collisions test the
vehicle's own contacters against its own collision triangles. Collision cabs never
meet static boxes or static triangles. Beams never collide with anything.

## 2. The rule, per step (RoR-exact)

In the node pass (`ActorForcesEuler.cpp: Actor::CalcNodes`), for each node i in
order, if the node does not carry the no-ground-contact flag:

1. Let x be the node's **stored world position**. This is the value written by the
   previous step's integration, origin + rel, rounded once in f32 at that time. It
   is not recomputed. After a frame's origin rebase, RoR keeps using the stored value
   from before the rebase until the node next integrates. In exact arithmetic it
   equals new origin + new rel. To match RoR bit for bit, keep a per-node stored
   world position and use it; recomputing origin + rel may differ by one rounding.
   This applies to the ground test too.
2. Ground (`Collisions::groundCollision`): as in `explicit-beams.md`.
3. Static elements (`Collisions::nodeCollision`):
   1. Cell: (cx, cz) = (trunc(x.x / 2), trunc(x.z / 2)), where trunc rounds toward
      zero and the cell size is 2 m. If x.y is greater than the hashed bucket's
      height (the largest box top, hi.y, among the elements hashed into that bucket),
      stop. This cull never removes a node that is strictly inside a box.
   2. For each element registered in that cell, in registration order (boxes and
      static triangles interleaved, as registered): skip disabled elements. For a
      box: if x is **strictly** inside on all three axes (lo < x < hi componentwise),
      pick the face with the six distances, in this order, replacing only when
      strictly smaller:
      * x.z − lo.z with normal (0, 0, −1);
      * hi.z − x.z with normal (0, 0, +1);
      * x.x − lo.x with normal (−1, 0, 0);
      * hi.x − x.x with normal (+1, 0, 0);
      * x.y − lo.y with normal (0, −1, 0);
      * hi.y − x.y with normal (0, +1, 0).

      Then add the contact force with that normal, penetration p = 0 and the concrete
      surface. The force reads the node's force F as it stands, including the ground
      contact and earlier boxes added in this pass. Set the node's
      last-collision surface to concrete (bookkeeping only).
   3. Static triangles: none in the harness, so out of scope here.
4. The node's has-ground-contact flag = ground hit or any box or triangle hit
   (bookkeeping; it does not feed back into this pass's forces). RoR then adds
   (stored world position − position before the tests) to rel. That difference is
   exactly zero, because the tests do not move the node.
5. Integrate (v ← v + F/m·h; rel ← rel + v·h; stored world position ← origin + rel),
   then restart the force, as in `explicit-beams.md`.

**What the contact force does here.** With p = 0 on concrete (solid level 0), the
fluid branch is off and the solid branch always runs. Let vn = v·n and fn = F·n.
Then R = −fn − (0.8·vn)·m/h when vn < 0 (the penetration term 0.2·p/h is zero),
else R = −fn. If R > 0, the normal part of the force after the contact is
fn + R = −0.8·vn·m/h. After integration, the node's normal speed becomes
vn + (−0.8·vn) = 0.2·vn. The node keeps 20 % of its inward speed per step and gets
no position correction, ever. The reaction has no upper bound. Whatever force
pushes the node into the face, it is cancelled exactly (the "steady" part −fn).
Friction then acts tangentially with Gr = R·strength·friction.

**Consequences.**

* **Entry depth.** A node entering with inward speed u reaches about
  u·h·(1 + 0.2 + 0.04 + …) ≈ 1.25·u·h past the face, as long as it keeps the same
  face. At 16.67 m/s that is ≈ 10.4 mm (8.3 mm on the first step).
* **Capture.** A node keeps its entry face only while its depth past that face is
  smaller than its distance to every other face, with the tie order above. A node
  whose lateral margin inside the box is greater than ≈ 1.25·u·h is held on the
  entry face indefinitely. It cannot get deeper, because the steady reaction cancels
  any push. One heavy node held this way anchors the body: the rest of the car wraps
  around the pole, beams bend and break, and the car stops.
* **Slip-out.** A node whose lateral margin is smaller than its entry depth switches
  to the side face. Its normal is then lateral, the reaction is only the lateral part
  of its force (zero when that part points outward), and the node slides on through.
* **No sweep.** A node that is never strictly inside at the instant of a node pass
  is never touched. A beam spanning the pole is never touched.

### 2.1 RoR spatial-hash quirks (RoR-exact, not reproduced by Rubble)

* A box is registered in the cells from trunc(lo/2) to trunc(hi/2) on x and z, with
  both ends clamped to [0, 32767]. A node at x ≤ −2 or z ≤ −2 looks up a negative
  cell, where no box is registered, so in RoR **boxes never act on nodes at
  x ≤ −2 m or z ≤ −2 m**, and likewise beyond 65 536 m. The cell id is
  (cx << 16) + cz as an unsigned int.
* A bucket can hold other cells' elements. They are filtered by cell id, so for
  boxes the result is the same as testing every box in registration order.

The harness scenes use positive coordinates (x ≈ 108, z ≈ 100), so these quirks do
not show up there. Rubble deliberately tests every box everywhere; matching RoR's
negative-coordinate blind spot would be an opt-in quirk, and it is not wanted.

## 3. Evidence: what stops the car at 0.233

All runs use the black-box harness binary, `ror_ref E36Sedan.truck free
pos=120,Y,100 vel=-16.67,0,0 box=108.19,2,Z,hx,2,hz roll=1 drag=0`. Speed is the
mass-weighted centre-of-mass x speed over the last 0.1 s of 1.5 s. The no-box
baseline is −13.93 m/s with the coordinator's spawn (Y = −0.059531).

| spawn Y | pole centre Z (half-width hz) | RoR CoM speed at 1.5 s |
|---|---|---|
| −0.059531 | 100 (0.1) | −0.48 (stopped) |
| −0.059531 | 100.1, 100.2, 100.233, 100.3, 99.767, 100.6, 100.7 (0.1) | between −0.85 and +0.75 (all stopped) |
| −0.059531 | 100.2189 to 100.233 in seven steps (0.1) | all stopped |
| −0.059531 | 100.233 (**0.01** or 0.03) | **−13.93** (passes, untouched) |
| −0.059531 | 100.308 (0.01), on node 84's lane at impact | −13.04 (passes) |
| −0.059531 | 100.233 (0.1), but only 0.7 m tall (hy 0.35) | −13.59 (passes) |
| **0.1225** (lowest node on the ground) | 50 (no contact) | −14.88 |
| 0.1225 | 100 (0.1) | +0.44 (stopped) |
| 0.1225 | **100.233** (0.1) | **−11.81 (passes)** |
| 0.1225 | 99.767 (0.1) | −14.41 (passes) |
| 0.1225 | 100.16 (0.1) | −14.88 (passes, untouched) |
| 0.1225 | 100.3 (0.1) | −0.28 (stopped) |
| 0.1225 | 100.381 (0.1) / (0.01) | +0.05 stopped / −12.00 passes |

Trace of the coordinator's run (Y = −0.059531, Z = 100.233, every step; per node:
the first step strictly inside and the face picked):

* The spawn puts the tyre nodes at y = −0.18. The ground throws the car up. Its
  centre of mass climbs from 0.42 m to 1.34 m by t = 0.45 s, and the lowest node is
  still 0.43 m up at the impact (t ≈ 0.6 s). In flight the car yaws (nose node 103
  drifts to z − 100 = −0.076 by step 1200) and rolls (upper hood node 191 drifts
  outward from 0.3169 to 0.337, lower frame node 84 drifts inward from 0.381 to
  0.309).
* The first node inside is **node 84** (17 kg, rest lane +0.381, lower front frame)
  at step 1217, on the +x face. Its lateral margin is 24.8 mm to the +z face, larger
  than its entry depth of 7.5 to 9.5 mm. It stays on +x for all 224 steps it is
  inside. Next is **node 81** (24 kg, the same lane, further back) at step 1296, on
  +x for 121 steps. These two hold the car. Node 191, the hood node expected to be in
  the lane, is outside the pole at impact (0.337 > 0.333).
* With the grounded spawn (Y = 0.1225, Z = 100.233), the only early node inside is
  hood node 191 (1.5 kg). It is captured on +x (margin 23 mm) for 65 steps, then the
  hood deforms and it slips to −z. The light node cannot stop the car, which goes
  through at 11.8 m/s.

So RoR does not prevent slicing past a box narrower than the node spacing. The car
stops only when a node, heavy enough and well connected, enters the box with a
lateral margin above ≈ 1.25·u·h. Which node does so depends on the body's exact pose
at impact. For this scene that pose is set by the launch from the sunken spawn.

## 4. Statements in `explicit-beams.md`: right, wrong, incomplete

The face rule, the strict-inside test, the tie order, p = 0, the concrete surface,
"each contact sees the node's force including contacts added before it", and the
contact-force formula all match RoR. None of them needs to change to reproduce RoR's
pole behaviour.

| statement | status | replacement |
|---|---|---|
| "with the world position x = origin + rel" | incomplete (bit level) | "with x the node's stored world position: origin + rel as rounded at its last integration, not recomputed after a frame's origin rebase" |
| "for every static box with x strictly inside …" | correct for Rubble; RoR additionally ignores boxes for nodes at x ≤ −2 or z ≤ −2 (§2.1) | add: "(RoR's spatial hash misses nodes at x or z ≤ −2 m; Rubble does not reproduce this)" |
| implicit: static boxes interact only with nodes | correct, but should be explicit | add: "no beam, collision-triangle or swept test against static boxes; no previous position; no per-node contact memory" |

No statement asserts a mechanism that RoR has and Rubble lacks. **Do not add a
swept test, a previous-position face choice or edge/beam-vs-box contact in the
RoR-compatible path.** Any of them would make Rubble stop where RoR passes (the
grounded 0.233, thin-pole and −0.233 rows above).

## 5. Rubble-only shapes

The oriented box, cylinder, capsule and sphere keep their current rule: strict
inside test, nearest face or surface from the stored world position, p = 0, the
concrete surface, after the axis-aligned boxes. This is Rubble's generalisation of
RoR's box rule and is consistent with it: the oriented box with the identity
rotation is the axis-aligned box bit for bit. No "previous position" or face memory
applies to them, because RoR has none to generalise. The capture and slip-out
analysis of §2 carries over: a node is held on the surface it entered through when
its distance to every other surface stays above its entry depth (≈ 1.25·u·h). For
round shapes there is only one surface, so a node is always pushed radially, and a
node entering a thin cylinder off-centre is deflected sideways rather than held.

If a Rubble profile ever wants thin static obstacles to stop vehicles regardless of
node spacing, that is a separate, non-RoR feature, for example a swept test using
the node's position before integration, or contact between beams and shapes. It
would need its own flag and the user's decision, and it must stay off in the
RoR-parity profile.

## 6. State and ordering

* Per node: the stored world position (§2 step 1). Nothing else: no previous
  position, no contact memory. The has-ground-contact and last-surface flags are
  outputs only.
* Ordering inside the node pass, per node: ground, then boxes in registration order,
  then (Rubble) shapes in order, then integration, then the force restart. All
  contacts read the force left by the previous step's beam and self-collision passes
  plus the contacts already added for this node.

## 7. For the Rubble/RoR comparison at 0.233

To find the real divergence, compare the pre-impact state, not the contact. Reference
values from RoR (Y = −0.059531, no box; z shown as z − 100):

| step | node 103 (nose) z | node 84 z | node 191 z | node 189 z | centre of mass y |
|---|---|---|---|---|---|
| 0 | 0.0000 | 0.3810 | 0.3169 | −0.3169 | 0.422 |
| 400 | −0.0281 | 0.3570 | 0.3217 | −0.3124 | 1.079 |
| 800 | −0.0534 | 0.3322 | 0.3286 | −0.3036 | 1.336 |
| 1200 | −0.0757 | 0.3091 | 0.3369 | −0.2919 | 1.198 |

If Rubble's car does not launch, yaw and roll the same way, node 84 stays near
0.38 at impact and misses the 0.133 to 0.333 lane, and Rubble passes, as RoR does
from a grounded spawn. For a pole test that does not depend on attitude, spawn on
the ground (Y = 0.1225) and/or centre the pole on a heavy node lane.

## 8. Black-box check of Rubble against RoR (2026-11-04)

Rubble's CPU solver (`ror_compare free`, which now also prints each engine's mean
node position and takes `out=` for its end positions) against `ror_ref`, 1.5 s, the
E36. No rule change was needed: Rubble already stops where RoR stops and passes where
RoR passes.

Front pole (0.2 m square, x = 108.19), 60 km/h along −x, mean node x at 1.5 s
(about 96.4 with no pole; ≈ 107–110 means stopped):

| pole z − 100 | spawn y −0.059531: RoR / Rubble / node RMS | spawn y 0.1225: RoR / Rubble / node RMS |
|---|---|---|
| 0 | 110.04 / 110.11 / 0.40 m | 110.09 / stopped / 0.40 m |
| 0.1 | 106.84 / stopped / 0.23 m | 96.44 (through) / 0.66 m |
| 0.16 | 107.37 / 107.41 / 0.14 m | 96.44 (through) / 0.05 m |
| 0.233 | 109.33 / 109.44 / 0.26 m | 98.97 / 97.18 (both through, slowed) / 2.15 m |
| −0.233 | 109.09 / stopped / 0.27 m | 96.76 (through) / 2.62 m |
| 0.3 | 109.84 / stopped / 0.39 m | 109.23 / stopped / 0.18 m |

The 2 m RMS rows are glancing hits whose outcome is chaotic; the ensembles below put
them inside RoR's own spread.

24-run ensembles at z − 100 = 0.233 (spawn speed +0.1 mm/s × k), the
`ACCURACY-RULE.md` statistics with K = 24:

| spawn | M_RR | M_XR | ratio (95 % CI) | broken (median in RoR range) | bent | mean node x, RoR / Rubble |
|---|---|---|---|---|---|---|
| y −0.059531 | 244 mm | 251 mm | 1.03 (0.93–1.17) PASS | 45 in [29, 68] PASS | 338 vs [347, 445] FAIL | 109.18–109.49 / 109.18–109.53 |
| y 0.1225 | 1620 mm | 1532 mm | 0.95 (0.75–1.09) PASS | 29.5 in [10, 71] PASS | 115.5 in [90, 354] PASS | 96.61–99.50 / 96.60–100.13 |

The one failure, the launched spawn's bent count (Rubble 338, RoR at least 347), is
a deformation-count difference and not a pass-through.

Side pole (0.2 m square at z = 101.6, beside the +z side), 32 km/h sideways
(vel = 0, 0, 8.89), grounded spawn; mean node z at 1.5 s (105.87 with no pole):

| pole x | RoR | Rubble | node RMS |
|---|---|---|---|
| 118.7 | 102.30 | 102.30 | 0.05 m |
| 119.2 | 101.16 | 101.49 | 2.09 m |
| 119.5 | 101.31 | 101.33 | 0.11 m |
| 119.8 | 100.47 | 100.46 | 0.07 m |
| 120.0 | 101.66 | 101.65 | 0.04 m |
| 120.2 | 105.87 (through, untouched) | 105.87 | 0.008 m |
| 120.6 | 101.44 | 101.02 | 0.43 m |
| 121.0 | 102.21 | 102.22 | 0.07 m |

At x = 120.2 the pole lies between the door's node lines (x 119.73 and 120.38), and
RoR passes it straight through the side, as Rubble does.

A four-node plate (`engine/crates/engine-softbody/tests/data/thin_box_plate.truck`,
node lines 0.5 m apart, 10 m/s along −x, pole half-width 0.1) through `ror_ref` and
Rubble: pole between the lines, both untouched; pole on a line (margin 0.1 m or
5 mm), both held and turned about the pole; margin 0.2 mm (below the entry depth),
both slide through. Node RMS 0.000–0.002 mm in every case.
`engine/crates/engine-softbody/tests/explicit_thin_box.rs` checks these from the
rule's definition and, with `RUBBLE_INSPIRATION`, runs `ror_ref` on the E36 front
(0, 0.16, 0.233) and side (x 120.0, 120.2) poles.
