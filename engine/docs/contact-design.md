# One contact model for thin deformable skins (proposal, 2026-11-24)

Status: **proposal, for the owner's decision.** Nothing here is the default. Why it exists:
known-bugs S4 (two generated sedans fuse after a head-on; a car's own panels pass through each
other at a pole) and the user's bar: solids never share space, for car against car, a car
against itself, and a fast rigid body (a cannonball) against a car.

## What the code does today (read from the code, 2026-11-24)

| Pair | Detection | Reach | Response | Gap |
|---|---|---|---|---|
| Rigid ↔ rigid | Swept boxes; per-pair look-ahead `pairReach` = relative speed·h + spin (`world_step_margins.slang:16`), only closing points kept (`closingPoints`, `world_step_rest_inserts.slang:359`); `BODY_CCD`: GJK time of impact, validate and clamp after the solve | Scales with relative speed | Position-level speculative row: close the gap, push past it | Vehicles are excluded (`bodyIsCcd`, `world_step.slang:153`) |
| Car node ↔ car triangle, self | Node against triangle; anti-clip side memory, swept crossing, 10 cm memory (`explicit_contact.rs`, `explicit_beams.rs` self pass) | **Fixed** band r = 25 mm; only the search box grows with speed | §37 dissipative law, depth capped at r, position step capped by the energy the hit removed | Edges cross between nodes; nodes held on the wrong side (S4) |
| Car edge ↔ car edge | `explicit_contact/edge.rs` (CPU, off) | Fixed band | Same law | Pairs at an edge's end belong to nobody |
| Car skin ↔ rigid shape | Surface contact (`explicit_surface.rs`, off by default): triangles, beam segments, nodes; casts by conservative advancement (`fq::toi`) for fast bodies | Cast reach scales with relative motion | Dissipative, depth capped at r | Only for bodies above half their thickness a step |
| Car node ↔ rigid body (coupling) | `node_contact_with` (`explicit_rigid.rs:387`) | Fixed | **Penetration 0**: nothing pushes a node out once in (known-bugs B1, still true) | Bricks and balls sink in |

Already tried and recorded:
* OGC-style displacement bounds in the AVBD soft solver (`DECISIONS.md` s29): a 100 km/h wall's
  penetrations 33 → 25, a head-on's car-to-car ones 1 → 12. Bounds on the approach do not stop
  nodes that are already behind at the step start under crash loads.
* The outward side alone (S4): no node held inside, but the panels cross between nodes and the
  contact fights (+867 / −640 kJ).
* A node-against-edge contact along the edge's bisector (S4): fought the face contacts beside it.
* A wider fixed band (S4): helps (80 mm: edges through 0) but is right for one speed only and
  leaves resting cars up to the band apart.

What best practice says:
* BeamNG: nodes against triangles only, forces from about 2.5 cm, anti-clip picks the side;
  its own documentation says fast nodes can clip through. Matching BeamNG keeps this bug.
* Offset Geometric Contact (Chen et al., SIGGRAPH 2025): every contact belongs to exactly one
  feature (a face's prism, an edge's wedge cut by planes square to its two faces, a vertex's cap
  cut by planes square to its edges), the force square to that feature, so no two contacts
  fight; a large contact radius; penetration-free by per-vertex movement bounds. Its bounds are
  absolute (the paper notes they slow fast motion), which does not suit cars at speed.
* Bridson et al. 2002, Brochu et al. 2012: vertex–face and edge–edge continuous tests are the
  complete set for two triangle meshes; anything less leaves a way through.

## The proposal: one rule for every soft pair

1. **A complete, non-fighting contact set.** For every pair of skins (two cars, one car with
   itself, a car with a rigid shape's surface): vertex against face, edge against edge, and the
   end cases routed by OGC's feature regions (a point belongs to the face, edge or vertex whose
   region holds it; the force is square to that feature). This replaces the bisector attempt and
   gives node-against-edge contact without fighting the faces beside it.
2. **The side from geometry, not memory.** A car's collision skin faces out of its solids (the
   generated sedan: 528 of 528 triangles); another car's node belongs outside, so its side is
   the outward one. Memory and the side vote stay for content without a consistent skin
   (checked at load, a clear error or the old rule, never a silent guess).
3. **Look-ahead per pair by relative closing speed**, as the rigid path's `pairReach` and
   `closingPoints`: a pair is in contact when its gap is within r + max(closing speed, 0)·h at
   the contact point. The response lets it close the gap this step and no further
   (speculative), so resting contact is unchanged (r = 25 mm) and 100 + 100 km/h gets the reach
   it needs. Replaces any fixed wide band.
4. **Recovery for what is already through.** s29's lesson: approach limits do not undo a node
   that is behind. Depth beyond the band is pushed out by a bounded position step that is not
   cancelled under load (today's §37 cap gives a loaded node none), with an explicit
   engine-wide rate and the energy it stores reported by the ledger. The same for nodes inside
   rigid bodies (fixes B1).
5. **Fast rigid bodies keep their time of impact**: surface contact's conservative-advancement
   casts (and the rigid path's `gjkCast` for shapes it lacks) for a cannonball against a skin,
   with the same feature regions and recovery.

## How it will be judged (tests first, CPU reference)

`engine-softbody-authoring/tests/gen_sedan_contact.rs`, worst moment of each crash, every one
an exact or sourced bound:
* head-on 60 + 60, **100 + 100**, **100 into 10** km/h, offset 0.8 m: no node inside the other
  car past the band, no edge through it, not fused;
* **slow push** (two cars pressed at 1 m/s): their skins end no further apart than today's band
  (no resting gap);
* pole 64 km/h: no new crossing of a car's own skin;
* **cannonball** (steel, 0.1–0.3 m, 30–300 m/s) into door and roof: never inside the hull,
  momentum within the coupling's tolerance;
* every scene: finite, mechanical energy never rises past 0.5 % of the start, the ledger closes;
* `ccd_invariants.rs`'s rigid invariants (no tunnelling, momentum, energy, a finer-tick
  reference) carried over to the soft pairs.

## Order of work

1. Tests for the new scenes (red today).
2. Feature regions for vertex–face, edge–edge and their end cases (CPU), with the outward side.
3. Speculative per-pair look-ahead (CPU); the fixed wide band removed.
4. Recovery step with its explicit rate (CPU), and the coupling's node contact (B1).
5. One shared Slang module for vehicle and self contact (today the anti-clip logic is in about
   ten kernel files), then the port, GPU == CPU bit for bit, the scenario matrix, Chrome.
6. Authoring: the generated sedan's skin crosses itself 89 times at rest and opens at its seams
   under crush; the rigger closes it (independent of the solver work, measured by the same
   tests).

## Progress (2026-11-24, CPU reference, all behind switches, off by default)

Steps 1–3 built: the tests (`gen_sedan_contact.rs`: 100 + 100, 100 into 10, offset failing and
listed; the slow push and the cannonball passing guards); node against edge in the edge's
wedge only, convex edges (`EDGE_VERTEX=1`, `explicit_contact/edge.rs`); speculative per-pair
look-ahead in the node and edge passes (`SPECULATIVE=1`: a pair whose gap would close past
the band this step gets a contact that lets it close to the band's edge and no further).

Worst moment of each crash, nodes inside the other car past the band / skin edges through it
(car A / car B):

| Scene | Today | Edge | Edge + node–edge | + look-ahead | Edge, fixed 80 mm band |
|---|---|---|---|---|---|
| 60 + 60 | 29/35, 78/88 | 8/12, 46/48 | 6/6, 33/43 | 4/5, 23/17 | 3/4, 0/0 |
| 100 into 10 | 21/26, 70/67 | 11/5, 32/48 | 10/4, 20/20 | 8/5, 8/12 | – |
| 0.8 m offset | 17/20, 58/65 | 12/7, 36/28 | 6/6, 20/21 | 8/8, 7/17 | – |
| 100 + 100 | 26/23, 96/90 | 10/11, 70/61 | 15/12, 85/114 | 12/15, 75/80 | – |

Look-ahead alone (node contact only) is worse (60 + 60: 14/33, 97/96). Not fused, slow push
touching (14–27 mm), cannonball and pole-energy guards pass in every variant.

What remains, 100 + 100 with everything on (`EDGE_TRACE=1`, probe): only 5 edge pairs first
seen through by jumping the band (look-ahead works); nodes enter through the skin's holes
(gap entries 24 / 34, all but one within 20 cm of an open edge), through faces under load
(20 / 17: step 4, the recovery step, not built), and the tyres have no collision shape (932 /
278 tyre vertices inside). Next: step 4, step 6 (close the skin), a tyre shape (known-bugs S2
item 3).

### Step 4 measured, and an energy bug in the edge pass (2026-11-24)

`pair_crashes_create_no_energy` (new, every step through the ledger; passes with the defaults)
found that the edge pass creates energy: the mechanical energy's peak over the start, head on
60 + 60 / 100 + 100 / 100 into 10 / offset (bound 0.5 % of the start: 2.1 / 5.6 / 2.9 / 2.1 kJ):

| Switches | Rise (kJ) |
|---|---|
| none (today) | 0.01 / 0.00 / 0.25 / 0.15 |
| look-ahead alone | 0.01 / 0.00 / 0.05 / 0.07 |
| edge | 7.4 / 8.0 / 3.2 / 1.9 |
| edge + node–edge | 14.2 / 7.6 / 8.9 / 3.3 |
| edge + node–edge + look-ahead | 17.5 / 7.0 / 9.1 / 2.3 |

(Earlier in S4 "energy only falls" with edge contact was read from the probe every 20 ms; the
ledger every step shows these rises.) Cause, by construction: the edge pass runs after the
node pass with its own mass splitting and both cancel the same nodes' forces from the pass's
start, and its hits skip the node pass's work guards (`guard_vehicle_pass`, `guard_final`,
the residual pass). **Fix first:** edge and node–edge hits in the node pass's hit list (a
four-node hit), one splitting, the same guards.

Push-out (`PUSH_OUT` m/s, both passes, a pair behind its surface moves out at that rate under
load) on top of edge + node–edge + look-ahead: 1 m/s 60 + 60 4/7 nodes, 33/28 edges; 100 + 100
9/8, 49/61; little change, and at 3 m/s no better. With the outward side as well it fuses the
cars again (parted 0.01 m of 4.6) and edges through rise to 133–521. Not adopted; re-measure
after the fix above.

### One hit list (2026-11-24)

Edge-against-edge and node-against-edge hits now join the owner's node hits in one list
(`Hit` with a second contacter node `h2` at weight `t`, `kind`): one sub-body splitting, the
residual pass, the work guards, staging and the ledger. Defaults unchanged bit for bit (the
head-on probe's whole log identical; the 46 contact unit tests pass).

Energy (`pair_crashes_create_no_energy`, bound 0.5 % of the start) now passes with node–edge
on: 60 + 60 rises 2.0 kJ (bound 2.1) with edge + node–edge, 1.6 kJ with look-ahead as well;
edge alone still fails the 60 + 60 (2.6 kJ). But with the splitting shared, each contact
pushes less, and the overlap the separate pass had reached was partly its excess:

| Worst moment, edge + node–edge + look-ahead | Separate pass (created energy) | One list |
|---|---|---|
| 60 + 60 nodes / edges | 4/5, 23/17 | 9/13, 77/84 |
| 100 into 10 | 8/5, 8/12 | 5/7, 56/47 |
| 0.8 m offset | 8/8, 7/17 | 8/11, 39/52 |
| 100 + 100 | 12/15, 75/80 | 22/14, 128/143 |

Not fused in every variant with node–edge; push-out (1 m/s) changes little. **What this says:**
one pass of per-hit responses, each with its share of a node's mass, cannot hold many
simultaneous contacts on the same nodes under crash loads without either overshooting
(energy) or giving way (overlap). The candidate next step is an iterative contact projection
over the turn's hits (a few Gauss–Seidel sweeps of the dissipative law, each hit seeing the
others' impulses, as rigid solvers do), which satisfies all contacts together with no positive
work. That is a change of the contact algorithm (the owner's decision).

### Judged over ensembles (2026-11-24) — supersedes the single-run tables above

The head-on is chaotic: car B moved 1 mm aside changes a single run's skin edges through the
other car from 51 to 179 (everything on). Single-run comparisons of edge counts above are
mostly noise. `tools/c2c/ensemble.sh KMH SECONDS N "NAME:VARS" ...` runs each variant N times
(car B 0..N−1 mm aside) and reports min / median / max of the worst moment (A + B):

| 60 + 60, 6 runs | nodes inside | edges through | overlap L |
|---|---|---|---|
| defaults | 53 / 71 / 74 | 170 / 183 / 194 | 152 / 191 / 227 |
| edge | 22 / 26 / 35 | 62 / 86 / 182 | 26 / 49 / 81 |
| edge + node–edge | 27 / 29 / 32 | 63 / 102 / 167 | 41 / 53 / 84 |
| + look-ahead | 27 / 28 / 32 | 48 / 66 / 179 | 29 / 48 / 57 |
| + 16 sweeps | 22 / 24 / 29 | 30 / 83 / 127 | 37 / 45 / 54 |
| + push-out 1 m/s | 27 / 29 / 34 | 38 / 98 / 166 | 29 / 54 / 58 |
| edge, 80 mm car-to-car band | 5 / 7 / 9 | 0 / 20 / 34 | 6 / 12 / 19 |

| 100 + 100, 6 runs | nodes inside | edges through | overlap L |
|---|---|---|---|
| defaults | 70 / 78 / 98 | 186 / 204 / 241 | 350 / 402 / 539 |
| edge | 41 / 49 / 52 | 199 / 232 / 310 | 256 / 304 / 318 |
| edge + node–edge + look-ahead | 44 / 48 / 51 | 231 / 248 / 277 | 257 / 288 / 332 |
| edge, 80 mm band | 18 / 22 / 29 | 40 / 167 / 248 | 176 / 213 / 248 |
| edge + look-ahead, 80 mm band | 17 / 26 / 31 | 72 / 97 / 296 | 226 / 259 / 319 |

**What holds up:** edge contact is the real gain (60 + 60: nodes 71 → 26, overlap 191 → 49 L;
100 + 100: 78 → 49, 402 → 304 L) and ends the fusing. Node–edge, look-ahead and push-out make
no difference beyond the spread; 16 sweeps a little. The 80 mm car-to-car band is the strongest
on overlap but fails the per-step energy test (60 + 60: 3.6 kJ, 100 into 10: 5.9 kJ, bound
2.1 / 2.9). Per step, edge alone fails the 60 + 60's energy narrowly (2.6 kJ); edge + node–edge
passes (2.0 kJ). 100 + 100 stays far from clean with every variant.

### Authoring: the collision skin's holes (2026-11-24)

`LOOPS=1` (probe) lists the skin's holes as loops. Over the generated sedan's front: a
0.42 × 0.45 m hole at each corner over the lamps (between the hood's side edge and the
fender's top front: the seam zipper's 0.3 m reach does not cross it; the lamp lenses are
triangulated over the body's nodes, far from the lamp), the radiator, rear seat and console
as one-sided flat faces, the doors' and trunk lid's outlines, and the underside. The skin
also has doubled layers at its seams: 91 shared edges walked the same way by both triangles
(the same defect as the 89 crossings at rest, `collision_skin_does_not_cross_itself_at_rest`).

`fill_holes` (gen2 rig, `RIG_FILL_HOLES_M`, off by default) closes loops up to a length by ear
clipping (2.5 m: 20 triangles; the left lamp corner and the flat parts close, the right lamp
corner does not, its rim shared with the doubled layers). Six runs of the 60 + 60 head-on:
node-only contact 71 → 60 nodes inside, 191 → 166 L; with edge contact 26 → 30, 49 → 58 L
(within the spread). So with edge contact the remaining overlap does not come through these
holes; kept off. The doubled seam layers remain an authoring bug to fix in `bridge_seams`.

### After the authoring fixes (2026-11-24, six runs each, 60 + 60, median nodes inside A + B / overlap)

The rigger fixes (seam bridges that do not fold or cross, holes filled, interior solids first;
`generated-vehicles-lessons.md` §18) on their own: defaults 71 → 66 nodes, 191 → 145 L. With
edge + node–edge contact: 33 nodes, 57 L (was 29, 53 L: within the spread).

* **The outward side, now that the outer skin's winding is consistent:** still worse (edges
  through 106 → 362, overlap 57 → 104 L). Rejected.
* **Push-out at full strength** (`PUSH_OUT=100`, up to 5 cm a step, through the accounted
  position step): blows up (mechanical energy 413 kJ → 22 MJ). Moving loaded nodes out by
  large steps stores the beams' energy, which returns as motion; §37 caps the step by the
  energy the hit removed for exactly this reason. Rejected; the slow rates (1–3 m/s) stay
  measured as no gain.
* **Self edge contact** (edge against edge within one car, outer skin, pairs in range at rest
  excluded; `edge_contact`): the 64 km/h pole's new crossings of the car's own skin 47 → 30,
  the front door and fender no longer through each other. Pole energy passes.
* **Edge contact takes the outer skin's edges only** (`ExplicitBeams::tri_outer`): an edge's
  outward normal averaged over interior faces (winding not checked) was wrong near them.
* The candidate default (edge + node–edge, both cars and within one): every guard passes
  (not fused, slow push touching, energy in every crash with 60 + 60 at 2.04 of 2.07 kJ,
  cannonball, pole); overlap remains (60 + 60: 11 / 12 nodes, 62 / 65 edges at the worst).

### The fix that holds: accumulated-impulse sweeps (2026-11-24)

**Trace** (`TRACE_FACE=1`, probe): the first node to enter the other car through a face (a
front apron node into the bumper cover, 60 + 60) was in contact for ten steps, pushed out at
9–16 kN every step, and still closed at 8–15 m/s: the per-hit response with each node's mass
split among its hits undershoots where many hits load a light panel (the cover's nodes carried
35–50 kN of other hits). Sweeps that only add force overshoot instead: the node separates at
2–4 m/s, the no-positive-work rule then gives it nothing while its beams drive it in at about
35,000 m/s², and it comes back at 13–17 m/s: a ratchet at the band's edge.

**Fix** (`CONTACT_GS=1`, `CONTACT_SWEEPS=n`; off by default): after the first pass, sweeps
over the owner's hits keep each hit's normal force (the first pass's plus its changes) and
change it toward the force that brings the pair's normal velocity at the step's end to zero, at
the forces now, on the nodes' own masses, clamped at zero (sequential impulses' accumulated
clamp: an excess is taken back). Gauss–Seidel applies each change at once.

| 60 + 60, median of 6 | nodes inside | overlap | energy rise, per step |
|---|---|---|---|
| node only | 66 | 145 L | 0.01 kJ |
| node only + 8 sweeps | 37 | 94 L | |
| edge + node–edge | 32 | 60 L | 2.04 kJ (bound 2.07) |
| edge + node–edge + 1 sweep | **28** | **38 L** | **0.13 kJ** |

Every guard passes with edge + node–edge + 1 sweep (not fused, slow push 23 mm, cannonball,
pole energy, every crash's energy). Self edge contact is left off (`SELF_EDGE=0`): with it the
head-on's edges through rise 99 → ~170 (self and vehicle contact are separate passes over
shared nodes); it stays a measured option for the pole case (47 → 30 new crossings).

### The GPU's form of the sweeps: Jacobi on the split masses (2026-11-24)

Gauss–Seidel is sequential: each change is applied before the next hit's is computed, which on
the GPU needs colouring or one lane walking each contact group. Two parallel forms (every
hit's change from the same sums, then all applied, the same accumulated clamp), over the same
six runs of 60 + 60 (median, min–max; edge + node–edge on):

| sweeps | nodes inside | edges through | overlap |
|---|---|---|---|
| Gauss–Seidel, 1 | 28 | 112 | 38 L |
| Gauss–Seidel, 8 | 28 | 79 | 42 L |
| Jacobi, own masses, 1 / 4 | 36 / 40 | 158 / – | 80 / 102 L |
| Jacobi, own masses, 8 | diverges (coordinates blow up) | | |
| **Jacobi, split masses, 1** (`CONTACT_SPLIT=1`) | **30** (28–35) | **80** (56–116) | **57 L** (32–68) |
| Jacobi, split masses, 4 / 8 | 31 / 29 | 100 / 100 | 50 / 45 L |
| no sweep (edge + node–edge alone) | 32 (30–38) | 99 (77–110) | 60 L (46–66) |
| no sweep, node only (today's default) | 66 (48–71) | 162 | 151 L |
| node only + split Jacobi 1 | 63 | 166 | 145 L |

Jacobi on each node's own mass overshoots where hits share nodes (each hit asks for the whole
stop) and diverges with more sweeps; on the split masses (each node's mass shared among its
hits, the first pass's split) it is the damped form, stable at every count, and converges
toward Gauss–Seidel. One sweep of it is the residual pass the GPU already runs
(`vvResidualPhase`: the same split, one pass) with one change: a hit may take back force down
to zero (the accumulated clamp) instead of only adding.

The sweep is not what removes the overlap: edge + node–edge contact does (66 → 32 nodes, 151 →
60 L); the clamp alone does nothing on node-only contact. What it buys is energy margin. The
crash tests' worst rise per step (≤ 0.5 % of the start's energy is the bound):

| 60 + 60 head on | 100 + 100 | 100 into 10 | 60 + 60, 0.8 m offset |
|---|---|---|---|
| edge + node–edge: 2032 J of 413 kJ (0.49 %) | 2702 J of 1129 kJ | 1289 J of 575 kJ | 165 J |
| + split Jacobi, 1 sweep: 1394 J (0.34 %) | 1305 J | 606 J | 14 J |

Both pass every guard (not fused: parted 9.0 / 6.0 m against the same with contact off; slow
push 25 mm; cannonball; pole energy). Chosen for the GPU: **edge + node–edge + split Jacobi, one
sweep**: it is the GPU's existing pass shape, and it halves the edge contact's energy rise.

### One setting (2026-11-24)

`ExplicitBeams::edge_contact` is now the whole fix: edge against edge, node against edge (on
with it; `EDGE_VERTEX=0` a measurement), and the accumulated clamp in the residual pass for
every hit with either vehicle's `edge_contact` (the existing split, one pass: split Jacobi, one
sweep). Self edge contact is off (`SELF_EDGE=1`, a measurement). `EDGE_CONTACT=1` alone, six runs
of 60 + 60: nodes inside 27 / **32** / 38, edges through 64 / 86 / 148, overlap 33 / **60** / 69 L;
every guard passes (not fused: 6.00 m against 6.00 m; 60 + 60 rise 1394 J of 413 kJ, bound
2065 J). Off, every step is as before (196 + 63 softbody host tests pass).

## What the GPU port carries (the candidate default)

1. **Edge hits in the owner's turn** (`explicit_contact/edge.rs`): edge against edge between
   two cars (each pair once, the lower index's turn; every live edge, its outward normal from
   the outer skin's faces, an inside face's side taken from the partner's outer edge), node
   against an outer convex edge in its wedge; as `Hit`s with a second contacter node (`h2`,
   weight `t`) and the owner's edge as `[i, j, j]`, weights `[s, 0, 1 − s]`.
2. **One splitting and the same guards** for every hit of the turn (first pass, staging, work
   guard, ledger, contact-force log), zero-weight nodes not counted for edge kinds.
3. **The accumulated clamp in the residual pass** for vehicle hits (split Jacobi, one sweep:
   the existing pass with a hit allowed to take back its first-pass force down to zero).
4. Settings and flag bits for each (vehicle flags word bits 21+), GPU == CPU bit for bit with
   bit-exact on, every backend; the scenario matrix and the Chrome lab as `AGENTS.md`'s ladder.
