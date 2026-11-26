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

## The GPU port: what it touches (read from the kernels, 2026-11-24)

Until it lands, the GPU upload refuses a vehicle with `edge_contact` (`ExplicitUpload::with_sleep`
asserts) rather than step it without. Today a GPU hit is always (gt, h): a triangle and one
contacter node, two words in the owner's list at `gP.vvHits`, its weights, normal and side
re-derived from positions wherever it is used (`vvTry`, `vvStagedHit`, `lvTryStage`).

**Live paths** (each must carry edge hits bit for bit with `bit_exact` on):
1. World: `contactGroup` (`explicit_beams_vv_levels.slang`), small groups `vvPassSmall`
   (≤ `VV_SIDE` members, module `explicit_beams_contact`), big `vvPass` (`explicit_beams_contact_big`).
2. Lab (group split on): the fleet dispatches (`FLEET_STATE`, `FRONT`, `ANY`), kept pairs
   (`fkTest` / `fkPairTry`), levels `fleetLevelsOnceTry` (`HITLAT` + `LEVELS_ONCE`) falling back
   to `fleetLevelsAny`, and `fleetGroupLevels` for an owner past `FL_SCAN_MOST` hits.
Measurement-only (`fleet_levels`, `fleetLevelsLat`, `vvwide`) and the frozen reference kernels
(`explicit_beams_ref_*`, `baseline*`) need not carry it; their scenes keep edge contact off.

**What an edge hit needs on the GPU:**
- *Static, per vehicle*: the skin edges' topology (i, j and their triangles, outer and inside
  faces apart; CSR in `gLists`, the `surf_seg` pattern: a `gStatic` record and a params word,
  `PARAM_WORDS` bumped) and a flag bit (vehicle flags bit 21).
- *Per step*: each edge's liveness and outward normal from its live triangles (the outer
  faces' when it has any), its convexity and opposite corners: one pass over the edges.
- *Detection*: per pair of members with overlapping boxes, A's edges against B's (each pair
  once, the lower index's turn) and B's contact nodes against A's outer convex edges (both
  ways, each owner's turn), the CPU's tests in `explicit_contact/edge.rs` exactly.
- *The list*: a hit kind bit in the first word (owner edge index) and the partner's edge or
  node in the second; s, t, n, gap re-derived each phase like today's hits. Order: after the
  owner's node hits, by (partner, kind, B edge, A edge) as the CPU appends them.
- *Split counts*: the owner edge's two nodes (weight > 0) in `vvSplitTri`, h and h2 (t > 0)
  in `vvSplitPoint`, and the same in the clear.
- *Snapshot*: h2 and both edge nodes (`vvSnapNode`).
- *Apply*: five-node variants of `vvTry`'s adds and of `resStage` / `resFinal` / `resMerge`
  (wired to four nodes by role today), and the residual's accumulated clamp
  (`vvResOf`: r = max(λ0 + d, 0) − λ0, λ0 the first pass's normal force) for edge-contact pairs.
- *Order of dependent hits*: an edge hit reads its contacter h's force like a node hit and also
  adds to h2, so per owner the edge hits run after every node hit (a second ordered sweep
  after `vvApplyOrderedIn`), in list order.
- *Proof*: `explicit_contact_gpu.rs` gains GPU == CPU scenes with edge contact on (the 60 + 60
  head-on, bit for bit); fast-check and the scenario matrix stay bit for bit with it off (no
  kernel word changes there); then a matrix row with it on for its cost.

## The GPU port: done (2026-11-24)

`slang/kernels/explicit_beams_vv_edges.slang`, called from the small contact groups' pass
(World: `vvPassSmall`; the lab's split path: the edge tests run in the group's levels dispatch,
before the levels, with the fleet-wide schedules off, `PERF_FLEET_FAMILY`, which are lossless
switches). Each step: every member's skin edges (outward normal from the live faces, the outer
skin's when it has any), the edges and contact points within reach of a partner's edges' box
listed, the edge-edge and node-edge tests of `explicit_contact/edge.rs`, the hits with their
split counts; then in the levels the snapshot of their nodes, the first pass, the residual
phases with the accumulated clamp (node hits of an edge-contact pair too) and the clear. An
owner with edge hits alone gets a level (`gsEdgeHas`). A group of more than four cars with an
edge-contact member reports FL_ERR_EDGE_GROUP (the big groups' pass has no edge contact yet).

**Proof.**
- GPU == CPU bit for bit with bit-exact on, both paths: `explicit_contact_gpu.rs`
  `edge_contact_steps_like_the_cpu`, `edge_contact_on_the_split_path_steps_like_the_cpu`
  (three tetrahedra: ridges crossing, a corner on a ridge; 162 steps in one- and 17-step
  frames; with edge contact off the CPU ends 0.057 m away). The 12 other contact tests pass.
- Off, nothing changes: the scenario matrix on CUDA, the build before the port against this
  one (`run.sh --cuda --base 56bb2def --head e7af10ae`, strict floats): every row bit for bit,
  none slower (−1.6 % to +0.1 %).
- The lab: `{ edge: true }` (page control "edge contact", a build option; `vehicle_lab --edge 1`).

### The GPU port's cost, and why its kernels are not shipped yet (2026-11-24)

Native, Metal (M3 Max), the lab's 4-car mountain run (`vehicle_lab --measure mountain-perf
--cars 4 --edge 1`, generated sedan), the explicit frame's GPU ms a World step (34 substeps):

| | median | p95 (the pile) |
|---|---|---|
| edge contact off | 12.9 | 14.6 |
| on: the fleet-wide schedules off alone (`PERF_FLEET_FAMILY`, edge off) | 20.8 (40 s run) | |
| on: all pairs of the listed edges | 62 (40 s run) | 136 |
| on: midpoint grid, 128 buckets | 24.5 | 117 |
| on: cells 2 × reach, 2,048 buckets, no dedupe, (edge, cell) work items | **25.1** | **79** |

Attribution (`RUBBLE_EDGE_ABLATE`, time only): the edges' step data and lists about +3 ms; the
grid build +5 ms median, +9 p95; the pair tests about +32 ms at p95. Every variant is the same
physics (the final energies agree to the joule) and GPU == CPU bit for bit, two procedural sedans
crashing included (5,678 edge hits).

Inlined into the contact modules the edge code also put 28 shipped kernels past their
compile-cost lines (`compile_cost.py --check`) and Chrome's Metal compile of
`explicit_beams_contact` failed (an internal compiler error after 180 s). So the edge kernels are
behind `EB_EDGE`: built only with `RUBBLE_EDGE_KERNELS=1` (native development, tests, the native
lab with `--edge 1`); the shipped kernels are as before the port (no edge code in their WGSL; the
compile-cost, inline-budget and duplication checks pass); a host without them refuses edge
contact with a clear error; the lab page's control is off until they ship.

**Next (to ship it and make it cheap):**
1. Edge contact as passes of its own (small modules of their own, dispatched beside the contact
   groups' levels on the split path), not inlined into the contact modules.
2. Frame-coherent pair lists: the grid and the pair tests once a frame with a travel margin (each
   pair's nodes' travel over the frame), the 34 substeps retesting only the listed pairs (the
   node contact's kept pairs, `PERF_FLEET_KEEP`, do this). Lossless; the pair tests are most of
   the cost.
3. The fleet-wide levels (`fleetLevelsOnceTry`, `fleetLevelsAny`, `fleetGroupLevels`) with
   the edge hits, so edge contact no longer turns the fleet-wide schedules off (+7 ms).
4. Then the Chrome A/B and the default decision.


## Edge pass and offsets on the GPU: measured state (2026-11-24, Modal CUDA, T4)

- Kernels, 88582403 against the head with the same `ssqrt`, fixtures' flags masked to ones
  both support: 12 rows bit for bit, none slower (pile8 -16.5 %, headon -7.2 %).
- The fleet-wide schedules with offsets (all but levels-once), on/off in one build: 12 rows
  bit for bit; pile8 -41.6 %, pile32 -83.2 %, but pile2 +14.4 % and barrier +9.4 %, so the
  gate fails. Not the missing levels-once module:
  with it on for offsets (an experiment, perf=4 on one build) 12 rows bit for bit and PASS, pile2
  -0.8 %, barrier -0.3 %, pile32 -28.3 %. It is not adopted: the module is then 22,198 statements,
  10.2 s predicted against the 10 s cap (about 220 statements to find). A car without offsets pays most of
  it too (the fleet family on/off, its own saved states): barrier +10.8 %, pile2 +5.5 %. So barrier
  is the schedule's own cost on a two-body group; of pile2's +14.4 % with offsets the rest may be
  offsets or the different saved state (not separated). A schedule chosen by group size would
  take both; the family's default predates this work.
- A portable fixture stores its planner words, perf flags included, and `apply` does not
  mask them: fixtures made by a build replay that build's schedule on any other. Earlier
  "7-11 % slower" figures of this day are void for that reason.
- Edge contact's rows (headon_e, side_e, pile8_e; `run.sh SWITCH headon_e ...`, not in the default
  gate), CUDA T4, first run ever there (its module was not in the replay's module list). ms a frame,
  without / with edge contact: headon 13.2 / 13.7, side 15.7 / 18.5, pile8 21.4 / 176. The saved
  states differ (each row ran its own way to its moment), so this is a size, not an A/B; pile8's
  factor of 8 is the pass: one workgroup a group, its owners one after another. Fleet schedules
  on/off with edges: bit for bit on the three rows. CUDA against the CPU is not tested with edges.
- Open: groups over VV_PART (8) have edge contact again (edgePass, an owner at a time; a 12-body test); compile,
  inline and duplication ratchet lines raised and to be brought back; a browser frame runs
  without edge contact until its module compiles; paint points are CPU-only (off).
- Edge turns after the node turns against one shared turn (CPU, the switch is in d38610b2 only), 12
  head-ons each, median nodes inside / edges through / overlap L: 60 km/h shared 16/60/26, after
  14/54/30, edge contact off 33/126/39; 100 km/h 24/75/113, 24/57/110, 38/154/173. The two orders
  are inside each other's spread (at 60 km/h the separate turn's worst overlap is larger, 48 against
  36 L); no energy rise; the same host tests pass and fail. Neither comes near zero overlap.
- The belly cap with the small holes filled after it (the fills ran before it and could not see
  its gaps), 12 head-ons each, edge contact on, median nodes inside / edges through / overlap L:
  60 km/h no cap 22/72/30, cap 14/54/30, cap and fill 10/58/31; 100 km/h 36/70/222, 24/57/110,
  14/61/116. Edge contact off: 60 km/h cap 33/126/39, cap and fill 20/116/33; 100 km/h 38/154/173,
  34/154/185 (inside the spread). The cap alone raised the lint's small holes 1 -> 4, with the fill
  3. Filling the 3.5 to 5 m outlines (doors, lid): no gain, 16 to 21 faces inward: not holes to
  patch, panels to author closed.
- The edge pass's time (8-car pile, CUDA, 175 ms at a 5 cm skin): hits not kept -8, then no pair
  tests -108, then no grids -9: the pair search, not the owners one after another. Skin 12 cm: 99.

### What is seen: paint cutting paint (vehicle_paint_overlap, 2026-11-24, CPU reference)

Length of the lines where one car's painted triangles cut the other's (m); min / median / max of 8
runs; control with contact off 141 / 145 / 145 (2 runs, 60 km/h), 0 at spawn. Paint only (no glass,
lamps, trim, tyres); torn triangles (3 x rest) left out.

| scene | worst, edge off | worst, edge on | at the end, edge off | at the end, edge on |
|---|---|---|---|---|
| head on 60, 1.5 s | 3.06/5.71/8.10 | 2.08/3.21/5.32 | 0/0/0 | 0/0/0 |
| head on 100, 1.5 s | 8.55/12.08/19.20 | 0.38/2.23/4.65 | 0.18/4.56/8.08 | 0/0/0 |
| head on 30, A pressing, 2.5 s | 7.11/7.27/7.50 | 0.87/0.98/1.10 | 0/0/0.75 | 0/0/0.19 |
| head on 60, A pressing, 2.5 s | 4.29/5.59/7.05 | 3.10/3.78/5.22 | 0.04/0.90/2.08 | 0/0/3.41 |

Side impact into B's middle (2 s; an earlier version was 1.4 m off centre): 30 km/h worst
1.86/2.01/2.08 off, 0/0/0 on; 60 km/h worst 1.40/1.43/2.16 off, 1.37/1.38/1.40 on, at the end
0/0.44/0.77 off, 0/0/0 on; 60 km/h pressed at the end 1.93/2.44/2.79 off, 0/0/0.53 on.
Tests: gen_sedan_paint.rs (three pass, three listed as expected failures).

Edge contact's cost on one saved state (CUDA T4, ms a frame, the same fixture with W_EDGE cleared
for "off"): pile8_e off 28.1, on 175 at a 5 cm skin, 99 at 12 cm, 89.5 with the lists' rebuild on
motion relative to the group's (147 at 5 cm); side_e off 14.3, on 18.4; headon_e off 12.3, on 13.7.
pile8_e without the pair search 50.9: the search and tests are 39 ms of the 61, the rest 23.
(The earlier "21 ms without" for the pile was another state.)

The edge pass's parts on pile8_e (89.8 ms; off 28.1), each switched off in turn (time only, the
physics differs): the lists' rebuild 32 (89.8 -> 57.9), the kept pairs' tests 10 (-> 47.7), the
motion check 0.4, the edges' per-step preparation 15 (-> 32.0), the rest 4. A box reject before a
kept pair's test gave nothing (89.6). Next: one owner's changed partners or motion rebuilds every
owner's list, and the positions at the build are kept a vehicle, not an owner (so one list cannot
be rebuilt alone yet); the preparation works out every edge each step, listed or not.
Edges near no partner (its box grown by the lists' reach) are not worked out: read per partner box
the pile got slower (89.5 -> 94), per member's joint box all three rows gain. Last run pile8_e,
side_e, headon_e: see the commit; before 89.5, 18.4, 13.7.

### Rebuilding the candidates per owner: built, lossless, slower (2026-11-26)

The idea: the group rebuilds every member's lists when any node of any member moves a skin, so
rebuild only the owners whose own pairs need it. Built two ways (commit 1ca3df761, reverted in
the next commit), both GPU == CPU bit for bit in `explicit_contact_gpu` (24 tests). CUDA T4, ms a
frame, `PROFILE_SPLIT=1 run.sh --cuda perf=0 pile8_e side_e headon_e` (generated sedan; one
container a run, so directional):

| | pile8_e | its edge pass | side_e | its edge pass | headon_e | its edge pass |
|---|---|---|---|---|---|---|
| base (b95f3f754) | 85.6 | 59.1 | 17.7 | 3.8 | 13.3 | 0.98 |
| per owner, a partner's drift bounded through its own reference | 136.8 | 109.8 | 17.8 | 4.0 | 13.4 | 1.08 |
| per owner, its partners' nodes copied at its build | 94.4 | 67.2 | 18.0 | 4.2 | 13.5 | 1.19 |

- The bound-only variant keeps no copies (an owner's bound on a partner grows by the partner's
  motion at each of its builds), so a partner's drift counts twice and owners rebuild in a
  cascade.
- The copy variant has exactly the old criterion per owner (its nodes and each partner's within
  two skins about one vector, from its own build), in the candidates' room (no layout change;
  9 % of a sedan's room for seven partners). It rebuilds about as often as the group did, pays
  the extra checks (the two-car head-on +0.2 ms), and is over the split_edges inline budget by
  about 570 lines. In a pile every car drifts, so an owner's partners trigger it as often as the
  group.

Where the rebuild's time is (pile8_e, the base with measurement-only edits, physics not kept):

| | edge pass ms a frame |
|---|---|
| base | 59.1 |
| the partner-set signature ignored | 59.8 (partner changes are not what rebuilds) |
| no rebuild on motion | 26.0 |
| a rebuild every step | 315 |
| a rebuild every step without its pair search | 45.6 |

So about one step in nine rebuilds, and a rebuild costs about 8.5 ms, 7.9 of it the pair
search at reach + 2 skins (lists and grids 0.6): one workgroup searching every (owner,
partner) pair in turn while the T4's other SMs wait. The lever is the search's cost, not how
often it runs: spread an 8-car group's (owner, partner) searches over workgroups (the hits
add in fixed point in any order, so lossless), or balance each search's lanes (an (edge,
cell) item scans its bucket serially, a lane each).

### A car through itself, as seen (2026-11-25, CPU)

A car's own paint cutting itself (m; pairs that did not as built; min/median/max of 4; the control,
standing, 60 and 120 km/h: 0): pole 64 worst 7.3/7.5/8.5, at the end 5.1/5.4/5.5; pole 40 11.4/11.7/
12.2, 5.0/5.1/5.5; head on 60 13.1/14.3/14.8, 4.8/7.5/7.8; side 60 7.5/8.5/9.3, 3.7/4.0/4.1. It is
more than car through car, which ends at 0 there. Mostly neighbouring panels: hood and fender, door
and fender, bumper and fender, body and hood.
- SELF_EDGE=1: pole at the end 5.4 -> 3.3, but head on 7.5 -> 8.3 and car through car worst 2.6 -> 6.8.
- Not the pairs excluded for starting inside the contact range: 20 of them, and with none excluded
  (SELF_EXCL=0) 5.39 and 8.01 against 5.38 and 7.50.
- Mostly geometry, not the solver: on the panels seen, the collision skin through itself at the end
  is 0.1 to 0.4 m at a pole (0.0 to 0.2 at the contact offsets) where the paint is 5.5; head on 0.4
  and 1.2 m (0.4, 1.1) against 4.2 and 6.1; side 0.1 and 0 against 1.7 and 4.0 (one run each; an
  open skin can also pass by another's edge uncut).
- The crossing painted triangles: 40 to 60 % lie over their part's collision skin (p50 11 to 17, p90
  24 to 63 mm; self contact's range is 20 mm and it ignores the offsets), the rest past its edge
  (p50 28 to 74, p90 127 mm and more). Parts are by the bound node's name, so some are misassigned.
  (The reading that the panels' skins stop short of their painted outlines was wrong: see below.)
- Self contact at the contact offsets (CPU, SELF_OFFSET: 1 the triangles, 2 the contacters too; off),
  4 runs, edge contact on, own paint at the end, median: pole 64 5.38 -> 2.97 -> 2.90, head on 60
  7.50 -> 6.17 -> 4.73, side 60 3.97 -> 5.75 -> 3.86; driving 0; car through car worst head on
  2.62 -> 4.00 -> 2.52. Mode 2: no test of gen_sedan_contact changes, energy rises inside the bar. With
  SELF_EDGE=1 on top it is worse (head on own 10.8, car through car worst 8.8). The GPU's self
  contact has no offsets, so it stays off until it does.
- As built the panels do reach their outlines (paint more than 2 cm past the car's collision skin:
  hood 0 %, fenders 5 %, body 7 %, doors 12 %, inner shell 10 %), no collision triangle dies in these
  crashes, and the earlier "past its edge" matched parts by name.
- The paint has its own weighted skinning (rubbleSkin) and leaves its collision panel in a crush: from
  where the panel it lies on carries it, driving p99 1 mm; pole 64 p50 2 to 3, p90 34 to 43, p99 103
  to 183 mm; head on 60 p90 46 to 58, p99 113 to 182.
- The paint carried rigidly by its collision triangle instead (PAINT_ON_SKIN=1) is four times worse
  (pole at the end 5.4 -> 21.9 m, head on 7.5 -> 27.2): it tears at each triangle's edge. Not a fix.
  What is left to try is the reverse: collide points and edges placed as the paint is.
- Where a car's own crossings are (PAINT_DEPTH=1, SELF_OFFSET=2, one run): small places at the seams
  of neighbours. Pole 64: door x fender 0.88 m of line over 0.26 m, bumper x fender 0.65 over 0.16,
  doors' inner x hood 0.34 to 0.38 over 0.15, the rest under 0.3 m. Side 60: fender x hood 1.0 m over
  0.52; bumper x itself 0.88 over 1.95. One triangle reaches through another by 8 mm at most (they are
  that small). The "behind" figures it prints are not a depth: they count any vertex within 30 cm
  under the other part's surface, curved panels included, with no baseline as built.
- The self edge experiment (SELF_EDGE=1) already solves its hits in the self node hits' turn; it
  hardly tests the seams. Pole 64, edge pairs within the range with both closest points inside
  (pair-steps): 49,018 excluded for starting within 25 mm (44 pairs, 27 of them facing alike, p50
  14.6 mm apart), 9,520 rejected for facing alike, 1,462 (2 %) tested. Many of the excluded are seams
  at rest, not crossings. A seam pair needs its side remembered from spawn and contact only at
  crossing; neighbours' edges are nearly parallel, where the edges' cross product gives no normal, so
  the pair's own faces must give the side.
- Built and dropped: those pairs keep the direction between their closest points while apart and are
  pushed back to it when it flips, or apart inside 80 % of the gap first seen (SELF_OFFSET=2, 4 runs,
  medians, own paint at the end / car through car worst). Alone: pole 2.90 -> 3.03, head on 4.73 ->
  8.19 / 2.52 -> 5.62; with the other self edge pairs: pole 3.16, head on 8.00 / 8.56; driving 0. The
  control (neither tested) is the baseline to the digit. So the seams' edge pairs are not the lever.

### An outside read (Fable, read-only, 2026-11-25) and its first tests

Its points found true in the code: the paint follows a smooth field of radius 0.42 m or more a part
(flexskin.rs R_MIN); the page draws stretched weighted paint (the measure leaves it out; counted,
PAINT_STRETCHED=1, the numbers are the same); clip memory keeps a node's two smallest keys, not its
deepest; the edge pass takes the cross product down to 1e-8 (no near-parallel rule). Taken from its
report, not rechecked: painted panels are double sheets, so one panel through another counts up to
four times, and inner door faces are in the count.
- "Ends clear" was mostly the cars apart: head on 60 at 1 s the centres are 8.5 to 10.8 m apart.
  In contact at the end: head on 100 (3.2 to 8.8 m, median 3.6), side 60 and 30 pressed (3.1 to 3.4).
- Its next experiment, the paint embedded continuously in its part's collision skin (closest point,
  each node carrying the rest offset by its rotation; all 354,966 vertices): own paint at the end pole
  5.38 -> 8.13, head on 7.50 -> 18.85: its stop rule (under 40 % less) is met. The driving control is
  not clean (0 -> 3.2 worst), so it says this embedding is no way, not that the drawing has no part.
- Paint points with edge contact (RUBBLE_PAINT_POINTS=1), worst median: head on 60 3.21 -> 4.07, 100
  2.23 -> 3.29, pressed 60 3.72 -> 3.23, pressed side 60 1.38 -> 1.41: dropped.
- The default path (no offsets, edge contact off) on CUDA, 46223bae against b596b26a: 12 rows bit for
  bit, none slower (head on +2.2 % the most), with the shipped ssqrt.
- Near-parallel edges by their closest points' direction (below a sine of 0.05 and 0.2; built,
  measured, dropped): edges through, medians of 12, head on 60 58 -> 67, 62; 100 61 -> 70, 72; overlap
  the same; pressed side 60 at the end 0.15 -> 0.24, 0.50 m. The tail cases at an edge's end were not
  tried. In contact and clear: head on 100 pressed, centres 3.0 to 3.7 m, 0 in all 8 runs.

### Edge contact had never run in a browser (fixed 2026-11-25)

- **The fault.** The `edges` module was missing from `ExplicitBeamsGpu::compute_specs`, so on a
  host that compiles ahead (the browser) `pipeline_ready("edges")` was never true and every
  frame went out with edge contact off. Every earlier statement here that `edge=1` ran on the
  lab page is wrong: those checks only saw the page load. Native and CUDA numbers stand.
- **The fix** lists the pass for any vehicle with `edge_contact`; the frame check waits while
  the split modules compile instead of panicking.
  Tests: `a_browser_host_compiles_the_edge_pass`, `edge_contact_waits_for_its_deferred_compiles`.
- **Proof on the page** (`mode=drive&car=gen&surface=1&cars=3`, pipelines counted by hooking
  `setPipeline`): `explicit_beams_split_edges_wide` 5400 against 5400 member steps, no
  engine error. The cars were not in contact in that run, so a crash on the page is not yet
  measured, nor is its frame time (the pane was hidden and throttled to 1 frame a second).
- **Until the compile finishes**, a page's first frames run without edge contact.
- **Default.** The lab turns it on for up to three generated cars on the flat arena (parked
  ones counted); at four the box clears (checked). Larger scenes do not afford it yet.

### On a page, edge contact started 111 s after the crash could (fixed 2026-11-25)

- **Measured** (Chrome, M3 Max, car at 60 km/h into a parked car's side, cold kernel cache): after "ready",
  111 s of frames ran on the groups' workgroup (28 ms a step, half real time) with edge contact
  silently off, on = off to the last digit, while its modules compiled in the background. The
  3-car proof above saw the pass only because it waited longer than that.
- **Fix:** with edge contact on, the modules its path runs (member, tests, groupbegin, grouplevels,
  grouplv, edges) compile before the first frame. A cold load is longer; `edge_gate.rs` holds it.
- **The crash once compiled:** 60 steps/s; resting contact 10.6 ms a step on, 11.6 off. The impact
  second: 13.7 ms. Frame-locked stills (`out/chrome-crash`, one run each): on, the cars part
  and the red one rolls on at 6 km/h (t 5 s); off, they stay locked at 0 km/h.

### Depth instead of line length, and what it found: authoring (2026-11-25)

Owner's bar: at most 1 cm from 0.3 s after a hit, at most 5 cm during it and above 1 cm for at most
0.2 s; 8 runs a case (`paint_overlap::depth`: winding number of the contact surface over 0.75, less
that surface's gap to the paint as built; contact off reads 626 mm; sampled every 20 ms).
- **B-pillar with no collision triangle:** side 60 88 mm -> 0. **Dogleg node ahead of the rear arch in
  no triangle (`cover_bare`):** side 60 pressed, worst 63 -> 34 mm.
- Edge contact on, after both, worst / from 0.3 s on (mm): side 30 0/0 and head 30 pressed 7/7, head 60
  pressed 36/0 (these three before the dogleg's faces), side 60 0/0, head 60 52/0, head 100 54/0,
  side 90 68/0, side 60 pressed 34/34 (a second hit at 0.94 s; the measure counts it as lasting).
  After a hit nothing stays in any of the eight by the end.
- Not adopted: paint carried by its own panel (`PAINT_ON_SKIN=1`): side 90 68 -> 30, head 100 54 -> 39,
  side 60 pressed 34 -> 18, head 60 52 -> 63, but a car's own paint crossing length about doubles
  (head 60 end 6.7 -> 17.8 m). A depth measure of a car against itself has to judge it.
- The pillar's own crossing (end median, side 60): 4.1 -> 5.1 m; 5.3 with the dogleg's faces.
- **Speculative look-ahead (`SPECULATIVE=1`, CPU only) by depth, 8 runs, worst (mm):** head 60 52 -> 44
  (passes), side 90 68 -> 53, head 100 54 -> 60. Not adopted: one row worse, and not on the GPU.
- The acceptance tests (`gen_sedan_paint.rs`, `accept_*`): 4 of 8 pass; the file 956 -> 475 s with the
  own-paint and line measures off where a test does not read them (`crash_measuring`).

### The car-to-car band (2026-11-25): 50 mm passes all eight scenes

Depth, 8 runs, edge contact on, the bar's verdict (worst mm / from 0.3 s on):

| Scene | 25 mm (today) | 35 mm | 50 mm | speed-scaled k 2 | k 4 |
|---|---|---|---|---|---|
| side 30, side 60, head 30 pressed | pass | pass | pass | pass | pass |
| head 60 pressed | 36/0 | 7/0 | 2/0 | 45/0 | 43/0 |
| head 60 | **52**/0 | 48/0 | 47/0 | 31/0 | 37/0 |
| head 100 | **54**/0 | **92**/0 | 10/0 | 47/0 | 11/0 |
| side 90 | **68**/0 | 18/0 | 8/0 | **68**/0 | **56**/0 |
| side 60 pressed | 34/**34** | 23/**23** | 7/7 | 34/**34** | 32/**32** |

(`VV_RANGE`, `VV_SPEED_BAND=k`: the band k x the cars' closing speed x the step, at most 50 mm.)
With 50 mm: energy never rises (all 8 rows of the energy tests within 0.5 %), the cars part
after a head-on as with contact off, side scrapes slide; a slow push ends with the paint 21 mm
apart (7 mm at 25 mm): the visible cost. Self contact keeps 25 mm. The GPU reads one range per
car for self and car-to-car contact (`gStatic[v].z`): a car-to-car band is a new parameter.

### Edge contact silently off on a cold page, again (fixed 2026-11-25, evening)

With contact offsets (the generated sedan) the levels-once kernels do not serve, so the split
path runs the groups' kernel with every pass (`explicit_beams_split_group*`), which stayed
deferred (and `prefer_self_split` deferred it again): until its minutes-long compile ended a page
ran the groups' workgroup, edge contact off, 30 ms a step. The afternoon's check passed only
because Chrome had that kernel cached; a kernel change exposed it. Now it is compiled before the
first frame with edge contact (`edge_gate.rs` checks the lab's order). Cold load 86 s on the M3 Max.
Chrome, side 60 into a parked car: split path and edge pass from the first frame; median 10.83 ms
a step, worst 23.5 ms at the impact (47-56 steps/s for about 3 s, then 60); the 50 mm car-to-car
band (lab control `car-to-car band`) the same to 0.01 ms.

### A per-pair speed-scaled band does not help; paint outside the collision surface does matter (2026-11-25)

- **Speed-scaled band** (`SPEED_BAND=k`: each pair's band 25 mm + k x its closing speed x the step, at
  most 75 mm; node, edge-edge and node-edge passes), worst depth / from 0.3 s on (mm), 8 runs: k 2:
  head 60 85/0, head 60 pressed 59/0, side 90 70/0, side 60 pressed 11/11 (the rest pass); k 3: head
  100 79/51, head 60 51/0, side 90 62/0, head 30 pressed 13/13. Worse than the flat 50 mm (all eight
  pass) and than 25 mm in places: the band thins as a pair slows inside it. Not adopted.
- **Why the flat band helps:** painted points more than 2 cm outside the car's own contact surface
  as built (`PAINT_CALIB=1`): body sides 6,698 (to 142 mm), each front door 3,961 (174 mm), wheel-arch
  lips 1,850-3,000 each (180-239 mm), sills 1,750 each (166 mm). Some are at openings (wheel wells,
  side windows: glass is visual only), but side 90's failure is the parked car's door top front, at
  the window frame: paint no contact surface covers. The vehicle lint's `skin_coverage` passes on its
  p99 and misses this 1 %. The fix belongs in the authoring (frames and lips covered), not the band.

### Edge contact's jitter after hard hits (2026-11-25, night)

The owner saw cars vibrating violently after hard hits. Measure: each car's nodes' rms speed about
its centre over the last 0.25 s (`JITTER`, `vehicle_paint_overlap`), 4 runs, worst (m/s), pressing on:
- side 100: edge on 2.06, off 0.31; head 60: edge on 1.18, off 0.13. Edge contact makes it.
- Side 100: `EDGE_DEPTH_M` 0.005 0.11, 0.01 0.11, 0.015 2.09 (one run), 0.1 (default) 2.06; the smooth
  fade (`EDGE_FADE=1`: a pair crossed past the band pushes less the deeper it is, to nothing at
  EDGE_DEPTH) 0.17. Cutting the depth to 5-10 mm loses the settled state (head 60 pressed 47 mm,
  head 100 one run 107 mm left from 0.3 s on); the fade keeps it (all scenes as before but head 100,
  one run 26 mm).
- Head 60 pressed with the fade: still 1.95; with it and the residual off 1.68, depth 50 mm 0.98, 30 mm
  1.19, node-edge off 1.84. Its fastest nodes: the braked car's crushed front bumper and radiator
  (4.6-7.6 m/s about the car's centre): light crushed parts trapped between the cars.
- Without throttle the edge contact calms hard hits (worst, m/s, edge on / off): head 130 0.83 / 0.76,
  side 100 0.09 / 0.50, side 130 0.15 / 1.64. The jitter needs a car pressing on at full throttle.
- Head 60 pressed: the braked car's front bumper corner (bf_o16) swings at about 62 Hz, peak 12 m/s
  about the car's centre (31 reversals in 0.25 s; JITTER_FREQ=car:node): a steady oscillation, not
  step-to-step chatter. Not friction (EDGE_NO_FRICTION=1: 0.98), not the contact throttle
  (THROTTLE_OFF=1: unchanged 1.18), not the residual (1.68), not the depth (fade 1.95).

### Impact cost by pass (CUDA T4, 2026-11-25, `PROFILE_SPLIT=1 run.sh --cuda perf=0 ROWS`)

ms a frame of 34 steps: head-on mid-crush 11.7 (members' step 8.9, edge pass 1.0); side mid-crush
17.0 (members 9.7, edge 3.7); pile of 8 86.6 (edge 58.2, members 12.7, groups 5.6, tests 4.2).
For two or three cars the members' step is most of an impact; the edge pass only dominates piles.
A pile rebuilds the group's edge candidates whenever any node moves a skin (12 cm) about the
group's mean; per-owner rebuilds (lossless when the two cars' motions sum under two skins) are the
lever. (`RUBBLE_EDGE_SKIN` cannot be A/B'd on the portable fixtures: they carry the upload's
statics, the skin with them.)
- Its trace (JITTER_TRACE=1, the last 80 steps of head 60 pressed): the bumper corner's velocity a
  smooth sine (+2.6 to -3.9 m/s over about 32 steps), 3-4 edge hits every step and no node hit: a
  steady 62 Hz swing of about 1 cm, fed while the throttle pushes, not hits switching on and off.
