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
