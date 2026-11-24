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
