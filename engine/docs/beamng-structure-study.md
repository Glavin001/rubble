# BeamNG structure study: what a good car is built from, and the specification for ours

2026-11-12, branch `docs/beamng-structure-study`. A research document: no engine,
rigger or generator code changed. It answers one question for Step 2d of the
plan (a real body structure and mechanical parts for the generated sedan):
**what does a good BeamNG car consist of, part by part and node by node, and
what should ours be?**

Three kinds of statement are kept apart and labelled:

- **[Doc]** what a public source says (cited; the list of sources is Appendix A);
- **[Measured]** a number read from five cars' data files with the committed
  tools (reproducible: §0.3);
- **[Hypothesis]** an explanation or a recommendation that follows from the two
  but has not been tested on Rubble.

The specification (§10) marks each item **BeamNG baseline** (the five cars do
it) or **beyond BeamNG** (they do not; we need it for our own goals).

## Summary

1. **A BeamNG car is 460–680 nodes and 2,800–4,400 beams; 40–66 of its parts
   own nodes** **[Measured]**. The body shell is one part of 118–138 nodes carrying
   30–38 % of the mass; the suspension and subframes carry 17–19 %; the
   powertrain 12–20 %.
2. **The shell's nodes lie on the visible body:** half are within 0.5–1.6 cm of
   the body mesh and 80–98 % within 5 cm. The shell is a lattice drawn on the
   floor, sills, rails, towers, cowl, pillars and quarter panels. There is no
   hidden frame.
3. **The shell is a braced space frame, not a grid of skin:** 5.0–5.8 beams per
   node, 9–11 neighbours per node, every node a corner of a tetrahedron, 12–28 %
   of beams skipping a node, 8–10 % crossing the car. No SUPPORT beams inside
   it (0–4 of 600–800).
4. **Heavy nodes where loads enter:** strut towers 8.5 kg, subframe mounts 6 kg,
   rails 5–5.5 kg, floor 4.5 kg, quarter panels 2.8 kg, pillars and roof
   1.5–1.7 kg (ETK). The nodes suspension lands on are 1.3–4× the shell's
   median.
5. **The crush path is graded by yield force, not by spring.** Front and rear
   beams are the stiffest (k 3–7e6 N/m) with the lowest `beamDeform`
   (4–13 kN); cabin beams are softer (1.5e6) with the highest (15–32 kN, up to
   65 kN). Along the car, the shell's longitudinal yield sums to 0.2–0.5 MN in
   the front, 0.7–0.9 MN at the firewall and 1.1–1.3 MN in the cabin.
6. **Every class of node sits at the same stiffness-to-mass ratio:**
   Σk/m ≈ 7–10e6 s⁻² per node, ω·h = h·√(Σk/m) ≈ 1.3–1.6 at 2 kHz, whether the
   node is a 22 kg engine corner or a 0.5 kg fender node. That is the stability
   edge, and BeamNG's staff say so. **It is the one number that explains "not
   jello".**
7. **Panels are one layer of nodes plus 1–3 non-colliding rigidifier nodes**
   0.2–0.4 m off the surface. No car models inner and outer skins separately.
8. **Parts hang on small named break groups** of 5–14 beams: fenders 12 kN per
   beam, bumpers 6–22 kN, hinges 8–55 kN, subframe mounts 80 kN–1.2 MN, wheel
   to hub 57–240 kN. 97–99 % of shell beams are unbreakable.
9. **11 % of beams are SUPPORT, all between parts:** engine to body (36–56 per
   car), closures to their openings, bumpers, lamps, exhaust. 36–46 % of them
   are precompressed.
10. **Mechanical parts are few heavy nodes:** engine 8–10 block nodes of
    12–28 kg, fully cross-braced at k 2e7; a hub is 4 nodes of 4–5 kg in a
    tetrahedron; a suspension arm is two beams, not a part with nodes; the
    driveshaft is one breakable beam.
11. **A wheel reaches the shell in two hops** (axle nodes → hub tetrahedron →
    arms and strut → body), and the weakest link is the wheel-to-hub group
    (ETK: 8 beams at 168 kN), then the spring and damper (180 kN), the tie rod
    (250 kN), then the subframe mounts (1.2 MN).
12. **Glass is a deform group, not a body:** 4–62 beams across the opening with
    a trigger ratio of 0.005–0.1, a mesh material swap, and (ETK, CCF) one
    centre node carrying the pane's mass.
13. **Flexbodies bind to one node group (60–75 % of rows),** each vertex to 3–4
    of its nodes; a vertex's nearest node is 7–16 cm away and its third nearest
    24–40 cm. Both faces of a panel mesh ride the same single layer of nodes.
14. **Weaknesses we can beat:** no interior structure beyond seats and a dash
    ballast; seats and radiators that cannot detach; roofs of 4–8 nodes; a
    body torsion mode at 6–12 Hz; frontal crush too short and too stiff
    against NHTSA bands; poles passing between shell nodes; lattice hand-tuned
    with 49–77 distinct beam settings.
15. **Whole-car ω·h is 1.80–1.90 on all four cars** (power iteration on
    Rubble's CPU reference; the explicit limit is 2). They are authored to
    90–95 % of the step's bound.
16. **The specification (§10):** a 131-node shell on a modelled body-in-white,
    mechanical parts as few heavy nodes, a 4-node hub per wheel with a
    breakable 8-beam wheel group, and 20 lint rules, each traced to a number
    here.

## Contents

- [0. Scope, corpus, method](#0-scope-corpus-method)
- [1. What BeamNG's own guidance says](#1-what-beamngs-own-guidance-says)
- [2. Part inventory and slot tree](#2-part-inventory-and-slot-tree)
- [3. The unibody](#3-the-unibody)
- [4. Mechanical parts and the wheel's load path](#4-mechanical-parts-and-the-wheels-load-path)
- [5. Closures, panels, interior, glass](#5-closures-panels-interior-glass)
- [6. Flexbody binding](#6-flexbody-binding)
- [7. Stability and stiffness](#7-stability-and-stiffness)
- [8. The guidance against the cars](#8-the-guidance-against-the-cars)
- [9. Weaknesses](#9-weaknesses)
- [10. Specification for the sedan (Step 2d)](#10-specification-for-the-sedan-step-2d)
- [11. Glossary of JBeam fields](#11-glossary-of-jbeam-fields)
- [12. What we still do not know](#12-what-we-still-do-not-know)
- [Appendix A. Sources read](#appendix-a-sources-read)
- [Appendix B. Reproducing every number](#appendix-b-reproducing-every-number)

Related: [beamng-authoring-census.md](beamng-authoring-census.md) (per-part
bands, hinges and latches, creases; not repeated here),
[etkk-audit.md](etkk-audit.md) (the ETK in crash scenes),
[generated-vehicles-lessons.md](generated-vehicles-lessons.md),
[jbeam-semantics-spec.md](jbeam-semantics-spec.md) (field semantics as
implemented).

---

## 0. Scope, corpus, method

### 0.1 The corpus

Five community mods, unpacked under `~/Downloads/jbeams/` (local only). They
are read as data: `.jbeam`, `.pc` and `.dae`. No Lua was read or run, and
nothing from them is committed: no file, node table, coordinate list or mesh.
This document holds counts, ranges and descriptions.

| Label | Vehicle | Configuration | Kind |
|---|---|---|---|
| ETK | ETK K-series (`etkk_lucasbe`) | `kc6` (hardtop coupe); `kr6` (soft top) where it differs | unibody, front engine, rear drive, double wishbones |
| CCF | CCF (`thw_ccf2`) | `f4_sport_M` | unibody roadster with a soft top |
| Vertex | Gavril Vertex (`innocent_gavril_vertex`) | `gavril_vertex` | unibody four-door sedan: **the nearest to our sedan** |
| SUV | full-size SUV (`fsw_fullsizesuv`) | `mid-sport` | body on a ladder frame: the contrast |
| Trailer | boat hauler (`boathauler`) | `gooseneck` | a trailer: frame, axles and wheels only |

The trailer has no body, closures, glass or powertrain. It appears where it
adds something (frames, wheels, field use) and is left out elsewhere.

Not used: BeamNG's own BX (in the earlier census; its files are not on this
machine now), and the hobby and tutorial vehicles under
`inspiration/beamng-mods/` (not complete road cars).

### 0.2 Conventions

- Frame: JBeam's, +X left, +Y rear, +Z up, metres. **Station s** is the
  longitudinal position as a fraction of the wheelbase: 0 at the front axle, 1
  at the rear axle. Stations: nose (s < −0.18), front axle and towers
  (−0.18…0.08), firewall and A-pillar (0.08…0.30), cabin front (0.30…0.62),
  cabin rear (0.62…0.90), rear axle and towers (0.90…1.12), tail (> 1.12).
- **Shell** = the body part's own nodes (for the SUV, body plus frame).
- **Class** of a part comes from its name (`classify()` in the script): body,
  frame, door, hood, trunk, fender, bumper, trim, lights, glass, interior,
  suspension, wheel, engine, driveline, exhaust, fuel, cooling.
- k = `beamSpring` (N/m), c = `beamDamp` (N·s/m), deform = `beamDeform` (N),
  strength = `beamStrength` (N), L = length at spawn. ε_y = deform/(k·L).
  h = 0.5 ms.
- Tyre nodes are generated by the game and are not in any count. Masses
  exclude tyres and rims' generated nodes.

### 0.3 Method and tools

| Tool | What it does |
|---|---|
| `structure_dump` in `engine/crates/engine-softbody/tests/authoring_census.rs` | Assembles each car with Rubble's own JBeam loader (slots, `.pc`, variables, inherited options) and writes one JSON per car, outside the repository |
| `engine/tools/beamng-structure-study/analyze.py` | Reads those files and prints the tables (standard library only) |
| `engine/tools/beamng-structure-study/run.sh` | Runs both for all five cars |
| `examples/sedan_stiffness.rs` (new `label=vehicle,config.pc` form) | Modes, kerb wobble and settle on the CPU reference, for any JBeam car |

Appendix B gives the commands. Everything is CPU only.

---

## 1. What BeamNG's own guidance says

All **[Doc]**. Sources are named by short keys; Appendix A gives each URL, the
date read (2026-11-12 for all) and what the page holds. Pages were read through
a summarising fetch tool, so numbers were spot-checked on the pages that carry
weight here (Nodes, Flexbodies, the basic car tutorial). Old wiki pages were
read from the Internet Archive.

### 1.1 The model

- Nodes are point masses with no size and no rotation. Beams are massless
  spring-dampers that carry axial force only, so every node is a free hinge and
  an unbraced box folds. [D-intro, W-theory, P-icse, R-concepts]
- The physics core advances 0.5 ms per step (2,000 Hz). The rate is compiled
  in; 2,000 is the supported rate. [D-tight, F-rate, P-icse]
- A beam is a parallel spring and damper with a deform threshold and a break
  threshold. [P-icse, D-beams]
- The published example car has 714 nodes, 4,479 beams and 976 triangles
  (tyres included). [P-icse]
- Only nodes collide: with terrain, with static objects and with triangles of
  vehicles. [D-tips, R-concepts]

### 1.2 Node placement and structure

- Model the mesh first, then place nodes on it as an evenly spaced lattice.
  Spacing trades deformation detail against cost and against gaps that objects
  pass through. [W-construct]
- A car is typically 3–4 nodes wide with similar density along its length.
  Extra nodes cost "weight budget": the node count sets how light the car can
  be. [D-tutorial]
- A starter guide says not to exceed about a hundred nodes for a body; its
  example body has 48. [W-intro]
- Start from the suspension mounts and component mounts, prefer straight lines
  and right angles, and reuse one node as both structure and mount. Find the
  main rigid box first. [D-tutorial]
- Floor: two matching grids one above the other, beamed along all three axes,
  with both diagonals in every rectangle, then rigidifiers that skip a node
  lengthwise and crosswise. [W-construct]
- Pillars, firewall and roof may have no thickness. Brace the corners of door
  and window openings with as few beams as possible. Do not span the cabin,
  engine bay or boot with beams. [W-construct]
- To hold a node, use at least three beams roughly perpendicular to each other.
  Too narrow a tripod raises stress; too wide risks the node flipping through.
  [D-tutorial]
- Fix wobble by topology before raising `beamSpring`. [D-tutorial]
- Flat panels need a rigidifier: a node with collision and self-collision off,
  about 0.5 m off the surface, beamed to every panel node. Its beams deform
  more easily than the surface beams (the tutorial uses deform 5,000, strength
  FLT_MAX) so the panel can fold. Panels should be at least 3 × 3 nodes.
  Box-shaped parts need none. [D-tutorial, D-issues, W-construct]
- Long wobbling structures get damping-only beams (spring 0) to a part not
  directly connected, in a break-group table with `breakGroupType` 1.
  [D-tips]
- Torsion bars stiffen flat structures (ladder frames, shock mounts, arms)
  without a rigidifier node. [D-torsion]
- Roof: tune so the roof-crush test reads about twice the real IIHS rating,
  which may need rigidifiers in the roof and pillars. [D-debug]
- Node names: a letter prefix, a row number and a side suffix. [D-nodes]

### 1.3 Mass, stiffness, damping, stability

- If a node's connected stiffness or damping is too high for its mass, it
  vibrates or the structure explodes. The cure is more node mass or less
  spring and damping. The illustrated case is unstable at 6 kg and stable at
  7 kg. No formula is given anywhere. [D-intro, W-theory, D-issues]
- Three grades of instability: the car vanishes with an "instability" message,
  parts break off at spawn, or parts pulsate. [D-issues]
- A BeamNG developer: "Most of the vehicles have the spring values very close
  to the max". Halving a part's mass means starting from half the spring and
  damping. He also says beam length does not change the limit: it is a
  stiffness-to-mass ratio, so small light structures have a stiffness ceiling.
  [F-goosah]
- Give all frame nodes the same weight except shock mounts; over-precise
  weights invite instability. [F-agenty-w]
- The step's bound is not stated by BeamNG. The textbook bound for an explicit
  step is ω·h ≤ 2 (not from a BeamNG source; `generated-vehicles-lessons.md`
  §2).
- Typical values: suspension spring 40,000 N/m with no damping; damper 4,500
  N·s/m with no spring; structural arms 8e6 / 125; steering rack 1.4e7 / 250.
  [D-intro, W-theory]
- Tutorial values: frame 1.2e6 / 150, deform 50,000, FLT_MAX,
  `deformLimitExpansion` 1.2; mounts 1.5e6 / 150, deform 30,000; panel
  attachment deform 15,000, strength 15,000; node weights 2–3 kg.
  [D-tutorial]
- Starter values (old wiki): node weight 10 kg, spring 2e6, damp 200, deform
  80,000, strength 800,000. Lower damping permits a higher spring. [W-intro]
- Metal is stiff and lightly damped, plastic softer and more damped, rubber
  very soft and heavily damped. [W-beams]
- Defaults: spring 4.3e6, damp 580, deform 220,000, strength FLT_MAX, node
  weight 25 kg. [D-beams, D-nodes]
- Rigs of Rods, the lineage: defaults 9e6 / 12,000 / 400,000 / 1e6; a car
  example 3e6 / 10,000 / 100,000 / 250,000; minimum node mass 50 kg by
  default; excess damping on light nodes is the commonest cause of explosions
  at spawn. Its values are for its own solver and are not BeamNG's.
  [R-truck, R-weight]

### 1.4 Deformation and breaking

- `beamDeform` is the force at which the beam's rest length changes;
  `beamStrength` the force at which it breaks. A broken beam also removes mesh
  polygons near it and makes its triangles non-colliding
  (`disableMeshBreaking`, `disableTriangleBreaking` prevent that).
  [D-beams, W-theory]
- The main shape should be nearly unbreakable, because beams tear badly;
  breakable beams are for parts that detach. [W-construct]
- Deform hierarchy on a floor: grid beams strongest (the structure should bend,
  not shrink), diagonals weaker, skip-node rigidifiers weakest. [W-construct]
- Strength should exceed deform, except for glass. Close values make a brittle
  beam, a wide gap a tough one. [W-beams]
- `deformLimitExpansion` caps permanent stretch as a fraction of the length;
  staff give 1.05–1.1 as typical and 1.001–1.01 for very small parts.
  `deformLimit` caps permanent compression. `deformLimitStress` caps how far
  the deform threshold grows. [D-beams, F-agenty-dle]
- `breakGroup`: when one member breaks, all break. `breakGroupType` 1 marks a
  beam that is broken by its group but does not trigger it. [D-beams]
- Detachable parts hang on breakable beams in a hinge group and a latch group;
  the hinge is the strongest and the latch breaks first. [D-tutorial]
- Support beams resist compression only, break past `beamLongBound`, and with
  precompression below 1 stay inactive until squeezed. They cover nearly every
  self-collision case. They are cheaper than bounded beams. [D-support,
  D-bounded, W-construct]
- Couplers latch doors, hoods and boots: a `couplerTag` node and a `tag` node
  at the same place, with `couplerStrength` (the documented hood example is
  12,000 N). A coupler can join a break group. [D-couplers]

### 1.5 Parts, mounts, hardware

- Anything that can detach or move is its own mesh and its own part: bumpers,
  hood, doors, engine, subframes. Parts with a material swap (glass, lamps) are
  separate meshes with one material. [D-split, D-model]
- Suspension: a node at every pivot, attached to as few chassis nodes as
  possible; a rigid spindle carries the two wheel-axis nodes; arms and spindles
  are as stiff as stability allows. [W-construct]
- A wheel is generated around two axle nodes that belong to the hub. Brake
  reaction goes through `nodeArm`; drive reaction through `torqueArm`,
  `torqueArm2` and `torqueCoupling`, which must be non-collinear nodes on one
  rigid structure; heavier and farther apart tolerates more torque. Brake and
  torque reaction nodes should be at least about 2 kg on cars.
  `axleBeams` are the half-shafts; breaking them cuts drive. [D-wheels,
  D-rotators]
- A coilover is a spring beam (no damping), a bounded damper beam and a
  bump-stop beam. [D-tutorial]
- Engine: an 8-node cube with every node beamed to every other, plus a
  transmission-end node; soft mounts with few beams; support and bounded beams
  stop the block clipping the body. Mounts are 3–4 beams per mount node.
  [W-construct, D-tutorial]
- `engineGroup` node groups (`engine_block`, `radiator`, `engine_intake`,
  `fuel`) are the powertrain's hold on the node-beam structure; `engine_block`
  and `radiator` need at least two nodes each. `torqueReactionNodes` are three
  engine nodes that take the torque reaction. `breakTriggerBeam` kills a
  device when that beam breaks. [D-nodes, W-engine]
- Rails and torsion bars cannot break alone; a dummy beam carries their break
  group. [D-tips]

### 1.6 Flexbodies

- A flexbody names one or more node groups, and only those nodes move it. Each
  vertex is bound to 3 or 4 of them: a centre node, a VX and a VY node whose
  lines to the centre are near a right angle, and a VZ node if one fits. The
  closest nodes that fit are chosen. No maximum distance is stated.
  [D-flex]
- A group needs at least three nodes that are not collinear and cover the mesh.
  Spikes come from outlier nodes or nodes sharing a position: keep the group a
  regular grid, exclude outliers, or add a help node. [D-flex, D-issues,
  W-problems]
- `deformGroup` on a flexbody swaps `deformMaterialBase` for
  `deformMaterialDamaged` when the group fires; such a mesh should have one
  material. [D-flex]
- Non-removable panels may share the body's group. [D-tutorial]
- Every vertex is recomputed each frame, so polygon counts should stay low.
  [D-model]
- Props are rigid meshes on three reference nodes; a break or deform group
  hides them. [D-props]

### 1.7 Collision

- Triangles are what other vehicles' nodes hit. They are one-sided; wind them
  counter-clockwise facing out. [D-tri, W-tri]
- A triangle's collision thickness is 2.5 cm. Keep nodes at least 3 cm from
  triangles at spawn (3.5 cm for flexible parts), or panels push apart.
  [D-tips, D-nodes]
- A node does not self-collide with triangles of its own first group. Every
  self-colliding node should own at least NONCOLLIDABLE triangles.
  [D-tips]
- Cost: a collidable triangle adds work for each node within 50 cm; it sleeps
  when no candidate node is within 10 cm. [D-tips, D-nodes]
- Thin edges gain collision points from slidenodes on rails. [D-tips]
- `collision` defaults to true, `selfCollision` to false. `frictionCoef`
  defaults to 1. `dragCoef` on triangles drives aerodynamics. [D-nodes,
  D-tri]

### 1.8 Glass

- Glass damage is binary: a deform group fires and the mesh swaps to a damaged
  material. [D-glass]
- Brittle duplicate beams with a `deformGroup` are the documented glass idiom.
  If the beam also has a `breakGroup`, breaking it spawns glass particles.
  `deformationTriggerRatio` is the fraction of length that fires the group:
  typically under 0.1; an old example uses 0.03 for a windscreen. It is the
  expensive property. [D-beams, D-tips, W-deform]

### 1.9 Checking a structure

- Debug views: beam stress, deformation, break groups, deform groups, support
  bounds; node weights and node stability. [D-debug]
- The stress check: zero gravity, engine off. A healthy car shows stress only
  in suspension, tyres and latches. [D-tips]
- Repository rules for mods are procedural (no console errors, closed groups,
  origin at zero). There is no numeric quality standard. [D-modsupport]

---

## 2. Part inventory and slot tree

All **[Measured]** (`analyze.py --only parts`, `--only summary`).

### 2.1 Size of each car

| | ETK kc6 | CCF | Vertex | SUV | Trailer |
|---|---|---|---|---|---|
| Nodes (tyres excluded) | 457 | 635 | 519 | 680 | 689 |
| Mass (kg) | 1,373 | 1,418 | 1,282 | 3,020 | 4,004 |
| Beams | 2,789 | 4,048 | 3,428 | 4,422 | 4,095 |
| Collision triangles | 508 | 529 | 556 | 611 | 360 |
| Parts in the assembled configuration | 67 | 158 | 118 | 133 | 21 |
| Parts that own nodes | 40 | 49 | 51 | 66 | 12 |
| Flexbody rows / props | 116 / 30 | 242 / 26 | 179 / 25 | 227 / 13 | 120 / 0 |
| Torsion bars | 30 | 39 | 8 | 25 | 86 |
| Break groups | 62 | 64 | 81 | 105 | 20 |
| Deepest slot level below the root | 4 | 6 | 6 | 5 | 4 |

40–70 % of parts own no nodes. They carry only a mesh (badges, plates,
brake discs), a powertrain device (ECU, final drive), wheel data, or a skin.

### 2.2 Nodes / mass / share of mass by class

| Class | ETK kc6 | CCF | Vertex | SUV |
|---|---|---|---|---|
| body | 124 / 515 / 38 % | 138 / 426 / 30 % | 118 / 455 / 36 % | 96 / 266 / 9 % |
| frame | – | – | – | 52 / 609 / 20 % |
| roof (separate top) | 18 / 18 / 1 % | 42 / 26 / 2 % | – | – |
| doors | 26 / 43 / 3 % | 26 / 36 / 3 % | 40 / 72 / 6 % | 36 / 85 / 3 % |
| hood | 23 / 35 / 3 % | 21 / 22 / 2 % | 28 / 23 / 2 % | 30 / 37 / 1 % |
| boot lid / tailgate | 15 / 14 / 1 % | 12 / 12 / 1 % | 17 / 14 / 1 % | 24 / 37 / 1 % |
| fenders | 26 / 13 / 1 % | 20 / 17 / 1 % | 24 / 18 / 1 % | 32 / 28 / 1 % |
| bumpers (covers and bars) | 44 / 27 / 2 % | 69 / 58 / 4 % | 76 / 33 / 3 % | 61 / 52 / 2 % |
| trim (skirts, mirrors, spoilers) | 34 / 11 / 1 % | 54 / 33 / 2 % | 34 / 19 / 2 % | 38 / 415 / 14 % ¹ |
| lights | 22 / 17 / 1 % | 22 / 11 / 1 % | 24 / 17 / 1 % | 37 / 18 / 1 % |
| glass | 3 / 15 / 1 % | 8 / 8 / 1 % | 41 / 108 / 8 % | 56 / 142 / 5 % |
| interior (seats, dash, steering) | 14 / 34 / 2 % ² | 35 / 100 / 7 % | 19 / 51 / 4 % | 45 / 164 / 5 % |
| suspension and subframes | 47 / 267 / 19 % | 55 / 239 / 17 % | 50 / 226 / 18 % | 25 / 214 / 7 % ³ |
| wheel (axle nodes) | 8 / 40 / 3 % | 8 / 40 / 3 % | in suspension | 8 / 46 / 2 % |
| engine | 13 / 187 / 14 % | 24 / 220 / 16 % | 23 / 128 / 10 % | 22 / 245 / 8 % |
| driveline | 14 / 68 / 5 % | 17 / 61 / 4 % | 3 / 27 / 2 % | 5 / 121 / 4 % |
| exhaust | 15 / 21 / 2 % | 21 / 21 / 1 % | in engine | 20 / 37 / 1 % |
| fuel tank | 9 / 42 / 3 % | 9 / 38 / 3 % | 6 / 40 / 3 % | 13 / 76 / 3 % |
| radiator | 2 / 5 / 0 % | 7 / 9 / 1 % | 2 / 14 / 1 % | 2 / 8 / 0 % |

¹ The SUV hangs 400 kg of ballast on its side-skirt nodes. ² The ETK's dash
is four 15 kg nodes inside the body part (60 kg, counted under body). ³ The
SUV's axles are named differently and fall partly under "other" (78 nodes,
419 kg).

### 2.3 The ETK's part list, as a list to adopt

Per part instance. "Mounts" are the beams joining the part to its parent
structure. k, deform and strength are medians over those beams.

| Part (visual mesh) | Role | Nodes | kg | Node kg | Own beams | Tris | Mounts to | Mount beams (types) | Mount k / deform / strength | Break groups |
|---|---|---|---|---|---|---|---|---|---|---|
| Body shell (unibody mesh, carpets, engine-bay dressing, pedal box, dash) | structural | 124 | 515 | 1.5–15, p50 4.0 | 677 | 258 | – | – | – | – |
| Hardtop (roof skin, liner, rear glass) | structural | 18 | 18.5 | 1.0 | 138 | 20 | body | 72 (44 NORMAL, 28 SUPPORT) | 2.4e5 / 5,500 / 11–15 kN | 4 of 10–14 |
| Door (skin, quarter glass) + inner panel + glass + mirror | closure | 11 + 2 + 1 + 7 | 18.5 + 3 + 1.7 + 2.2 | 1.7 (mirror 0.25–0.5) | 82 + 12 + 41 | 9 + 2 + 6 | body | 26 (17 NORMAL, 5 SUPPORT, 4 BOUNDED) | 3e6 / 30 kN / 30 kN | upper hinge 14, lower hinge 8; coupler latch 35 kN |
| Hood | closure | 23 | 34.8 | 1.5 | 117 | 23 | body | 41 (20 SUPPORT, 12 NORMAL, 9 BOUNDED) | 6e5 / 9,000 / 10–200 kN | 2 hinges of 7; 2 couplers 30 kN |
| Boot lid | closure | 15 | 13.9 | 0.9 | 86 | 13 | body | 29 (17 SUPPORT, 12 NORMAL) | 6e5 / 26 kN / 9 kN–1.1 MN | 2 hinges of 6; coupler 40 kN |
| Fender | bolt-on | 13 | 6.5 | 0.5 | 68 | 10 | body | 26 (22 NORMAL, 4 SUPPORT) | 4e5 / 9,000 / 12 kN | 4 of 5–6 |
| Front bumper cover | bolt-on | 25 | 17.5 | 0.7 | 175 | 28 | body, fenders | 37 + 12 (28 NORMAL, 9 SUPPORT) | 2.4e5 / 5,500 / 11 kN | 2 of 14 to body, 2 of 6 to fenders |
| Rear bumper cover | bolt-on | 19 | 9.5 | 0.5 | 104 | 22 | body | 35 (23 NORMAL, 12 SUPPORT) | 4.5e5 / 4,000 / 6–12 kN | 3 of 7–8 |
| Side skirt | bolt-on | 10 | 3.4 | 0.3–0.5 | 83 | 6 | body | 45 (34 NORMAL, 11 SUPPORT) | 2e5 / 5,000 / 12 kN | 4 of 8–10 |
| Headlight (housing, 2 lenses) | bolt-on | 6 | 6.8 | 1.1 | 38 | 8 | body | 16 NORMAL | 1.3e6 / 6,000 / 26 kN | 1 of 16 |
| Tail light | bolt-on | 5 | 1.9 | 0.4 | 36 | 6 | body | 26 (14 SUPPORT, 12 NORMAL) | 3.8e5 / 1,400 / 15 kN | 1 of 12 |
| Windscreen (glass, damaged glass, wipers, mirror) | glass | 1 | 12 | 12 | 16 | 5 | body | 11 NORMAL | 1.5e6 / 6,750 / FLT_MAX | deform group of 62 |
| Seat (seat, rails) | interior | 6 | 12 | 2.0 | 39–42 | 4 | body | 26 (22 NORMAL, 4 SUPPORT) | 2.8e5 / 6,000 / FLT_MAX | none |
| Engine (block, intake, airbox, header) | mechanical | 10 | 182 | 22 (8 block nodes) | 40 | 2 | subframe, body | 6 BOUNDED mounts; 38 SUPPORT stops to body, 4 to subframe | mounts 32,000 / 20 kN / 90 kN | 2 mounts of 3 |
| Oil pan, intake | mechanical | 2, 1 | 3, 2 | 1.1–2 | 16, 8 | 0 | engine | 16, 8 | 1.3e6 / 16 kN / FLT_MAX; 1e5 / 12 kN / 36 kN | deform groups |
| Transmission (case, shifter) | mechanical | 11 | 25.9 | 1 × 22, 10 × 0.3–0.5 | 66 | 0 | engine, body | 24 to engine; 9 BOUNDED to body | 1.4e5 / 5,000 / 160 kN | 1 mount of 9 |
| Driveshaft | mechanical | 0 | 0 | – | 1 | 0 | transmission ↔ differential | 1 BOUNDED, no spring | – / 1,500 / 3,500 | 1 of 1 |
| Differential (rear) | mechanical | 3 | 42 | 14 | 24 | 0 | rear subframe | 24 NORMAL | 4.8e6 / 27 kN / FLT_MAX | – |
| Half-shafts | mechanical | 0 | 0 | – | 2 | 0 | differential ↔ axle node | 1 BOUNDED each | – / 3,600 / 9,500 | 1 each |
| Exhaust (two pipes, silencer) | mechanical | 15 | 21 | 1.4 | 96 | 7 | body, engine | 55 to body (28 NORMAL, 27 SUPPORT), 4 to engine | 47,000 / 4,000 / 15 kN (engine 65 kN) | 4 of 4–16 |
| Fuel tank | mechanical | 9 | 41.5 | 4.6 | 30 | 0 | body | 30 NORMAL | 30,000 / 5,000 / 35 kN | 1 of 26, passive |
| Radiator (core, hoses) | mechanical | 2 | 5.4 | 2.7 | 18 | 0 | body, engine | 12 + 5 NORMAL | 1e6 / 7,500 / FLT_MAX (hoses 3,600) | deform groups |
| Front suspension (subframe, arms, hubs, rack mounts) | mechanical | 19 | 97.6 | 3.8–7 | 145 | 10 | body | 84 (68 NORMAL, 12 BOUNDED, 4 SUPPORT) | 1.3e6 / 157 kN / 1.24 MN | 2 of 32 |
| Rear suspension (subframe, arms, hubs, tie rods) | mechanical | 24 | 164 | 5.5–8 | 171 | 18 | body | 48 (40 NORMAL, 8 BOUNDED) | 2.1e6 / 17 kN / 80–150 kN | 4 of 10 |
| Steering (rack, tie rods) | mechanical | 2 | 10 | 5 | 11 | 0 | subframe, hubs | 2 tie rods | 1.7e7 / 85 kN / 250 kN | – |
| Anti-roll bar (each) | mechanical | 2 | 2.6–3 | 1.3–1.5 | 6 | 0 | subframe, arms | 6 NORMAL | 2.5e6 / 15 kN / 250 kN | – |
| Struts (each axle) | mechanical | 0 | 0 | – | 6 | 0 | hub ↔ tower | BOUNDED spring, damper, bump stops | 37–48,000 N/m; damper 3,100–4,200 / 13–17 kN / 150–180 kN | – |
| Wheel (rim; tyre generated) | mechanical | 2 axle nodes each | 10 each | 5 | 1 | 0 | hub | 8–9 NORMAL | 6e6 / 168 kN / 168 kN (rear 280 kN) | 1 of 8–9 per wheel |
| Brakes, badges, plates, gauges, skins, ECU, final drive | visual or device only | 0 | 0 | – | 0 | 0 | – | – | – | – |

**What attaches to what** (the slot tree, all five cars):

- The root part holds only slots for the body and the paint.
- **Everything hangs off the body part's slots:** closures, fenders, bumpers,
  lights, glass, seats, fuel tank, engine, radiator, both suspensions.
- **The engine part's slots** hold its mounts, oil pan, intake, exhaust, ECU
  and the transmission (and through it the flywheel or converter).
- **Each suspension part's slots** hold its brakes, wheel, wheel data, struts,
  anti-roll bar, and (rear) the differential with driveshaft, half-shafts and
  final drive, or (front) the steering.
- **A door's slots** hold its inner panel, glass and mirror; a bumper's hold
  plates and fog lamps.
- On the body-on-frame SUV the frame is the root structure and the body one of
  its children, joined by 13 body-mount break groups (§4.5).

Structural: body (and frame), roof. Closures: doors, hood, boot lid.
Bolt-on: fenders, bumpers, skirts, lights, mirrors. Mechanical: engine,
transmission, differential, exhaust, tank, radiator, subframes with arms and
hubs, steering, wheels. Interior: seats, dash, steering wheel (a prop).

---

## 3. The unibody

All **[Measured]** unless marked (`--only body`, `grade`, `brace`, `summary`).

### 3.1 The lattice in numbers

| Shell | ETK | CCF | Vertex | SUV (body + frame) |
|---|---|---|---|---|
| Nodes / kg | 124 / 515 | 138 / 426 | 118 / 455 | 148 / 875 |
| Node mass p10 / p50 / p90 / max (kg) | 2.6 / 4.0 / 5.5 / 15 | 1.5 / 3.0 / 5.0 / 7 | 1.4 / 3.7 / 7.0 / 15 | 2.2 / 3.3 / 13.3 / 13.3 |
| Distinct node masses | 16 | 11 | 18 | 17 |
| Nearest-neighbour spacing p10 / p50 / p90 (m) | 0.11 / 0.18 / 0.29 | 0.09 / 0.16 / 0.22 | 0.12 / 0.22 / 0.39 | 0.09 / 0.21 / 0.38 |
| Nodes on the centre line | 11 | 16 | 13 | 24 |
| Collision off / self-collision on | 2 / 114 | 13 / 121 | 7 / 86 | 17 / 114 |
| Shell beams / per node | 624 / 5.0 | 805 / 5.8 | 601 / 5.1 | 720 / 4.9 |
| Neighbours per node p10 / p50 / p90 / min | 6 / 9 / 14 / 5 | 7 / 11 / 17 / 5 | 5 / 9 / 15 / 2 | 5 / 9 / 15 / 2 |
| Beam length p10 / p50 / p90 (m) | 0.22 / 0.44 / 0.78 | 0.22 / 0.52 / 0.94 | 0.25 / 0.58 / 0.91 | 0.29 / 0.65 / 1.30 |
| Edge / diagonal / skip beams (%) | 48 / 39 / 13 | 33 / 39 / 28 | 49 / 38 / 12 | 32 / 41 / 28 |
| Beams crossing the centre plane (%) | 10 | 8 | 8 | 16 |
| Beam triangles / non-flat tetrahedra | 947 / 517 | 1,149 / 524 | 972 / 627 | 991 / 354 |
| Nodes in at least one tetrahedron | 123 of 124 | 131 of 138 | 114 of 118 | 138 of 148 |
| Nodes braced in one plane only | 0 | 0 | 12 | 8 |
| Beams over 0.9 m through the cabin volume | 6 | 23 | 6 | 34 |
| FLT_MAX strength (%) | 99 | 98 | 97 | 88 |
| `deformLimitExpansion` set (%) / values | 53 / 1.2, 1.1 | 51 / 1.1, 1.2 | 44 / 1.1 | 39 / 1.2, 1.1 |
| Distinct (k, c, deform, strength) settings | 49 | 77 | 72 | 61 |
| k p10 / p50 / p90 (N/m) | 1.5e6 / 1.6e6 / 7e6 | 8e5 / 1.5e6 / 3e6 | 5e5 / 3e6 / 4e6 | 5e5 / 1.9e6 / 1.1e7 |
| c p10 / p50 / p90 | 70 / 120 / 200 | 100 / 130 / 175 | 110 / 130 / 200 | 115 / 141 / 705 |
| deform p10 / p50 / p90 (N) | 7,800 / 12,500 / 32,000 | 2,000 / 10,000 / 28,000 | 5,000 / 12,500 / 27,600 | 4,100 / 14,000 / 35,000 |
| SUPPORT beams touching the shell / inside it | 233 / 0 | 149 / 4 | 356 / 0 | 285 / 56 ⁴ |
| Torsion bars defined by the shell | 20 | 10 | 4 | 14 |
| Collision triangles on shell nodes / shell nodes on one | 258 / 97 % | 217 / 88 % | 192 / 91 % | 147 / 70 % |
| Node to body-mesh surface p50 / p90 / max (cm) | 0.9 / 2.8 / 10 | 0.5 / 3.8 / 23 | 1.2 / 5.9 / 13 ⁵ | 1.6 / 8.5 / 48 |
| Nodes within 2 cm / 5 cm of the surface | 81 % / 94 % | 78 % / 98 % | 70 % / 86 % | 55 % / 80 % |

⁴ The SUV's 56 are body-to-frame rests. ⁵ Against all meshes bound to the
shell; against the main body mesh alone the Vertex reads 1.7 / 19.6 / 56 (its
quarter panels are separate meshes).

Definitions: an **edge** beam is at most 1.3 × the local spacing (the mean of
its ends' third-nearest-neighbour distances), a **diagonal** 1.3–2.2 ×, a
**skip** beam longer. A tetrahedron counts when its volume is over 0.2 % of its
longest edge cubed. "Braced in one plane" is a smallest eigenvalue share under
0.02 of the node's beam-direction covariance.

**Reading it:**

- **Nodes are on the visible body.** Half are within about 1 cm of the mesh.
  The far ones are the dash ballast, pillar rigidifiers and a tunnel node.
- **The shell is rigid by topology.** A pin-jointed frame needs 3 beams per
  node; these have 5–6, arranged so nearly every node is a corner of a
  tetrahedron. That is 1.7–1.9 × the minimum.
- **Skip beams are the long-range stiffness:** 12–28 % of beams run past a
  neighbour, at k up to 5.5e6 on the ETK, with the lowest yield strain
  (0.35–0.4 %).
- **Few beams span the cabin** (6 on the ETK and the Vertex), as the old
  construction guide asks. The CCF and the SUV use more (23, 34).
- **No SUPPORT inside a unibody.** All SUPPORT beams run from the shell to
  another part.
- **The shell does not break:** 97–99 % FLT_MAX. The few finite beams are
  glass triggers.

### 3.2 Where the nodes sit (ETK, 124 nodes)

The author's node rows map onto the structure as follows. Positions are given
as stations and descriptions, not coordinates.

| Location | Nodes | Node mass (kg) | Notes |
|---|---|---|---|
| Bumper beam, lower and upper edge (s −0.23) | 3 + 3 | 4.0 | three across at two heights; also the tow hook and hood latch |
| Radiator support, top (s −0.15) | 3 | 4.0 | three across; carries the radiator |
| Front rails at the subframe's front (s −0.08) | 2 | 5.0 | anti-roll bar mounts |
| Upper-arm and subframe mounts on the rail (s −0.05, 0.05) | 4 | 6.0 | two per side |
| Strut towers (s 0.04) | 2 | 8.5 | spring, damper and strut brace land here |
| Upper fender rail (s 0.02) | 2 | 3.5 | fender and hood hinge mounts |
| Toe board and lower A-pillar (s 0.21–0.23) | 4 + 4 | 3.5, 4.5 | floor row 1 and the row above it |
| Cowl / top of firewall (s 0.26–0.35) | 4 | 4.2 | windscreen base, dash, strut brace |
| Dash ballast (s 0.40) | 4 | 15 | two heights; bound to the dash mesh |
| Floor rows 2–5 (s 0.35, 0.54, 0.74, 0.84) | 4 each | 4.5 | four across: two sills, two tunnel sides, at floor height |
| Sill tops and tunnel top over rows 2–5 | 3–4 each | 3.5 | 0.11–0.22 m above the floor row: the floor is a double layer |
| A-pillar, upper (s 0.43–0.46) | 6 | 1.5 | two of them stand 8–10 cm off the mesh |
| Windscreen header (s 0.50–0.55) | 3 | 1.7 | three across |
| Door latch posts (s 0.77) | 2 | 2.6 | B-post latch tags |
| Quarter panels, lower and upper rows (s 0.79–1.21) | 10 + 10 | 2.8 | five stations per side at two heights |
| Rear rails / boot floor, lower (s 0.92, 1.08, 1.24) | 2, 2, 3 | 5.5, 5.5, 4.6 | subframe mounts on the first two |
| Rear rails, upper (s 0.99, 1.09, 1.28) | 2, 2, 3 | 5.5, 5.5, 4.6 | with the row below, a boxed rail |
| Rear strut towers (s 1.00) | 2 | 5.5 | |
| Parcel shelf and boot opening (s 1.00–1.27) | 8 | 2.8 | lid hinge and seal rests |
| Tail panel top (s 1.28) and latch | 3 + 1 | 2.6, 3.1 | |

Per station and height band (nodes / kg):

| Station | Floor band | Sill to belt | Belt to roof | Roof | All |
|---|---|---|---|---|---|
| Nose | 3 / 12 | 3 / 12 | – | – | 6 / 24 |
| Front axle, towers | 4 / 22 | 9 / 48 | – | – | 13 / 70 |
| Firewall, A-pillar | 6 / 25 | 2 / 7 | 2 / 8 | – | 10 / 40 |
| Cabin front | 12 / 50 | 5 / 40 | 10 / 47 | 3 / 5 | 30 / 142 |
| Cabin rear | 12 / 50 | 9 / 27 | 4 / 11 | – | 25 / 89 |
| Rear axle, towers | – | 14 / 66 | 8 / 22 | – | 22 / 89 |
| Tail | – | 12 / 44 | 6 / 17 | – | 18 / 61 |

Across cars:

- **Lateral lanes** (|x| over the half width): centre 11–16 nodes, inner
  (tunnel, rails) 28–31, rail lane 32–44, side (sills, quarters) 38–50. That
  is four to six nodes across at the floor, as the tutorial's "3–4 across"
  plus a centre line.
- **Member spacing:** floor rows are 0.2–0.5 m apart along the sill (ETK
  0.28, 0.46, 0.48, 0.22 m); quarter-panel stations about 0.25 m; rails
  0.25–0.4 m.
- **Roofs are sparse.** The Vertex, a four-door sedan, has 4 shell nodes in
  the roof band and 16 between belt and roof; the CCF's windscreen frame has
  6 + 7. Roof nodes weigh 1.4–2.0 kg.
- **The floor is a double layer** on the ETK (rows 0.11–0.22 m apart
  vertically), which is what the construction guide describes. The CCF and
  the Vertex do the same with sill-top and tunnel-top rows.

### 3.3 Node mass by location

| Location | ETK | CCF | Vertex | SUV |
|---|---|---|---|---|
| Shell median | 4.0 | 3.0 | 3.7 | 3.3 (body 2.4, frame 12.9) |
| Nodes the suspension or subframe lands on, median | 5.5 | 4.0 | 6.7 | 12.9 |
| Floor band, cabin, median | 4.5 | 4.0 | 7.0 | – |
| Belt to roof, median | 1.5–2.8 | 1.9–4.0 | 1.6–2.2 | – |
| Heaviest shell node | 15 (dash ballast) | 7 | 15 | 13.3 |

**[Hypothesis]** Mass follows load: the nodes that take suspension, subframe
and engine-mount forces are the heaviest, because a stiff mount beam needs a
heavy node at each end to stay under the step's bound (§7). Roof and pillar
nodes are light because little stiffness is asked of them.

### 3.4 Crush-path grading

Shell-internal NORMAL beams by the station of their midpoint: k p50 (N/m) /
deform p50 (N) / ε_y p50 (%) / **Σ deform of the longitudinal beams** (those
within 45° of the car's axis; N). The last is the plastic force the shell
offers along the crush path at that station.

| Station | ETK | CCF | Vertex | SUV |
|---|---|---|---|---|
| Nose | 7e6 / 13,500 / 0.67 / – | 3e6 / 4,500 / 0.36 / 18 k | 5e5 / 9,500 / 3.7 / 17 k | 1.5e6 / 9,000 / 1.1 / 70 k |
| Front axle, towers | 7e6 / 8,000 / 0.38 / 0.26 M | 3e6 / 4,000 / 0.34 / 0.22 M | 3e6 / 8,000 / 0.52 / 0.53 M | 3.5e6 / 9,000 / 0.52 / 0.16 M |
| Firewall, A-pillar | 1.6e6 / 24,000 / 2.0 / 0.67 M | 2e6 / 16,500 / 1.1 / 0.91 M | 3e6 / 14,000 / 0.59 / 0.32 M | 3.5e6 / 18,800 / 0.83 / 0.81 M |
| Cabin front | 1.6e6 / 17,000 / 2.1 / 1.2 M | 1.5e6 / 15,000 / 1.9 / 1.1 M | 3e6 / 15,300 / 0.97 / 0.33 M | 1.8e6 / 17,800 / 0.79 / 2.2 M |
| Cabin rear | 1.6e6 / 24,000 / 2.4 / 1.3 M | 1.2e6 / 15,500 / 3.1 / 1.2 M | 1.5e6 / 15,000 / 1.0 / 0.46 M | 1.7e6 / 14,000 / 0.82 / 1.0 M |
| Rear axle, towers | 4.5e6 / 10,500 / 0.91 / 0.42 M | 1.1e6 / 10,000 / 0.95 / 0.82 M | 1.5e6 / 14,000 / 1.2 / 1.0 M | 1.9e6 / 14,000 / 0.98 / 1.1 M |
| Tail | 4.5e6 / 12,000 / 0.81 / 0.42 M | 1.7e6 / 10,000 / 0.94 / 0.25 M | 4e6 / 12,000 / 0.90 / 0.42 M | 1.9e6 / 14,500 / 1.0 / 2.0 M |

- **The grading is in `beamDeform`, and it is reversed against `beamSpring`.**
  The ETK's front and rear beams are 3–4.4 × stiffer than its cabin's, and
  yield at a third to a half of the force. Stiff and weak: they hold their
  shape while driving and fold early in a crash. The cabin is soft and strong.
- **Yield strain follows:** 0.34–0.7 % in the front, 1–3 % in the cabin.
- **The cabin offers 4–5 × the front's longitudinal yield** on the ETK and the
  CCF (1.1–1.3 MN against 0.22–0.26 MN). The Vertex is flatter (its cabin
  0.33–0.46 MN).
- **The ETK's strongest beams** (50,000–65,000 N, k 2.1e6, c 400) number 20,
  all in the cabin.
- **Stretch is capped where crush happens:** `deformLimitExpansion` is on
  64–78 % of front beams (1.2 on the ETK) against 37–55 % in the cabin (1.1).
- **Damping is nearly flat:** c 110–160 everywhere on the ETK shell.

**Front load path by height** (every NORMAL beam of any part ahead of s 0.2
that runs along the car; Σ deform):

| Height of the beam's midpoint | ETK | CCF | SUV |
|---|---|---|---|
| Low (under 0.45 m: subframe, rails, engine) | 7.4 MN (96 beams) | 4.4 MN (165) | 8.1 MN (142) |
| Middle (0.45–0.75 m: fenders, upper rails, hood edge) | 4.0 MN (131) | 3.5 MN (162) | 2.7 MN (108) |
| High (over 0.75 m: hood) | 0.4 MN (13) | 0.2 MN (22) | 2.8 MN (214) |

**The primary frontal load path is low:** at rail, subframe and engine
height, below the centre of mass (ETK 0.43 m, CCF 0.46 m). The hood band
carries 3–6 % of it on the cars. (The Vertex's model is offset vertically and
is not banded here.) This bears on our nose-lift bug (known-bugs S2), where
our load went through the upper band.

### 3.5 Bracing patterns

- **Every rectangle has both diagonals;** boxed members (double floor, rails
  at two heights, quarter panels at two heights) are closed into tetrahedra.
- **Skip beams** run along rails, sills and across the floor, past one node.
- **Cross-car beams:** 51–68 per unibody (8–10 %), median length 0.8–0.9 m:
  the floor rows, the cowl, the radiator support, the shelf and the tail
  panel.
- **Torsion bars across the floor** (ETK 20, CCF 10): four-node torsion
  springs (5–8e5 N·m/rad, deform 7,000–8,000, strength 15–20 kN on the ETK)
  along the floor rows. They resist the floor twisting where a beam cannot.
  **[Hypothesis]** They stand in for the bending stiffness of the floor pan.
- **Rigidifier nodes in the shell** are few: 2 non-colliding nodes on the ETK
  (A-pillar), 7 on the Vertex, 13 on the CCF.
- **`deformLimit` and `deformLimitStress` are not used** by any of the five
  cars.

### 3.6 SUPPORT placement

SUPPORT beams from the shell to another class (count):

| To | ETK | CCF | Vertex | SUV |
|---|---|---|---|---|
| Engine | 38 | 38 | 56 | 36 |
| Doors | 10 | 36 | 66 | 32 |
| Hood | 20 | 20 | 28 | 16 |
| Boot lid | 17 | 8 | 32 | 16 |
| Bumpers | 21 | 4 | 78 | 6 |
| Fenders | 8 | 10 | 28 | – |
| Lights | 28 | – | 14 | 32 |
| Exhaust | 27 | 4 | – | 31 |
| Roof / top | 28 | 7 | – | – |
| Suspension, wheels | 4 | – | 18 | 14 |
| Interior | 8 | – | 16 | 14 |

- **Engine stops are the largest single use:** 36–56 beams from the engine's
  nodes to the body, and a few more to the subframe (42–78 in all; k 1–5e6,
  deform 20–50 kN on the ETK, median length 0.8–1.0 m). **Nearly all are
  precompressed, 0.5–0.95** (ETK 42 of 42, CCF 42 of 46, Vertex 76 of 78, SUV
  44 of 44), so they are slack at rest. They let the engine move on its
  mounts but not pass through the firewall, rails or hood.
- **Precompression** is set on 36–46 % of SUPPORT beams. Common values are 0.9
  (most), 0.85, 0.7 and 0.5: the beam is inactive until it has shortened by
  10–50 %.
- 62–83 % of SUPPORT beams have a finite strength, and `beamLongBound` on
  them is mostly 2 or 4 (they break when stretched to 3–5 × their length, when
  the part has left).
- Whole car: SUPPORT is 6–15 % of beams (ETK 11 %), BOUNDED 1.7–2.4 %.

### 3.7 The body-on-frame contrast (SUV, trailer)

- The SUV's frame is 52 nodes of 9–13 kg (609 kg): k p50 3.8e6, 21 spring ends
  per node. Its body is 96 light nodes (2.4 kg, 266 kg).
- Body to frame: 13 break groups of about 6 beams, weakest member 140 kN,
  group sum about 1 MN, plus 56 SUPPORT rests.
- The trailer's frame is 113 nodes of 4–17 kg, beams 0.5–2 m long at k 4e6
  and deform 80–600 kN, 7 beams per node, 48 torsion bars, no
  `deformLimitExpansion`. 97 % of its beams cannot break.
- **[Hypothesis]** A ladder frame concentrates mass in a few heavy nodes and so
  can run stiffer beams than a unibody of the same weight; the body on top can
  then be light and soft.

---

## 4. Mechanical parts and the wheel's load path

All **[Measured]** (`--only mech`, `wheelpath`, `attach`).

### 4.1 Classes across cars

Nodes / kg / heaviest node; beams to other classes / their median finite
strength.

| Class | ETK | CCF | Vertex | SUV |
|---|---|---|---|---|
| Engine | 13 / 187 / 22; 95 / 50 kN | 24 / 220 / 25; 139 / 15 kN | 23 / 128 / 12; 147 / 15 kN | 22 / 245 / 28; 112 / 16 kN |
| Driveline | 14 / 68 / 22; 65 / 160 kN | 17 / 61 / 20; 89 / 90 kN | 3 / 27 / 9; 18 / 60 kN | 5 / 121 / 38; 47 / 11 kN |
| Exhaust | 15 / 21 / 1.4; 59 / 65 kN | 21 / 21 / 1; 72 / 11 kN | (in engine) | 20 / 37 / 2; 110 / 24 kN |
| Fuel tank | 9 / 42 / 4.6; 30 / 35 kN | 9 / 38 / 4; 45 / – | 6 / 40 / 7; 22 / – | 13 / 76 / 8; 42 / 93 kN |
| Radiator | 2 / 5 / 2.7; 17 / 3.6 kN | 7 / 9 / 2; 40 / 10 kN | 2 / 14 / 7; 64 / 16 kN | 2 / 8 / 4; 12 / – |
| Suspension, subframes | 47 / 267 / 8; 204 / 280 kN | 55 / 239 / 8; 216 / 150 kN | 50 / 226 / 7; 197 / 180 kN | 25 / 214 / 14; 105 / 350 kN |
| Wheel axle nodes | 8 / 40 / 5 | 8 / 40 / 5 | (in suspension) 4.5 | 8 / 46 / 5.8 |
| Interior | 14 / 34 / 5 | 35 / 100 / 10 | 19 / 51 / 8 | 45 / 164 / 10 |

### 4.2 Engine, transmission, driveline (ETK in detail)

- **Engine block: 8 corner nodes of 22 kg** and 2 light nodes, 40 beams among
  the 10 (a box with every diagonal) at k 2e7, deform 250 kN, unbreakable. The
  beams are in a deform group with a trigger ratio of 0.001: any real
  deformation of the block damages the engine. All four cars have 8 heavy
  block nodes (12, 22, 25 and 28 kg).
- **Two engine mounts:** one mount node per side on the block, each held by 3
  BOUNDED beams to the **front subframe** (k 32,000, c 600, deform 20 kN,
  strength 90 kN), one break group per mount. Soft and few, as the guide says.
- **Transmission:** one 22 kg node behind the block, 24 beams to the engine
  (k 4.8e6, deform 27 kN, unbreakable) and **one mount of 9 BOUNDED beams to
  the body** (k 1.4e5, c 2,700, deform 5,000, strength 160 kN). The gear lever
  is 10 light nodes (0.3–0.5 kg).
- **Engine stops:** 38 SUPPORT beams block-to-body and 4 block-to-subframe
  (§3.6).
- **Driveshaft:** no nodes. One BOUNDED beam without spring from the
  transmission node to the differential, deform 1,500, strength 3,500, with a
  deform group that hides the shaft's prop when it fires. The torque path is
  the powertrain device tree, not the beam.
- **Differential:** 3 nodes of 14 kg, 24 beams to the rear subframe
  (k 4.8e6, unbreakable).
- **Half-shafts:** one BOUNDED beam each from a differential node to the
  wheel's axle node, strength 9,500; they are the wheel's `axleBeams`, so
  breaking one cuts drive to that wheel.
- **Exhaust:** 15 nodes of 1.4 kg in a chain; 28 soft NORMAL hangers and 27
  SUPPORT rests to the body (k 47,000, deform 4,000, strength 15 kN) in three
  break groups, and 4 beams to the engine at 65 kN.
- **Fuel tank:** 9 nodes of 4.6 kg (the fuel's mass is shared over them), 30
  very soft beams to the body (k 30,000, c 500, deform 5,000, strength 35 kN),
  no internal beams: the tank is carried entirely by the body, in a passive
  break group.
- **Radiator:** 2 nodes of 2.7 kg, 12 unbreakable beams to the radiator
  support, in a deform group (trigger 0.05) that is the coolant-leak trigger;
  5 hose beams to the engine at 3,600 N.
- **Devices:** engine, clutch, gearbox, driveshaft, differential, a torsion
  reactor on the differential, and a shaft per wheel. `engineGroup` nodes:
  8 block, 5 intake, 2 radiator, 9 fuel.

The other cars follow the same pattern: 8 block nodes of 12–28 kg, 2–4
mounts of 3–8 beams at 15–90 kN, 42–78 stops, a tank of 6–13 nodes, a radiator
of 2–7 nodes.

### 4.3 Front axle (ETK, double wishbone)

| Element | Nodes | kg | Beams | k / c / deform / strength |
|---|---|---|---|---|
| Subframe | 4 corners + 2 rack mounts + 1 centre | 7 each (centre 4.5): 46.5 | 21 among them | 5–6.5e6 / 120–180 / 80–96 kN / FLT_MAX |
| Subframe to body | lands on 4 rail mount nodes (6 kg) | – | 64 in 2 break groups (one per side) | 1.3–1.8e6 / 180 / 157 kN / 1.24 MN |
| Lower arm (per side) | **none of its own** | – | 2 beams: subframe front and rear corner to the hub's lower ball joint | 1.0e7 and 1.4e7 / 2,000 / 155 kN / FLT_MAX |
| Lower-arm stiffeners (per side) | 2 (the damper's lower mount above the arm, and a node below it) | 4.5, 4.0 | 7 | 1e6 / 2,000 / 155 kN / FLT_MAX |
| Upper arm (per side) | none of its own | – | 2 beams: **body** rail mount nodes to the hub's upper ball joint | 1.0e7 and 1.4e7 / 2,000 / 155 kN / FLT_MAX |
| Hub / knuckle (per side) | 4: lower joint, upper joint, steering arm, a fourth for rigidity | 4.8, 4.0, 4.5, 3.8: 17.1 | 6 (a full tetrahedron) | 6e6 / 100 / 129 kN / FLT_MAX |
| Wheel axle nodes (per side) | 2, on the wheel's axis 0.27 m apart | 5 each | 1 between them; 8 to the hub's four nodes | 6–6.5e6 (two at 1e6) / 100 / 168 kN / **168 kN**, break group per wheel |
| Spring, damper, bump stops (per side) | none | – | 4 BOUNDED beams: hub lower joint to the strut tower | spring 37,000 N/m; damper 4,200; deform 17 kN / 180 kN |
| Travel limits (per side) | none | – | 4 SUPPORT (precompression 0.82, 0.9) and 4 damping-only BOUNDED, hub to the rail mounts and to the subframe | 1e6 / 250 / 200 kN / FLT_MAX |
| Steering | 2 rack-end nodes, 5 kg | 10 | 2 tie rods to the hubs' steering arms; 2 hydros; the rack mounts ride a slide rail | tie rod 1.7e7 / 100 / 85 kN / 250 kN |
| Anti-roll bar | 2 | 3 | 6, and torsion bars | 2.5e6 / 100 / 15 kN / 250 kN |

Totals: 19 nodes, 97.6 kg, 145 beams defined, 84 beams to the body.

The rear axle is the same scheme with 24 nodes and 164 kg: a subframe carrying
the differential, hubs of 4 nodes at 6–8 kg, arms as beams, four subframe
mount groups of 10 beams (80 kN and 150 kN per beam; a few at 540 kN).

### 4.4 The load path of a wheel hit

Tyre → **2 axle nodes** → (8 beams, 168 kN, one break group) → **hub
tetrahedron (4 nodes, 17 kg)** → three ways into the car:

1. lower arm beams (155 kN yield, unbreakable) → subframe corners (7 kg) →
   64 mount beams (1.24 MN) → rail mount nodes (6 kg);
2. upper arm beams (155 kN yield, unbreakable) → rail mount nodes directly;
3. spring and damper (180 kN) → strut tower node (8.5 kg).

Plus the tie rod (85 kN yield, 250 kN break) into the rack.

| | ETK | CCF | Vertex | SUV |
|---|---|---|---|---|
| Axle-node mass (kg) | 5.0 | 5.0 | 4.5 | 5.8 |
| Hops from an axle node to a shell node | 2 | 2 | 1 | 1–2 |
| Nodes one hop from the axle nodes / their mass (kg) | 7–10 / 30–46 | 6–7 / 36–50 | 7–8 / 33–42 | 8–9 / 65–92 |
| Beams leaving that cluster / of them to the shell | 24–46 / 3–14 | 19–31 / 5–12 | 24–37 / 9–15 | 44–72 / 12–18 |
| Wheel-to-hub break group: beams / weakest (kN) / sum (MN) | 8 / 89 ⁶ / 1.8 | 8 / 57 / 0.76 | – | 11 / 240 / 3.6 |
| Suspension ↔ shell beams / with finite strength | 136 / 128 | 114 / 112 | 133 / 73 | 82 / 20 |
| Their strength p10 / p50 / p90 (kN) | 80 / 890 / 1,240 | 50 / 1,200 / 3,000 | 85 / 110 / 900 | 240 / 500 / 1,000 |
| Shell nodes they land on / those nodes' median mass (shell median) | 32 / 5.5 (4.0) | 38 / 4.0 (3.0) | 44 / 6.7 (3.7) | 32 / 12.9 (3.3) |

⁶ Median over front (168 kN) and rear (9.5 kN half-shaft member, 280 kN wheel
beams).

**What gives first under a rising wheel load (ETK front) [Measured, by
thresholds; not simulated]:**

1. the tie rod bends (85 kN): toe changes;
2. the hub's own beams bend (129 kN), then the arms (155 kN): camber and
   position change, nothing detaches;
3. **the wheel leaves the hub** when one of its 8 beams reaches 168 kN (the
   break group takes the rest);
4. spring and damper break (180 kN);
5. the tie rod breaks (250 kN);
6. the subframe leaves the body (1.24 MN per beam, 64 beams): in practice it
   bends its mount beams (157 kN yield) long before.

The rear subframe is held far more weakly (80–150 kN per beam) and is the
part that tears off in a hard side or rear hit.

Rim: the generated hub is 16 rays, so 32 nodes of 0.40–0.42 kg (about 13 kg),
with hub beams at deform 37–44 kN and strength 67–78 kN. Hubcaps have their
own break group but are disabled on these wheels.

### 4.5 Our constraint: raycast wheels on real hubs

**[Hypothesis; the recommendation is §10.4.]** The faithful minimum is what
the ETK's structure reduces to when the generated tyre and rim nodes are
removed: per wheel, 2 axle nodes and a 4-node hub tetrahedron, arms as beam
pairs, the strut as a spring-damper pair to a tower node, one wheel break
group. Everything else in §4.3 is steering, anti-roll and travel limits, which
the raycast model already owns.

---

## 5. Closures, panels, interior, glass

### 5.1 Panels are a single layer with rigidifiers

**[Measured]** "Depth" is the node cloud's smallest extent: the rigidifier's
offset from the surface.

| Part | Nodes | kg | Depth (m) | Non-colliding nodes | Internal beams per node | Triangles |
|---|---|---|---|---|---|---|
| Door (4 cars) | 8–13 | 17.6–23.6 | 0.20–0.38 | 1–3 | 3.1–4.7 | 5–10 |
| Hood | 21–30 | 17.6–36.8 | 0.21–0.39 | 3–4 | 3.3–5.5 | 21–33 |
| Boot lid / tailgate half | 11–17 | 11.6–22 | 0.21–0.35 | 2–3 | 2.7–4.6 | 0–18 |
| Fender | 10–16 | 6.5–14.2 | 0.19–0.38 | 1–2 | 2.9–3.6 | 7–10 |
| Bumper cover | 19–28 | 9.5–25 | 0.31–0.66 | 0–2 | 3.6–4.9 | 18–30 |
| Bumper bar (CCF, Vertex, SUV) | 10 | 3–14.5 | 0.05–0.11 | 0 | 2.7 | 8 |

- **Every closure and fender carries 1–4 non-colliding nodes** set 0.2–0.4 m
  off its surface and beamed to 5–24 of its nodes. This is the documented
  rigidifier, at a smaller offset than the tutorial's 0.5 m.
- **No car has an inner skin layer.** A door's "inner panel" part is 2 nodes
  (ETK) that double as rigidifiers; the door mesh's inner and outer faces
  both ride the single layer.
- **A bumper is a cover plus, on three cars, a separate 10-node bar** behind
  it (5 across at 2 heights). The ETK instead puts 6 bumper-beam nodes in the
  body.
- Hinges, latches, seals: `beamng-authoring-census.md` §3. In short: two
  hinge break groups of 4–14 beams at 8–98 kN per beam; one coupler latch at
  30–40 kN (hood: two, latch and safety catch); 4–30 SUPPORT seals and rests.
  The coupler radius is 1.35 m with a start radius of 0.1 m and
  `couplerLock` on all cars.

### 5.2 How panels are held (break groups by the classes they join)

Groups / beams per group (median) / weakest member (median, N) / group sum
(median, N).

| Joins | ETK | CCF | Vertex | SUV |
|---|---|---|---|---|
| Body ↔ fender | 8 / 6 / 12,000 / 66 k | 5 / 7 / 25,000 / 220 k | 6 / 6 / 48,000 / 290 k | 6 / 7 / 45,000 / 320 k |
| Body ↔ bumper | 5 / 8 / 6,000 / 78 k | 5 / 8 / 12,000 / 96 k | 11 / 10 / 10,000 / 98 k | 6 / 8 / 22,500 / 120 k |
| Bumper ↔ fender | 2 / 6 / 7,550 / 45 k | – | 2 / 5 / 12,500 / 63 k | 4 / 6 / 9,150 / 60 k |
| Body ↔ door (hinges) | 4 / 11 / 19,000 / 480 k | 4 / 6 / 16,500 / 220 k | 6 / 10 / 8,000 / 450 k | 8 / 9 / 54,600 / 950 k |
| Body ↔ hood | 2 / 7 / 10,000 / 530 k | 4 / 3 / 82,500 / 460 k | 4 / 8 / 12,750 / 190 k | 2 / 6 / 10,000 / 870 k |
| Body ↔ boot lid | 2 / 6 / 9,000 / 340 k | 2 / 7 / 30,000 / 240 k | 2 / 9 / 10,000 / 280 k | 1 / 30 / 34,000 / 3.8 M |
| Body ↔ lights | 4 / 14 / 20,000 / 300 k | 4 / 18 / 11,000 / 270 k | 9 / 4 / 20,500 / 90 k | 5 / 6 / 17,500 / 100 k |
| Body ↔ trim (skirts) | 8 / 8 / 12,000 / 96 k | 5 / 15 / 22,500 / 240 k | 4 / 10 / 14,000 / 140 k | 9 / 7 / 22,000 / 130 k |
| Door ↔ mirror | 2 / 22 / 4,000 / 310 k | 2 / 22 / 4,000 / 310 k | 2 / 14 / 5,800 / 81 k | 2 / 12 / 15,300 / 150 k |
| Door ↔ glass | 2 / 12 / 4,000 / 8 k | 2 / 2 / 3,000 / 6 k | 6 / 8 / 6,000 / 48 k | 3 / 9 / 9,100 / 82 k |
| Body ↔ exhaust | 3 / 6 / 15,000 / 90 k | 4 / 15 / 11,000 / 160 k | – | 1 / 18 / 23,500 / 420 k |
| Engine ↔ subframe (mounts) | 2 / 3 / 90,000 / 270 k | 2 / 5 / 90,000 / 450 k | 1 / 5 / 15,000 / 75 k | – |
| Body ↔ suspension (subframe mounts) | 6 / 10 / 150 k / 2.3 M | – | 5 / 7 / 90 k / 630 k | – |
| Wheel ↔ hub | 4 / 8 / 89 k / 1.8 M | 4 / 8 / 57 k / 760 k | – | 2 / 11 / 240 k / 3.6 M |
| Body ↔ frame | – | – | – | 13 / 6 / 140 k / 960 k |

- **A panel is split into several small groups,** so it tears off a corner at
  a time: a fender has 4–8 groups of 5–7 beams, a skirt 4, a bumper 3–5 (left,
  middle, right, and two to the fenders).
- **Strength ladder** (weakest member): mirror and door glass 3–6 kN < bumper
  6–22 kN < fender 12–48 kN ≈ lights ≈ skirts < hinges 8–55 kN (group sum
  0.2–0.95 MN) < exhaust < engine mounts 15–90 kN < subframe mounts 90–150 kN
  (front ETK 1.24 MN) < wheel 57–240 kN.
- 3–10 % of grouped beams are passive (`breakGroupType` 1): SUPPORT rests and
  tank straps that should go with the part but never trigger its release.
- **Strength below deform** occurs on 46–336 beams per car: brittle beams
  (glass triggers, panel clips, hoses) that snap without bending.

### 5.3 Interior

**[Measured]**

| | ETK | CCF | Vertex | SUV |
|---|---|---|---|---|
| Front seat: nodes / kg / beams to body / breakable | 6 / 12 / 26 / none | 9 / 18 / 40 / 120 kN (one side all 44, the other 4) | 6 / 12 / 19 / none | 8 / 24 / 43 / none |
| Rear seat | – | – | 2 / 16 / 12 / none | 9 / 21 / 40 / none |
| Dash | 4 × 15 kg in the body | 5 × 10 kg, 35 beams to body, unbreakable | no dash part (the body's heaviest node is 15 kg) | 8 × 10 kg, 28 beams to body, unbreakable |
| Gear lever | 10 light nodes | 10 light nodes | – | 10 light nodes |
| Steering wheel | a prop | a prop | a prop | a prop |

- Seat-to-body beams are soft and weak in yield (k 2.8e5, deform 6,000), so a
  seat shifts and stays bent, but on three of four cars it can never come
  loose.
- The dash is ballast: 40–80 kg on 4–8 heavy nodes beamed to the cowl and
  floor. There is no cross-car beam, console structure or steering column.
- Occupants are not modelled.

### 5.4 Glass and lamps

**[Measured]** (`--only glass`, `summary`)

| | ETK | CCF | Vertex | SUV |
|---|---|---|---|---|
| Deform groups / on glass | 18 / 7 | 33 / 6 | 41 / 17 | 42 / 12 |
| Beams in any deform group (% of all) | 10 | 17 | 19 | 17 |
| Beams per glass group, median (min–max) | 12 (5–62) | 29 (16–33) | 20 (4–62) | 13 (4–28) |
| Trigger ratios on glass (× beams) | 0.1 × 62, 0.005 × 46, 0.02 × 24, 0.03 × 16, 0.0001 × 10 | 0.05 × 100, 0.0001 × 42, 0.02 × 24 | 0.01 × 184, 0.05 × 120, 0.02 × 32, 0.03 × 31 | 0.05 × 84, 0.02 × 37, 0.04 × 24, 0.01 × 16 |
| Glass beams with finite strength / median strength | 45 % / 26 kN | 57 % / 12 kN | 60 % / 18.5 kN | 89 % / 12 kN |
| Glass beams also in a break group | 35 % | 41 % | 34 % | 58 % |
| Glass beams k / deform (median) | 1.5e6 / 7,400 | 7e5 / 11,000 | 7e5 / 9,150 | 7e5 / 10,000 |
| Nodes owned by glass parts / kg | 3 / 15 | 8 / 8 | 41 / 108 | 56 / 142 |
| Flexbody rows with a deform group | 17 | 82 | 52 | 58 |

Two ways to build a pane:

- **ETK, CCF: the pane is a set of trigger beams across the opening.** The
  windscreen is one 12 kg centre node and 62 beams among 18 body nodes around
  the aperture (46 at trigger 0.005, 16 at 0.03); only 5 can break. A door
  glass is one 1.7 kg node and 12 beams (trigger 0.02; two break at 4 kN and
  release particles).
- **Vertex, SUV: the pane has its own nodes** (the Vertex 41 nodes, 108 kg of
  glass), held by breakable beams, so a pane stiffens the greenhouse until it
  shatters.

Either way, when the group fires: the flexbody's material swaps to the
damaged glass, props in the group are hidden, and where a break group is
shared the beams break and the pane stops carrying load. No car has glass
fragments as bodies.

Lamps use the same mechanism with 22–31 beams between the lamp's 5–6 nodes
and 4–5 body nodes (trigger 0.08–0.1, strength 15–26 kN), swapping lens and
lamp materials and switching off their light props.

The same field also carries **mechanical damage triggers** with no visual:
engine block (trigger 0.001), oil pan (0.005), intake (0.01), radiator (0.05),
hoses (0.01), driveshaft and half-shafts.

