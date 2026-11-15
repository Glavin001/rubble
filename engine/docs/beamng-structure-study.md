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
   node, 9–11 neighbours per node, 93–99 % of nodes a corner of a tetrahedron, 12–28 %
   of beams skipping a node, 8–10 % crossing the car. No SUPPORT beams inside
   it (0–4 of 600–800).
4. **Heavy nodes where loads enter:** strut towers 8.5 kg, subframe mounts 6 kg,
   rails 5–5.5 kg, floor 4.5 kg, quarter panels 2.8 kg, pillars and roof
   1.5–1.7 kg (ETK). The nodes suspension lands on are 1.3–4× the shell's
   median.
5. **The crush path is graded by yield force, not by spring.** Front and rear
   beams are the stiffest (k 3–7e6 N/m) with the lowest `beamDeform`
   (4–13 kN); cabin beams are softer (1.5e6) with the highest (15–32 kN, up to
   65 kN). On the ETK and the CCF the shell's longitudinal yield sums to
   0.2–0.3 MN in the front, 0.7–0.9 MN at the firewall and 1.1–1.3 MN in the
   cabin.
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
9. **6–15 % of beams are SUPPORT (ETK 11 %), all between parts:** engine to
   body (36–56 per car), closures to their openings, bumpers, lamps, exhaust. 36–46 % of them
   are precompressed.
10. **Mechanical parts are few heavy nodes:** an engine is 8 block nodes of
    12–28 kg, cross-braced (ETK k 2e7); a hub is 4 nodes of 4–5 kg in a
    tetrahedron; a suspension arm is two beams, not a part with nodes; the
    driveshaft is one breakable beam.
11. **A wheel reaches the shell in two hops** (axle nodes → hub tetrahedron →
    arms and strut → body), and the weakest link is the wheel-to-hub group
    (ETK: 8 beams at 168 kN), then the spring and damper (180 kN), the tie rod
    (250 kN), then the subframe mounts (1.2 MN).
12. **Glass is a deform group, not a body:** 4–62 beams across the opening with
    a trigger ratio of 0.005–0.1, a mesh material swap, and (ETK, CCF) one
    centre node carrying the pane's mass.
13. **Flexbodies bind to one node group (58–76 % of rows),** each vertex to 3–4
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
    breakable 8-beam wheel group, and 24 lint rules, each traced to a number
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

---

## 6. Flexbody binding

All **[Measured]** (`--only flex`, `summary`) except where a doc is cited.

### 6.1 Groups

| | ETK | CCF | Vertex | SUV | Trailer |
|---|---|---|---|---|---|
| Flexbody rows / with a mesh found / props | 116 / 98 / 30 | 242 / 218 / 26 | 179 / 158 / 25 | 227 / 201 / 13 | 120 / 55 / 0 |
| Rows naming 1 / 2 / 3 or more node groups | 83 / 33 / 0 | 140 / 47 / 55 | 133 / 34 / 12 | 173 / 42 / 12 | 74 / 46 / 0 |
| Node groups | 61 | 85 | 73 | 88 | 63 |
| Nodes that belong to two or more groups | 24 % | 23 % | 21 % | 23 % | 1 % |
| Rows with a `deformGroup` (material swap) | 17 | 82 | 52 | 58 | 39 |

- **One mesh, one group** is the rule (58–76 % of rows). A second group is
  named where a mesh spans two structures: a wheel on wheel and hub, the dash
  on dash and console, a subframe on its mounts.
- **A node carries several groups** when it is shared hardware: a strut-tower
  node is in the body's group, the engine bay's, the damper's and the strut
  brace's, so each of those meshes can use it.
- **The body mesh binds to the whole shell** (101–137 nodes). Sub-assemblies
  of the body get smaller named groups: engine bay 13 nodes, dash and console
  9, windscreen frame 12.
- Small non-detachable meshes ride the body's group (carpet, pedal box, mount
  brackets), as the tutorial allows.

### 6.2 Distances

Per class: meshes / nodes in the pool (median) / vertex to **nearest** pool
node, median over vertices (m) / to the **third nearest** (m) / the worst
vertex's nearest node (m). A vertex needs three or four nodes [D-flex], so the
third-nearest distance is the reach the binding really uses.

| Class | ETK | CCF | Vertex | SUV |
|---|---|---|---|---|
| Body | 11 / 114 / 0.15 / 0.36 / 0.41 | 1 / 129 / 0.11 / 0.24 / 0.37 | 7 / 137 / 0.12 / 0.28 / 0.46 | 2 / 101 / 0.14 / 0.36 / 0.79 |
| Door | 4 / 12 / 0.14 / 0.32 / 0.24 | 7 / 13 / 0.13 / 0.28 / 0.28 | 12 / 12 / 0.14 / 0.35 / 0.41 | 12 / 13 / 0.16 / 0.40 / 0.38 |
| Hood | 1 / 20 / 0.07 / 0.30 / 0.34 | 1 / 18 / 0.12 / 0.40 / 0.29 | 2 / 12 / 0.13 / 0.38 / 0.36 | 1 / 26 / 0.11 / 0.36 / 0.32 |
| Boot lid | 1 / 13 / 0.10 / 0.28 / 0.31 | 1 / 14 / 0.13 / 0.26 / 0.23 | 3 / 15 / 0.11 / 0.33 / 0.26 | 2 / 15 / 0.11 / 0.32 / 0.30 |
| Fender | 3 / 12 / 0.08 / 0.28 / 0.30 | 4 / 69 / 0.10 / 0.27 / 0.26 | 2 / 11 / 0.10 / 0.32 / 0.23 | 6 / 6 / 0.10 / 0.38 / 0.31 |
| Bumper | 3 / 25 / 0.08 / 0.24 / 0.20 | 6 / 23 / 0.08 / 0.23 / 0.23 | 3 / 18 / 0.11 / 0.26 / 0.28 | 5 / 26 / 0.12 / 0.30 / 0.37 |
| Lights | 8 / 6 / 0.09 / 0.17 / 0.27 | 12 / 6 / 0.09 / 0.20 / 0.24 | 14 / 7 / 0.09 / 0.22 / 0.21 | 15 / 5 / 0.09 / 0.26 / 0.30 |
| Glass | 4 / 12 / 0.14 / 0.38 / 0.39 | 2 / 13 / 0.10 / 0.31 / 0.20 | 6 / 74 / 0.18 / 0.47 / 0.38 | 9 / 13 / 0.12 / 0.36 / 0.51 |
| Interior | 7 / 26 / 0.16 / 0.37 / 0.44 | 16 / 14 / 0.13 / 0.31 / 0.40 | 17 / 29 / 0.16 / 0.45 / 0.50 | 16 / 10 / 0.17 / 0.43 / 0.77 |
| Suspension | 17 / 6 / 0.05 / 0.27 / 0.46 | 25 / 4 / 0.06 / 0.33 / 0.46 | 10 / 8 / 0.10 / 0.32 / 0.31 | 15 / 13 / 0.12 / 0.37 / 0.56 |
| Engine | 4 / 8 / 0.21 / 0.43 / 0.40 | 6 / 12 / 0.12 / 0.25 / 0.39 | 12 / 9 / 0.16 / 0.33 / 0.41 | 11 / 8 / 0.31 / 0.54 / 0.69 |
| Exhaust | 2 / 14 / 0.11 / 0.29 / 0.27 | 2 / 25 / 0.07 / 0.32 / 0.35 | 1 / 9 / 0.24 / 1.12 / 0.27 | 3 / 9 / 0.14 / 0.44 / 0.55 |

- **A vertex's nearest node is 7–16 cm away on panels** and its third nearest
  24–40 cm. BeamNG's binding reaches about a third of a metre as a matter of
  course; the worst vertices are 0.2–0.5 m from any node of their group.
- **Rigid machinery rides few nodes:** an engine mesh on 8–12, a hub on 4–6, a
  lamp on 5–7.
- **Inner and outer faces:** a door, hood or lid is one mesh (or a skin mesh
  and a trim mesh) bound to the same group, and that group is a single layer
  of nodes plus its rigidifiers (§5.1). Thickness is carried by the binding's
  local frame at each vertex, not by a second layer of nodes.
  **[Hypothesis]** That is why BeamNG panels keep their thickness when bent:
  both faces follow the same three or four nodes.
- **Visual swaps** are per mesh: a `deformGroup` on the flexbody row, one
  material per such mesh. Glass and lamp lenses are split from their frames so
  only the lens swaps.
- Node to surface, and nodes on creases: `beamng-authoring-census.md` §5.

---

## 7. Stability and stiffness

### 7.1 Every node is tuned to the same stiffness-to-mass ratio

**[Measured]** (`--only stab`, `summary`). Per node: Σk over all its spring
beams, divided by its mass (1e6 s⁻²), and ω·h = h·√(Σk/m) at h = 0.5 ms:
median Σk/m / ω·h p50 / p90 / max.

| Class | ETK | CCF | Vertex | SUV |
|---|---|---|---|---|
| Body | 9.1 / 1.51 / 2.00 / 2.21 | 8.4 / 1.45 / 1.89 / 2.12 | 9.8 / 1.56 / 1.91 / 2.68 | 9.4 / 1.53 / 2.05 / 2.68 |
| Frame | – | – | – | 7.4 / 1.36 / 1.85 / 2.06 |
| Door | 7.1 / 1.33 / 2.02 / 2.36 | 9.0 / 1.50 / 1.76 / 1.85 | 10.5 / 1.62 / 2.04 / 2.09 | 10.6 / 1.63 / 1.98 / 2.13 |
| Hood | 4.6 / 1.07 / 1.19 / 2.17 | 11.5 / 1.69 / 1.98 / 2.20 | 7.4 / 1.36 / 1.73 / 2.04 | 7.9 / 1.40 / 1.57 / 1.97 |
| Boot lid | 10.2 / 1.60 / 1.77 / 1.80 | 10.5 / 1.62 / 1.73 / 1.87 | 11.0 / 1.66 / 1.78 / 2.00 | 7.0 / 1.32 / 1.76 / 2.00 |
| Fender | 9.2 / 1.52 / 1.60 / 1.66 | 9.8 / 1.57 / 1.87 / 2.10 | 13.0 / 1.80 / 1.98 / 2.03 | 7.6 / 1.38 / 1.72 / 1.93 |
| Bumper | 6.0 / 1.22 / 1.49 / 1.72 | 6.2 / 1.25 / 1.56 / 1.86 | 5.0 / 1.12 / 1.57 / 1.92 | 3.4 / 0.92 / 1.35 / 1.97 |
| Lights | 9.3 / 1.53 / 1.92 / 1.92 | 10.2 / 1.60 / 1.77 / 1.82 | 6.9 / 1.32 / 1.54 / 1.56 | 10.7 / 1.63 / 1.88 / 1.97 |
| Suspension | 8.3 / 1.44 / 1.71 / 1.84 | 11.0 / 1.66 / 1.97 / 2.14 | 11.8 / 1.72 / 1.94 / 2.16 | 8.0 / 1.41 / 1.81 / 2.24 |
| Wheel axle nodes | 5.4 / 1.16 / 1.35 / 1.35 | 6.6 / 1.28 / 1.32 / 1.32 | – | 7.1 / 1.33 / 1.91 / 1.91 |
| Engine | 7.1 / 1.34 / 1.49 / 1.51 | 7.1 / 1.33 / 1.41 / 1.85 | 8.8 / 1.48 / 1.60 / 1.77 | 6.2 / 1.24 / 1.41 / 1.57 |
| Exhaust | 5.0 / 1.12 / 1.29 / 1.32 | 4.0 / 1.01 / 1.23 / 1.36 | – | 4.8 / 1.10 / 1.72 / 1.75 |
| Fuel tank | 0.02 / 0.07 | 0.2 / 0.24 | 0.04 / 0.10 | 0.03 / 0.09 |

- **Structure sits at Σk/m ≈ 7–11e6 s⁻²** on every car and nearly every
  class, with node masses from 0.5 kg (fenders) to 28 kg (engine). A 0.5 kg
  fender node carries Σk 4.6e6; a 22 kg engine node 1.4e8.
- **That is the edge of the step.** For a lattice of equal masses the largest
  eigenvalue is at most 2·Σk/m (Gershgorin), and the explicit step needs
  ω_max·h ≤ 2, so the safe per-node value is ω·h ≤ 1.41. The cars' medians are
  1.3–1.7 and their p90 1.6–2.0. They hold because a node's beams point in
  different directions, which the bound ignores.
- **Whole-car ω·h by power iteration** (the car's true top frequency on
  Rubble's CPU reference, springs only): **ETK 1.80, CCF 1.82, Vertex 1.80,
  SUV 1.90.** The limit is 2. This is the measured meaning of the developer's
  remark that spring values are very close to the maximum [F-goosah].
- **Soft things are deliberately far below it:** bumper covers 3–6e6, exhaust
  4–5e6, the fuel tank 0.02–0.2e6.
- **Damping:** h·Σc/m is 0.29 on the ETK's shell (median), 0.55–0.66 on its
  light bolt-on panels; per-beam ζ 0.03–0.04 on structure, 0.07–0.33 on
  bumpers (census).
- The census's "row" (the same sum with damping and unequal masses, over
  Rubble's cap) reads 1.1–1.6 at the median for these parts. Rubble's
  generated car is capped at 1.0. **Adopting BeamNG's level is an engine-wide
  setting and the user's decision** (`generated-vehicles-lessons.md` §11.6).

**The design rule that follows [Hypothesis]:** give each node a stiffness
budget Σk ≈ (6.8–9)e6 × m N/m (ω·h 1.3–1.5) and share it among its spring
ends. A 4 kg shell node with 15 ends gets a mean k of 1.8–2.4e6; a 1.5 kg door
node with 12 ends 0.85–1.1e6; a 0.5 kg fender node with 10 ends 3.4–4.5e5.
Those are the cars' actual medians (1.6e6, 1.0e6, 4e5). **Stiffness is bought
with node mass and nothing else.** Fewer, heavier nodes are stiffer over a
distance because k·L ∝ m·L/n.

### 7.2 Modes, wobble and settle

**[Measured]** `sedan_stiffness` on the CPU reference, JBeam mode, pressure
wheels, lids latched by their couplers. First mode of each part with its
neighbours held (Hz):

| Part | ETK kc6 | CCF | Vertex | SUV |
|---|---|---|---|---|
| Front door | 34.9 | 31.5–31.9 | 15.7–15.9 | 20.4–20.6 |
| Rear door | – | – | 20.5–20.6 | 15.5–26.4 |
| Hood | 12.5 | 19.2 | 11.1 | 15.1 |
| Boot lid / tailgate | 27.0 | 26.0 | 23.3 | 30.3–40.1 |
| Fender | 41.8 | 17.2–20.4 | 40.3 | 54.5–55.4 |
| Front / rear bumper cover | 23.8 / 9.2 | 16.8 / 8.5 | 19.5 / 13.4 | 14.5 / 19.1 |
| Bumper bar | – | 61–73 | 68–81 | 89 |
| Body shell alone, free, first global mode | 5.6 | 11.9 | 19.6 | 27.1 |
| Whole-car ω·h | 1.80 | 1.82 | 1.80 | 1.90 |
| SUPPORT share of loaded beams (tyres included) | 7.2 % | 3.5 % | 10.3 % | 5.1 % |

5 cm × 0.5 m bump at 30 km/h on pressure wheels: peak movement of each
category about the body's rigid fit (mm).

| | ETK kc6 | CCF | Vertex | SUV |
|---|---|---|---|---|
| Body (own ripple) | 1.37 | 0.56 | 8.4 | 0.42 |
| Doors | 1.70 | 0.74 | 10.6 | 1.02 |
| Hood | 3.44 | 1.13 | 16.1 | 0.95 |
| Boot lid | 3.61 | 1.41 | 13.3 | 0.52 |
| Fenders | 2.21 | 0.76 | 7.8 | 0.77 |
| Bumpers | 4.24 | 1.22 | 21.3 | 0.79 |
| Beams bent by the bump | 0 | 0 | 157 | 0 |

- **Baseline bands:** doors 16–35 Hz, hood 11–19 Hz, lid 23–40 Hz, fender
  17–55 Hz, bumper covers 9–24 Hz, bumper bars 60–90 Hz. Bump wobble 0.4–4 mm
  on pressure wheels. On raycast wheels the ETK reads 0.09–0.23 mm and
  settles within 0.6 s (`generated-vehicles.md`, kr6).
- **The Vertex's numbers are not a baseline.** The bump bent 157 of its beams
  on Rubble (front floor and subframe mounts), so its wobble includes
  permanent set. Its content is tuned for BeamNG's solver, not ours
  (`physics-evidence-first`). Its modes are valid; its bump row is not.
- **Body torsion is soft everywhere:** the bare shells' first global mode is
  5.6–27 Hz. Real unibodies are 25–50 Hz (general engineering knowledge, not
  from a source read here). The open ETK and CCF shells are the softest.
- After a 2 cm spawn drop the ETK's categories ripple 0.25–0.84 mm and are
  under 0.2 mm in 0.1–0.4 s (bumpers 1.3 s).

### 7.3 Why these cars are not jello

**[Hypothesis, ranked by the evidence above]**

1. **Every node is at the step's bound** (§7.1). There is no stiffness left on
   the table anywhere; ours had a stability pass removing 44–65 % of closure
   stiffness.
2. **Mass is concentrated:** 118–138 shell nodes of 3–4 kg, 10–13 door nodes
   of 1.2–1.9 kg. Half the nodes means twice the mass each, twice the k each,
   and beams twice as long: about four times the stiffness over a distance.
3. **Topology first:** tetrahedra everywhere, 12–28 % skip beams, double-layer
   floor, torsion bars along the floor.
4. **Panels get depth from a rigidifier,** 0.2–0.4 m off the surface: bending
   stiffness goes as k·d².
5. **Closures are preloaded on SUPPORT rests** and held by one strong coupler.
6. **Mounts land on the heaviest nodes** (§3.3), so mount beams can be stiff.

---

## 8. The guidance against the cars

| Topic | The guidance [Doc] | The five cars [Measured] | Verdict |
|---|---|---|---|
| Nodes on the mesh, evenly spaced | W-construct | shell nodes 0.5–1.6 cm from the mesh (median); spacing p10–p90 0.09–0.39 m, so not even | followed for position; spacing is by member, not uniform |
| "3–4 nodes across" | D-tutorial | 4 across the floor plus a centre line; 5 on bumper bars | followed |
| Body about a hundred nodes | W-intro | 118–138 | followed |
| Double-grid floor, both diagonals, skip rigidifiers | W-construct | double layer; 38–41 % diagonals; 12–28 % skip beams | followed |
| Do not span the cabin with beams | W-construct | 6 (ETK, Vertex); 23 (CCF); 34 (SUV) | followed by two, bent by two |
| Deform hierarchy: grid > diagonal > skip | W-construct | ETK edge 13,500 / diagonal 12,000 / skip 12,000; ε_y 1.9 / 1.1 / 0.37 % | followed in strain, weakly in force |
| Main shape unbreakable | W-construct | 97–99 % FLT_MAX | followed |
| Rigidifier 0.5 m off, non-colliding, weaker beams | D-tutorial | 1–4 per panel, 0.2–0.4 m off, non-colliding | followed, at a smaller offset |
| Panels at least 3 × 3 | D-tutorial | doors 8–13 nodes, fenders 10–16 | followed |
| Same weight on all frame nodes except shock mounts | F-agenty-w | 11–18 distinct masses per shell; mounts 1.3–4× heavier | the cars grade mass more finely |
| Node weights 2–3 kg (tutorial), 10 kg (old starter) | D-tutorial, W-intro | shell median 3.0–4.0; panels 0.5–1.7 | between the two |
| Frame beams 1.2e6 / 150 / 50,000 | D-tutorial | 1.5–3e6 / 120–140 / 10,000–14,000 | similar spring and damping; **yield 4–5 × lower** on real cars |
| Springs "very close to the max" | F-goosah | whole-car ω·h 1.80–1.90 of 2 | confirmed, quantified |
| Beam length does not change the limit | F-goosah | Σk/m is the same across parts with L from 0.3 to 1.2 m | consistent |
| `deformLimitExpansion` 1.05–1.1 | F-agenty-dle | 1.1 and 1.2, on 26–35 % of beams | higher than the stated typical |
| Hinge strongest, latch breaks first | D-tutorial | hinges 8–98 kN per beam, sums 0.2–0.95 MN; latch one coupler at 30–40 kN | followed |
| Latch = coupler, 12 kN in the example | D-couplers | 30–40 kN, radius 1.35 m, `couplerLock` | followed, stronger |
| Support beams for self-collision | W-construct, D-support | 6–15 % of beams, 36–46 % precompressed | followed |
| Long wobble: damping-only beams | D-tips | 69–74 beams with k = 0 per car: 40–55 are BOUNDED dampers and limiters; **15–32 are NORMAL damping-only beams** (mirror to door 14, door to body 4–8, lids, seats; 8–10 inside the CCF's and the Vertex's shells, none inside the ETK's) | used sparingly: about 1 % of beams. (The census's "no damping-only beams" should read "almost none".) |
| Torsion bars for flat structures | D-torsion | 4–20 across shell floors | followed |
| Engine: 8-node cube, soft mounts, stops | W-construct | 8 block nodes; mounts k 32,000; 42–78 stops | followed exactly |
| Brake and torque nodes ≥ 2 kg | D-wheels | hub nodes 3.8–8 kg, axle nodes 4.5–5.8 kg | followed |
| Flexbody group ≥ 3 non-collinear nodes, regular grid | D-flex | pools of 4–137; single-layer panels plus an off-plane rigidifier | followed |
| Nodes ≥ 3 cm from triangles | D-tips | not measured | unknown (§12) |
| Glass: trigger ratio under 0.1 | D-beams | 0.0001–0.1 | followed |
| Roof at about 2 × IIHS | D-debug | ETK hardtop 9.6–9.8 × weight, soft top 1.6–6.1 × (`etkk-audit.md`) | not met either way on Rubble |

---

## 9. Weaknesses

Things the baseline does poorly, each with its evidence. These are where the
generated car can be better.

| # | Weakness | Evidence |
|---|---|---|
| 1 | **No interior structure.** Seats and a dash ballast only: no cross-car beam, console, column, B-pillar trim structure or occupants | §5.3; `etkk-audit.md` finding 7 |
| 2 | **Seats cannot detach** on three of four cars; radiators and dashes never | §5.3, §2.3: all mount beams FLT_MAX |
| 3 | **Sparse greenhouse.** A sedan roof is 4 nodes; pillars are 2–3 nodes each; the roof is out of band both ways | §3.2; `etkk-audit.md` finding 5 |
| 4 | **A pole passes between shell nodes.** Beams are 0.44–0.65 m long; a 254 mm pole reaches 1–1.6 m into the ETK | §3.1; `etkk-audit.md` finding 4 |
| 5 | **Frontal crush too stiff and too short** against NHTSA bands: 489–511 mm against 620–750, 56–66 g against 34–47, restitution 0.26–0.35 against 0.13–0.20 | `etkk-audit.md` finding 3 (on Rubble; the car is also a short compact) |
| 6 | **Soft body torsion:** first global shell mode 5.6–27 Hz | §7.2 |
| 7 | **Suspension arms and the front subframe cannot break** (FLT_MAX; subframe mounts 1.24 MN per beam × 64) | §4.3 |
| 8 | **Arms, driveshaft and half-shafts have no nodes:** their meshes ride other parts' nodes or are props that vanish | §4.2, §4.3 |
| 9 | **The fuel tank has no internal structure** (9 nodes, no beams among them) and would collapse if freed | §4.2 |
| 10 | **Panels have no inner layer:** a torn-off door is a sheet with a floating rigidifier node, not a closed solid; its collision surface is one-sided | §5.1; D-tri |
| 11 | **Hand-tuned values:** 49–77 distinct beam settings and 11–18 node masses per shell; nothing derives from section or material | §3.1 |
| 12 | **Ballast instead of parts:** 60 kg of dash nodes (ETK), 400 kg on skirts (SUV), a 12 kg windscreen node | §2.2, §5.4 |
| 13 | **Glass does not exist as an object** on two cars (trigger beams only); none has fragments | §5.4 |
| 14 | **Not at rest on our solver:** the ETK keeps 0.5 J and 0.15 m/s after 10 s; the Vertex bends 157 beams on a 5 cm bump | `etkk-audit.md` finding 2; §7.2 |
| 15 | **Rear subframe held by 40 beams at 80–150 kN** against the front's 1.24 MN: an order of magnitude apart with no physical reason given | §4.3 |
| 16 | **Mesh faceting under crush:** 418 edges change by more than 60° in the wall | `etkk-audit.md` finding 9 |

Weaknesses 4, 5, 14 and 16 were measured on Rubble, not in BeamNG. They are
weaknesses of this content on our solver.

---

## 10. Specification for the sedan (Step 2d)

The deliverable. Every item is marked **[B]** BeamNG baseline (the five cars do
it; the number comes from the section cited) or **[+]** beyond BeamNG (needed
for our principles: closed solids, nothing faked, real crash bands). All of it
is **[Hypothesis]** until built and measured side by side with the ETK.

Positions are given as stations s and as members of the body-in-white, so the
generator derives them from its own key lines. Masses assume a kerb mass of
about 1,450 kg.

### 10.1 Mass budget

| Class | Target | Share | Baseline (§2.2) |
|---|---|---|---|
| Body shell (body-in-white plus the fixed trim it carries) | 430–500 kg | 30–35 % | 30–38 % **[B]** |
| Closures, fenders, bumpers, lights, trim | 130–160 kg | 9–11 % | 8–13 % **[B]** |
| Glass | 30–40 kg | 2–3 % | real mass **[+]** (cars: 1–8 %) |
| Interior (seats, instrument panel, console) | 80–110 kg | 6–8 % | 2–7 % **[B]**; as parts, not ballast **[+]** |
| Suspension, subframes, hubs, steering | 240–270 kg | 17–19 % | 17–19 % **[B]** |
| Wheels (rim, tyre, as axle nodes) | 4 × 18–20 kg | 5 % | axle nodes 3 % plus generated tyre **[B]** |
| Engine | 150–190 kg | 10–13 % | 10–16 % **[B]** |
| Transmission, driveshaft, differential | 60–70 kg | 4–5 % | 2–5 % **[B]** |
| Exhaust, tank (with fuel), radiator | 70–80 kg | 5 % | 5–6 % **[B]** |

The body-in-white's steel weighs about 300–350 kg. The shell's nodes also
carry sealer, wiring, carpet, insulation and paint, which have no part of
their own: 100–150 kg, spread over the floor, firewall and quarter nodes in
proportion to panel area. It is listed as "fixed trim" in the mass report,
not hidden **[+]**.

### 10.2 Body-in-white members

Closed solids in the generator, one-to-one with the shell's nodes. Section
sizes are typical steel-unibody values (general engineering practice, not
from a source read here): **starting values for the generator, to be replaced
by its own key lines.** Gauge sets the visual wall and the real yield force;
the physics lattice takes its masses and yield forces from them.

| Member | Count | Section (mm), gauge | Shell nodes on it |
|---|---|---|---|
| Front crash boxes | 2 | 70 × 110 box, 1.5; 0.15–0.20 m long | 4 at the front face, 4 at the rear face |
| Front rails | 2 | 70 × 110 box, 1.8 | 2 at the subframe's front mount, 2 at its rear mount (kick-down) |
| Radiator support: upper tie bar, lower crossmember, two uprights | 1 | 40 × 60, 1.0 | 3 on top, 1 lower centre |
| Upper fender rails | 2 | 40 × 60, 1.2 | 2 each (front, middle) |
| Strut towers with inner aprons | 2 | pressed, 2.0 (tower), 0.9 (apron) | 1 at the top, 1 at the base each |
| Firewall and toe board | 1 | panel, 0.9 | 4 across at mid height; its foot is floor row 1 |
| Cowl box | 1 | 100 × 150, 0.9 | 4 across (the outer two are the A-pillar bases) |
| Floor pan with tunnel | 1 | panel 0.8; tunnel 180 wide × 160 high, 1.0 | 5 rows × 2 tunnel sides at pan height; 5 tunnel-top nodes |
| Sills | 2 | 110 × 140 box, 1.4, with reinforcement | 5 rows × (bottom, top) each |
| Seat crossmembers | 2 | 50 × 60 hat, 1.2 | on floor rows 2 and 3 (no extra nodes) |
| A-pillars | 2 | 60 × 80, 1.6 | base (in the cowl), middle, top |
| B-pillars | 2 | 60 × 140 tapering, 1.6, reinforced | foot (sill top, row 3), belt, top |
| C-pillars | 2 | 80 × 120, 1.2 | foot (quarter, upper row), belt, top |
| Roof rails | 2 | 50 × 70, 1.2 | the A, B and C tops |
| Windscreen header, rear header, roof bow at the B-pillar | 3 | 40 × 80, 1.0; bow 25 × 60, 0.9 | 1 centre node each |
| Roof panel | 1 | 0.75 | none of its own |
| Rear seat bulkhead and parcel shelf | 1 | panel, 0.8 | 2 bulkhead top, 4 shelf |
| Rear rails | 2 | 60 × 90 box, 1.6 | 3 stations × (lower, upper) each |
| Rear strut towers and inner arches | 2 | pressed, 1.8 | 1 each |
| Boot floor | 1 | panel, 0.8 | 3 on the centre line |
| Tail panel | 1 | panel, 0.8 | 3 across × 2 heights |
| Quarter panels (outer and inner, part of the shell) | 2 | 0.75 | 5 stations × 2 heights each |

Separate parts mounted to it (§10.5): bumper beams, subframes, and everything
mechanical.

### 10.3 The shell lattice: 131 nodes, about 487 kg

| Group | Location (station s) | Nodes | kg each | kg | Baseline |
|---|---|---|---|---|---|
| A1 | Crash-box front faces: upper and lower edge, left and right (−0.30) | 4 | 3.5 | 14 | ETK bumper-beam rows, 4.0 **[B]** |
| A2 | Crash-box rear faces / rail front at the radiator support (−0.20) | 4 | 4.0 | 16 | **[B]** |
| A3 | Radiator support: top left, centre, right; lower centre (−0.18) | 4 | 3.0 | 12 | ETK 3 at 4.0 **[B]** |
| A4 | Rails at the subframe's front mount (−0.08) | 2 | 6.0 | 12 | ETK 5.0–6.0 **[B]** |
| A5 | Rails at the subframe's rear mount, kick-down (0.08) | 2 | 6.0 | 12 | ETK 6.0 **[B]** |
| A6 | Strut tower tops (0.02) | 2 | 8.5 | 17 | ETK 8.5 **[B]** |
| A7 | Tower bases on the inner apron (0.02) | 2 | 4.0 | 8 | **[B]** |
| A8 | Upper fender rails: front (−0.12), middle (0.05) | 4 | 3.5 | 14 | ETK 3.5 **[B]** |
| B1 | Firewall at mid height, 4 across (0.20) | 4 | 3.5 | 14 | ETK 3.5 **[B]** |
| B2 | Cowl top, 4 across; the outer two are the A-pillar bases (0.28) | 4 | 4.2 | 16.8 | ETK 4.2 **[B]** |
| C1 | Floor rows 1–5 at pan height: sill, tunnel side, tunnel side, sill (0.20, 0.36, 0.53, 0.70, 0.86) | 20 | 4.5 | 90 | ETK 4.5 **[B]** |
| C2 | Sill tops over rows 1–5 | 10 | 3.5 | 35 | ETK 3.5 **[B]** |
| C3 | Tunnel top over rows 1–5 | 5 | 3.5 | 17.5 | ETK 3.5 **[B]** |
| D1 | A-pillar middle and top (0.36, 0.48) | 4 | 1.75 | 7 | ETK 1.5–1.7 **[B]** |
| D2 | B-pillar belt and top (0.55) | 4 | 2.5 | 10 | Vertex 1.6–2.2 **[B]**; heavier **[+]** |
| D3 | C-pillar belt and top (0.90) | 4 | 2.5 | 10 | **[+]** |
| D4 | Header centre, B-bow centre, rear header centre | 3 | 1.7 | 5.1 | ETK 1.7 **[B]** |
| E1 | Rear rails: 3 stations (0.92, 1.08, 1.26) × lower and upper × left and right | 12 | 5.0 | 60 | ETK 4.6–5.5 **[B]** |
| E2 | Rear strut towers (1.00) | 2 | 6.0 | 12 | ETK 5.5 **[B]** |
| E3 | Boot floor centre line (0.95, 1.10, 1.26) | 3 | 4.5 | 13.5 | ETK 4.6 **[B]** |
| E4 | Tail panel: 3 across × 2 heights (1.30) | 6 | 3.0 | 18 | ETK 2.6–4.6 **[B]** |
| E5 | Quarter panels: 5 stations (0.80–1.25) × 2 heights × left and right | 20 | 2.8 | 56 | ETK 2.8 **[B]** |
| E6 | Parcel shelf (1.00–1.10) | 4 | 2.8 | 11.2 | ETK 2.8 **[B]** |
| E7 | Rear seat bulkhead top (0.92) | 2 | 3.0 | 6 | **[B]** |
| | **Total** | **131** | p50 3.5 | **487** | 118–138 nodes, 426–515 kg |

Rules for placing them:

- **On the member's visible surface,** within 2 cm (80 % of nodes) and never
  over 5 cm (§3.1) **[B]**. A boxed member's two nodes sit on opposite faces
  (rail lower and upper, sill bottom and top), so the box has depth.
- **Four across at the floor** plus tunnel top; lateral lanes as §3.2 **[B]**.
- **Spacing** along members 0.25–0.50 m; nearest-neighbour p50 0.16–0.22 m
  **[B]**.
- **All self-colliding and colliding** except declared rigidifiers **[B]**.
- **Node mass is assigned by location, not by tributary area alone:** mount
  nodes first (their mount beams fix a minimum mass, §10.6), then members by
  section, then the fixed trim. At most 12 distinct values **[B]** (§8).
- Optional **[+]**: side contact rows on the sills and roof rails at
  ≤ 0.25 m, as light nodes (1 kg, k from the Σk/m rule), for the pole lane
  (weakness 4). They add about 16 nodes and are not counted in the 131. Decide
  after the pole sweep is measured on the new shell.

### 10.4 Bracing and beam values

Topology **[B]** (§3.1, §3.5):

- Edges along and across every member; both diagonals in every rectangle;
  boxed members and the double floor closed into tetrahedra.
- Skip beams along rails, sills, roof rails and across the floor: 12–28 % of
  beams.
- Cross-car beams at each floor row, the cowl, the radiator support, the
  shelf and the tail panel: 8–10 % of beams.
- At most 6 beams over 0.9 m through the cabin volume.
- 10–20 torsion bars along the floor rows (5–8e5 N·m/rad, deform
  7,000–8,000).
- Targets: 650–760 beams (5.0–5.8 per node); 9–11 neighbours per node at the
  median, 5 at least; every node a corner of a non-flat tetrahedron;
  direction spread ≥ 0.02 at every node.
- No SUPPORT beams inside the shell. No damping-only beams inside it either
  (ETK 0; CCF 8, Vertex 10): damp each spring beam.

Values by zone (k N/m, c N·s/m, deform N; strength FLT_MAX except as §10.5)
**[B]** (§3.4):

| Zone | k | c | deform | `deformLimitExpansion` | ε_y target |
|---|---|---|---|---|---|
| Crash boxes (A1–A2) | 3–7e6 | 110–120 | 4,000–6,000 | 1.2 | 0.35–0.5 % |
| Front rails, radiator support, upper rails (A2–A8) | 3–7e6 | 110–120 | 8,000–13,500 | 1.2 | 0.4–0.7 % |
| Towers and subframe mount nodes (A4–A7) | 3.5–7e6 | 120–160 | 26,000–29,000 | 1.2 | – |
| Firewall, cowl, floor, sills (B, C) | 1.5–2.0e6 | 120–135 | 15,000–32,000 | 1.1 | 1–2.4 % |
| Sill and door-frame primaries | 2.1e6 | 400 | 50,000–65,000 | 1.1 | – |
| Pillars, roof rails, headers (D) | from the Σk/m rule | 100–125 | 15,000–24,000 pillars; 8,000–12,000 bows | 1.1 | 1–2 % |
| Rear rails, towers, boot floor (E1–E3) | 1.5–4.5e6 | 125–160 | 8,600–12,500 | 1.2 | 0.8–1.2 % |
| Quarter panels, shelf, tail (E4–E7) | 1.5e6 | 125 | 6,000–9,000 | 1.2 | 1–2 % |
| Skip beams in any zone | the zone's, up to 5.5e6 | the zone's | 0.5–0.9 × the zone's edges | as the zone | 0.35–0.4 % |

Every k is then checked against the node budget (§10.6) and lowered where a
node would exceed it; the budget, not the table, is the rule.

Crush-path force **[+]** (the cars' fronts are too stiff for the NHTSA band,
weakness 5): the sum of `beamDeform` over all longitudinal beams of all parts
crossing a station, at bumper-to-rail height, should rise along the car:

| Station | Target Σ deform (all parts, low band) | Why |
|---|---|---|
| Bumper beam and crash boxes | 120–180 kN | a 1,450 kg car at 56 km/h holds 175 kJ; over 0.62–0.75 m that is a mean of 235–280 kN |
| Rails, ahead of the towers | 200–300 kN | |
| Towers to firewall (rails, subframe, engine stacked) | 400–570 kN | peak 34–47 g × 1,450 kg = 480–670 kN |
| Cabin (sills, tunnel, floor) | ≥ 1.2 MN | ≥ 2 × the front's peak; ETK and CCF 1.1–1.3 MN **[B]** |
| Rear rails | 250–400 kN | ETK 0.42 MN **[B]** |

- **The low band carries at least 55 % of the front's longitudinal yield;
  the hood band at most 6 %** **[B]** (§3.4: ETK 63 % / 3.6 %, CCF 55 % /
  2.1 %). This is the authoring answer to the nose lift (S2).
- These are force targets for authoring. The crash bands themselves
  (`crash-reference-targets.md`) remain the acceptance test.

### 10.5 Parts mounted to the shell

Nodes, masses and mounts. "Group" = a named break group. Mount values are per
beam.

**Structure and running gear**

| Part | Nodes | kg | Internal | Mounts to | Mount beams: k / c / deform / strength | Groups | Mark |
|---|---|---|---|---|---|---|---|
| Front bumper beam | 10 (5 across × 2 heights) | 8 | 27 beams, k from budget, deform 6–10 kN | A1 | 2 × 8 NORMAL: 1e6 / 100 / 20 kN / 60 kN | 2 | **[B]** CCF, Vertex, SUV |
| Rear bumper beam | 10 | 6 | as front | E4, E1 | 2 × 8: 1e6 / 100 / 15 kN / 40 kN | 2 | **[B]** |
| Front subframe | 7 (4 corners 7 kg, 2 rack mounts 7 kg, 1 centre 4.5 kg) | 46.5 | 21 beams: 5–6.5e6 / 120–180 / 80–96 kN / FLT_MAX | A4, A5 | 2 × 32 NORMAL: 1.3–1.8e6 / 180 / 157 kN / **400 kN** | 2 | **[B]** layout; strength **[+]** (ETK 1.24 MN, weakness 7, 15) |
| Rear subframe (or axle carrier) | 6 (4 corners, 2 centre) at 7 kg | 42 | 15 beams as front | E1 (first two stations) | 4 × 10 NORMAL: 2.1e6 / 500 / 18–26 kN / 150 kN | 4 | **[B]** |
| Hub / knuckle, each wheel | 4 in a tetrahedron (lower joint, upper joint or strut foot, toe arm, one more) | front 17, rear 24 | 6 beams: 6e6 / 100 / 129 kN / FLT_MAX | arms, strut, toe link | see below | – | **[B]** |
| Lower arm, each wheel | none (2 beams) | in the subframe and hub | – | subframe corners → hub lower joint | 2: 1.0–1.4e7 / 2,000 / 155 kN / **300 kN** | 1 per arm | **[B]** beams; strength **[+]** |
| Upper arm, each wheel (or the strut on a MacPherson front) | none (2 beams) | – | – | shell mount nodes → hub upper joint | 2: 1.0–1.4e7 / 2,000 / 155 kN / 300 kN | 1 per arm | as above |
| Strut, each wheel | none (1 beam) | in the hub and tower | – | hub → tower top (A6, E2) | 1: from budget / 200 / 17 kN / 180 kN | 1 per strut | **[B]** strength; a rigid link in place of the spring because the raycast wheel owns the travel **[+]** |
| Toe link or tie rod, each wheel | none (1 beam) | – | – | rack end or subframe → hub toe arm | 1: 1.7e7 or budget / 100 / 85 kN / 250 kN | – | **[B]** |
| Steering rack | 2 at 5 kg | 10 | 1 beam | subframe rack mounts | 4 beams, unbreakable | – | **[B]** |
| Wheel (rim and tyre mass) | 2 axle nodes on the spin axis, 0.25–0.27 m apart, plus 3 frame nodes | 18–20 | 10 beams (rigid) | hub's 4 nodes | 8: 6e6 (two at 1e6) / 100 / 168 kN / **168 kN** | 1 per wheel | **[B]** axle nodes and group; frame nodes **[+]** |
| Hub cap | 3 at 0.2 kg | 0.6 | 3 beams | the wheel's outer nodes | 6: budget / 20 / 300 / 600 N | 1 per cap | **[B]** (the wheel data's hubcap group) |

- **Six beams hold each hub:** two lower-arm beams, two upper-arm beams (or
  one strut-foot pair), the strut and the toe link. A four-node rigid body
  needs six, so the hub is held exactly, each beam is a visible member, and
  bending any one changes camber, caster, toe or ride height. **[+]** in
  being exact; the ETK adds 8 limiters per side.
- **Arms have no nodes of their own [B].** The arm's solid binds to its three
  joint nodes plus one hub node for the out-of-plane direction (a flexbody
  group needs non-collinear nodes, §1.6). When the arm's group breaks, the
  arm's mesh stays with the side that still holds two of its nodes.
- **The wheel's raycast frame follows the two axle nodes** (as today it
  follows the hub). When the wheel group breaks, the five wheel nodes are a
  free rigid part with the rim and tyre mesh **[+]**.
- **Rim damage [+], a decision for later (§12):** with 5 nodes the rim cannot
  dent locally; a bent wheel shows as its frame nodes yielding (wobble).
  BeamNG's dent-able rim is 32 generated nodes per wheel at 0.4 kg (§4.4),
  which only makes sense with a node-beam tyre.
- **Failure order to verify (§4.4):** toe link bends (85 kN) → hub and arms
  bend (129–155 kN) → wheel leaves (168 kN) → strut breaks (180 kN) → toe
  link breaks (250 kN) → arms break (300 kN) → subframe leaves (400 kN per
  beam).

**Powertrain and services**

| Part | Nodes | kg | Internal | Mounts to | Mount beams: k / c / deform / strength | Groups | Mark |
|---|---|---|---|---|---|---|---|
| Engine | 8 block corners + 2 (mount ears) | 150–190 (block nodes 18–22 kg) | 40 beams, box with all diagonals, 2e7 / – / 250 kN / FLT_MAX, deform group at ratio 0.001 | front subframe corners | 2 mounts × 3 BOUNDED: 32,000 / 600 / 20 kN / 90 kN | 2 | **[B]** |
| Engine stops | – | – | – | firewall, rails, tower bases, radiator support, subframe | 40–50 SUPPORT: 1–3e6 / 150 / 20–50 kN / FLT_MAX, precompression 0.85–0.95 | – | **[B]** |
| Transmission | 2 (case tail, one more for the frame) | 45 | 24 beams to the engine's rear face: 4.8e6 / 150 / 27 kN / FLT_MAX | tunnel nodes (C1, C3) | 1 mount × 9 BOUNDED: 1.4e5 / 2,700 / 5,000 / 160 kN | 1 | **[B]** (ETK: 1 heavy node) |
| Driveshaft (if rear or all-wheel drive) | 2 light end nodes (0.5 kg) | 8 (in the ends' neighbours) | 1 beam | transmission tail, differential | 1 each end: budget / – / 1,500 / 3,500 | 1 | nodes **[+]** (ETK: one beam, a prop) |
| Differential (if rear drive) | 3 at 14 kg | 42 | 3 beams | rear subframe | 24: 4.8e6 / 150 / 27 kN / **200 kN** | 1 | **[B]**; strength **[+]** |
| Half-shafts | none (1 beam each) | – | – | differential or transmission → axle node | 1: no spring / – / 3,600 / 9,500 | 1 each | **[B]** |
| Exhaust | 12–15 in a chain at 1.4 kg | 17–21 | chain with diagonals, 1.5e6 | E1, C1, tunnel; engine | 4 hanger groups × 4–16: 47,000 / 370 / 4,000 / 15 kN; 20–27 SUPPORT rests; 4 to the engine at 65 kN | 4 | **[B]** |
| Fuel tank | 8, boxed, at 4.5–5.5 kg (fuel included) | 36–44 | 28 beams, soft: budget / – / 5,000 / FLT_MAX | floor rows 4–5, E1 | 26–30 straps: 30,000 / 500 / 5,000 / 35 kN | 1, passive | straps **[B]**; internal box **[+]** (weakness 9) |
| Radiator and condenser | 4 at 1.4 kg | 5.6 | 6 beams | A3 | 12: 1e6 / 150 / 7,500 / **20 kN**, deform group at 0.05 (leak) | 1 | **[B]** 2 nodes unbreakable; 4 nodes and a break **[+]** |

**Interior and glass**

| Part | Nodes | kg | Mounts to | Mount beams | Groups | Mark |
|---|---|---|---|---|---|---|
| Front seat, each | 6–8 at 2.5–3 kg | 18–20 | floor rows 2–3 (4 feet) | 26: 2.8e5 / 38 / 6,000 / **30 kN** | 4 (one per foot) | **[B]** nodes and softness; break **[+]** |
| Rear bench | 6–9 at 2.3 kg | 16–21 | floor row 5, bulkhead | 18–40: 2.8e5 / 38 / 6,000 / 30 kN | 2 | **[B]**; break **[+]** |
| Instrument panel with cross-car beam and column | 6–8; the beam's ends at the A-pillar bases | 40–50 | B2, B1, tunnel top | 28–35: 4e6 / – / 22–42 kN / **60 kN** | 3 | mass **[B]**; a real cross beam and a break **[+]** (weakness 1) |
| Console | 3–4 at 1.5 kg | 5 | tunnel top | 12, 10 kN | 1 | **[+]** |
| Windscreen, rear screen | 1 centre node with the pane's mass (12 and 8 kg) | – | the aperture's shell nodes | 40–62 trigger beams across the aperture: 1.5e6 / 120 / 7,000–12,000, ratio 0.005–0.03; 5–10 % with strength 12–26 kN | deform group, shared break group | **[B]** ETK, CCF |
| Door and quarter glass | 1 node (1.7 kg) each | – | the door's frame nodes | 12 trigger beams: ratio 0.02; 2 at 4,000 N | deform group and break group | **[B]** |

Glass is visual only by the user's decision: when its group fires the pane's
mesh switches to shattered. The trigger-beam pattern does exactly that and
needs no fragment physics.

**Closures and panels** keep the census's bands
(`beamng-authoring-census.md`, Authoring rules), with two changes from this
study:

- **[B]** each closure and fender gets 1–3 **rigidifier nodes** 0.2–0.4 m
  inside its surface, non-colliding, beamed to at least half its nodes with
  beams that yield below the surface beams (§5.1). On our solid parts the
  rigidifier sits on the inner frame, so it is a visible point of the part.
- **[+]** our doors are closed boxes (the parts lab needs a free door to be
  rigid and collidable on both faces): keep the 17-node box, and make sure
  its inner-face nodes are the rigidifier row.
- **[B]** bolt-on panels in several small groups: fender 4 groups of 5–6
  beams at 12 kN; bumper cover 3 to the body and 2 to the fenders at 6–12 kN;
  lamps 1 group of 12–16 at 15–26 kN (§5.2).
- **[B]** SUPPORT: 7–11 % of the car's beams, all between parts, 36–46 % of
  them precompressed at 0.85–0.95 (§3.6).

### 10.6 Per-part target bands

Node budget **[B]**: Σk/m per node within the band, checked on the assembled
car (every beam at the node, any part). ω·h = h·√(Σk/m).

| Part | Nodes | kg | Node kg p50 | Σk/m p50 (1e6 s⁻²) [ω·h] | k·L p50 (N) | ε_y p50 | ζ p50 | First mode, neighbours held (Hz) |
|---|---|---|---|---|---|---|---|---|
| Shell | 131 (118–138) | 430–500 | 3.0–4.0 | 7–9.5 [1.3–1.55] | 0.8–1.1e6 | 1.0–1.3 % (front 0.35–0.7 %, cabin 1–2.4 %) | 0.03–0.04 | free shell ≥ 12 **[B]** CCF; ≥ 20 **[+]** Vertex |
| Door | 13–17 | 17–22 | 1.2–1.9 | 7–10.5 [1.3–1.6] | 0.4–1.0e6 | 1.3–1.6 % | 0.03–0.05 | 16–35; aim ≥ 30 |
| Hood | 21–30 | 18–35 | 0.8–1.5 | 4.6–11.5 [1.1–1.7] | 2.2–4.7e5 | 1.5–5.5 % | 0.03–0.06 | 11–19; aim ≥ 12.5 |
| Boot lid | 12–17 | 12–16 | 0.8–0.9 | 7–11 [1.3–1.65] | 2.2–5e5 | 2–5 % | 0.04–0.06 | 23–40; aim ≥ 26 |
| Fender | 10–13 | 6.5–9 | 0.5–0.8 | 7.6–13 [1.4–1.8] | 1.5–4.3e5 | 1.4–4.6 % | 0.04–0.07 | 17–55; aim ≥ 40 |
| Bumper cover | 19–28 | 9.5–18 | 0.5–0.7 | 3.4–6.2 [0.9–1.25] | 0.4–1.3e5 | 2.7–21 % | 0.07–0.33 | 9–24 |
| Bumper beam | 10 | 6–8 | 0.6–0.8 | 7–9 | – | 1–2 % | 0.03 | 60–90 |
| Subframe | 6–7 | 42–47 | 7 | 8–12 [1.4–1.7] | – | – | 0.03 | – |
| Hub | 4 | 17–24 | 4–6 | 8–12 | – | – | – | – |
| Wheel | 5 | 18–20 | 5 (axle) | 5.4–7.1 [1.15–1.35] | – | – | – | – |
| Engine | 10 | 150–190 | 18–22 | 6.2–8.8 [1.25–1.5] | – | – | – | – |
| Exhaust | 12–15 | 17–21 | 1.4 | 4–5 [1.0–1.1] | – | – | – | – |
| Fuel tank | 8 | 36–44 | 4.5–5.5 | 0.02–0.2 for its straps | – | – | – | – |
| Seat | 6–8 | 18–20 | 2.5–3 | 5–9 [1.1–1.5] | – | – | – | – |

- **If the row cap stays at 1.0,** the reachable Σk/m is about 6.5–7.5e6
  (ω·h 1.27–1.37) at BeamNG's damping, which is 75–80 % of the ETK's shell.
  The mode targets then scale by √0.78 ≈ 0.88: door ≥ 30 Hz, hood ≥ 11 Hz, lid
  ≥ 24 Hz, fender ≥ 37 Hz. **At the cars' own level (rows 1.1–1.6) the "aim"
  column applies.** The choice is the user's (§7.1).
- **Whole-car ω·h** (power iteration, `sedan_stiffness`): 1.6–1.9. Under 1.5
  means stiffness was left unused; the cars are 1.80–1.90.
- **Dynamic bar (the plan's exit criterion):** kerb wobble and settle at the
  ETK's level measured side by side on the same wheels: 0.09–0.23 mm and
  ≤ 0.6 s on raycast wheels; 1.4–4.2 mm on pressure wheels (§7.2).

### 10.7 Lint rules to add

Each is a check on the rigged car, with the measurement it comes from. "Hard"
fails the build; "report" prints the number beside the band.

| # | Rule | Band | Level | From |
|---|---|---|---|---|
| 1 | Shell node count | 110–140 | hard | §3.1 |
| 2 | Shell nodes within 2 cm of their member's surface; none over 5 cm unless a declared rigidifier | ≥ 80 % / 0 | hard | §3.1 |
| 3 | Every shell node is a corner of a non-flat tetrahedron of shell beams | all | hard | §3.1 |
| 4 | Shell beams per node; neighbours per node (min) | 5.0–5.8; ≥ 5 | hard | §3.1 |
| 5 | Beam-direction spread at every shell node | ≥ 0.02 | hard | §3.1 |
| 6 | Skip beams / beams crossing the centre plane, share of shell beams | 12–28 % / 8–10 % | report | §3.1 |
| 7 | Shell beams over 0.9 m through the cabin volume | ≤ 6 | hard | §3.1; W-construct |
| 8 | No SUPPORT inside the shell; damping-only beams in the whole car | 0; ≤ 1 % of beams | hard | §3.1, §8 |
| 9 | Node budget: Σk/m at every structural node | ≤ the cap's equivalent; p50 ≥ 6.5e6 | hard / report | §7.1 |
| 10 | Whole-car ω·h (power iteration) | 1.6–1.95 | report; hard above 1.95 | §7.2 |
| 11 | Mount nodes heavier than the shell median: nodes that subframe, strut, engine or transmission mounts land on | ≥ 1.3 × | hard | §3.3, §4.4 |
| 12 | Crush grading: Σ deform of longitudinal beams by station rises nose → firewall; cabin ≥ 2 × the front's peak | monotone; ≥ 2 | hard | §3.4, §10.4 |
| 13 | Front load path by height: low band's share / hood band's share of longitudinal yield ahead of s 0.2 | ≥ 55 % / ≤ 6 % | hard | §3.4 |
| 14 | Yield strain by zone: front, cabin | 0.35–0.7 %; 1–2.4 % | report | §3.4 |
| 15 | Shell beams with finite strength (glass triggers excepted) | ≤ 3 % | hard | §3.1 |
| 16 | Each hub is held by exactly six named beams (or a listed limiter), all visible members; each wheel has one break group of 8 beams | 6; 1 × 8 | hard | §4.3, §10.5 |
| 17 | Failure ladder: toe link deform < arm deform < wheel strength < strut strength < arm strength < subframe mount strength | ordered | hard | §4.4 |
| 18 | Every part that owns nodes has a mount with finite strength in a named break group (the shell excepted); no group over 16 beams except subframe mounts | all | hard | §5.2; plan 2e |
| 19 | SUPPORT share of beams; share precompressed; engine stops | 7–11 %; 36–46 %; ≥ 36 | report | §3.6 |
| 20 | Each closure and fender has 1–3 off-surface nodes 0.2–0.4 m behind its skin, beamed to at least half its nodes | 1–3 | hard | §5.1; D-tutorial |
| 21 | Mass shares by class | §10.1 bands | report | §2.2 |
| 22 | Flexbody reach: vertex to third-nearest node of its part, median / worst | ≤ 0.40 / ≤ 0.60 m | report | §6.2 |
| 23 | Every mesh binds to its own part's nodes only (shared hardware nodes excepted and listed) | all | hard | §6.1 |
| 24 | Glass: each pane has a deform group of ≥ 12 beams, trigger ratio 0.005–0.05 | all | hard | §5.4 |

Existing rules that this study confirms and leaves as they are: closure
node counts and masses, hinge and latch groups, the collision skin's
closure.

### 10.8 How to accept it

1. `vehicle_lint` with rules 1–24.
2. `structure_dump` and `analyze.py` on the generated sedan's own JBeam beside
   the ETK and the Vertex: the tables of §3.1, §3.4, §4.4 and §7.1 side by
   side. (Add the sedan to `RUBBLE_CENSUS_CARS`; it is JBeam like the others.)
3. `sedan_stiffness sedan etkk`: modes, ω·h, kerb wobble and settle against
   §10.6.
4. The crash scenes of `vehicle_compare` against
   `crash-reference-targets.md`; head-on pitch for the nose lift.
5. The parts lab and the press test (plan 2c, 2e) for each new part.

---

## 11. Glossary of JBeam fields

What the documentation says, and what the five cars use. "Cars" are the four
road cars unless the trailer is named. Rubble's implementation of each field:
`jbeam-semantics-spec.md`.

### Nodes

| Field | Documentation [Doc] | Use in the five cars [Measured] |
|---|---|---|
| `nodeWeight` | kg; default 25 (D-nodes) | 11–18 distinct values on a shell; 0.25–28 kg overall; medians: car 1.2–2.0, shell 3–4, door 1.2–1.9, fender 0.5–0.8, engine block 12–28 |
| `group` | one or more groups; used to bind flexbodies (D-nodes) | 61–88 groups per car; 21–24 % of nodes in two or more |
| `collision` | default true (D-nodes) | on for 88–93 % of nodes; off for rigidifiers and some mounts |
| `selfCollision` | default false (D-nodes) | on for 61–78 % (shells 73–92 %) |
| `staticCollision` | follows `collision` (D-nodes) | set false on a handful of nodes |
| `frictionCoef` | default 1 (D-nodes) | 0.5 on 55–87 % of nodes; 0.7–0.8 on 10–37 %; 1.0 on a few |
| `nodeMaterial` | sound and particles only (D-nodes) | metal 68–72 %, plastic 23–30 %, a few glass, rubber, cloth |
| `couplerTag`, `tag`, `couplerStrength`, `couplerRadius`, `couplerStartRadius`, `couplerLock`, `couplerLatchSpeed` | latch pairs; strength default FLT_MAX; lock radius 0.025 m, latch speed 0.3 m/s (D-couplers) | 2–7 coupler nodes per car: doors 35 kN, hood 2 × 30 kN, boot 40–50 kN, roof latches; radius 1.35, start radius 0.1, lock on; the trailer's hitch 1.5 MN |
| `breakGroup` on a node | breaks a coupling for good (D-couplers) | on latch nodes, tied to the hinge groups |
| `engineGroup` | `engine_block`, `radiator`, `engine_intake`, `fuel` (D-nodes) | block 4–8, radiator 2, intake 1–5, fuel 6–13 nodes |
| `nodeOffset`, `nodeMove`, `nodeRotate` | slot-level transforms (D-nodes) | not examined (the loader applies them before the dump) |

### Beams

| Field | Documentation [Doc] | Use in the five cars [Measured] |
|---|---|---|
| `beamSpring` | N/m; default 4.3e6 (D-beams) | over 70 distinct values per car (ETK 72, Vertex 78); shell median 1.5–3e6; 2e5 (covers) to 2e7 (engine); 69–74 beams with 0 (§8) |
| `beamDamp` | N·s/m; default 580 (D-beams) | about 40 distinct; 50–150 common; shell 120–140; ζ 0.03–0.04 on structure |
| `beamDeform` | N; default 220,000 (D-beams) | 86–100 distinct (ETK, Vertex); shell median 10,000–14,000; front 4,000–13,500; cabin up to 65,000; front subframe mounts 157,000 |
| `beamStrength` | N; default FLT_MAX (D-beams) | FLT_MAX on 57–63 % of beams (shell 97–99 %); finite values 3–1,240 kN in break groups |
| `beamType` | NORMAL, SUPPORT, BOUNDED, ANISOTROPIC, PRESSURED, LBEAM, HYDRO (D-beams) | NORMAL 83–92 %, SUPPORT 6–15 %, BOUNDED 1.7–2.4 %; ANISOTROPIC only the ETK's soft top (1.2 %); no PRESSURED or LBEAM in the body (tyres generate them) |
| `beamPrecompression` | spawn-length ratio; default 1 (D-beams) | ≠ 1 on 3–7 % of beams, 36–46 % of SUPPORT: 0.5–0.95; a few 1.01–1.1 (preloaded seals) |
| `precompressionRange`, `beamPrecompressionTime` | metres; seconds of ramp (D-beams, D-issues) | range on 4 beams (suspension springs); time 0.5 on 6–16 beams (ETK, Vertex) |
| `deformLimitExpansion` | cap on permanent stretch (D-beams); typical 1.05–1.1 (F-agenty-dle) | 26–35 % of beams, 39–53 % of shell beams; values 1.1 and 1.2 only; none on the trailer |
| `deformLimit`, `deformLimitStress` | caps on compression set and on deform growth (D-beams) | not used |
| `beamLongBound`, `beamShortBound` | fractions of length for BOUNDED; break extent for SUPPORT (D-bounded, D-support) | on SUPPORT mostly 2 or 4; larger (12–150) where a rest must never break |
| `longBoundRange`, `shortBoundRange`, `boundZone`, `beamLimitSpring`, `beamLimitDamp` | metres; limit stiffness (D-bounded) | 4–6 % of beams: suspension travel, bump stops, mounts |
| `beamDampFast`, `beamDampRebound`, `beamDampVelocitySplit` | damper curve (D-bounded) | the 4 dampers only |
| `dampCutoffHz` | damping above a frequency; costly (D-beams) | 500 on 51–73 beams, 750 on 4–8 (ETK, Vertex) |
| `breakGroup` | all members break together (D-beams) | 17–26 % of beams in 62–105 groups of 5–14 (median) |
| `breakGroupType` | 1 = broken by the group, does not trigger it (D-beams) | 3–10 % of grouped beams |
| `deformGroup` | fires material swaps and damage (D-beams) | 10–19 % of beams in 18–42 groups |
| `deformationTriggerRatio` | fraction of length that fires the group; under 0.1 (D-beams) | 0.0001–0.1; glass 0.005–0.05; engine 0.001 |
| `optional` | no error if a node is missing (D-beams) | 227–257 beams (ETK, Vertex): the beams between optional parts |
| `disableMeshBreaking`, `disableTriangleBreaking` | keep mesh and triangles when the beam breaks (D-beams) | 78–115 and 466–480 beams (ETK, Vertex) |

### Other sections

| Field or section | Documentation [Doc] | Use in the five cars [Measured] |
|---|---|---|
| `triangles` | one-sided collision and aero surfaces; NORMALTYPE or NONCOLLIDABLE; `dragCoef`; `groundModel` (D-tri) | 508–611 per car, 25–46 NONCOLLIDABLE; 88–97 % of unibody shell nodes are on one |
| `flexbodies` | `mesh`, `[group]`, `deformGroup`, `deformMaterialBase`, `deformMaterialDamaged` (D-flex) | 116–242 rows; 58–76 % name one group; 17–82 with a deform group |
| `props` | rigid meshes on three nodes, driven by a function (D-props) | 13–30: steering wheel, gauge needles, lamp lights, the driveshaft |
| `slots` / `slots2` | part tree; `coreSlot`; `slots2` adds allow and deny lists (D-slots) | 67–158 parts per configuration; the ETK uses `slots` with a few `slots2`, the trailer `slots2` only |
| `torsionbars` | four nodes; N·m/rad; cannot break alone (D-torsion, D-tips) | 8–41 per car: anti-roll bars, wheel alignment, 4–20 across shell floors |
| `slidenodes`, `rails` | a node held on a line of nodes (D-rails) | 2–10 slidenodes (steering rack first); 32 on the trailer |
| `hydros` | beams driven by an input (D-hydros) | 2–10 (steering first) |
| `pressureWheels` | hub and tyre generated around two axle nodes; reaction nodes; `axleBeams` (D-wheels) | 4 (trailer 10); on the ETK 16 rays, hub nodes 0.40–0.42 kg, `hubcapBreakGroup` named, caps disabled |
| `powertrain` | no reference page found | 12 devices on the ETK (engine, clutch, gearbox, shafts, differential, torsion reactor) |
| `energyStorage` | fuel capacity; `fuel` nodes share the mass (D-energy) | tank nodes 6–13 |
| `glowMap` | material swap from electrics (D-glow) | no rows found by the dump on the ETK or the trailer; not examined further |
| `refNodes` | the vehicle's axes (D-refnodes) | not examined |

---

## 12. What we still do not know

1. **BeamNG's integrator and its exact stability bound.** No public source
   names them. We infer the bound from the cars (ω·h 1.80–1.90).
2. **How BeamNG's flexbody binding weighs its 3–4 nodes** and whether there is
   a maximum distance. The page gives the roles only.
3. **How these cars behave in BeamNG itself.** Every dynamic number here is
   from Rubble's CPU reference. The Vertex bending on a kerb is our solver
   with BeamNG content, not a property of the car.
4. **Node-to-triangle clearance at spawn** (the 3 cm rule) on the five cars:
   not measured.
5. **Official cars.** The corpus is five community mods, all recent and well
   made, but not BeamNG's own. The BX in the earlier census agreed with them
   per part.
6. **Which values are deliberate.** We see 49–77 beam settings per shell and
   cannot tell tuning from history. The rear subframe at a tenth of the
   front's strength (weakness 15) may be either.
7. **Self-collision groups and the cost model** (the 10 cm and 50 cm rules)
   against our contact: not examined.
8. **The old wiki's remaining pages** (JBeam reference, PressureWheels,
   Slidenodes) and the BeamNG.tech technical PDF (over the fetch limit) were
   not read.
9. **Real section sizes** in §10.2 are typical values from general practice,
   not from a source read for this study; the generator's key lines should
   set them.
10. **Rim deformation** on a raycast wheel (§10.5): 5 rigid nodes, or a ring
    that co-rotates. Needs a design decision and a measurement.
11. **The pole lane** (weakness 4): whether light contact rows on sills and
    roof rails fix it without costing stiffness. To be measured on the new
    shell.
12. **The row cap** (§7.1): at 1.0 the shell reaches about 78 % of the ETK's
    stiffness-to-mass. Whether that meets "ETK level, side by side" is not
    known until measured.
13. **Whether a six-beam, exactly held hub is stable enough** under raycast
    wheel loads without the ETK's extra limiters.

---

## Appendix A. Sources read

All read on 2026-11-12. `D` = `https://documentation.beamng.com/modding/vehicle`.
Web Archive pages give the snapshot in the URL. Nothing here is source code.

| Key | Source | What it holds |
|---|---|---|
| D-intro | `D/intro_jbeam/` | The model: nodes, beams, spring, damp, deform, strength; typical values; the 6 kg / 7 kg instability picture |
| D-tips | `D/intro_jbeam/jbeamtips/` | Triangle thickness 2.5 cm, 3 cm clearance, self-collision groups, collision cost, slidenode edge collision, damping-only beams, the stress check, dummy beams for rails |
| D-issues | `D/intro_jbeam/common_issues/` | Instability grades, collapsing panels and rigidifiers, precompression time, flexbody errors, property leaks |
| D-debug | `D/intro_jbeam/debugtools/` | Debug views; roof crush at 2 × IIHS; self-collision detector |
| D-slots | `D/intro_jbeam/partslotsystem/`, `D/sections/slots/`, `D/sections/slots/slots2/` | Part tree, `slotType`, `coreSlot`, `slots2` |
| D-nodes | `D/sections/nodes/` | Every node field and default; collision distances; engine groups |
| D-couplers | `D/sections/nodes/couplers/` | Coupler fields, defaults, hood-latch and hitch examples |
| D-beams | `D/sections/beams/` | Every beam field and default; break and deform groups; mesh breaking |
| D-bounded | `D/sections/beams/bounded/` | Bounded beams, ranges, limit spring and damp, damper curve; cost note |
| D-support | `D/sections/beams/support/` | Support beams, precompression, long bound; steering-limiter example |
| D-aniso, D-pressured, D-lbeam | `D/sections/beams/anisotropic/`, `…/pressured/`, `…/l-beam/` | The other beam types (tyres, ropes, soft tops, air springs) |
| D-tri | `D/sections/triangles/`, `…/triangles/quads/` | Triangles, winding, types, drag, pressure groups |
| D-flex | `D/sections/flexbodies/` | Binding to groups; 3–4 nodes per vertex; errors; deform materials |
| D-props | `D/sections/props/` | Props on three nodes |
| D-torsion | `D/sections/torsionbars/`, `D/sections/torsionhydros/` | Torsion bars and torsion actuators; examples |
| D-rails | `D/sections/rails/` | Rails and slidenodes; steering rack recipe |
| D-hydros | `D/sections/hydros/` | Hydros; steering example |
| D-wheels | `D/sections/wheels/` | pressureWheels fields, reaction nodes, axle beams, ray counts |
| D-rotators | `D/sections/rotators/` | Reaction-node rules: far apart, heavy, non-collinear |
| D-energy | `D/sections/energystorage/`, `D/sections/nodes/fire/`, `D/sections/nodes/exhaust/` | Fuel nodes, rupture beams, exhaust node flags |
| D-glow | `D/sections/glowmaps/` | Material swap from electrics |
| D-vars | `D/sections/variables/`, `D/sections/components/` | Tuning variables driving node positions and beam values; shared keys between parts |
| D-refnodes | `D/sections/refnodes/` | The reference axes |
| D-tutorial | `D/tutorials/basic_car_tutorial/` | **The official build guide:** 3–4 nodes across, weight budget, tripod rule, rigidifiers, frame and mount values, hinges and latches, coilover recipe |
| D-model | `D/vehicle_modeling/` | Mesh rules: origin, detachable parts as separate meshes |
| D-split | `D/vehicle-art/final-mesh-splitting/` | What to split a car's model into |
| D-glass | `D/vehicle-art/texturing/glass-damage-textures/` | Glass damage is a binary material swap |
| D-modsupport | `https://documentation.beamng.com/modding/mod-support/` | Repository rules; procedural only |
| D-tight | `https://documentation.beamng.com/beamng_tech/tools/cosimulationeditor/tight_coupling/` | 0.5 ms physics step; own thread |
| D-args | `https://documentation.beamng.com/beamng_tech/arguments_and_settings/` | 2,000 steps per second default |
| W-construct | `https://web.archive.org/web/20170611113451/http://wiki.beamng.com/Construction_Guide` | **The best single authoring source:** lattice on the mesh, double-grid floor, deform hierarchy, no cabin spans, engine cube, suspension, panels, support beams |
| W-theory | `https://web.archive.org/web/20211220124859/https://wiki.beamng.com/JBeam_Physics_Theory` | The model and typical values (as D-intro) |
| W-problems | `https://web.archive.org/web/20210827030712/https://wiki.beamng.com/Vehicle_Modding_Common_Problems` | Instability causes; mesh spikes |
| W-intro | `https://web.archive.org/web/20220125134611/https://wiki.beamng.com/Introduction_to_Vehicle_Creation` | Starter values; "about a hundred nodes"; axle layout |
| W-beams | `https://web.archive.org/web/20161127173956/http://wiki.beamng.com/Beams` | Material character; strength against deform |
| W-tri | `https://web.archive.org/web/20240805113753/https://wiki.beamng.com/Collision_Triangles` | Triangle winding; drag coefficients |
| W-engine | `https://web.archive.org/web/20211016013251/https://wiki.beamng.com/CombustionEngine`, `https://web.archive.org/web/20220103002018/https://wiki.beamng.com/Powertrain` | Torque reaction nodes; device tree; `breakTriggerBeam` |
| W-flex | `https://web.archive.org/web/20170613054113/http://wiki.beamng.com/Flexbodies` | Groups; mesh breaking |
| W-deform | `https://web.archive.org/web/20161128061438/http://wiki.beamng.com/DeformGroups` | Windscreen trigger ratio 0.03; one material per swapped mesh |
| B-physics | `https://www.beamng.com/game/about/physics/` | The public description of the model |
| B-tire21 | `https://beamng.com/game/news/blog/tire-physics-changes/` | Tyres are node-beam; rolling resistance history |
| B-tire15 | `https://www.beamng.com/game/news/blog/a-look-at-tire-development-in-beamng-part-2/` | pressureWheels; 16 rays |
| F-goosah | `https://www.beamng.com/goto/post?id=898127` | **Staff: springs are very close to the maximum; the limit is stiffness to mass; length does not matter** |
| F-agenty-w | `https://www.beamng.com/goto/post?id=1511689` | Staff: equal frame node weights except shock mounts |
| F-agenty-dle | `https://www.beamng.com/posts/1313359/` | Staff: `deformLimitExpansion` 1.05–1.1 |
| F-rate | `https://www.beamng.com/threads/is-there-a-way-to-turn-up-physics.51381/` | Developer: the rate is compiled in |
| G-blender | `https://github.com/BeamNG/Blender-JBeam-Editor/blob/main/docs/user/user_docs.md` | The editor's user manual (documentation, not code): import, export; no structural guidance |
| R-concepts | `https://docs.rigsofrods.org/vehicle-creation/vehicle-concepts/` | Rigs of Rods: the same model |
| R-truck | `https://docs.rigsofrods.org/vehicle-creation/fileformat-truck/` | Its defaults, minimum mass, detacher groups, flexbody `forset`, contacters |
| R-weight | `https://docs.rigsofrods.org/vehicle-creation/weight-tuning/` | Its method for curing spawn explosions |
| R-panels | `https://docs.rigsofrods.org/vehicle-creation/flexbody-deforming-detaching-parts/` | Hood as a 4 × 4 grid with stabiliser nodes 0.3 m off; hinge and latch |
| R-wheels | `https://docs.rigsofrods.org/vehicle-creation/wheels-axles-steering/`, `…/suspension/` | Ray counts; reaction arm node |
| P-icse | `inspiration/research-ror-beamng/gambi-icse-seip-softbody.pdf` (local) | The BeamNG authors' paper: 2 kHz loop, beam model, node-triangle collision, 714-node example, one thread per vehicle |
| P-itec | `https://air.unipr.it/handle/11381/2870805` | Abstract only: a soft-body vehicle model validated against multibody |

Not found or not readable: a `powertrain` reference page (404); a hubWheels
page; the live old wiki (redirects); the BeamNG.tech technical PDF (too
large); several blog posts with no text in the fetch.

## Appendix B. Reproducing every number

Corpus: the five unpacked mods under `~/Downloads/jbeams/` (override with
`JBEAMS`). Raw outputs go to `~/Development/rubble-artifacts/beamng-structure-study/`
(override with `OUT`); they hold per-car detail and are not committed.

```bash
# Everything in §2–§6 and §7.1: dumps the cars, then writes <car>.<section>.md
engine/tools/beamng-structure-study/run.sh

# The cross-car tables
python3 -I engine/tools/beamng-structure-study/analyze.py \
  ~/Development/rubble-artifacts/beamng-structure-study/dump \
  etkk_kc6 etkk_kr6 ccf vertex fullsuv boathauler --only summary

# One section of one car: parts, attach, body, grade, brace, mech, wheelpath, glass, flex, stab, types
python3 -I engine/tools/beamng-structure-study/analyze.py <dump dir> etkk_kc6 --only wheelpath

# The dump alone (opt-in test; skipped unless both variables are set)
cd engine && RUBBLE_STRUCTURE_OUT=<dir> \
  RUBBLE_CENSUS_CARS="etkk_kc6=$HOME/Downloads/jbeams/etkk_lucasbe/vehicles,etkk,kc6.pc" \
  cargo test --release -p engine-softbody --test authoring_census -- --ignored --nocapture structure_dump

# §7.2: modes, whole-car ω·h, spawn, half-sine loads and the bump (CPU, about 4 minutes for four cars).
# RUBBLE_COMPARE_MODS is a folder of links: <label> -> the mod's vehicles/ folder.
cd engine && RUBBLE_COMPARE_MODS=<mods> cargo run --release -p engine-softbody --example sedan_stiffness -- \
  etkk=etkk,kc6.pc ccf=ccf,f4_sport_M.pc vertex=gavril_vertex,gavril_vertex.pc fullsuv=fullsuv,mid-sport.pc
```

| Section | Command output |
|---|---|
| §2.1, §2.2, §3.1, §3.4 (table), §4.1, §4.4 (table), §5.2, §5.4, §6, §7.1 | `--only summary` |
| §2.3 | `etkk_kc6.parts.md`, `etkk_kc6.attach.md`, `etkk_kc6.mech.md` |
| §3.2, §3.3 | `<car>.body.md` |
| §3.4 beam classes, §3.5, §3.6 | `<car>.grade.md`, `<car>.brace.md` |
| §4.2, §4.3 | `etkk_kc6.mech.md`, `etkk_kc6.wheelpath.md`, `etkk_kc6.attach.md` |
| §5.1 (depth, non-colliding nodes) | `<car>.panels.md` |
| §3.4 (load path by height), centre of mass | `<car>.height.md` |
| §2.1 (parts, break groups, slot depth), §3.6 and §4.2 (engine stops), §8 (beams with no spring) | `<car>.counts.md` |
| §7.2 | `sedan_stiffness` |
| §11 | `<car>.types.md` |

Two readings are not from these tools and are cited where used: the ETK's
wheel options (ray count, hub node weight; read from its wheel files' rows as
data) and the crash-scene numbers (`etkk-audit.md`).
