# Deformable vehicles in Rubble: how they work

**A living document.** It explains, end to end and in plain terms, how a Rubble vehicle is modelled, simulated, drawn and broken. Update it whenever the model changes; the changelog is at the end.

**Inspired by, not copied from, BeamNG.** BeamNG.drive showed that a coarse net of masses and springs, stepped very fast and dressed in a detailed mesh, gives convincing soft-body cars. Rubble uses the same family of ideas. It reads BeamNG's public JBeam file format so community vehicles load, and it follows BeamNG's *public documentation* where the format has a meaning. Its solver is our own, written clean-room: we never read or copy BeamNG or Rigs of Rods code. **The target is to be as good as BeamNG or better,** and to go where BeamNG doesn't: hundreds of vehicles on the GPU, and vehicles in the same physics world as 100k+ destructible rigid bodies.

The detailed specifications this summarises:

| Topic | Spec |
|---|---|
| The solver, step by step | [explicit-beams.md](explicit-beams.md) |
| JBeam fields and their meaning in Rubble | [jbeam-semantics-spec.md](jbeam-semantics-spec.md) |
| Vehicles in the rigid World | [explicit-vehicles-in-world.md](explicit-vehicles-in-world.md) |
| How we verify it | [vehicle-test-strategy.md](vehicle-test-strategy.md), [verification-plan.md](verification-plan.md) |
| Authoring vehicles | [vehicle-authoring.md](vehicle-authoring.md) |
| Open problems | [handoff/known-bugs.md](handoff/known-bugs.md) |

---

## 1. The idea in one picture

A vehicle is three layers. Only the first is physics.

```
 ┌───────────────────────────────────────────────────────────────┐
 │ 3. What you see: detailed meshes ("flexbodies", props)        │  ~100k triangles
 │    skinned to the nodes; deform groups swap visuals           │  no physics
 ├───────────────────────────────────────────────────────────────┤
 │ 2. What touches things: collision triangles between nodes,    │  ~1–3k triangles
 │    raycast or pressure wheels                                 │  contact + aero
 ├───────────────────────────────────────────────────────────────┤
 │ 1. What moves: nodes (point masses) joined by beams           │  ~300–1,000 nodes
 │    (springs + dampers that bend permanently and snap)         │  ~3–7k beams, 2 kHz
 └───────────────────────────────────────────────────────────────┘
```

Real numbers:
- the BeamNG ETK K-series as Rubble loads it: 457 nodes, 3,800 beams, 1,453 kg;
- Rubble's authored sedan: 987 nodes, 6,865 beams.

The trick that makes few nodes look good: **the physics net is coarse but laid out along the real structure, and the detailed mesh follows it smoothly.** A dent is a few nodes moving, and the bonnet's thousands of vertices bend smoothly between them.

## 2. Layer 1: nodes and beams

### Nodes
A node is a point mass: position, velocity, mass and a friction coefficient. Masses come from the vehicle file. In JBeam, `nodeWeight` sets them per node or per group, and a part's total is close to the real component's (a bonnet weighs what a bonnet weighs). Heavy things are small clusters of heavy nodes, held by mount beams: the engine block, gearbox, fuel tank (Rubble adds the fuel's mass) and battery.

### Beams
A beam joins two nodes. Each step it measures its length ℓ against its rest length L and pushes or pulls both nodes along the line between them:

  **F = −k·(ℓ − L) − c·(relative speed along the beam)**

with k the spring stiffness (N/m) and c the damping (N·s/m). Each beam also has:
- **a deform threshold** (`beamDeform`, N). Push or pull harder and the beam **yields**: its rest length moves toward its current length, so the shape stays bent. That is plastic deformation, the reason a crushed bonnet stays crushed. Rubble's law (the docs give none): the rest length moves so the force returns halfway to the threshold, and the threshold hardens by the length ratio. Yield follows the elastic force only, so plastic flow never stores energy (jbeam-semantics-spec §3);
- **deform limits** (`deformLimit`, `deformLimitExpansion`): how far the rest length may move, as a fraction of the original;
- **a strength** (`beamStrength`, N). Past it the beam **breaks** and stops existing.

Beam kinds, each a named rule in the solver:

| Kind (JBeam) | What it models | Rubble rule |
|---|---|---|
| NORMAL | Sheet metal, frame members, mounts | Spring–damper, yield, break |
| SUPPORT | Members that only resist compression (e.g. keeping panels from collapsing inward) | No force when stretched; breaks past a stretch factor |
| BOUNDED | Shock absorbers, bump stops, limiters | Free inside a range, stiff with its own damping outside; rebound damping |
| ANISOTROPIC | Different stiffness in compression and extension | Switches rate over a transition zone |
| PRESSURED | Gas springs (air suspension) | Gas law on its surface |
| HYDRO | Steering racks, actuators | A beam whose rest length the game drives |
| LBEAM | Resists bending at a joint, not stretching | Angle-based |
| Torsion bars and hydros | Anti-roll bars, twist | Torque between node pairs |
| Couplers | Latches, tow hitches | Join and release |
| Rails and slidenodes | Sliding mechanisms | A node held on a rail |

### Triangles
Triangles stretched between three nodes are the vehicle's **collision skin and aero surface**. Other vehicles' nodes collide with them, as do the vehicle's own nodes (self-collision), and air pushes on them. A triangle stops colliding when one of its side beams breaks (triangle breaking), unless the beam is marked `disableTriangleBreaking`.

### Groups: how a car comes apart and shows damage
- **Break groups:** beams sharing a group break together. When one hinge beam snaps, the whole hinge lets go, so a door falls off cleanly instead of one strand at a time. Passive members (`breakGroupType` 1) are broken by the group but never set it off.
- **Deform groups:** a beam that yields past `deformationTriggerRatio`, or breaks, sets a flag that stays on. The flag doesn't change physics; the visual layer reads it (glass cracking, lamps breaking).

### Parts
A vehicle is assembled from **parts** (chassis, bonnet, each door, bumpers, radiator, engine, suspension, wheels…). A `.pc` configuration picks a part for each slot. Each part brings its own nodes, beams, triangles and visuals, and attaches to its parent through specific beams that can break. That's why damage reads as real: parts detach at their joints.

## 3. One step at 2 kHz

The vehicle is stepped with h = 0.5 ms, 33 steps per 1/60 s World step, the remainder carried over. Each step has three passes:
1. **Nodes:**
   - contact with the ground and static shapes;
   - integrate: semi-implicit Euler, v += F/m·h, then x += v·h;
   - restart the force with gravity.
2. **Beams:** each live beam computes its force, yields or breaks, and adds its force to both ends.
3. **Contact between triangles and nodes:** self-collision, and other vehicles in the same contact group.

Why so fast? Steel is stiff and nodes are light. An explicit step is stable only when h is well under the period of the stiffest beam–mass pair, and 2 kHz is what BeamNG uses too.

The pay-off of explicit steps:
- stress waves, rebound and ringing come out naturally;
- force history is fine-grained, so yield and break decisions are accurate;
- contact moves at most a couple of centimetres a step at highway speed, so nothing tunnels.

We measured the alternative: implicit AVBD vehicles were either jello (cheap settings) or 4–9× the cost (rigid-looking settings).

Every vehicle has a CPU reference (`engine-softbody`, f64 where it matters) and a GPU kernel (`slang/kernels/explicit_beams*.slang`). They agree bit for bit in tested scenes, which is how we test the GPU.

**Profiles.** The same solver runs two rule sets:
- **The Rigs of Rods profile** (for `.truck` content and black-box comparison with Rigs of Rods).
- **The BeamNG profile** (for JBeam content): exact beam lengths, no idle-triangle throttle, two-body contact with mass splitting, anti-clip, a 2.5 cm contact band, contact that never adds energy, JBeam plasticity, exact friction, and per-triangle aero.

Each switch is a separate setting (jbeam-semantics-spec §36).

## 4. Layer 2: contact

| Contact | How |
|---|---|
| Ground | Each node against the ground height; friction law with static and sliding regimes |
| Static shapes (boxes, convex, mesh) | Each node against each shape |
| Self-collision | Node against the vehicle's own triangles, within a contact band (2.5 cm in the BeamNG profile); a grid finds candidate pairs |
| Vehicle against vehicle | Touching vehicles join one **contact group**, stepped together; two-body contact splits the response by the lumped masses so light skin nodes don't get flung |
| Anti-clip | Each node remembers which side of a triangle it was on, plus a swept crossing test, so fast nodes can't pass through a panel |
| Surface contact (opt-in) | The vehicle's **triangles and beam edges** against rigid shapes, not just its nodes. This closes the gap where a narrow object slips between nodes (§9) |
| World rigid bodies | Through the coupling (§8) |

**Limit to know:** with node-only contact, anything narrower than the node spacing (about 0.3–0.5 m) can slip between nodes: a lamp post, or a brick corner pressed in slowly. BeamNG and Rigs of Rods share this limit. Rubble's surface contact is how we intend to beat it; it isn't on by default yet (cost and calibration, jbeam-semantics-spec §38.7).

**Second limit, a current bug:** contact against *moving* World rigid bodies has no penetration correction. Nothing pushes a node back out once it's inside a brick, and the brick's motion inside a coupling window differs from the World's. So slow pushes let bricks sink into the car (known-bugs B1).

## 5. Layer 3: what you see

The visual model is separate, detailed meshes (from `.dae` files in JBeam vehicles). Each mesh vertex is **bound to nearby nodes once, at load**, and **recomputed from the node positions every frame** on the GPU. Rubble has three bindings:
1. **Three-node frames** (`engine-softbody/src/flexbody.rs`, the JBeam-style default). Each vertex takes its nearest allowed node as an origin, plus two more nodes chosen so the two directions are as close to perpendicular as possible. The vertex is stored in that little frame. A rigid motion moves it exactly, and a deformation carries it along.
2. **Node-frame skins** (`authoring/node_frames.rs`, a **Rubble extension**, the authored showcase cars). Each node of a part gets a smooth frame (position and rotation) from its own part's neighbourhood, and each vertex blends the frames of its 8 nearest own-part nodes with weights continuous in its rest position. Neighbouring vertices therefore always move together (no shards or saw-tooth), a part carries only its own skin (a pulled-off door takes its whole skin), and the skin is cut only where beams broke (a vertex follows the side of its nodes, grouped by live stiff beams, that carries most of its weight); folds and crushes are drawn as they are. Rest pose and rigid motion are exact. Why and the measurements: [measurements/skin-shards.md](measurements/skin-shards.md).
3. **Weighted skins** (`flexskin.rs`, a **Rubble extension**: a `.rskin` sidecar named by `"rubbleSkin"`). Each vertex stores explicit node weights from a smooth moving-least-squares fit over its own part's nodes. Rigid motion is reproduced exactly, and the surface stays smooth: no creases where a three-node frame would switch nodes. Normals are recomputed from the deformed field. This is how Rubble does better than per-vertex frames on dense panels.

Like skeletal animation for a character, with three differences:
- there are hundreds of "bones" (nodes) with no hierarchy;
- the bones themselves bend permanently and break;
- the mesh never affects physics.

**Props** (lights, mirrors, steering wheel) are rigid meshes pinned to a node triple. **Deform groups** switch materials or meshes (cracked glass). In the renderer, paint wears where the car slid, which is per-node visual state.

## 6. Wheels and driving

Two wheel models:

| | Raycast wheels (used for driving) | Pressure wheels (opt-in) |
|---|---|---|
| What it is | The rigid World's raycast vehicle model (suspension spring and damper, bump stop, anti-roll bars, tyre friction curves within a friction ellipse, brakes, steering with Ackermann), evaluated **as forces on the hub nodes inside each 0.5 ms step** | Real node–beam tyres: hub and tread rings joined by beams, plus a pressure group (air in the tyre) |
| Deforms with crashes | Yes: forces go through the beams, a crushed suspension moves the hub, and the ray moves with it | Yes, fully |
| Cost | Cheap | Many more nodes and beams |
| Limits today | Rays see the ground and static shapes, **not World rigid bodies** (known-bugs F7). Drive is a direct-drive simplification (no engine, clutch, gearbox or differentials) | The converted ETK has no tyre content, so it rides on its rims |

**Powertrain is a gap.** BeamNG simulates engine, gearbox, differentials and damage to them. Rubble doesn't yet.

## 7. Aerodynamics

Per triangle (BeamNG profile): drag ½ρ·C·(`dragCoef`/100)·A·|u|²·sin α along the flow, lift by angle of attack up to a stall angle, each node of the triangle taking a third (jbeam-semantics-spec §31). The documented default `dragCoef` is 100, a flat plate; about 10 is typical for body panels.

Content that leaves it at 100 on every triangle gets far too much drag.

The Rigs of Rods profile instead puts drag on **every node** (F = −0.05·|v|·v each, independent of shape). That grows with the node count, not the frontal area: a 987-node car gets 127× a physical car's drag and tops out near 41 km/h (known-bugs B2). Dense Rubble cars need triangle aero or C_d·A-calibrated drag, set in the car's source.

## 8. Vehicles in the rigid World

Each 1/60 s World step:
1. Rigid bodies near each vehicle become **proxies**: shape, pose, velocity and mass.
2. The vehicle runs its 33 steps against them, each proxy moving as a free body.
3. The impulses go back to the rigid solver for its next step.

Prefractured buildings take the vehicle's strike into their threshold fracture before their rigid step: forward-only, never re-simulated.

Measured:
- momentum closes at the coupling;
- an immovable proxy matches the solver's own wall within about 5 %;
- adding vehicles costs rigid-only scenes nothing.

Details: [explicit-vehicles-in-world.md](explicit-vehicles-in-world.md).

## 9. Running at scale (where Rubble goes beyond BeamNG)

- **One GPU workgroup per vehicle** steps its 33 substeps in one dispatch. Large single vehicles use "wide" passes across workgroups; touching vehicles share one contact group (splitting a group across workgroups is next).
- **Vehicle sleep** (an explicit engine-wide setting): a still vehicle isn't stepped and resumes exactly where it froze.
- **Origin rebasing:** positions are relative to a per-vehicle origin, moved every 100 m, so precision holds anywhere on a large map.
- **GPU safety:** every kernel loop is budgeted. A runaway fails the step with an error instead of hanging the GPU.
- **Same source everywhere:** CPU reference, WebGPU (Chrome), Metal and CUDA, built from one Slang source.
- **Measured:** the ETK runs at 60 World steps/s in Chrome inside a 110,441-brick World (≈10 ms GPU a step).

## 10. Authoring and importing

- **JBeam (BeamNG format):** parts, slots, `.pc` configs, nodes, beams (all kinds above), triangles, break and deform groups, flexbodies and props. Community vehicles are local-only data: never committed, never executed. See the `beamng-community-content` skill.
- **`.truck` (Rigs of Rods format):** for Rigs of Rods content and black-box comparison. Rubble's extensions (`rubble_*` lines) carry JBeam features `.truck` can't express.
- **Rubble's own cars:** the authored sedan and the vehicle generator (`engine/tools/vehicle-gen`, art models; gen3 physics paused).

**Authoring rules that matter:**
- lay nodes along the real load paths (rails, pillars, crumple zones), and give parts realistic masses;
- set deform and strength per member group, so crumple zones yield before the cabin;
- attach parts with breakable joints and break groups;
- give panels realistic `dragCoef`;
- keep node spacing below the size of things the car must not pass through, or use surface contact.

## 11. How we know it's right

Physical accuracy and mathematical soundness come first; how it looks comes last. The order of trust:
1. closed forms;
2. conservation (momentum, an energy ledger that never rises, Newton III across every coupling);
3. CPU and GPU agreeing;
4. two code paths agreeing;
5. Rigs of Rods' own solver as a black box on Rigs of Rods content;
6. BeamNG's public docs;
7. real crash data under matching conditions;
8. visual judgement.

**BeamNG's solver is closed, so "matches BeamNG" can only mean "follows its documented semantics,"** unless someone records BeamNG itself. Behaviour that looks wrong is a hypothesis until a valid reference says so. See the `physics-evidence-first` skill and [verification-plan.md](verification-plan.md).

## 12. Rubble vs BeamNG today

| Area | Rubble | Status |
|---|---|---|
| Node–beam physics at 2 kHz | Yes, our own solver | Done |
| JBeam beam kinds, groups, parts | Most documented features | Done; gaps in the spec |
| Flexbody visuals | Three-node frames, plus weighted MLS skins (ours) | Done |
| Contact | Node and triangle; anti-clip; two-body; surface contact (ours, opt-in) | Surface contact not default yet |
| Tyres | Raycast (driving) and pressure wheels | Converted mods lack tyre content |
| Powertrain and damage to it | Direct drive only | Gap |
| Many vehicles on the GPU | Yes | Contact groups on one workgroup |
| Vehicles + 100k destructible rigid bodies in one world | Yes | Done; see known bugs |
| Bit-exact CPU reference for every GPU kernel | Yes | Done |

## Glossary

- **Node:** a point mass.
- **Beam:** a spring–damper between two nodes that can yield (bend permanently) and break.
- **Triangle:** a collision and aero surface between three nodes.
- **Flexbody:** a detailed visual mesh skinned to nodes.
- **Prop:** a rigid visual pinned to nodes.
- **Part, slot, `.pc`:** the vehicle's assembly and its configuration.
- **Break group:** beams that break together.
- **Deform group:** a damage flag for visuals.
- **Contact group:** touching vehicles stepped together.
- **Proxy:** a rigid body's copy inside a vehicle frame.
- **Profile:** a rule set (Rigs of Rods or BeamNG) for the same solver.

## Changelog

- 2026-11-13: node-frame skins for the authored cars (the triangle binding drew shards in crashes).
- 2026-11-07: first version, written for the PR #105 handoff. Same day: node drag vs triangle aero, and the moving-body penetration bug (from `diag/ring-phase-through`).
