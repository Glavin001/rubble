# BeamNG-quality destruction for the showcase sedan

Date: 2026-11-05. Branch: `feat/showcase-car-visual`. This is a research and design document only; no code has changed.

The goal, in the user's words, is "BeamNG quality, visuals and destruction". The car should crumple and keep its substance, not behave like sheets of paper. A T-boned door should be able to fall off because its hinges and latch broke. Every visual layer should follow the physics: the skin, glass, lamps, torn edges and the debris.

Clean-room note. Everything about BeamNG here comes from public documentation, staff forum posts and the BeamNG.tech technical paper, summarised in my own words, plus counts I measured from local JBeam content (read as data, not copied). No BeamNG or Rigs of Rods code was read or translated. The local content is in `inspiration/` directories, which are gitignored. Nothing here makes a build or test depend on it.

---

## 0. Summary

**What makes BeamNG look right.** It is mostly not node density. The real BeamNG car I measured, the Ibishu BX, has 611 explicit nodes in its default configuration. Its panels have 20–25 cm node spacing, and a door is 11 nodes. The realism comes from three things:

1. **Structure that is honest about load paths:**
   - Each panel is its own part.
   - Its hardware (each hinge, the latch, each bolt) is one break group.
   - A seal carries damping only.
   - Compression-only SUPPORT beams make up 17% of all beams. They make panels bear on the structure behind them, which gives the car its "substance".
   - Some parts have an inboard "rigidifier" node that gives a single-sheet panel depth.
2. **Correct part masses.** The BX bonnet is 15.6 kg, a door is 11.5 kg of skin nodes plus 16 kg of door-panel nodes, and the front bumper is 9.9 kg.
3. **A visual layer driven by physics events.**
   - Glass and lamps switch to broken materials when their beams deform or break.
   - Props such as lamps go dark.
   - Each detached part carries its own mesh.

**Rubble's showcase sedan today.** I measured these on both the visual worktree and the re-noding worktree.

- **Physics:**
  - **No break groups anywhere.** 0 of 7,195 beams are in a detacher group. Hinges, latch and bolts are individual beams.
  - **A door can't come off in a crash.** The front-left door is held by 50 tension-carrying joint beams with a summed break strength of about 1.37 MN. The FMVSS 206 minimum is 11 kN per hinge system and 11 kN per latch.
    - In my car-to-car T-bone probes at 50 and 80 km/h, and in oblique rigid-wall impacts up to 100 km/h, at most 5 of the 50 joints broke. The door never detached.
  - **Panel seals act like welds.** They carry full tension at half the panel grade.
  - **Bonnet and fenders are held by more than their hardware.** Bidirectional "backing" ties to the body frame also hold them on.
  - **Little compression-only support.** Only 104 support beams (1.4%), and only at the door ends, the deck-lid stops and the engine block's faces.
  - **Panels are 3–8× too light.** Bonnet 4.9 kg, fender 1.6 kg, boot lid 1.6 kg, front bumper 2.1 kg.
  - **Collision triangles never break, and the collision skin spans the shut lines.** 32 collision triangles join door_f_l to the body. A detached door would drag a stretched "ghost sheet" of collision triangles with it.
  - **Glass behaves like soft metal.** Its yield strain is 4.4% and it yields before it breaks. Real tempered glass breaks at about 0.17% strain, with no yield.
- **Visual:**
  - **Detached panels do carry their visuals** (good): binding is per part.
  - **Damage visuals are not tied to physics events.** Glass shatters on 12% strain of a skeleton triangle, not when glass beams break.
  - **Mirrors and lamps vanish instead of falling.** They are discarded when their mount moves 4 cm.
  - **No inner panels.** Nothing sits between the outer skin and the door card 10 cm inside. Torn edges are a shaded lip, not geometry with thickness.
  - **No particles or sound** in the showcase renderer. `car-render.mjs` already has sparks, shards, chips and dust.

**First five plan items** (§3; details and acceptance tests there):

1. Hinge, latch and bolt hardware as break groups, with strengths from FMVSS 206, and damping-only seals.
2. Compression-only SUPPORT beams in place of the bidirectional backing ties.
3. Panel masses from sheet area × gauge × density.
4. A per-part collision skin, so no collision triangle spans a shut line. This needs no solver change; triangle breaking is the optional solver alternative.
5. Physics damage events (part detached, glass broken, lamp broken) exported to the renderer. Mirrors and lamps become small physical parts that fall.

---

## 1. How BeamNG achieves its realism

Sources: the URLs are listed in §6, and each point gives its key.

- **[DOC-*]** pages under `documentation.beamng.com/modding/vehicle/`
- **[RTP]** the BeamNG.tech technical paper (2021)
- **[FORUM-*]** staff and modder posts

Counts marked **(measured)** are mine. I measured them from the local BX content with a tolerant JBeam reader I wrote in the scratchpad, counting nodes, beams by type, triangles, flexbodies, breakGroups and deformGroups per part. The data is at `/Users/glavin/Development/rubble-beamng-parity/inspiration/beamng-bx/vehicles/bx/`, the Ibishu BX, a BeamNG-authored 1990–94 coupe with `"Author": "BeamNG"` in its `info.json`.

### 1.1 Part decomposition: parts, attachment beams, break groups, detachment

- **A car is a tree of parts plugged into slots** [DOC-slots, DOC-partslot].
  - Each part has its own nodes, beams, triangles and flexbodies.
  - Beams may join nodes of different parts. These are the attachment beams.
- **Break groups** [DOC-beams]:
  - When any beam of a group breaks, every beam in that group breaks.
  - `breakGroupType 1` makes a beam a passive member. The group can break it, but it never sets the group off.
  - The docs advise passive membership for the support beams between neighbouring panels, so they go when either panel leaves [DOC-tips].
  - Triangles can carry a breakGroup too, so collision surfaces split with the parts [DOC-triangles].
- **What detaches** [DOC-split]: separable parts are modelled as separate meshes. The docs list the bonnet, doors, bumpers, boot, engine, gearbox, driveshafts, arms, axles, lights and grille.
- **Detached parts stay physical.** No page says it outright (flagged unverified by the research). It follows from the model: a broken beam stops acting, and the part's nodes keep their node and triangle collision [RTP, DOC-nodes].

**The BX door's hardware (measured)**, read from `bx_doors.jbeam` as data:

| Element | Count | Parameters (as authored) | Role |
|---|---|---|---|
| Upper hinge | 11 beams, breakGroup `doorhinge_a_*` | spring 1.3 MN/m, deform 9.8 kN, strength 78 kN | One group: when its first beam reaches 78 kN, the whole hinge lets go |
| Lower hinge | 8 beams, breakGroup `doorhinge_b_*` | same | Second, independent group: one hinge can fail and the door hangs from the other |
| Latch | 1 coupler (`advancedCouplerControl`), its own breakGroup `door*_latch` | coupling strength 35 kN | A releasable latch; also carries open and close sounds |
| Seal | ≈ 27 beams | **spring 0**, damping 50, strength 102 kN | Damping only: it carries no static load, so it can't act as a weld |
| Door-to-body supports | 44 beams, `SUPPORT`, breakGroup `door*_supportBeams` | spring 1.0 MN/m, deform 8 kN, strength 200 kN, break on stretch past `beamLongBound` 6 | Compression only: a door pushed in bears on the pillars, sill and inner structure. Pulled out, it carries nothing |
| Pop-open | 1 beam, SUPPORT, precompressed 1.02, passive member of the upper-hinge group | | Pushes the door ajar once the latch releases |
| Detents and stops | 3 BOUNDED beams | | Hold the door's open positions |

Every other BeamNG part I measured follows the same pattern. The general rule is per-mount groups, not one group per part, so a panel peels and hangs before it falls:

- **Fender:** 4 mount groups (`fender_a`…`fender_d`) plus a support group.
- **Front bumper:** 8 mount groups.
- **Rear bumper:** 5 mount groups.
- **Side skirt:** 4 mount groups.
- **Bonnet and boot:** 2 hinge groups each, plus a support group.
- **Lamps and mirrors:** each has its own group.
- **Wheels:** a `wheel_FL` group per corner on the suspension beams.
- **Subframe:** 2–4 mount groups.

In the BX's default configuration there are **90 distinct break groups (measured)**.

### 1.2 Beam types and parameters

All from [DOC-beams] and its sub-pages, except where marked.

- **Parameters.**
  - Stiffness and damping: `beamSpring` (N/m, default 4.3 MN/m) and `beamDamp` (N·s/m).
  - `beamDeform` (N) is the force above which the beam yields plastically, changing its rest length.
  - `beamStrength` (N) is the break force.
  - `deformLimit` / `deformLimitExpansion` cap how far plastic flow can go.
  - `beamPrecompression` (or `precompressionRange`) is a spawn length factor, ramped in by `beamPrecompressionTime`.
- **Lasting creases come from plastic yield** [RTP p.4].
  - Elastic deflection springs back. Above `beamDeform` the rest length moves, so the panel stays bent.
  - Sponge-like parts use a low deform and a high strength.
  - **Brittle parts such as glass use deform ≥ strength**, so they break before they bend.
- **SUPPORT beams** [DOC-support].
  - They resist only compression and are free in tension.
  - They break when stretched past `beamLongBound`, measured from spawn length, or past an absolute `beamLongExtent`.
  - They are limiters that stop one part's nodes going through another part. They are not attachments.
- **BOUNDED beams** [DOC-bounded]: normal stiffness inside a range, then a stiffer limit spring and damper (`beamLimitSpring`, `beamLimitDamp`, with a `boundZone` transition). They are used for dampers, bump stops and door detents. They are costly.
- **ANISOTROPIC beams** [DOC-aniso]: a different spring and damper in extension (`springExpansion`, `dampExpansion`). They are used for tyre sidewalls and ropes.
- **PRESSURED beams** [DOC-pressured]: gas cylinders. They break above `maxPressure`.
- **Other kinds:** HYDRO, LBEAM and torsion bars.
- **Deform groups.**
  - `deformationTriggerRatio` is the deformation fraction (typically < 0.1) at which a beam fires its `deformGroup`.
  - Breaking also fires it, and is cheaper [DOC-beams, DOC-tips].
  - This is the hook that swaps glass or lamp materials and triggers radiator leaks.

**Measured share of beam types on the BX (default configuration, 3,927 beams):**

| Type | Count | Share |
|---|---|---|
| NORMAL and BOUNDED (BOUNDED: suspension, mounts, detents) | 3,265 | 83.1% |
| **SUPPORT** | **662** | **16.9%** |

Per panel, the SUPPORT share is:

| Part | SUPPORT beams | Share |
|---|---|---|
| Door | 45 of 101 | 45% |
| Bonnet | 38 of 169 | 22% |
| Fender | 23 of 104 | 22% |
| Front bumper | 76 of 309 | 25% |
| Body shell | 46 of 893 | 5% |

Panels are where the "substance" lives.

### 1.3 Node density and weights (measured)

**BX, default configuration** (`rubble_default.pc`, which takes the slot defaults):

| Quantity | Value |
|---|---|
| Parts walked through the slots | 84 |
| Explicit nodes | 611 (the tyres come from a `common` part that is not in the local copy) |
| Beams | 3,927 |
| Collision triangles | 599 |
| Flexbodies | 140 |
| Break groups | 90 |
| Deform groups | 29 |
| Sum of node weights | 1,245 kg |

**Per part (nodes, kg, median node weight, nearest-neighbour node spacing):**

| Part | Nodes | Mass (kg) | Median node (kg) | Median nn spacing (p10–p90) |
|---|---|---|---|---|
| Body shell (`bx_body_coupe`) | 145 | 328.9 | 1.7 (0.3–6.75) | 0.19 m (0.10–0.33) |
| Door (each) | 11 | 11.5 (+ 10 door-panel nodes, 16.3 kg) | 1.0 | 0.25 m (0.24–0.29) |
| Bonnet | 26 | 15.6 | 0.55 | 0.24 m (0.07–0.37) |
| Fender (each) | 13 | 5.4 | 0.4 | 0.19 m (0.13–0.27) |
| Front bumper | 33 | 9.9 | 0.3 | 0.12 m (0.11–0.16) |
| Rear bumper | 24 | 7.2 | 0.3 | — |
| Boot lid | 18 | 11.2 | 0.6 | 0.23 m (0.08–0.36) |
| Windscreen | 10 | 34.0 | 3.0 | — |
| Headlight (each) | 6 | 2.4 | 0.4 | — |
| Mirror (each) | 7 | 2.2 | 0.25 | — |
| Engine (i4 2.0) | 10 | 142 | 17 | — |
| Front suspension | 28 | 107.7 | 4.7 | — |
| Rear suspension | 27 | 133.8 | 4.75 | — |

**Cross-checks.**

- **BeamNG's own template car is not a damage reference.** It has 59 body nodes at 10–20 kg, so a rigid placeholder (measured, `beamng-template-car`).
- **Staff on node counts:** cars have "about 300–400 nodes", with 30–40k render triangles including the interior. A dense mesh on a weak skeleton gives "spiky deformations" [FORUM-highpoly].
- **Engine limits:** in 2018 the per-vehicle limits rose to 3,000 nodes and 15,000 beams [PATCH-0.13].

**Conclusion.** The memory note "realism from ≤ 10 cm node density" does not match the real content. BX panels run at 19–25 cm. Only the bumper, at 12 cm, is near 10 cm. The density BeamNG pays for sits in the *visual* mesh. The physics skeleton is sparse but structurally honest.

### 1.4 Panel construction: skin, inner structure, bracing; the door

From the BX door's node and beam layout (measured, described in my own words):

- **Skin.** A 3 × 3 grid of skin nodes at about 1.0 kg (1.3 kg at the corners) spans the door. There are about 23 skin beams: the grid edges, "surficial crossing" diagonals and corners. Each beam yields at 3.5–15 kN and is never authored to break (`beamStrength` FLT_MAX).
- **Rigidifier.** One node, 1.6 kg in the light door variant and 3.0 kg in the heavy one, sits about 30 cm inboard of the skin. Beams fan from it to the skin nodes. This makes the one-sheet door a 3D truss with depth.
  - About 9 beams join it to the skin.
  - It stands in for the inner panel and the intrusion beam.
  - It is excluded from the flexbody group, so no visual vertex binds to it.
- **Latch node.** It sits at the rear edge (0.6 kg) and is the coupler's end.
- **Door panel part.** A second part (`bx_doorpanel_*`) puts 10 heavier nodes (16.3 kg) on the same grid. It carries the trim and mass. It has its own flexbody, and the same node group joins the door and its panel.
- **Panel to structure.**
  - Hinge groups at the front.
  - The latch coupler at the rear.
  - Damping-only seal beams round the edge.
  - 44 SUPPORT beams from the door's nodes to the body's nodes, so the door bears on the pillars and sill.

The general lesson is that a panel's "substance" comes from three sources. None of them requires shell thickness in the physics:

1. **Mass:** realistic kilograms per part.
2. **Depth:** an off-plane node or an inner frame that makes the sheet a truss.
3. **Bearing:** support beams onto the structure behind.

### 1.5 Collision

- **All collision is node-based** [DOC-nodes, DOC-triangles].
  - Triangles collide only with nodes, never with other triangles.
  - Terrain (heightmap) contact is node vs ground.
  - Static map objects are node vs collision-mesh faces.
  - Dynamic contact is node vs vehicle triangles, both self and other vehicles.
  - **BeamNG's static world collision is node-only, the same as Rubble's.**
- **`selfCollision`** (default off) lets a node hit its own vehicle's triangles.
- **Node groups control what skips what.** Nodes skip triangles made of nodes in their own `group`. That is how a part does not collide with itself, while a detached door does collide with the body [DOC-nodes].
- **Triangle breaking.** When a beam under a triangle breaks, the triangle loses collision and aero unless `disableTriangleBreaking` is set. So torn sheets and broken glass open physical holes, and seams do not drag ghost surfaces [DOC-beams].
- **Spacing guidance** [DOC-nodes, DOC-tips]:
  - Keep colliding nodes about 3–3.5 cm from triangles.
  - Keep collidable parts more than 10 cm apart.
  - Keep the underside sparse, because static collision is expensive.
- **Anti-snag triangles.** `NONCOLLIDABLE` triangles exist only to stop cars snagging on each other.

### 1.6 Visual layer

- **Flexbodies** [DOC-flexbodies].
  - Each mesh vertex binds to 3–4 nodes of the flexbody's node group: a centre node plus nodes roughly at right angles.
  - Poor node layouts (collinear, coincident or outlier nodes) make spikes.
  - **Flexbodies are per part**, so a detached part carries its mesh.
  - `disableMeshBreaking` (default off) removes the flexbody polygons over a broken beam, so meshes do not stretch across tears [DOC-beams].
- **Deform groups switch materials** (`deformMaterialBase` → `deformMaterialDamaged`) when their beams deform past the trigger ratio or break [DOC-flexbodies].
  - On the BX (measured), the windscreen's 18 beams carry `windshield_break` (trigger ratio 0.03) and the backlight's carry `backlight_break` (0.001).
  - Each side glass has its own group and each lamp lens its own group: `headlightglass_*_break` on 39 headlight beams, `taillight_*_break` on 22.
- **Glass is binary, intact or broken** [DOC-glasstex].
  - Tempered glass shows a dense granular texture with a jagged hole.
  - Windscreen cracks are modelled as 3D mesh.
  - Lamp covers lose shards to expose the reflector.
  - Glass meshes are split inner and outer, and lamps into frame, glass and unit, so each can swap material on its own [DOC-split].
- **Props** are rigid meshes on three nodes. A deform or break group disables them, so a lamp goes dark when its glass breaks [DOC-props].
- **Damage materials (scratches, chips, dirt).** The research found **no official built-in scratch or chip system**; there are only community mods [MOD-scratch]. Dynamic Decals is a livery tool. Treat paint damage as our own design, not as BeamNG parity.
- **Panel thickness and inner surfaces.** No official page covers this. It is a property of the art: each panel's flexbody models its inner skin, flanges and jambs, as the mesh-splitting guide implies [DOC-split, DOC-modeling].
- **Mesh and skeleton are separate** [FORUM-highpoly]. Render density can be 100× the node density. Our skinning has the same split (`skin.rs`).

### 1.7 Particles and sound

- **Break particles.** A beam with both a break group and a deform group spawns glass or wood particles when it breaks, chosen by its nodes' `nodeMaterial`. Plastic nodes spawn plastic bits [DOC-beams, DOC-nodes].
- **`nodeMaterial`** (metal, plastic, glass, wood and so on) changes only impact sounds and particles, not the physics [DOC-nodes].
- **Sound:**
  - Collision samples graded by impact strength, plus deformation-energy tracking, since v0.13 [PATCH-0.13].
  - Scrape and grind audio recorded in a studio and driven by physics parameters [BLOG-sneak6].
  - Latches have attach and detach sound events (measured on the BX coupler).
- **Sparks from metal scraping:** only a forum report says this (unverified officially). Rigs of Rods' node option `f` ("no sparks") implies nodes spark by default there [RORDOC].

### 1.8 Other contributors

- **Wheels and tyres** [DOC-wheels].
  - Pressure wheels are built from nodes, beams and triangles, with a pressure group per tyre.
  - Hub beams have their own deform and break values.
  - Axle beams cut drive torque when they break.
  - Wheels detach through a `wheel_XX` break group on the suspension.
  - The deflation and puncture mechanism is not documented (unverified).
- **Suspension and subframe mounts** have their own break groups (measured: `subframemount_*`, `enginemount_*`).
- **Engine and thermal damage** run as separate simulations driven by deform groups: radiator leaks, oil pan damage, engine block "mainEngine" [DOC-fire, PATCH-0.5.3]. On the BX (measured), `mainEngine` covers 44 engine beams and `oilpan_damage` covers 8.
- **Fuel tank rupture and fire:** a container beam break spills fuel; heat can burst tyres [DOC-fire].
- **Couplers** hold doors, bonnets and tailgates shut and release them [DOC-couplers].
- **Physics rate:** 2 kHz, fixed in drive and spring-mass explicit, the same as ours [RTP, DOC-arch].

### 1.9 What actually produces the look (synthesis)

1. **Failure is decided at the hardware, not across the sheet.** Hinges, latches and bolts are groups that fail as units. Seals do not hold. So a panel comes off whole and hangs by its last mount first.
2. **Panels bear and have mass.** Support beams plus realistic part mass give the sense of substance. A pushed panel loads the structure behind it, and a detached panel falls with momentum.
3. **Visuals follow physics state.** Material swaps on deform or break groups, per-part meshes and props that switch off.
4. **Sparse physics, dense art.** About 20 cm node spacing on panels, with 30–40k triangles of art bound to it.

---

## 2. Gap analysis: Rubble's showcase sedan

### Measurements

**Code read.**

- Authoring, at `/Users/glavin/Development/rubble-showcase` (`feat/softbody-showcase-car`, including the re-node work not yet committed there) and in this worktree:
  - `authoring/vehicle.rs`: `Joint`, `DOOR_HOLD`, `DOOR_BEAR`, the seam loop, backing, `coarse_skin`.
  - `authoring/sedan.rs`: the panels and `door_joint`.
  - `authoring/material.rs`: the grades.
- Solver, in `explicit_beams.rs`:
  - The beam pass, with detacher groups and the compatible-profile break exception.
  - `collision_pass`, which has no triangle liveness.
- Visual: `authoring/skin.rs` and `parts/*`.
- Renderer: `engine-webgpu/showcase-car-render.mjs`.
- Tests: `tests/showcase_car.rs` (tests `g`, `m`, `p`).

**Probes.** Small programs in the scratchpad, linked against each worktree's crate. They run on the CPU reference for 1.5–2 s of simulated time, with no repo changes. All numbers are for the Medium resolution.

| | Visual worktree (HEAD 8afa00be) | Re-node worktree (uncommitted state) |
|---|---|---|
| Nodes / beams | 1,237 / 8,264 | 1,105 / 7,195 |
| Support beams | 122 (1.5%) | 104 (1.4%) |
| Beams in detacher groups | **0** | **0** |
| door_f_l: nodes, mass | 46, 29.8 kg | 42, 30.4 kg |
| door_f_l: tension-carrying joint beams | 51; Σ strength 1,389 kN (median 23 kN, max 85 kN) | 50; Σ 1,366 kN |
| door_f_l: contacters | 19 of 46 | 19 of 42 |
| Collision triangles spanning door_f_l and another part | 32 | 32 |

**Part masses** (re-node worktree) against the BX and against sheet physics:

| Part | Rubble (kg) | BX (kg) | Area × gauge × ρ estimate (kg)* |
|---|---|---|---|
| Bonnet | 4.9 | 15.6 | outer ≈ 1.3 m² × 0.75 mm × 7,850 ≈ 7.7, plus inner ≈ 15 |
| Fender (each) | 1.6 | 5.4 | ≈ 0.45 m² → 2.6, plus brackets ≈ 3–4 |
| Boot lid | 1.6 | 11.2 | ≈ 0.9 m² outer + inner ≈ 10–11 |
| Front bumper | 2.1 | 9.9 | fascia plus reinforcement beam: several kg |
| Front door | 30.4 | 11.5 + 16.3 | ✓ in range |

\*Outer-panel gauge is 0.7–0.9 mm [GAUGE]. The inner stamping is thicker than the outer [PATENT-5605371]. These are my estimates, and the plan below makes them a computed, tested quantity.

**Side-impact probes.** Each probe reports the door_f_l joints broken out of 50 or 51, and whether the door detached.

| Scenario | Joints broken | Door detached |
|---|---|---|
| Car-to-car T-bone, striker front-first, 90°, 50 km/h (striker an identical sedan, at the front door's centre) | 0 | no |
| Car-to-car T-bone, 90°, 80 km/h | 5 | no |
| Car-to-car, 60° oblique, 80 km/h | 2 | no |
| Lateral slide into a rigid wall, 50 km/h | 0 | no |
| Lateral slide into a rigid wall, 70 km/h | 0 | no |
| Oblique rigid wall, 20° at 60 km/h | 1 | no |
| Oblique rigid wall, 30° at 80 km/h | 3–5 | no |
| Oblique rigid wall, 15° at 100 km/h | 1 | no |

Test `g` does detach the door, but it pulls with the joints' *summed* break strength: about 1.3 MN on the Low car.

### 2.1 Point by point

| BeamNG ingredient | Rubble today | Status |
|---|---|---|
| Parts with their own nodes | `skeleton.parts`: body, bumpers, bonnet, boot lid, fenders, 4 doors, radiator, blocks, suspension | ✓ exists |
| Hardware as break groups (per hinge, latch, bolt) | `Joint::Bolt { group }` exists, but the sedan sets `group: false` everywhere. `Joint::Hinged` builds individual beams. Door-frame hinge and latch ties are individual `DOOR_HOLD` beams (yield 15 kN, break 22.5 kN each, 2–4 per door end) | ✗ missing (the solver supports groups; authoring does not use them) |
| Seals carry no load | The "flange and seal" seam beams are full beams at 0.5 × the panel grade, tension included (`vehicle.rs`, the seam loop's `else` branch) | ✗ wrong: seals act as welds |
| Attachment strength from a standard | The `DOOR_HOLD` comment cites FMVSS 206 as a floor. The real door system holds about 1.37 MN summed, roughly 100× the 11 kN minimum | ✗ far too strong (no ceiling) |
| SUPPORT beams for substance | 104–122 in total: door frame ends to pillars (`DOOR_BEAR`), deck-lid stops, engine bearing faces. No panel-to-structure supports on the bonnet, fenders, bumpers or door skin | ◐ partial |
| Panels backed without extra attachment | `backed` panels (door, fender, bumper, bonnet) tie skin nodes to frame nodes with `attach(… backing)`: **bidirectional** ties. For a frameless panel (fender, bonnet) the ties go to the **body's** frame, adding attachment the joint system does not count | ✗ wrong for bolted and hinged panels |
| Off-plane depth | Doors have their own frame members (`door_belt`, `door_mid`) through `frame_through`; the bonnet and boot lid have none | ◐ doors only |
| Realistic part mass | Doors ✓. Bonnet, fenders, boot lid and bumpers are 3–8× light (table above) | ✗ "sheets of paper" |
| Plastic creases | Plastic yield with `PLASTIC` keep-fraction; panel hinge (brace) beams yield, so creases stay | ✓ exists |
| Brittle glass | `glass` grade: k·L 9 kN, yield 400 N, break 760 N; yield strain 4.4%; plastic before break | ✗ glass bends like soft metal |
| Detachment works physically | Detacher-group cascade exists in the solver (`group > 0`). The break exception (beams whose node is on a collision triangle with fewer than 3 live plain beams get stronger instead of breaking) applies to truck-mode beams | ◐ solver ready; authoring doesn't use it |
| Detached parts collide with the ground | Every node has node-vs-ground and node-vs-box contact | ✓ |
| Detached parts collide with the car | Only contacters (the coarse skin's nodes, about 45% of a door's) hit triangles. No self-group exclusion is authored. **Collision triangles never break**, and 32 skin triangles span the door seam: after detachment they stretch into a large ghost surface between door and body | ✗ risk of false contact; ◐ partial coverage |
| Collision skin per part | One closed lofted skin over the whole shell (`coarse_skin`), capped at 2,900 of the solver's 3,000 triangles | ✗ spans shut lines |
| Visual follows the part | `skin.rs` binds each piece to one physics part's single-part triangles (vote), so a detached door carries its skin, rolled edges and handle | ✓ |
| Visual damage tied to physics | Glass shatter: binding-triangle strain > 0.12, then discard. Windscreen cracks with strain. Mirrors, lamps and handles discard when their mount moves 4 cm (`ANCHOR_BREAK`). The renderer reads node positions only, never beam state | ✗ heuristic, not event-driven; small parts vanish |
| Inner panels and thickness when torn | Rolled edges with a return flange at shut lines. A door card offset 10 cm inside. Headliner. Torn edges are cut by the tear field with a bright "bare steel lip" shade, and back faces are dark e-coat. No inner door skin, intrusion beam, jamb or shut faces, inner bonnet frame, wheelhouse or apron mesh | ✗ missing |
| Particles | None in `showcase-car-render.mjs`. `car-render.mjs` has a GPU particle system (sparks, glass, chips, dust from strain changes) | ◐ reusable code exists |
| Sound | None | ✗ (out of scope for now) |
| Wheels detaching | Corner links are individual beams; no wheel group | ✗ |
| Tyre deflation | No pressure groups in the authored tyres | ✗ (solver feature) |
| Damage materials (scratches, chips, dirt) | None (the paint is uniform) | ✗ (BeamNG has none built in either) |

### 2.2 The five questions, answered

1. **Can a door detach in a T-bone today?** No.
   - The door is held round its whole perimeter by 50 tension-carrying beams, about 1.37 MN summed.
   - In a T-bone the door is pushed inward, and the inward load goes into compression-only bearings and the seam beams.
   - At most 5 joints broke in every probe up to 80 km/h car-to-car and 100 km/h oblique wall.
   - Only test `g`'s pull, at the summed break strength, tears it off.
   - This is partly correct physics: real doors usually stay latched in a pure T-bone, which is what FMVSS 206 is for. But no load in a crash can ever separate the Rubble door.
2. **Do detached parts collide with the ground and with the car?**
   - With the ground: yes, every node.
   - With the car: only through contacters (about 45% of a panel's nodes) against the car's single coarse skin.
   - That skin includes triangles spanning the seams, which never break. So a detached panel would be pushed by, or would push, a stretched ghost sheet.
   - No test covers it, because nothing detaches in the crash tests.
3. **Do panels have compression-only support?** Only partly. The door frame's ends bear on the pillars (`DOOR_BEAR`, `Support(10)`) and the deck lid has stops. The bonnet, fenders, bumpers and door skin are backed by **bidirectional** ties, and the bonnet and fenders are backed to the body frame. BeamNG uses SUPPORT beams for exactly these ties.
4. **Do the visuals of a detached part move with it?** Yes for panels, because per-part binding is in place. No for mirrors, lamps and handles, which are discarded when their mount strains instead of riding a physical part.
5. **Are there inner panels with thickness?** No. There are rolled edges with return flanges at shut lines, and a door card 10 cm inboard. Between them is empty space, and a tear shows a shaded lip, not a second skin.

---

## 3. Prioritised implementation plan

The rules apply throughout:

- No lossy changes. Nothing silently degraded.
- Solver changes need the user's approval and are flagged **[SOLVER]**.
- Each limit is an explicit configuration that fails loudly.

Everything else is authoring, export, or renderer work. Each item has an acceptance test that is grounded either physically (a standard, a conservation law or a material constant) or in measured data. Tests go in `tests/showcase_car.rs` unless noted.

### P1. Hardware joints as break groups, with seals that carry no load (authoring)

**Change:** replace the joints with hardware, in `authoring/vehicle.rs` and `authoring/sedan.rs`.

- **Hinges:**
  - `Joint::Hinged` gets explicit hardware: upper hinge, lower hinge, latch.
  - Each is one detacher group through `Skeleton::beam_with(…, group)`.
  - It is built at the hinge and latch nodes of the door *frame* tier (`DOOR_HOLD` ties) and the skin seam in that zone.
- **Latch:** one group on the latch zone's ties.
- **Seals:** the "flange and seal" seam beams become damping-only (k = 0, BeamNG's seal) or compression-only supports (`BeamType::Support`). They then carry no tension. I prefer Support: it also keeps the shut line from closing under a push.
- **Hardware strength:**
  - Taken from FMVSS 206 S4.1.2.1 and S4.1.1: a hinge *system* holds 11,000 N longitudinal and 9,000 N transverse without separating, and a latch holds 11,000 N and 9,000 N.
  - A system test may split the load among the hinges (S4.1.2.2).
  - Set each hinge group's ultimate so the *system* separates between the FMVSS minimum and a stated ceiling.
  - My recommendation: **ultimate = 1.5 × the FMVSS minimum**, a typical design margin. The real ultimate exists only in compliance reports, which need the user's permission to download (`crash-reference-targets.md` §7).
  - Make the factor one named constant (`HARDWARE_MARGIN`), not a magic number per beam.
- **Bolts:** `Joint::Bolt` gets per-mount groups (`mounts: n`): the seam is split along its length into n groups. The fender gets 4, matching the BX's 4 mount groups and a real fender's 4–6 bolts. Each bumper gets 6–8.
- **Bonnet and boot lid:** 2 hinge groups and 1 latch group each, from the same FMVSS-style hardware rule. FMVSS 206 covers side doors only, so use the door hinge's per-hinge strength and state that this is an assumption.
- **Break exception:** check the compatible-profile exception (`explicit_beams.rs`, the `saved` rule) against the new groups. A hinge node that keeps 3 or more live internal plain beams is unaffected. Assert this in a structural test rather than changing the solver.

**Acceptance tests:**

1. **Quasi-static FMVSS 206 rig.** The door is held at its hinge pillar. Ramp a load at the latch and hinges, longitudinal and then transverse.
   - The hinge system carries 11,000 N longitudinal and 9,000 N transverse without separating (S4.1.2.1).
   - It separates below 1.5 × those loads.
   - The same holds for the latch (S4.1.1.1 and S4.1.1.2).
   - The ramp is slow enough that the door's inertia is negligible; test `p`'s rule applies.
2. **Updated test `g`.** A pull of 1.5 × (11 kN + 11 kN) longitudinal tears the door off. A pull of 0.9 × the FMVSS minimum does not.
3. **Hanging test.**
   - Break the lower hinge group alone, by command in the test.
   - The door hangs from the upper hinge: its centre of mass ends up below the hinge axis within 2 s, and no other group breaks.
   - Then break the upper hinge group. The door falls freely: its centre-of-mass acceleration equals g within 2% until first ground contact.
4. **Inertial latch load, FMVSS 206 S4.1.1.4 (30 g).** A 30 g longitudinal and a 30 g transverse body pulse (sled) do not open the latch group.
5. **No regression:**
   - Test `m` (the boot lid holds at 56 km/h).
   - Test `p` (FMVSS 214 static door crush thresholds).
   - Test `o` (side-pole residual crush in the NHTSA band).

   The door must still carry a side load into the pillars. That is the supports' job, not the seals'.

### P2. Compression-only support: substance without extra attachment (authoring)

**Change:**

- `attach(… backing)` for `backed` panels becomes two kinds of beam:
  - Within a panel's own part (door skin to door frame): normal ties, as now.
  - To another part (fender, bonnet or bumper to the body's frame): `BeamType::Support`, as passive members of the panel's mount groups, so they go when the panel leaves (BeamNG's support beams with `breakGroupType 1`). This needs follower groups (negative group ids). The solver has them; check the `.truck` export path writes them, and if it can't, that is a small export change, not a solver change.
- **Door skin bears on the structure:** add supports from the door skin and frame inner nodes to the B-pillar, sill and the rear door's front frame, as the BX door's 44 supports do. They break on stretch past `SUPPORT_BREAK_FACTOR`.
- **Bonnet bears on the aprons and towers:** add bonnet-to-apron and bonnet-to-tower supports. This replaces today's tie path, which the code comment explains as "its inner panel bears on the aprons".

**Acceptance tests:**

1. **Statics.** Sum every tension-capable beam joining a panel to another part. That sum must equal the panel's hardware groups' strength, within 1%. A structural test enforces this: no hidden attachment.
2. **Substance.** FMVSS 214 S6.1 (test `p`) still meets its thresholds: ≥ 10 kN mean over 152 mm, ≥ 15.6 kN over 305 mm, and the peak rule. With seals and backing now unable to carry tension, the bearing supports must carry the crush.
3. **No pass-through.** In the 56 km/h wall, no bonnet node ends up behind the firewall's plane, and no fender node ends up inside the wheel-arch inner wall (the geometry from the profile). Today's comment names this failure mode ("sinking through the wheelhouses' skin").

### P3. Part masses from sheet physics (authoring)

**Change.** Each panel gets `gauge` (m) and `inner` (a share of its area as inner stamping, or an explicit inner-frame member). Its mass becomes area × gauge × 7,850 kg/m³ for steel (1,000–1,200 kg/m³ for a plastic bumper fascia), plus its hardware. The mass goes into `solve_masses` as a target per part.

**Acceptance tests:**

1. **Part mass.** Each part's node mass is within ±10% of its computed sheet mass. The test computes the panel's area from its own skin triangles, so the number is not typed in.
2. **Kerb mass and the existing ledgers still hold**: tests `a` and `b`.
3. **Stability floor.** Every panel node stays at or above its stability floor at the 0.5 ms step (`SLACK=1` already reports this). Heavier panel nodes raise the floor margin, which is a lossless gain.

### P4. Collision skin per part, so nothing spans a shut line (authoring; optional [SOLVER])

**Change (authoring only, preferred).**

- `coarse_skin` builds each part's collision triangles from that part's own nodes, and drops every triangle whose nodes belong to more than one part. Each detachable panel's open edges close with no cross-part triangle: a single-sheet panel is a valid two-sided collision surface in a node-vs-triangle model.
- Every node of a detachable part becomes a contacter, not just the coarse skin's nodes. That adds about 300 candidates at Medium, which the self-collision grid handles. Measure the step cost before and after; the result is unchanged except where it was wrong.
- Self-collision groups: a part's nodes skip their own part's triangles (the solver's `self_groups`). Check that the `.truck` path can carry them; that may need an export addition.
- `MAX_COLLISION_TRIANGLES` stays the explicit limit. The build fails loudly if the per-part skin exceeds it, as today.

**[SOLVER] alternative, needs approval:** triangle breaking. A collision triangle stops colliding once any beam along its edges breaks, like BeamNG's default (`disableTriangleBreaking` off). This also fixes torn panels and broken glass, which today leave intact collision triangles. It is a GPU kernel change as well. I recommend the authoring fix first, and triangle breaking only if torn-sheet collision proves visible.

**Acceptance tests:**

1. **No seam triangles.** No collision triangle has nodes from two parts (structural).
2. **The detached door stays outside the body.** Detach the door by command (P1), with the car at rest, and drop it from 0.5 m onto the sill and rocker area.
   - No door node ends up inside the body's closed collision skin.
   - No body contacter ends up inside the door's sheet by more than the collision range (2 cm).
   - The test is an inside-closed-mesh check by ray parity.
3. **Ground.** In every crash test that detaches a part, no node is ever below y = −collision range, and the part comes to rest within 3 s: its kinetic energy falls below its standing-still chatter.

### P5. Physics damage events reach the renderer; small parts become physical (authoring, wasm export, renderer)

**Change.**

- **Damage-state buffer (new).** Export a per-vehicle damage-state buffer next to the node buffer. Per part, it holds:
  - detached: all its tension joints broken
  - per glass pane: broken (its group broke)
  - per lamp: broken
- **Where the buffer comes from.** It is computed from beam `broken` flags in a tiny pass, CPU or GPU. The GPU solver already holds the flags. Reading them is not a solver change, but adding a compute pass to the WebGPU path needs a look at its cost.
- **Glass and lamps.**
  - `showcase-car-render.mjs` replaces the `strain > 0.12` glass discard with the pane's broken bit.
  - The windscreen's crack amount comes from its deform state (glass beams' plastic or elastic strain against the glass failure strain, P6), not from skeleton-triangle strain.
  - Lamps use their broken bit for the lens swap and the emissive switch-off (BeamNG's prop switch-off).
- **Mirrors (and later handles) become physical parts.** Each mirror becomes a 4–7 node part of about 2 kg, like the BX mirror (7 nodes, 2.2 kg), with one mount group. The visual rides it as a rigid piece (the existing `Bind::Rigid` on its own triangle). It falls when the mount breaks, instead of `ANCHOR_BREAK` discarding it.
- **Lamps.** These stay rigid visual pieces on the bumper and body but use the broken bit. A lamp becomes a physical part only if it has a mount group in the physics (BX: 6 nodes and 2.4 kg per headlight, with its own groups).

**Acceptance tests:**

1. **Consistency (exact).** In every crash test, each pane's visual state equals its physics state at every frame (CPU `deform` against the broken bits).
2. **The mirror falls.** A 10 km/h sideswipe on a post at mirror height breaks the mirror group. The mirror's nodes then fall under gravity within 2% of g until they hit the ground. The visual mirror's vertices stay within 1 mm of their rigid frame.
3. **Visual check.** `crash-lab-check.mjs --car showcase` shows the mirror lying on the floor after the sideswipe preset (P10).

### P6. Brittle glass from its material constants (authoring)

**Change.**

- The `glass` grade becomes brittle: deform ≥ strength, so it breaks without yielding, as BeamNG authors brittle parts [RTP].
- Its break strain comes from the material:

  | Glass | Characteristic bending strength σ | E | Break strain ε_f = σ / E |
  |---|---|---|---|
  | Tempered | ≈ 120 MPa | ≈ 70 GPa | ≈ 0.17% |
  | Annealed (laminated windscreen plies) | ≈ 45 MPa | ≈ 70 GPa | ≈ 0.064% |

  Sources: [GLASS-TUE, GLASS-MIF].
- The lumped beam's strength = ε_f × its k·L, with k·L the pane's real in-plane rigidity per lumped beam: E × thickness (≈ 4 mm side glass) × the strip width the beam represents.
- If that k·L exceeds the step's stability floor for the glass nodes' mass, that is a genuine limit. Report it and make the node mass physical (glass is 10 kg/m² at 4 mm). Do not soften the glass silently.
- Each pane is one detacher group: tempered glass fails whole.
- The laminated windscreen has no group. Its beams break and crack locally but the pane stays in place (the interlayer). Visually it uses P5's crack amount.

**Acceptance tests:**

1. **Racking.** Rack the front door's window frame quasi-statically (shear the belt against the header). The pane's group breaks when the in-plane diagonal strain across the pane reaches ε_f within ±10%, and not before.
2. **Windscreen.** In the 56 km/h wall (test `b`), the windscreen may crack but no windscreen node leaves the car: none of its nodes ends up farther than 5 cm from the cowl–header–A-pillar ring. This is the physical meaning of a laminated screen; the regulation is FMVSS 212, for which there is no open data.
3. **No glass breaks in the at-rest test** (test `a`): exact.

### P7. Inner panels and thickness in the visual (parts library, renderer)

**Change.** Model the second skin that BeamNG's art carries:

- **Door:**
  - An inner skin at the door's real section depth (≈ 10–15 cm). It replaces the empty space before the door card.
  - Shut faces (the door's edge walls), plus the body's door jambs and sill flange.
  - A visible intrusion bar inside.
  - Each piece is bound to the door part (`Bind::Part`), so it leaves with the door.
- **Bonnet:** an inner frame lattice under the bonnet.
- **Front end:** inner wheelhouses and aprons, so a bonnet that is gone shows an engine bay rather than a void.
- **Torn edges:** the cut edge is extruded as a narrow wall of the gauge's width, cheaply in the vertex stage from the tear field's gradient, in place of the shaded lip. At normal viewing distance a 0.75 mm wall is sub-pixel. The visible "substance" is the inner skin seen behind a tear, so the inner panels are the priority.

**Acceptance tests (render, through `showcase-car-check.mjs` and `crash-lab-check.mjs`):**

1. **Inner skin behind every tear.** For every torn painted triangle visible in the 56 km/h wall and the T-bone presets, a ray from the eye through the tear hits a non-paint inner surface (the inner skin, jamb or bay) before the far side of the car.
   - The test reads back an ID buffer, not a person's judgement.
   - It holds in at least 95% of torn-edge pixels. The remaining ones see through a fully open hole, which is correct.
2. **Section depth.** The door's inner skin sits at the real section depth: the mean of outer-to-inner distance over the door equals the authored section (±5 mm). This is a structural test in `skin.rs`.
3. **Cost.** The render pass's GPU time (`perf=1`) stays within the visual docs' budget. Measure it and report the delta; nothing is dropped to meet it without the user's decision.

### P8. Wheels that can come off (authoring)

**Change.**

- One `wheel_XX` detacher group per corner over the knuckle's links to the arms, tie rod and strut, as BeamNG authors it.
- The ultimate strength comes from the arm sections' plastic moment, the same rule the side members now use (`M_p = σ_y·Z_p`, commit 7d720867). It does not come from a typed number.

**Acceptance tests:**

1. **Kerb strike.** A 40 km/h lateral strike of a front tyre into a 15 cm kerb keeps the wheel on. Real cars survive kerb strikes at this speed with bent parts, not lost wheels. This is an empirical expectation; IIHS and NHTSA have no open kerb data.
2. **Small-overlap.** A 64 km/h small-overlap rigid-pole impact at the front corner (the IIHS small-overlap geometry, at 25% overlap) may detach the wheel.
   - Measured: the wheel's group breaks only when the arm load passes the section's computed `M_p / arm length` (the test logs both).
   - The wheel then rolls or falls as a physical body. Its kinetic energy decays and it never passes through the ground (P4 test 3).

### P9. Particles: sparks, shards, chips (renderer; no solver change)

**Change.** Port `car-render.mjs`'s GPU particle system into `showcase-car-render.mjs`:

- **Shards** on a pane's broken bit, from P5, spawned over the pane's area.
- **Paint chips** on new tears.
- **Sparks** from metal nodes (body, bumper reinforcement, wheels) sliding on the ground. This uses the node buffer's position and velocity: within the collision range of y = 0 and sliding faster than 2 m/s. No new solver output is needed.
- **Dust** on hard ground hits.

The particles are visual only: they don't feed back.

**Acceptance tests:**

1. **Shard conservation.** The total number of shards equals the broken panes' area × a fixed density (exact count). Shards fall at g within 2% and stop on the floor (y ≥ 0).
2. **Sparks only where metal slides.** In the drop and roll scene (15 m drop, test `e`), sparks appear only on frames where some metal node is within the collision range of the ground and sliding over the threshold (exact, checked from the replay). A resting car shows none.

### P10. Crash-lab scenarios that show detachment (crash lab)

These add to `LAB_PRESETS` in `crash-lab.mjs`. The `tbone` explicit scene already exists in `soft_crash.rs` and `explicit_beams_scaling.rs`. Each is also a CPU test with the assertion given.

| Preset | Setup (from a standard or test protocol) | What it shows | Assertion |
|---|---|---|---|
| `tbone_fmvss214` | FMVSS 214 MDB: a 1,368 kg moving deformable barrier crabbed 27°, at 54 km/h into the driver's side | Doors pushed in, latches hold | Latch and hinge groups intact (FMVSS 206 S4.1.1's purpose). Momentum ledger as test `b` |
| `tbone_iihs2` | IIHS side 2.0: a 1,900 kg barrier at 60 km/h | Heavier hit: possible hinge-pillar crush and glass breakage | Side glass breaks only if its frame strain passed ε_f (P6). Doors still latched |
| `tbone_car_80` | Car to car, striker front-first, 80 km/h, at the B-pillar | Pillar tear-out; the door may come off when its hinges' longitudinal load passes the P1 ultimate | Each hinge group breaks only after its logged load passed the ultimate. A detached door obeys P4 |
| `sideswipe_post` | 50 km/h, 10° glancing strike on a rigid post at door height | Door peeled rearward: hinges loaded longitudinally (the direction doors really come off) | The door separates only if the longitudinal hinge load passed 1.5 × 11 kN. The mirror falls (P5) |
| `rollover_drop` | Existing 15 m drop (test `e`), plus a 2 m roll-off | Bonnet and boot lid pop, panels hang by one hinge, glass shatters, sparks | Ground never passed (P4). Energy ledger |
| `kerb_40` | 40 km/h lateral into a 15 cm kerb | The wheel stays on | P8 test 1 |
| `small_overlap_64` | 64 km/h, 25% overlap, rigid pole or barrier at the front corner | Wheel off, fender and bumper peel mount by mount | P8 test 2. Bumper mount groups break in order of their logged loads |
| `door_pull` | Test `g` as a lab demo, with a ramped pull that reads out force | The door lets go at its FMVSS-derived ultimate | P1 test 2 |

### P11. Paint damage and dirt (renderer; later)

BeamNG has no built-in paint damage. This is Rubble's own design.

- A per-vertex accumulation buffer, written by a compute pass, records scrape (contact sliding distance near the ground or another car) and chip (tear proximity).
- The paint shader mixes in primer and bare metal by those amounts.

**Acceptance test:** the scrape accumulator equals the integral of sliding speed over contact time at each vertex (exact, checked against a CPU replay). Untouched panels read zero.

### P12. Tyre deflation [SOLVER]

This needs pressure groups (a closed triangle volume with gas pressure), which BeamNG and Rigs of Rods both have. Our authored tyres are beam-only. The change is a solver feature and needs approval.

**Acceptance test:** an intact tyre's deflection under the corner load matches the inflation-pressure relation, contact patch area ≈ load / pressure, within 15%. A punctured tyre (its pressure dropped by command) settles to the rim within the leak time set.

### Order and rationale

1. **P1–P4 first.** They are the physics that decides whether a part *can* come off and what happens after. P1 without P4 would let a detached door snag on ghost seam triangles. P3 is cheap and improves every crash's look by giving panels momentum.
2. **Then P5–P7, the visuals.** They depend on physics state existing.
3. **Then P8–P10.** P10's presets can land early, each as its item's demo, beginning with `door_pull` and `sideswipe_post`.
4. **P11 and P12** are later. P12 needs the user's approval.

---

## 4. What can't match BeamNG within Rubble's rules, and options

- **Static contact is node-only, which BeamNG shares** [DOC-nodes]. This is not a gap against BeamNG.
  - Both engines let a sharp world edge slip between nodes.
  - BeamNG's answer is node placement (3–3.5 cm from triangles, sparse undersides). Ours is the same.
  - **Options:**
    - Place nodes along the edges most likely to strike: sill, bumper corners, wheel-arch lips.
    - Model world obstacles with rounded edges.
  - Adding vehicle-triangle-vs-world contact would be a [SOLVER] change, and BeamNG does not do it either.
- **The 2 kHz explicit step bounds stiffness per node mass.** A beam's stiffness must stay under about m·(2/h)²: roughly 4.8 MN/m for a 0.3 kg node at h = 0.5 ms, less when many beams share the node.
  - BeamNG lives under the same bound: the BX door's skin springs are 1.1 MN/m on 1 kg nodes.
  - **So neither engine has real 0.75 mm sheet bending stiffness in its physics.** Substance comes from mass, depth and bearing (§1.9), all of which we can author.
  - **Options:**
    - Physical panel mass (P3), which raises the stiffness allowance losslessly.
    - Depth nodes and inner frames.
    - Supports.
    - Coarsening the timestep is not an option without the user's decision; it is lossy.
- **Collision triangle limit.** It is 3,000 per vehicle (`MAX_COLLISION_TRIANGLES`). BeamNG's per-vehicle caps are 3,000 nodes and 15,000 beams. A per-part skin (P4) fits within today's 2,900 budget at Medium. If it doesn't at High, the build fails loudly, as now. Raising the limit is a [SOLVER] decision.
- **Fine creases between nodes.** At 20–25 cm node spacing a crease is a polyline of beams, the same as BeamNG's. BeamNG's sharper visual creases come from art density and flexbody binding. Ours binds to one triangle per vertex.
  - **Options (visual-only and lossless):**
    - Bind vertices to 4 nodes (a tetrahedral frame) for smoother bending.
    - Add a normal-perturbation crease pass driven by plastic strain at the binding nodes.

    Both are later work, after P7.
- **Deformation-triggered events at no cost.** BeamNG warns that deform triggers are costly. Ours would read beam state each frame for the damage buffer. That cost is a compute pass, so measure it.
- **Clean-room limits.**
  - BeamNG's closed C++ core means there are details we can't know: the exact triangle-breaking rule, coupler internals and puncture logic. The plan uses only documented behaviour.
  - Where the docs are silent, as on detached-part collision, the plan states the physical rule we test instead.
- **Forward-only destruction** (AGENTS.md) is compatible with all of the above. Every break is decided within the step by force against strength. No item rewinds.

---

## 5. Open decisions for the user

1. **Hardware margin.** I recommend 1.5 × the FMVSS 206 minimum per hinge and latch system (P1). Real ultimate strengths need compliance reports, which need permission to download.
2. **Triangle breaking [SOLVER]** as the alternative or complement to the per-part collision skin (P4).
3. **Tyre pressure groups [SOLVER]** (P12).
4. **Adding a damage-state compute pass to the WebGPU path** (P5). This is not a solver rule change, but it is new GPU work per frame.

---

## 6. Sources

**BeamNG documentation**

- [DOC-slots] https://documentation.beamng.com/modding/vehicle/sections/slots/
- [DOC-partslot] https://documentation.beamng.com/modding/vehicle/intro_jbeam/partslotsystem/
- [DOC-beams] https://documentation.beamng.com/modding/vehicle/sections/beams/
- [DOC-support] https://documentation.beamng.com/modding/vehicle/sections/beams/support/
- [DOC-bounded] https://documentation.beamng.com/modding/vehicle/sections/beams/bounded/
- [DOC-aniso] https://documentation.beamng.com/modding/vehicle/sections/beams/anisotropic/
- [DOC-pressured] https://documentation.beamng.com/modding/vehicle/sections/beams/pressured/
- [DOC-tips] https://documentation.beamng.com/modding/vehicle/intro_jbeam/jbeamtips/
- [DOC-nodes] https://documentation.beamng.com/modding/vehicle/sections/nodes/
- [DOC-couplers] https://documentation.beamng.com/modding/vehicle/sections/nodes/couplers/
- [DOC-fire] https://documentation.beamng.com/modding/vehicle/sections/nodes/fire/
- [DOC-triangles] https://documentation.beamng.com/modding/vehicle/sections/triangles/
- [DOC-flexbodies] https://documentation.beamng.com/modding/vehicle/sections/flexbodies/
- [DOC-props] https://documentation.beamng.com/modding/vehicle/sections/props/
- [DOC-wheels] https://documentation.beamng.com/modding/vehicle/sections/wheels/
- [DOC-split] https://documentation.beamng.com/modding/vehicle/vehicle-art/final-mesh-splitting/
- [DOC-modeling] https://documentation.beamng.com/modding/vehicle/vehicle-art/modeling/part-creation/
- [DOC-glasstex] https://documentation.beamng.com/modding/vehicle/vehicle-art/texturing/glass-damage-textures/
- [DOC-arch] https://documentation.beamng.com/beamng_tech/architecture/

**BeamNG papers, patch notes and forum posts**

- [RTP] BeamNG.tech technical paper (2021): https://beamng.tech/blog/2021-06-21-beamng-tech-whitepaper/bng_technical_paper.pdf
- [PATCH-0.13] https://beamng.com/game/news/patch/the-automation-collaboration-version-0-13-released
- [PATCH-0.5.3] https://beamng.com/game/news/patch/version-0-5-3-released
- [BLOG-sneak6] https://beamng.com/game/news/blog/development-sneak-peek-6
- [FORUM-highpoly] BeamNG staff on node and triangle counts: https://www.beamng.com/threads/how-do-i-make-a-high-poly-car-mod.67584/
- [MOD-scratch] https://beamng.com/resources/dynamic-scratches-mod.38613/

**Rigs of Rods**

- [RORDOC] Rigs of Rods truck file format: https://docs.rigsofrods.org/vehicle-creation/fileformat-truck/

**Standards and materials**

- [FMVSS-206] 49 CFR 571.206 (door locks and retention): https://www.law.cornell.edu/cfr/text/49/571.206
  - S4.1.1.1–2: latch 11,000 N and 9,000 N.
  - S4.1.2.1–2: hinge system 11,000 N longitudinal and 9,000 N transverse.
  - S4.1.1.4: 30 g inertial load.
- [FMVSS-214] 49 CFR 571.214: https://www.law.cornell.edu/cfr/text/49/571.214 (already in `crash-reference-targets.md`).
- [GLASS-TUE] Strength of annealed, heat-strengthened and fully tempered float glass: https://research.tue.nl/en/publications/the-strength-of-annealed-heat-strengthened-and-fully-tempered-flo/ (characteristic strengths 45, 70 and 120 MPa).
- [GLASS-MIF] Glass elastic modulus about 70 GPa: https://www.makeitfrom.com/compare/Annealed-Soda-Lime-Glass/Toughened-Tempered-Soda-Lime-Glass
- [GAUGE] Outer panel gauge 0.7–0.9 mm (secondary source): https://thedailyautomotive.com/how-thick-is-car-sheet-metal/
- [PATENT-5605371] The inner stamping is thicker than the outer: https://patents.google.com/patent/US5605371

**Local data (read only, not committed)**

- BX JBeam: `/Users/glavin/Development/rubble-beamng-parity/inspiration/beamng-bx/vehicles/bx/*.jbeam` (doors, body, hood, fenders, glass, bumpers, suspension, lights, mirrors).
- BeamNG template car: `/Users/glavin/Development/rubble-beamng-parity/inspiration/beamng-template-car/current/vehicles/template_car_sedan/`.
