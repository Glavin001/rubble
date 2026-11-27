# Generated vehicles: lessons, conclusions and decisions

A reference so this research never has to be repeated. It records what we learned building crash physics for generated vehicles between 2026-11-07 and 2026-11-12: what was measured, what was concluded, what was decided, and what's still open. Add to it; don't rewrite history. Each entry says whether it is a **Fact** (from code or docs), a **Measurement**, a **Decision** (the user's) or a **Hypothesis**.

**Where the detail lives:**

| Topic | Document or branch |
|---|---|
| How Rubble vehicles work (the model) | [deformable-vehicles.md](deformable-vehicles.md) |
| The pipeline, how to run it, results tables | [generated-vehicles.md](generated-vehicles.md) |
| BeamNG cars measured (five cars, per-part bands) | [beamng-authoring-census.md](beamng-authoring-census.md) |
| BeamNG structure study: sourced guidance, five cars measured, glossary, and the Step 2d body specification (§10) | [beamng-structure-study.md](beamng-structure-study.md) |
| Real crash targets (NHTSA, FMVSS) | [crash-reference-targets.md](crash-reference-targets.md) |
| Known bugs (B1–B3, S1, S2, F1, F7…) | [handoff/known-bugs.md](handoff/known-bugs.md) |
| Soft bodies in the rigid World (the engine side) | [handoff/soft-rigid-world.md](handoff/soft-rigid-world.md) |
| The current plan | `~/.claude/plans/dazzling-honking-iverson.md` (local) |
| Raw measurements (not in git) | `~/Development/rubble-artifacts/{sedan-stiffness,vehicle-gen-physics,solid-parts,etk-strength,ring-phase-through}/` |

**Branches** (as of 2026-11-12; none merged):
- `feat/soft-rigid-world`: PR #105, the base.
- `feat/vehicle-gen-physics`: the rigger.
- `fix/c2c-interpenetration`: the closed collision skin.
- `feat/sedan-stiff`: the BeamNG-style closures.
- `feat/solid-parts`: solid visual parts.
- `diag/sedan-stiffness`, `diag/ring-phase-through`: probes.

---

## 1. Principles (Decisions, 2026-11-12)

1. **One definition, everything derived.** A part is described once in code: shape, section, gauge, material, joints. From that come its visual solid, nodes, masses, beams, joints, collision skin and skin binding. Visuals and physics can't drift apart.
2. **Parts are closed solids** (outer, edges, inner). Each must look and behave right on its own (parts lab: drop, throw, shoot), and better assembled.
3. **A real body structure.** A unibody and mechanical parts as visible solids, with physics one-to-one on them. No hidden invented structure.
4. **Nothing faked.** Real masses and real yield and break forces; nothing unbreakable by decree. The proof is a slow press between plates until the car is flat.
5. **Stiffness at BeamNG (ETK) level,** measured side by side. Nothing that resembles jello. The step is fixed at 2 kHz.
6. **Wheels:** keep the fast raycast wheel and suspension, for scale. Hubs, arms, mounts, rim and hub cap are real, breakable parts. A crushed body moves the wheel.
7. **Glass:** collisions break it, but only visually for now.
8. **Interior parts** have breakable mounts.
9. **Evidence first.** "Looks wrong" is a hypothesis. Fix authoring, never the solver, to suit one car (`physics-evidence-first` skill).
10. **Prove one car end to end** (the Rust generator's sedan). The other designs reuse the parametric rigger.

## 2. The stiffness law at a fixed step (Fact)

For a member with modulus E, density ρ and node spacing L:
- wave speed c = √(E/ρ);
- node mass m = ρAL;
- beam stiffness k = EA/L;
- yield force F = σ_y·A.

The explicit step is stable only while ω·h ≲ 2, and ω ≈ 2c/L. That gives **L ≥ c·h**.

- **At h = 0.5 ms, real steel (c ≈ 5,000 m/s) would need 2.5 m beams.** Every 2 kHz node–beam engine runs softened steel. Our skins carry about 160 m/s.
- **Per node:** k ≲ m·(2/h)²/n, with n spring ends on the node. **Heavier nodes allow stiffer beams.**
- **What can stay real** at any spacing: mass, yield force, break force.
- **What cannot:** wave speed, and so stiffness. Record it as one explicit softening factor per part.
- **Bending stiffness** comes from depth: inner nodes a distance d behind the skin give about k·d².
- **Multi-rate stepping** (finer regions at 4–8 kHz, soft or idle regions at 0.5–1 kHz) would remove the conflict. **Decision: not now; assume 2 kHz.**

## 3. Why a generated car was jello (Measurement, `diag/sedan-stiffness`)

Sedan v2 against the ETK, CPU reference:

| | Sedan v2 | ETK |
|---|---|---|
| Nodes: body / door / hood | 346 / 47 / 48 | 124 / 13 / 23 |
| Node mass (kg): body / door / hood | 1.6 / 0.5 / 0.28 | 4.0 / 1.7 / 1.5 |
| Closure stiffness left by the stability pass | 35–56 % | not cut |
| SUPPORT share | 1.9 % | 7.1 % |
| First modes (Hz): door / hood / lid | 7.8 / 8.0 / 7.8 | 34.9 / 12.5 / 27.0 |
| Kerb wobble, body (mm) | 0.87 | 0.09 |
| Settle (s) | 1.4–2.9 | 0–0.6 |

Real lids measure 17.7–41.5 Hz.

**Conclusions:**
- **More nodes made the car softer, not more detailed.** Light nodes force the stability pass to strip stiffness. It even stripped the door inner frame that was meant to give depth.
- **Seals and stops (SUPPORT) are the main out-of-plane restraint of closures.** Counted as two-way, they'd move closure modes from 5–8 Hz to 21–31 Hz.
- **Low modes ring for seconds** at BeamNG's damping ratios. The ETK's same ratios die quickly because its modes are 2–5× higher.
- **A long visual binding reach amplifies body wobble 2–3× on screen.**
- **Detail belongs in the skinned mesh,** not in node count.

**After rebuilding the closures in BeamNG style** (`feat/sedan-stiff`: door 17 nodes at about 1.1 kg, hood 24, lid 18; seals; 3×3 hinges):

| | Before | After | ETK |
|---|---|---|---|
| Closure stiffness kept | 35–56 % | 79–92 % | — |
| Door mode (Hz) | 7.8 | 15–16 | 35 |
| Body wobble (mm) | 0.87 | 0.41 | 0.09 |
| Settle (s) | 1.4–2.9 | 0.7–1.4 | 0–0.6 |

**Not at ETK level.** The body lattice is unchanged (346 light nodes, 28 % of its stiffness stripped), SUPPORT is at 5.6 %, and the hood and fenders lack depth.

**Conclusion:** rebuild the body at about 130 heavy nodes on real members, with internal stops. That is the same work as principle 3.

**Practical findings:**
- A 13-node door failed the static door test (7.7 kN), so a door needs a closed box: 17 nodes with an inner panel.
- Four near-parallel hinge beams let a door swing on its latch; use 3×3.
- One under-braced node (`rrh0`, 3.8 Hz) moved 30 mm at rest.
- Use `sedan_stiffness MODE_NODES=1` to find such nodes.

## 4. Collision lessons

- **(Decision) The engine owns "solids never interpenetrate"; authoring owns a closed collision skin.** Node spacing sets how a car deforms, never what it touches. Density can't fix slip-through: there is no smallest obstacle, every node costs at 2 kHz, and lighter nodes mean softer cars.
- **(Measurement) The head-on pass-through was mostly authoring.**
  - **How bad:** 23 % of the frontal area had no collision triangle. 32 % had its first triangle more than 0.6 m behind the visible surface. 224 of 264 node entries came through holes.
  - **Causes in the skin builder:**
    - inner nodes were used for the skin;
    - a tilt test checked one render triangle, which sat on a fold;
    - quads got one triangle or none;
    - there was no node row at the grille;
    - patches stopped short of seams;
    - wheels had no geometry.
  - **Fix** (`fix/c2c-interpenetration`):
    - skin from surface nodes only;
    - averaged normals;
    - both triangles per quad;
    - seam bridges;
    - a grille row;
    - wheel cones.
  - **Result:** overlap 1,256 L → 98 L.
  - **The lint had missed it,** because its coverage rule ignored front and rear faces. **Check every face.**
- **(Measurement) Seam bridges must not die with one bolt.** Tying each to its nearest bolt killed 19 bridges when a single bolt broke.
  - **Open:** a triangle should die only when its part has really separated. That needs more than one break group per triangle.
- **(Measurement) Light contact-only nodes plus surface contact can blow up.** Pole-lane nodes of 0.15 kg, used as triangle corners, injected 3.7e29 J in the side-pole test (known-bugs S1). Generated cars use no lanes.
- **(Measurement) Remaining engine-side gaps:**
  - vehicle contact is node against triangle only (about 30 node entries per car came through a live triangle, and edges cross);
  - raycast tyres have no collision shape (the outer 8 cm overlaps);
  - contact against moving World bodies has no penetration push-back (B1: bricks sink 25 cm).

  Each is the user's decision.
- **(Measurement) Surface contact** (triangles and edges against shapes):
  - a static brick stays fully out (0.000 against 21 cm);
  - cost is 0.20 ms a step for one car;
  - it's expensive with many cars (32-car town: 37 ms against 17);
  - it isn't enough against moving bodies until B1 is fixed.

## 5. Nose lift in the head-on (Measurement; known-bugs S2)

- **Symptom:** after the collision skin was closed, both noses rose (front hubs 0.27 → 0.93 m). Real head-ons pitch nose-down.
- **Not the cause:** hoods ramping on each other (vertical contact impulse under 0.6 of 20 kN·s).
- **Cause:**
  - 65 % of the contact impulse entered above 0.6 m, against a centre of mass at 0.57 m;
  - the hood and fenders carried about half the load, and a hood beam yielded at about 10 kN against a rail beam's 2.8 kN;
  - the front structure rotated nose-up about the strut towers;
  - the floor struck the ground, and the ground threw the car up.
- **Reduced to 0.46 m** by making the rails and crash cans stronger (×4) and the upper band weaker. On that lattice, further adjustment trades against overlap.
- **Conclusion:** the frontal load path (bumper beam → crash boxes → rails, at bumper height) must be real geometry, stronger than the hood and fender band. This is part of the body rebuild.
- **Hypothesis:** the remainder may need edge contact between cars.

## 6. Solid visual parts (Measurement, `feat/solid-parts`)

- **Method that works:** inner face on the skin's own topology (one inner vertex per skin vertex) pushed back by a depth field, plus edge walls round every border. All 36 solid parts close; 439k triangles; 352 ms build in wasm.
- **Lessons:**
  - **Smooth the depth field.** Sampling it per vertex on 20 mm cells stair-steps diagonal ribs. Smooth it on the mesh and limit its slope.
  - **Wall rows can't all scale together.** Where a wall is shallow or turns tightly, drop rows down to a plain wall.
  - **Pillars and roof rails** need a channel section: deep jamb walls with a shallow face between. Offsetting thin rows along one direction made pillar walls lean through the doors.
  - **Corner clearances.** Glass and lens slabs need their own clearance at tight corners, because rebates shrink there.
  - **Some defects were in the sheet skin already** (the lens crease at the nose corner, the rear door-card fold). Fix those at the source, not in the solid.
  - **Don't feed solids' bare outer skins to the rigger untested.** A hood node drifted 16 mm. Until the rigger reads the outer tag, it uses the sheet build as its model.
- **Open:** 2,149 triangles past 6:1 (target about 600); glass against rebate at corners; the sail tip; grille plates.
- **Process lesson:** inspect every rendered still. Twice, half were left unread.

## 7. What a BeamNG car has that ours lacked (Fact; the ETK's model files, read as data)

- **A modelled unibody:** floor pan, tunnel, firewall, engine bay, strut towers, inner arches, sills, jambs.
- **Separate mechanical parts,** each also its own JBeam part: subframes front and rear, upper and lower arms, hubs, shocks, steering rack, strut brace, engine, transmission, driveshaft, differential, exhaust, radiator, fuel tank, seat rails.
- **The ETK's JBeam part files** number about 33: body, doors, hood, trunk, fenders, bumpers, glass, lights, interior, engine, engine mounts, transmission, differential, exhaust, fuel tank, radiator, suspension F/R, brakes, and more.

Our first sedan was an outer shell with panels and placeholder boxes; its structure existed only as invisible nodes. See principle 3.

## 8. BeamNG against Rigs of Rods, and what "matching BeamNG" can mean (Fact / Measurement)

- **Same family, different solver.** JBeam content is tuned for BeamNG's closed solver. A JBeam car converted to `.truck` broke at rest in Rigs of Rods.
- **Equivalence to BeamNG can't be proven without recording BeamNG itself.** Our baseline is the public documentation plus measured properties of the best cars, judged in our harness against real crash data.
- **The ETK in our solver** (56 km/h wall): 0.55 m crush, 67–70 g, restitution 0.37, residual 233 mm. That's stiffer and springier than NHTSA sedans (0.62–0.75 m, 34–47 g, 0.13–0.20), because four front subframe mounts stay elastic in its data. BeamNG cars are tuned to look right, not derived from materials. That's a weakness we can beat.
- **Loose 25 kg bricks peak at about 100 kN.** The ETK's front yields near 600 kN, so it barely crumples through a brick wall. The coupling was correct.

## 9. Driving and drag (Measurement)

- **Per-node drag (the Rigs of Rods rule, 0.05·|v|·v on every node) scales with node count.** A 987-node car gets 127× real drag and tops out at 41 km/h. Generated cars use triangle aero sized to the design's C_d (0.30); the sedan reaches 213 km/h.
- **The soft car's raycast wheels** are the rigid vehicle model applied as forces on hub nodes inside the 2 kHz step. In the World their rays also see rigid bodies and other cars and push back on them (F7, fixed 2026-11-18: `wheels_hit_bodies`).
- **Camera jitter** on the drive page is suspected to come from readback pacing, not physics (body shake 0.15–0.25 mm).

## 10. Process lessons

- **Rest fidelity first.** A generated car must sit within 1 mm / 1° of its art model, parked, before crash numbers mean anything.
- **Lint with a synthetic failing car per hard rule.** A rule without a failing test will miss things; the coverage rule did.
- **Stop and report when a bar isn't met.** Don't render clips of a car that fails its exit criteria.
- **One agent per set of files.** Queue work that touches the same rigger or generator files; parallel agents only on disjoint files.
- **Agents run out of context on large geometry tasks.** Give ordered, separately committed items, and require a push after each.
- **Clips so far are CPU reference renders (Blender),** not the web page. Say so.
- **GPU, Chrome and determinism lessons** are in [handoff/soft-rigid-world.md](handoff/soft-rigid-world.md) and the skills in `.cursor/skills/`.

## 11. Open questions and next steps (2026-11-12)

1. **The BeamNG structure study** (public docs plus five cars), leading to the body specification.
2. **Body-in-white and mechanical parts** in the generator, with the body lattice (about 130 heavy nodes) placed on them. Aim: ETK-level stiffness and a real frontal load path.
3. **Binding:** both faces of a solid part to one node set, with thickness and poke-through metrics. An optional crumple-detail visual layer, off until judged from stills.
4. **Parts lab** (each part alone), then the press test (nothing faked).
5. **The generated sedan in the web demos;** then one PR. PR #105 merges first (the user's go-ahead).
6. **Engine decisions pending:**
   - B1 penetration response;
   - edge contact between cars;
   - a tyre collision shape;
   - S1;
   - surface contact as the default;
   - the stiffness row cap (BeamNG content runs at rows 1.1–4; ours is capped at 1.0).

## 12. The body rebuilt from definitions: first findings (2026-11-13, `feat/sedan-body`)

State and tables: [generated-vehicles.md](generated-vehicles.md), "Body structure and mechanical parts". The definitions and the shell's lattice exist and are tested. The rigger does not use them yet, so there is no stiffness or crash measurement of the new body.

- **(Fact) There is no room for a second pillar or roof-rail box.** At the roof rail the shell's solid is already 30 mm deep, the headliner sits 40 mm under the skin, and the door frame closes beside it. The upper A-pillars, C-pillars, roof rails, headers and bow are that section; their nodes sit inside it. Redrawing the shell's own walls as channel sections (section 6) is still open.
- **(Fact) The floor pan's skin is the shell's.** Its solid has an end wall 30 mm high at each door line. Members on the pan stop inside those walls; the heel kick stands on the rear one.
- **(Measurement) A wheel-well liner with a small centre hole blocks every arm.** The inner wall's opening was 0.55 r (0.206 m). Arm legs from the rail's stations cross that wall at 0.21–0.22 m. The solid build opens it to 0.74 r.
- **(Measurement) The rear rail has to arch over the half-shaft.** With the floor step where the bench allowed, the rail cut the shaft. The step moved 3 cm forward and the rail rises with it in upright sections.
- **(Measurement) Inner bumper ribs reach 71 mm behind the rear skin.** The rear impact beam sits 80–130 mm in, and the rails stop behind it.
- **(Measurement) A k-nearest lattice leaves roofs and pillar middles flat.** After bracing each node to ten neighbours, the roof bow's centre had a direction spread of 0.003 and four nodes were in no tetrahedron. Each flat node needs a fourth corner along its weakest axis, and the study's six long cabin braces are exactly enough on the reference car when the flattest nodes take them first.
- **(Measurement) The zone table's springs sit under the node budget in the cabin and tail.** With the table's values the median Σk/m was 5.5e6 s⁻² and the shell's ω·h 1.64. Letting each beam rise to the budget (8.5e6 per node, at most 7e6 N/m per beam) gives a median of 8.5e6 and ω·h 1.73 for the shell alone, with no global softening needed.
- **(Fact) Where the stiffness cap is.** `Car::assign_springs` (`engine-softbody/src/authoring/gen2/build.rs`) has two limits. The row cap is `row_cap()` × `ROW_BOUND` (line 252: `GEN2_ROW`, default 3.0; `ROW_BOUND` 4.8). The whole-car limit is `OMEGA_H` 1.85 (line 282), enforced by power iteration that softens the beams carrying the top mode by 3 % a round. On this branch the row cap is 3.0, not 1.0, so the binding limit is the whole-car 1.85. A row of 1.0 is Σ(h²k + 2hc)(1/m + 1/√(m·m′)) = 4.8, which for equal masses and no damping is Σk/m = 9.6e6.
- **(Measurement) The front's load path by height depends on the band's edge.** Rule 13's low band ends at 0.45 m, the ETK's rail and engine height. With rails centred at 0.47 m half of each rail counted as middle band (39 % low for the shell alone). With rails at 0.44 m and the mechanical parts counted, the reference car is at 59 % low and 2 % high. The short preset is at 52 %.
  - **Open (the user's decision):** whether the band should follow the car's own centre of mass.
- **(Measurement) The merged base fails one existing test.** With the solid parts and the census rigger together, `gen3_rig`'s wall test gives stretch p99 1.64 against 1.6 (`VG_SHEET=1`: passes). Each branch passed alone. It is the binding of the solids' walls (plan step 2b), not the structure.
- **Process:** zsh expands `$M:u` as a modifier. Write `${M}:view` when a shot list is a shell variable.

## 13. The new body crashed, against the ETK (2026-11-13, `feat/sedan-body`)

Tables: [generated-vehicles.md](generated-vehicles.md), "Crash character and the frontal crashes against the ETK".

- **(Decision, the user's) The primary bar is the ETK's look and feel; the real crash bands are the guard.** The crash character is one setting, `Physics.crash`, with presets `beamng_feel` (default) and `real_crash`.
- **(Measurement) The sedan as defined is nearer NHTSA's wall bands than the ETK is,** and crushes 15–20 % deeper than the ETK with 40–70 % more residual crush. Front members at 1.8 times their defined crush force, towers and cabin at 1.5 and panels at twice the census yield strain bring the three frontal scenes within 25–60 mm of the ETK's dynamic crush.
- **(Measurement) Part of an "over-crush" seen on the page was the contact mode.** With node contact only, a 40 % barrier's edge passes between nodes and the offset's cabin travel is 900 mm against 720 mm with surface contact. Generated cars have a closed collision skin; the page turns surface contact on for them.
- **(Measurement) Measure intrusion in the cabin's own frame.** An offset hit turns the car about 7°; against the rear floor's centre that reads as 85 mm of "intrusion" on the struck side and −60 mm on the other. `sedan_frontal` fits the rear floor's nodes rigidly first.
- **(Measurement) A side impact needs a carrier.** A car launched sideways on raycast wheels is stopped by its tyres' lateral grip before the pole.
- **(Measurement) Mass a part by its geometry.** A 136 mm centre cap at a full wheel cover's 0.6 kg broke its own clips in every frontal crash, at the far wheels too.
- **(Measurement) An occupant's mass belongs where the belt takes it.** A quarter of the driver on each top corner of the seat's back folded the back 240 mm forward in the wall.
- **(Measurement) "The exhaust's skin" was not the parked normal's fault.** Print the worst mesh (`GEN3_WORST=1`) before fixing: it is a hub cap following the wheel's frame nodes.
- **(Measurement) A stronger front breaks the fenders' bolts.** Both front bolt groups break in the full wall at `beamng_feel`; none did as defined. Hardware has to be checked whenever the character changes.
- **(Fact) The page's beam count is the solver's records:** 3,467 JBeam beams + 2 aero records per collision triangle (726) + 2 per torsion bar (24) + 7 couplers + 4 raycast-wheel ties = 4,228.
- **Process:** a web build reads the sources while it runs; do not edit Rust until its host and wasm compiles have finished. Never use the shared Browser pane while the user tests; `engine/crates/engine-webgpu/page-shot.mjs` opens a URL headless.

## 14. The car that flew apart: energy bombs (2026-11-13, `feat/sedan-body`)

Record: [handoff/known-bugs.md](handoff/known-bugs.md) S3; spec §39.6; `examples/sidepole_sweep`, `examples/spin_probe`; `tests/sedan_energy.rs`, `tests/centred_damping_objective.rs`.

- **(Measurement) Two causes, found with the energy ledger on the CPU reference.** The page's reports (side pole 186 km/h, a 31 m drop at 189 km/h, a 42° wall at 106 km/h, a 25 km/h bump in the yard) all reproduce: 20 of 59 runs of the scene matrix diverge on the 32cbc954 export.
- **(Fact) A BOUNDED beam's limit spring acts on its whole extension.** A "gapped stop" written as a bounded beam with no base spring is a force wall where it engages (2e6 N/m × 0.34 m over a 2 cm zone) and a tether at its long bound: 100 lengths is 55 m, and a torn-off wheel gets there in 0.7 s. Write a stop as a push-only spring whose rest length is the length it engages at (SUPPORT, `beamPrecompression` = that share), inside the step's bound, that lets go when stretched.
- **(Measurement) The solver's centred damping added energy to anything that spins** (a hub cap: 3.75 J to 1,800 J in 0.75 s). Its correction read a turning beam as shortening. The objective form fixes it. Since 2026-11-13 the GPU kernels carry it too (bit for bit with the CPU reference, per-vehicle frame, contact groups and wide passes) and it is on by default (`ExplicitBeamsSettings::centred_objective`; spec §39.6 "The kernels"), so the page no longer needs it switched on; `centred_objective = false` gives the old form for comparison.
- **(Measurement) With both, 82 runs never exceed their starting energy** (walls ±20/42/60° at 56/106/150 km/h, side poles 32-320 km/h at two places, drops 5/15/31 m at 0/100/189 km/h, surface contact on and off). What is left are single-step rises of the discrete energy, in the beams: 0.3-4 kJ where an engine mount's bound is crossed in one step (the lint's `bound_wall` gives 1.3 kJ a beam), and up to 28 kJ at 320 km/h on a 0.47 kg body edge node (`ebody13`): the light edge nodes on soft ties that the jello work removes.
- **(Decision, the user's) Catch the class in the lint,** statically (`bound_wall`, `tether`) and dynamically (tear every part off, fling and spin it; the ledger never rises).
- **(Measurement) A hub cap leaves at a 25 km/h bump onto its wheel** (9 clips of 600 N a beam, the study's value). Not changed: open whether real clip retention holds at that load.
- **Process:** write the test from the physical invariant first and commit it failing. Set energy bounds against the plain step on the same body: single precision alone moves an undamped body's energy by 0.3 % a second.

## 15. Jello and collisions inside one car (2026-11-13, `feat/sedan-body`)

Tables: [generated-vehicles.md](generated-vehicles.md), "Jello against the ETK, and parts that collide inside one car".

- **(Measurement) Ask each mode which springs hold it.** `sedan_stiffness` with `MODE_BEAMS=1` prints a mode's strain energy by spring, `MODE_NODES=1` its shape. The doors' softness was one spring: the latch's coupler (55 % of the energy), not the door's lattice.
- **(Fact) A coupler is as stiff as its nodes are heavy.** `coupler_rates`: k = 0.16·μ/h², ω·h 0.4 on the reduced mass. A latch on two light tag nodes is a soft spring. Give the latch and striker nodes their hardware's mass, the striker's from the member it stands on.
- **(Measurement) "Nearest n" breaks symmetry by rounding.** A hub cap's nine clips took one of two equally near frame nodes per corner; two corners shared one and the cap was a pendulum (2–22 Hz, a different value per wheel, never at rest). A mount's beams are chosen by a rule that is the same on every wheel, or all of them are taken.
- **(Measurement) The order hardware is built in is a rule.** Seals made before bolts took the bolts' pairs (`has_beam`): a fender leg hung on a push-only beam.
- **(Measurement) A node whose ties lie in one plane swings across it, whatever their stiffness.** Check the spread before making the node, not after (`EDGE_SPREAD`).
- **(Measurement) A tie sized by a sheet rule on a light node is stripped by the stability pass.** Size it by the node's budget.
- **(Measurement) Removing the body's edge nodes removes its collision skin's corners** (126 triangles; coverage p99 0.16 → 0.25 m). A stiffness fix has to be checked against the lint, not only the modes.
- **(Measurement) The kerb gap to the ETK is input and mounting, not panel stiffness.** Panels ripple a third as much about their own fit as about the body's; the body's worst nodes are the strut towers, loaded through a rigid strut link where the ETK has its spring.
- **(Measurement) A scene has to drive the door into its opening to show a door passing through the cabin.** A pole to the ground and a ram over the sill both push the sill and pillar through the door's stops, and the seat moves with the floor. A second car into the door at 50 km/h showed it: the sill's top edge and the A-pillar through the door's triangles.
- **(Fact) Self-collision is node against triangle, and a pair inside 2 cm at spawn never collides.** Give every solid faces on its own nodes (hulls for seats, panel, engine, tank; boxes for doors; faces for the sills, floor, tunnel and firewall) and make its nodes contacters. Edges still cross between nodes, and neighbouring panels' edge nodes at a 5 mm shut line are excluded for good: both are the solver's, not authoring's.
- **(Measurement) Interior collisions changed the frontal crashes for the better.** With the hood landing on the engine the wall's residual crush fell from 299 / 215 to 239 / 169 mm (the ETK 228 / 192) and the pole's flipped triangles from 616 to 80.
- **(Measurement, 2026-11-15) A part's one damping value is not a damping ratio.** The parked sedan's empty passenger seat rocked on its feet at 12 Hz and decayed at 0.6/s (modal ζ 0.008, 0.07 mm still 4–5 s after spawn; at 10 s 0.004 mm: a slow decay, not a limit cycle: no beam crossed an engage point, no plasticity, no contact). Its definition gave every beam 38 N·s/m, on springs set by the node budget (2.6–3.8e6 N/m): ζ 0.006–0.03 a beam against the census 0.04. Every defined spring now takes at least [`ZETA`] on its final spring (`build::defined_damping`; stops and rubber mounts keep theirs): 1,027 of 4,251 beams, no spring changed, the seat's modal ζ 0.012 and 0.037 mm. A heavy part rocking on short mounts is still the car's least damped mode.
- **Process:** this shell (zsh) does not split `$cfg` into words: pass each value as its own variable.

## 16. Hinges that snap instead of bending (2026-11-14, `integration/lab-skin`)

Measurement: [measurements/hinge-breaks.md](measurements/hinge-breaks.md).

- **(Measurement) A break group is as weak as its weakest beam.** One beam past its strength sets off the whole group, so a hinge of nine unyielding beams (deform FLT_MAX) snapped at the first beam's 17-29 mm of stretch while the door or lid was barely bent (2-6 % plastic strain). Hardware that should bend needs a finite deform below its strength (now 0.4 × strength, the census's door band) and a bounded bend.
- **(Measurement) With a fixed spring per beam, the long beams of a mount reach their strength first,** and a beam to a crush member changes length when that zone folds whatever the panel does. Mount reach and crush members are rules (`HINGE_REACH`, `crush_member`).
- **(Fact) The solver keeps a JBeam rest length at 0.1 m or more (`MIN_REST`):** a shorter beam cannot yield in compression and breaks on its strength. Hardware beams are 0.12 m or longer.
- **(Measurement) A per-step observer finds the trigger:** `examples/hinge_breaks` (CPU reference, `CpuLab::step_observed`) names the beam past its strength, its force and stretch, and the panel's state at that step. One mountain run is one sample: compare seeds.

## 17. Dark patches and dots in a crushed car's paint (2026-11-16)

- **(Measurement) They are back faces, not z-fighting, tears or shadows.** On a heavily wrecked sedan (230 of 4,251 beams broken) the paint showed black blocks with jagged triangle edges and regular dots. The same state drawn several ways (`vehicle-lab-speckle.mjs`): the tear field is 0 there, the car's shader samples no shadow map, and every dark pixel is a back-facing paint triangle. A panel is a closed solid whose inner face sits millimetres behind its skin and follows other nodes; a crush pushes it out through the skin, and the shader drew a paint back face as a dark "inside".
- **(Decision) A closed solid has no inside to see while it is whole.** Every ray meets a front face first, so a paint back face in view is the solid turned through itself: not drawn, and the skin behind it shows. Round a tear (tear field over 0.05) the cavity is open and stays dark. Sheet-built cars (the authored sedan, imported mods) keep their dark insides.
- **(Fact) The renderer learns "closed solids" from the model, not the page:** the generator writes `_SURFACE` only on solid parts; `beamng-mod.mjs` `closedSolids()` reads it and sets `visual.closedPanels`.
- **(Process) A picture someone saw depends on the files they ran.** The page's default run rests on its roof, lightly damaged, and shows nothing; the wreck came from a local `__mods/sedan_new` whose body JBeam was two days older than the generator's. Check the mod folder a checkout links to before trying to reproduce.

## 18. Adding a vehicle (2026-11-24)

A design is one Rust file in `engine/crates/engine-vehiclegen/src/designs/` plus one line in `designs::REGISTRY` (`designs/mod.rs`: name, title, defaults, build). Nothing else names a design (CLI, wasm designer, web build, tests, lab); the gates read the registry, so a registered design is checked without editing any test.

- **Variant or standalone.** A *variant* wraps a design and overrides what differs: `police.rs` (194 lines: livery and a function adding parts), `wagon.rs` (226: key-line tables, tail face, tail layout, roof rails) and `mach1_gasser.rs` (162: livery, running gear, pose, parts) delegate every `Design` method to the base. The sedan exposes `Variant` hooks (keys, rear face, panel rule, tail, tailgate, rear lines: `Sedan::with_variant`); Mach 1 exposes `Running` (`Mach1::with_running`). A *standalone* design (`mach1.rs`, `suv4x4.rs`) brings its own params struct (serde, `#[serde(default)]`, `validate`), tube, layout, caps and parts. Port the Python (`tools/vehicle-gen/designs/`) piece by piece; its pinned numbers (`tests/mach1.rs`) are for the legacy sheet build (`BuildOpts { solid: false }`).
- **What the rigger needs from a design** (all in `RigSpec` and the two definitions; never a design's name in the rigger):
  - `structure::Lines` (`structure()`): reference-frame door, screen, B-pillar, seat and rear lines, the wheel-well liner (`well_x`, `well_r`), `wheel_y` (the opening's centre), `y_pan` (the floor pan, a little over the skin's floor line: the sills and the firewall's foot stand on it). Reference z is the sedan frame's: a design with its own proportions maps its z through its `Frame` (`f.zi`), as `suv4x4.rs` does; the `Frame` also lifts the floor (`dy`) and scales the roof (`kr`).
  - `mechanical::Gear` (`mechanical()`): track, wheel centre and axles, tyre and rim sizes, the nose and tail faces; `Gear::rear` when the rear axle's wheel differs.
  - `RigSpec::doors` (two- or four-door), `rear_closure` (`"trunk_lid"` or `"tailgate"`; a tailgate's `hinge_tailgate_*` / `latch_tailgate` markers are read as the lid's), `arches`, `front_z` / `rear_z`, `physics`.
  - Rear lines for a tailgate car: the back light's top and base on the centre line (the rear header and the parcel shelf stand there); they must be ahead of the tailgate's skin (`tests/every_design_structure.rs` fails a node outside it).
  - `Design::pose()` and `RigSpec::pose` for a posed car (a pitch about the rear axle and a lift). Build every part in the body frame, wheels included: the build places the whole car by the pose; the rigger rigs the body frame. `RigSpec::rear_wheel` (with `Gear::rear`) for different front and rear tyres.
- **The gate** (run from `engine/`, `--release`, no GPU): `cargo test -p engine-softbody-authoring --test every_design -- --nocapture` (every registered design rigs, parks 3 s on raycast wheels: drift within 5 mm of design, 0 broken, 0 bent, the lowest point on the ground within 30 mm, then drives 3 s); `cargo test -p engine-vehiclegen` (`tests/registry.rs`: every part builds and every solid is closed; `tests/every_design_structure.rs`: the body-in-white lies inside its skin, members 8 mm under it; `tests/build.rs` and `tests/mach1.rs`: the pinned sedan and Mach 1 builds). A stale test binary can omit a new design: `touch crates/engine-vehiclegen/src/lib.rs` and rerun, and check that every registered design prints a line.
- **Gate results (2026-11-24, same build):**

| Design | Nodes | Beams | Mass | Worst drift | Rust |
|---|---|---|---|---|---|
| sedan | 548 | 4251 | 1410 kg | 2.0 mm | `sedan.rs` 1,212 lines |
| police_sedan | 548 | 4251 | 1410 kg | 2.0 mm | `police.rs` 194 |
| wagon | 531 | 4064 | 1410 kg | 1.8 mm | `wagon.rs` 226 |
| mach1_1969 | 478 | 3548 | 1363 kg | 2.6 mm | `mach1.rs` 1,198 |
| mach1_gasser | 478 | 3550 | 1363 kg | 2.7 mm | `mach1_gasser.rs` 162 |
| suv_4x4 | 539 | 4090 | 1410 kg | 2.7 mm | `suv4x4.rs` 816 |

- **(Fact) Two things a tall or posed car taught the rigger,** each fixed at its source and neutral for the earlier designs (their results are bit for bit the same): a tailgate over 0.9 m tall had nodes only along its top edge and its foot, one plane, so no flexbody skin fit was well posed ("no well-posed fit": it now has rows up its face); and the shell's brace lengths (1.25 m, and 0.9 m before a beam counts as long) were the study's for a cabin up to 1.07 m from the tunnel to the roof bow, so a 4x4's roof bow centre had no beam with a vertical part and sagged 42 mm (they now scale with a taller cabin).
- **(Process)** A gate that only compares a car with its own design cannot see a car that fell through the ground, since a rigid fit removes the fall: `every_design` also asserts the parked car's lowest point is at the ground.

## 19. The collision skin is checked at build time (2026-11-24, S4)

- **(Process) Defects are detected, not discovered.** The S4 crash study found the
  generated sedan's collision-skin defects by hand: holes over the lamps, doubled layers
  at seams, a skin through itself at rest. The vehicle lint (`authoring::lint`) existed
  but ran only by hand (`vehicle_lint` example), so even its hard `skin_coverage` rule
  failed unseen (p99 0.149 m against 0.127 m).
- **(Decision) Every generated design is linted in CI.**
  `tests/generated_vehicle_lint.rs` rigs each design in `engine_vehiclegen::designs::REGISTRY`
  and gives every hard rule a test of its own; `every_hard_rule_has_a_test` fails when
  the lint gains a hard rule with no test. Failing rules sit on
  `tests-expected-failures.txt` with their measured values (CI fails on a new failure
  and on a listed one that starts passing). A design added to the generator goes in
  `DESIGNS` and is linted from then on.
- **(Rules) The collision skin's own** (`lint::collision_skin`, proven on synthetic
  boxes in `tests/host/vehicle_lint.rs`: a sound skin passes, each broken one fails its
  rule): `skin_winding` (outer edges walked oppositely by their two triangles),
  `skin_facing` (outer triangles face out; one under a second layer within 5 cm is
  tagged), `skin_self_crossing` (no edge through the skin at rest), `skin_holes` (no
  loop of open edges under 2.5 m round), and warnings for inner faces' winding,
  edges on three or more triangles and slivers.
- **(Process) The rig tool prints the verdict.** `vehicle_gen_rig` ends with the lint's
  rules that need no render skin on what it wrote, and a warning with the count of hard
  failures.
- **(Measurement) The generated sedan today:** coverage p99 0.149 m; 15 outer edges
  wound the same way and 27 outer faces under a doubled layer (hood–fender and
  body–rear-bumper seam bridges over the parts' own triangles); 89 crossings at rest
  (sill and floor faces through the dash, rear bumper through the body); 12 holes under
  2.5 m round (door edges, the corners over the lamps).

- **(Fix, 2026-11-24) The lint's first findings, fixed in the rigger** (`gen2/rig.rs`):
  seam bridges and zipper triangles may not fold over the triangle they share an edge
  with or cross the skin (`RIG_BRIDGE_OVERLAP=1`: as before); a body-shell triangle
  crossing another part's goes (`RIG_HULL_KEEP_CROSSING=1`); `fill_holes` closes holes up
  to 2.5 m round, split where a loop meets itself, each patch running against its rim
  (on by default; `RIG_FILL_HOLES_M=0`); interior solids are placed before the shell's
  faces and a shell face takes a diagonal that does not cross. Outer winding 15 → 0,
  facing 27 → 2, holes 12 → 1, closed interior solids 5 → 8; crossings at rest 89 → 60
  (dropping the crossing sill faces took them to 6 but let the T-bone's door pierce the
  sill, `sedan_parts_collide`: they stay). Every other authoring and generator test as
  before. Six runs of the 60 + 60 head-on: node-only contact 71 → 66 nodes inside,
  191 → 145 L; with edge contact 26 → 33, 49 → 55 L (within the spread).

- **(Fix, 2026-11-27) Two coplanar overlaps in the shared rigger** (`gen2/rig/zones.rs`;
  the lab refuses a car whose own skin primitives sharing no node are closer than 2 mm at
  rest, the lint's `skin_clearance`). Both laid a second surface in the plane of the first,
  which `tri_crosses` cannot see (coplanar edges meet, they do not pass through):
  - *A hull face with a point inside it.* `hull_faces` orders a face's points by angle
    round their centre; the rear seat's `sear_c` *is* the centre of its four corners, so its
    angle was noise. On the sedan it happened to sort first (a fan round the centre); on the
    Mach 1s and the 4x4 it sorted between two corners, the polygon folded over itself and
    its fan laid `sear_rl,sear_fr,sear_rr` across the centre's triangles (0.00 / 0.07 mm).
    **Rule:** a point strictly inside its face starts the face's polygon, so the fan is the
    centre's to each rim edge (fill_holes closes the last one, as on the sedan).
  - *A quad given both diagonals.* On the Mach 1s the outer skin already had the first
    floor-pan quad (`sh_c1_1o*`, `sh_c1_2t*`, one diagonal); `interior_skins` found that fan
    "crossing" something else and took the other diagonal over the same quad. **Rule:** a
    face one of whose fans is already on the skin whole is not laid again.
  Sedan, police_sedan and wagon rig byte for byte as before (neither case occurs there).
  Mach 1 / gasser `skin_clearance` 4 → 0, the 4x4's `sear_*` pairs gone (15 → 13; the rest
  are the hood/fender and `sh_e5` seams). With the doubled quad gone, the Mach 1s' floor pan
  also comes out whole in rows 3–4 (open edges 172 → 160, 169 → 157), which leaves
  the tunnel's open rear end (`sh_c1_5t*`–`sh_c3_5`, two open edges, open on every design
  including the sedan, where it runs into a larger opening) standing alone: the Mach 1's
  `skin_holes` now reports it, as the gasser and 4x4 already did for the tunnel's front end.

- **(Fix, 2026-11-27) The 4x4's own skin clearance** (`gen2/rig.rs` `fender_top_x`,
  `gen2/rig_fill.rs` `cover_bare`; the PR preview refused to spawn the 4x4 with continuous
  contact on):
  - *A fender's top row under the hood's edge.* The fender's top row stands 2.5 cm out from
    the fender's inner edge, and the hood's edge row 2.5 cm in from the hood's edge, both on
    the top surface. The 4x4's flared fenders narrow at their ends (x 0.65 against the hood's
    0.78), so the two rows crossed at the same height, 0.47–1.6 mm apart (10 pairs).
    **Rule:** a fender's top row is never inside the hood's own edge (|x| ≥ the hood's
    half-width there).
  - *A bare node's cover across a skinned quad.* `cover_bare` gives a skin node in no
    triangle a fan to its two nearest skin neighbours and nearest shell node. At the 4x4's
    rear the fan's shell edge was the other diagonal of a quad already skinned, 0.3 and
    0.6 mm from that quad's diagonal (and one edge 0.8 mm from the rear bumper's).
    **Rule:** the nearest choice whose faces keep `SKIN_CLEARANCE` from every placed face
    (`near_skin`), nearest neighbours first.
  - Sedan, police sedan, wagon and Mach 1 rigs are byte for byte as before. Every design
    now passes `skin_clearance`. The lint test checks every design, not only the sedan, so a
    car the lab would refuse fails at build time.

## 20. Glass binding: a pane interpolates its frame (2026-11-27)

- **(Measurement) The 4x4's windscreen spike.** After a roof crush (vehicle lab, shoot, a
  1.5 m steel sphere at 6 m/s from above, or a 0.1 m one at 49 m/s), a grey cone of the
  windscreen rose 0.7–0.9 m above the car. On the CPU reference
  (`engine-world-lab --example skin_spike_probe`) it is one region of the pane, its top
  middle (vertex 3064 of `glass_windshield`, 18 cm below the header). Its weighted skin
  (`.rskin`) had Σ|A| 4.46: −1.45 on the header's middle node `sh_d4_h`, +0.95 on each
  header corner, +0.40 on the roof's next row, −0.28 on the roof's rear, the cowl 0.01.
  The roof crush pushed `sh_d4_h` down 0.5 m more than the corners, so the vertex went up:
  0.86 m past its rest distance from its nodes (the rest of the car: 0.37 m at most).
- **(Cause) Extrapolation, not a bad node.** No node broke off and no weight went to
  another part: the moving-least-squares fit (`flexskin::bind`) reached only the roof's
  sparse nodes (long kernels) from the pane's top; the cowl's dense nodes (short kernels)
  were out of reach, so the affine fit extrapolated below the roof plane with large
  weights of both signs. A glass pane spans an opening with no nodes in it, so it is
  where this happens. Panels have nodes on them.
- **(Fix) `gen3::GLASS_AMP_MAX` = 3.0** (`flexskin::bind_with`, `amplification`): a glass
  mesh with a vertex past Σ|A| 3 grows its kernels, like an ill-posed fit, until the frame
  on every side is in reach. The 4x4's windscreen now binds at Σ|A| 1.32 (its rear glass
  3.02 → 1.45, its quarter lights 3.43 → 2.64); after the same crush its worst vertex is
  0.20 m past rest (from 0.85 m), the small shot's 0.14 m (from 0.67 m). Glass is visual
  only: no physics changed.
- **(Decision) Other designs unchanged.** Their glass reaches at most 2.98 (the wagon's
  quarter lights) and none spikes under the same crush (glass 0.10–0.13 m past rest, below
  their body shell's). Sedan, police sedan, wagon and both Mach 1 rigs are byte for byte as
  before.
- **Checks:** `every_design` fails a design with a glass vertex past the bound;
  `flexskin::tests::a_pane_across_an_opening_interpolates_its_frame` holds the case (a
  sparse roof, a dense cowl: unbounded the pane rises 0.89 m as its header falls 0.3 m,
  bounded 3 cm).
- **Open:** the same measure is high on some non-glass pieces (the 4x4's fenders 8.1, the
  rigid headlamps 4.6–9.1, rigid pieces' centroid weights); none was seen to spike, but a
  rigid piece's frame extrapolated that far turns with the nodes' bending.

## 21. The 4x4's kerb mass, and crash zones from the definitions (2026-11-27)

- **(Fact) Source of the 4x4's kerb.** `suv_4x4` had the sedan's `Physics::default()`
  (kerb 1450 kg). Its wheelbase is 2.70 m (wheel axle node to axle node). The closest
  published match is the 2012 Nissan Xterra Pro-4X 4WD, a mid-size body-on-frame 4x4 on the
  same 2.70 m (106.3 in) wheelbase. Its curb weight is 4,417-4,425 lb (AP fact sheet;
  cars.com and Guide Auto list 4,417-4,418 lb for the manual 4WD), which is 2,004-2,007 kg.
  The 4x2 is 4,143 lb (1,879 kg). The design now says `kerb` 2000 kg
  (`designs/suv4x4.rs` `SUV_KERB_KG`).
- **(Measurement) The rigged mass did not change: 1410 kg, every JBeam byte as before.**
  Cars built on the body definitions (all six designs) take their masses from those
  definitions: the shell groups (`structure::GROUPS`), each `MechPart`'s nodes and the
  closures' `Physics` masses (`body2::masses`). Only the earlier census rigger
  (`rig.rs`, "the rigger's own masses") reads `kerb`, putting the remainder in a
  front-end module. So the 4x4 is the sedan's masses on a taller body: the sedan's shell
  groups, the sedan's running gear and powertrain (`mechanical::sedan_parts`), the
  Accord's closures and engine. The lint's `mass` rule now fails it at −26.5 %
  (1410 kg against 1920 kg ± 10 %), inside the already-listed
  `every_other_design_passes_every_rule`.
- **(Fix, 2026-11-27) The 4x4's mass, built up part by part** (`designs/suv4x4/mass.rs`;
  each figure says whether it is published or an engineering estimate, with its
  uncertainty; none was fitted to the kerb, nothing is scaled by a factor). What a
  body-on-frame mid-size 4x4 has that the unibody sedan lacks or has heavier, each on the
  part that is it or the nodes where it is: the ladder frame's middle section (+100 kg: a
  140 kg frame less the horns the unibody's own rails carry, half on each subframe's inner
  nodes), a 4.0 L V6 (+40), a heavier automatic (+30) and a transfer case (+35) with its
  skid plate (+6), a front differential and propshaft on the front crossmember (+40) and
  its skid plate (+10), front CV shafts and brakes (+11 a hub), a solid rear axle's carrier
  (+25) and tubes, shafts and half-leaves (+18 a hub), 265/75 R16 steel wheels (+10 each),
  an 80 L tank full (+30) and its skid (+4), the V6's exhaust (+5): +423 kg of mechanical
  parts. Rock sliders (+20) and the roof rack when fitted (+12) on the shell. Taller doors,
  a steel hood, flared fenders, steel bumpers and a full-height tailgate (`Physics` closures,
  +39.3), and the spare wheel on the tailgate when fitted (+32): +526 kg in all.
  **Measured:** rigged 1,447.7 → 1,974.0 kg (nodes 1,410 → 1,897 kg: the lint's `mass`
  passes at −1.2 % of 1,920 kg); topology unchanged (539 nodes, 3,509 beams), stability
  ω·h 1.848 → 1.841. Parked 4 s (`tests/suv4x4_rest.rs`): 0 broken, 0 bent, ride height
  still within 0.09 mm after 1.5 s, kinetic energy decaying (0.015 J → 0.003 J). The gate's
  worst drift rose 2.7 → 4.9 mm (the bar is 5 mm): the 15.5 kg front bumper's lower row
  (`bf_g2`) sags on its bolts; the same row hangs forward after S02 (below). Open. GLB, `.rskin`, `crash.json` and every other design's
  output byte for byte as before (the look is unchanged: no visible part was added).
- **(Measurement) The axle split is the sedan lattice's, not an SUV's.** Before 49.6 % front
  / 50.4 % rear, after 48.7 / 51.3 (the spare, the tailgate and the tank sit behind the rear
  axle). A front-engined body-on-frame 4x4 is nearer 52-55 % front (engineering estimate);
  the sedan the 4x4's lattice comes from is itself 49.6 % front, where a front-drive mid-size
  sedan is about 60 % (estimate). Not retuned: the masses are where the parts are; the
  inherited layout (engine 0.09 m ahead of the front axle, a rear differential the
  front-drive Accord does not have) is the open question.
- **(Fact) The ladder frame is mass only.** The shared part builders have no frame (its
  rails, crossmembers and rubber body mounts), so its stiffness and strength are not
  modelled; the unibody lattice carries the load.
- **(Fact) Strengths were checked, none changed.** The generator derives a member's
  strength from its section, gauge and grade (`Member::crush_n`, the crash boxes' and
  rails' deform), never from mass; the mechanical parts' mounts, the hardware and the
  closures' hinges and latches are literals (the study's, FMVSS for the seats). Heavier
  nodes only raise the springs the node budget gives (Σk/m). Open: whether the engine
  mounts (90 kN, set for 170 kg) and the tailgate's hinges and latch (with 54 kg on them
  now) hold a 2,000 kg car's loads.
- **(Measurement) A re-massing rule that was not kept.** A shell member's sheet goes with
  its sheet area at the same gauge, so its nodes could be re-massed by this car's area over
  the sedan's. On the 4x4 that takes 12 kg off: the shared definition gives its rear pillar
  (`c_pillar`, the rear header to 0.16 m above the back light's base) 0.014 m² against the
  sedan's 0.154 m², a 0.14 m pillar where the 4x4's D-pillar stands 1.1 m tall, and its
  upright A-pillar is shorter than the sedan's raked one. The definition does not describe
  a two-box body's rear pillar; the shell keeps the sedan's group masses.
- **(Measurement) S02 with the 4x4 at 2,000 kg** (CPU, 8 runs each of discrete, edge,
  continuous): crush 0.25-0.27 m, rebound 0.25-0.31 (the 1,448 kg 4x4: 0.33), no energy
  created; "the crush stays" reads −0.10 to −0.19 m because the car ends longer than it
  started: the front bumper's lowest row (`bf_g0`-`bf_g4`) ends 0.08-0.18 m ahead of where
  it began relative to the centre of mass. At the old mass the same measure was 0.000.
- **(Fact) `crash.json`** (`gen2/crash_zones.rs`, written beside the JBeam by
  `vg_input::files_with` and `web_files`). This is the crash reference suite's zone map
  and gauges. Each node's zone comes from what it belongs to, by one rule. A shell node
  takes its member type's zone (`engine_vehiclegen::crash::member_zone`, checked for
  every design's members by `engine-vehiclegen/tests/crash_zones.rs`). A mechanical part
  takes its own `MechPart::zone`, which every part states when it is made. A panel takes
  `crash_zones::part_zone` (an exhaustive match). Any other body node (shut-line edge
  nodes, strikers) takes the zone of the nearest shell node. The names are written as the
  fewest prefixes that give every node its zone (100 % of nodes on all six designs).
  The gauges are picked from geometry. `tests/crash_zones.rs` (authoring) holds both.

## 22. A heavy vehicle: the 4x2 tractor unit `truck_cab` (2026-11-27)

For the crash reference suite (a truck T-boning a sedan at ~70 km/h, a sedan crushed
against a parked trailer). Design `designs/truck_cab.rs`; parts `engine-vehiclegen::heavy`
(ladder frame, beam axles and raycast wheels, cab, front underrun, box parts, mounts: the
trailer reuses them); rigger `gen2::parts_rig`; tests `engine-softbody-authoring/tests/truck_cab.rs`.

- **(Fact) The class and its published figures.** A European 4x2 tractor of the Renault
  Trucks T class, from its personalised specification sheet (T High T4X2 13L, model code
  NLT_10XA036STD00, 17 Sept 2025, renault-trucks.co.uk): wheelbase 3,800 mm, front overhang
  1,365, rear overhang 825, overall chassis length 5,990; frame width 851 mm (rear),
  side member height 266 mm and thickness 8 mm; frame top 938–958 mm unladen; fifth wheel
  1,152 mm over the ground, 675 mm ahead of the drive axle; track 2,109 / 1,854 mm; width
  over the rear wheels 2,518 mm; cab width 2,534 mm; centre of gravity 1,159 mm behind the
  front axle and 960 mm up; kerb 8,499 kg = 5,930 front + 2,569 rear (69.8 % front; full
  AdBlue, 90 % fuel, a 75 kg driver; high sleeper cab 3,845 mm). Front underrun
  protection: UN R93, its lower edge at most 400 mm over the ground (its test loads,
  up to 160 kN at a rail's line). Not on the sheet, typical values: flange 90 mm, tyres
  315/70 R22.5 (1,013 mm) on 9.00 × 22.5 steel wheels, frame steel S500MC-class
  (500 / 600 MPa).
- **(Decision) Ours:** a day cab with a standard roof (3.0 m), kerb 7,500 kg at the
  sheet's split (5,233 / 2,267 kg). Mass budget (kg): cab with interior, glass and
  driver 1,400; engine, gearbox and cooling 2,000; front axle with springs and steering
  600; drive axle with air springs 1,000; wheels 2 × 105 front, 2 × 200 (twins) rear;
  frame 840 (rails' steel 308, cross-members 125, frame-mounted equipment 407); bumper
  with underrun beam 120; fuel tank 350; batteries/AdBlue/silencer box 400; fifth wheel
  180. `ballast_kg` adds a stated gross on the fifth wheel.
- **(Fact) The car rigger cannot rig a ladder-frame vehicle.** `rig::rig_with` (~3,000 lines)
  lays a unibody lattice on a tube body with doors, hood, boot, fenders and the study's
  131-node shell. A design made only of defined parts (`MechPart`s, an empty
  `Structure`) goes to `gen2::parts_rig` instead (`parts_rig::applies`, a hook in
  `vg_input::rig_only`): the definitions' nodes, masses, beams, mounts and links; springs
  from the node budget under the step's bound; one closed convex collision hull per
  colliding part (so a colliding part is convex in its nodes: the frame is two rails, not
  one part); raycast wheels on `wheel_<corner>` parts; reference nodes on the frame.
- **(Measurement) The census node budget is too soft for a heavy frame.** At Σk/m 8.5e6
  (the cars') the truck's whole-car ω·h was 1.50 and the frame parked 6 mm out of shape
  (18 mm at 10 bays). The step's bound, not the census, is the limit: the parts rig asks
  for 2.0e7 and the power iteration takes back 15 % (ω·h 1.84). Fewer, heavier stations
  also help: a chord's EA at the budget grows as the station spacing squared (node mass and
  length both grow), so the rails have 7 stations (0.95 m) and one diagonal a face, not two.
  Worst parked drift 18 mm → 4.5 mm.
- **(Measurement) A part's mount must hold it in all three directions, not only in a
  plane.** Tanks mounted at one station hinged (15 mm); on two stations with every beam
  in the stations' planes they slid along the rail at 2 Hz (kinetic energy 0.25 J
  decaying 10 %/s); fore-and-aft bracket legs between the stations fixed both (0.006 J).
  Mount beams are at least 0.12 m (`MOUNT_MIN_LEN`; the solver's `MIN_REST`).
- **(Measurement, the tests)** 160 nodes, 918 JBeam beams (1,330 in the solver), 216
  collision triangles. Parked: worst drift 4.6 mm, 0 bent, 0 broken, lowest point −0.3 mm,
  ride height steady to 0.01 mm from 2 to 4 s, kinetic energy peaks per second 0.20 /
  0.034 / 0.019 / 0.012 / 0.009 J (never rises). Axle loads parked 5,203 / 2,293 kg
  (69.4 % front; target 5,233 / 2,267). 30 km/h full-width rigid wall (260 kJ): rails
  shortened 13.1 mm dynamic, 1.4 mm residual, no frame beam broken; bumper 193 mm dynamic;
  the cab's front reaches the wall (the bumper stands 65 mm proud of it): both front cab
  mounts (6 beams each) and the right belt rail between the cab's front and door pillars
  break. Lint: every hard rule passes (coverage p99 0.072 m, ω·h 1.858).
- **(Fix, an explicit setting) The raycast suspension was sized for the mean wheel load**
  (`pack::core_plan`: one k and c for every wheel). The truck's front axle carries 1.39 ×
  the mean, so it parked 38 mm nose-down past its design (`every_design`'s lowest point
  −38 mm, its tyres drawn into the ground). `Physics::suspension_by_load` (off by default;
  the truck turns it on) is written as the wheels' `rubbleSuspensionByLoad` and makes
  `core_plan` size each wheel's spring and damper for its own axle's static load (the lever
  rule between two axles; the same frequency and damping ratio, so the same sag): lowest
  point −0.3 mm. With it off the six cars' generated files are byte for byte those of the
  base (51f90c713) and their `every_design` lines identical.
- **(Measurement) A mirror-symmetric rig was not given mirror-symmetric springs.** Two
  causes: the underrun bumper's middle column stood 10 mm off the centre plane (`f64::signum(0.0)`
  is 1: fixed), and the step's bound (`Car::assign_springs`' power iteration softens the beams
  carrying the top mode) finds a pair of equal left and right modes as a combination of the
  two and softened one side more (184 mirrored beams differed; at the census budget, with no
  softening, 0). The 30 km/h wall then broke the left front cab mount alone. The parts rig now
  gives each mirrored pair the lower of its two springs (`parts_rig::mirror_springs`: only
  lower, so the bound holds): 0 differ, and both front mounts break. What still differs left
  to right in the wall (one belt rail, one of the bumper's two middle beams) is at its
  threshold: once one of two equal beams breaks the load leaves its mirror, and which goes
  first is decided by float-level differences (the solver's summation order, the collision
  hulls' diagonals, which the ear clipping does not mirror; the battery box's 400 kg against
  the tank's 350 kg). With the masses made equal the same beams break.
- **(Measurement) The frame's bending strength against the section's.** The published
  side member, a C 266 × 90 × 8 mm in S500MC (σ_y 500 MPa): A 3,440 mm², I 3.44e7 mm⁴,
  elastic modulus 2.59e5 mm³ (M_y 129 kN·m), plastic modulus 3.11e5 mm³ (two flanges
  2 × 720 mm² × 129 mm plus the web 8 × 250² / 4), M_p = σ_y·Z_pl = 155 kN·m a rail. The
  rigged rail is four chords of σ_y·A/4 = 430 kN at its corners, 0.266 m apart: it yields in
  bending at 2 × 430 × 0.266 = 229 kN·m, 1.47 × M_p (1.77 × M_y), and breaks at 275 kN·m
  (σ_u 600 MPa). The four 90 mm section ties (the flanges' width, under the solver's 0.1 m
  `MIN_REST`) never yield, so the section cannot distort; its webs and diagonals yield at
  215 kN. A real rail of this section is slender in its flanges (outstand c/t 10.3 against
  the class-3 limit 14ε = 9.6 at 500 MPa) and buckles locally before M_y. Not changed: chords
  of M_p / (2 × 0.266) = 292 kN would match the bending but put its axial squash at 1,168 kN
  against the section's 1,720; a decision for the crash scenes.
- **(Open)** Twin tyres are one raycast wheel 665 mm wide; the raycast wheel is the
  loader's 20 kg (the rest of each wheel is on its part's nodes), so its spin inertia is
  a twentieth of a real truck wheel's. A mesh with two materials' primitives had half its
  vertices (the other primitive's, in no triangle of this one) counted as paint by the
  coverage rule; the truck's meshes have one material each.
- **(Crash zones)** The underrun bumper is `FrontLower`, the cab `Cabin`, axles and wheels
  `Wheel`; the ladder frame, engine, tanks, battery box and fifth wheel `Other` (a ladder
  frame has no counterpart among a car body's sections). `crash.json` gauges:
  wheelbase each side, cab width and height, frame length, front overhang
  (`parts_rig::gauges`).

## 23. A parked semi-trailer `box_trailer`, its rear impact guard, and a laden tractor (2026-11-27)

For the crash reference's S04 (a sedan under the trailer's rear), S05 (the tractor T-bones a
sedan) and S06 (a sedan crushed between the tractor and the trailer). Design
`designs/box_trailer.rs`; new part builders `heavy/guard.rs` (`RearGuard`, `rear_guard`) and
`heavy/trailer.rs` (`CargoBox`, `RearFrame`, `LandingGear`); tests
`engine-softbody-authoring/tests/box_trailer.rs`.

- **(Fact) The class and its figures**, the US standard 53 ft dry van (the IIHS underride
  tests use US trailers; its tandem fits the rigger's four wheels):
  - length 53 ft (16.154 m), width 102 in (2.591 m), height 13 ft 6 in (4.115 m): the US
    federal width and the usual height;
  - kingpin 36 in (0.914 m) behind the front (a Vanguard VXP listing's "36 in pin setting");
  - a sliding tandem, 49 in (1.245 m) spread (listings: Utility 4000D-X, Vanguard VXP);
  - 40 ft (12.19 m) from the kingpin to the rear axle (the California kingpin-to-rear-axle
    limit), so the rear overhang is 3.05 m;
  - empty 14,000 lb (6,350 kg; a Con-way 53 ft dry-van brochure; 53 ft vans run
    13,500-15,500 lb, otrucking.com);
  - floor 50 in (1.27 m) over the ground: *estimate* (47-50 in is usual; a Con-way figure of
    48.25 in could not be attributed to its label);
  - tyres 295/75 R22.5 (1,014 mm) twin on 8.25 × 22.5 steel wheels, 1.82 m track: *estimate*;
  - landing gear 1.70 m behind the kingpin, its legs at ±0.60 m: *estimate*; a pair's static
    rating about 700 kN (*estimate*, the usual 155,000-160,000 lb).
- **(Fact) The rear impact guard's rules** (49 CFR 571.223 and 571.224, read 2026-11-27 at
  law.cornell.edu; 223 as amended in 2022 to CMVSS 223's levels). FMVSS 224: the guard's lower
  edge at most 560 mm over the ground across its width, measured unladen; its ends within
  100 mm of the sides and not past them; its rear face at most 305 mm ahead of the rear
  extremity. FMVSS 223: a horizontal member at least 100 mm high; a forward quasi-static push
  at 2-9 cm/min; at P1 (3/8 of the guard's width from the centre plane, either side) and at P2
  (the centre plane), both 50 mm above the guard's bottom, 50,000 N with no more than 125 mm
  of deflection; a uniform distributed load (a rigid block 203 mm high, wider than the outer
  supports) of at least 350,000 N, absorbing at least 20,000 J plastically within the first
  125 mm (or more than 700,000 N without the energy requirement); in every test no load path
  eliminated. Ours models FMVSS 223/224 as amended (CMVSS 223 has the same 350 kN and 20 kJ;
  UN R58, Europe's, is not modelled). The pre-2022 FMVSS 223 had other point loads (from
  memory, not checked against the old text: 50 kN at P1 and P3, 100 kN at P2 with 5,650 J):
  not used.
- **(Decision) The guard is the minimally compliant one; nothing in it is a free number**
  (`heavy/guard.rs` derives each value):
  - lower edge 0.52 m at design, ends 75 mm in from the sides, rear face 0.10 m ahead of the
    rear extremity (*estimates* inside the regulation's limits);
  - the bar: the lightest standard 100 mm square tube whose plastic moment carries P1's 50 kN
    outboard of a support, 100 × 100 × 6.3 mm in ASTM A500 Grade C (345 / 427 MPa): M_p
    28.7 kN·m (5 mm: 23.4). Its four chords yield at M_p / (2·d), so the bar yields in
    bending at the tube's plastic moment; each face's diagonal at the face's plastic shear;
  - two supports at ±0.70 m (outboard of the rails, *estimate*), each a post (the bar's top to
    the rails' last station) and a brace (to the station ahead), four beams each: their yields
    from one statically admissible distribution of the 350 kN (the bar a rigid body on beams of
    equal EA), each member's beams at its largest share, so the guard collapses at no less than
    350 kN (the lower-bound theorem); breaking at σ_u/σ_y × yield, the connections
    capacity-designed (*estimate* of the margin: the regulation only says nothing breaks in its
    tests);
  - 100 kg (*estimate*: the bar 44 kg plus supports and brackets).
- **(Measurement) The guard against FMVSS 223** (`the_guard_carries_fmvss_223s_loads_before_its_attachments_fail`:
  the trailer held, a kinematic platen at 0.08 m/s, the force the supports' beams hold):
  | point | at 125 mm | largest before an attachment broke | energy in 125 mm | first break |
  |---|---|---|---|---|
  | P1 (50 kN) | 44 kN | 234 kN (none broke in 0.4 m) | 2.8 kJ | none |
  | P2 (50 kN) | 22.5 kN | 98 kN (none broke in 0.4 m) | 1.4 kJ | none |
  | uniform (350 kN, 20 kJ) | 464 kN | 604 kN | 48.9 kJ | the left brace at 206 mm |

  Every point carries the regulation's load before an attachment fails (the asserted test).
  P1 and P2 do not reach 50 kN within 125 mm: the bar's elastic stiffness is the step bound's
  (§2: a 100 kg bar's light nodes cap its springs; its P2 stiffness is about 180 kN/m where a
  real 100 × 100 × 6.3 tube simply supported over 1.4 m is 48·EI/L³ ≈ 12 MN/m, about 65 times
  stiffer). Its strength
  is real, its stiffness is not; a finding, not a band to widen.
- **(Measurement) What a moving static box does to a point load.** Pushing a `StaticBox` 0.2 m
  wide through the bar let the two P1 nodes slip out over its top after 26 mm (the force read
  4 kN where the bar carries 50). A rigid platen is kinematic: the nodes it reaches are carried
  with it.
- **(Measurement) Ringing and drift, fixed in the layout** (each a rule now in the builders):
  supports under the rails left the bar's ends cantilevered 0.8 m, ringing in plan at 6.7 Hz
  with about 0.5 % modal damping for tens of seconds after the spawn's 2 cm drop; at ±0.70 m
  the ends ring at about 3× the frequency and die in about 5 s (the bar's first modes take their
  damping only from the beams' relative damping). An axle hung from one rail station hinged
  (19 mm): the hangers reach two. The deck's 10.5 t on floor nodes at the walls only, tied to
  the rails by near-level beams, left the rails over the tandem 18 mm off the box's line: floor
  nodes over each rail seat it on the rail's station below (the box's stations now on the rails'
  0.80 m). The landing gear on the rails alone drifted 18 mm: its brackets also take the floor's
  nodes over it, and fore-and-aft braces stop the legs swinging as cantilevers. The 250 kg
  kingpin plate on near-level bolts rocked: it bolts to the floor over the rails too.
- **(Fix, the shared rigger) A hull face's ear clipping left a node on a triangle's edge.**
  `parts_rig::hull` clips the best-shaped ear first; on the box's long sides (17 then 21
  stations of collinear edge nodes) it cut an ear whose new edge ran over another ring point,
  so one node sat on a triangle's edge (lint `skin_clearance`, 0.00 mm) and the last three
  points were collinear (a triangle short, an open hole). An ear whose new edge passes over a
  ring point is not taken (`on_segment`); test `a_long_box_with_collinear_edge_nodes_has_a_closed_hull`.
  The truck's (and every other design's) generated files are byte for byte as before.
- **(Measurement, the tests)** 374 nodes, 2,339 JBeam beams (3,547 in the
  solver), 560 collision triangles, ω·h 1.849; 15,000 kg (nodes 14,920 + four 20 kg raycast
  wheels). Parked 10 s: 0 bent, 0 broken, lowest point −0.1 mm, box height steady to 0.03 mm
  from 2 to 10 s, kinetic energy per second 27.5 → 0.08 J (never rising), every node but the
  guard's under 0.025 m/s at the end (the guard 0.06 m/s); `every_design` worst drift 4.5 mm (the
  guard bar). Support split parked: the tandem 9,228 kg, the landing gear 5,773 kg, against the
  lever rule on the design's own centre of mass (3.92 m ahead of the tandem's middle) 9,163 /
  5,837 kg. Lint: every hard rule passes.
- **(Finding) The raycast springs are sized for the mean wheel load, and a trailer's wheels do
  not carry all of it.** `pack::core_plan` sizes every wheel for total/4 (and the per-axle
  lever rule, `suspension_by_load`, needs exactly two wheel axles carrying everything: on a
  tandem with the centre of mass 3.9 m ahead it would give a negative load). The tandem
  carries 61 % of the trailer, so its springs keep 39 % of their 97 mm sag unused and the rear
  stands about 40 mm high: the guard's lower edge parked is 0.570 m (0.525 m in the template,
  FMVSS 224 at most 0.560). Proposed fix (engine-softbody `pack.rs`, not in this change): a
  vehicle standing on non-wheel supports sizes its wheels for their share of the static load
  (the lever rule between the wheel group and the other supports' line).
- **(Measurement, not kept) A laden tractor.** The crash reference's templates are a design's
  defaults, so a ballasted tractor needs a registry entry of its own. `truck_cab_laden`
  (4,500 kg on the fifth wheel, 12 t, M/m 8.3) rigged and linted clean but failed
  `every_design`: parked, its frame stood 13.5 mm out of shape at the rear rail (5 mm the bar;
  cab 9 mm, tanks 9-10 mm), the 2 kHz node budget's soft frame (§22) under 4.7 t on two
  stations. Not registered; S05 and S06 use `truck_cab` at its kerb (7.5 t, M/m 5.2). The
  trailer's default is laden (`ballast_kg` 8,650: 15 t gross), since the trailer exists for
  the crash scenes, and it passes the gate laden.
- **(Crash zones)** The guard is the trailer's `Rear` (the crash reference reads it as
  `trailer.zone.rear.*`: its bar's and supports' beams, bent, broken and plastic work); box,
  rear frame, landing gear, kingpin plate and frame `Other`; axles and wheels `Wheel`.
  `crash.json` gauges: the tandem's spread (as `wheelbase_left/right`), `kingpin_to_axle` (the
  kingpin plate's rear edge to the rear axle), `box_width`, `box_height`, `frame_length` and
  `guard_drop` (the rear sill's lowest node to the guard's: its change in a crash is the guard
  pushed against the trailer).
- **(Open)** Brakes: the crash scenes set no inputs, so a parked trailer's tandem rolls free
  (a parked trailer's spring brakes are on) and only the landing gear's feet grip. The lamps'
  lens tints are not drawn in the studio renders. The cargo box's walls are mass and stiffness
  only (no panel to dent: `members` 80 / 160 kN a beam, *estimate*).

## Changelog

- 2026-11-27: section 23, the parked 53 ft dry van `box_trailer` (sources, the FMVSS 223/224 guard and its strength test), the laden tractor, the hull's ear fix.
- 2026-11-27: section 21, the 4x4's mass built up part by part (1,410 → 1,897 kg of nodes; the lint's `mass` passes), the axle split, strengths checked, S02 at the new mass.
- 2026-11-27: section 22, the heavy tractor `truck_cab`, the parts-only rigger, published figures.
- 2026-11-27: section 21, the 4x4's kerb (2000 kg, the Xterra's) and why the rigged mass stays 1410 kg; `crash.json`.

- 2026-11-27: section 20, glass binding: a pane interpolates its frame (the 4x4's windscreen spike).
- 2026-11-27: section 19, two coplanar overlaps fixed in the rigger (a hull face's inside point, a quad's second diagonal).
- 2026-11-24: section 18, adding a vehicle (the registry, variant or standalone, what the rigger needs, the gate); mach1_gasser (pose, per-axle wheels) and suv_4x4 registered.
- 2026-11-16: section 17, dark patches in a crushed car's paint (closed solids' back faces).
- 2026-11-14: section 16, hinges that snap instead of bending.
- 2026-11-13: section 14: the GPU kernels carry the centred damping's objective form; on by default for CPU and GPU.
- 2026-11-13: section 15, jello against the ETK (modes in band, kerb not), collisions inside one car.

- 2026-11-13: section 14, energy bombs: the gapped stops, the centred damping on spinning parts, the lint's rules and the tests.

- 2026-11-13: section 13, the new body crashed against the ETK; the crash character setting.

- 2026-11-12: first version.
- 2026-11-12: linked the BeamNG structure study (`beamng-structure-study.md`): every node at Σk/m ≈ 7–10e6 s⁻² (whole-car ω·h 1.80–1.90), the 131-node shell specification and 24 lint rules.
- 2026-11-13: section 12, the body rebuilt from definitions (structure, mechanical parts, shell lattice); the rigger integration and all measurements still to do.
