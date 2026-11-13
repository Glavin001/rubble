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
- **The soft car's raycast wheels** are the rigid vehicle model applied as forces on hub nodes inside the 2 kHz step. Their rays see static geometry only (F7).
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
- **(Measurement) The solver's centred damping added energy to anything that spins** (a hub cap: 3.75 J to 1,800 J in 0.75 s). Its correction read a turning beam as shortening. The objective form fixes it on the CPU reference (off by default until the kernels follow).
- **(Measurement) With both, 82 runs never exceed their starting energy** (walls ±20/42/60° at 56/106/150 km/h, side poles 32-320 km/h at two places, drops 5/15/31 m at 0/100/189 km/h, surface contact on and off). What is left are single-step rises of the discrete energy, in the beams: 0.3-4 kJ where an engine mount's bound is crossed in one step (the lint's `bound_wall` gives 1.3 kJ a beam), and up to 28 kJ at 320 km/h on a 0.47 kg body edge node (`ebody13`): the light edge nodes on soft ties that the jello work removes.
- **(Decision, the user's) Catch the class in the lint,** statically (`bound_wall`, `tether`) and dynamically (tear every part off, fling and spin it; the ledger never rises).
- **(Measurement) A hub cap leaves at a 25 km/h bump onto its wheel** (9 clips of 600 N a beam, the study's value). Not changed: open whether real clip retention holds at that load.
- **Process:** write the test from the physical invariant first and commit it failing. Set energy bounds against the plain step on the same body: single precision alone moves an undamped body's energy by 0.3 % a second.

## Changelog

- 2026-11-13: section 14, energy bombs: the gapped stops, the centred damping on spinning parts, the lint's rules and the tests.

- 2026-11-13: section 13, the new body crashed against the ETK; the crash character setting.

- 2026-11-12: first version.
- 2026-11-12: linked the BeamNG structure study (`beamng-structure-study.md`): every node at Σk/m ≈ 7–10e6 s⁻² (whole-car ω·h 1.80–1.90), the 131-node shell specification and 24 lint rules.
- 2026-11-13: section 12, the body rebuilt from definitions (structure, mechanical parts, shell lattice); the rigger integration and all measurements still to do.
