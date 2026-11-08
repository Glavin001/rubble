# Verification plan: soft bodies, rigid bodies and their seams

Status: **plan for review** (2026-11-07). Docs only. It now includes the measured probes of `fix/etk-strength` (49a30f40, §0.1).
Branch `docs/verification-plan`, from `perf/explicit-2khz` (2820e47e).

**The ask** (user, 2026-11-07). Before more performance work, get test coverage we can be
certain of: "mathematically and physically sound", not made up off the top of our heads.
It must cover five things:

* BeamNG vehicle files are handled correctly;
* the soft body is correct;
* the rigid body is correct;
* soft and rigid are correct together;
* "all the seams" are handled.

The user suspects the soft↔rigid coupling. It may not give the same reaction forces as
soft↔soft contact. The evidence: the ETK looks right when it hits other vehicles, but it
crushed only about 15 cm at 113 km/h through the 110k ring of 25 kg bricks.

**Rule of the investigation.** Never tweak working physics without evidence. The order is:

1. physical accuracy and mathematical soundness;
2. then empirical checks;
3. if the maths is right but the look is wrong, fix the authoring or the conversion, not
   the solver.

This plan changes nothing in the physics. Every fix it might lead to is a decision for
the user (AGENTS.md).

**Sources read for this plan:**
* the docs: `vehicle-test-strategy.md`, `jbeam-semantics-spec.md`,
  `explicit-vehicles-in-world.md`, `explicit-beams.md`, `vehicle-acceptance-criteria.md`,
  `explicit-beams-vehicle-contact-spec.md`;
* the code of the coupling's contact rule: `engine-softbody/src/explicit_rigid.rs`,
  `explicit_beams.rs`'s node pass, and `slang/kernels/explicit_beams_rigid.slang`;
* the headers and test lists of every test file cited below.

Tests are cited as `file::test` only where I read them. `engine-world/tests/explicit_drive_world.rs`
was on the reading list, but it does not exist on this branch.

---

## 0. Findings from reading (before any test is written)

These come from the code and docs alone. Each one becomes a test in §3 or §4.

**F1. Rigid proxies use RoR's contact rule, whatever the vehicle's profile.**
* **Static boxes and static shapes.** The node pass gives them
  `ContactSurface::BOX.exact_if(exact).dissipative_if(dissipative)`
  (`explicit_beams.rs`, node pass). For a JBeam vehicle such as the ETK, that is the
  BeamNG profile's §37 rule: the approach is stopped, no positive work, and exact
  exponentials.
* **World rigid proxies.** They get plain `ContactSurface::BOX`. This is RoR's rule:
  80 % of the approach is removed, with Schraudolph exponentials.
  * CPU: `ProxySet::node_contact_with`, both the infinite-mass and the finite-mass
    branches.
  * GPU: `explicit_beams_rigid.slang` lines 441 and 454, `boxSurface()` with no
    `surfaceV(…, vehicle)`.
* **The consequence.** `explicit-vehicles-in-world.md` §1 says "a partner of infinite
  mass at rest is bit-identical to a static shape". That holds only in the RoR profile.
  `explicit_rigid.rs::static_proxy_is_the_static_shape_bit_for_bit` checks it on the
  `.truck` procedural sedan only, which runs the RoR profile.
* **For the ETK, contact with a World body follows a different law from contact with
  the same wall as a native static shape.** This is a documented-design deviation, and
  its size is unmeasured.

**F2. The contact law depends on the pair: there are five different pairings.**

| Pair | Law | Penetration term | Profile-aware |
|---|---|---|---|
| node ↔ ground plane | `contact_response`, GROUND surface | yes (§37 lift in the BeamNG profile) | yes |
| node ↔ static box/shape | `contact_force`, BOX surface, p = 0 | none | yes |
| node ↔ World rigid proxy | `contact_force`, BOX surface, p = 0, two-body share and reduced mass | none | **no** (F1) |
| node ↔ collision triangle (soft↔soft, self) | §10.3 two-body: the pair's relative acceleration cancelled (normal and tangential), reduced mass with mass splitting, then the §37 rule with a position step | yes (position step) | yes |
| triangle/edge ↔ rigid shape (surface contact, opt-in) | §10.3 two-body with the shape as contacter, then §37 always, §38.8 stick friction | yes | always §37 |

So soft↔soft and node-only soft↔rigid differ in four ways:
* the normal law (stop 100 % of the approach, against 80 %);
* what is cancelled (the relative acceleration of the pair, against the node's force share
  `s·f`);
* the penetration response (a position step, against none);
* the friction stick law.

Under identical masses and velocities they are **not expected** to give identical
impulse histories. The question for §3 is whether they give the same *physically
determined* quantities: momentum exchanged, the bounds on restitution, and the energy
dissipated, with crush as its consequence.

**F3. Inside a World window, a proxy body is a free body.**
* §1 "The proxy inside the frame": no gravity, no other contact, no joint.
* The bricks in the ring therefore do not push on each other during the 33-step window
  (1/60 s). The car meets each touched brick's own 25 kg, not the mass of the
  contact-connected column behind it.
* That column's resistance reaches the car only through the next rigid step's AVBD
  solve. The ring runs 4 iterations (`brick_ring_drive.rs` `ring_world_def`), and the
  following window then starts with the bricks' new velocities.
* §3.5 derives what this implies, and §3.6 gives the tests that decide it.

**F4. The ring drive runs node-only contact.**
* `brick_ring_drive.rs` sets no surface contact, and the ETK's nodes are about 0.4–0.5 m
  apart (585 nodes; `measurements/explicit-2khz.md`).
* A brick's struck face is 1.0 × 0.5 m (1 × 0.5 × 0.5 m bricks laid tangentially). Front
  structure between node lines can pass through bricks with no force
  (`explicit_thin_box.rs::a_box_between_node_lines_lets_the_body_pass_untouched`, and the
  ETK's pole, `jbeam-semantics-spec.md` §38).

**F5. The ETK's own rigid-wall crush is not recorded.**
* `jbeam-semantics-spec.md` §29.1 gives the ETK's 56 km/h wall run (2 broken, 359 bent,
  7.1 kJ KE left of 172 kJ), but no crush distance.
* Without that baseline, a small crush in the ring cannot yet be blamed on the coupling.
  It could be the car's content.

**F6. Every coupling test that exists uses the RoR-profile procedural sedan.** This
covers:
* `explicit_rigid.rs` (all four tests);
* `explicit_rigid_gpu.rs`;
* `explicit_vehicles.rs`;
* `explicit_surface_proxy_gpu.rs`;
* engine-metal's `explicit_parity.rs::metal_coupled_sedan_into_a_rigid_box_matches_cpu`.

None covers the BeamNG profile (JBeam content, §37 contact, exact friction). That is the
path the ETK takes.

**F7. Raycast wheels do not see rigid bodies.** The ring drive and the World ETK run on
raycast wheels. Those wheels meet the vehicle's own ground plane only, never a World body.
Reported by the coordinator's measurements (§0.1), and not yet a test.

### 0.1 Measured evidence (branch `fix/etk-strength`, 49a30f40, 2026-11-07)

These are diagnostic probes, not committed tests. They change no solver, kernel or
importer code.

* `engine-softbody/examples/crush_probe.rs`: a car into a rigid wall on the CPU
  reference.
* `engine-softbody/examples/coupling_probe.rs`: one car into the same obstacle through
  native colliders and through the coupling's CPU reference (walls, a free box, brick
  proxies, drops).
* `engine-world/examples/coupling_wall_gpu.rs`: the ETK into an anchored box, a free box
  or a brick wall in a small GPU World. It covers World rates, momentum while flying, and
  a rigid-only shooter.
* Logs: `~/Development/rubble-artifacts/etk-strength/`.

They answer several of this plan's questions in part. Each result below is labelled as
**evidence** (it stands as a measurement) or as a **gap** (a committed test is still
needed).

| # | Measured | Answers | Evidence / gap |
|---|---|---|---|
| E1 | **ETK at 50 km/h into an immovable wall, four paths:** native box/shape, static rigid proxy, heavy (free) proxy, and GPU World anchored body. They agree within about 5 % in momentum exchanged. Without raycast wheels: 0.8 % in momentum and 5 % in peak force. | X2 (§3.2), the cross-path for a wall | **Evidence** that the coupling transmits a wall's reaction as the native wall does, to within the F1 law difference. **Gap:** it is a probe; T-X2 should commit it with the C tolerance (nudged-twin spread). The 5 % with wheels is F7 (wheels see only the native ground). |
| E2 | **Two real coupling differences** that `static_proxy_is_the_static_shape_bit_for_bit` cannot see, because it uses an RoR-profile car. (a) Rigid-body contact uses plain `ContactSurface::BOX`, so it ignores the BeamNG profile's exact friction and no-positive-work rule (CPU `explicit_rigid.rs` `node_contact_with`; GPU `explicit_beams_rigid.slang` proxy branches against the static-slot branch that uses `surfaceV(boxSurface(), vehicle)`; line numbers differ between branches). (b) Raycast wheels do not see rigid bodies (F7). | F1, F7, T5.1 | **Confirmed** by measurement. Still a **gap** for a committed test (T5.1, T5.5) and a **decision** for the user. |
| E3 | **Newton III and momentum.** CPU box: residual 0.004 N·s. GPU World box: 0.0. GPU ETK into floating bricks: ±7 %. **A rigid-only World with no car** (a 1,453 kg block shot into floating bricks) **loses 5.5 % of its momentum.** | §3.3, G3.1 | **Evidence** that the coupling itself closes momentum, and that the ±7 % is the AVBD multi-body solve, not the coupling. **New gap:** a rigid-only momentum test (T3.6) belongs to the rigid seam, not the coupling seam. |
| E4 | **Drop from 1 m onto every ground path** (native plane, static shape, static proxy, box, heavy proxy): peak force 697–720 kN, impulse 10.3 kN·s on every path. | X1 (§3.2) | **Evidence**: the ground paths agree in impulse and within 3 % in peak force at 1 m. **Gap:** 0.5 m and 15 m, rebound and settle height, and a committed form. |
| E5 | **ETK at 56 km/h into a rigid wall (CPU) against the NHTSA small-sedan band.** Dynamic crush 0.55 m (band 0.62–0.75 m). Peak 67 g through a 5 ms box filter (band 34–47 g, CFC 60). Stiffer and springier than the band. Importer values match an independent parse for all 2,767 beams. | D0 (§3.1), T1.1 | **Evidence** that the import is faithful (every beam) and that the ETK is stiffer than measured sedans in the frontal: a content property, not an importer error. **Gap:** the peak g is not comparable until it uses a proper CFC 60 filter (SAE J211); until then the 67 g is reported, not judged. |
| E6 | **A loose wall of 25 kg bricks converges with World rate.** The car's momentum lost is 9.1, 10.1 and 10.1 kN·s at 60, 120 and 240 Hz. The bricks' peak force on the car is about 100 kN. The ETK front's yield against a rigid wall is about 600 kN. | §3.5 hypothesis (C), R1/R2 | **Evidence against** the window-latency hypothesis at these rates (see §3.5's revision). **Gap:** the iteration count was not swept, and a column in compressive contact (R1's row against one block) is still unmeasured. |

**Status of the plan's items after these probes.** The probes are examples, not
asserting tests. "Measured" means a number exists. It does not mean the item is covered:
each still needs a committed test with a typed tolerance.

| Item | Status |
|---|---|
| D0, the ETK into a rigid wall (§3.1) | **Measured** (E5): 0.55 m dynamic crush at 56 km/h, `crush_probe.rs` |
| X2, the wall cross-path (§3.2) | **Measured** (E1): four paths within about 5 % momentum, `coupling_probe.rs`, `coupling_wall_gpu.rs` |
| X1, the drop cross-path (§3.2) | **Measured** at 1 m (E4): 697–720 kN and 10.3 kN·s on every path, `coupling_probe.rs` |
| §3.3, the Newton III residual | **Measured** (E3): CPU 0.004 N·s; GPU World box 0.0 |
| F1, rigid proxies ignore the BeamNG-profile surface | **Open**: confirmed, untested, a decision for the user |
| F7, raycast wheels do not see rigid bodies | **Open** |
| AVBD multi-body momentum loss (5.5 %, rigid only) | **Open, a top gap**: T3.6 |
| X3, rigid box against a soft block of the same mass | Not measured |
| R1, a row of bricks against one block | Not measured |

**What this changes in the plan:**
* D0 and the wall and drop cross-paths have measured answers. They move from "first to
  investigate" to "commit as tests with typed tolerances".
* The investigation's open questions are now three:
  * the user's decision on F1/F7;
  * AVBD's own momentum loss in multi-body contact (T3.1, T3.6);
  * the CFC 60 comparison for the ETK's content.

---

## 1. The oracle hierarchy

What each kind of oracle can prove, and what it cannot. A claim of correctness names its
oracle. A test that only says "looks right" names (h).

| | Oracle | Proves | Cannot prove |
|---|---|---|---|
| (a) | **Closed form / analytic.** The step's exact discrete solution, or a continuum solution with a stated discretisation error. Examples: the 1-DOF spring's discrete ω_d, σ_d (`vehicle-test-strategy.md` layer 1); the plastic/elastic bounds of a two-body collision; an inelastic chain's common speed; a sliding stop distance v²/2μg. | That one feature implements its stated law, to rounding or to a derived discretisation bound. | That the law is the right one for the content, or that features compose correctly. |
| (b) | **Conservation and invariants.** Linear momentum with no external force; angular momentum about a fixed point; the energy ledger closes and mechanical energy never rises (beyond the integrator's bounded oscillation ω_max·h/2·E); Newton III per coupling (Σ action + Σ reaction = 0 to the fixed-point quantum); no growth of penetration; invariance under rigid frame motion (translation, rotation), origin rebase and spawn order (where specified); convergence under timestep refinement (where the step is not fixed by design). | That nothing creates momentum or energy, that couplings are equal and opposite, and that results do not depend on arbitrary choices (frame, origin, order). They are necessary conditions, and they apply to every scene, however complex. | Correct *magnitudes* of dissipation or force. A contact that dissipates too much or too little passes every one of them. |
| (c) | **Cross-implementation equivalence.** CPU reference ↔ WebGPU (native wgpu) ↔ Chrome WebGPU ↔ Metal (Slang→MSL) ↔ CUDA, bit-exact where the arithmetic is strict and order-fixed, else bounded by a derived float bound or by the chaos-bounded ensemble (§2.0). | That each port computes what the CPU reference computes. | That the CPU reference is right. A shared bug passes. |
| (d) | **Cross-path equivalence.** The same physical scenario through two code paths. For example: a wall native to the soft solver against the same wall as an immovable World body; a rigid box against the same mass as a stiff soft block; native ground against a rigid slab. | That the seams add nothing physical: a coupling path gives the physics the other path gives, or differs by an amount the differing *laws* (F2) predict. This is the oracle the user's suspicion needs. | Anything both paths get wrong in the same way. |
| (e) | **Black-box ground truth on content tuned for that solver.** RoR's unmodified solver (`inspiration/ror-harness`, `REF=ror soft_calibrate`, `explicit_thin_box.rs`'s opt-in E36 cases) on real RoR `.truck` content. | That Rubble's RoR profile reproduces a real, independently written explicit node–beam solver on its own content: per-step node trajectories, broken and bent sets. | Anything about BeamNG semantics; anything about the World coupling (RoR has no AVBD bodies). Opt-in only: never a CI dependency (AGENTS.md). |
| (f) | **BeamNG.** Closed source. (f1) Spec tests per documented field, each citing its documentation.beamng.com page (`jbeam-semantics-spec.md` [B], [S], [BD] …), with every undocumented rule marked **Choice**. (f2) *Optional, if the user can run BeamNG*: behavioural recordings from BeamNG itself through BeamNGpy, BeamNG's open-source Python API (it drives BeamNG.tech and logs vehicle state through documented sensors). That needs Windows, a BeamNG.tech licence and a check of the EULA's terms for recordings. Use the API as documented; do not read BeamNGpy's or BeamNG's code (clean room). | (f1): that Rubble implements what BeamNG *documents*. (f2): that a given JBeam car behaves as in BeamNG in the recorded scenes (same file, same nodes: rest sag, a wall pulse, crush and deformation per node group if the sensors expose node positions). | (f1) cannot prove BeamNG *equivalence*: the documentation is silent on, for example, node–ground response, the yield law and contact (§28's and §3's Choices). (f2) proves only the recorded scenes. **Without (f2), BeamNG equivalence is not provable** (§5). |
| (g) | **Real-world data under matching conditions.** NCAP/NHTSA 56 and 64 km/h rigid barrier, FMVSS 214/216a/206 (`crash-reference-targets.md`), drop tests. | That a car *as authored* lands in a measured spread. **Labelled sanity checks**: they validate content plus solver together. | That the solver is right: a wrong solver with compensating content passes. |
| (h) | **Visual judgement.** | Nothing on its own. It finds candidates for (a)–(g). | Everything; it comes last. |

---

## 2. The seam-by-seam matrix

### 2.0 Conventions for tolerances

Every tolerance is one of the following kinds, and says which.

* **R (rounding).** Strict f32 single-order arithmetic: bit for bit (0 ulp). Fixed-point
  contact sums: the quantum is 1/32 N for forces (`explicit-vehicles-in-world.md` §3),
  so a sum of n contacts has error ≤ n/64 N per step. Position steps: 2⁻²⁴ m.
* **D (discretisation).** A derived order. Examples:
  * symplectic Euler's frequency shift ≈ (ωh)²/24;
  * the staggered coupling's latency of one window, Δ;
  * a node's tunnelling depth ≤ 1.25·v·h (`invariants_small.rs`).
* **C (chaos-bounded).** Crash outcomes are chaotic under node-only contact
  (`explicit-vehicles-in-world.md` §10, the 0.02 mm nudge giving 199 mm).
  * Two runs of the same scenario through different paths are compared against the
    spread of an **ensemble of nudged twins** of one path: N = 8 runs, initial positions
    perturbed by 10⁻⁵ m.
  * The tolerance is the ensemble's max–min range of the measured quantity.
  * The assertion: path B's value lies within path A's range widened by its own range.
    It is not a Choice: the ensemble *measures* how well the scenario determines the
    quantity.
* **Choice.** Anything else, stated as such.

### 2.1 JBeam/BeamNG import → solver parameters

**Must be true.** Every JBeam field that the census finds in a supported car maps to the
solver parameter that BeamNG's documentation defines, with units and frame converted
(+X left, +Y rear, +Z up → +X left, +Y up, +Z forward; `jbeam-semantics-spec.md` §2).
Undocumented behaviour is a labelled Choice. No field is silently dropped.

**Oracle.** (f1) spec, plus (a) for each rule's force law.

**Existing:**
* `jbeam.rs` unit tests: `relaxed_json_accepts_jbeam_syntax`, `expressions_follow_lua`,
  `loads_parts_slots_variables_and_tables`, `config_selects_another_part_and_clamps_vars`,
  `fuel_tank_mass_spreads_over_its_fuel_nodes`, `descriptors_round_trip_through_json`,
  `lbeam_third_node_in_row_options_with_the_node_colon`.
* `explicit_jbeam.rs`:
  * `kinds_precompression_and_groups_load`
  * `passive_members_break_with_their_group_but_never_set_it_off`
  * `supports_push_only_and_break_by_length`
  * `bounded_beams_move_to_their_limit_rates_over_the_bound_zone`
  * `torsion_bar_holds_its_angle_and_breaks_with_its_hinge`
  * `node_couplers_couple_at_spawn_and_break_by_strength_and_group`
  * `torsion_forces_are_the_exact_lever_law`
* `explicit_jbeam_ext.rs` (§16–§27, one rule each):
  * `several_groups_broken_by_any_without_propagation`
  * `triangles_break_with_their_sides_unless_disabled`
  * `deform_groups_trigger_on_yield_past_the_ratio_or_on_break`
  * `anisotropic_rates_switch_at_rest_over_the_transition_zone`
  * `pressured_beam_follows_the_gas_law_on_its_surface`
  * `rebound_damping_takes_the_expansion_values`
  * `precompression_ramps_in_over_its_time`
  * `lbeam_resists_angle_not_arm_length_and_conserves_momentum`
  * `slide_node_hangs_on_its_rail`
  * …
* `elements_oracle.rs` (LBEAM, torsion bar, slidenode, anisotropic, beam options).
* Layer 1, `invariants_small.rs`:
  * `support_beam_pushes_only_and_breaks_by_length`
  * `bounded_beam_follows_its_blended_law`
  * `anisotropic_beam_is_a_bilinear_oscillator`
  * `pressured_beam_follows_the_gas_law`
  * `lbeam_conserves_momentum_angular_momentum_and_energy`
* `explicit_jbeam_ext.rs::community_vehicles_do_not_creep_at_rest` (opt-in content).

**Gaps:**
* **G1.1 No field-by-field conformance table.** The rule tests check force laws. No test
  walks the importer's output field by field: for each documented field, its default, an
  explicit value, unit scaling, and the parameter it lands in. Examples: `nodeWeight`,
  `beamSpring`, `beamDamp`, `beamDeform`, `beamStrength`, `deformLimit`,
  `deformLimitExpansion`, `beamPrecompression`, `precompressionRange`, `frictionCoef`,
  `collision`, `selfCollision`, `group`, `breakGroup`, `breakGroupType`, `dampCutoffHz`,
  the `$variables` and the property-row inheritance across sections.
* **G1.2** The census ("no unsupported physics feature", §29.1) is a measurement on local
  content. No committed test fails when a new field appears.
* **G1.3** The importer's **mass** is not checked against the file. The total of the node
  weights, after property inheritance, should equal the file's explicit sum.

**Proposed:**
* **T1.1 `jbeam_field_conformance`** (CPU, no content needed). One synthetic JBeam part
  per documented field, in Rubble's own fixture style (`jbeam_fixture`). Each row asserts:
  * the importer's `SoftVehicleDesc` value;
  * the solver's spawned parameter;
  * the documentation URL, in a `const` table that the test prints.

  Tolerance: R (exact, except declared unit conversions). Rows for undocumented
  behaviour carry `Choice` and the spec section.
* **T1.2 `census_rejects_unknown_fields`.** The importer, fed a field outside the table,
  reports it in the census with a count. Committed fixtures only.
* **T1.3 `imported_mass_is_the_files`.** Σ node mass equals Σ of the file's
  `nodeWeight` after inheritance, exactly. On fixtures, and on community cars opt-in.

### 2.2 The soft solver alone

**Must be true:**
* each beam kind follows its law (a);
* the step is symplectic Euler with exact discrete frequency and decay;
* free flight conserves momentum (M·g·t) and never gains energy beyond ω_max·h/2·E;
* the ledger closes;
* plastic and break laws match their closed forms;
* contact with the ground never tunnels and does no positive work (BeamNG profile).

**Oracle.** (a), (b), (e) for the RoR profile.

**Existing.** All of layer 1 (`invariants_small.rs`, 19 tests, and
`invariants_small_gpu.rs`, 6). Also:
* `energy_ledger.rs`:
  * `ledger_closes_in_a_sedan_rear_end`
  * `ledger_closes_in_the_showcase_t_bone_off_the_door_middle`
  * `self_contact_never_adds_energy_in_the_120_kmh_wall`
* `crash_invariants.rs`:
  * `crash_ledgers_close_and_energy_never_rises`
  * `rest_no_plastic_no_breaks_no_creep`
  * `rigid_fit_recovers_a_rigid_motion`
  * plus the `#[ignore]`d content failures, listed in `vehicle-test-strategy.md`
* `explicit_beams.rs` (RoR rules from definition, 31 tests).
* `explicit_oracle.rs`:
  * `car_rests_without_yielding_or_clipping`
  * `wall_56_crushes_the_front_without_clipping`
* `showcase_car.rs`:
  * `contacts_do_no_positive_work`
  * `fixed_contact_quantisation_does_no_work`
  * the NHTSA/FMVSS scenes
* `explicit_thin_box.rs::e36_front_pole_matches_ror_at_offsets_0_016_0233` and
  `e36_side_pole_matches_ror_at_door_and_between_door_nodes` (opt-in RoR black box).

(`cpu_reference.rs` tests `CpuSoft`, an older implicit solver, not the explicit solver.
It is no evidence for the explicit path.)

**Gaps** (beyond `vehicle-test-strategy.md`'s own Coverage gaps, which stand):
* **G2.1 Momentum in crashes is asserted only in free flight.** Ground and wall impulses
  are logged as work, not impulse. (Listed in `vehicle-test-strategy.md`.)
* **G2.2 No frame-rotation invariance test** for a single vehicle against static
  geometry. Only vehicle↔vehicle has one:
  `explicit_contact.rs::a_turned_spawn_turns_every_node_about_the_spawn_point`, and
  `explicit_far_origin_gpu.rs`, which covers translation.
* **G2.3 No test of the BeamNG profile against the RoR black box.** It is not expected
  to match, so this is fine; but there is also no other external check of the BeamNG
  profile (§5).
* **G2.4 No timestep-refinement study.** 2 kHz is fixed by design, so this is not a
  correctness criterion. As a *measurement* it would show how far the 2 kHz answer is
  from the continuum answer for a crash metric. That needs a test-only step override,
  which does not exist; it is a measurement only (AGENTS.md).

**Proposed:**
* **T2.1 `ground_and_shape_impulse_log`.** A per-step log of the Σ impulse that the
  ground, static boxes and static shapes give each vehicle. Then the momentum identity
  ΔP_vehicle = Σ J_ext + M·g·t holds to R in every crash scene. Add it to
  `crash_ledgers_close_and_energy_never_rises`.
* **T2.2 `turned_scene_is_the_turned_result`.** The sedan, and a JBeam fixture car, into a
  wall, with the whole scene (car, wall and ground normal kept vertical) yawed by 37°.
  * Node trajectories up to first contact + 20 ms: within R-growth, 10⁻⁵ m.
  * Crush and peak g after that: C.

  Yaw only, because the ground plane is horizontal.

### 2.3 Rigid AVBD alone, and unchanged with vehicles absent

**Must be true:**
* (i) The rigid solver's laws:
  * free fall exact for its integrator;
  * Coulomb friction stops a slider in v²/(2μg) (to D) and holds on an incline of
    tan θ < μ;
  * free-body collisions conserve linear and angular momentum;
  * an inelastic chain of n touching equal bodies, struck at one end with impulse J and
    converged, moves at J/(n·m) (no restitution between rigid contacts);
  * resting stacks gain no energy.
* (ii) Adding the vehicle machinery changes **nothing** in a World with no vehicles:
  every rigid state word bit for bit, and no measurable time.

**Oracle.** (a) and (b) for (i); (c) bit-identity for (ii); the pinned three-avbd
reference as (e)-like regression.

**Existing:**
* `engine-metal/tests/oracle_accuracy.rs` (criteria taken from pinned three-avbd's tests):
  * `stack_settles_like_three_avbd_ref`
  * `rope_joint_error_bounded_like_three_avbd`
  * `pyramid_stays_finite`
  * `resting_brick_stack_does_not_gain_energy`
  * `ground_friction_slows_a_sliding_box`: asserts only that the box slows, not v²/2μg.
* `engine-webgpu/tests/ring_hold.rs::ring_110k_stands_away_from_impact_and_ball_rolls_through`
* `gables_hold.rs::gables_506k_keeps_unhit_walls_standing_and_both_balls_advancing`
* `columns_hold.rs` (4 tests)
* `broadphase_oracle.rs` (brute-force pair oracle), and
  `engine-oracles/tests/convex_slang_cpu.rs` (narrowphase against exhaustive SAT/distance
  oracles)
* `ccd_tunnel.rs` (5 tests)
* `engine-world/tests/world.rs::world_physics_is_the_raw_simulation_bit_for_bit`
* `engine-world/tests/equivalence.rs` (`record_scene` and `compare`, both `#[ignore]`d).
  It records and compares two builds over the house, arch, grid, ring, gables, city and
  vehicle scenes.
* `explicit_rigid_gpu.rs::coupling_off_is_bit_for_bit`: the vehicle side.
* engine-metal `webgpu_parity.rs::box_drop_matches_metal`, `columns_wgpu_parity.rs`.

**Gaps:**
* **G3.0 (top gap, measured): AVBD loses momentum in a multi-body impact.**
  * Measured in a rigid-only World with no vehicle (E3, `coupling_wall_gpu.rs`'s shooter
    check): a 1,453 kg block shot into floating bricks loses **5.5 %** of its momentum.
  * The coupling's own residual is 0.004 N·s on the CPU and 0.0 on the GPU World box.
    So the ±7 % in the ETK-into-bricks run belongs to this seam, not to the coupling.
  * No test asserts momentum in any multi-body AVBD impact. The cause is not yet known:
    iteration count, the penalty ramp, or something else. Measuring it is T3.6, and any
    fix is the user's decision.
* **G3.1 Few analytic tests of AVBD itself.** Its suite is mostly
  reference-checkpoint and regression. There is none for the stop distance, the incline
  threshold, momentum in free collisions, or **momentum propagation through a contact
  chain against iteration count**. That last one is the property the brick ring
  depends on (§3.5).
* **G3.2 "Unchanged with vehicles absent" has no committed, bounded check.** The
  equivalence harness can do it, but it is manual and `#[ignore]`d. Memory
  (goal-drive-through-brick-ring) requires rigid-only scenes to be benchmarked against
  main whenever vehicle work touches shared code.

**Proposed** (GPU, each run bounded ≤ 10 min under a kill timer):
* **T3.1 `inelastic_chain_converges_with_iterations`.**
  * Setup: n = 10 boxes, 1 × 0.5 × 0.5 m, 25 kg, touching in a row, on no ground, no
    gravity. One World step with an impulse record J = 25 kg × 30 m/s on the first box,
    along the row.
  * Measure, after the step: each box's velocity, and the spread max v_i − min v_i.
  * Converged rigid contact: every v_i = J/(n·m) = 3 m/s.
  * Sweep iterations ∈ {4, 8, 16, 32, 64} and dt ∈ {1/60, 1/120}.
  * Assert that momentum is closed to R. Report the spread against iterations. The
    assertion on the spread is a Choice to set after seeing the curve; the measurement
    is the deliverable.
* **T3.2 `slider_stops_at_v2_over_2_mu_g`.**
  * Setup: a box at 3 m/s on a static slab, μ 0.5.
  * Stop distance v²/(2μg) = 0.917 m, within D. AVBD integrates with the velocity at
    the step's end: the discrete stop distance is Σ (v − kμgΔ)·Δ, which differs from the
    continuum by about vΔ/2 = 2.5 cm at 60 Hz. Derive the exact discrete sum in the test.
* **T3.3 `incline_holds_below_atan_mu`.** μ 0.5: holds at 25° (tan 0.466), slides at
  28° (tan 0.532).
* **T3.4 `free_boxes_conserve_linear_and_angular_momentum`.** Two boxes in an off-centre
  collision, no gravity. P and L about the origin closed to R plus the solver's
  per-step tolerance. Report the latter.
* **T3.6 `multi_body_impact_conserves_momentum`** (rigid only, no vehicle). Measured
  today at −5.5 % (E3).
  * Setup: the probe's scene as a test. A 1,453 kg block into floating bricks, with no
    gravity and no ground, so there is no external impulse. Also a 2-body and a 10-body
    row for the trend.
  * Measured: total linear momentum each step, and its loss against the iteration count
    {4, 8, 16, 32} and dt {1/60, 1/120, 1/240}.
  * Bound: R, plus the solver's stated per-step tolerance. That bound must first be
    derived from AVBD's formulation; until then, report the loss and do not assert it.
    It is `#[ignore]`d with the measured value while the loss stands.
* **T3.5 `no_vehicles_is_bit_for_bit_main`** (CI-able).
  * Run the ring (28k) and gables-12 for 200 steps with no `ExplicitVehiclesDef`, and
    with one that has zero vehicles.
  * Hash every body state word per step and compare it with a hash recorded **by the
    same test on main** (`equivalence.rs`'s record mode), not with any `inspiration/`
    output.
  * Plus a timing gate. The A/B/A median of the World step, one GPU job at a time, must
    agree within its own A/A spread.

### 2.4 Soft ↔ soft

**Must be true:**
* contact forces are equal and opposite (the triangle nodes take −w·f);
* linear momentum is conserved by contact;
* no positive work (BeamNG profile, §37);
* no tunnelling below r/h = 40 m/s normal speed;
* swapping vehicle order changes results only as the spec says.

**Oracle.** (a), (b), (c), (e) (`explicit-beams-vehicle-contact-spec.md` §9, RoR
profile).

**Existing:**
* `explicit_contact.rs` (19 tests), notably:
  * `one_node_over_one_triangle_gets_the_contact_force_and_the_triangle_its_reaction`
  * `the_contact_conserves_momentum`
  * `vehicles_that_never_meet_are_bit_identical_to_running_alone`
  * `swapping_vehicle_order_changes_the_result`
  * `candidate_order_within_a_triangle_changes_the_result_by_rounding_only`
  * `anti_clip_catches_a_point_faster_than_the_band_per_step`
* `invariants_small.rs::node_never_passes_a_triangle_below_the_band_speed`
* `energy_ledger.rs` (rear end, T-bone)
* `showcase_car.rs::contacts_do_no_positive_work`, `t_car_to_car_t_bone_at_80_kmh`
* `crash_invariants.rs::head_on_contact_speeds_within_the_band` (`#[ignore]`d: 15 of
  about 300k samples at 60 m/s)
* GPU, `explicit_contact_gpu.rs`:
  * `two_body_contact_steps_like_the_cpu`
  * `anti_clip_steps_like_the_cpu`
  * `two_sedans_rear_end_match_cpu`
  * `self_collision_two_body_steps_like_the_cpu`
  * …
* `explicit_far_origin_gpu.rs`, all four tests.
* Metal and CUDA: `metal_two_sedans_rear_end_match_cpu`,
  `cuda_two_sedans_rear_end_match_cpu`.

**Gaps:**
* **G4.1 Angular momentum through soft↔soft contact is not asserted.** Only linear
  momentum is (`the_contact_conserves_momentum`).
* **G4.2 No head-on test between two different cars in the BeamNG profile with a
  momentum assertion.** That is the ETK's case. The ledger tests use the procedural
  sedan and the showcase car.
* **G4.3 RoR §9's acceptance list** (head-on, T-bone, slow touch, no-triangle vehicle
  against the harness). No committed opt-in test runs it; I found only binaries
  (`explicit_vv_compare`).

**Proposed:**
* **T4.1 `vehicle_contact_closes_angular_momentum`.**
  * Setup: two JBeam fixture cars, flying (no ground), offset 0.5 m laterally.
  * L about the initial centre of mass closed to R (fixed point: n/64 N × lever × h per
    step).
* **T4.2 `etk_head_on_ledger`** (opt-in content).
  * Setup: ETK into ETK at 56 km/h each, flying, so momentum is exact.
  * Assert ΔP ≤ R, and that the ledger closes with no positive contact work. Record the
    impulse history (§3.3) for use as the soft↔soft reference.
* **T4.3 `ror_vehicle_contact_acceptance`** (opt-in, `RUBBLE_INSPIRATION`). Spec §9
  cases 1–5 as one test, skipped without the harness.

### 2.5 Soft ↔ rigid coupling

**Must be true:**
* For each node contact the body takes −f_c at the node's position. The window's
  impulse J_s and moment M_s are delivered whole to the body that owns the shape now
  (after splits).
* Linear and angular momentum close to R across the frame and through the World's
  impulse path.
* Energy does not rise beyond the integrator's free-flight oscillation.
* An immovable body at rest is the static shape bit for bit (§1's claim, **in both
  profiles**: F1).
* A free body of mass M struck by a vehicle of mass m at v ends with speed in
  [m·v/(m+M), 2m·v/(m+M)].
* Results converge as the World window Δ → h (§3.5).
* Sleeping bodies behave as awake bodies at rest.
* Fracture is forward-only, and freed pieces never exceed the whole body's change of
  motion (§4 of the design).

**Oracle.** (a), (b), (c), (d).

**Existing:**

| Test | Profile | Covers |
|---|---|---|
| `explicit_rigid.rs::static_proxy_is_the_static_shape_bit_for_bit` | RoR sedan | static proxy = static shape |
| `explicit_rigid.rs::momentum_closes_and_energy_does_not_rise` | RoR | P to 10⁻⁴, E ≤ free-flight floor; 200, 1500 and 6000 kg |
| `explicit_rigid.rs::box_of_known_mass_takes_the_momentum_exchange_physics_predicts` | RoR | plastic/elastic bounds; 300, 1200 and 4000 kg |
| `explicit_rigid.rs::a_light_box_moves_away_inside_the_frame` | RoR | 100 kg box ≤ elastic bound |
| `explicit_rigid_gpu.rs::coupled_crash_matches_the_cpu_and_closes_momentum`, `more_proxy_bodies_than_lanes_are_all_integrated`, `far_from_the_origin_the_coupling_is_the_origins_bit_for_bit`, `a_rebase_moves_the_proxy_records_with_the_origin`, `coupling_off_is_bit_for_bit`, `a_corrupt_proxy_header_fails_loudly_not_slowly` | RoR | parity, rebase, limits |
| `engine-world/tests/explicit_vehicles.rs::a_vehicle_shoves_a_free_box_and_momentum_closes` | RoR | World path: ΔP ≤ 2·10⁻⁴; box in [plastic, 2·plastic]; KE falls |
| `explicit_vehicles.rs::a_vehicle_breaks_a_prefractured_house` | RoR | forward-only; breaks follow at once; freed pieces ≤ 2·v |
| `explicit_vehicles.rs` (6 bit-exactness tests: cull, workgroup copies, node ranges, margin, parallel emit, parallel plan) | RoR | lossless optimisations |
| `explicit_vehicle_broadphase.rs` (3 tests) | RoR | plan from the broadphase = scan, word for word |
| `explicit_sleep.rs::a_parked_vehicle_sleeps_and_a_falling_box_wakes_it` | RoR | vehicle sleep |
| `explicit_surface.rs::the_body_takes_the_equal_and_opposite_force`, `explicit_surface_cast.rs` (3), `explicit_surface_friction.rs` (3), `explicit_surface_proxy_gpu.rs` | surface contact | Newton III, casts, friction, parity |
| engine-metal `metal_coupled_sedan_into_a_rigid_box_matches_cpu`, `explicit_surface_proxy_parity.rs` (one `#[ignore]`d: turned proxy) | RoR | Metal parity |

**Gaps:**
* **G5.1 (F1)** No coupling test in the BeamNG profile. The static-proxy bit-identity is
  known to be untested there, and from reading the code it is false.
* **G5.2 Angular momentum across the coupling is never asserted.** Only horizontal
  linear momentum is, although the design carries M_s for it.
* **G5.3 No cross-path test.** None compares the coupling with soft↔soft contact, with a
  native wall, or with native ground (§3).
* **G5.4 No window-rate convergence test.** Every World test runs at one dt.
* **G5.5 No test against many light bodies in contact with each other.** The coupling
  is tested only against isolated boxes (one free box, house chunks under threshold
  fracture). Nothing covers a stack, a row or the ring (§3.5).
* **G5.6 No mass-ratio sweep below 100 kg.** The lightest tested box is 100 kg; the
  ring's bricks are 25 kg and the demo's 0.25 kg.
* **G5.7 Sleeping rigid bodies struck by a vehicle.** The plan treats sleeping bodies as
  not waking the vehicle. Nothing checks that a sleeping box, once struck, takes the same
  impulse as the same box awake at rest.
* **G5.8 Bonds-model destructibles** (chunks as AVBD bodies with bond joints) are not
  tested with vehicles; only the threshold house is.
* **G5.9 No per-step, per-pair impulse history** is available for diagnosis (§3.3).

**Proposed:** §3 in full, plus:
* **T5.1 `static_proxy_is_the_static_shape_in_the_beamng_profile`** (CPU). The
  existing test with the JBeam fixture car (`jbeam_fixture::PART2_CAR`) or a `.truck`
  with `rubble_beamng_profile`.
  * Expected to **fail** today (F1). The test is the evidence. The decision whether
    proxies should follow the vehicle's profile is the user's (it changes physics). Until
    then, keep it `#[ignore]`d with its measured divergence in the doc comment, as
    `crash_invariants.rs` does.
* **T5.2 `coupling_closes_angular_momentum`** (CPU, then GPU parity).
  * Setup: a vehicle hits a free 1200 kg box 0.8 m off its centre line, flying.
  * L about a fixed world point, nodes plus box, closed to R. That is
    n_contacts·(1/64 N)·|r|·h per step, summed.
* **T5.3 `sleeping_box_is_an_awake_box_at_rest`** (World, GPU).
  * Setup: two Worlds, identical except that the box is asleep in one. A car at 10 m/s
    strikes it.
  * The box's velocity after the window and the car's momentum: bit for bit, or within
    the World's sleep-wake tolerance if wake changes the solve order (C).
* **T5.5 `raycast_wheels_meet_rigid_bodies`** (F7, open). A car on raycast wheels
  driven onto a static box ramp that is a World body. Today the wheels see only the
  vehicle's own plane: the test records that, `#[ignore]`d, until the user decides.
* **T5.4 `bonds_house_takes_the_vehicles_impulse`** (World, GPU). The house test in
  `StrengthModel::Bonds`. Momentum closed through the joints (ΔP ≤ 2·10⁻⁴·P0, as the
  existing World test), and the bonds loaded.

### 2.6 Soft ↔ terrain

**Must be true:** a vehicle meets the ground under it, and its reaction goes to the
terrain where the terrain is dynamic.

**State.**
* The explicit vehicles keep their **own flat ground plane** (`ExplicitBeams::ground`).
* Heightfields and planes are never proxies (design §1). Their feature queries exist
  (§38.3) but no proxy carries them.
* `brick_ring_drive.rs` takes the World's ground slab out of the vehicles' collision mask
  for this reason.
* The MPM terrain coupling (`terrain_surface.rs::a_car_crosses_an_open_area_on_its_patch_and_leaves_a_rut`)
  is for the rigid vehicle.

**Gaps:**
* **G6.1** There is no soft↔terrain seam to verify yet: no heightfield, no slope, no
  deformable ground for explicit vehicles.
* **G6.2** The flat ground and the World's ground slab are two grounds that must agree
  in height. Nothing asserts it beyond construction (`GROUND = 0.5` in
  `brick_ring_drive.rs`).

**Proposed:**
* **T6.1 `vehicle_ground_is_the_worlds_slab`.** For every scene that builds explicit
  vehicles in a World with a slab, the vehicles' plane equals the slab's top (R, exact).
* Heightfield contact gets its tests when it is built, with T3.x-style analytic slopes
  (a node at rest on a slope holds below atan μ).

### 2.7 World integration

**Must be true:**
* explicit time never drifts from World time by more than one step (h);
* rebasing the origin changes nothing;
* sleep is lossless for awake vehicles;
* contact groups are deterministic;
* every loop has a budget that fails loudly;
* proxy limits fail loudly (nothing is dropped).

**Existing:**
* `explicit_beams.rs::frames_carry_the_remainder` and
  `origin_rebases_to_node_zero_past_one_hundred_metres`
* `explicit_rigid_gpu.rs::a_rebase_moves_the_proxy_records_with_the_origin`
* `explicit_far_origin_gpu.rs`
* `explicit_sleep_gpu.rs`, `explicit_sleep.rs`
* `explicit_loop_budget.rs::a_world_step_past_the_pass_budget_fails_fast`
* engine-metal `explicit_loop_cap.rs`, engine-cuda `explicit_loop_cap.rs`
* `explicit_rigid_gpu.rs::a_corrupt_proxy_header_fails_loudly_not_slowly`

**Gaps:**
* **G7.1** Exceeding `PROXY_BODIES`/`PROXY_SHAPES` fails with a named error, by design.
  No test of it was found in the files read.
* **G7.2** The time-coupling invariant (Σ explicit steps per World step = Δ/h with the
  remainder carried) is not tested at the World level for 120 Hz (17, 17, 16) and
  60 Hz (33, 33, 34).

**Proposed:**
* **T7.1 `proxy_limit_fails_by_name`.** One vehicle among PROXY_SHAPES + 1 tiny static
  boxes; the step fails with the limit's name.
* **T7.2 `world_and_explicit_clocks_agree`.** Over 600 World steps at 60, 120 and
  90 Hz, explicit time minus World time ∈ [0, h).

### 2.8 Platforms

**Must be true:** the GPU ports compute the CPU reference.
* Bit for bit where the arithmetic is strict and the order is fixed (wgpu: `explicit_beams_bitwise.rs`, the sleep and far-origin tests).
* Elsewhere bounded: Metal and CUDA compile with fast math, so positions to millimetres
  and counts.

**Existing:**
* wgpu: `explicit_beams_bitwise.rs` (14 tests, including `cpu_reference_per_frame`),
  `explicit_contact_gpu.rs`, `explicit_rigid_gpu.rs`, `invariants_small_gpu.rs`,
  `explicit_jbeam_community_gpu.rs` (opt-in), `explicit_surface_proxy_gpu.rs`.
* Metal: `explicit_parity.rs` (7 tests), `explicit_drive_parity.rs`,
  `explicit_surface_proxy_parity.rs`, `vehicle_parity.rs` (the rigid vehicle).
* CUDA: `explicit_parity.rs`, 3 parity tests.
* Chrome: `explicit-wgsl-check.mjs` compiles every module in Tint;
  `explicit-check.mjs` runs end-to-end qualitative checks (moves, deforms, no GPU error);
  `community-check.mjs` checks broken and bent counts against the CPU.

**Gaps:**
* **G8.1 No coupled parity test in the BeamNG profile** on any backend (F6).
* **G8.2 Chrome has no numeric parity.** Its checks are qualitative and counts-based.
  Tint against naga codegen differences (the uniform-barrier issue in memory) can change
  numbers without failing any check.
* **G8.3 The Metal turned-proxy case is open**
  (`metal_surface_contact_meets_a_turned_box_proxy_like_the_cpu`, `#[ignore]`d).
* **G8.4 CUDA has no coupling parity test.** Only the explicit solver alone is covered.

**Proposed:**
* **T8.1** Extend `explicit_rigid_gpu.rs`, Metal `explicit_parity.rs` and CUDA
  `explicit_parity.rs` with the JBeam fixture car (BeamNG profile) into a free box.
  Tolerances:
  * first contact frames: R (wgpu) / their existing bars (Metal, CUDA);
  * momentum: R;
  * outcome: C.
* **T8.2 `chrome_frame_hash`.** One fixed-length run (the sedan into a free box, 120
  frames) in Chrome and in native wgpu from the same WGSL, comparing per-frame state
  hashes. Bit for bit is expected where Tint and naga agree on strict arithmetic. When
  they differ, report the first frame and buffer that differ, not a pass or fail.

---

## 3. The coupling-force investigation

**Goal.** Decide, with evidence, whether soft↔rigid transmits the reaction forces that
physics requires. Separate three possible causes of the ETK's 15 cm in the ring:
* (A) the car's content (its stiffness and strengths);
* (B) the contact law and geometry (F1, F2, F4);
* (C) the time coupling (F3).

Every scene runs on the CPU reference where possible (no GPU), and on the GPU World only
for what needs AVBD.

### 3.1 Baseline: the car alone (separates A from B and C)

**D0 `etk_rigid_wall_crush`** (CPU, opt-in content; minutes)
* **Scenario:** the ETK (BeamNG profile, the ring's wheels and drag settings) into a
  native static wall (`StaticShape`) at 56 and 113 km/h, 1.5 s.
* **Measured:** dynamic and residual crush (bumper centre-line nodes relative to the
  rear-seat crossmember), the deceleration pulse filtered at 100 Hz, and restitution.
* **Reference:** NHTSA 56 km/h spread (`crash-reference-targets.md`), labelled (g). At
  113 km/h there is no measured reference, so the run is reported only.
* **Reading:** if the ETK crushes far less than the NHTSA spread at 56 km/h into a
  *rigid* wall, then 15 cm in the ring is (A), content. The fix is in authoring and
  conversion (the user's rule), and no coupling test can say otherwise.
* **Measured** (E5, `crush_probe.rs`, a probe, not yet a test):
  * dynamic crush 0.55 m at 56 km/h, against the NHTSA small-sedan band of 0.62–0.75 m;
  * peak 67 g through a 5 ms box filter, against 34–47 g at CFC 60. The figures are not
    comparable until a proper SAE J211 CFC 60 filter is used;
  * stiffer and springier than the band;
  * the importer matches an independent parse for all 2,767 beams.

  So the ETK reaches a substantial crush against a rigid wall. The 15 cm in the ring is
  not explained by the car being unable to crush. Its front yields at about 600 kN
  against a rigid wall (E6).
* **Still to commit:** the test with a CFC 60 filter, the residual crush, and the
  113 km/h report.

### 3.2 The three cross-path scenes

Each runs once per path. The paths are:
* (N) native: the soft solver's own static shape or ground;
* (P) World rigid proxy, CPU reference (`ProxySet`): static (M = ∞) and dynamic;
* (S) soft: the same object as a stiff soft block (a vehicle, two-body contact);
* (W) World on the GPU, where AVBD is needed.

| Scene | Paths | Must agree, and why |
|---|---|---|
| **X1 Drop onto ground.** The JBeam fixture car and the ETK (opt-in) dropped flat from 0.5, 2 and 15 m. | N: native plane. P∞: a static slab proxy, top at the plane. P: a 10⁶ kg dynamic slab. W: the World slab as a proxy (mask changed for the test). | Momentum: the ground's impulse = M·(v_impact + v_rebound) ± M·g·t (T2.1's log, R). Rebound ratio and settle height, against the F2 law differences: N uses GROUND (§37 lift); P uses BOX, non-dissipative, p = 0. **Expected to differ.** The test measures by how much, and checks that P∞ and P(10⁶ kg) agree to D: the slab's motion is ≤ M_car/M_slab·v. |
| **X2 Into a wall.** 56 km/h, full width. | N: static wall (`StaticShape`). P∞: the same wall as a static proxy. P: a 10⁶ kg free wall. | N = P∞ bit for bit in the RoR profile (existing test). In the BeamNG profile: report the divergence (F1, T5.1). Crush, pulse and restitution: C between P∞ and P(10⁶ kg). |
| **X3 Into another body of the same mass.** A free block of mass M = m_car (and M = m_car/4, 4·m_car), flying, no ground. | P: a rigid box proxy of mass M. S: a soft block of mass M: a 3 × 3 × 3 lattice at the stiffest stable spring (ω_max·h ≤ 0.5, ζ 0.1), collision triangles on its faces, two-body contact. Also S′: the ETK into a parked ETK, since the ETK "looks right" against vehicles. | **Physically determined quantities must agree.** Momentum exchanged at separation: both in [m·v/(m+M), 2m·v/(m+M)], P closed to R. Energy: kinetic energy lost = ½μv²(1 − e²) with μ = mM/(m+M). Measure e on each path and assert that the car's dissipated energy (ledger: plastic + damping) matches ½μv²(1 − e²) − ΔE_elastic(block) within the ledger's closure, 10⁻³. **Then compare the cars' plastic energy between P and S after correcting for each path's measured e.** That correction is exact physics, so a remaining difference beyond C is the contact law's (F2), and it is measured, not assumed. |

**Why X3 answers the user's question.** If the coupling transmitted too little force,
the car's plastic work in P would be smaller than in S at equal e. The momentum balance
cannot hide it: the box's final speed fixes the impulse exactly.

### 3.3 The impulse history (instrumentation, both paths)

A test-only log, per step and per contact pair:
* soft↔rigid: (vehicle, shape) → Σ f_c, Σ moment, contact count k, and the mass-split
  count;
* soft↔soft: (vehicle, vehicle) → Σ f, count;
* the ground and static shapes: T2.1.

* **CPU.** `ProxySet` already sums per shape per step (`ProxyShape::sum`, then `impulse`,
  `moment` and `peak` per window). It needs a per-step tap. The vehicle contact needs
  the same tap in `ExplicitBeamsWorld`.
* **GPU.** The window's J_s, M_s and peak per shape are in the coupling buffer. A
  diagnostic readback gives the per-window history; per-step needs a debug ring, which
  is opt-in and off by default (lossless when off, bit for bit: T-test as
  `coupling_off_is_bit_for_bit`).

**Newton III residual**, per step and per pair:

ρ = |Σ_nodes f_c(float) + unfixed(Σ fixed(−f_c))|

**Bound:** n_contacts/64 N (R). Report max ρ / max |Σ f_c|.

### 3.4 Dependence sweeps (CPU unless marked)

| Sweep | Values | Scene | What theory predicts |
|---|---|---|---|
| Rigid mass ratio M/m_car | 0.0002 (0.25 kg) to 10 (25, 100, 300, 1200, 4000, 15000 kg) | X3, P path | Box speed inside [plastic, elastic] for every M. Below ~100 kg the in-window body is lighter than many single nodes are effectively, so check that mass splitting (k sub-bodies) never gives a box speed above the elastic bound (`a_light_box_moves_away_inside_the_frame` at 100 kg only). |
| Contact stiffness | The car's front stiffness ×0.5, ×1 and ×2 (content scaling in a test copy, never the shipped car) | X2, X3 | Peak force ∝ √k for a linear front (F = v·√(k·μ)), and the momentum exchanged unchanged. Separates content from contact. |
| World rate 1/Δ (GPU) | 30, 60, 120, 240, 480 Hz | X3-W, R1–R3 below | **Convergence**: the results at Δ and Δ/2 differ by O(Δ) and shrink (the staggered scheme is first order in Δ: §3.5). A quantity that does not converge is a coupling error, not physics. |
| AVBD iterations (GPU) | 4, 8, 16, 32 | R1–R3 | As T3.1: momentum propagation through a contact chain. |
| Surface contact | off / on | R2, R3 | F4: geometric force loss between node lines. |

### 3.5 What the time coupling implies for many light bodies

From `explicit-vehicles-in-world.md` §1 ("The proxy inside the frame") and §2 ("Time
coupling"), with:
* m_b, a brick's mass (25 kg);
* M_c, the car's mass;
* v, the closing speed (31.4 m/s at 113 km/h);
* Δ, the World window (1/60 s in the ring: `ring_world_def`);
* h = 0.5 ms.

**(i) Inside the window, each struck brick is an isolated free body.**
* No gravity, no ground friction, no contact with its neighbours.
* Its effective inertia, seen by the car, is m_b, split k ways over the nodes touching
  it.
* A brick struck at relative normal speed u takes at most (1+e)·μ_b·u, with
  μ_b = m_b·M_c/(m_b + M_c) ≈ m_b. It then separates and takes no more force this
  window: it is *faster than the car*, with no neighbour to stop it.
* The mean force the car can feel in one window is therefore bounded:

  F̄_window ≤ Σ_struck (1+e)·m_b·u_i / Δ

**(ii) Between windows,** the rigid step applies each brick's impulse at its start and
solves it with its neighbours. A converged inelastic contact chain shares the momentum
over the contact-connected cluster. With 4 iterations it may not converge (T3.1).

**(iii) The next window starts from the solved velocities.** The struck bricks may now
be slower than the car, and are struck again.

**What exact rigid physics says.** Bricks stacked without gaps form a column in
compressive contact. A car meeting the column meets its whole cluster mass M_cl at once,
because rigid contacts have no compliance. A crude count for the ring's base, labelled:
* the car's front touches about 2 bricks across × 2–3 courses;
* the base is 10 rows deep;
* so M_cl ≈ 4–6 × 10 × 25 kg ≈ 1.0–1.5 t, about the car's own mass, plus friction from
  the courses above.

For an equal-mass, perfectly plastic collision, ½·μ·v² with μ = M_c/2 is dissipated:
the energy of the same car hitting a rigid wall at v/√2. That is 80 km/h for 113 km/h.

So exact rigid physics predicts a **large** crush. The NHTSA 56 km/h band is around
0.5–0.7 m for sedans (`crash-reference-targets.md`); at 80 km/h it is more. This holds
only if the ETK's content is calibrated (D0).

**What the coupling gives in the limit of large Δ.** Per window the car meets about
4–6 isolated 25 kg bricks. It loses at most about (1+e)·N·m_b·v per window, and the
column's mass enters only after AVBD's solve, iteration-limited. So:
* crush depends on Δ and on the iteration count, as **numerical** parameters;
* the force is spread over many windows instead of one rigid-impact pulse;
* **peak force, and with it crush, is underestimated.**

The momentum is conserved and still transferred overall (the World test proves the
accounting). It is the force-time profile that is wrong, and crush depends on peak force
against the structure's collapse load.

**Order of the error.** The scheme is a staggered (explicit, partitioned) coupling. For
smooth contact its error is first order in Δ. For contact between bodies whose
coupled stiffness exceeds what a window can resolve (rigid neighbours), it is not small
at any practical Δ until Δ ≈ h. Stated as a **hypothesis**; R1–R3 decide it.

**Revision after measurement (E6, `coupling_wall_gpu.rs`, 2026-11-07).** The fact, the
measurement and the hypothesis are kept apart here.

* **Fact (design).** Inside a window, a proxy is free of its neighbours (F3). This is
  unchanged.
* **Measurement.**
  * A loose wall of floating 25 kg bricks struck by the ETK takes 9.1, 10.1 and
    10.1 kN·s of the car's momentum at 60, 120 and 240 Hz. This converges by 120 Hz,
    and 60 Hz is within 10 % of the converged value.
  * The bricks' peak force on the car is about 100 kN. The ETK's front yields at about
    600 kN against a rigid wall.
* **Revised hypothesis.**
  * **The window length is not the driver at these rates.** The window-latency
    mechanism of (i)–(iii) would make the result depend on Δ, and it does not
    (60 Hz is within 10 %).
  * The small crush in the ring follows from the force the wall can apply. A loose wall
    of light bricks loads the front to about 1/6 of its collapse force, so the front
    barely yields. That is the physics of such a wall, provided AVBD transmits the
    column's resistance correctly.
  * The "column mass at once" estimate above assumed a rigid column in compressive
    contact. The probe measured loose bricks. The two cases are not yet compared.
* **Still open:**
  * a column in compressive contact (R1, a row against one block). There, rigid physics
    predicts a much larger force;
  * AVBD's iteration dependence (T3.1);
  * AVBD's own 5.5 % momentum loss in multi-body impacts (G3.0, T3.6). That loss, if it
    lies in the contact chain, would reduce the force a column can transmit.

**Alternatives.** Listed for the user's decision after the evidence, none adopted here:
* proxy↔proxy contact inside the frame (neighbouring bricks collide at 2 kHz);
* a shorter World window near vehicles;
* more AVBD iterations near vehicles;
* surface contact (F4);
* the profile-consistent proxy law (F1).

### 3.6 The ring tests (GPU, after D0 and X1–X3)

| Test | Scenario | Measured | Pass criterion |
|---|---|---|---|
| **R1 `row_of_bricks_vs_one_block`** | Flying JBeam fixture car (then the ETK) at 31.4 m/s, no gravity or ground: (a) into 10 touching 25 kg bricks in a row (W); (b) into one 250 kg block (W); (c) the same 250 kg block on the CPU (P). | The car's momentum P_c(t), the bricks' common speed at the end, crush, and the impulse history. | Momentum closed (R) everywhere. (a) must converge to (b) as Δ → h and the iterations grow. Exact rigid physics: a touching row pushed at one end is one body, within the converged T3.1 spread. Report the (a)/(b) crush ratio against Δ and iterations. (b) against (c): within C plus the one-window latency of the World impulse (D). |
| **R2 `column_wall_strike`** | The same car into a free-standing 10-row × 4-course × 6-brick wall section with gravity, ground and friction (a ring slice). | As R1, plus bricks moved and their velocities. | Convergence in Δ and iterations, as R1. The converged crush is the reference; the 60 Hz × 4-iteration value is reported against it. |
| **R3 `ring110k_etk_113`** | The real scene: `brick_ring_drive_probe` with BR_CAR=etk, BR_SPEED=31.4, surface contact off and on, Δ ∈ {1/60, 1/120, 1/240}, iterations ∈ {4, 16}. | Crush, the car's speed after the wall, bricks moved, per-pair impulse peaks. | No pass/fail: it is the measurement the user asked about. It reads only after R1 and R2 have shown which factors matter. One GPU job at a time, each run ≤ 10 min under a kill timer. |

---

## 4. Priorities: implementation order

Smallest and most decisive first. "GPU" means AVBD or a backend is required. Every GPU
run is bounded (≤ 10–20 min, kill timer, one job at a time).

| # | Item | GPU | Why first |
|---|---|---|---|
| — | *Measured by probes (§0.1), still to commit as tests:* D0 (E5), X2 wall (E1), X1 drop at 1 m (E4), Newton III residual (E3), the loose-brick rate sweep (E6) | mixed | Convert the probes `crush_probe.rs`, `coupling_probe.rs` and `coupling_wall_gpu.rs` into asserting tests with typed tolerances. D0 needs a CFC 60 filter first. |
| 1 | **T3.6** AVBD multi-body momentum (rigid only, measured −5.5 %) with **T3.1** chain against iterations | yes | Top open gap: the rigid seam itself loses momentum, and every soft↔rigid result in a pile inherits it. |
| 2 | **T5.1** static proxy = static shape in the BeamNG profile (F1), and **T5.5** raycast wheels against rigid bodies (F7) | no / yes | Both confirmed by measurement and untested. Each measures a law divergence that every World test of the ETK inherits; changing them is the user's decision. |
| 3 | **X3** car into a rigid box against a soft block of the same mass, with the e-corrected energy comparison (§3.2) | no | The soft↔rigid against soft↔soft question, with exact momentum and energy; not yet measured. |
| 4 | **R1** a row of bricks in compressive contact against one block, with Δ and iteration sweeps | yes | The case E6 did not cover; decides whether a column transmits rigid-body force. |
| 5 | **§3.3** per-step impulse tap (CPU) | no | Instrumentation for items 3 and 4. |
| 6 | **T5.2, T4.1** angular momentum across the coupling and across vehicle contact | no | Closes the conservation set. |
| 7 | **T2.1** ground and shape impulse log, giving momentum in every crash | no | Closes `vehicle-test-strategy.md`'s gap. |
| 8 | **R2, then R3** wall slice, then the ring | yes | The user's scene, read through 1–7. |
| 9 | **T8.1** BeamNG-profile coupling parity: wgpu, Metal, CUDA | yes | The ETK's path on every backend. |
| 10 | **T3.5** no vehicles = main, bit for bit, plus the timing gate | yes | Protects rigid scale (north-star rule). |
| 11 | **T5.3, T5.4, T7.1, T7.2, T6.1** sleep, bonds, limits, clocks, ground | mixed | Remaining seams. |
| 12 | **T1.1–T1.3** JBeam field conformance, census, mass (E5 already checked all 2,767 ETK beams against an independent parse) | no | Raises import certainty to every documented field and every car. |
| 13 | **T3.2–T3.4** AVBD analytic laws | yes | Rigid correctness beyond regression. |
| 14 | **T2.2, T8.2, T4.2, T4.3** rotation invariance, Chrome hash, ETK head-on, RoR §9 | mixed | Completeness. T4.3 is opt-in. |
| 15 | **(f2)** BeamNG recordings via BeamNGpy | user's machine | Only if the user can run BeamNG (§5). |

Items 2 (T5.1), 3, 5, 6 and 7 need no GPU. A test that finds a real
defect is `#[ignore]`d with its measured value, and the fix goes to the user as a
decision.

---

## 5. What we can be certain of, and what we cannot

**Certain, once the tests above pass** (mathematical facts checked to stated bounds):
* Each beam, contact and element law is the law written in the spec, to rounding or a
  derived discretisation bound (layer 1, plus T1.x).
* No path creates linear or angular momentum (to the fixed-point quantum), and no path
  creates energy (beyond the integrator's bounded oscillation).
* Every coupling is equal and opposite, and the coupling's impulses reach the right
  bodies.
* The GPU ports compute the CPU reference: bit for bit on wgpu where the arithmetic is
  strict, within stated bounds on Metal and CUDA.
* Optimisations are lossless: the bit-exactness tests.
* The coupling either converges to the rigid answer as its window shrinks, or it does
  not. That is R1, a yes/no fact with a measured rate.
* Whether soft↔rigid and soft↔soft give the same momentum exchange (exact), and how far
  their dissipation differs at equal restitution: measured, with the cause attributed to
  the law difference F2 or not.

**Not certain today** (measured, open):
* AVBD loses 5.5 % of the momentum in a rigid-only multi-body impact (G3.0). Until T3.6
  explains or bounds it, "no path creates or destroys momentum" holds for the coupling
  (residual 0.004 N·s on the CPU, 0.0 on the GPU World box) but not for AVBD piles.

**Certain only relative to a reference:**
* RoR profile ≈ RoR's own solver on RoR content (e). That is a strong statement about
  the solver's RoR mode, and only that mode.
* Content ≈ NCAP/FMVSS spreads (g): a statement about car plus solver together.

**Not provable without BeamNG itself:**
* **BeamNG equivalence.** BeamNG's documentation leaves out the node–ground response,
  the plastic law's details and the contact law. Rubble's choices there are labelled
  Choices (`jbeam-semantics-spec.md` §3, §28, §37).
* A JBeam car can therefore behave in Rubble exactly as its spec tests require and
  still differ from BeamNG. Only (f2) recordings of the same file in BeamNG can bound
  that difference, and only for the recorded scenes.
* Until then the honest claim is: "implements BeamNG's *documented* semantics; matches
  RoR where RoR applies; conserves what physics conserves".

**Not provable by tests at all:** that the content is a good model of a real car. That
is content work, judged against (g) and then (h).
