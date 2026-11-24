# Host tests on the expected-failure list

`engine/tests-expected-failures.txt` lists the host tests (CI job "Engine host tests (no
GPU)") that fail today. Every one still runs on every CI run with its own bar unchanged;
the job passes only when the failures are exactly listed ones
(`.github/scripts/expected-failures.py`). A new failure, a listed test that starts
passing, or a listed test that no longer runs all fail the job.

None of these fail on `main`. They came into PR #109 with the generated-car work merged
from `feat/sedan-body` (2026-11-13/14). Status as of 2026-11-15 (the attempt to fix them
for real is time-boxed; this file is updated with what it finds).

| Test | Since | What fails | Cause, as found | Plan |
|---|---|---|---|---|
| `sedan_jello::kerb_wobble_is_at_the_etk_level` | 7b9b0e58 (committed failing as a target) | Ripple 0.20/0.29/0.37/0.35/0.27/0.53 mm against 1.25 × the ETK's 0.09/0.15/0.23/0.21/0.09/0.18 | Measured 2026-11-15 (fix/kerb-wobble 1f76c5b1, lessons §15): the hub is rigid with the body. The definition `strut_<corner> vals(0.0, …)` copies the ETK's BOUNDED travel limit (spring 0), but in our `Vals` 0 means "from the node budget", so the strut is built at 3.04e6 N/m, plus 1.2e7 N/m front clamps. The kerb's raycast step lands on the towers (sh_a6 0.56 mm against the ETK's 0.14–0.16). Measurement switches (off by default): both coilovers + a guided front knuckle give 0.12/0.15/0.20/0.18/0.14/0.36 mm, settle 0/0/0/0/0/0.9 s (bar 0.11/0.19/0.29/0.26/0.11/0.23). Rear only: door and trunk reach the bar; front softened without guidance is 3× worse | Design change (needs the owner): struts as coilovers (25–40 kN/m spring, damper, preload from the corner load, BOUNDED limits); the front knuckle guided by a defined upper link (or slidenodes in the import); then the bumper covers' ring (0.36 vs 0.23 mm) |
| `sedan_jello::kerb_wobble_dies_as_fast_as_the_etks` | 7b9b0e58 | Settle 0.3/0.7/0.7/0.7/0.3/1.0 s against 0/0/0.3/0.6/0/0 (+0.1) | Same as above | Same |
| `showcase_car::o_side_pole_matches_the_nhtsa_sedans` | 31c20eb2 | Mean crush 295 mm (n=8) against 297–445 | Not statistics: the same perturbation box Halton-spread gives n=8 278 ± 25, n=64 289 ± 7.2, n=256 286 ± 3.5 mm (≈0.5 s a run; 256 runs ≈ 50 s on 8 threads), so the car crushes 10–15 mm too little at the pole. No dominant member: plastic work at the pole is spread over door_f_l's outer panel group (strength 12.5 kN, deform 1.95 kN: ~2.9 kJ, 36 beams), body structure groups (0.7–1.7 kJ each) and a few door beams (out of 57 kJ). Sensitivity (n=8): `DOOR_FRAME` 5 → 2 moves the mean 295 → 294 mm (sd 80 → 40); `PILLAR_B` 3 → 1.5 → 1 changes nothing at all (bit-identical: no B-pillar beam reaches its threshold in this scene). (An earlier "bit-identical" door sweep was a zsh word-splitting slip: the values never reached the build; they do.) | Find which beams carry the pole load elastically (the mean barely depends on the yield of the door frame or pillar): next probe is the door outer panel and the sill/rocker section at the pole line, by beam force share at peak crush. A powered ensemble (n ≥ 64) for the test once the mean moves, since n=8 has SE 25–28 mm |
| `showcase_car::contacts_do_no_positive_work` | 770875dd | +0.076 J in one tenth of the T-bone (80 km/h) | Real energy bug (proved on the CPU reference): the vehicle contact work guard predicts the end velocity as v + h·F/m, but the next step's centred-damping correction moves it by h·c/m (also shared nodes, the residual pass, the fixed-point sums). Striker node 947 stuck on the struck door's triangle (151, 171, 173) at 2.5 kN: predicted work −0.1…−0.2 mJ a step, realised +0.5…+1 mJ. The unpredicted velocity change is exactly 0 with centred damping off; mean 1.4 mm/s (objective form), 28 mm/s (old form). **CPU fix done** (6da4a1f6): `ExplicitBeamsWorld::guard_final`, a work guard on the final velocities after every vehicle's node update and correction; **off by default** (`final_guard`, `FINAL_GUARD=1`) until the kernels carry it. With it on this test passes (+0.0000 J) and nothing else changes pass/fail; `tests/vehicle_contact_work.rs` is the unit test (fails +1.0e-5 J with it off). GPU verification pending (the kernels are not written; the Mac GPU was wedged from 06:41). | The kernels (plan below), then default on and remove this line |
| `showcase_skin_integrity::wall_at_90_kmh_three_times` | a405eb78 / e4c5ec3c | 4 shards up to 0.40 m (hit 1, 1.5 s) | Measured 2026-11-15 (fix/skin-shards 00c3b864, skin-shards.md "The remaining shards are not folds over intact beams"): every shard's nodes are still connected, but three mechanisms: (C) real stretch along intact soft beams (21–42 kN/m, 1.5–2.7× with no yield or break; 943 of the body's 4,068 stiff beams neither yield nor break before 40 % strain): wall 90, random, side 60 rear door, t-bone struck car; (A, B) the vertex grouping in `eval_breaks` rejoins nodes that have come apart through a third node of ~0 weight: grab bootlid, t-bone striker. A prototype that refuses such merges takes grab bootlid 288 → 30 and the striker 184 → 37, but cuts 26 triangles with no broken beam directly under them | (C) needs a beam-physics decision (sheet beams that stretch 2× without yielding are not steel); (A/B) the grouping fix also needs the renderer's `blendVertex` mirror and a rule for the 26 cuts. A denser lattice or helper nodes would not remove these |
| `showcase_skin_integrity::car_into_the_rear_door_at_60_kmh` | same | 14 shards (struck, f 15); also a 114-triangle glass speck (glass_side_q_l) behind it | Class C (above); the speck not investigated | Same |
| `showcase_skin_integrity::t_bone_118_into_61_kmh` | 31c20eb2 | Shards struck 115/106/79, striker 184/171/173 | Struck: class C; striker: class A | Same |
| `showcase_skin_integrity::random_impacts` | same | 11 shards (random 0, 103 km/h at 3°) | Class C (underbody, arch liner) | Same |
| `showcase_skin_integrity::grab_rear_and_roof` | same | 288 shards (bootlid 2 m) | Classes A and B | Same |
| `gen_sedan_contact::head_on_cars_are_not_fused` | 2026-11-24 (written ahead of the fix) | Two generated sedans after a 60 + 60 km/h head-on, pushed apart (3 m/s each) from one state: part 0.21 m in 1 s with contact against 6.11 m without (need 80 %) | Vehicle contact holds them: nodes inside the other car held there by anti-clip (known-bugs S4) | Edge-against-triangle vehicle contact, close the skin, then the outward side (S4) |
| `gen_sedan_contact::head_on_no_node_inside_the_other_car` | same | 29 / 35 nodes past the band inside the other car at the worst sample to 0.6 s, deepest 189 / 268 mm, against 0 (edge contact on: 8 / 12) | Most enter through gaps in the collision skin | Same |
| `gen_sedan_contact::head_on_no_edge_through_the_other_car` | same | 78 / 88 skin edges through the other car's triangles at the worst sample against 0 (edge contact on: 46 / 48) | Node-only contact: edges pass between nodes | Edge-against-triangle contact |
| `gen_sedan_contact::pole_own_panels_stay_apart` | same | 52 new own-skin crossings at a 64 km/h pole (front left door through the left fender), against 0 | Self contact is node against triangle | Edge-against-triangle self contact |
| `gen_sedan_contact::collision_skin_does_not_cross_itself_at_rest` | same | 89 own-skin crossings at spawn against 0 | Authoring: the rigger's skin passes through itself (rear bumper, sills, fender corners) | Fix in the rigger |
| `gen_sedan_contact::fast_head_on_no_overlap` | 2026-11-24 (ahead of the fix) | 100 + 100 km/h head on, worst sample: 26 / 23 nodes inside the other car (243 / 201 mm), 96 / 90 edges through, against 0 | Fixed 25 mm band against 28 mm a step; node-only and memory side rules (`docs/contact-design.md`) | The contact design |
| `gen_sedan_contact::unequal_head_on_no_overlap` | same | 100 into 10 km/h: 21 / 26 nodes (133 / 228 mm), 70 / 67 edges | Same | Same |
| `gen_sedan_contact::offset_head_on_no_overlap` | same | 60 + 60 km/h at 0.8 m offset: 17 / 20 nodes (347 / 286 mm), 58 / 65 edges | Same | Same |

Fixed and removed from the list:
- `gen3_rig::rest_fidelity_at_spawn_and_parked` (the test's allowance now counts every node a vertex is bound to; 1.057 → 0.542 mm against 1 mm).
- `sedan_jello::the_parked_car_stands_still` (bd9af021: not a sustained oscillation but the passenger seat's rocking mode dying at ζ ≈ 0.008: defined beams bypassed the rigger's ζ 0.04 rule; a defined spring's damping is now at least that; 0.0708 → 0.0366 mm against 0.05).
- `showcase_skin_integrity::grab_doors` (267a44c1: the 18-triangle speck was intact body skin left on the body by the part split's 8 cm margin on door_r_r's rear edge; margin islands now go to the part they lie on).

Working notes and probes: `engine/out/patches/item6-*` and the agent reports quoted in
PR #109's description.

## Final contact work guard: metrics with it on (`FINAL_GUARD=1`, whole CI host suite, CPU)

Pass/fail: only `contacts_do_no_positive_work` changes (fails → passes). Printed metrics that
move (off → on); every other printed number is identical:

| Scene | Metric | Off | On |
|---|---|---|---|
| T-bone 80 km/h (`contacts_do_no_positive_work`) | vehicle contact work, most over a tenth | +0.0763 J | +0.0000 J |
| | self-contact work, most over a tenth | −12.24 J | −4.98 J |
| | position steps' stored energy, most over a tenth | +23,965 J | +37,169 J |
| | together after 1 s | −499.9 J | −209.8 J |
| sedan1 HeadOn (crash ledgers) | closure | 0.0105 | 0.0076 |
| | rear | 0.0104 | 0.0094 |
| `sedan_parts_collide` | door reach past the seat's face; seat moved inboard | −55 mm; 198 mm | −70 mm; 195 mm |
| `sedan_energy` offset −1.2 (two cars) | fastest node at the end | 1.5 / 1.1 m/s | 0.1 / 0.0 m/s |

Unchanged: walls, poles, the n/o/p/q scenes and every skin-integrity scene (single vehicles,
or the RoR contact rule, which stages no dissipative two-body hits). Logs:
`engine/out/suite-guard{0,1}-20261115-061351.log` in the fix worktree.

Self contact (`friction_guard_pass`) has the same structure (hits guarded at the end of a
step, integrated with the correction after): no self-contact tenth is positive in these
runs; the same guard on a vehicle's own staged self hits between `step_nodes` and
`step_rest` would close it.

### Kernel plan (not done; the GPU must match the CPU bit for bit before the default flips)

1. **Staging.** Where the vehicle contact pass writes a dissipative two-body hit's total
   (the CPU's `apply_vehicle_hit` for an `on` hit), also append a record (a, b, h, tri[3],
   w[3], n, f) to a bounded per-world list (capacity = the existing hit capacity; overflow
   is an error flag, never a silent drop). Clear it at the pass's start.
2. **v₀.** The velocities before the node phase for those nodes: with the objective form the
   kernel already keeps them (`centredDv + 2·totalNodes`); otherwise save them in the node
   phase for nodes with a staged hit.
3. **The pass.** After `centredPhase` (all three call sites: the fused step, the wide path's
   `explicit_beams_wide_centred_nodes`, the persistent step) and before the beam phase, a
   world-level pass over the list **in list order, one lane** (hits share nodes; the CPU is
   sequential, so the GPU must be for bit parity; lists are tens to hundreds long). On the
   wide path this is a new dispatch between the centred nodes and the beams; on the fused and
   persistent paths both vehicles' node phases must be done first, so it needs the
   cross-workgroup sync those paths already use for vehicle contact, or a split of the step
   when the list is non-empty. Keep it its own function behind an early return (list empty)
   for the Metal fast-math rule, with a capped loop and uniform control flow round barriers.
4. **The arithmetic, in this order (f32, as the CPU):** q_b = toFixed(f)/32 per component;
   q_k = toFixed(−f·w_k)/32; work = q_b·(v₀_h + v_h) then + q_k·(v₀_k + v_k) for k = 0, 1, 2;
   lose = q_b·q_b·im_h then + q_k·q_k·im_k (im 0 for immovable or retired); skip if work ≤ 0
   or lose ≤ 0; r = min(work / (h·lose), 1); each node v += q·(−r·h·im), x += that·h.
5. **Tests.** A GPU parity case of `tests/vehicle_contact_work.rs`'s scene with the guard on
   (CPU == GPU bit for bit), the existing contact suites with it off (unchanged) and on; the
   frozen `_ref`/`_baseline` kernels untouched (they have no final guard: compare them only
   with it off). Then flip the default and remove `contacts_do_no_positive_work` from the list.

