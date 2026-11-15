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
| `sedan_jello::kerb_wobble_is_at_the_etk_level` | 7b9b0e58 (committed failing as a target) | Ripple 0.21/0.29/0.37/0.36/0.27/0.53 mm against the ETK's 0.09/0.15/0.23/0.21/0.09/0.18 | generated-vehicles-lessons §15: the gap is in input and mounting, not panel stiffness | Body/mount authoring at ETK level; may need a body redesign |
| `sedan_jello::kerb_wobble_dies_as_fast_as_the_etks` | 7b9b0e58 | Settle 0.3/0.7/0.7/0.7/0.3/1.0 s against 0/0/0.3/0.6/0/0 | Same as above | Same |
| `showcase_car::o_side_pole_matches_the_nhtsa_sedans` | 31c20eb2 | Mean crush 295 mm (n=8) against 297–445 | Not statistics: the same perturbation box Halton-spread gives n=8 278 ± 25, n=64 289 ± 7.2, n=256 286 ± 3.5 mm (≈0.5 s a run; 256 runs ≈ 50 s on 8 threads), so the car crushes 10–15 mm too little at the pole. No dominant member: plastic work at the pole is spread over door_f_l's outer panel group (strength 12.5 kN, deform 1.95 kN: ~2.9 kJ, 36 beams), body structure groups (0.7–1.7 kJ each) and a few door beams (out of 57 kJ). Sensitivity (n=8): `DOOR_FRAME` 5 → 2 moves the mean 295 → 294 mm (sd 80 → 40); `PILLAR_B` 3 → 1.5 → 1 changes nothing at all (bit-identical: no B-pillar beam reaches its threshold in this scene). (An earlier "bit-identical" door sweep was a zsh word-splitting slip: the values never reached the build; they do.) | Find which beams carry the pole load elastically (the mean barely depends on the yield of the door frame or pillar): next probe is the door outer panel and the sill/rocker section at the pole line, by beam force share at peak crush. A powered ensemble (n ≥ 64) for the test once the mean moves, since n=8 has SE 25–28 mm |
| `showcase_car::contacts_do_no_positive_work` | 770875dd | +0.076 J in one tenth of the T-bone (80 km/h) | Real energy bug (proved on the CPU reference): the vehicle contact work guard predicts the end velocity as v + h·F/m, but the next step's centred-damping correction moves it by h·c/m (also shared nodes, the residual pass, the fixed-point sums). Striker node 947 stuck on the struck door's triangle (151, 171, 173) at 2.5 kN: predicted work −0.1…−0.2 mJ a step, realised +0.5…+1 mJ. The unpredicted velocity change is exactly 0 with centred damping off; mean 1.4 mm/s (objective form), 28 mm/s (old form). **CPU fix done** (6da4a1f6): `ExplicitBeamsWorld::guard_final`, a work guard on the final velocities after every vehicle's node update and correction; **off by default** (`final_guard`, `FINAL_GUARD=1`) until the kernels carry it. With it on this test passes (+0.0000 J) and nothing else changes pass/fail; `tests/vehicle_contact_work.rs` is the unit test (fails +1.0e-5 J with it off). | The kernels (plan below), then default on and remove this line |
| `showcase_skin_integrity::wall_at_90_kmh_three_times` | a405eb78 / e4c5ec3c | Skin shards (10) | "Cut only where beams broke" leaves skin pieces whose carriers separate without a broken beam on a sparse lattice | skin-shards.md "Open": denser lattice or skin helper nodes at the underbody, rear door and boot panel |
| `showcase_skin_integrity::car_into_the_rear_door_at_60_kmh` | same | Shards (20) | Same | Same |
| `showcase_skin_integrity::t_bone_118_into_61_kmh` | 31c20eb2 | Shards (115–352 a car) | Same | Same |
| `showcase_skin_integrity::random_impacts` | same | Shards | Same | Same |
| `showcase_skin_integrity::grab_rear_and_roof` | same | Shards (bootlid 354) | Same | Same |

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

