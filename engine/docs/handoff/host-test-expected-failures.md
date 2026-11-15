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
| `sedan_jello::the_parked_car_stands_still` | 7b9b0e58 (was 1.08 mm, now 0.07) | 0.07 mm against 0.05 | A sustained ~12 Hz ±0.04 mm oscillation at the passenger seat-back top; seat feet move ~20 µm the same way; not seat collision | Find the energy source (wheel/raycast or solver level) |
| `showcase_car::o_side_pole_matches_the_nhtsa_sedans` | 31c20eb2 | Mean crush 295 mm (n=8) against 297–445 | Not statistics: n=256 gives 286 ± 3.5 mm. The side structure at the pole is too stiff against the NHTSA p10 | Soften door / B-pillar / sill at the pole, with justification |
| `showcase_car::contacts_do_no_positive_work` | 770875dd | +0.076 J in one tenth of the T-bone | Real energy bug: the vehicle contact work guard predicts v + h·F/m but the centred-damping correction later in the step changes v; a stuck contact then does positive work (both damping forms have the hole) | Run the guard after the correction as a velocity-level clamp (CPU reference and kernels together) |
| `showcase_skin_integrity::wall_at_90_kmh_three_times` | a405eb78 / e4c5ec3c | Skin shards (10) | "Cut only where beams broke" leaves skin pieces whose carriers separate without a broken beam on a sparse lattice | skin-shards.md "Open": denser lattice or skin helper nodes at the underbody, rear door and boot panel |
| `showcase_skin_integrity::car_into_the_rear_door_at_60_kmh` | same | Shards (20) | Same | Same |
| `showcase_skin_integrity::t_bone_118_into_61_kmh` | 31c20eb2 | Shards (115–352 a car) | Same | Same |
| `showcase_skin_integrity::random_impacts` | same | Shards | Same | Same |
| `showcase_skin_integrity::grab_rear_and_roof` | same | Shards (bootlid 354) | Same | Same |
| `showcase_skin_integrity::grab_doors` | 31c20eb2 | An 18-triangle scrap of body skin cut off with the rear right door pulled 10 m; its 8 carrier nodes still have live stiff beams | Present with two-body contact off too | Decide whether a torn-hinge scrap is physical; else fix the cut |

Fixed and removed from the list: `gen3_rig::rest_fidelity_at_spawn_and_parked` (the test's
allowance now counts every node a vertex is bound to; 0.542 mm against 1 mm).

Working notes and probes: `engine/out/patches/item6-*` and the agent reports quoted in
PR #109's description.
