# One World for the fracture lab

**Goal:** every fracture lab run is one `World` (rubble's product engine, the one Unreal uses
through the C library) on one GPU device, with one renderer and one HUD. What you see is set by
four independent choices:

| Choice | Examples |
|---|---|
| **Scene** | cuboid house, suburbs, GLB levels, scene packs, the 2×2 tower block with its city, glass tower |
| **Actions** | click to shoot, row shot, volley, topple, meteor, a pack's authored shots, ball mass/speed/radius |
| **Bond model** | none, breakable D6 joints, simple threshold, load path, local impact, vibe-land's GPU stress solver (`blast_gpu`) |
| **Settings** | physics rate, CCD, strike prediction, in-step contact fracture, crushing, bond strength |

Any scene runs with any action, bond model and setting. A combination that cannot work fails with
a clear error naming why; it is never hidden, changed or substituted.

Rubble is one library: GPU work in Slang, host work in Rust. The web (wasm), Unreal or any C
engine (the C ABI) and native demos are thin bindings over the same Rust API, so what works in
one host works in the others.

## Where it stands

| Phase | What | State |
|---|---|---|
| 1 | One profile set-up (stress-rubble `ProfileSetup`, `apply_profile`) for `ThresholdWorld` and `World`; `StrengthModel::{LoadPath, ImpactLocal, BlastGpu}` and `FractureOptions`; C ABI values; packs imported per profile | Done. `tests/stress_models.rs`: every model, `World` against `ThresholdWorld`, house-1story and the cuboid house |
| 2 | Several destructibles under every stress model (one shared fracture, laid out again at each placement for the models `GpuFracture::append` cannot carry) | Done. `tests/stress_block.rs`: house-1story's 2×2 block under all four; the fractured tower under threshold, impact_local, blast_gpu |
| 3 | The lab as a Rust API: `rubble::lab` (`LabPlan`, `Lab`, `LabAction`, `LabStats`), the C ABI (`rbCreateLab`, `rbLab_Queue`, `rbLab_Step`, `rbLab_GetStats`), `examples/lab.c` | Done. `tests/lab_api.rs`; `lab.c` runs the tower block |
| 4 | The lab page over it: one load path (`createLab`), every scene under every bond model, actions queued on World; `blast_parity` the one model on its own engine | Done. `lab-matrix-check.mjs` |
| 5 | One set of settings (explicit physics rate, 60 Hz default; the fracture switches for every stress model; crushing; directed debris, a split's pieces included); the whole tick's GPU time (`World::tick_gpu_ms`, marker passes over the buffers it writes) | Done |
| 6 | The matrix check (`lab-matrix-check.mjs`), the cross-host check (`lab-cross-host.mjs`), docs (fracture README) | Scripts in place; results below |

### Decisions taken (recommended in the plan)

1. `StrengthModel` and its C ABI gained the stress models additively; `Threshold` is unchanged.
2. `blast_parity` (on `GpuWorld`, which waits on the GPU every frame) stays on the stress world,
   labelled as a separate engine, until it is ported.
3. One physics rate for every model: 60 Hz by default (`lab::DEFAULT_PHYSICS_HZ`), 120 Hz
   selectable.

### Found and fixed on the way

- The in-step contact fracture's splits were lost to a host reading events: the frame cleared the
  event lists after the contact fracture had written them (`GpuFracture::encode_contact_fracture`
  now opens the step's lists).
- Breaks by the load path, the analytic strike, both contact-fracture verdicts, crushing and
  Blast were not recorded in the broken-bond event list (only the threshold rule's were).
- `World` skipped the fracture frame while nothing was awake, which left the load path's and
  Blast's warm-started solves at their first, unconverged answer; it now steps them, as the lab's
  world does.
- `World`'s fracture errors name every error bit.
- The page's device holder raced: a scene chosen before the page's first load finished made
  a second GPU device, and the renderer drew on one while the world ran on the other
  (`lab-world.mjs` now shares the one promise).

## Results (2026-11-04, M3 Max, Chrome headless WebGPU)

**Native** (`cargo test --release -p engine-world --test stress_models --test stress_block
--test lab_api`): every stress model agrees with the lab's fracture world on house-1story and
the cuboid house (the strike's first break exactly; broken sets' Jaccard 0.82–1.00; bodies
within 13%); `blast_gpu`'s forces are the f32 reference's frame for frame; 2×2 blocks break as
one copy does.

**Matrix** (`lab-matrix-check.mjs`): the cuboid house, the arch, the veneer house, the 7×7
cuboid grid (35,035 bodies), the punch-through wall, the glass tower and the fractured-tower
block (14,844 pieces, its city), under none, D6, threshold, load path, local impact and
`blast_gpu`, with click, row, volley, topple and authored shots: every cell steps, fires its
action and draws with no WebGPU error, except those listed under Open. GPU time per step
1–16 ms.

**Cross-host** (`lab-cross-host.mjs`): the fractured-tower block under `blast_gpu`, the
volley, 480 steps:

| host | shots | chunks | bonds | bodies | broken |
|---|---|---|---|---|---|
| native Rust | 11 | 14,844 | 43,280 | 9 | 112 |
| C (`lab.c`) | 11 | 14,844 | 43,280 | 9 | 112 |
| web | 11 | 14,844 | 43,280 | 10 | 121 |

## Regressions this change introduced, fixed

- The lab's `?broadphase=full` (an A/B check) was dropped by the new load path; restored
  (`WebWorld.createForLab`'s `full_broadphase`).
- The View menu's **Bodies** coloured each chunk under the stress models now drawn from World's
  poses (they coloured by rigid body on the old stress host); the pose records carry each
  chunk's body slot (`Lab::poses` word 3, `fracture_draw.slang` `mode.w`).
- `fr_il_run` dispatched past WebGPU's limit on World's larger shared fractures; dispatched in
  rows.

Every suite that runs locally passes but for the failures below, each of which fails the same
way on main (4c7d54cf): engine-world, engine-ffi, engine-world-wasm, stress-gpu, stress-rubble,
the JS tests, `build-ffi.sh` (with `lab.c`).

## Pre-existing on main (out of this change's scope)

- **The load path on the fractured tower** (vibe-city `fracturedTower`, alone, in the block and
  in the page's volley): "a load-path block was not positive definite" (fracture.slang
  `fr_lp_block`, error bit 64) once a chunk is left hanging by one of two of its 10,820 bonds:
  each of those bonds' single-bond block factors in f64 and not in f32 (stress-rubble example
  `load_path_blocks`). The lever arm's coupling swamps the rotational stiffness and the
  Cholesky's Schur complement cancels in f32. The kernel is unchanged here (the plan named it as
  a known failure); World fails the step with the reason where the lab's world held the chunk
  still. The lossless fix: factor each block about its bonds' elastic centre.
- **D6 on the punch-through wall** (`crush-wall` pack): the settle overflows the warm-start hash
  (`OVERFLOW_WARM_HASH`) at step 88–107, at 60 and at 120 Hz; `preload_destructible_asset` on main
  does the same.
- **`engine-world` tests/threshold.rs `cuboid_house_breaks_and_splits_as_in_the_lab`**: the lab's
  world breaks 2 bonds settling at frame 2, `World` none; same numbers on main.
- **`engine-ffi` tests/soft_car_visual.rs `a_wall_crash_reports_strain_tears_and_break_points`**
  and **`engine-world-wasm` tests/broadside.rs `the_raze`** (step 991, `Overflow(8)`): fail on main.
- **The glass tower under D6 and threshold** breaks nothing to one click of the default ball, at
  60 and at 120 Hz (those models' code paths are the old ones); the matrix reports it.
- **Non-convex hulls**: stress-rubble's cooker replaces nine of the fractured tower's hulls with
  their boxes (with a warning).
