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

## Open

- **The load path on the fractured tower** (vibe-city `fracturedTower`, alone, in the block and
  in the page's volley): "a load-path block was not positive definite" (fracture.slang
  `fr_lp_block`, error bit 64) once a chunk is left hanging by one of two of its 10,820 bonds:
  each of those bonds' single-bond block factors in f64 and not in f32 (stress-rubble example
  `load_path_blocks`). The lever arm's coupling (|r|² × axial stiffness) swamps the rotational
  stiffness, and the Cholesky's Schur complement cancels in f32. The lab's world on the same
  shot does not leave such a chunk; World's run does. World fails the step with the reason
  (the lab's world holds the chunk still and reports it). The lossless fix: factor each block
  about its bonds' stiffness-weighted centre (the elastic centre, where the coupling vanishes
  for one bond), the sweeps transforming back; it changes the load path's kernels and their
  host-twin parity, so it is its own change.
- **D6 on the punch-through wall** (`crush-wall` pack): the settle overflows the warm-start
  hash (`Overflow(256)`, `OVERFLOW_WARM_HASH`) at step 107. The settle's world is sized as the
  old page sized it; not yet checked on main.
- **The cuboid house settling**: the lab's world breaks 2 bonds at frame 2, `World` none (the two
  rigid engines differ in contact count from step 1). Present on main before this work; the
  strike's breaks agree.
- **The default ball on strong scenes**: the glass tower under D6 and threshold breaks nothing to
  one click at its top (its other models do); the tower block under D6 at the pack's strength
  breaks little (its suggested settings: 0.2×, 16× balls).
- **Non-convex hulls**: stress-rubble's cooker replaces nine of the fractured tower's hulls with
  their boxes (with a warning); pre-existing, a candidate for re-export from Shardwright.
