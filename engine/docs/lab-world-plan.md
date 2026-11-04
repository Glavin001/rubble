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

### Open

- **The load path on the fractured tower** (vibe-city `fracturedTower`): the struck tower reports
  "a load-path block was not positive definite" (fracture.slang `fr_lp_block`) at step 89, alone
  as in the block. The lab's own fracture world on the same shot does not reach it, and every
  chunk's block in that run factors in f64 and in f32 (stress-rubble example
  `load_path_blocks`); the cause is not found yet.
- **The cuboid house settling**: the lab's world breaks 2 bonds at frame 2, `World` none (the two
  rigid engines differ in contact count from step 1). Present on main before this work; the
  strike's breaks agree.
