# One World for the fracture lab: plan

**Goal:** every fracture lab run is one `World` (rubble's product engine, the one Unreal uses
through the C library) on one GPU device, with one renderer and one HUD. What you see is set by
four independent choices:

| Choice | Examples |
|---|---|
| **Scene** | cuboid house, suburbs, GLB levels, scene packs, the 2×2 tower block with its city, glass tower |
| **Actions** | click to shoot, row shot, volley, topple, meteor, a pack's authored shots, ball mass/speed/radius |
| **Bond model** | none, breakable D6 joints, simple threshold, load path, local impact, vibe-land's GPU stress solver (`blast_gpu`), learned impact |
| **Settings** | physics rate, CCD, strike prediction, in-step contact fracture, crushing, bond strength |

Any scene runs with any action, bond model and setting. When a combination genuinely cannot work,
it fails with a clear error naming the reason. It is never hidden, silently changed or quietly
substituted. Comparisons between bond models are then like for like: the same rigid-body
engine, timestep, projectiles, renderer and timing.

## Today: two simulators behind one panel

The lab page drives two separate physics hosts and switches between them by bond model.

| | A: rubble's `World` | B: the stress world (`stress-web`) |
|---|---|---|
| Bond models | none, D6, simple threshold, learned | blast_parity, load path, local impact, `blast_gpu` |
| Code | `engine-world` via `engine-world-wasm` (`lab-world.mjs`, `lab-sim.mjs`) | `engine/stress-oracle/crates/stress-web` (`stress-mode.mjs`) |
| GPU device, canvas, renderer, camera | the page's (`#view`, `render.mjs`) | its own (`#stress-view`, `stress-rubble` `render.rs`) |
| Scenes | everything | one structure: a pack, a GLB level, or a cuboid layout |
| Actions | click, row shot, volley, topple, meteor, authored shots | click, authored shots |

How it came to be: the GPU stress solver reached the lab (2026-10-18) by embedding the stress
oracle's own browser viewer. `World` gained the same GPU fracture stage later (D1b, 2026-10-24),
but with the simple threshold rule hard-coded (`Profile::Threshold` in
`engine-world/src/threshold.rs`). Every stress model since then was added only to the viewer.

### What the split causes (`fracture/main.mjs`)

1. **Scenes.** In B, the tower block, the city (cars and pedestrians), the meteor and the glass
   tower load one structure or not at all.
2. **Actions.** Volley, topple, meteor and the row shot are `World` launches and do nothing in B.
3. **Physics rate.** "Auto" is 120 Hz on A and 60 Hz on B.
4. **Projectiles.** The CCD checkbox is ignored on B, which always sweeps. The mass, speed and
   radius scales go through two different code paths.
5. **Strike prediction** is a toggle only on B. A's threshold model always predicts, and its
   checkbox is locked.
6. **In-step contact fracture** exists only on B.
7. **Directed debris and the bystander count** exist only on A.
8. **The camera, view modes and bond-graph inspector** are each implemented twice.
9. **HUD timing differs.** B's GPU timing is correct (`stress-gpu` `FrameTimer`). A's
   `gpu_marks.rs` uses unbound marker passes, which Chrome can run concurrently with the work,
   so its "ms GPU/tick" likely reads low in Chrome.
10. **Two GPU devices.** Switching between A and B can draw with a buffer from the other device
    ("Buffer b0 … is associated with Device rubble world").
11. **Selecting a scene forced a bond model.** The tower scene switched to D6 and 576 m/s balls
    (fixed in PR #87; the general pattern is removed below).

## Target

### 1. `World` takes any stress model (engine-world, stress-rubble)

- **`StrengthModel`** gains the stress model:
  - `Threshold` keeps its meaning (the simple threshold rule) and its C ABI value;
  - new values select load path, local impact and `blast_gpu`;
  - the options move into a struct shared with the lab: strike prediction, in-step contact
    fracture sweeps, crushing.
- **One source for a profile's set-up of `GpuFracture`.** Today `ThresholdWorld::with_profile`
  (`stress-rubble/src/threshold.rs`) builds each profile's tables and `World` builds only
  threshold's. Move that set-up into one function, called by both:
  - the load model (threshold, or load path with its sweeps and per-bond tables);
  - the strike model (Hertz, or the analytic local impact);
  - in-step contact fracture;
  - the crush tables;
  - `blast_gpu`'s graph and materials, `BlastFrame`.
- **Importing.** `StressSource::from_pack` imports with the chosen profile's `import_options()`,
  not threshold's. Packs keep their own solver inputs for `blast_gpu`.
- **Several destructibles.** `World` puts every threshold destructible on one shared
  `GpuFracture`. The stress models' per-bond tables, and `blast_gpu`'s graph, must cover all of
  them, each structure's chunks and bonds at its own offset. `blast_gpu` already derives one
  graph over several structures for scenes without authoring. Authored packs need the same
  across copies: the tower block places one pack four times.
- **Events.** `World` already reads the fracture's GPU-built event log (`fracture_log`, about
  60 KB a step at most) and emits `BondsBroken` and `Split`. Every stress model reports through
  it unchanged.

### 2. The lab runs every bond model on `World` (engine-webgpu/fracture, engine-world-wasm)

- **One load path.** The scene builder produces the lab scene and its `World` placements for
  every bond model. `loadStress` and the `stressOn` branches go, and so do `#stress-view` and
  `stress-mode.mjs` in the lab.
- **Scenes, each placeable under every bond model:**
  - GLB levels (every copy of a layout, not one);
  - the cuboid house and its layouts;
  - scene packs and the reference rigs;
  - the glass tower (a generated pack);
  - the 2×2 tower block: four pack copies plus the city's cars and pedestrians as plain rigid
    bodies, the bystander zones, and the meteor.
- **Actions:**
  - every action is a `World` launch: click, row shot, volley, topple, meteor, authored shots;
  - one projectile path: mass, speed and radius scales, CCD as the checkbox says, and each
    projectile registered for strike prediction (`World::body_set_projectile`) whenever the
    bond model predicts.
- **One renderer, camera and HUD:** `render.mjs`, `lab-sim.mjs`. The stress counters (bonds
  broken, bodies, crushed) come from `World` stats.
- **No scene-driven presets.** Selecting a scene never changes the bond model or the settings.
  A scene's suggested settings, such as the tower block's fast CCD volley, become an explicit
  "use the scene's suggested settings" action.

### 3. Settings are the same for every bond model

- **Physics rate:** one explicit value, shown in the panel. "Auto" goes; no bond model picks its
  own rate.
- **Strike prediction and in-step contact fracture:** a switch for every stress model, with the
  same meaning in each.
- **Crushing:** for every model that has chunk materials.
- **Directed debris:** for every bond model.
- **GPU timing:** `World`'s `gpu_marks.rs` binds the buffers each segment writes, as
  `FrameTimer` does, so that Chrome orders its marks. The HUD shows GPU physics per tick, GPU
  draw, the display's refresh and the uncapped FPS for every model.

### 4. Checks

- **Native:** for each stress model, a `World` destructible against `ThresholdWorld` on the
  same scene and strike. They must agree on:
  - bodies after the split;
  - broken bonds, as a set;
  - `blast_gpu`'s forces, against the f32 reference.

  Native wgpu worlds are not bit-reproducible run to run, so these compare a
  documented tolerance or the same GPU state, not bitwise lockstep.
- **Headless Chrome matrix** (a new `lab-matrix-check.mjs`): every scene × bond model × action.
  Each cell asserts:
  - no WebGPU or page error;
  - frames advance;
  - the action fires;
  - a breakable bond model breaks something;
  - the HUD's GPU timing is present and plausible.

  A combination that cannot work must fail with its clear error. The check lists every
  combination that is an error.
- **Vercel preview:** the matrix's main cells checked by hand in a visible browser.

## Phases

| Phase | Work | Exit criteria | Estimate |
|---|---|---|---|
| 1 | Shared profile set-up; `StrengthModel` stress models with options; C ABI values; pack import per profile | Native `World`-vs-`ThresholdWorld` checks pass for threshold, load path, local impact and `blast_gpu` on house-1story and the cuboid house; engine-world and engine-ffi suites pass | 1–2 days |
| 2 | Several destructibles under every stress model (offsets in the load path, local impact and `blast_gpu` tables; authored packs copied) | The tower block (4 copies) stands at rest and breaks under each stress model, natively | 1 day |
| 3 | The lab on `World` alone: every scene builder, every action, one renderer and HUD; the stress host removed from the lab | Every scene loads under every bond model in Chrome; volley, topple and meteor work under `blast_gpu` | 2–3 days |
| 4 | One set of settings (physics rate, CCD, prediction, contact fracture, crush, directed debris); `gpu_marks` bound; scene presets made explicit | The HUD is the same for every model; the timing matches throughput in Chrome | 1 day |
| 5 | The matrix check, a browser check on Vercel, docs (fracture README, `public-api.md`) | The matrix is green except for listed, explained errors | 1 day |

About a week in all. Each phase is committed and verified on its own.

## Decisions needed

1. **Public API:** `StrengthModel` and its C ABI gain the stress models and their options,
   additively; `Threshold` is unchanged. Recommended.
2. **blast_parity** (the "Stress solver (GPU, blast_parity)" entry) runs on `GpuWorld`, which
   waits on the GPU every frame. It cannot move into `World` in this plan. Options:
   - **keep it on the stress world for now**, labelled in the menu as a separate engine, and
     port it later (recommended; the menu then says plainly that it is the one exception);
   - remove it from the lab until it is ported (the code stays).
3. **Default physics rate:** one value for every model. Recommended: 60 Hz, the stress models'
   authored frame, with 120 Hz selectable. D6 joints and the rigid-only models are then compared
   at the same rate as the stress models.

## Risks

- **Speed of `blast_gpu` in `World`.** The stress world never reads back. `World` reads the
  event log each step and keeps host state for the C ABI. If `World` turns out measurably
  slower, report it per model; the comparison stays like for like.
- **The tower block under the load path.** The load path already fails on one fractured tower
  ("a load-path block was not positive definite"). It must be diagnosed. A real limit becomes a
  clear error, never a hidden one.
- **Kernel capacity.** Four towers plus the city can exceed the contact ingest's limit or the
  storage-buffer limits on some devices. Capacities are explicit and fail clearly (AGENTS.md).
- **Lab regressions.** The page is rewired. The matrix check is the guard, and it lands before
  the old paths are deleted.

## Out of scope

- The head-to-head between `blast_gpu` and vibe-land's solver: next, on top of this.
- Speed work on any model.
- Unreal-side UI for the new stress models. They are available through the C ABI.
