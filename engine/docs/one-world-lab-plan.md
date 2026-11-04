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

## Principles: one library, every host

Rubble is one library: GPU work in Slang, host work in Rust. Every front end is a thin binding
over the same Rust API:
- the web, through wasm and a little JavaScript;
- Unreal or any C engine, through the C ABI;
- native Rust demos, directly.

What works in one host then works in the others, with the same capabilities and results. For
this plan that means:

1. **Every capability the lab shows is a Rust API in `engine-world`** (with `engine-fracture`
   for scene content). That covers placing a scene, the bond models and their settings,
   launching and scheduling actions, and stats and GPU timing, with clear errors for
   combinations that cannot work. Each is reachable unchanged through wasm and the C ABI.
2. **JavaScript only wires the page:** DOM controls to Rust calls, the canvas, and drawing what
   Rust reports. It makes no physics decision and holds no scene, action or bond-model logic.
   The same goes for the C layer.
3. **No second implementation.** A capability exists once, in Rust or Slang. The stress world's
   separate host, renderer and HUD go away. They don't get ported to JavaScript.
4. **Parity is tested across hosts.** The same scenario, run from native Rust, from C and from
   the web, gives the same result within a documented tolerance.

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

### 2. Scenarios and actions in Rust (engine-world, engine-fracture)

Today the scene content and shot plans are already Rust: `engine-fracture` `towers.rs`,
`plan.rs`, `samples.rs`, reached through `wasm.mjs`. But `main.mjs` (584 lines) still decides
real behaviour:
- which bond model a scene gets, and the scene presets;
- when each shot of a volley or topple fires (`startShots`);
- how the ball scales apply to each host;
- which branches run per model;
- the HUD's timing arithmetic.

None of that can be reached from C or a native demo. It moves to Rust:

- **A scenario API on `World`:** place a lab scene (GLB level and layout, cuboid layout,
  pack, tower block with its city, glass tower) under a chosen bond model and settings, as one
  call. Invalid combinations return a clear error.
- **An action schedule on `World`:** queue an action plan (click shot, row shot, volley,
  topple, meteor, authored shots), each with its launch step. `World` launches each one at its
  step and registers the projectiles for strike prediction. The ball mass, speed, radius and
  CCD settings apply in one place.
- **Stats and timing from `World`:**
  - the counters for every model (bodies, bonds broken, crushed, awake, contacts);
  - GPU time per physics step (the bound markers);
  - the derived figures: maximum steps per second, and the share of each second at real time.

  The page's own draw timing and display refresh stay the page's, because drawing is the host's.
- **Bindings:** the same calls in `engine-world-wasm` and the C ABI (`rubble.h`), with a C
  example that places a scene, runs an action plan and reads the stats.

### 3. The lab is a thin page over that API (engine-webgpu/fracture)

- **One load path.** The page calls the scenario API for every bond model. `loadStress`,
  `loadThreshold` and the `stressOn`/`thresholdOn`/`towersBlock` branches go, and so do
  `#stress-view` and `stress-mode.mjs`.
- **Scenes, each placeable under every bond model** (built by the scenario API, section 2):
  - GLB levels (every copy of a layout, not one);
  - the cuboid house and its layouts;
  - scene packs and the reference rigs;
  - the glass tower (a generated pack);
  - the 2×2 tower block: four pack copies plus the city's cars and pedestrians as plain rigid
    bodies, the bystander zones, and the meteor.
- **Actions:** each button queues its plan on `World`'s action schedule (section 2): click,
  row shot, volley, topple, meteor, authored shots. The page holds no timing or projectile logic.
- **One renderer, camera and HUD:** `render.mjs`, `lab-sim.mjs`. Every number comes from
  `World`'s stats and timing; the page only formats them.
- **Target size:** `main.mjs` shrinks to control wiring and display.
- **No scene-driven presets.** Selecting a scene never changes the bond model or the settings.
  A scene's suggested settings, such as the tower block's fast CCD volley, become an explicit
  "use the scene's suggested settings" action.

### 4. Settings are the same for every bond model

- **Physics rate:** one explicit value, shown in the panel. "Auto" goes; no bond model picks its
  own rate.
- **Strike prediction and in-step contact fracture:** a switch for every stress model, with the
  same meaning in each.
- **Crushing:** for every model that has chunk materials.
- **Directed debris:** for every bond model.
- **Settings live in Rust:** each one is a `World` option with the same name and meaning in
  wasm and in the C ABI.
- **GPU timing:** `World`'s `gpu_marks.rs` binds the buffers each segment writes, as
  `FrameTimer` does, so that Chrome orders its marks. The HUD shows GPU physics per tick (from
  `World`), GPU draw and the display's refresh (from the page), and the uncapped FPS, for every
  model.

### 5. Checks

- **Native:** for each stress model, a `World` destructible against `ThresholdWorld` on the
  same scene and strike. They must agree on:
  - bodies after the split;
  - broken bonds, as a set;
  - `blast_gpu`'s forces, against the f32 reference.

  Native wgpu worlds are not bit-reproducible run to run, so these compare a
  documented tolerance or the same GPU state, not bitwise lockstep.
- **Across hosts:** one scenario, the tower block under `blast_gpu` with a volley, run three
  ways:
  - a native Rust test;
  - the C example through `rubble.h`;
  - the web page.

  All three must report the same bodies, bonds broken and crushed chunks, within the native
  tolerance. This is the check that what works in one host works in the others.
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
| 3 | The scenario API and action schedule on `World` (scene placement, action plans, projectile settings, stats and timing) in Rust, wasm and the C ABI, with a C example | A native test and the C example place the tower block under each bond model, run the volley and report the same stats | 2 days |
| 4 | The lab as a thin page over it: every scene and action under every bond model, one renderer and HUD; the stress host and the page's mode branches removed | Every scene loads under every bond model in Chrome; volley, topple and meteor work under `blast_gpu`; `main.mjs` holds no physics, scene or action logic | 2 days |
| 5 | One set of settings (physics rate, CCD, prediction, contact fracture, crush, directed debris) as `World` options; `gpu_marks` bound; scene presets made explicit | The HUD is the same for every model; the timing matches throughput in Chrome | 1 day |
| 6 | The matrix check, the cross-host check, a browser check on Vercel, docs (fracture README, `public-api.md`) | The matrix and the cross-host check pass except for listed, explained errors | 1 day |

About a week and a half in all. Each phase is committed and verified on its own.

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
