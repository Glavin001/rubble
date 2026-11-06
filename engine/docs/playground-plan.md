# One playground for every demo: plan

**Goal:** every rubble demo is one scene in one registry, played by one page shell (web),
one viewer (native) and the C API, on one `World`. A demo is a choice of **scene**,
**actions**, **settings** and **camera/controls**, not a page of its own. The fracture lab
(`rubble::lab`, [lab-world-plan.md](lab-world-plan.md)) already works this way. This plan
extends that pattern to the rest.

Principles (unchanged from the lab):

- Every capability is a Rust API in `engine-world`. Scenes, actions and scripts are Rust,
  so C/Unreal and native get them too.
- JavaScript only wires the page: device, canvas, input events, DOM. Draw shaders are Slang.
- No second implementation of anything: one scene builder, one ball/launch, one camera,
  one HUD, one frame loop.
- A combination that can't work fails with a clear error naming why. It is never hidden
  or substituted.
- Lossless: no scene loses bodies, contacts, iterations or detail in the move. Every page
  keeps its frame time (checked before and after each phase, one GPU run at a time).

## Status (2026-11-05, branch `claude/amazing-euler-eefda7`)

| Phase | State |
|---|---|
| 1 | Done: the preset registry (`play::presets`), the shared balls (`play::balls`), Broadside and the timber yard in `engine-world`. |
| 2 | Done: `demo.html`'s built scenes are made in Rust (`engine-scenes::web`, `play::rigid`). |
| 3 | Done for vehicles, soft bodies, motion and terrain (`play::{vehicles, softbody, motion, hills}`); the city was already Rust (`rubble::city`). The street and range layouts moved to `engine_vehicle::streets`. |
| 3b | Done: one `SceneDef` over those families (`play::scene`): its world, its placement, a page's URL switches read in Rust (`from_query`, so a preset's query is its page's scene), and `Playground`. Bindings: wasm `createScene`/`placeScene`/`sceneFromQuery`/`presetScene`; C `rbCreatePlay`/`rbPlay_Step`/`rbPlay_GetStats`; native `tests/play_presets.rs` (all 44 placed presets). |
| 3c | Done for vehicles, soft bodies, motion and terrain: `play::Action` and `play::Director` (`play::director`, `play::vehicle_director`). Each scene's script (the gallery's volleys, the playground's NPCs and platform, the T-bone ram, the crusher's winch, the traffic drivers, the gauntlet's shelling, the range's volleys and scoring, the meteor run's director and judge, directed debris) runs in Rust before each step or frame, and the player's actions (fire, cannon, ram, crater, rain, guide, targets, reset) are `act` calls. Pages call `world.act()` and follow `takeEvents()`; natively `Playground::act`/`frame`. `World::begin_body_read` reads body records without waiting, as the pages' `readBodies`. The gauntlet, range and run scripts moved to `engine_vehicle::scripts`. |
| 3d | The city (`play::city_scene`: `city.html`'s switches, every house placed, shell and scenario actions) and the range scenes (Broadside, the timber yard: `stage::RangeSceneDef`) run through `Playground` (`Playground::with_assets` for the GLB; native hosts read the repository's). The meteor impact (`play::impact`: the prediction, the plain, the terrain and the world from `impact.html`'s switches; its script launches, strikes after the approach, bakes and drives the truck out) is in `SceneDef`; checked in Chrome: heightfield, bodies, terrain settings and prediction byte for byte, the strike's run (particles identical), the crater and the truck (within 0.2 m: the truck now comes 1.5 s of sim time after the bake, not 1.5 s of wall time). The page records its strata start positions a step after the strike now (render only). The meteor run (`play::showcase`, `Family::Showcase`: every scene, drive/drive1/pass/bts/play) likewise: its terrain, dunes or basin, track, car and world, its strike schedules, the car's autopilot and the play scene's director in Rust (`Action::Strike`, `Action::Drive`); checked in Chrome: setup byte for byte for four scenes, the same strikes (times, points, regions) for drive, drive1 and play (seed 7); the car within ~1 m (its pose is read a step earlier now). Not yet: the lab (its own Rust API, `rubble::lab`; folded in with its page in phase 4). |
| 4–6 | Not started. |

Every move was checked in Chrome against the page before it: the body records before the
first frame (and every soft body's entry, the heightfield) byte for byte. Demo 12
configurations and the Morton slot order, vehicle 16, motion 3, terrain 3, softbody 27.

How this differs from the target below, so far:

- `SceneDef` is an enum of the families' own definitions, not freely composed parts
  (rigid content + vehicles + ground + characters in one scene). Composition comes with
  the shell (phase 4), when one page needs it.
- No `Action` type yet: firing, ramming, shelling, meteors and crash runs are still each
  page's script over what placement returns.
- Not in `SceneDef` yet: the lab, impact and showcase (the range and the city run
  through `Playground`, each on its own placement, not composed parts); the explicit solver's own scene (the page builds the sim, Rust places its shapes);
  imported vehicles (the page adds the file to the scene).
- `World::settle_uploads` (new): bulk placement hands its uploads over in batches;
  natively, one write per body piled up Metal staging buffers (gables506k placed in over
  10 minutes, now 7.7 s).

How the directors' results compare with the pages before them (outcome checks in Chrome,
the pages' own scripts):

- Same: the CCD gallery, bullets and character checks (positions to 1 mm), the NPC
  circles, the crusher, the meteor range (static and moving: identical hits, turns and
  scores), the meteor run's take (identical score, near misses and hits), cruise traffic
  (the player's car identical), the cannons.
- Different, and why: the pages' frame-stepped hooks (`__step`, the takes) skipped some
  of what the interactive page did. The T-bone ram fired before the page had read the
  car (its default spot; the interactive page aimed at the car, as the director now
  always does: 14 broken beams, as the old page gives once it has read the car), and the
  gauntlet take never latched the car's rest height for its zones (rAF stubbed).
  The derby (chase) traffic is chaotic and differs run to run.

Known on main, not from this work: R (reset) in `motion.html`'s ccd and character scenes
fails ("forcing a body to sleep" is unsupported).

## Today (2026-11-05, main after PR #102)

The physics is already one engine. Every page creates a `World` through
`engine-world-wasm` (`rubble-world.mjs`); the only exception is the lab's `blast_parity`
(stress world, labelled as a separate engine). The duplication sits around it.

### Pages

| Page | Lines | Scenes (`?scene=`) | Own page parts |
|---|---|---|---|
| `demo.html` | 1,135 | wall, ring, gables, pyramid, pile, columns, ring110k, gables506k, joints, ragdolls, ragdolls10k, hingechains10k, motorfield10k | scenes built in JS (`demo-config.mjs`, `demo-scene.mjs`, `scene-builders.mjs`), solver A/B flags, instanced renderer, ring regression |
| `vehicle.html` | 1,684 | bridge, buggies, carry, destruction, gauntlet, meteors, playground, run, seesaw, sidewall, slab, tanklab, traffic, vibeland | driving input, chase camera, `buggy-render.mjs`, routes/AI flags |
| `softbody.html` | 1,877 | arena, crates, crusher, drop, headon, import, limbo, pole, ramp, rollover, sandwich, sidepole, tbone, traffic, wall, yard | RoR/JBeam import, `car-render.mjs`, skin, crash cameras, 30 query flags |
| `showcase.html` | 1,845 | meteor run (profile, rule, car) | MPM ground WGSL, touch controls, buggy renderer |
| `impact.html` | 1,154 | meteor on deformable ground (surface, mode, bake) | MPM ground WGSL (`terrain-render.mjs`) |
| `motion.html` | 712 | bullets, ccd, character | character input, autofire |
| `city.html` | 632 | large-maps city (scenario, budget, pool…) | streaming HUD, car |
| `terrain.html` | 489 | heightfield (n) | heightfield renderer |
| `broadside.html`, `timber.html` | 48 each | Broadside, the timber yard | `fracture/range-page.mjs` over `play.rs`/`range.rs` |
| `fracture/` | 477 (`main.mjs`) | every lab scene × bond model × action | already one World (`rubble::lab`) |

Diagnostics that stay as tools: `dispatch-diagnostic.html`, `chrome_wgsl_load.html`,
`fracture/gpu-limits.html`, `fracture/headless.html`, `tests/reference-gables.html`.

### Duplication

1. **Several Rust "scene plus actions" APIs.**
   - `rubble::lab` covers shots, volley, topple, meteor and the shot schedule.
   - `engine-world-wasm` `range.rs` and `play.rs` (Broadside, the timber yard) are a GLB
     threshold destructible with its own cannonball, launch, splinter and pose code: the
     lab's setup again.
   - `engine-vehicle-wasm` holds `Gauntlet`, `MeteorRange` and `MeteorRun`.
   - `engine-world-wasm` `vehicle.rs`, `soft.rs`, `city.rs`, `motion.rs` and `terrain.rs`
     are per-page bindings.
2. **Scenes built in JavaScript.** `demo.html`'s scene set exists only in JS, so C, Unreal
   and native can't run it.
3. **Page shells.** Eight self-contained pages, each with its own render pipelines, inline
   WGSL, camera, keyboard handling, HUD and frame loop. About five overlapping renderers:
   the fracture lab's (`fracture_draw.slang`), `buggy-render`, `car-render`,
   `terrain-render` and `demo.html`'s instanced one.
4. **A second wasm module.** `engine-vehicle-wasm` runs no physics (layout constants,
   vehicle and soft-body descriptors, RoR/JBeam import), but every vehicle page downloads
   and starts it, a second copy of `engine-vehicle`.
5. **Check scripts per page.** `motion-check`, `softbody-check`, `regression`,
   `lab-matrix-check`, `record-*`, `gables-regression`, `ring-regression`.
6. **The legacy engine.** Top-level `crates/` (`rubble3d`, `rubble-wasm`…) and `web/`
   (three.js) are still in the repo; the README calls them legacy.

## Target

### 1. A scene registry (`rubble::play`, in `engine-world`)

`rubble::lab` grows into the general API; the lab's types become its fracture part.

- **`SceneDef`**: what's in the world. These parts compose freely:
  - rigid content (stacks, rings, piles, joints/ragdolls/chains/motor fields: today's
    `demo.html` builders, ported to Rust);
  - destructibles (GLB, cuboid, pack, prebuilt, under any bond model);
  - vehicles (rigid and soft, by descriptor, RoR `.truck` or JBeam);
  - ground (plane, heightfield, MPM deformable surface, city tiles);
  - characters.
- **`Action`**: what happens. Today's lab actions plus drive, AI routes, fire bullets,
  meteor shower, crash runs (softbody `headon`/`tbone`…). Scripts like Broadside's director
  and the timber yard's take are an action schedule.
- **`Settings`**: one struct, with the lab's settings plus World's solver features.
  `demo.html`'s A/B flags become `WorldFeatures` fields; flags the plan already fixed stay
  rejected with their reason, as `demo-config.mjs` does now.
- **`Preset`**: a name, a scene, actions, settings, a camera and controls. Every current
  URL becomes a preset (`vehicle.html?scene=gauntlet` → preset `gauntlet`). Broadside, the
  timber yard, the showcase and the meteor impact are presets, not engines.
- **`Play`** (`play.rs` today, generalised): steps the world, fires the schedule, writes
  the pose and draw buffers, reports stats (`tick_gpu_ms`).

Bindings: wasm (`Playground`), C (`rbCreatePlay`, `rbPlay_Step`, `rbPlay_Queue`,
`rbPlay_GetStats`), native (`engine-viewer` `world_viewer`).

### 2. One page shell

`play.html?preset=…` (and `?scene=&bonds=&action=…` for free combinations) owns:

- device and canvas;
- the frame loop, interpolation and timing;
- the HUD (stats from `Play`, plus a preset's own fields);
- the menu (scene, bond model, action, settings);
- camera rigs (orbit, chase, crash cam, fly).

A preset's **controls** (drive, shoot, character) are small input maps the shell routes to
`Play` actions.

Old URLs keep working by redirecting to their preset, so README links, media and recordings
don't break.

### 3. Render layers in Slang

A layer is drawn when the scene has its content:

- rigid bodies and chunks (one instanced path that scales to ring110k and gables506k);
- vehicles and wheels;
- soft-body skins;
- heightfield;
- MPM particles and the deformable surface;
- splinters and debris;
- city tiles.

The layers are today's renderers moved into `engine/slang/render`, with inline WGSL
removed. Native draws the same layers through `engine-viewer`.

### 4. One check over the registry

One matrix (preset × host) extends `lab-matrix-check.mjs` and `lab-cross-host.mjs` to every
preset. Each preset carries its check: it steps, its action fires, it draws with no WebGPU
error, and its preset-specific invariants hold (ring regression, gables reference, vehicle
parity, softbody calibration). It runs natively, through C and on the web, so every demo is
covered by the parity tests the lab already has. The per-page scripts fold into it; their
assertions stay.

### 5. One wasm module

Descriptors, constants and RoR/JBeam import move into `engine-world-wasm` (or are exposed
by `engine-world`). `engine-vehicle-wasm` remains only if a test needs it standalone.

## Phases

Each phase ships on its own, keeps every page working, and is checked before and after
for frame time.

| Phase | What | Done when |
|---|---|---|
| 1 | **Registry core.** `rubble::play` with `SceneDef`/`Action`/`Settings`/`Preset`/`Play`. The lab becomes its first user (no behaviour change). Broadside and the timber yard move onto it (`range.rs` cannonball → the lab's shot). | Lab matrix and cross-host unchanged; Broadside and timber tests pass on `Play`; `range.rs` ball code deleted |
| 2 | **demo.html scenes in Rust.** Port `demo-config`/`demo-scene`/`scene-builders` to the registry; JS keeps only flag parsing. The `scene_builders` and `spatial_order` invariant tests run against the Rust builders. | Ring regression and gables reference identical; ring110k and ragdolls10k frame time unchanged; the scenes run from C |
| 3 | **Vehicles, soft bodies, motion, terrain, city as presets.** Their page bindings (`vehicle.rs`, `soft.rs`, `motion.rs`, `terrain.rs`, `city.rs`, `engine-vehicle-wasm` scene builders) become `SceneDef` parts and `Action`s. | vehicle parity, softbody check, motion check, terrain and city tests pass through `Play` |
| 4 | **The shell and render layers.** `play.html` with Slang layers; each page in turn becomes a redirect to its presets, largest renderer overlap first (demo, vehicle, showcase/impact, softbody, then motion, terrain, city). | Screenshot comparison per preset; frame time per preset unchanged; old URL redirects |
| 5 | **One check, one wasm.** The preset matrix (native, C, web); fold the per-page scripts; merge `engine-vehicle-wasm`. | One matrix covers every preset on three hosts; one wasm download per page |
| 6 | **Docs and cleanup.** README gallery links to presets; `public-api.md` covers `rubble::play`; delete the replaced page code. | No page-specific engine code left outside presets |

## Decisions (2026-11-06)

1. **The legacy engine (`crates/`, `web/`).** Retired in its own PR (deleted, with a git
   tag for history). The playground PRs leave it alone.
2. **`blast_parity`.** Stays on the stress world as the one labelled exception until it's
   ported onto `World` (separate work, as decided in the lab plan).
3. **Old page URLs.** Redirect to their presets indefinitely (README media and recordings
   depend on them).
4. **Page-only tuning** (camera paths, HUD fields, crash-cam timings). Carried into each
   preset as data; nothing dropped. Any difference is listed in the phase's report.

## Risks

- **Scale.** ring110k, gables506k and the 35k-body lab grid depend on specialised draw
  paths. The shared rigid layer must keep them; phase 4 checks each before replacing it.
- **Lost behaviour.** The pages hold a lot of small tuning. Each page moves only with a
  screenshot comparison, its check script passing, and a list of its query flags mapped
  to preset settings (an unmapped flag fails loudly).
- **Native wgpu isn't bit-reproducible.** Cross-host checks use tolerances, as the lab's do.
- **Size of the move.** Phases 1–3 are Rust-only and invisible to users; the user-visible
  change (phase 4) happens one page at a time.

## Out of scope

- New physics, new scenes or retuning existing ones.
- Pre-existing failures on main (listed in [lab-world-plan.md](lab-world-plan.md)), which
  remain follow-ups.
- Porting `blast_parity` onto `World`.
- The Unreal plugin's own UI (it gets `rubble::play` through the C API; using it there is
  separate work).
