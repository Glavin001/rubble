# One engine, thin bindings: plan

**Goal.** Whatever a host uses (C for Unreal and other game engines, the web with
three.js, Rust on its own or with Bevy), it calls the same Rust code through a thin
wrapper. Anything a host can do, the others can do too. The TypeScript library the
website uses stays, but as a typed layer over the wasm module, with its types
generated from Rust rather than written by hand.

This continues [playground-plan.md](playground-plan.md), whose phases moved the
scenes, the Director and the render manifest into Rust (`rubble::play`,
`World::render_manifest`). Its status table showed these gaps:

| Check | Before this plan |
|---|---|
| Physics is the Rust `World` (wasm on every migrated page) | ✅ |
| Scenes built in Rust (`SceneDef`), the same code C's `rbCreatePlay` uses | ✅ rigid, vehicle, motion, terrain |
| Scene scripts and player actions in Rust (`Director`) | ✅ wasm; ❌ C has only `rbCreatePlay`, `rbPlay_Step`, `rbPlay_GetStats` |
| What to draw decided in Rust (`render_manifest`) | ✅ wasm; ❌ not in C |
| Joints, ragdoll and legacy scenes | ⚠️ world settings assembled in JS (`demo-config.mjs`) |
| Settings described once in Rust, UI generated from that | ❌ each family module wires its own controls |
| Generated TypeScript types | ❌ the world wasm is built `--no-typescript`; JS objects cross as untyped JSON |
| One wasm module | ⚠️ the shell loads only the world wasm; old pages still load a second one |

## Rules

- **One implementation.** Each operation (place a scene, act, run a frame's scripts,
  read events, get the render manifest) is one Rust function in `engine-world`. A
  binding converts arguments and errors and does nothing else: no defaults of its
  own, no logic, no second copy of a struct.
- **Same names everywhere.** `WebWorld.placeScene` ↔ `rbWorld_PlaceScene` ↔
  `rubble::play::place`; `act` ↔ `rbWorld_Act` ↔ `Director::act`, and so on. A test
  checks that every scene operation has a C function and a wasm method
  (`engine-world/src/play/api.rs`, `HOST_API`).
- **One data contract.** Structured data (scene definitions, actions, events, presets,
  stats, the render manifest) is the Rust serde type. The web gets it as JS objects
  with generated `.d.ts` types (`ts-rs`). C gets the same JSON, documented by the same
  generated file. Hot data (poses, transforms, buffers) stays typed arrays or POD
  structs.
- **Generated files are checked in, and a test fails if they're stale** (as
  `engine-ffi/tests/header.rs` does for `rubble.h`). CI and fresh clones need no extra
  tool to read the types.
- **Coordinates.** C converts at the boundary, as every existing `rbWorld_*` call does.
  Scene JSON, actions, events and the manifest are in the world's coordinates
  (`RbCoordinates`). The web and Rust are native (Y up, metres).

## Phases

| Phase | What | Done when |
|---|---|---|
| 1 | **Generated types.** `ts-rs` on `SceneDef` (every family), `Action`, `Event`, `PlayStats`, `PresetInfo`, `RenderManifest`; written to `engine-webgpu/types/rubble-play.d.ts`. | A Rust test regenerates the file and fails on any difference |
| 2 | **One scene API.** `rubble::play::place` (place + Director) used by `Playground`, wasm `placeScene` and C. `presetList` returns `PresetInfo` (typed), not ad hoc JSON. | wasm keeps only conversion; `play_presets` passes |
| 3 | **C parity.** On `RbWorld`: `rbCreateSceneWorld`, `rbWorld_PlaceScene`, `rbWorld_Act`, `rbWorld_BeforeFrame`, `rbWorld_AfterFrame`, `rbWorld_PrepareFrame`, `rbWorld_ScriptPending`, `rbWorld_SetPlayer`, `rbWorld_SetAutofire`, `rbWorld_GetScriptState`, `rbWorld_TakeEvents`, `rbWorld_GetRenderManifest`; and `rbPresetList`, `rbSceneFromQuery`. JSON in caller-sized buffers. | An ABI test runs a preset through C with actions and events and compares it with Rust; the `HOST_API` test passes |
| 4 | **Typed web layer.** The world wasm built with TypeScript; its `JsValue` arguments typed (`typescript_type`); `rubble-world.mjs` and the shell's modules checked with `tsc --checkJs` against the generated types. | `npm run typecheck` passes in `engine-webgpu` |
| 5 | **World settings once.** `WorldDef` and its parts get serde and `ts-rs`; the wasm's hand-read `defs.rs` and C's `RbWorldDef` field copies go through it; `demo-config.mjs`'s settings for the joints, ragdoll and legacy scenes move into their `SceneDef`. | `defs.rs` field reading deleted; demo scenes from C |
| 6 | **Settings described once, UI generated.** Each `SceneDef` field carries its control (range, choices, label) in Rust (`SettingsSchema`); the shell builds its menu from it instead of each family module. | Family modules have no hand-wired scene controls |
| 7 | **One wasm module.** The old pages' second module (`engine-vehicle-wasm`) goes through the world wasm. | One wasm download on every page |

Phases 1–4 are this branch. Each later phase ships on its own.

## Status (2026-11-18, branch `claude/unified-engine-bindings-1afddd`, on `claude/playground-threejs`)

| Phase | State |
|---|---|
| 1 | Done. `play::api::typescript()` writes `engine-webgpu/types/rubble-play.d.ts` (25 types, from `ts-rs` derives on the Rust types). `engine-world/tests/api_types.rs` fails when it is stale (`BLESS=1` rewrites it) or names a type it doesn't declare. Inputs the Rust side defaults (`{action: 'fire'}` with no aim) are optional in TS; the render manifest's arrays are `ArrayLike<number>`, true of the web's typed arrays and of C's JSON arrays. |
| 2 | Done. `play::api::{place, preset_list, preset_scene, scene_from_query}`; `Playground`, wasm and C all place through `place`. `presetList` returns `PresetInfo` instead of hand-built JSON. |
| 3 | Done. The C functions above on `RbWorld`, plus `rbCreateSceneWorldOnMetal` (Unreal shares its Metal device, as `rbCreateWorldOnMetal`). `rbWorld_Step` runs the placed scene's script first, as `WebWorld.step` does. An action's positions and directions convert from the host's coordinates (unit test for Unreal's); scene JSON, script state, events and the manifest are in Rubble's frame, as `rbWorld_ExportPoses`' poses are. JSON out stays valid until the next JSON call on the same world (or thread, for the world-less calls). `tests/scene_abi.rs`: the `bullets` preset through C and through `Playground` gives the same scene JSON, the same events at the same steps (the player's fire and the script's own) and the same draw groups. `HOST_API` lists the 16 operations; `tests/header.rs` checks each has its C function, `api_types.rs` its wasm method. |
| 4 | Done for the scene API. The world wasm is built with TypeScript (`build-web.sh`, no `--no-typescript`); `createScene`, `placeScene`, `act`, `sceneFromQuery`, `presetList`, `presetScene`, `takeEvents` and `renderManifest` carry the generated types, which the module's own `.d.ts` includes. `rubble-world.mjs` is `// @ts-check`; `npm run typecheck` in `engine-webgpu` (CI's web job) checks it. The other wasm methods still take untyped `JsValue`s (their types come with phase 5). |
| 5–7 | Not started. |

Still untyped across the boundary: a family's placement (`placeScene`'s return,
`rbWorld_PlaceScene`'s JSON: `Placed::to_json`), the Director's script state
(`serde_json::Value`) and an event's `data`. Each becomes a type when a host needs to
read it; the shell reads them today by name.

`rbCreatePlay`/`rbPlay_*` (a preset with its own world, steps and counts) stay for hosts
that only run a preset; a host that draws or acts uses `rbCreateSceneWorld`.
