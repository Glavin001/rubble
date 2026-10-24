# RubbleDemo: Rubble's demos in Unreal Engine 5.8

This is a single-player Unreal client for Rubble's web demos: rigid scenes (`demo.html`), vehicles
(`vehicle.html`), soft-body cars (`softbody.html`) and the fracture lab. It drives the real engine
through its C ABI (`engine/crates/engine-ffi/include/rubble.h`). Unreal adds the look: Lumen,
virtual shadow maps, a physical sky, volumetric clouds and fog, PBR materials and cosmetic debris.

## Run

```bash
scripts/unreal/build-ffi.sh       # librubble_ffi.a (Build.cs also runs this; RUBBLE_FFI_PREBUILT=1 skips it)
scripts/unreal/build-editor.sh    # the RubbleDemoEditor target
scripts/unreal/build-content.sh   # materials, generated headless from Content/Python/build_content.py
scripts/unreal/run.sh fracture:house
scripts/unreal/run.sh soft:wall?speed=56&overlap=0.25
scripts/unreal/test.sh            # Rubble.Clock, Rubble.Mock, Rubble.Abi (headless), Rubble.Render (in game)
```

Scenes are `family:id?key=value`, using the families `rigid`, `vehicles`, `soft` and `fracture`
and the pages' own ids and parameters. Press Tab in game for the menu.

| Key | Action |
|---|---|
| F, left click | Shoot |
| WASD, arrows | Drive, or move the camera |
| Right-drag | Orbit |
| Wheel | Zoom |
| C | Camera: orbit, chase or fly |
| K | Cinematic |
| T | Side ram |
| R | Reset |
| Z | Slow motion |
| P | Pause |
| G | Effects |
| H | HUD |

Only Unreal 5.8 on Apple silicon is set up. The Mac toolchain settings (the force-included
`Build/Mac/MacSDK14Compat.h`, no PCHs, UBA off, `-NoHotReload`) are copied from vibe-land's
VibeCity, where they were found.

## How it fits together

```
renderers (primitives, fracture, vehicles, soft bodies, effects, environment)
   ^ OnFrame (merged per display frame), Alpha (interpolation)
URubbleSubsystem: fixed-step clock, merge, input
   ^ IRubbleBackend
FRubbleAbiBackend --> librubble_ffi (rubble.h, Unreal coordinates)
   | scenes the ABI cannot build yet
FRubbleMockBackend (stand-in motion; the HUD says "mock: <reason>")
```

- **Coordinates.** `RB_COORDINATES_UNREAL` makes the ABI convert everything exactly, so the plugin
  only copies floats. Shardwright GLB meshes are read by `RubbleGlb` and converted with the same
  axis map (Unreal X = -Rubble Z, Y = Rubble X, Z = Rubble Y, ×100).
- **Stepping.** The subsystem turns real time into fixed steps, capped per frame and never
  coarsened. It polls one step at a time and merges the steps of each display frame. Renderers
  interpolate across the latest step by the clock's alpha.
- **Scenes.** Scene layouts are built through the public API from the same layouts the mock draws:
  bodies and shapes, vehicle presets, and destructibles from the lab's GLBs. The ground is one
  static box, 4 m thick.

## What runs on the real engine

| Family | On the C ABI | Still on the mock, and why |
|---|---|---|
| Rigid | Every stack, ring, gable, pile, grid and scatter scene | `joints`, `ragdolls`, `hingechains10k`, `motorfield10k`: their builders are not on the ABI |
| Vehicles | Every scene, using the presets | — |
| Fracture | `house`, `arch` (Shardwright GLBs at level 0, as the lab loads them; Bonds model) | The cuboid, row and grid scenes: procedural destructibles are not on the ABI |
| Soft-body cars | — | All: the ABI takes Rigs of Rods `.truck` text only |

## C ABI gaps found integrating it (for the public API)

1. **Vehicles have no chassis or geometry query.** `rbCreateVehiclePreset` returns an index
   only. The plugin places the body from the wheels and takes the wheel radius from the contact
   distance. Spare and detachable wheels also appear in the moves, parked at Rubble y = -5000 m,
   so a renderer has to know to hide them. Needed: a `rbVehicle_GetInfo` returning the chassis body, chassis extents (or a render
   mesh), and the wheels' radius, width and rest hubs; the part bodies for destructible presets
   would help too.
2. **No procedural soft car, and no flexbody mesh.** The demo's car is
   `engine_softbody::procedural::car_desc`/`car_truck`. Needed: an `rbCreateSoftVehicleProcedural`
   (or the `.truck` text it makes), and the detailed render mesh from `visual.rs` with UVs and
   parts, to pass to `rbSoft_BindSkin`.
3. **No body commands on the ABI.** The Rust API has set transform, set velocity, impulse and
   wake. Without them a shot creates a new ball every time (nothing is parked or reused), and
   props such as the seesaw cannot be driven.
4. **The destructible render contract is not reported for GLB assets.**
   `rbDestructible_GetChunkBody` and the docs describe chunk world = body × chunk local, with
   chunk local the identity in the Bonds model ("meshes authored in the asset frame"). A
   Shardwright GLB's chunk meshes are local to the fragment. Rubble places each one at the
   fragment's centre of mass (the glTF node translation) plus a per-asset placement:
   `build_scene` centres the selected level in X and Z and lifts it onto the ground, from the
   collision hulls' bounds. The plugin recomputes that placement from the render meshes, which
   is correct to about half a millimetre. Needed: an `rbDestructible_GetChunkLocal` that returns
   the real per-chunk transform (fragment pose plus placement), so hosts never re-derive it.
   Also, the GLB `level` the lab uses is 0 (the arch's 23 stones). The quickstart and the ABI
   test use 1, which for the arch subdivides only one stone.
5. **Destruction events carry no point or normal.** Dust and sparks are placed at the chunks that
   woke this frame.
6. **No scene builders on the ABI.** The joint scenes and `DestructibleAssetDesc::cuboid_house`
   and its family are missing. `rbCreateDestructibleAssetFromDesc` (descriptor first) would cover
   the lab's procedural scenes.
7. **No engine features on `RbWorldDef`.** `grid_levels` is the one that matters most.
   Separately, at most 1024 statics wider than half a grid cell are allowed
   (`OVERFLOW_LARGE_STATICS`), so a ground made of tiles overflows; this one is a single box.
8. **Capacities are guesswork.** Brick walls need about 30 contacts per body. A helper that sizes
   a `RbWorldDef` from body counts, or reports high-water marks, would help.
9. **`coalesce` folds in the earliest previous pose.** That suits motion vectors, but to
   interpolate by an alpha within the latest step the host needs the latest step's previous pose.
   The plugin therefore polls step by step and merges the steps itself.
10. **Hit events (P4) and kinematic bodies (P3).** These are already planned; impact effects and
   animated props wait on them.

## Layout

- `RubbleDemo.uproject`, `Config/`, `Build/Mac/`, `Saved/UnrealBuildTool/BuildConfiguration.xml`
  (UBA off).
- `Source/RubbleDemo/`: the game mode, the pawn (cameras, input), the Slate menu and HUD, and the
  in-game render tests.
- `Plugins/Rubble/Source/RubbleRuntime/`:
  - the types and backend seam: `RubbleTypes.h`, `RubbleBackend.h`;
  - the ABI and mock backends;
  - the clock, the subsystem and the director;
  - the renderers;
  - `RubbleGlb`;
  - the tests.
- `Plugins/Rubble/Source/ThirdParty/RubbleFFI/`: links `librubble_ffi.a`, built by
  `scripts/unreal/build-ffi.sh` and not committed.
- `Content/Python/build_content.py`: the materials. The content it generates is not committed.
  Without it the game runs on flat engine materials.
