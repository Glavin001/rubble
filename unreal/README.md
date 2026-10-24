# RubbleDemo: Rubble's demos in Unreal Engine 5.8

This is a single-player Unreal client for Rubble's web demos: rigid scenes (`demo.html`), vehicles
(`vehicle.html`), soft-body cars (`softbody.html`) and the fracture lab. It drives the real engine
through its C ABI (`engine/crates/engine-ffi/include/rubble.h`). Unreal adds the look: Lumen,
virtual shadow maps, a physical sky, volumetric clouds and fog, PBR materials and cosmetic debris.

The at-scale design is in [DESIGN.md](DESIGN.md), and the plan and status are in
[PLAN.md](PLAN.md).

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
   | the few scenes the ABI cannot build yet
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
| Vehicles | Every scene. Presets are drawn from their descriptors (`rbGetVehiclePresetJson`) | — |
| Soft-body cars | Every crash test, the yard and the traffic derby. The demo's `procedural_car` (`rbGetSoftVehiclePresetJson`) uses the calibrated 240 Hz settings, rigid shapes are soft obstacles, and a fitted shell is bound with `rbSoft_BindSkin` | `crates`: no lattice-crate preset |
| Fracture | `house`, `arch`: Shardwright GLBs at level 0, with the lab's own solver and bond preload | The cuboid, row and grid scenes: `rbCreateDestructibleAssetCuboidHouse` exists, but nothing reports chunk geometry to draw them with |

The ground is an infinite plane (`rbCreatePlaneShape`). Shots reuse parked balls through the
body commands.

## C ABI gaps found integrating it (for the public API)

PR #66 and PR #67 closed several earlier gaps: body commands, kinematic bodies, soft-vehicle
presets, the cuboid house, plane shapes, vehicle descriptors and the lab's solver settings.
These remain:

1. **`rbSoft_BindSkin` binds in the vehicle's own frame, not the world.** The header says
   "world coordinates at spawn", but `SoftHost::bind_mesh` binds against the descriptor's node
   positions. A world-space mesh binds to the wrong nodes and stays where it was spawned. The
   plugin passes the descriptor frame. Fix either the docs or the binding (transform by the
   spawn pose).
2. **The destructible render contract is not reported for GLB assets.** The docs say chunk
   world = body × chunk local, with chunk local the identity in the Bonds model, but a
   Shardwright GLB's chunk meshes are local to the fragment. Rubble places each one at the
   fragment's centre of mass (the glTF node translation) plus `build_scene`'s per-asset
   placement, which centres the level in X and Z and lifts it onto the ground. The plugin
   re-derives that placement from the render meshes, correct to about 0.5 mm. Needed:
   `rbDestructible_GetChunkLocal`. Also, the GLB `level` the lab uses is 0; the quickstart
   and the ABI tests use 1, which for the arch subdivides only one stone.
3. **No chunk geometry for procedural destructibles** (the cuboid house). Debug draw shows the
   boxes but not which body each belongs to.
4. **Vehicles still have no chassis body id.** The descriptor gives the shape, but the pose
   comes from the wheels. Parked spare wheels appear in the moves at y = -5000 m.
5. **No flexbody render mesh for the procedural car** (`engine-softbody/src/visual.rs`). The
   plugin fits its own shell to the node cage.
6. **Destruction events carry no point or normal.** Dust and sparks are placed at the chunks
   that woke.
7. **No builders for the joint scenes** (ragdolls, chains, the motor field) and no crate
   preset.
8. **No engine features on `RbWorldDef`** (`grid_levels`). At most 1024 statics wider than half
   a grid cell are allowed (`OVERFLOW_LARGE_STATICS`).
9. **Capacities are guesswork.** Brick walls need about 30 contacts per body. A helper that sizes
   a `RbWorldDef`, or reports high-water marks, would help.
10. **`coalesce` folds in the earliest previous pose.** To interpolate by an alpha within the
    latest step, the host needs that step's own previous pose. The plugin polls step by step.
11. **Hit events (P4).** Impact effects wait on them.

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
