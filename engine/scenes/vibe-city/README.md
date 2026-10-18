# vibe-city `/destructible-stress` presets

The 17 structure presets of vibe-city's stress-solver page, kept here as rubble's source
of truth and exported as scene packs that the fracture lab loads (`engine-fracture`
`pack.rs`). Catalogue of every scene still to port: [`docs/scene-port-catalog.md`](../../../docs/scene-port-catalog.md).

## Layout

| Path | What |
|---|---|
| `src/scenarios/*.ts`, `src/core/{types,autoBonding}.ts` | copied unchanged from vibe-city `feat/stress-destruction-april-6` (bb16c52), `src/lib/stress/` |
| `src/scenarios/gantryPresets.ts` | reinforcedTower, tiltingGantry, kinematicDumbbell from vibe-city `feat/tilting-gantry` (2665b34) |
| `models/lion.glb` | the fracturedGlb preset's model (Git LFS) |
| `shims/rapier.ts` | stands in for `@dimforge/rapier3d-compat`: records each node's `ColliderDesc.cuboid` / `convexHull` |
| `shims/dom.ts` | `ProgressEvent` / `createImageBitmap` so three's GLTFLoader runs under bun |
| `export.ts` | builds every preset with the page's arguments and writes `engine/crates/engine-webgpu/generated/vibe-city/<id>.json` (git-ignored) |

```sh
cd engine/scenes/vibe-city
bun install
bun run export.ts            # all presets, ~8 s, ~90 MB
bun run export.ts hut tower  # a subset
cargo test --release --manifest-path ../../Cargo.toml -p engine-fracture --test pack -- --nocapture
```

`tsconfig.json` maps `@/lib/stress/*` to `src/`, Rapier to the shim and
`blast-stress-solver` to its ESM build (its CJS build cannot find its wasm under bun).
Math.random is seeded per preset (mulberry32, seed 1337 + index) so Voronoi presets are
reproducible.

## Pack contents

ScenePack v1 (`scenario.nodes`, `bonds`, `nodeSizes`, `nodeColliders`, optional
`nodeMeshes`) plus `source` (repo, ref, commit, seed), `parameters` (the builder's JSON
parameters) and `defaults`:
- projectile: the page's ball, r 0.5 m, 15,000 kg, 36 m/s;
- solver limits: the page's Blast limits, base x materialScale 1e8 (tension/compression
  fatal 270,000, shear fatal 360,000).

Node sizes and colliders follow the page: `fragmentSizes`, else `spacing`; the builder's
collider, else a box (supports x0.999).

## Loading in rubble (`engine-fracture/src/pack.rs`)

- Mass nodes are dynamic bodies; mass-0 supports are static bodies, and bonds to them weld
  to the world.
- Boxes are SHAPE_BOX colliders; hulls are cooked with `ConvexHull::from_points` in their
  principal frame (`compound` nodes: several hulls on one body).
- Bonds are the lab's breakable welds. Capacity = limit x area / BLAST_DT (1/60 s):
  blast-stress-solver measures stress as impulse / area per 60 Hz step. Bending =
  tension limit x area x centroid distance / 2 / BLAST_DT. Springs use one modulus
  (30 GPa) and square-face moments; the lab's scene-wide penalty scale sets the level.
- Shot: the pack's projectile.

## Hull clean-up (`hulls.ts`)

three-pinata's Voronoi pieces carry float noise from its f32 slicing (sliver triangles,
near-parallel steps up to ~1 mm). The engine's cooker needs exactly planar faces and at
most 64 vertices, so the exporter rebuilds each piece as an intersection of half-spaces:

1. Start from the planes of the piece's hull, each offset out to the farthest original
   point (the shape always contains the piece).
2. Remove small or near-parallel planes, smallest face first, while the shape stays
   closed (volume does not drop) and within `HULL_TOLERANCE` (0.1 mm) of the piece.
3. Remove planes the cooker cannot separate (near-parallel within ~0.08 deg, corners
   closer than 0.1 mm) while within `HULL_MAX_OUTSIDE` (1 mm).
4. Cut shapes over 64 vertices into exact convex parts (`compound` colliders).

Each hull is written with its corners and faces; `pack.rs` cooks it with
`ConvexHull::from_faces`, as baked Shardwright hulls are. The tolerances and the largest
extension past the original piece are recorded per pack (`hullCleanup`); the export
fails when a piece would exceed `HULL_MAX_OUTSIDE`.

| Preset | Hulls | Split | Largest extension |
|---|---|---|---|
| fracturedWall | 199 | 0 | 0.60 mm |
| fracturedWallHut | 400 | 0 | 0.09 mm |
| fracturedTower | 3,675 | 0 | 0.10 mm |
| fracturedBridge | 280 | 0 | 0.10 mm |

## Status (2026-10-18)

Done:
- 16 presets build as lab scenes with welds, pass `tests/pack.rs` and show in the
  fracture lab: the 12 box presets plus fracturedWall, fracturedWallHut, fracturedTower
  and fracturedBridge. Each one's support welds hold its weight.

Pending: fracturedGlb (lion). After clean-up ~10 of its 120 hull parts still fail the
cooker, and its clean-up takes ~3 minutes, so a full export skips it
(`bun run export.ts fracturedGlb` runs it).

Known behaviour: with breakable bonds on, structures sag under their own weight (the
per-piece weld creep in the fracture lab README). vibe-city held intact islands rigid.

Not done yet:
- tiltingGantry / kinematicDumbbell: platform motion is not modelled (supports are static).
- vibe-city's cutter, push and damage tools.
- Pack size: Voronoi packs are 10–40 MB of JSON (fine for generated files; compact later).
