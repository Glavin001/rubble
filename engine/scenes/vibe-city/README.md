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

## Status (2026-10-18)

Done:
- All 17 presets export (`bun run export.ts`).
- 12 box presets build as lab scenes with welds and pass `tests/pack.rs`: wall, brickWall,
  hut, bridge, beamBridge, tower, townhouse, courtyardHouse, vaultedLoft, reinforcedTower,
  tiltingGantry, kinematicDumbbell. Each one's support welds hold its weight.

Blocked, needs a decision: the 5 Voronoi presets (fracturedWall, fracturedWallHut,
fracturedTower, fracturedBridge, fracturedGlb). three-pinata's pieces have ~15 real faces
plus 0.01 mm² sliver triangles and near-parallel steps up to ~1 mm. The engine's cooker
needs exactly planar faces and at most 64 vertices, so they fail with "non-convex" or
"N vertices > 64". Making them load changes the geometry slightly (AGENTS.md: needs the
user's call). Options:
1. Bake them with Shardwright instead (same dimensions and piece counts, rubble's normal
   asset path; pieces differ from vibe-city's).
2. An explicit, engine-wide hull clean-up tolerance (weld faces within e.g. 0.1 mm, then
   split hulls over 64 vertices into exact convex parts), reported per asset.
3. Keep three-pinata's render mesh and use a documented collision approximation.

Not done yet:
- Fracture lab wiring: `Sample` pack entries, a wasm `Scene.fromPack`, loading
  `generated/vibe-city/*.json` in `fracture/main.mjs`, hub link, and the export step in
  `engine/scripts/build-web.sh` (Vercel needs bun or npx bun).
- GPU check that each preset stands and the default shot breaks welds.
- tiltingGantry / kinematicDumbbell: platform motion is not modelled (supports are static).
- vibe-city's cutter, push and damage tools.
- Pack size: Voronoi packs are 10–40 MB of JSON (fine for generated files; compact later).
