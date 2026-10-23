# Scene port catalogue

Every destruction and physics scene built in earlier repos, and how each one gets into rubble
so rubble can be compared against all of them. Surveyed 2026-10-18 across every branch, stash
and worktree of: vibe-city (1/2/3), vibe-land (+2/3/old/matter), blast-stress-solver (+old),
PhysX (blast/, destruction/stress-oracle, demos/blast-stress-demo), Shardwright, three-pinata,
vibe-world (+2), snap-machines and rubble's own branches.

Baseline is rubble `main` (7de17f0b). `feat/avbd-attempt-2` is 483 commits behind main and
lacks the stress oracle, fracture samples, vehicles and legacy scenes listed under "Already in
rubble".

## Port routes

Each scene gets one route. The route decides the work, not the scene.

| Route | Path into rubble | Covers |
|---|---|---|
| **SW** Shardwright | source GLB (or `frac-cli` structure) → `prefracture` bake → `.fracture.glb` → `engine-fracture` sample + fracture lab Scene menu | authored buildings, GLB fractures, Voronoi scenes |
| **PK** ScenePack | ScenePack JSON (`nodes`, `bonds`, optional `nodeMeshes`) → converter into an `engine-fracture` scene (box chunks via `cuboid.rs`, hull chunks from `nodeMeshes`), reusing `stress-ref/src/scene_pack.rs` parsing | every blast-stress-solver / vibe-land pack (~50) |
| **RS** Rust builder | procedural builder ported to Rust (`engine-fracture` for bonded chunk scenes, `engine-scenes` for rigid-only) | rectilinear grids, brick walls, arches, flywheels, procedural towers |
| **OR** Oracle to web | `stress-ref` builders already in rubble → wasm export → fracture lab | stress-oracle b1–b9 and s_* |
| **RB** Rigid only | `engine-scenes` + `demo.html?scene=` | non-destruction physics (stacks, ragdolls, heightfields) |

Every ported scene needs: (1) a builder or asset in rubble, (2) an entry in the web demo
(fracture lab Scene menu or `demo.html` SCENES + `index.html` hub link), (3) the original impact
setup (projectile mass/speed/radius, ram, removal, blast) as a named case, and (4) where the
original had a stress solver, the breakable-bond path (`JOINT_FLAG_BOND`, currently opt-in) or
the stress-rubble pipeline turned on. Without (4) the comparison is against unbonded rubble.

Bench integration: add each ported scene to `bench/scenes` as a `fracture`-tier scene so the
existing cross-engine adapters (rubble, rapier, physx, jolt, three-avbd) run it.

## Already in rubble main

| Scene | Where | Origin |
|---|---|---|
| Shardwright house L0–L1, arch L0–L3 (+ row, grid, suburb, cuboids, cuboidGrid) | `engine-fracture/src/samples.rs`, fracture lab | Shardwright / vibe-land town-kit |
| veneer-house-hull16 | `fracture/assets/` | Shardwright |
| stress-oracle b1–b9, s_* showcases (native only) | `engine/stress-oracle/crates/stress-ref` | PhysX stress-oracle |
| Packs: brick-building, comp-wall-bay, house-1story, house-2story, rig-portal, rig-toppled, veneer-house-l0, villa-savoye (native stress only) | `engine/stress-oracle/packs/` | blast-stress-solver structures |
| stress-rubble scenes: wall_ram, b5_v10/20/40, house_ram, house_idle, town(16), town_idle, veneer, wall_row, striker | `stress-rubble` | PhysX stress-physx |
| Vehicle scenes: playground, destruction, buggies, traffic, slab, seesaw, carry, sidewall, bridge, tanklab | `vehicle.html` | rubble |
| Dune buggy (vibe-land garage "Sand rail": recipe mesh, rig, measured joints → breakable part mounts) | `engine/scenes/dune-buggy`, `vehicle.html?scene=buggies`, preset `dune_buggy` | vibe-land `client/src/vehicles/dune` |
| three-avbd paper/scale scenes, legacy scenes, convex piles, joints | `engine-scenes`, `demo.html` | three-avbd, legacy rubble |

Gaps in this group: the oracle scenes and the 8 packs have no web page; shardwright_house L2/L3
skip on degenerate hulls (recook in Shardwright).

## To port

Priority: **P0** core structures the comparison needs, **P1** wider coverage, **P2** nice to have.

### 1. ScenePacks (route PK; one converter covers all)

Source: `PhysX/blast/blast-stress-demo-rs/assets/scenes/`, `vibe-land/destruction/assets/scenes/`,
`PhysX/blast/blast-stress-solver/assets/{mini-city,reference,conformance}/`. Generator:
`PhysX/blast/blast-stress-solver/structures/*.mjs`, `scripts/export-*.mjs`.

| Scene | Chunks / bonds | Notes | P |
|---|---|---|---|
| fractured-tower | 204 / 546 | Voronoi hull chunks; 1/2/3-floor variants 83/148/204 (`vibe-land/destruction/src/variants.rs`); pinned by `vibe-land/destruction/tests/fractured_tower.rs` | P0 |
| fractured-wall | 91 / 254 | Voronoi | P0 |
| fractured-bridge | 65 / 102 | Voronoi | P0 |
| brick-building | 248 / 494 | 2-storey brick shell (pack already in rubble) | P0 |
| high-rise-3f-local | 318 / 1,083 | boxes; vibe-land /city default | P0 |
| high-rise-10f-local | 1,032 / 3,624 | boxes; perf grids | P0 |
| fractured-highrise-10f | 1,096 / 3,451 | exact-contact bonds | P1 |
| house-1story, house-2story | 319, 599 | already in rubble packs | P0 |
| villa-savoye | 1,045 / 2,464 | already in rubble packs | P1 |
| parking-garage | 3,177 / 9,194 | | P1 |
| algedra-tower | 7,959 / 21,049 | | P1 |
| neighbourhood | 8,877 / 23,581 | | P1 |
| park-432, petronas | 24,452, 24,766 | | P1 |
| minas-tirith, minas-tirith-rebuilt | 49,049, 64,213 | | P2 |
| skyline-stable, skyline | 29,592, 62,525 | | P2 |
| fractured-city, mini-city-v2 | 949, 1,267 | mini-city | P1 |
| fractured-district, fractured-downtown, fractured-town | 15,918, 24,105, 49,070 | downtown = netlab scene | P1 |
| crush-wall, reference-building(-crush) | 216, 64 | reference rigs | P0 |
| structure-conformance-v2 | 7 | conformance | P0 |
| comp-wall-bay ×1/2/4/8, comp-frame-bay ×1/2/4/8, comp-room, comp-storey, comp-stack | 35–284 | stability components | P1 |
| rig-column, rig-pane, rig-portal, rig-toppled, rig-wall, rig-cantilever, rig-garage | 6–725 | with the 8 rig scenario tests in `vibe-land/destruction/tests/rig_scenarios_sim.rs` | P1 |
| embedded-penetration, embedded-four-buildings | 444, 1,776 | vibe-land `origin/codex/gpu-destruction-integration` | P2 |

### 2. vibe-city `/destructible-stress` presets

Source: `vibe-city3` branch `feat/stress-destruction-april-6`, `src/lib/stress/scenarios/`
(3 extra presets on `feat/tilting-gantry`). Impact: ball 15 t, r 0.5, 36 m/s; also cutter/push/damage.

| Preset | Construction | Route | P |
|---|---|---|---|
| wall | rectilinear grid 6×3×0.32 m, 12×6×1, 10 t | RS | P0 |
| brickWall | running-bond bricks 24×12, mortar clumps, 40 t | RS | P0 |
| hut | rectilinear shell 6.5×3.4×5.2 m, 18×9×14, door + window, 19 t | RS | P0 |
| bridge | suspension footbridge 18 m, 40×12×12, cables, 24 t | RS | P1 |
| beamBridge | deck 18×5×0.6 m on 4+4 posts, 60 t | RS | P0 |
| tower | frame tower 16×58×16, ~24 m, 280 t | RS | P0 |
| reinforcedTower | tower + column grid, outriggers, 310 t (`feat/tilting-gantry`) | RS | P1 |
| tiltingGantry | ring of 12 block columns on tilting/spinning kinematic platform (`feat/tilting-gantry`) | RS | P2 |
| kinematicDumbbell | 2 bonded blocks on spinning pad (`feat/tilting-gantry`) | RS | P2 |
| townhouse | 26×18×16 grid, gable roof, 620 t | RS | P1 |
| courtyardHouse | 28×14×28 grid, courtyard, 880 t | RS | P1 |
| vaultedLoft | 24×18×14 grid, barrel roof, mezzanine, 810 t | RS | P1 |
| fracturedWall | three-pinata Voronoi 6×3×0.32 m, 120/200 pieces | SW | P0 |
| fracturedWallHut | 4 Voronoi walls, 30/100 per wall, foundation, 19 t | SW | P0 |
| fracturedTower | Voronoi walls/floors/columns, 15 floors 45×45 m | SW | P1 |
| fracturedBridge | Voronoi deck + posts, 200/10 pieces | SW | P1 |
| fracturedGlb | `lion.glb` Voronoi 120 pieces (+ tower-big, bridges, game-level GLBs on disk) | SW | P2 |

Other vibe-city destruction pages (main): `/destructible-wall` single-wall (48 Voronoi pieces)
and mini-building (4 walls, 158 pieces), wrecking ball 0.45 m / 900 t — route SW, P1.
`/three-pinata` random block shatter — SW, P2. `/shockwave-demo` ball-pit and box-tower
(12×5×5) with blast presets — RB, P2. Heightfield craters/ragdoll — RB, P2 (ragdolls exist).

### 3. blast-stress-solver demos

Source: `PhysX/blast/` on HEAD unless a branch is named (branches in `blast-stress-solver`).

| Scene | Description | Route | P |
|---|---|---|---|
| wall / tower / bridge (procedural) | `blast-stress-solver-rs/src/scenarios/{wall,tower,bridge}.rs`; 12×6 wall, 4×4×8 tower; shot presets in `blast-stress-demo-rs/src/main.rs` | RS | P0 |
| high-rise (Rust demo) + shot presets | high-rise-low/mid-infill, low-column | PK | P1 |
| wall-contract, wall-arch-contract | resim contract walls 5×4 and 10×8 bricks, set projectiles (`fix/resim-2`) | RS | P1 |
| JS: Tower Collapse, High-Rise Demolition (wrecking ball), Destructible House, Wall Demolition, Mini-City (meteor + charges) | `js_stress_example/*.html` | PK/RS | P1 |
| JS: Destructible Vehicle (buggy GLB, CoACD pieces) | ported from vibe-land's own buggy recipe instead (`vehicle.html?scene=buggies`, rigid parts on breakable mounts) | done | — |
| JS: Stress Cracks (cantilever), Centrifugal Spinner, Fracture Policy | | RS | P2 |
| brick-castle | ~2,700 bricks, towers, gatehouse, keep; cannon, wrecking ball, boulders (`origin/claude/brick-castle-demo-wmeuo7`) | RS | P1 |
| keystone-arch | 15 voussoirs r 7 m, keystone strike (`origin/claude/novel-library-example-luowr2`) | RS | P1 |
| flywheel-burst | hub, 6 spokes, 72-block rim, zero g, spin to burst (`origin/claude/relaxed-allen-ttk1ji`) | RS | P2 |
| frame-tower, concrete-hut, courtyard-bungalow | vibe-city presets re-ported (`origin/claude/serene-bohr-43rcua`) | = section 2 | — |
| hierarchical-wall / -tower / -bridge | 2-level chunk hierarchy (PhysX `origin/claude/hierarchical-destruction-system-VkM7k`) | SW (levels) | P1 |
| native geometry set: tower64, bridge64, cantilever64, dense12, chain256, chain32, panel32, ladder128 | `demos/blast-stress-demo/native_scenario_geometry.h` (PhysX `origin/claude/realtime-destruction-20260915`) | RS | P1 |
| native mini-city grid 1–64, city16/64/256-staggered (113k chunks), building-weak/strong/heavy-shot | `mini_city_main.cpp`, `native_destruction_main.cpp` | PK (fractured-tower tiled) | P1 |
| stress-gpu micro cases (impactor_onset … settled_wide) | `stress-gpu/src/micro.rs` | OR | P2 |
| Shardwright two_storey_building L2 cases idle/ram/projectile/remove | `stress-ref` `shardwright_destruction` (already in rubble native) | SW | P0 |

### 4. Shardwright structures (route SW)

Source: `Shardwright/benchmarks/assets/*.glb`; procedural structures on branch
`structures-vibe-land` (`crates/frac-cli/src/structures.rs`, "ported from the vibe-land town kit").

| Asset | Parts | P |
|---|---|---|
| two_storey_building | ~50 (collapse GIF: remove f0_col_21, ball −15 m/s) | P0 |
| building_v0, building_v1 | 132, 419 | P0 |
| brick_wall_window | 4×3 m wall | P0 |
| pratt_truss_bridge, rc_viaduct | 45, 26 | P1 |
| steel_office | 223 | P1 |
| rc_column, rc_slab_on_columns | rebar FEM golden; 5 | P1 |
| glass_annealed_pane, glass_tempered_pane, timber_beam, ceramic_bowl, messy_scan | material cases | P2 |
| tree, conifer | 46, 53 | P2 |
| held-out: concrete_pipe, drywall_door, stone_arch | | P2 |

### 5. vibe-land structures not covered by packs

Source: `vibe-land` (`feat/unreal-client`) unless a branch is named.

| Scene | Description | Route | P |
|---|---|---|---|
| calibration: bridge-piers, truss-members, frame-column, demolition, house-studs, house-headers, masonry-arch (+ house-headers-removal on `dev/dynamic-sequence`) | `structures/calibration/scenarios/*.mjs` — removal/charge cases with GSA/UFC expectations | SW (arch, truss, viaduct, house already in Shardwright) + removal cases | P0 |
| town-kit buildings: victorian-corner, porch-house, corner-grocery, workshop, bungalow, strip-shop, framed-houses, library, cinema, fire-station, office-tower, veneer-bungalow/house, stairs | `structures/town-kit/src` | SW | P1 |
| town-kit towns: bayline-town (+gardens-market), bayline-district (82k), small-town (183k), civic-town (189k) | | SW tiled | P2 |
| town-kit outdoor props (20), trees (15), interior props (14), playground (24 exhibits) | | SW | P2 |
| Vibe Town, hero, Bayline Heights showcase | `native-mac.sh` presets | SW | P2 |
| vehicle-lab trials: house, framed-house, wall, rubble, meteor-house, cannonball-framed-house, truck-ball-house, town-chase; wall-matrix targets | `structures/vehicle-lab/` | SW + vehicle.html | P1 |
| perf: city-10f-mass (25.8k), -7x7 (51.7k), mass-bodies/meteors, fleet/stock idle+meteors; audit intact-g1..16, pile-g5/g7, hibernate-g5 | `scenarios/perf/` (`feature/debris-hibernation` for audit) | PK tiled | P1 |
| netcode: wedge, sustained, meteor-volley; netlab on downtown | `scenarios/netcode/` | PK | P2 |
| fracture-lab specimens: rc-wall, concrete-wall, brick-wall, drywall, glass-pane, rc-slab, rc-column, timber-stud | `feat/fracture-lab` `client/src/city/fracture/specimens.ts` | SW | P1 |
| destructible demo huts (drywall/wood/plaster) | `origin/cursor/destructible-sheet-materials-e885` | RS | P2 |
| practice wall (12×6) and tower (4×4×8) | April 2026 history a1e2816c | RS (= blast procedural) | P1 |
| material-showcase | `feat/matter-materials` | SW | P2 |

### 6. Older and smaller sources

| Scene | Source | Route | P |
|---|---|---|---|
| WoodYard (plank, post, beam, log, wall, door, …), Voronoi stone/rubble/radial/wood | three-pinata `feat/wood`, `codex/implement-different-fracture-modes-and-demos` | SW | P2 |
| BreakableWall, BuildingsCollision (WreckingBall, SkyscraperDemolition, BreakableBuildings, LargeCity …), MultiStoryBuilding (Tower, LShaped, UShaped, CourtYard, CrashTest …), LaserCutter, ExplodingWalls | vibe-world storybook (`src/stories/`) | RS | P2 |
| Snap machines: car, crane, trebuchet, fortress, building under construction | snap-machines | RB (joints) | P2 |
| rubble `gpu-optimize-april-*`: Stress Mixed 20k | rubble branches | RB | P2 |

### Out of scope

Not rigid-body destruction: HTN block stacking, navmesh demos, fire propagation, tornado and
GPU debris particles, dust/rain/VFX, decals/CSG bullet holes, ballistic raycasts, SDF physics,
Cesium map colliders.

## Suggested order

1. Work from `main`. Make breakable bonds usable for all fracture-lab scenes, since most
   originals ran a stress solver.
2. PK converter + fracture-lab Scene menu fed from one manifest; land fractured-tower,
   fractured-wall, fractured-bridge, brick-building, high-rise-3f/10f, house-1story/2story,
   crush-wall, reference-building with their original shots.
3. Oracle scenes (b*, s_*) and the 8 native packs on the web page.
4. vibe-city rectilinear builder in Rust (wall, brickWall, hut, beamBridge, tower, then houses).
5. Shardwright bakes: two_storey_building, building_v0/v1, brick_wall_window, vibe-city
   Voronoi scenes, calibration set.
6. P1 rows, then P2.
7. Each landed scene goes into `bench/scenes` fracture tier for cross-engine runs.
