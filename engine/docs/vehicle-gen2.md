# gen2: the generated sedan as BeamNG content

`engine_softbody::authoring::gen2` is Rubble's second vehicle generator. One Rust
source builds the physics skeleton and the render mesh together from one analytic
shape, and writes them as the same format the imported community cars use:

- JBeam parts (one file per part) and a `.pc` configuration;
- a Collada render mesh with its `materials.json`, whose meshes are flexbodies
  bound by node groups.

The generated car then runs through the unchanged import path: `jbeam::load_jbeam`
on CPU, the GPU solver, the page's `?mod=`. It is plain, editable JBeam. sedan1 and
the two-tier sedan are untouched; gen2 is a new selectable car.

```bash
cargo run --release -p engine-softbody --example gen2_export          # writes engine/crates/engine-webgpu/__mods/gen2 (gitignored)
cargo run --release -p engine-softbody --example gen2_crash -- \
    engine/crates/engine-webgpu/__mods/gen2 gen2 gen2.pc rest wall pole   # CPU references
cargo run --release -p engine-softbody --example vehicle_compare -- --cars sedan1,etkk,gen2
# page: softbody.html?solver=explicit&mod=/__mods/gen2&pc=gen2/gen2.pc&wheels=raycast&xscene=wall56
```

## Files

| File | What |
|---|---|
| `gen2/shape.rs` | The outer surface: plan outline (superellipse), belt line, greenhouse, crowned top. Nodes and mesh both sample it. |
| `gen2/build.rs` | Builder: nodes, beams by role, panels, pole lanes; the mass, stiffness, damping and yield rules. |
| `gen2/sedan.rs` | The car: structure, closures, hardware, masses. |
| `gen2/export.rs` | JBeam and `.pc` writer. |
| `gen2/mesh.rs` | Render mesh, materials, deform groups. |
| `examples/gen2_export.rs` | Writes the mod folder; prints the census, mass and CG. |
| `examples/gen2_crash.rs` | Rest, wall56 and the side-pole sweep, for gen2 or any JBeam car. |

## Rules (adopted from the census, `beamng-authoring-census.md`)

- **Nodes on features, about 0.5 m apart.** Body stations at the sill, belt,
  roof rail, pillars and door seams; doors 3 × 3 outer plus a 3 × 2 inner frame.
- **Light contact-only lane nodes** (0.15 kg) on every side-facing row edge
  longer than 0.20 m (doors, sills, quarters, fenders, the hood's outer edges),
  tied to the edge, the triangle corners and the nearest backing node. The
  collision skin is re-triangulated through them. Every outer panel is collidable.
- **Mass.** Each part's real mass (DOT HS 811 666) over its structural nodes by
  member kg/m × length and sheet area (a lumped Voronoi), with a floor of half
  the part's mean. Lane nodes carry a fixed light mass. Named masses (occupants,
  fuel, the front-end module and so on) go on the structure that carries them.
  The engine is 12 nodes of 22 kg.
- **Stiffness.** Each role's census k·L band (k = k·L / max(L, 0.4 m)). The cap
  is the real stability bound, not sedan1's per-node formula. A power iteration
  finds the highest mode of M⁻¹K, with damping counted as 2c/h and the loader's
  latch couplers included. The beams of the nodes that carry that mode are
  softened until ω_max·h ≤ 1.85.
- **Damping.** ζ = 0.04 on every spring beam; no dashpots.
- **Yield.** deform = ε_y · k · max(L, 0.4 m). ε_y is the census's 1.15 % on the
  body sheet and 1.45 % on doors. The crush zone and the cabin carry calibrated
  multiples (below). The structure is FLT_MAX; deformLimitExpansion is 1.2 on
  member chords.
- **Hardware.** Hinges are 2 break groups of 4 beams at 40 kN. Latches are
  node couplers: doors 40 kN, hood 30 kN, trunk 45 kN. Seals and stops are
  SUPPORT at 150 kN. Fenders and fascias are bolted in break groups.
- **Binding.** Each mesh binds to its own part's group; latch nodes and rim
  proxies are in no group.

## Deviations from the census, and why

- **Yield on the safety cage and the crush zone.** The census gives body
  ε_y 1.0–1.3 %. gen2's body sheet is at 1.15 %. The crush zone's members and
  webs carry 4.5× that (`GEN2_CRUSH`), and the cabin's 6× (`GEN2_CABIN`, the
  Accord's DP980/PHS cage). That puts the body's p50 at 5.7 %. At census yield
  the frontal crushed 1.3 m and the cabin folded. gen2 has 8 spring ends per
  node against BeamNG's 11–18, so each beam carries more of the section.
- **deformLimit 0.7 on the crush zones** (no census car uses it). Crushed bays
  stack at 30 % of their length instead of inverting. Without it the rail's
  first bay inverted (the tip node passed the next station, 0.67 m).
- **Doors: 17 structural nodes, not 9–13.** The Accord's doors are 24–30 kg
  against BeamNG's 17–22 kg; node mass stays in band (p50 1.4 kg).
- **Stops are SUPPORT at 150 kN, not preloaded**, and their long bound is 100 so
  they never break by stretching: short stops reach 2× their length within a few
  cm of travel.
- **Rim proxies.** Raycast wheels have no geometry, so each hub carries 6 rim
  nodes (2 kg each, out of the unsprung mass): contacters with no triangles,
  tied off the axle line. Without them the outer frontal zones crushed past
  the wheel.
- **Engine-to-dash BOUNDED stops** that engage with 8 cm left, and torque boxes
  from the rail's end to the sill and the floor rail.

## Measured (2026-11-06)

Mass 1,450.0 kg; CG 1,190 mm behind the front axle (band 1,001–1,248), 546 mm
up, 33 mm left (the driver). 517 nodes (154 contact-only), 2,276 beams, 430
collision triangles; ω_max·h 1.846.

| Part | nodes (lane) | node kg p50 | k·L p50 | ε_y p50 | census |
|---|---|---|---|---|---|
| body | 238 (68) | 2.74 | 8.0e5 | 5.7 % (zone multiples) | 2.5–4 kg, 0.8–1.1e6, 1.0–1.3 % |
| front door | 32 (14) | 1.39 | 4.6e5 | 1.45 % | 1.2–1.9 kg, 0.4–1.0e6, 1.3–1.6 % |
| hood | 29 (12) | 0.82 | 3.3e5 | 3 % | 0.55–1.5 kg, 2.2–4.7e5, 1.5–5.5 % |
| trunk | 10 | 1.26 | 3.5e5 | 3 % | 0.6–1.6 kg, 2.2–5e5, 2–5 % |
| fender | 24 (13) | 0.26 | 1.1e5 | 3 % | 0.4–0.8 kg, 1.5–4.3e5 |
| bumper cover | 31 | 0.19 | 6.5e4 | 9 % | 0.3–0.9 kg, 0.4–1.3e5 |

CPU, JBeam mode, `vehicle_compare` (pressure wheels, as the shared harness runs
every JBeam car):

| | gen2 | ETK kr6 | band |
|---|---|---|---|
| rest 10 s: broken / bent | 0 / 0 | 0 / 0 | 0 / 0 |
| wall56 dynamic crush | 774 mm | 489 | 620–750 |
| t_m | 85.5 ms | 46 | 63–76 |
| peak (rear cabin) | 39.5 g | 66.3 | 34–47 |
| residual max / avg | 422 / 272 mm | 228 / 193 | 380–580 / 261–512 |
| Campbell b1 | 48.9 /s | 69.2 | 22.9–39.2 |
| restitution | 0.208 | 0.353 | 0.13–0.20 |
| springback | 0.455 | 0.533 | about 0.25 |
| feet intrusion | 365 mm | 63 | 1–67 |
| pole 32 km/h residual | 400 mm | 57 | 297–445 |
| FMVSS 214 door: initial / intermediate / peak | 7.9 / 13.6 / 57.5 kN | 5.7 / 10.7 / 38.5 | ≥ 10.0 / 15.6 / 31.1 |
| mesh at the wall: stretch p99; dihedral p99; seams > 5 cm | 5.6; 80°; 76 | 1.5; 31°; 0 | |

`gen2_crash` (raycast wheels, as the page's crash lab): rest 10 s 0 broken,
0 bent, KE 0.01 J. wall56 residual 510 mm, rear-cabin dynamic crush 786 mm at
92 ms, e 0.17, 41.2 g, b1 40.7 /s. A-pillar 4 mm and cowl 2 mm rearward,
door apertures −1 / +18 mm. Side-pole sweep (90°, every 0.25 m): the deepest
node is ≤ 11 mm inside the pole behind the A-pillar and 17–34 mm at the front
fender and hood corner (band 25 mm).

## Open

- The mesh in the crush zone: stretch, folds and opened seams are worse than the
  ETK's. The loader binds a vertex to its nearest node's frame. At 3 cm
  resolution on a 0.5 m lattice with light lane nodes, a crushed panel shows
  staircase folds. Smooth binding (a vertex to its physics triangle's three
  nodes) would need a loader change.
- Footwell intrusion on pressure wheels (365 mm): the tyre stacks against the
  toe board.
- t_m and b1 are a little stiff-late and stiff; e and dynamic crush are at the
  top of their bands.
- The front fender pole clip (17–34 mm): the fender's structural nodes are light
  (0.26 kg) because its 13 lane nodes take 2 kg of its 5.1 kg.
