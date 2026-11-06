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
