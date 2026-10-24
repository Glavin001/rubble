# vibe-land's dune buggy (and its garage catalog)

The whole catalog bakes from here. `bake-physics.mjs <model>` and `bake-mesh.mjs` cover
`buggy`, `trophy`, `rally`, `monster`, `derby`, `sprint` and `semi`; the presets are
`dune_buggy` and `vibe_<model>`. The rest of this page describes the buggy, which is also
the only model with a hybrid (soft-frame) build. The other models differ in:

* `parts.mjs`'s body rules: one ordered part list, each model keeping the parts it has;
* panel colliders: the box fitted to each panel's corner points;
* the semi's trailer: its own vehicle, `vibe_semi_trailer`, whose parts follow its body
  and wheels.

Joint contacts under vibe-land's 1 cm² solver floor are kept in the mount ratings: a
mount is rated as a whole, and a small far contact is what gives it its moment capacity.

vibe-land's garage "Sand rail" (`buggy` in its vehicle catalog), ported to Rubble as the
`dune_buggy` vehicle preset and the `vehicle.html?scene=buggies` demo.

* In vibe-land the buggy is a procedural Three.js recipe: about 930 parts, each tagged
  with its system and suspension role.
* Its physics ran server-side on PhysX Vehicle2 with a Blast-style stress stage. Neither
  ports.
* Rubble rebuilds it from the same recipe:
  * a raycast-wheel vehicle with 17 rigid parts on breakable mounts (`MountDesc`, see
    [`engine/docs/vehicles.md`](../../docs/vehicles.md));
  * the recipe's full mesh, with the suspension posed from the wheels.

Catalogue entry: [`docs/scene-port-catalog.md`](../../../docs/scene-port-catalog.md).

## Layout

| Path | What |
|---|---|
| `src/buggy.mjs`, `live-geometry.mjs`, `vehicle-rig.mjs`, `visual-rig.mjs`, `vehicle-catalog.mjs`, `vehicle-body.mjs`, `vehicle-specials.mjs`, `construction.mjs`, `SOURCE.md` | copied unchanged from vibe-land `client/src/vehicles/dune/` at 344c2d97 (the recipe's own provenance is `SOURCE.md`: supplied by the author as `dune-buggy-source.zip`) |
| `src/strength-profile.mjs`, `src/real-joint-capacity.mjs` | copied unchanged from vibe-land `client/src/vehicles/` (joint strengths; the section bound) |
| `parts.mjs` | which Rubble part, wheel corner and hybrid class every recipe part belongs to (both bakes read it) |
| `bake-physics.mjs` | writes `engine/crates/engine-vehicle/assets/dune_buggy.json` (**committed**: the Rust preset `include_str!`s it) |
| `bake-mesh.mjs` | writes `engine/crates/engine-webgpu/generated/dune-buggy/` (git-ignored; `build-web.sh` runs it) |

The copies are not imported from a vibe-land checkout by path. Rubble builds on its own
in CI and in fresh clones.

```sh
cd engine/scenes/dune-buggy
bun install
bun run bake:physics    # ~40 s (exact CSG build); commit the asset
bun run bake:mesh       # < 1 s
```

## Frames

The recipe is in metres, +Y up, -Z front, with left at x < 0. Rubble vehicles are +Z
forward and +X left.

Every point goes through `R(x, y, z) = (-x, y, -z)`, a 180° turn about Y. It maps the
recipe corners fl, fr, rl, rr onto Rubble's wheel order FL, FR, RL, RR. Steering (+ =
left) keeps its sign. Wheel spin flips: recipe `rotationRad = -W_W.y`.

## Physics bake (`dune_buggy.json`)

The recipe is built twice, part for part:

* The **exact build** (manifold-3d CSG) gives each part's solid volume, and so its mass
  (`construction.mjs` hollows tyres, castings and skins). It also measures the interface
  area between every pair of touching parts.
* The **live build** (Three.js primitives) gives each part's primitive and placement.

**Parts.** `parts.mjs` groups the recipe into 17 Rubble parts. Each Rubble part's mass is
the sum of its members.

| # | Part | Role |
|---|---|---|
| 0 | frame | root |
| 1 | front_bumper | body |
| 2 | nose | body |
| 3, 4 | headlamp_l / headlamp_r | body |
| 5, 6 | mirror_l / mirror_r | body |
| 7 | rear_bumper | body |
| 8, 9 | rear_lamp_l / rear_lamp_r | body |
| 10 | engine | engine |
| 11 | transmission | drive shaft, all four wheels |
| 12 | fuel_tank | body |
| 13, 14 | seat_l / seat_r | body |
| 15 | extinguisher | body |
| 16 | steering_column | steering, front wheels |

The moving suspension parts count toward the frame. Each corner's wheel and hub parts are
its wheel (128 kg).

**Colliders.** Each part's structural members become primitives:

| Recipe primitive | Collider |
|---|---|
| tube, or a cylinder at least as long as it is wide | capsule |
| box | box |
| short cylinder | box |
| sphere | sphere |

These have no colliders:

* Small hardware: fins, pleats, lenses, fasteners, harness webbing.
* The moving suspension and the wheels. The raycast wheels and their shell spheres stand
  in for these.

**Mounts.** Each joint's capacity is its area × the weaker material's fatal strength
(tension or shear), from vibe-land's `strength-profile.mjs`. The area is bounded by the
smaller member's section (vibe-land's `real-joint-capacity.mjs`): two overlapping solids
are joined by a weld, not by their whole overlap.

* A part's mount is the sum over its joints to the frame. A part that touches only another
  part takes that part's joints instead (headlamps on the nose, rear lamps on the bumper,
  the gearbox on the engine).
* The moment limit is the joints' couple about their force centroid, plus half each
  joint's width.
* A wheel's mount is its corner's suspension (arms, tie rod, damper) where it meets the
  frame.

| Mount | Force limit |
|---|---|
| headlamps, rear lamps, extinguisher, steering column, seats | 58–92 kN |
| fuel tank, mirrors | 157–172 kN |
| bumpers, gearbox | 460–760 kN |
| nose, wheels | ~1 MN |
| engine | 1.5 MN |

**Handling** (`presets.rs`), as vibe-land's garage setup:

* springs at 30% static sag, critically damped;
* bump stops ten times as stiff as the springs;
* all-wheel drive at 7.5 m/s², 30 m/s top speed;
* grip 1.4;
* centre of mass 0.2 m low;
* full steering lock to 6 m/s, tapering to 35% at 28 m/s.

## Render bake (`generated/dune-buggy/`)

The bake writes three files:

* `buggy-mesh.bin`: the live build's 50 shared primitives (positions, normals, indices).
* `buggy-mesh.json`:
  * the 927 parts as instances, each with its rest matrix, colour, Rubble part, class,
    corner and suspension role;
  * the rig definition.
* `rig-fixtures.json`: the original `VisualRig`'s matrices at 12 poses.

`crates/engine-webgpu/buggy-rig.mjs` is a dependency-free port of that rig.
`tests/buggy_rig.test.mjs` checks it against the fixtures.

`buggy-render.mjs` draws each primitive once, instanced over every part on every buggy. A
part is drawn in one of these places:

* on the chassis;
* on its own body once its mount broke;
* with its wheel's body once the wheel came off.

The suspension is posed from each buggy's wheel records: travel from the suspension
length, the steer angle, and the spin.

## Hybrid build

`frame-graph.mjs` turns the frame classes into a node–beam structure:

* **Tubes:** tube junction nodes (with T-junction splits); long spans subdivided to 4
  nodes a tube.
* **Braces:** direction-aware, so no node is free to move across its tube.
* **Detachable pieces:** the front bumper, rear bumper and nose get nodes of their own.
  They are tied to the frame by mount beams in one break group each.
* **Mass:** frame-class mass on the nodes.
* **Render bindings:** each frame tube is skinned along its node chain; any other frame
  part rides three nodes.

`bake-physics.mjs` writes it into the asset's `hybrid` section (nodes, beams, the core's
attachment nodes, and the nodes each bolted part attaches to). `bake-mesh.mjs` writes the
render bindings (`instances[].bind`, `frameNodes`).

`engine_softbody::buggy::dune_buggy_soft()` assembles the soft vehicle. The beam
calibration and the table of what bends and what snaps are in
[`engine/docs/softbody.md`](../../docs/softbody.md).

`parts.mjs` classes:

| Class | Parts |
|---|---|
| `cell` | the safety cell: hoop, pillars, roof, side bars, floor rails |
| `crumple` | front frame horns, nose tubes, bumpers, rear stays |
| `panel` | nose panel, floor pans |
| `rigid` | bolted-on parts that snap off |
| `wheel` | wheel and hub parts |
| `rig` | the suspension |
