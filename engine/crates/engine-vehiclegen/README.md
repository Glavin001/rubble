# engine-vehiclegen

Procedural, part-based vehicle models to the [vehicle art spec](../../../docs/vehicle-art-spec.md),
in Rust: the same generator as the Python [vehicle-gen](../../tools/vehicle-gen) (ported line for
line), now runnable natively (CLI, CI) and in the browser (WebAssembly), fast enough to rebuild a
whole car while a slider moves.

| | |
| --- | --- |
| [`tube`](src/tube.rs) | the body skin S(z, v): G1 Bézier sections through ten key lines |
| [`layout`](src/layout.rs), [`carbody`](src/carbody.rs) | block-structured quad mesher: z-lines × girth lines, wheel arches, end caps, chord counts solved by union-find |
| [`panels`](src/panels.rs) | shut-line insets, rolled hems, inner panels, glass and lenses from openings |
| [`solid`](src/solid.rs) | every panel and pane as a closed solid: skin, edge walls, inner face; surface tags; closedness, thickness and crossing checks |
| [`parts`](src/parts.rs) | tyres, rims (alloy, steel, five-window), discs, lamps, handles, mirrors, wipers, seats, steering wheel, wheel wells, pipes, mouldings |
| [`assemble`](src/assemble.rs) | `Design` trait, body + parts -> named parts and markers |
| [`frame`](src/frame.rs) | proportions: maps a design's reference car onto a new wheelbase, overhangs, width, ride and roof height, keeping wheel arches round |
| [`uvpack`](src/uvpack.rs), [`export`](src/export.rs) | UV atlas; render buffers (42° split normals) and glTF 2.0 binary |
| [`designs::sedan`](src/designs/sedan.rs) | the spec's mid-size sedan, parametric (`SedanParams`) |

## Use

```bash
cargo run -p engine-vehiclegen --release --bin vehiclegen -- sedan out.glb            # the reference sedan
cargo run -p engine-vehiclegen --release --bin vehiclegen -- sedan out.glb p.json     # with parameters
```

`p.json` holds any subset of `SedanParams` (metres; rims in inches; paint in linear RGB):
`{"wheelbase": 2.55, "front_overhang": 0.8, "width": 1.78, "height": 1.42, "ride": 0.0, "tyre_d": 0.64, "rim_in": 16, "rim_style": "steel"}`.
Out-of-range values are refused with a message (`SedanParams::validate`).

In the browser: [`engine-vehiclegen-wasm`](../engine-vehiclegen-wasm) (`build_vehicle(design,
params_json, uv)` -> per-part buffers, markers, `to_glb()`), used by
[`vehicle-designer.html`](../engine-webgpu/vehicle-designer.html) in a worker.

## Solid parts

Every body panel, pane, lamp lens, door card, wheel-well liner and the headliner is a closed
solid, not a sheet ([`solid`](src/solid.rs)), so a part looks whole on its own (a door open, a
wing torn off). A panel is:

- its **outer skin**: exactly the sheet build's skin (same vertices, same order);
- an **inner face**: the skin's own topology pushed below it by a depth field (one inner vertex
  per skin vertex). The field gives each kind of part its features: a door's frame, recessed
  centre, belt reinforcement and beam (to 50 mm); a bonnet's or boot lid's frame, spine, cross
  member and braces with pockets where the skin's back shows (9–34 mm); a wing or bumper cover
  as a 5 mm pressing; the body shell as a 30 mm box-section wall (14 mm floor pan);
- **edge walls** round every border, by what the border is: the rolled hem and a shut face at a
  shut line (a return flange on pressings), a rebate the glass sits in at a glazed opening, a
  rolled lip and a return towards the liner at a wheel arch.

Glass is a 4.5 mm slab, a lens a 3 mm moulding drawn in one direction. Every vertex carries a
tag (`mesh::Surf`: `Outer`, `Edge`, `Inner`): on `Mesh::surf`, on `RenderPart::surf`, in
`RigSpec::surfaces` (`assemble::rig_spec_with`) and in the `.glb` as the vertex attribute
`_SURFACE` (0, 1, 2). `Mesh::outer_only` / `assemble::outer_parts` give the skins back.

```bash
cargo run -p engine-vehiclegen --release --bin vehiclegen -- sedan out.glb --solids   # closed? volume, thickness, crossings per part
cargo run -p engine-vehiclegen --release --bin vehiclegen -- sedan out.glb --sheet    # the legacy sheets
Blender -b --python tools/vehicle-gen/tools/render_parts.py -- out.glb out/parts      # isolated and exploded stills
```

`tests/build.rs` checks every solid: no open or non-manifold edge, every piece a positive
volume (faces outward), wall thickness in a band per part, no face cutting through another
(known exceptions pinned as ceilings, below), the art spec's triangle budgets (body 278,856 of
300,000; lamps, glass and trim 57,596 of 60,000; 442,448 in all), and the designer's five presets.

Known, not yet fixed:

- Wall rows are thin where a wall is shallow or turns a tight corner: about 4,100 panel
  triangles are past the art spec's 6:1 aspect (the sheet build has 600).
- The headlamp lens crosses itself where it wraps the nose's corner (its sheet is creased there;
  about 80 edge crossings a side), the rear door cards behind the wheel arch (about 85), and a
  few wall rows at lamp-opening and screen corners (body shell 19, front doors 6, wings and front
  bumper 4 each).
- Panes and lenses touch their rebates at corners (`check.py` "no part pairs cross": 24 pairs,
  the sheet build 10).
- Lamp reflectors and the grille's back plates are still sheets. The body has no B-pillar (the
  doors meet over it), so an open door shows the cabin, not a pillar.

## The legacy sheets and the Python generator

`BuildOpts { solid: false, .. }` (CLI `--sheet`) builds the parts as sheets: a single-sided
skin, a rolled hem and, on closures, a loose inner panel. That build is the Python sedan (the
Python designs are not ported to solids: they are the legacy sheet version): 68 parts with
identical triangle and vertex counts and bounding boxes, vertex sums equal to ~1e-16 per
vertex, all 42 markers in the same places. Check it:

```bash
cargo run -p engine-vehiclegen --release --bin vehiclegen -- sedan /tmp/s.glb --sheet --stats rust.json
Blender-python tools/vehicle-gen/tools/stats.py sedan py.json     # Blender's python3.11 has numpy
python3 tools/vehicle-gen/tools/compare_stats.py rust.json py.json
```

`tests/build.rs` pins the sheet build's counts to the Python's and the solid build's to its own.
`panels` and `parts` carry unit tests whose numbers were produced by the Python functions on the
same inputs.

## Speed

A full solid sedan (442k triangles): ~0.25 s native and ~0.34 s in wasm (Node) without panel UVs
(the live preview); ~0.34 s native with them. As sheets (289k triangles) it is ~0.2 s native
without UVs and ~0.27 s with. The Python takes ~18 s. The speed-ups are lossless (cached key
points and Bézier segments per z, a fast hasher): the output is unchanged.

## Not yet ported

The 1969 Mach 1, the 4x4 and the variants (police, wagon, gasser) are still Python designs; the
photo-matching tools and Blender renders stay in Python, reading the generated `.glb`.
