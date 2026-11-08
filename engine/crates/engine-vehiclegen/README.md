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

## Parity with the Python generator

At the reference parameters the Rust sedan is the Python sedan: 68 parts with identical triangle
and vertex counts and bounding boxes, vertex sums equal to ~1e-16 per vertex, all 42 markers in
the same places. Check it:

```bash
cargo run -p engine-vehiclegen --release --bin vehiclegen -- sedan /tmp/s.glb --stats rust.json
Blender-python tools/vehicle-gen/tools/stats.py sedan py.json     # Blender's python3.11 has numpy
python3 tools/vehicle-gen/tools/compare_stats.py rust.json py.json
```

`tests/build.rs` pins the reference counts. `panels` and `parts` carry unit tests whose numbers
were produced by the Python functions on the same inputs.

## Speed

A full sedan (289k triangles): ~0.2 s native, ~0.24 s in wasm (Node), without panel UVs (the live
preview); ~0.3 s with them. The Python takes ~18 s. The speed-ups are lossless (cached key points
and Bézier segments per z, a fast hasher): the output is unchanged.

## Not yet ported

The 1969 Mach 1, the 4x4 and the variants (police, wagon, gasser) are still Python designs; the
photo-matching tools and Blender renders stay in Python, reading the generated `.glb`.
