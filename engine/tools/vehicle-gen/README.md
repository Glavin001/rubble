# vehicle-gen

Procedural, part-based vehicle models for Rubble's soft-body crash vehicles.
Each design is a Python file; the generator turns it into a `.glb` that
follows the [vehicle art spec](../../../docs/vehicle-art-spec.md) (part names, markers, metres,
+Y up, +Z forward), plus the `.blend` source and studio renders.

**The generator is moving to Rust:** [engine-vehiclegen](../../crates/engine-vehiclegen) is a
line-for-line port (its sedan matches this one part for part) that runs natively and in the
browser, where [vehicle-designer.html](../../crates/engine-webgpu/vehicle-designer.html) rebuilds
the car live as sliders move. Designs not yet ported (Mach 1, 4x4, variants) and the
photo-matching and Blender render tools stay here; `tools/stats.py` + `tools/compare_stats.py`
compare the two generators, `tools/render_glb.py` renders any `.glb` in the studio.

The point of generating rather than hand-modelling: the panel layout, edge
loops, shut lines and markers are exactly what the node/beam builder will
need next, and we know every vertex's surface parameters.

| Vehicle | Design | Output |
| --- | --- | --- |
| 1969 fastback Mach 1, electric blue | [designs/mach1_1969.py](designs/mach1_1969.py) | [engine/scenes/vehicles/mach1_1969](../../scenes/vehicles/mach1_1969) |
| Mid-size four-door sedan (the spec's car) | [designs/sedan.py](designs/sedan.py) | [engine/scenes/vehicles/sedan](../../scenes/vehicles/sedan) |
| Off-road 4x4 SUV | [designs/suv_4x4.py](designs/suv_4x4.py) | [engine/scenes/vehicles/suv_4x4](../../scenes/vehicles/suv_4x4) |
| Police interceptor (from the sedan) | [designs/police_sedan.py](designs/police_sedan.py) | [engine/scenes/vehicles/police_sedan](../../scenes/vehicles/police_sedan) |
| Estate wagon (from the sedan) | [designs/wagon.py](designs/wagon.py) | [engine/scenes/vehicles/wagon](../../scenes/vehicles/wagon) |
| Fastback gasser (from the Mach 1) | [designs/mach1_gasser.py](designs/mach1_gasser.py) | [engine/scenes/vehicles/mach1_gasser](../../scenes/vehicles/mach1_gasser) |

### Variants

A design can start from another and override only what differs
([vehgen/design.py](vehgen/design.py)):

```python
from vehgen.design import derive
derive('sedan', globals())        # the sedan's tables and functions, in this module
NAME = 'wagon'
K[9] = key(0.0, [...])            # a new roof centre line
def rear_z(x, y): ...             # a new tail face
def build_tail(cb): ...           # a new tail cap; the nose stays the sedan's
_sedan_parts = parts
def parts(cb):
    out, extras, mk = _sedan_parts(cb)
    out['roof_rails'] = ...       # extra parts
    return out, extras, mk
```

The base's functions read the variant's globals, so overriding a constant,
key line or function changes everything that uses it. The three variants
above are 90-150 lines each. What a variant cannot easily change: proportions
along the car (z positions are absolute in the base's tables and parts) and
door count.

## Running

Blender 4.3 (its bundled Python has numpy). From this folder:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b --python build.py -- mach1_1969
```

Options: `--out DIR`, `--no-render`, `--samples N`, `--views front34,side,rear34,top`.
A build takes about 30 s plus about a minute of Cycles renders.

Acceptance checks (the spec's automatic list) on a delivered file:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b --python check.py -- ../../scenes/vehicles/sedan/sedan.glb --sedan
```

`--sedan` adds the spec's dimension tolerances, `--doors FL,FR` sets the door
list for a coupe, `--tailgate` expects a tailgate instead of a boot lid,
`--md FILE` writes the table. The Khronos glTF Validator:

```bash
npm install
node validate.mjs ../../scenes/vehicles/sedan/sedan.glb
```

Look-dev helpers: `preview_body.py` (panels only, Workbench or Cycles) and
`render_blend.py` (views of a built `.blend`; `--quick --wire` for wireframe;
`at:x,y,z:dx,dy,dz` aims a close-up at a glTF-space point; a `--views` list
holding one separates views with `;`).

Showcase video: `showcase.py` circles the car once in the studio set and
encodes an mp4 with ffmpeg (frames kept beside it; a rerun resumes).

```bash
VEHGEN_CPU=1 /Applications/Blender.app/Contents/MacOS/Blender -b ../../scenes/vehicles/mach1_1969/mach1_1969.blend \
  --python showcase.py -- out/mach1_1969_showcase.mp4 --seconds 10 --samples 32
```

### Matching photographs

Reference photos are not committed; `refs/<design>/photos.json` lists their
URLs and `fetch_refs.sh <design> <dir>` downloads them. Each view's json
(`refs/mach1_1969/side.json`, `front34.json`, `rear34.json`) pairs photo pixels
with model points (wheel hubs, tyre contacts) and holds `lines` traced off the
photo.

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b ../../scenes/vehicles/mach1_1969/mach1_1969.blend \
  --python compare.py -- <dir>/front34.jpg refs/mach1_1969/front34.json out/f.png --outdoor --lines
```

`compare.py` solves the photo's camera, renders the model from it (side by
side and a 50/50 overlay) and, with `--lines`, draws the model's visible edges
in cyan over the photo with the traced lines in yellow. `tools/zoom.py` crops
and enlarges a photo with a pixel grid for tracing; `tools/backproject.py`
turns pixels into rays from the solved camera (onto a known plane, or at the
depth where a left/right pair is mirror-symmetric) and projects 3D points
back, to place a feature in depth.

### Viewer

`viewer/index.html` shows every built vehicle (the list is
`engine/scenes/vehicles/index.json`, which `build.py` rewrites; or run
`tools/vehicles_manifest.py`): orbit, turntable, explode, parts grouped by
kind with triangle counts (click to isolate, hover to name), markers,
wireframe, paint colour, `?car=<name>`. Serve `engine/` and open it:

```bash
python3 -m http.server 8813 --directory engine     # from the repo root
# http://localhost:8813/tools/vehicle-gen/viewer/
```

The web build (`engine/scripts/build-web.sh`) publishes it at `/models/`
with the models beside it.

### Mesh debugging

`tools/aspect_at.py` lists the worst triangles of a part near a point and
`tools/crossings_at.py` where two parts cross, to trace `check.py` failures
to their cause.

### Agent skills

`.claude/skills/vehicle-from-photos` (the photo-matching procedure) and
`.claude/skills/vehicle-gen-design` (designs, variants, checks, preview) hold
the working procedures for agents; `.cursor/skills` links to them.

## How a car is built

1. **One skin surface.** [tube.py](vehgen/tube.py): every cross-section is a
   G1 chain of cubic Béziers through ten key points on the left half (floor
   centre, sill, door bottom, widest line, waist line, shoulder, belt, roof
   rail, glass edge, top centre). Each key point's x and y is a monotone
   spline of z, so a design is drawn as plan and elevation lines (read off
   reference photos for real cars). Segments are arc-length parametrised.
   Small displacement fields add flares, arch lips, bonnet bulges and sculpted
   sides.
2. **A block layout over it.** [carbody.py](vehgen/carbody.py),
   [layout.py](vehgen/layout.py): z-lines (stations: nose, arches, door
   edges, pillars, tail) cross girth lines (the key points and a few
   fractions). Wheel openings are blocks whose border nodes sit on the arc.
   Nose and tail are caps in their own chart. Every block is a Coons patch;
   chord counts are solved over the whole layout, so neighbouring blocks share
   edge samples and weld exactly, and every character line is an edge loop.
3. **Parts from blocks.** A design's `assign()` names each block's part
   (`door_FL`, `glass_windshield`, `headlight_L_lens` …).
   [panels.py](vehgen/panels.py) welds a part's blocks, relaxes sheared cells
   in the tangent plane, insets borders that meet another panel (a 4.6 mm
   shut line), leaves openings (arches, glass, lamps) where drawn, rounds
   corners, rolls a hem under every border, and adds an inner panel to
   closures. Glass and lenses grow under the opening's flange.
4. **Everything else** comes from [parts.py](vehgen/parts.py) (tyres, rims,
   discs, lamps, mirrors, handles, wipers, seats, wheel wells, pipes …) placed
   by the design's `parts()`, which also returns the spec's markers.
5. [uvpack.py](vehgen/uvpack.py) gives painted panels a non-overlapping UV0
   at one texel density; [blender.py](vehgen/blender.py) makes the objects,
   materials (the spec's ten names) and empties, saves the `.blend`, and
   exports the `.glb` with beauty-triangulated quads.

## Known limits

- **Triangle aspect at multi-panel corners.** A few hundred triangles of
  ~200k exceed 6:1 where three or four panels meet (cowl, pillar feet, lid
  corners) and, on the 4x4, where a girth band is much thinner over the
  bonnet than in the cabin (one tensor layout couples them). The next step is
  chord-breaking transition blocks so a band can change density along the car.
- **A handful of inverted triangles** at the same corners.
- **"No part pairs cross (any depth)"** is our stricter check than the
  spec's 1 mm; it still lists lamp housings touching lenses and similar.
- No textures yet: flat PBR factors only (the spec allows base, ORM and
  normal maps; nothing requires them).
- The 1969 Mach 1 reproduces a real production car's shape (no badges or
  logos). Its proportions are from photographs, not drawings, so expect
  centimetre-level differences from the real car.
