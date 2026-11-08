---
name: vehicle-gen-design
description: Build, vary and debug part-based vehicle models with engine/tools/vehicle-gen - write a new design or a variant of an existing one with derive(), rebuild and render, run the art-spec checks and fix what they find (slivers from chord coupling, crossings, inverted cap columns), preview in the model viewer, and publish renders, video and the manifest. Use when adding or changing a vehicle in engine/tools/vehicle-gen/designs, when check.py fails after a change, or when asked to preview, render or show a generated car.
---

# Vehicle designs with vehicle-gen

`engine/tools/vehicle-gen` turns a design file into an art-spec `.glb`
(`docs/vehicle-art-spec.md`): named parts, markers, metres, +Y up, +Z forward.
Read its README section "How a car is built" once; this skill is the working
procedure. For matching a real car's photos use the `vehicle-from-photos` skill.

## Commands (from engine/tools/vehicle-gen)

```bash
B=/Applications/Blender.app/Contents/MacOS/Blender
$B -b --python build.py -- <design>                    # glb + blend + renders into engine/scenes/vehicles/<design>/
$B -b --python build.py -- <design> --out out/x --no-render   # quick iteration, no renders
$B -b --python check.py -- ../../scenes/vehicles/<d>/<d>.glb [--sedan | --doors FL,FR | --tailgate] --md ../../scenes/vehicles/<d>/checks.md
node validate.mjs ../../scenes/vehicles/<d>/<d>.glb    # Khronos validator (nvm use first)
$B -b <d>.blend --python render_blend.py -- out/look/<d> front34 "at:x,y,z:dx,dy,dz:lens" --samples 32
VEHGEN_CPU=1 $B -b <d>.blend --python showcase.py -- out/showcase/<d>.mp4 --seconds 10   # turntable video
python3 tools/vehicles_manifest.py                     # viewer list (build.py runs it)
```

Several `at:` views in one `--views` list are separated by `;`. GPU renders run
one at a time behind a kill timer (`perl -e 'alarm shift; exec @ARGV' 300 ...`);
`VEHGEN_CPU=1` when the GPU is busy. Outputs worth keeping go in `out/`
(gitignored) or `~/Development/rubble-artifacts/` (videos), never /tmp.

## A new car: copy the closest design, or derive from it

Prefer a variant ([vehgen/design.py](../../../engine/tools/vehicle-gen/vehgen/design.py)):

```python
from vehgen.design import derive
derive('sedan', globals())       # runs the base design in this module
NAME = 'wagon'; PAINT = (...)    # constants
K[9] = key(0.0, [...])           # key lines (roof, belt, glass edge ...)
def rear_z(x, y): ...            # end faces
_base_assign = assign
def assign(zf, zt, lo, hi, side): ...   # which panel each block becomes
def build_tail(cb): ...          # the sedan splits build_caps into build_nose/build_tail
_base_parts = parts
def parts(cb):
    out, extras, mk = _base_parts(cb); ...; return out, extras, mk
_base_post = post                # (if the base has one) whole-car transforms after assembly
```

The base's functions read the variant's globals, so an override reaches every
use. Worked examples: `police_sedan.py` (equipment only), `wagon.py` (new roof
and tail), `mach1_gasser.py` (re-posed body, new wheels, `PAINT_COAT = 0`). Not
yet easy: proportions along the car (z values are absolute in the base) and
door count; those need a new design file.

## The checks and what their failures mean

Run `check.py` after every geometry change and diff against the previous
`checks.md`. Compare a variant with its base: failures inherited unchanged are
the base's; only new ones are yours.

- **Panel triangle aspect > 6:1** (slivers). Cause is almost always *chord
  coupling*: a block's opposite edges share one chord count, and union-find
  chains that through every block along a band, so one long edge anywhere
  (a steep wall, a tall cap row) multiplies the rows of a band everywhere,
  and a thin band elsewhere (a door frame, a gutter) becomes slivers.
  Locate with `tools/aspect_at.py <glb> <part> x,y,z radius`, then find the
  long edge that couples to that band. Fixes that worked: put a steep wall
  exactly on a cap boundary instead of inside a block (Mach 1 headlamp
  tunnels); keep cap rows a constant distance below the edge they meet, so no
  cap row is taller mid-cap than at its side (wagon tailgate); keep columns
  from crossing (a cap column line must stay inside the body outline:
  `min(xg, x_side - 0.05)`).
- **No part pairs cross**: the check prints each pair with a location;
  `tools/crossings_at.py <glb> a/b` lists both sides' triangle centroids.
  Typical: accessory bases sunk into a curved panel (lift by a few mm),
  coplanar faces between lens and housing (offset 1-2 mm), a pipe through a
  panel (end it behind the panel and put a bezel on the face), lathe profiles
  touching the axis (`r = 0` points give non-manifold edges: start at 2 mm).
- **Shut-line gaps under 3 mm / folded seams**: a shut line running along a
  sharp body edge (a roof's rear edge) folds; move the panel split to a flat
  area (the wagon's roof header).
- **No inverted normals / interpenetration**: usually a few corner faces in
  the base; compare counts with the base before chasing.
- Never fix a check by loosening it, decimating, or skipping a part (AGENTS.md:
  no lossy shortcuts); fix the layout.

## Preview and publish

- Viewer: `engine/tools/vehicle-gen/viewer/index.html`, served from `engine/`
  (`python3 -m http.server 8813 --directory engine`, then
  `/tools/vehicle-gen/viewer/?car=<name>`); the web build publishes it at
  `/models/`. It lists `engine/scenes/vehicles/index.json`, which `build.py`
  rewrites after each full build. Check new cars there: orbit, explode, solo a
  part, markers.
- Each car folder gets: glb + blend (LFS), `renders/`, `checks.md` and a
  README (what it is, base design, rebuild command, checks table, what fails
  and why).
- Commit each verified car; stage explicit files (the main checkout is shared
  by other sessions).
