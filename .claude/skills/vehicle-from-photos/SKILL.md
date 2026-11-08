---
name: vehicle-from-photos
description: Recreate a real car accurately from reference photographs with engine/tools/vehicle-gen - gather references, solve each photo's camera, trace the photo, overlay the model's edges on it, place features in depth with rays and symmetry, and iterate coarse to fine until the traced lines land. Use when asked to model a specific real vehicle (by name or from supplied images), to make a generated car "look like the real one", or to check a model against photos.
---

# Recreating a real car from photos

The generator is `engine/tools/vehicle-gen` (read its README first: how a
design is built, the art spec, `check.py`). This skill is the accuracy loop on
top of it. It was learned on `designs/mach1_1969.py`; its refs are the worked
example (`refs/mach1_1969/*.json`).

The rule that matters most: **judge only from overlays, never from side-by-side
looks.** Side by side, a parallelogram window passes for a trapezoid and a
headlamp 0.17 m too far forward looks fine. Overlaid lines do not lie.

## 0. Ground rules

- Large or long-lived outputs (photos, comparison renders, frames) go in the
  gitignored `engine/tools/vehicle-gen/out/`, never a /tmp scratchpad (restarts
  wipe it). Photos are not committed: their URLs are, in `refs/<car>/photos.json`,
  and `fetch_refs.sh <car> out/refs/<car>` downloads them.
- Blender is `/Applications/Blender.app/Contents/MacOS/Blender -b`; its python
  (`.../Resources/4.3/python/bin/python3.11`) has numpy, system python3 may not.
- GPU renders: one job at a time, behind a kill timer
  (`perl -e 'alarm shift; exec @ARGV' 300 Blender ...`). If the GPU is busy or
  wedged, `VEHGEN_CPU=1` renders on the CPU.
- Trade dress: no badges, lettering or logos; say in the car's README that the
  shape reproduces a real car.

## 1. References

- One physical car, photographed from every side: auction galleries
  (Barrett-Jackson, Mecum) give 30+ high-resolution shots of one car. Want:
  near-square side, front 3/4, rear 3/4, head-on, rear-on (top if any).
- Published dimensions: wheelbase, length, width, height, track, tyre size.
  Dimensions set scale; photos set shape. If they disagree (a photo's nose is
  shorter than the published length), trust the photo for shape and say so.
- If the user only names a car, find the gallery and spec sheet yourself and
  show the user the chosen references before modelling.

## 2. Solve each photo's camera

For each view write `refs/<car>/<view>.json`:

```json
{"pairs": [[u, v, x, y, z], ...],          // photo pixel -> glTF point
 "prior": {"cam_y": [0.9, 10], "f": [3500, 10]},
 "lines": {"name": [[u, v], ...]}}         // traced lines, step 3
```

Pairs are points known in 3D from the dimensions: wheel hub centres
(`±track/2, tyre_d/2, ±wheelbase/2`) and tyre contact points (`y = 0`), four to
six of them. `compare.py` solves position, orientation and focal length
(Levenberg-Marquardt from a ring of starts) and renders the model from that
camera:

```bash
Blender -b <car>.blend --python compare.py -- out/refs/<car>/<view>.jpg refs/<car>/<view>.json \
  out/compare/<view>.png --outdoor --paint r,g,b --lines
```

Accept the solve at a few pixels RMS (the Mach 1: side 1.7 px, 3/4 views 4-6 px).
Priors keep weak views (side views: depth vs focal length) from drifting. Read
pixel coordinates off `tools/zoom.py` crops (3x, 10 px grid), never by guessing
from a thumbnail.

## 3. Trace and overlay

Trace into the view's `lines`: window openings, the A-pillar, shut lines, the
bonnet line, lamp outlines (circles as 16-point polygons), grille surround,
bumper top and bottom, chin, tail panel, side markers. `--lines` writes
`<view>_lines.png`: the model's visible edges (Freestyle, cyan) over the photo
with the traces (yellow). `outline.py` does the same for glass borders and
chosen tube lines. Zoom into the result with `tools/zoom.py`; a 3x crop with
the grid makes 2-3 px misses obvious.

## 4. Coarse to fine, one question per view

1. Side view: overhangs, roofline, belt line, glass shape (trace it), wheel
   openings, bonnet and boot lines. Fix the key-line tables (`K[...]`) and
   z-line overrides until the side traces land.
2. Head-on / rear-on: widths. Measure lamp centres, grille width, body edges
   in pixels, convert with the body width; check how square the plan is.
3. 3/4 views: depth. This is where flat-end assumptions fail.
4. Features last: lamps, grille, bumpers, markers, trim, wheels.

## 5. Depth: rays, planes and symmetry

One photo cannot separate "further back" from "further out".
`tools/backproject.py` turns pixels into rays from the solved camera
(`<view>.png.cam.json`):

```bash
python tools/backproject.py out/compare/r.png.cam.json z -2.30 775 340 971 340   # rays onto z = -2.30
python tools/backproject.py out/compare/r.png.cam.json mirror 775 340 971.7 340 -2.6 -2.0
python tools/backproject.py out/compare/f.png.cam.json project 0.79 0.62 2.03   # 3D -> pixel
```

- Put a feature on a plane you know (a measured height, the centre line, a
  measured width from the head-on photo) and solve the rest.
- `mirror`: the depth where a left/right pair (lamps, exhaust tips) is
  symmetric about x = 0. This found the Mach 1's tail panel 90 mm inside the
  tail's outline, and with `project` the outer headlamps 165 mm deep in tunnels.
- Avoid rays nearly parallel to their plane (a camera at about the plane's
  height): ill-conditioned; use another plane or photo.
- Every feature gets confirmed in two views before it is "done".

## 6. Guard the mesh on every change

After each geometry edit: rebuild, `check.py`, and compare with the previous
`checks.md` (counts and locations, not just pass/fail). See the
`vehicle-gen-design` skill for reading and fixing failures. A visual fix that
adds slivers or crossings is not finished.

## 7. Done means

- every view's camera solve within a few px;
- every traced line on the model's edges within a few px (zoomed);
- spec checks no worse than before, the Khronos validator clean;
- renders and the turntable (`showcase.py`) judged next to the photos;
- the car's README states what still differs (tyre lettering, stripes,
  emblems left out, lighting) and how it was checked;
- refs (`photos.json`, view jsons with traces) committed so the next session
  can re-run every comparison.
