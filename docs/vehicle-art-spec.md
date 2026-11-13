Delivery spec · v1 · 2026-11-06

# Rubble Vehicle Art Spec

How to model a car for Rubble, a real-time soft-body crash engine. Your mesh is what people see, and our tools build the crash physics from it, so how the model is split and named matters more than its polygon count.

Deliverable: one .glb per vehicle, its source file, and turntable renders.

## How your model is used

Rubble simulates a car as a few hundred heavy points (nodes) joined by springs (beams) that bend, stay bent and break. From your model we generate:

- the node lattice, placed on your panel edges and character lines;
- the hinges, latches and bolts that hold each part on, taken from the markers you place;
- the binding from every vertex to the nodes, so the visible surface follows the crash smoothly.

Three properties of the model therefore decide the crash quality:

- **Separate parts:** anything that can come off must be its own object.
- **Even, fairly dense topology** on panels, so they can crease.
- **Accurate markers** for wheels, hinges and latches.

## The car

A generic modern mid-size four-door sedan of the Accord or Camry class. The design must be your own, with no real make's styling, badges or logos.

| Dimension | Target | Tolerance |
| --- | --- | --- |
| Overall length | 4,880 mm | ±50 mm |
| Overall width (excluding mirrors) | 1,860 mm | ±30 mm |
| Height | 1,450 mm | ±30 mm |
| Wheelbase | 2,830 mm | ±20 mm |
| Track, front and rear | 1,600 mm | ±20 mm |
| Front overhang | about 970 mm | ±50 mm |
| Ground clearance | 140 mm | ±15 mm |
| Tyre | 235/45 R18, outer diameter 683 mm | ±5 mm |

## File format and conventions

| Item | Requirement |
| --- | --- |
| Format | glTF 2.0 binary (`.glb`), one file per vehicle. Also send the source file (`.blend` preferred). |
| Units and axes | Metres, glTF standard axes: +Y up, the car's front facing +Z. |
| Origin | On the ground (y = 0), centred left to right, midway between the axles. |
| Transforms | Apply all transforms. No negative scale. No unapplied modifiers; subdivision is baked in. |
| Stance | Tyres touch y = 0 at static ride height, wheels straight. |
| Hierarchy | Flat. Every part is a top-level object under one root named `vehicle`; markers are empties under the same root. |

## Parts and names

Use these names exactly. FL, FR, RL and RR mean front-left, front-right, rear-left and rear-right, from the driver's seat. Each part is a separate object with its own mesh. Mirror pairs are modelled separately, not instanced with negative scale.

| Group | Object names | Notes |
| --- | --- | --- |
| Body | `body_shell` | Roof, pillars, sills, quarter panels and the outer floor pan, as one object. |
| Closures | `hood`, `trunk_lid`, `door_FL`, `door_FR`, `door_RL`, `door_RR` | Each with an outer skin and a simple inner panel. |
| Bolted panels | `fender_FL`, `fender_FR`, `bumper_front`, `bumper_rear`, `grille` | Bumper covers are separate from any structure behind them. |
| Glass | `glass_windshield`, `glass_rear`, `glass_door_FL`, `glass_door_FR`, `glass_door_RL`, `glass_door_RR`, `glass_quarter_L`, `glass_quarter_R` | Single-sided, about 5 mm thick, or a thin closed shell. |
| Lamps | `headlight_L/R` and `taillight_L/R`, each split into `_lens` and `_housing` | Lens and housing are separate so the lens can shatter on its own. |
| Mirrors and trim | `mirror_L`, `mirror_R`, `handle_FL` … `handle_RR`, `wiper_L`, `wiper_R` |  |
| Wheels | `rim_FL` … `rim_RR`, `tyre_FL` … `tyre_RR`, `brake_disc_FL` … `brake_disc_RR` | Each centred on its `wheel_center_*` marker. |
| Interior | `dash`, `steering_wheel`, `seat_FL`, `seat_FR`, `seat_rear`, `console`, `door_trim_FL` … `door_trim_RR`, `headliner` | Visible through the glass and when the car is crushed. Medium detail. |
| Under the bonnet and below | `engine_block`, `radiator`, `firewall`, `underbody`, `exhaust`, `wheel_well_FL` … `wheel_well_RR` | Simple, believable forms. Seen when the bonnet opens or the front crushes. |

## Markers (empties)

Place empties (glTF nodes without a mesh) at the real mounting points. We use them to build the hardware that breaks. Accuracy within 10 mm is fine.

| Marker | Where |
| --- | --- |
| `wheel_center_FL` … `wheel_center_RR` | The hub centre. Its local X axis points along the axle, outward. |
| `hinge_door_FL_upper`, `hinge_door_FL_lower` (and the same for each door) | The hinge pivots on the door's front edge. The rear doors hinge on the B-pillar. |
| `latch_door_FL` (and the same for each door) | The striker on the trailing edge, at mid-height. |
| `hinge_hood_L`, `hinge_hood_R`, `latch_hood` | The hinges at the cowl; the latch at the front centre. |
| `hinge_trunk_L`, `hinge_trunk_R`, `latch_trunk` | The hinges under the rear window; the latch at the rear centre. |
| `bolt_fender_FL_1` … (3–5 per fender), `bolt_bumper_front_1` … (4–6 per bumper), `mount_mirror_L/R` | Where the part is screwed to the body. |
| `seat_H_point_FL`, `seat_H_point_FR` | The occupant's hip point. We measure crash intrusion relative to it. |

## Topology and quality

| Rule | Requirement | Why |
| --- | --- | --- |
| Panel edge length | 15–30 mm on body panels, closures and bumpers, evenly spaced. Smaller only where curvature needs it. | A panel can only crease where it has vertices. Long thin triangles fold into stripes. |
| Triangle shape | Quad-dominant before triangulation. Exported triangles have an aspect ratio of at most 6:1. No slivers or zero-area faces. | Slivers flip and flicker when bent. |
| Manifold | No non-manifold edges, duplicate faces or stray vertices. Each part's surface is closed or has clean open borders. | Binding and normal recomputation need a clean surface. |
| Normals | Outward and consistent. Smooth by default, with hard edges only on real sharp features. | We recompute normals as the panel bends. |
| Panel edges | Model a hem: the skin rolls over 6–10 mm at every closure edge. Shut-line gaps of 3–6 mm between neighbouring parts. | Real gaps let parts separate cleanly. Hems keep torn edges from looking paper-thin. |
| Thickness | Outer skin single-sided plus the hem. Closures also have a simple inner panel 20–60 mm behind the skin. | Crushed panels show depth instead of empty shells. |
| Intersections | No two parts interpenetrate by more than 1 mm at rest. | Anything that overlaps at rest snags in contact. |
| Character lines | Put an edge loop along every crease and feature line: shoulder line, bonnet creases, arch lips. | We place nodes on these lines, so the crash folds where the design does. |

### Triangle budget

| Group | Triangles |
| --- | --- |
| Body, closures, bumpers, fenders | 150,000–300,000 |
| Each wheel (rim, tyre, disc) | at most 15,000 |
| Interior | at most 80,000 |
| Lamps, glass, trim, under-bonnet | at most 60,000 |
| **Total** | **at most 500,000** |

## Materials and textures

- **Shading:** glTF PBR metallic-roughness only. No custom shaders and no baked lighting.
- **Material names** (one each, shared across parts):
  - `paint`: a single colour that we can tint;
  - `chrome`, `trim_black`, `rubber`, `glass`, `lamp_lens`;
  - `lamp_emissive`, `interior_fabric`, `interior_plastic`, `metal_bare`.
- **UVs:**
  - UV0 is non-overlapping across all painted panels, at even texel density. We use it for scratches, dirt and dent shading.
  - Tyre sidewalls, lamps and the interior may use their own textures.
- **Textures:** PNG or JPG, at most 2048 × 2048, packed as base colour, ORM (occlusion, roughness, metallic) and normal maps. Total texture memory at most 64 MB uncompressed.
- **Glass:** transparent through base-colour alpha (BLEND) or `KHR_materials_transmission`, at roughness 0.05 or less.

## What to deliver

1. `sedan.glb`, following this spec.
2. The source file (`.blend`, or FBX with the textures).
3. Four renders in neutral studio light: front three-quarter, side, rear three-quarter, and top.
4. A short `README` giving the measured dimensions, triangle counts per group, and the licence.

**Licence and originality.** The model must be your own original work, and we need rights to use, modify and redistribute it in an open-source engine (CC-BY 4.0 or an assignment to us). Don't use meshes or textures taken from games (BeamNG, Rigs of Rods or others), from model stores whose licences forbid redistribution, or from real manufacturers' CAD.

## Acceptance checks

We run these automatically on delivery, then review the car visually in our renderer.

- The Khronos glTF Validator reports zero errors.
- Every part and marker in the tables above is present with its exact name.
- All dimensions are within tolerance. Tyres touch y = 0, and each wheel's centre is within 5 mm of its marker.
- No non-manifold edges, zero-area triangles or inverted normals on any part.
- On panels, the 95th-percentile edge length is 30 mm or less, and the triangle aspect ratio is 6:1 or less.
- No part pair interpenetrates by more than 1 mm. Shut-line gaps are 3–6 mm.
- Triangle counts and texture memory are within budget.
- Materials use only the names listed. UV0 has no overlaps on painted panels.
- Rest fidelity: after we build the physics and bind the mesh, every vertex is within 1 mm of your model at rest.
- Visual review: rendered in Rubble's browser page under its studio lighting, beside our reference car, from the four angles. It looks like a real, well-made car.

## Questions you may have

### Do I need to model the structure (rails, pillars inside)?

No. We generate the structure. The inner panels, firewall and wheel wells above only need to look right when exposed.

### Should parts be thick solids?

No. Use a single-sided outer skin plus a hem at the edges, and an inner panel on closures. Solid thick shells double the vertex count without helping.

Our own generator goes further: it builds every panel and pane as a closed solid (outer skin, edge walls, inner face) and tags each vertex, so a part looks whole on its own while the physics follows the outer skin only. See [engine-vehiclegen, "Solid parts"](../engine/crates/engine-vehiclegen/README.md#solid-parts). A delivered model does not have to.

### Can I go denser than 15 mm?

Only where the curvature needs it, such as tight arch lips or lamp bezels. Very dense panels cost memory at scale (hundreds of cars) without improving how they bend.