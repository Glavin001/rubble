# Showcase sedan skin: shards, confetti and saw-tooth (2026-11-13)

The page `softbody.html?solver=explicit&car=showcase` (drive yard, raycast
wheels, explicit 2 kHz solver) drew the authored sedan's visual skin after
crashes and node grabs as long blades, a confetti of flipped slivers, holes
and saw-tooth seams. Measured on the CPU reference only (no GPU, no browser):
`engine-softbody/examples/skin_shards_probe.rs`, tests
`engine-softbody/tests/showcase_skin_integrity.rs`. Stills and `.obj` files:
`~/Development/rubble-artifacts/skin-shards/{before,after}/`.

## Method

The car the page builds (`showcaseCar({ wheels: 'raycast' })`: medium sedan,
1,053 nodes, 7,258 beams, 207,271 visual vertices, 345,815 triangles), spawned
as `ExplicitSim::spawn` does (lowest wheel 2 cm up, the page's default
settings: drag and turbulence on). The visual is the page's
(`skin::raycast_visual`, now in Rust; the wasm calls it). What is drawn is
the renderer's: `deform_beams` positions, triangles cut per fragment where
the tear interpolated across them passes ½ (`skin::vertex_tears`, the CPU
mirror of `showcase-car-render.mjs`).

Per drawn triangle: an edge as drawn (only its part where the tear is under
½) longer than 3× its rest length and 5 cm longer is a **shard**. Also
counted: drawn triangles facing away from their rest orientation
(**flipped**), drawn vertices that move more than 3 mean rest edges from the
mean motion of their mesh ring (**saw-tooth**), vertices whose bound nodes
lie in different live-beam components (**spans a break**), nodes farther
than the car's rest radius + 0.5 m from its median centre (**flung**).

## Root causes

### (C/B) The triangle binding broke the skin into independent pieces: the shards, confetti and saw-tooth

Each vertex rode one triangle of three nodes (a skin triangle under it, or
a three-node frame of its part), chosen per vertex, with (α, β, h) fixed at
rest. Neighbouring vertices of one visual triangle rode **different**
binding triangles, so the visual triangle had no consistent map: when the
coarse lattice folded, each vertex followed its own frame's rotation.

| 70 km/h wall, end (1.5 s) | Measured |
|---|---|
| Shard triangles | 7,004 (longest 0.37 m drawn) |
| Of them with all three vertices on one binding triangle | **0** |
| Edge stretch, p99 / max | 16× / 2,300× |
| Flipped drawn triangles | 13,621 |
| Saw-tooth vertices | 3,858 |
| Skeleton stretch over the shards' own bound nodes | 1.2–2.0× |
| Affine vertices outside their binding triangle (extrapolating) | 29.7 % (54.6 % of shard vertices) |
| Vertices more than 3 cm off their triangle (normal offset) | 27.8 % (39.2 % of shard vertices) |
| Vertices on a skin triangle (the rest on node frames) | 58.7 % (38.1 % of shard vertices) |

The skeleton under the shards stretched at most 2×; the skin over it 16×
(p99). So the shards were the binding, not the physics: every one was a
visual triangle across two binding frames, and extrapolated vertices (α, β
outside the triangle, up to 1.2 and −0.6) and offset vertices amplified the
frames' different rotations.

### (B) Binding across parts and across breaks: the long blades toward pulled-off parts

The shut-line vertices rode skin triangles spanning two parts (with an
own-part fallback that switched only when that triangle tore past 2.5× or a
listed beam broke). Pulling the front left door 10 m with the page's grab
(50 kN/m) tore the door off: 2,461 vertices were then bound across the
separated pieces and 737 shards ran from the car toward the door.

### (A) Physics: no node was flung by created energy in the crashes

In the 70 and 90 km/h walls (three hits), the 60 km/h side impact and the
grab pulls, every node far from the car was a legitimately detached piece
lying still (detached suspension and radiator nodes at 0–3 m/s, all their
beams broken), or the grabbed node and its torn-off part. No energy rise
above 50 J a frame in a crash except one of 349 J in the third 90 km/h hit
(against hundreds of kJ dissipated; not investigated further). The
`centred_objective` damping fix is not relevant here: this car runs the
Rigs of Rods `.truck` path with centred damping off.

**Physics findings, separate from the skin:**

1. **The grab at the lab's strongest setting makes the car non-finite.**
   The slider (`crash-lab.mjs`) reaches 10^6.5 ≈ 3.16 MN/m. The front
   bumper's outermost node weighs 0.23 kg: h·√(k/m) = 1.86 (h = 0.5 ms) for
   the grab's spring alone, past the explicit step's stability once its beams
   add theirs. In the first frame every node goes non-finite. Test
   `grab_at_the_strongest_setting_keeps_the_car_finite` (ignored, known
   failure). The fender at 3.16 MN/m does the same. Fixing it needs a
   decision: an explicit limit on the grab's stiffness per node mass that
   fails with a clear error, or an implicit grab spring (a solver change).
2. **Damper-only beams draw as long low-strain beams.** 876 of the 7,258
   beams are body dashpots with k = 0 (the BX node damping pattern), all
   within one part. A node pulled out of its panel stays joined to it by
   these (they never break by stretch: they carry only c·v), and the lab's
   strain colour (k·|ℓ − L| over strength) draws them blue-grey at any
   length: the "metre-long blue beams" of the screenshots. Physically a
   dashpot whose material has gone should go with it (break with its part's
   structure); that is an authoring change to the car's physics and the
   owner's decision. The lab could colour k = 0 beams by their geometric
   strain instead (render only).

## The fix: node-frame skin (`authoring/node_frames.rs`)

Two levels, continuous by construction:

1. **Per node** a frame from its own part's neighbourhood (shape matching
   over the nodes within its kernel radius, `flexskin`'s radii), gated by
   each neighbour's tear and by whether the pair has pulled apart. Smooth
   across nodes: neighbouring nodes' frames differ smoothly even where the
   lattice folds.
2. **Per vertex** a linear blend of its 8 nearest own-part nodes' frames
   (weights continuous in the rest position: a node enters the set exactly
   where its weight is zero). A vertex is cut (the renderer's tear) where a
   quarter of its support has torn away or its nodes disagree on where it is
   (structure folded or crushed within its reach). Rigid pieces take their
   centroid's weights; structure-group pieces blend the group nodes they
   were bound to.

The rest pose and any rigid motion are exact (4.8 µm; normals 2·10⁻⁶).

### Before → after (CPU, the page's car)

| Scene | Shards | Longest | Stretch p99 / max | Flipped | Saw-tooth |
|---|---|---|---|---|---|
| Wall 70, end | 7,004 → **0** | 0.37 → 0 m | 16 / 2,300 → 1.22 / 2.1 | 13,621 → 443 | 3,858 → 0 |
| Wall 90 × 3, end | 16,649 → **1** (5 cm) | 0.52 → 0.05 m | 34.7 / 4,677 → 1.31 / 3.1 | — → 657 | — |
| Side 60, struck | 5,662 → **0** | 0.75 → 0 m | 11 / 2,882 → 1.11 / 2.3 | → 210 | |
| Side 60, striker | 5,641 → **0** | 0.54 → 0 m | 9.9 / 1,116 → 1.12 / 1.8 | → 113 | |
| Grab door 10 m (50 kN/m), held | 881 → **0** | 0.27 → 0 m | → 1.5 max | | |
| Grab boot lid 2 m (50 kN/m), held | 1,110 → **0** | 0.76 → 0 m | 1,879 max → 1.2 | | |

(The test thresholds count a shard as 3× and +5 cm; the probe's "wall 90 × 3
end" 5 cm shard sits exactly at the bound and the test's own measurement
points pass.) Every test scene: 0 shards, 0 drawn triangles across a break,
0 saw-tooth vertices, flipped share under the skeleton's own folded share +
0.5 %.

**What it costs to look at:** in the heaviest crush (90 km/h three times)
the skin smooths a crumple into rounded bulges (the node frames are smooth
over about 0.4 m), and opens holes where the structure folded or crushed past
what one sheet can follow (drawn triangles 304,106 → 292,362 of 345,815). It
never draws a shard there. Finer crumples need a finer lattice (or helper
nodes) in the car, not a sharper binding.

## How node–beam games keep flexbodies sane (public docs, read 2026-11-13)

Sources (public documentation only; clean-room, no code):
- BeamNG flexbodies: <https://documentation.beamng.com/modding/vehicle/sections/flexbodies/>
- BeamNG common issues: <https://documentation.beamng.com/modding/vehicle/intro_jbeam/common_issues/>

What they say:
- Each vertex is mapped to 3 or 4 nodes: a centre node and VX, VY (and if
  possible VZ) nodes whose lines to the centre are roughly perpendicular: a
  per-vertex local frame, not a weighted average of positions.
- A flexbody lists node groups; it is deformed only by those nodes. The
  ETK's own data binds each mesh to its own part's group (door to door,
  bumper to bumper, bay clutter to the engine bay).
- Bad mappings make "spikes"; the nodes should form a grid with no outliers.
  Errors: VX (only a centre node found: a group of one), VY (no node at
  about 90°). At least 3 nodes per flexbody, not all in a line or a plane;
  no part of the mesh far from its nodes; helper nodes may exist only for
  the flexbody.
- Deform groups switch a mesh's material when beams deform or break (glass,
  lamps); they are not tearing. Break groups, `disableMeshBreaking` and
  `disableTriangleBreaking` exist; their exact mesh-breaking rule is not in
  the pages above.

How our bindings compare:
- **Own group only.** The triangle binding broke it at shut lines (skin
  triangles spanning two parts, with a fallback that switched late): the
  blades toward a pulled-off door. The node-frame skin uses only the
  vertex's own part (or its structure group's nodes) by construction.
- **Per-vertex three-node frame.** The triangle binding was this (no VZ), and
  it is exactly what produced the shards here: neighbouring vertices on
  different frames, a third of them extrapolating. **The node-frame skin
  deliberately deviates from it** (a smooth field of per-node frames, each
  vertex a continuous blend of eight), because a per-vertex frame choice is
  discontinuous between neighbours by construction.
- **Lint equivalents** of VX/VY (≥ 3 non-degenerate nodes near every vertex,
  a reach limit, no outliers) are not yet enforced as a lint; see "Open".

## Binding lint (`authoring/skin_lint.rs`, `tests/skin_lint.rs`)

Measures of a car's skin binding, with no solver: static (A) and exact probe
deformations of the nodes (B). Run on every authored car; the generator's
cars should run it the same way.

| Measure | Definition | Threshold (hard) | Node frames | Triangle binding |
|---|---|---|---|---|
| A1 group purity | vertices with a driver (a node its position depends on directly) outside its piece's part (or its structure group's nodes) | 0 (BeamNG: a flexbody is deformed only by its own groups) | **0** | 53,089 |
| A3 weight norm (static spike predictor) | a vertex's Σ\|w\| over its drivers (a triangle binding: \|1 − α − β\| + \|α\| + \|β\|); 1 for a convex blend, above where it extrapolates | ≤ 1.001 (unorm16 and f32 rounding) | **1.00005; 0 over** | 12.6; 51,167 over |
| B3 gain (spike predictor) | per probed node (every 7th, 1 cm along x, y, z): a vertex's displacement over the node's; max and p99 | ≤ 1.5 (a blend of rigidly moved frames gives 1 + the frames' turn × the vertex's offset; past 1.5 it extrapolates) | **1.10** (p99 1.00) | 5.66 (p99 1.31) |
| A6 coherence (saw-tooth predictor) | mesh edges whose two vertices share no driver and no beam joins their drivers | 0 | **0** | 63 |
| A7 group health (VX/VY) | parts under 3 nodes or on a line | none | none | none |
| B1 locality | one part's nodes moved rigidly alone: vertices elsewhere that move (summed over parts); its own vertices' error from the rigid motion | 0; ≤ 0.1 mm | **0; 1.2 µm** | 201,030; 10 mm |
| B4 separation | one part moved 1 m away: drawn triangles with an edge 1 cm longer than at rest (summed over parts) | 0 (nothing bridges parts) | **0** | 6,042 |
| A4 reach (informational) | distance to the nearest driver over the part's median node spacing | reported, not enforced (B3 bounds its effect) | max 7.5; 9,919 vertices past 3 | max 6.5; 3,928 |

A4 is reported, not a rule: a far vertex amplifies only through its frame's
turn, which B3 measures directly. Its worst vertices are interior and bay
pieces far from their part's nodes: a generator or authoring follow-up (helper
nodes or a denser lattice there), not something the binding can fix.

Mutation tests (`the_lint_catches_broken_bindings`): a vertex driven by
another part's node (A1, B1, B4 catch it), a vertex extrapolating with weights
3 and −2 (A3: Σ|w| = 5; B3 does not see it, since one node's probe moves the smooth node frames almost alike), a vertex driven by its part's farthest node (A6), a
part's nodes collapsed onto a line (A7).

Not yet measured: the ETK through our loader (calibration), the generated
sedan (`feat/sedan-body`), B2 affine and bend reproduction, the A2 frame
angle and determinant measures (they apply to per-vertex three-node frames;
the node-frame skin's equivalent is B3).

## Open

- The GPU renderer (`showcase-car-render.mjs`: `nodeFrames`, `blendVertex`)
  mirrors `node_frames::eval`; its WGSL validates with naga, but it has not
  been run on a GPU. Check `softbody.html?solver=explicit&car=showcase`
  (drive yard), crash into the wall and the parked cars, grab panels; and
  `?bind=triangles` for the A/B (URL-only measurement switch, as `?bind=legacy`).
- The binding lint's remaining items (above).
- The generated sedan (`feat/sedan-body`) is not measured.
