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

**Superseded (2026-11-14):** the cuts this version made (stretch past 35 %,
a quarter of a vertex's support gated, nodes disagreeing at a fold) drew
dented panels as holes: see "v2: cut only where beams broke" below.

## Video (CPU, side by side: triangle binding left, node frames right)

`~/Development/rubble-artifacts/skin-shards/*_before_after.mp4`, from
`skin_shards_probe film:<scene>` (frames in the worktree's `out/skin-film/`).
The same solver run skinned both ways under one camera each frame; then an
orbit of the settled car, skin and with the skeleton (live beams, green;
yellow → red by plastic strain). The CPU rasteriser shades flat, cuts by the
renderer's tear rule and does not draw the raycast wheels (the page draws them
from the drive state).

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

## v2: cut only where beams broke (2026-11-14)

**What still looked wrong.** The node-frame skin above had no shards, but it
cut triangles for stretch, fold and disagreement, none of which is a tear: in
the 70 km/h wall most of the bonnet front and bumper area was missing, the
struck door in the side hit was half gone, the grabbed fender vanished.
Sheet metal dents; BeamNG deforms the mesh continuously and removes polygons
only where a beam breaks. Measured with the new integrity assertions (below):
side 60, struck, 0.25 s: **7,930 triangles cut with no broken beam under
them** (door_f_l 6,060 of 13,592 drawn; body 128,722 of 140,026).

**The rule now** (`node_frames::Binding::Breaks`, the default; CPU
`eval_breaks`, GPU `nodeComponents`, `nodeFrames`, `blendVertex`):
- A node's frame is steered by the members of its neighbourhood still joined
  to it by **live stiff beams** (k > 0: a dashpot holds no material) within
  the neighbourhood; a member with no such path even at rest stays while its
  part's live stiff beams still join them at all (per-part components).
- A vertex groups its 8 nodes the same way (two stay together unless they
  were joined at rest and no live path joins them now), follows the group
  carrying most of its weight, and is cut (tear = 2 × the far group's weight
  share, cut at ½) only where its support straddles a break. A node left
  without a frame by breaks keeps its weight at its own position, turned as
  its group's framed nodes turn.
- Nothing else cuts. Folds, crushes and stretch are drawn as they are.
- Rest and rigid motion stay exact (test `at_rest_and_under_rigid_motion`).

**New assertions** (`tests/showcase_skin_integrity.rs`): a triangle is cut
only where a broken beam joins two of the nodes carrying one of its vertices,
or its glass shattered; no new boundary edge (hole) beside a cut no break
explains. Every crash and grab scene: **0** of either.

### Four bindings on the same trajectories (`examples/skin_compare.rs`)

Shards (drawn edge > 3× rest and +5 cm), longest, edge stretch p99 / max,
flipped (share of drawn; the skeleton's own folded share), saw-tooth, triangles
uncut, cut with no broken beam. "frames+cuts" is the previous node-frame
version (`Binding::Stretch`); "BeamNG-style" is `bind_beamng` (below).

| Scene | Binding | Shards | Longest | Stretch | Flipped | Saw | Uncut | Cut w/o break |
|---|---|---|---|---|---|---|---|---|
| Wall 70, 1.5 s (154 broken) | triangles | 6,337 | 0.62 m | 18.3 / 2,879 | 4.49 % (1.22) | 3,151 | 90.9 % | 16,817 |
| | frames+cuts | 0 | 0 | 1.21 / 2 | 0.20 % | 0 | 97.9 % | 2,558 |
| | **frames, breaks** | **0** | 0 | 1.22 / 2 | 0.25 % | 0 | **98.9 %** | **0** |
| | BeamNG-style | 3,933 | 0.85 m | 4.98 / 1,073 | 5.66 % | 1,274 | 96.0 % | 0 |
| Wall 90 ×3, end (501) | triangles | 12,444 | 0.65 m | 29.8 / 3,328 | 8.05 % (2.16) | 4,981 | 85.0 % | 20,229 |
| | frames+cuts | 0 | 0 | 1.28 / 4 | 0.29 % | 0 | 82.5 % | 17,912 |
| | **frames, breaks** | **20** | 0.59 m | 1.48 / 5 | 0.43 % | 0 | **97.5 %** | **0** |
| | BeamNG-style | 9,164 | 2.31 m | 8.72 / 683 | 10.6 % | 2,338 | 80.1 % | 0 |
| Side 60 struck, 2 s (114) | triangles | 4,521 | 0.75 m | 11.1 / 2,881 | 2.38 % (0.68) | 1,270 | 94.9 % | 7,994 |
| | frames+cuts | 0 | 0 | 1.11 / 2 | 0.07 % | 0 | 91.4 % | 7,172 |
| | **frames, breaks** | **21** | 0.16 m | 1.26 / 4 | 0.11 % | 0 | **98.6 %** | **0** |
| | BeamNG-style | 3,227 | 1.37 m | 4.13 / 532 | 3.28 % | 1,024 | 96.2 % | 0 |
| Side 60 striker, 2 s (116) | frames+cuts | 0 | 0 | 1.12 / 2 | 0.04 % (1.89) | 0 | 92.0 % | 5,029 |
| | **frames, breaks** | **14** | 0.25 m | 1.32 / 5 | 0.12 % | 0 | **99.5 %** | **0** |
| Grab door 10 m, held (84) | triangles | 466 | 0.23 m | 1.70 / 388 | 0.47 % | 135 | 94.7 % | 15,605 |
| | frames+cuts | 0 | 0 | 1.04 / 1 | 0.00 % | 0 | 98.3 % | 192 |
| | **frames, breaks** | 0 | 0 | 1.08 / 2 | 0.00 % | 0 | **100 %** | 0 |
| Grab bumper_f 2 m, held (8) | frames+cuts | 0 | 0 | 1.00 / 1 | 0 | 0 | 93.9 % | 0 |
| | **frames, breaks** | 0 | 0 | 1.16 / 2 | 0.00 % | 0 | **100 %** | 0 |
| Grab boot lid 2 m, held (31) | triangles | 946 | 0.77 m | 2.20 / 1,860 | 0.66 % (0.41) | 223 | 85.5 % | 13,221 |
| | frames+cuts | 0 | 0 | 1.03 / 1 | 0.00 % | 0 | 87.8 % | 5,891 |
| | **frames, breaks** | **354** | 0.27 m | 1.81 / 24 | 0.20 % | 8 | **97.0 %** | **0** |
| | BeamNG-style | 2,110 | 0.80 m | 4.35 / 327 | 2.64 % | 812 | 92.2 % | 0 |

Full output (every checkpoint, every binding, per part):
`skin_compare` (`LINT=1 PARTS=1`), 4 minutes on the CPU.

**Choice, on these numbers: node frames, cut only at breaks.** BeamNG-style
per-vertex frames (implemented from the public documentation as faithfully
as we can: own part only, centre = the vertex's nearest part node, VX/VY/VZ
among its stiff-beam neighbours chosen nearest a right angle, every vertex of
a centre sharing its frame, three nodes + normal where flat) make 3,000–9,000
shards a crash: neighbouring vertices on different centres' frames, and the
affine coordinates extrapolate (A3 Σ|w| up to 28.9 on 147,340 vertices). On
this lattice (spacing 10–20 cm under 1–3 cm visual triangles) a per-vertex
frame choice is discontinuous between neighbours by construction; the blend
is not. Kept as `Binding::Beamng` (CPU, measurement only).

**What is still wrong (honest coarse-lattice state, no break under it).**
Shards remain where the lattice folded or crushed with every beam intact:
- body (underbody and rear floor) in the 90 km/h walls: 10–20 shards up to
  0.6 m; the underbody's visual triangles are 13–28 cm, across nodes whose
  predictions differ by 0.2 m in the crush;
- the struck rear door (side 60): 20–21 shards up to 0.2 m;
- the body round the boot (lamp bar, plate surround, rear bumper pieces) when
  the boot lid is pulled 2 m and drags the rear panel: 354–434 shards up to
  0.27 m, saw-tooth 8–26.
Each is a fold of a coarse lattice blended linearly (two node groups rotated
by a large angle against each other; a vertex between them takes a weighted
mix of their predictions). Flipped triangles stay under the skeleton's own
folded share in every scene. The integrity tests that see these shards fail
(`wall_at_90_kmh_three_times`, `car_into_the_rear_door_at_60_kmh`,
`random_impacts`, `grab_rear_and_roof`) and are kept as they are.
**Authoring finding:** the underbody, the rear door and the rear body panel
round the boot need a denser lattice (or helper nodes for the skin) so that
one node spacing is not bent through a large angle; tried and rejected: a
blend of node positions plus one blended turn (no lever arms) made it worse
(wall 70: 0 → 162 shards).

**Coverage per part** (frames, breaks; parts not listed draw 100 %): every
cut has a broken beam under it.

| Scene | Part: uncut / rest triangles |
|---|---|
| Wall 70 | body 98.7 %, bonnet 95.6 %, boot lid 98.8 %, fender_r 99.8 % |
| Wall 90 ×3 | body 96.8 %, bonnet 95.6 %, rear doors 89.7 %, boot lid 98.8 % |
| Side 60 struck | door_f_l 89.6 % (was 44.6 %), door_r_l 90.7 % (was 0 %), body 98.5 %, boot lid 98.9 % |
| Side 60 striker | body 99.9 %, bonnet 95.6 % (fender_l was cut almost whole before; now 100 %) |
| Grab door 10 m | 100 % (the door comes off whole along its broken hinges) |
| Grab boot lid 2 m | boot lid 77.5 %, body 96.0 % |

The same 1,532 bonnet and ~250 boot-lid triangles are cut in every crash
(beams inside them break each time; which, not investigated).

### Binding lint, completed

B2 (informational): every node moved by a homogeneous strain (10 % stretch,
5 % shear) or bent round a 6 m radius; a vertex's distance from the same map
of its rest position. A2: frame conditioning (per-vertex frames: angle between
VX − C and VY − C, and |det| of the three unit directions; node frames:
atan(λ₂/λ₁) of a driving node's neighbourhood, 45° even, 0° a line; triangle
binding: the binding triangle's angle).

| Measure | Triangles | Frames, breaks | BeamNG-style | ETK (our loader) | Generated sedan |
|---|---|---|---|---|---|
| A1 group purity | 53,089 | **0** | 0 | 0 (no parts: per piece) | n/a (no parts) |
| A3 Σ\|w\| max; over 1.001 | 12.6; 51,167 | **1.000; 0** | 28.9; 147,340 | 5.85; 37,596 (48.8 %) | 9.08; 383,376 (78 %)* |
| B3 gain max (p99) | 5.66 (1.31) | **1.10 (1.00)** | 7.77 (3.01) | 5.11 (1.72) | 2.96 (0.86) |
| A6 incoherent edges | 63 | **0** | 1,608 | 42 | 0 |
| B1 leaks; rigid error | 201,030; 10 mm | **0; 1.2 µm** | 0; 2.7 µm | n/a | whole car 0.14 mm |
| B4 bridges | 6,042 | **0** | 0 | n/a | n/a |
| A4 reach max; past 3 | 6.5; 3,928 | 7.5; 9,919 | 8.6; 11,115 | 9.2; 9,257 | n/a |
| B2 affine max | 4.1 cm | 6.8 cm | 0.0 (exact) | 7.3 cm (p99 3.3) | 6.4 cm (p99 4.0) |
| B2 bend max | 5.5 cm | 9.5 cm | 5.3 cm | 4.0 cm (p99 2.5) | 9.3 cm (p99 2.6) |
| A2 angle min (p1) | 12° (19°) | 1° (6°) | 12° (22°); det 0.074 | 30° (43°) | n/a |

*The generated sedan's weighted skin (`flexskin`: per node (A, ∂A/∂T, ∂A/∂B))
has negative A by design, so A3's convex-blend rule does not apply to it as
written; B3 2.96 is past the 1.5 bound. Measured from `origin/feat/sedan-body`
(1fb87819) in a scratch worktree (`rig_vehicle("sedan")`, the skin as
`vehicle_film.rs` builds it; 490,997 vertices on 548 nodes), read only.
**Calibration:** BeamNG's own ETK (default config, 77,034 vertices; kc6:
the same) fails A3 on half its vertices, B3 at 5.1 and A6 at 42: our hard
bounds are much stricter than BeamNG content as our loader binds it, so a
binding that passes them is well inside what BeamNG ships. Node frames give
up affine reproduction (6.8 cm under a 10 % strain: offsets are turned, not
sheared) for continuity; the per-vertex affine frame reproduces it exactly
and is the one that shards.

### GPU check (not run here)

The renderer mirrors `eval_breaks`; both skin modules validate with naga
(`node showcase-wgsl-dump.mjs out/wgsl && cargo run -p engine-webgpu --release
--example wgsl_validate -- out/wgsl/*.wgsl`; naga 24 needs `LOD_PX` as a float,
which the dump writes). Not run on a GPU or in Chrome (Tint's uniformity
rules are stricter: `nodeComponents` loops on `workgroupUniformLoad`). Check:
1. `softbody.html?solver=explicit&car=showcase` (drive yard), Chrome.
2. Drive into the wall at 70 km/h: the bonnet and bumper dent and stay
   drawn; holes only where the beam overlay (V) shows broken beams.
3. Drive into a parked car's rear door: the door dents, no hole into the
   cabin unless beams there broke.
4. Grab (crash lab) a door, the fender and the front bumper and pull 0.5, 2
   and 10 m, then release: the panel stretches and comes off whole along its
   broken beams; no confetti, no long blades.
5. The "skin" select (next to the car select) switches node frames /
   triangle binding / legacy on the same physics, for an A/B.
6. Console free of WebGPU validation errors (the params buffer is 96 bytes,
   NS = 13 vec4 per node).

## Open

- The GPU page: the steps under "GPU check" above.
- Shards at folds with no break (underbody, struck rear door, rear body round
  the boot): a denser lattice or skin helper nodes there (authoring).
- Physics findings G1 (grab blow-up at the strongest setting) and G2 (dashpot
  tethers) above: unchanged.
