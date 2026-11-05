# Vehicle authoring

`engine_softbody::authoring` builds node–beam vehicles the way a car is designed:
a body shape drawn as curves, panels cut from it, a structure of members in steel
grades, named masses, suspension, and a look. A definition (`VehicleDef`) is plain
data; `build` turns it into a skeleton, `.truck` text (the explicit solver's input,
Rigs of Rods format) and, through `skin::showcase_visual`, a dense visual mesh bound
to the skeleton. The first definition is the showcase sedan (`authoring/sedan.rs`).

```rust
use engine_softbody::authoring::{build, style, BuildParams, Resolution};
let def = style("sedan").unwrap();
let built = build(&def, &BuildParams { resolution: Resolution::Medium, ..Default::default() })?;
let truck: &str = &built.truck;                       // ExplicitBeams::from_truck, WebWorld.explicitSim
let visual = engine_softbody::authoring::skin::showcase_visual(&built)?;
```

From a page: `showcaseCar({ style, resolution, strength, paint })` (wasm) returns the
`.truck` and the visual; `showcase-car-render.mjs` draws it from the solver's nodes.

## Coordinates

x across (+x the car's left), y up, z along the car from the rear end (0) to the
front (`profile.length`). The `.truck` is centred on the length; its forward is +z.

## The definition

| field | what it is |
|---|---|
| `profile: BodyProfile` | the shape (below) |
| `grades: Grades` | materials: the frame's `crush`, `hsla`, `uhss`, `boron`, `alu`; the skin's `panel`, `floor`, `frame` (lid frames), `plastic`, `glass`; `rigid`, `link`, `mount` |
| `stations` | ring stations kept exactly (door seams, pillars, screens) |
| `panels: Vec<Panel>` | parts cut from the shell: bumpers, bonnet, boot lid, fenders, doors, each a `Region`, a grade and a `Joint` |
| `zones` | grades of shell regions inside a part (glass, the floor pan) |
| `crumple: Curve` | threshold scale of the shell along z: weak ends, strong cell |
| `members: Vec<MemberSpec>` | trusses along paths: rails, aprons, towers, sills, pillars, roof rails, cross members, bumper beams, door beams |
| `webs` | shear webs between members (inner wheelhouses between rails and aprons) |
| `blocks`, `plates` | lattice blocks with mass (powertrain, battery), plates (radiator) |
| `masses: Vec<NamedMass>` | named masses spread over the structure in a box: driver, seats, dashboard, fuel, interior, spare wheel |
| `corners: Vec<Corner>` | double-wishbone corners (left side, mirrored): track, wheel centre, tyre, spring, damper, drive |
| `kerb_mass` | everything, wheels included (kg); the structure takes what the named masses leave |
| `frame_every`, `frame_grade`, `frame_through` | the shell's braced frame (below; the sedan has none: `frame_every: 0`) |
| `shell_braces` | X-brace the shell's quads (off on the sedan: its triangles hold their shape; panels keep theirs as hinge beams) |
| `braces: Vec<BraceSpec>` | lumped shear panels: long diagonals (and mirrors) across the cabin box's faces (floor, roof, firewall, screens, rear bulkhead, door rings) and from the strut towers into it, each spread over the member sections it bears on; `strength` caps a brace's yield (the front towers' fold with the crumple zone) |
| `shear: Vec<ShearSpec>` | zigzag shear panels between two members (an X across each bay): inner fenders, rear floor, wheelhouses, subframes, the cabin's faces |
| `backed` | panels backed by their own frame (doors): every skin node tied to the part's frame behind it |
| `tiered` | fit the step per tier (below; on for the sedan) |
| `body_damping`, `body_damping_reach`, `body_damping_strength` | body dashpots (below) |
| `look: Look` | paint, side glass outlines, screens, lights, grille, intake, handles, plates, mirrors, panel gap |

### The shape: `BodyProfile`

Editable control-point curves over z (`Curve::new(&[(z, value), …])`, a monotone cubic
through the points: no overshoot):

* `top`: the centre line of the top surface: deck, rear window, roof, windscreen, bonnet;
* `bottom`: the underbody;
* `shoulder`: where the side meets the greenhouse or bonnet;
* `half_width`: the side's half width (plan view);
* `top_half_width`: the top surface's edge (narrower over the cabin: tumblehome);
* `crown`, `side_bulge`, `lower_radius`, `top_radius`: section shaping;
* `front_sweep`, `rear_sweep`: the nose and tail domes;
* `wells`: wheel arches (centre, radius, inner wall).

`BodyProfile::half_section(z)` is the cross-section; the shell lofts rings of nodes
along it (spacing per surface from the `Resolution`: 25–40 cm, a sparse body of
heavy nodes, as real node–beam cars are), caps both ends with domed rows, and braces
it: every triangle edge and bend beams that skip a node (panels add a hinge beam
across each shared edge). Ring `stations` are kept exactly (merged only within 5
cm), so door seams and pillars sit where the definition puts them.

### Parts and joints

A `Region` picks shell nodes: `Side(outline)` in side view (z, y), `Top { z0, z1 }`,
`Cap { front, y0, y1 }`, `Box`, `Floor`, or `Any(…)`. The first panel whose region
holds a node owns it (`sided` panels become `<name>_l`, `<name>_r`). Beams across a
seam take the panel's `Joint`: `Weld`, `Bolt { scale, group }` (a detacher group makes
the panel leave whole), or `Hinged { hinge, latch… }` (strong hinges along one edge, a
latch zone on the other, weaker flange beams elsewhere).

### Materials

`Material::steel(kl, zeta, yield, elong, density)`: the axial rigidity k·L (N, the
same at every length: a beam's stiffness is k·L / length), the damping ratio at a
30 cm beam between 0.4 kg nodes (damping `c = 2ζ√(k μ)`; `stiffer(s)` keeps ζ), the
yield force (N), the elongation at break (the break force is yield + elong × k·L:
a yielded beam loses strength as it stretches, so it tears after stretching that
share of its length, as the metal does) and the density. The grades come in two
tiers (below): the frame's are what heavy frame nodes allow at the 0.5 ms step, the
skin's what light skin nodes do.
The explicit solver keeps every yielded rest length ≥ 0.1 m, so a beam under 10 cm
cannot yield: it is elastic up to 2× its yield force and then breaks
(`skeleton::SHORT_BREAK`).

Ties (`Skeleton::tie`: welds, gussets, mounts, catcher braces, a section's ring)
are as stiff as a 30 cm beam however short: what they join sits close by, and a
short beam's stiffness would cost its nodes mass for nothing.

`Built::stats()` gives the medians real cars are compared by: node mass, k·L and ζ
over the structure (the shell's and members' spans: not ties, glass, panels, mounts
or blocks) and over every beam.

### Members

`MemberSpec { path, width, height, up, grade, scale: [start, end], beads, mirrored,
attach, ride, section, station, catch, stiff }`: a box (4 nodes a station) or ladder
(2) truss along a path; every path vertex is a station (members meet at their
paths' vertices) and each segment is split at most 1 m apart (crush members 60
cm; `station` overrides: the B-pillar 0.6 m and the doors' rails 0.42 m, so a
pole at mid-span meets a station, the centre hinge of its 4·M_p/L collapse). `stiff` is the section's k·L against its grade's (rails
×1.4, rear rails ×3.5, sills ×2.3: k·L ≥ 5e5 N). Its grade's
thresholds graded from `scale[0]` to `scale[1]` (a crush rail: weak at the bumper,
strong at the firewall), weaker rings every `beads` stations (it folds there
first), tied to the nearest skin nodes (`attach`) and, at its ends, to the members
it meets. A station node within 5 cm of an existing node of its part reuses it
(`members::SNAP`), and within 15 cm of another member's frame node (`SNAP_FRAME`):
joints are shared nodes, not gusset ties (ties are where compliance and step cost
collect). Ties break by position, so mirrored members pick mirrored nodes. Impact members (`catch`: bumper beams, door beams, sills,
B-pillars, firewall, floor cross members, tunnel, front subframe) get catcher nodes
along one line of each bay at most 15 cm apart (`members::CATCH`): static contact
is per node, so a 20 cm pole always meets one. Catchers are skin-tier nodes (light;
`vehicle::CATCHERS_IN_FRAME` puts them in the frame tier: fewer side-pole
pass-throughs, but they ring after a crash), each with a body dashpot
`CATCH_DAMPING` × `body_damping` strong; joints, braces and mounts never pick
them.

`knee: Some([t, s])` puts a knee in the threshold gradient (scale `s` at arc
fraction `t`): the rear rails are strong (`REAR_RAIL`, 4 × HSLA) from the kick-up
to the rear subframe, where they carry the suspension, fuel and spare wheel
(about 40 kN a side at 40 g in a frontal crash), and fold only behind it.

Blocks (`BlockSpec`) are rigid lattices with their mass; `bear` (m) adds
compression-only beams from the block's rear and side faces to the structure behind
and beside it (an engine driven back meets the firewall and tunnel).

## Lateral strength (sections)

The members that carry a side impact take their thresholds from a real section
(`sedan::side_sections`), not a tuned scale. For a thin-walled closed section
(a box, or a hat closed by its panel) of depth d along the load, width w and
wall t, the plastic section modulus is Z_p = t·(w·d + d²/2), and M_p = σ_y·Z_p.
The truss carries a moment through its chords: a box truss M = 2·T·lever (two
chords a side), a ladder M = T·lever. Each member's chords therefore yield at
T = M_p / (chords · lever), so the truss's plastic moment is the section's. A
centre load on a span L collapses at 4·M_p/L when its supports are simply
supported, or 8·M_p/L with fixed ends; both are given below. The yield stresses
are the grades' minimum Rp0.2: EN 10338 for the dual-phase steels, EN 10268 for
CR340LA, and about 1,100 MPa for hot-stamped 22MnB5. The sections are typical
of compact-sedan bodies, not one car's drawings.

| Member | d × w × t (mm) | Grade, σ_y (MPa) | Z_p (cm³) | M_p (kN·m) | Span L (m) | 4M_p/L (kN) | 8M_p/L (kN) | Chord yield T (kN) |
|---|---|---|---|---|---|---|---|---|
| `sill` (rocker: inner, outer, reinforcement) | 100 × 140 × 1.8 | DP980 (CR590Y980T-DP), 590 | 34.2 | 20.2 | 0.56 | 144 | 288 | 56.1 |
| `pillar_b` (hat closed by its outer) | 70 × 120 × 1.8 | PHS1500 (22MnB5, hot-stamped), 1,100 | 19.5 | 21.5 | 1.05 | 82 | 164 | 76.7 |
| `tunnel` (hat closed by the floor) | 200 × 150 × 1.2 | DP600 (CR330Y590T-DP), 330 | 60.0 | 19.8 | 0.56 | 141 | 283 | 61.9 |
| `floor_x1` (front seat cross member) | 80 × 60 × 1.4 | DP780 (CR440Y780T-DP), 440 | 11.2 | 4.9 | 0.79 | 25 | 50 | 61.6 |
| `floor_x2` (B-pillar cross member) | 80 × 60 × 1.6 | DP780 (CR440Y780T-DP), 440 | 12.8 | 5.6 | 0.79 | 29 | 57 | 70.4 |
| `floor_x3` (rear seat cross member) | 100 × 60 × 1.2 | HSLA (CR340LA), 340 | 13.2 | 4.5 | 0.79 | 23 | 45 | 44.9 |
| `kickup` | 100 × 60 × 1.2 | HSLA (CR340LA), 340 | 13.2 | 4.5 | 0.79 | 23 | 45 | 44.9 |

The span is between the member's supports along the pole's line: for the sill
and tunnel the floor cross members, for the B-pillar the sill and roof rail,
for a floor cross member the sill and tunnel. The floor cross members are hit
end-on by a side pole; their two chords then carry 2·M_p/d, within a third
of the section's squash load A·σ_y. The firewall's cross member stays at its
NCAP-measured strength (`FIREWALL`), and the roof's lumped panels came down
from 20 to 17 kN (`ROOF_YIELD`), because the PHS B-pillar now carries more of
the FMVSS 216a platen.

With these sections, the NCAP side pole's residual crush at the pole fell from
a median of 444 mm to 375–396 mm, and to 340 mm after the re-node (the
measured median is 356, p10–p90 297–445).

The pole sweep (`--ignored`, side) still finds pass-throughs: of 178 cases,
17 at 30 km/h, 59 at 45 and 91 at 60 (front: 0 of 66 at each speed). At
30 km/h they fall in three places. The sweep's offset is along −z from the
car's middle; `showcase_probe … side` takes it along +z, so negate it there.

* The rear overhang, z 0.35–0.80: no cross member between the rear subframe
  (0.62) and the bumper beam (0.15–0.21), so only the rear rails and the boot
  floor's skin bear on the pole.
* The rear door, z 1.65.
* Behind the front wheel, z 3.55–3.65.

Neither the front and rear subframe cross members at real sections
(90 × 60 × 2 mm, S420), nor catchers in the frame tier, nor a lower radiator
cross member changed the 30 km/h count. Node-only contact lets the pole pass
between stations 0.4–0.6 m apart once the catchers' ties fail.

## Node counts (BeamNG's)

The structure is noded as BeamNG's cars are: a frame of a few hundred heavy
nodes with dense node lanes only on the impact faces (the catchers, the doors'
rails). Medium (`counts_per_resolution` asserts 800–1,000 nodes, ≥ 3,000
beams, a frame of 300–350 besides the doors' frames, median 1–4 kg):

| | f447ed8f | re-noded |
|---|---|---|
| nodes / beams | 1,125 / 7,804 | 993 / 6,735 |
| frame (besides the doors') | 490 (392) | 398 (324) |
| frame node median | 1.21 kg | 1.44 kg |
| powertrain lattice | 80 (0.17 m cube) | 40 (faces 0.17 m, two layers) |
| catchers | 108 | 108 |
| skin scale | ×0.43 | ×0.59 |
| frame grades' k·L | ×1.0 | ×1.5 |

Low: 915 nodes, skin ×0.64; High: 1,054, skin ×0.45 (the resolution sets the
skin only).

### The rear body

`showcase_probe -- sedan medium wall 56` with `REAR=1` measures the rear frame
(z < 1.2) in the cell's best-fit frame (sills, pillars, roof rails, tunnel,
the B-pillar cross member) each millisecond: its RMS displacement from rest,
and with `REAR_FIT=1` its own best fit (pitch, centre) against its own
deformation. `… static` with `BEND=y|z` solves it under 30 g with the cell
clamped, and `TOPBEAMS=1` lists the beams storing the most energy.

Before the re-node the rear frame moved 107 mm (peak RMS; 71 mm over the
pulse): it hinged ±4° about the kick-up at about 12 Hz (δ = a/ω²: 30 g on a
12 Hz mode is 52 mm) and slid 85 mm forward on the floor's cross members,
which are ladders that do not bend in plan. Now 21 mm (13 mm over the pulse),
5× less:

* the rear rails start at the sill under the B-pillar (they were held by the
  rear seat cross member alone);
* floor-pan shear panels from the front and rear rails to the sills;
* the rear body's upper chord: the quarter inner and C-pillar ×6
  (`QUARTER_STIFF`, `C_PILLAR_STIFF`), the rear quarter's web ×3
  (`QUARTER_WEB`).

Statically, 30 g vertical on the rear moves it 39 mm (73 before), so its
pitch mode is about 14 Hz. Ten times less motion needs ω × 3.2 (about 45 Hz).
Stiffening every rear and cell member ×2–4 runs the step budget out first
(skin ×0.04, frame ×0.76); what is left is spread across the C-pillar, the
roof header, the door-ring panels and the skin.

## Suspension corners

A corner (`members::corner`) is a five-node knuckle (the axle nodes, the upper
and lower ball joints, the steering arm) on two wishbones and a toe link to a
six-node pivot bracket (four arm pivots, the toe-link pivot, the strut top),
each pivot tied to the body within 35 cm. Content that keeps the knuckle where
it belongs in a crash:

* the toe link is `TOE` (4) × the arms' stiffness: it alone holds the toe, on a
  13 cm lever, against the wheel's inertia 10 cm outboard of the kingpin;
* `Corner::cross` (k·L, the rear: `REAR_CROSS` 1.5e6 N) adds the subframe's
  cross tubes, each arm pivot to its mirror: the wheel's inertia pushes one
  pivot of each arm in and pulls the other out, and the tubes carry that
  across the car instead of bending the rails;
* articulation stops (`Skeleton::straps`, written as spring-less shocks after
  the real ones): the upper ball joint stays within ±`STRAP` (20 %) of its
  spawn distance to the lower arm's pivots (about ±13 % over ±9 cm of travel);
  past that the knuckle would fold round the lower ball joint.

## Body damping

Each beam's damper damps that beam's own mode (hundreds of hertz) but barely the
body's (10–30 Hz: bending, torsion, panels), which would ring for seconds after a
crash. A real body's joints, sealer and trim damp those; here each node gets one
dashpot (k = 0, `body_damping` N·s/m) to a node of its part about
`body_damping_reach` m away along an axis, torn off past `body_damping_strength` N.

## Mass and the explicit step: two tiers

Nodes carry their own mass (`l, kg` in the `.truck`): named masses where they were
put, the rest of the kerb mass spread by beam length × density. No node is lighter
than its beams need for a stable 0.5 ms step: a Gershgorin bound on h²ω² + 2hγ per
node (`Skeleton::floors`). At a fixed kerb mass and step, the sum of all beams'
stiffness (and damping) is fixed; the design question is where to spend it.

The body is built the BeamNG way, in two tiers:

* **Frame** (primary): every member's stations: rails, aprons, towers, sills,
  pillars, roof rails and bows, cross members, tunnel, firewall, cowl, bumper
  beams, subframes, the doors' own beams. About 320 nodes of 1–4 kg besides the
  doors' frames (74 more), median 1.44 kg (the named masses sit on them), its
  grades at 1.5 × RoR-level k·L on the load paths (`sedan::FRAME_STIFF`; see
  Node counts below). It carries
  torsion (the cabin box's lumped shear panels, the strut towers braced into it),
  crash load paths and the door rings.
* **Skin**: the shell (panels, floor pan, roof skin; 25–40 cm, `SKIN_DENSITY`
  1), catchers, ties: light nodes (0.07–0.1 kg), where dents and creases happen.
  Panels are backed (`backed`): doors by their own frame (intrusion beam, belt
  and mid rails, 14 cm deep, hinged and latched to the pillars), fenders,
  bumper covers and the bonnet by the body's frame behind them (the bonnet on
  the aprons and towers: crushed, it folds over the engine bay instead of
  sinking through the wheelhouses' skin, where it rattled in self-contact).

A member's diagonals and bend beams yield at the same strain as its
longitudinals (their thresholds scale with their stiffness): a rail folds
plastically instead of bending back elastically, so the crash's energy goes into
plastic shortening, not rebound. A block's bearing on the structure behind it
yields at `BEAR_SCALE` × UHSS (the engine crushes into the firewall).

The passenger cell's members (A-, B- and C-pillars, sills, floor cross members,
tunnel, cowl, dash, parcel shelf) yield at `CELL` × their grade (4: the cell keeps
its shape at 56 km/h; the A-pillars are boxes, the hinge pillar and screen
pillar in one); the roof's (roof rails, bows, headers) at `ROOF` × (1.2: it
crushes in a roll-over). Lumped panels yield at their grade's strain
unless `strength` caps them: the strut towers' braces (`END_YIELD`), the door
rings (`CABIN_YIELD`), the roof (`ROOF_YIELD`), the screens (`GLASS_YIELD`).

Named masses fill the stability floors of the frame nodes in their box first
(`Skeleton::add_mass_by_need`), then spread by stiffness: mass a node already
carries beyond what its beams need is budget spent twice. The sedan's boxes
span the structure that carries each mass (occupants and seats: the floor
members, sills, tunnel and the pillars' feet; trim: the whole cabin), and its
seat frames weigh only what their own lattices need (7.5, 7.5, 12.5 kg; the
rest of the seats is a named mass on the floor). That freed about 100 kg of
the budget: the skin's scale went from ×0.19 to ×0.47. `showcase_probe --
sedan medium info` with `SLACK=1` lists the mass beyond the floors per group.

The crumple zone's plateau (`CRUSH_PLATEAU`, 0.9 × the crush grade) and the
crush grade's rigidity (`CRUSH_RIGIDITY`, 0.15 × the frame's) are set from
NCAP's measured pulse (`crash-reference-targets.md` §8): the front springs
back 0.10–0.24 m from its dynamic crush to its residual, so it unloads over a
long, soft tail. The front rails are graded from 1.0 under the floor
(`RAIL_CABIN`) to 0.7 at the bumper (`RAIL_BUMPER`): the force rises with the
crush, as Campbell's model has it.

`VehicleDef::tiered` fits the step per tier (`Skeleton::fit_tiers`): the frame's
beams at their grades (`Built::stiffness_scale` = 1), the skin's stiffness and
damping scaled to what is left (`Built::skin_scale`). The frame keeps its yield
strains; the skin keeps its yield and break forces (what holds a panel on and where
it dents are forces). Blocks' and plates' mounts and the suspension are kept as
built (`fixed`).

`showcase_probe -- sedan medium static` solves the body-in-white's torsion as a
linear static problem (block-Jacobi conjugate gradients on the built stiffnesses):
kN·m/deg at the body-side shock-tower mounts, the twist along z, and where the
strain energy sits (`COST=1`: each group's share of the step's cost beside its
share of the energy: what to stiffen and what is spent for nothing; `DOOR=1`: a 4
kN push on the door instead). `MASSES=1` lists node mass by class;
`SHOWCASE_FLOORS=1 … info` the floors by group, tier and beam kind.

## Hardware, bearing and sheet masses (2026-11-05)

Parts hang on hardware, not on their seams.

- **Doors.** Each door has an upper hinge, a lower hinge, a latch and a
  secondary latch. Each is one detacher group of tension-only ropes from one
  pin node, the outer node of a member end, to the pillar.
  - Using one pin lets the door turn about its hinge line.
  - Each pin's ropes run toward the pillar, inboard and up, so every pull
    meets a rope sized for it.
- **Strengths.** They come from a static FMVSS 206 rig solve
  (`rig_tensions`, `size_hardware`). The fixture moves the pins along the
  pull. A rope at cosine c carries k·c/Σk·c² per newton.
  - Each rope's strength is the largest load × share over a fan of pulls.
    The fan runs from longitudinal (away from the pillar) to transverse
    (outward) on the ellipse through the two measured loads, and, for doors,
    from both of those to the door's weight (down) at the longitudinal load.
  - Group strengths, front door, Medium:
    - latch: 28.0 kN longitudinal, 20.5 kN transverse;
    - secondary latch: 19.0 / 17.8 kN;
    - hinge pair: 73.8 / 46.9 kN. Each hinge on its own holds the 36.2 /
      29.8 kN pair floor (2005 Camry, report 637726). No report separated a
      pair, so neither hinge failed. At half that floor, the intrusion beam's
      pull tore the lower hinge off in every side pole run.
  - The source is `crash-reference-targets.md` §9. The secondary engages
    softer, so the assembly fails with its primary.
- **Bearing.**
  - The member ends of each door edge bear inboard on one flange node of
    the pillar. Each end gets one 50 mm strip, with capacity σ_y·t·w: A-pillar
    DP980 1.8 mm 53 kN, B-pillar PHS 1.8 mm 99 kN, C-pillar DP780 1.4 mm
    31 kN.
  - The door's inner end faces (`door_face_*`, HSLA 0.8 mm × 150 mm, 41 kN)
    carry the members' pull to the hardware.
  - The skin edge is compression-only. The shut faces bear through the hem;
    the rest is seal.
- **Lids.** The bonnet and boot lid have a hinge on each side and a latch at
  the centre of the far edge, sized with the door values. FMVSS 206 covers
  side doors only, so this is an assumption.
  - Their seals are compression-only.
  - An inner-frame ladder (`lid_inner_*`, `bonnet_inner_*`) gives them depth.
    It is in the skin tier; each chord's strength is M_p / lever of a
    100 × 60 × 0.75 mm hat at 210 MPa.
- **Backed panels** (fenders, bumpers, bonnet, boot lid) bear on the body
  frame through supports. Only their hardware carries tension.
- **Sheet masses** (`SheetSpec`). A panel's mass is its skin area (a third
  of each triangle to each node's part) × Σ gauge × share × density:
  - bonnet 14.3 kg, boot lid 7.7 kg: 0.75 mm outer plus an inner of the same
    gauge over half the area;
  - fenders 2.7 kg: 0.75 mm;
  - bumper fascias 2.5 / 1.3 kg: 3 mm at 1,000 kg/m³.

  These parts take no share of the structure's mass. The kerb mass stays
  1,450 kg.
- **Tests.**
  - `r`: the FMVSS 206 rigs, static and in the solver.
  - `s`: 30 g inertial load.
  - `g`: door pull. The door comes off at 82 kN, the sum of its parts'
    transverse strengths (2 × 29.8 + 20.5 kN).
  - Bearing: a door end bears on the pillar beside it along the car, and
    inboard on a flange strip (one flange node behind each member end, both
    chords bearing on it). It never bears mostly outward on a node outboard
    of it, because that would hold the door shut.
  - `o`: FMVSS 214 S9.2.3 door retention in the side pole.
  - `t`: car-to-car T-bone, reported. `TB_KMH`, `TB_MASS`, `TB_DZ` and
    `TB_ANG` vary it.

## Example: tweak a parameter

A longer, lower roof and stronger front rails:

```rust
let mut def = sedan();
def.profile.top = def.profile.top.offset(-0.04);            // 4 cm lower everywhere
let rail = def.members.iter_mut().find(|m| m.name == "rail_f").unwrap();
rail.scale = [2.5, 0.6];                                     // 25 % stronger at the cabin end
def.grades.crush = def.grades.crush.scaled(1.2);             // every crush member 20 % stronger
let built = build(&def, &BuildParams::default())?;
```

## Example: a new body style

`authoring/hatchback.rs` reuses the sedan's builders: a shorter body, a taller
roof that runs to a steep tailgate, the boot lid's region replaced by a tailgate
over the rear window:

```rust
pub fn hatchback() -> VehicleDef {
    let mut d = sedan();
    d.name = "Rubble Showcase Hatchback".into();
    let s = 4.25 / 4.70;                       // shorter, same front
    d.profile.length = 4.25;
    // Roof held to the tail, then a steep tailgate.
    d.profile.top = Curve::new(&[(0.0, 1.05), (0.25, 1.30), (0.45, 1.40), …]);
    …                                          // stretch the other curves, move the rear axle
    d
}
```

## The visual car

The visual is generated by `authoring/parts` and bound to the skeleton by
`authoring/skin` (`showcase_visual`). It is built as a constructed car: a body
surface cut into separate panels, plus modelled parts. Nothing in it follows the
skeleton's node spacing; every vertex rides the solver's triangles through the
binding, so the same mesh fits a coarse or a dense skeleton.

### Design: an original mid-size sedan

The proportions come from the package (`BodyProfile`): 4.70 × 1.85 × 1.40 m, a
2.75 m wheelbase, 0.92 m of front overhang and 1.03 m at the rear (a low
greenhouse, the body 1.5 cm down on its wheels: `sedan::STANCE`; the
structure's points were drawn on the earlier, taller package and
`sedan::restyle` carries them to this one). The
greenhouse sits rearward: the windscreen's base is about 0.45 m behind the
front axle (a long dash-to-axle), the roof peaks over the B-pillar, and a fast
backlight runs into a short, high deck. The visual reads the package only;
the cabin's fixed points (the dash beam, the column's top) come from the
definition's members (`interior::Cabin`), so the dashboard and wheel follow
the structure.

The character lines are:

* **Shoulder crease.** One crisp line, the body's widest point, runs from above
  the headlamp at about 0.75 m, level through the handles at about 0.81 m, and
  kicks up over the rear wheel into the tail lamps at about 0.90 m
  (`BodyStyle::crease`; a 25–30° change of slope). Above it the side leans in
  to the belt.
* **Flanks.** The side is not a slab: below the crease it bows, tucking
  in 6.5 cm toward the sill with its steepest turn low (`SectionShape::bow`,
  the lower line's slope as a multiple of the mean), so the lower doors face
  the floor and read darker; above the crease the shoulder leans in hard
  (`a_above` 0.85) and catches the sky, a bright band over the handles.
* **Lower-door sculpture.** Under a straight feature line rising from 0.46 m
  behind the front arch to 0.50 m ahead of the rear one (parallel to the
  rocker in side view), the side steps in 2.4 cm with a
  crisp upper edge, easing out toward the sill (`BodyStyle::scoop`). A band row
  lies on the line, so its highlight runs straight.
* **Fillets.** Two extra band rows a few millimetres either side of the crease
  and of the feature line: the smooth normals turn within that fillet, so each
  line reads as a crisp line of light instead of a blur one row (2.2 cm) wide.
* **Bonnet.** Two power-dome creases run from 0.22 m off the centre at the nose
  to 0.60 m at the cowl (out toward the A-pillars), the dome 1.4 cm proud
  between them (`hood_crease`), faded in and out over 30–35 cm C2 ramps; a
  column of the top lies on each crease, with a fillet column 6 mm either
  side. The bonnet's leading edge and the boot deck roll over in 4–5 cm
  (`nose_ramp`, `tail_ramp`; the ramp's length is blurred round the corners,
  σ 8 cm, so the corner's surface does not twist); the bonnet's edges round
  over a third of the fender ramp (`bonnet_edge`); the deck kicks up 2.6 cm
  into an integrated lip spoiler (`deck_lip`).
* **Deck ledge and C-pillar.** Behind the cabin the ramp is a second family
  (`rear_deck`): a crisp roll off the belt, its width solved so it lands on
  the deck's level carried forward as the quarter's ledge, a 3.5 cm fillet
  where the C-pillar's sail rises off the ledge, the rail round into the
  backlight. The fillet's line (the sail's base) sweeps out from the
  backlight's foot to the belt under the quarter light; as the ledge comes
  down to the belt it tilts up to the glass's slope, so it merges into the
  shoulder without a notch.
* **Fascia.** Raked: the widest line low (0.56 m), the upper fascia leaning
  14 cm back into the bonnet. A low trapezoid grille (0.66 m wide at the
  top, 0.89 m at the bottom, 0.22 m tall, its top edge just under the
  lamps' inner ends: no blank band of paint between them) with a
  1.6 cm gloss-black surround and a honeycomb; slim full-LED headlamps (3.5 cm
  at the grille, 1.5 cm at the tip) just under the bonnet's edge, wrapping the
  corners to a pointed tip, with a DRL blade along their upper edge
  (`LampSpec::guide_above`); an L-shaped blade intake in each corner (a slim slot along the lower edge
  rising into a tall air curtain with two gloss-black fins), so the corner
  stays a sculpted surface;
  a gloss-black splitter blade standing 3 cm proud along the bumper's lower
  edge; no front plate (`plate_front_h` NaN). The bumper rises into a
  body-colour collar round the grille (1 cm proud, its crest 2.2 cm outside
  the opening, `collar`), and under a crisp crease at 0.405 m (a band row:
  the lower feature row, `lip`) the chin stands 1.4 cm proud. At the tail the
  same row carries a crease at 0.53 m with the bumper 1 cm proud under it.
* **Arch flares.** The front arches swell 1.8 cm over a 19 cm band (blending
  into the door, not bulging past it), the rear haunches 3.6 cm over a
  tighter 15 cm (`rear_flare`: a firm shoulder over the rear wheel), giving a
  plateau at the rolled lip
  and a wider-track look. The arch is cut 1.8 cm outside the tyre, and the
  wheels (a 20-inch look: the rim 0.80 of the tyre's radius, a low-profile
  sidewall) sit 4.2 cm outboard of the physics tyres (`design::WHEEL_OUT`),
  flush with the arch.
* **Tail.** Wrapped tail lamps, deep at the outer end, joined by a 7 cm
  smoked light band across the boot lid (an LED outline and one LED blade
  through it, the badge on its centre); a high-mounted stop lamp inside the
  backlight's top edge; a black diffuser insert rising into pockets round
  the exhaust tips, with five fins under it; an original badge (a split
  hexagon, `fixtures::emblem`) on the light band and the nose; a rear plate in the
  bumper.
* **Tumblehome.** The side glass leans in by about 22°, from the belt at a half
  width of about 0.86 m to the roof rail at about 0.66 m. The roof, bonnet and
  deck are crowned (bonnet and deck 3.4 cm).
* **Belt kick and DLO.** The deck ledge falls to the belt along a C2 curve
  (`body::ledge_y`, z 0.85 to 1.50 m), so its highlight runs into the belt
  as one curve. `BodySurface::belt(z)` is that feature line:
  the quarter light sits on it and its rear edge sweeps back along the
  C-pillar to a point on the belt, so the daylight opening reads as one swept
  shape tapering rearward. The B-pillar blackout is 8 cm; the rear door's
  glass and the quarter light share one 2 cm sash.
* **Plan view.** The nose and tail are bowed and the corners are rounded (a
  smooth intersection), so the bumpers wrap round without end caps.

The shut lines are:

* the bonnet runs from the cowl forward, cut on the fender tops just outside the
  top line, with its front edge at the headlamps' top;
* the fender/bumper cut runs diagonally from the lamp's outer corner to the arch;
* the front door's front edge leans 5° back;
* the B-pillar cut sits at z = 2.24;
* the rear door's rear edge slants and dog-legs round the rear arch;
* the deck lid wraps over the tail between the lamps;
* the rear bumper is cut under the lamps;
* there is a black sill below the doors.

Each side of a shut line is shaded darker over about 1.3 cm (baked AO), so the
lines still read at the three-quarter distance where the 4.5 mm gap itself is
under a pixel. A shut line's wall is the paint in shadow (`gap`); a window
opening's edge, the doors' belt edges included, is black rubber.

### Part list and how each is built

| Part | Construction |
|---|---|
| Body surface (`parts/body.rs`) | Patches between explicit feature curves. **Band** (sides, bumpers, fascia, tail): a section swept round the plan outline per arc length θ, blended between side, nose and tail shapes, a cubic span between each pair of feature rows (curl, lower line, lower-door crease or lip crease, shoulder crease, belt); the crisp lines lie on rows with fillet rows a few millimetres either side. **Ramp** (fender tops, greenhouse sides and pillars, C-pillar sails, quarter tops, the bonnet's and deck's roll-overs): between the belt and the top line (the belt offset inward by the ramp's length), per column a section of slope plateaus joined by smoothstep windows (`Ramp`): it leaves the belt on the band's slope and curvature, rounds over a controlled width (the shoulder; a crisp roll onto the deck ledge), runs straight, and rounds into the top on the top's slope and curvature, so curvature is continuous across both feature curves and a crisp line is a window of known width. Its rows lie on the windows' edges, spaced by arc length and by turn (a row per 15° of slope). **Top** (bonnet, windscreen, roof, backlight, deck): an analytic crowned height field over plan view with the power domes and the boot lip; its rows are anchored on the top line's vertices (with rows between them, fanning into the next vertex, where they are far apart) and a column lies on each dome crease. Every normal is the patch function's own (central differences of it, not of the mesh); the lookups are Catmull–Rom, so no normal steps between samples. The profile's curves are blurred (σ 8 cm), and the styling's crease line (σ 12 cm). Grid: 2.2 cm across the sections (coarser over the lower body, the flat spans and the top), 3.5 cm along the car (about twice that at the distant LOD). The highlight check: `showcase-car.html?zebra=1|2|3` (reflected stripes along the car, across it, level bands) and `cargo run --release -p engine-softbody --example showcase_zebra -- DIR` (a software render of the cut pieces; `RAW=1 WIRE=1` the uncut surface tinted by patch). |
| Panels (`parts/trim.rs`, `design.rs`) | Each panel is a region (an SDF made of smooth cut lines in side, plan or arc view), made metric on the surface and clipped by marching triangles, so edges follow the curve, not the grid. Every cut is rolled (2.8 mm radius) into a 12 mm wall and a return flange, set back 2.2 mm, giving a 4.5 mm shut line. Arch lips get an 8 mm roll. The inside of a panel renders dark. Panels: bonnet, cowl (black), lip and bumper front, fenders, doors, deck lid, valance and bumper rear, sills (black), body. |
| Glass | The body surface over each opening, set 3 mm in (nearly flush), with a black frit band printed inside the windscreen and backlight. The side windows sit in slim (1 cm) black sash frames, with a satin-black B-pillar applique (`mat::PILLAR`, one half on each door: no mirror-bright sky once the doors move) and a chrome belt moulding. The windscreen is laminated (it cracks); the side glass is tempered (it shatters). The rear doors', quarter lights' and backlight's glass is privacy-tinted (`mat::PRIVACY`). |
| Lamps (`fixtures::lamp`) | A lens continuing the body surface; a housing extruded along the lamp axis; a reflector back; small projector cups with lens domes; light guides that follow the outline (the DRL ring; two tail rings; the light bar; the fog lamps' LED line). |
| Grille (`fixtures::grille`) | Gloss-black surround, a dark throat recessed 8 cm and a honeycomb back mesh (drawn by the shader). The side intakes are grilles too, their throats along the lamp axis. |
| Wheels (`fixtures::wheel`, `caliper`) | Tyre lathe: four tread grooves, rounded shoulders, a rim protector and lettering band, proportioned to the sidewall's height. Texture coordinates on the tyre (u round it, v up the sidewall or across the tread) carry raised "RUBBLE SPORT 245/35 R20" (matching the 20-inch rim) lettering on the outer sidewall and tread blocks on the shoulders. Rim: lip and barrel, a dished face of ten split spokes (each forks toward the lip) cut on a polar grid, the faces diamond cut (`mat::MACHINED`), the 4 cm spoke walls gunmetal (`mat::RIM`); centre cap, five lug nuts. Vented disc (38 cm front, 35 cm rear) with its hat. The lettering's u seam has its own vertices (no text run backwards). The caliper is a monoblock: a flat-faced outboard piston housing, a bridge over the disc's rim and a slimmer inboard half, each a rounded section swept round the disc's edge with rounded ends; it rides the axle (it does not turn). |
| Mirror, handles, wipers | An egg housing (paint above, black below) on an arm from the door's sail, with its glass; a chrome pull handle over a dark pocket; two wipers parked along the windscreen's base. |
| Plates, badges | Plates are the body surface over their outline, set proud, with texture coordinates: the shader prints a border and "RBL 470" (a 5 × 7 pixel font), fading to grey when too small to read. The badge is extruded chrome facets on a black backing. |
| Interior (`parts/interior.rs`, `design.rs`) | Superellipsoid shapes: front seats (cushion, back, side bolsters, headrest on chrome posts, pedestal), a rear bench with two shaped places and three headrests; a dashboard lofted from a side section across the cabin, with a binnacle, a lit instrument cluster and a floating centre screen (the `display` material draws dials and a map from texture coordinates), vents and a trim strip; a three-spoke steering wheel with its hub and column shroud; a console with armrest and selector. Door cards are each door's skin 10 cm in (dark above the armrest); the headliner and pillar trim the greenhouse's skin 1.8 cm in; carpet; a parcel shelf with the high-mounted stop lamp. |
| Underbody and others | Arch liners, underbody plate, exhausts with chrome tips. |

### Binding, damage and cost

* **Voted pieces.** A piece binds to the physics part most of it lies on (a vote
  of the nearest single-part skin triangles). Where it reaches more than 8 cm
  past that part, it is cut along the smooth zero line of the distance
  difference (marching triangles), and the far side rides the other part
  (`Bind::Part`). So it parts along a clean line instead of stretching. A
  vertex leaves its piece's part only beyond 9.5 cm (1.5 cm past the cut), so
  the vertices on the cut line itself stay on their side and the line opens
  as one clean crack, not a sawtooth of triangles riding alternate parts.
* **Rigid pieces.** These ride one triangle's orthonormal frame
  (`ShowcaseVisual::rigid`): they move rigidly and never stretch. They are
  the lamps, mirrors, handles, and the tempered side glass (each pane its own
  piece: `glass_side_{f,r,q}_{l,r}`, riding its own door) and backlight.
* **Damage follows the physics** (`skin.rs` damage units; plan P5). Each
  glass pane, lamp and mirror is a `DamageUnit`: a set of the solver's beams.
  * A pane with a glass zone owns its zone's beams (both nodes in the zone,
    on its side). Every other unit owns its mount: the solver beams along the
    skin triangles under it.
  * A unit is broken when one of its beams is broken, or when its live
    beams' largest plastic strain |rest − rest₀| / rest₀ passes its trigger:
    tempered glass's break strain σ/E = 120 MPa / 70 GPa (0.17 %; glass has
    no plastic range, so a glass beam that has yielded that far is a broken
    pane, as BeamNG's deform groups switch glass), a lamp lens's 3 % (PMMA's
    elongation at break is 2–10 %). A pane without its own glass beams (the
    quarter light) and a mirror break only when a mount beam breaks.
  * The renderer computes this on the GPU every update from the solver's beam
    buffer (or the crash lab's compact rest buffer, live or replayed): a
    broken unit's glass, lenses and mirror glass are dropped and throw their
    shards, its lamps go dark. `unit_states` and `deform_with` are the same
    on the CPU. The laminated windscreen never drops: it cracks where its
    nodes' beams yielded past annealed glass's 0.064 % or broke.
  * `showcase-car-check.mjs` runs `damage_*` (`?check=damage`): every
    simulated frame it reads back the solver's beams and the renderer's unit
    states in the same submission and fails on any difference.
  * When the physics' own break groups land in `Built` (`DamageUnit`s by
    name: `glass_*`, `headlight_*`, `mirror_*`, hinges, latches), they
    replace this derivation; mirrors and lamps that become their own small
    parts then ride (and fall with) those parts.
* **Substance** (plan P7). Behind every door's outer skin sits its inner
  skin at the door's section (`DOOR_SECTION` 8.5 cm, just outboard of the
  door card), with shut faces round its edge; under the bonnet and boot lid
  an inner frame (`LID_SECTION` 3.2 cm: a 7 cm band round the edge and two
  ribs along it, the rest cut away as lightening holes, every hole flanged).
  These stampings (`mat::INNER`, the body colour, satin) bind where the outer
  skin over them binds (`Bind::Inset`: each vertex picks its triangle at the
  outer skin, then sits at its own depth), so a tear shows sheet with depth
  behind it and a detached door carries its inner skin. The engine bay
  (`parts/bay.rs`): the engine on the `powertrain` block, the radiator on its
  plate, apron walls under the apron rails with the strut towers' domes, and
  the bulkhead on the firewall, so a crushed or missing bonnet shows a bay.
  Round a tear the paint has cracked off the stretched steel (a grey primer
  band inside the bare lip).
* **Debris** (plan P9; `fx-particles.mjs`, shared with `car-render.mjs`).
  Sparks from metal nodes sliding on the floor or a static box faster than
  2 m/s; glass and lens shards from a breaking unit (area × `SHARD_DENSITY`,
  400 a square metre: each a few centimetres of granules); paint and plastic
  flecks where the skin tears or a panel yields suddenly; dust on hard
  landings. A ring of 8,192 particles shared by every car. Only with
  `update`'s `dt` > 0 (the simulated time since the last update).
* **Wear** (plan P11). Per node the renderer keeps its plastic strain and
  the metres it slid in contact with the floor or a static box. The
  clearcoat dulls with plastic strain (from 0.4 % to 6 %); scraped paint is
  scratched to primer, then bare steel, in streaks, more with every metre
  slid. Scrapes accumulate forward in time (a replay scrubbed back keeps
  them).
* **Cabin.** Seats and the steering wheel ride one triangle of their frame
  rigidly (`Bind::GroupRigid`), so a far binding triangle never swings them
  out of the cabin; the dashboard and console ride their members' triangles.
* **Folds.** A back face whose skinned normal still faces the eye is a
  sliver folded over in a crush, not a panel's inside: it shades as paint.
* **Tears.** The renderer keeps a per-node tear field: a skin triangle's
  perimeter against its rest perimeter (`node_rest`), interpolated over each
  vertex's binding nodes. Panels are cut where it crosses ½, a smooth line,
  with a thin bright lip of bare steel at the edge (the field's last 6 %: a
  few millimetres, lit by the studio) inside a band of grey primer. A
  panel's inside is dark e-coat grey.
* **Shading.** A procedural showroom: a cyclorama with a bright horizon band
  fading to a dark ceiling, three long softbox strips overhead (crisp lines
  along the roof, bonnet and deck), two low side strips, a key box and the
  sun; below the horizon the floor reflected from the shaded point (short
  rays land in the car's own shadow: under the sills, in the arches; far rays
  fade into the horizon), so the flanks carry a crisp horizon line.
  `showcase-car.html` draws the same cyclorama and floor behind the car,
  and the car reflected in the floor (`drawReflection`: the opaque level
  mirrored, a depth pass then one cheap diffuse shade of the nearest surface,
  blended at 0.22 reflectance fading with height; `?reflect=0` turns it off,
  `?reflect=0.3` sets it). Only the opaque triangles under 0.95 m at rest
  are reflected (a second index list; the reflection has faded to 3 % there).
  Contact shadows follow it (`drawReflection`'s `contact`): the triangles
  within 25 cm of the ground at rest (tyres, sill and bumper edges) drawn
  flattened onto the floor, each layer darkening it by up to 0.3 by its
  height, spread a little along its horizontal normal, so the car sits on the
  floor wherever it is and however it has turned. The crash lab draws both
  on its navy floor when the body is shown (`softbody.html?labReflect=0`
  off, `=0.2` the reflectance; 0.16 by default; the floor then leaves no
  depth). The crash lab's backdrop is a navy cyclorama with a lit horizon
  band. The paint deepens toward grazing (flop); tail lenses are a deep
  ruby; tyres satin. Metallic flakes are fixed in the paint's rest space (they
  never swim as the body deforms) and glint near the sun's highlight in a
  close view, fading out where a flake is under a pixel. The clearcoat has a
  mirror-sharp sun glint over a broader sheen; its sky reflection is eased
  at grazing angles so a flat bonnet does not read milky. Diffuse light is an analytic lobe fitted to the studio.
  Ambient occlusion is baked from analytic occluders (downward faces low on
  the car, faces inside the arches); the wheels, which turn, are darkened by
  height instead. Cabin trim is lit without studio reflections.
* **Normals.** Smooth normals follow the node normals. Where they disagree with
  the triangle's frame by more than about 15°, the faceted frame normal wins,
  so physics kinks read as creases.
* **Level of detail.** `showcase_visual_lod(built, 0.45)` (wasm `visualLod`) is
  the same car with a coarser body, single-step rolls and fewer segments:
  about 75k triangles against about 240k.
  * The renderer picks a level per car on the GPU from the last frame's eye
    (indirect dispatch and draw), so only that level is skinned and drawn.
  * Measured with 10 cars on an M3 Max at 1080p with 4× MSAA
    (`showcase-car.html?perf=1&n=10`): about 0.50 ms per car with the
    interior, the floor reflection and the contact shadows (about 300k
    triangles, 99k at the distant level; 0.41 ms without the reflection),
    0.31 ms at the distant level. Another GPU user moves these by ±10 %.
  * With the damage units, wear, debris and the inner stampings and bay
    (348k triangles, 121k at the distant level; 2026-11-05): about 0.62 ms
    per car (the previous renderer on the same mesh: 0.55; the stampings
    and bay add 0.03), 0.37 ms at the distant level. The skinned vertex is
    32 bytes (tear and damage, plastic strain and scrape packed as f16
    pairs); worn paint and the primer band shade through the generic metal
    and matt paths rather than code of their own, because the paint path is
    occupancy bound (a second inlined `env()` costs it).

To restyle, change `design::sedan_styling` (cut lines, outlines, section
shapes) and the profile. The tools stay the same.

## Imported JBeam vehicles' visual layer

`engine-softbody/src/flexmesh.rs` (clean room: BeamNG's public flexbody and
prop documentation, the Collada 1.4 specification) gives an imported BeamNG
vehicle the same visual format as the authored car, so
`showcase-car-render.mjs` skins, shades and damages it the same way.

* **Meshes.** `read_collada` reads a `.dae`: triangles, polylists and
  polygons (fanned), scene-node transforms (matrix, translate, rotate,
  scale), `Z_UP` or `Y_UP`. Corners equal in position, normal and texture
  coordinates are welded into one vertex (exporters write each corner on its
  own; the supercar's 4.6M corners are 0.67M vertices).
* **Flexbodies.** Each `flexbodies` row (`jbeam::assembled_rows`) binds its
  mesh (a scene node's or a geometry's name) to its node groups' nodes
  (`desc.node_group_nodes`): every vertex on the nearest node and the most
  perpendicular pair of others (`flexbody::bind`). Wheel meshes, whose groups
  are the nodes a `pressureWheels` row makes, bind to those (the solver
  appends them after the desc's nodes). A row's `pos`, `rot`, `scale` move
  the mesh. A desc node maps to the solver node at its rest position, so the
  binding holds on either spawn path (`.truck` or JBeam mode).
* **Props.** Each `props` row puts its mesh rigidly on `idRef`, `idX`,
  `idY`; a lamp's (`func` a light) lit parts are emissive.
* **Deform groups.** A row's `deformGroup` is a damage unit: the beams of
  that group (`desc.beam_group_names`), broken once one breaks or yields past
  the group's `deformationTriggerRatio`, the solver's own rule. Glass then
  shatters (a `windshield` mesh cracks), a lamp goes dark. The renderer
  computes it from the beam buffer each update (C1's check applies).
* **Materials.** A Collada material picks a shading class by its name
  (glass, lamp lens, chrome, tyre, rim, trim, interior, engine, else paint);
  the vehicle's `*.materials.json` makes it glass when translucent, emissive
  when it glows, and gives untextured materials their diffuse colour.
  Textures are not read (a textured material shades flat grey; paint takes
  the paint).
* **Level of detail.** The distant level clusters each mesh on a 3 cm grid
  per material and normal direction (`cluster`, `LOD_CELL`), switching where
  a cell is 2.6 pixels at 1080p (16 m). The near level is the full mesh.
* **Pages.** wasm `jbeamVisual(spec, meshes)`; `showcase-car.html?mod=<folder>`
  renders a mod's visual at rest (no solver; `?n=` copies); the crash lab
  loads `softbody.html?...&mod=<folder>&jbeamMode=native` (a folder served
  with a `files.json` listing). `showcase-car-check.mjs mod_*` and
  `crash-lab-check.mjs --car supercar|tutorial` use the gitignored
  `inspiration/beamng-mods` (`RUBBLE_MODS`), skipped when absent.
* **Measured** (M3 Max, 1080p, 4× MSAA, 10 copies): the Monster supercar
  (95 flexbodies, 1.52M triangles: about 40 times a typical BeamNG car's
  30–40k) costs 1.57 ms a car near and 0.91 ms a car at the distant level
  (275k triangles). Both are over the 0.5 ms budget; the supercar's own mesh
  is the outlier, and a third, coarser level is the next step.

## Tests and tools

* The side pole's intrusion is the deepest door node near the pole across
  the car in the far side's frame (the right sill and the tunnel): the car
  spins 55–75° round the pole, so world x would measure the spin. The sweep
  `SWEEP_LIST=1` lists the offsets that passed through; the boot-lid test
  `M_KMH`.
* `cargo test --release -p engine-softbody --test showcase_car -- --nocapture`:
  every physical assertion comes from `crash-reference-targets.md` (measured
  NHTSA data, Campbell's model, FMVSS) or is exact physics. Structure: counts
  (≥ 1,000 nodes), closed skins, the kerb mass, symmetry, the visual binding,
  the tiers (the frame unscaled, every beam under-damped), torsion
  (reported). Ledgers (§4.1–4.2): every scene's nodes' momentum change is its
  external impulses (the contacts recomputed as the solver's node pass
  applies them, gravity, drag) to f32 rounding, the internal forces sum to
  zero, the mechanical energy (a tenth of a second's mean) never rises by
  more than the car's standing-still chatter, and it loses at least what the
  beams lost to yield and fracture: at rest (`a`, no yield either), the
  56 km/h wall (`b`, Low and Medium), the poles (`c`, `d`), the 15 m drop
  (`e`), the 120 km/h wall (`f`: Campbell's b1 inside the measured spread)
  and every NCAP wall of `n`. A door pull as large as its joints' summed
  break strengths tears it off (`g`); the crashed car comes as still as it
  stands (`h`); the rear suspension keeps its geometry in the 56 km/h wall
  (`k`); the boot lid holds within FMVSS 206 (`m`). Reference tests: `n`
  (frontal NCAP), `o` (side pole), `p` (FMVSS 214 door), `q` (FMVSS 216a
  roof). `--ignored` adds the pole sweep (`SWEEP=front|side`,
  `SWEEP_KMH=60` for a subset).
* `cargo test --release -p engine-softbody --test showcase_rest_wobble -- --ignored --nocapture`:
  the materials and rest wobble beside real Rigs of Rods cars.
* `cargo test --release -p engine-webgpu --test showcase_car_gpu`: GPU against CPU.
* `cargo run --release -p engine-softbody --example showcase_probe -- sedan low wall 56`
  (`VISUAL=<dir>` renders the skin before and after, `DRAW=<dir>` the beams;
  scenes `wobble`, `settle`, `torsion`, `pole`; `BREAKS=1` sums broken, bent and
  elastic energy per member pair, `WHEEL=<i>` traces a wheel's knuckle;
  with `BRACKET=1` its pivot bracket's strain and rotation in the cabin's
  frame, the knuckle's in the bracket's, and each body tie, `BRACKET=2` the
  corner's nodes' displacements; `ESPLIT=4,8` the elastic energy by zone at
  those frames; `RELAX=<zones>` removes a zone's stored energy at `RELAX_AT`
  (which zone drives the rebound); `BROKEN_PAIR=a~b` lists a pair's broken
  and bent beams; `FASTEST=1 CONTACTS=1` the self-contact triangles under the
  fastest nodes). `PULSE=3.36` (with `PULSE_AXIS=x`, `PULSE_ALL=1`)
  traces the wall force and the force across a plane by member every 2 ms;
  `ESPLIT_YIELD=1` splits off the elastic energy held by yielded beams and
  `RELAX=yielded` removes it; `SLACK=1 … info` lists node mass beyond its
  stability floor per group.
* `showcase-car.html?res=medium&scene=wall&speed=56`: the car on the GPU in a page
  (`freeze=0` shows the rest pose, `perf=1` times the car's render pass;
  `cam=` a named view or a direction `x,y,z`, `look=x,y,z` offsets the target,
  `fov=` radians); `node showcase-car-check.mjs --size 1920x1080 [--only
  rest_three,...]` takes headless screenshots (close-ups: `rest_low`,
  `rest_nose`, `rest_back`, `rest_head`, `rest_tail`, `rest_wheel`,
  `rest_interior`, `rest_dash`, `rest_screen`, `rest_fplate`; zebra close-ups
  `zebra_three`, `zebra_bonnet`, `zebra_apillar`, `zebra_side`,
  `zebra_cpillar`, `zebra_deck`, `zebra_nose`, `zebra_tail`).
* `node --experimental-websocket crash-lab-check.mjs --car showcase [--size 1920x1080]`:
  the crash lab's presets with the skinned body (`--size` sets the
  screenshots' size, 1280x720 by default).

## What the visual binds to (`Built`)

* `shell`: rings, surface labels (`seg`), the mirror map, skin triangles. The
  shell is the skin tier now; its spacing and meaning are unchanged.
* `skeleton.parts` and each node's part: `body`, `bumper_f`, `bumper_r`, `bonnet`,
  `bootlid`, `fender_l/_r`, `door_f_l/_r`, `door_r_l/_r`, `radiator`, the blocks
  (`powertrain`, `seat_fl`, `seat_fr`, `seat_rear`, `battery`), `suspension`.
* The windscreen's base sits at z 3.33 (it was 3.50) and its top at 2.55
  (2.70): `profile.top`, `look.windscreen`, the glass zone, the screen pillars,
  header, cowl, dash, column and the front side glass moved back 0.15 m (a
  longer dash-to-axle); the hinge pillars stay at the doors' front edge.
* `groups` and `bind_tris` keyed by member, block and corner name, as before
  (`seat_fl`, `seat_fr`, `seat_rear`, `dash`, `tunnel`, `column` kept). A member's
  group may now include nodes it shares with the shell or another member (snapped),
  and its catchers lie on one line of each bay. New groups: `bow_r` (a rear roof
  bow), `door_belt_f/_r`, `door_mid_f/_r` (the doors' belt and inner rails, with
  `_r` mirrors); `floor_brace1`–`4` are gone (lumped shear panels instead).
