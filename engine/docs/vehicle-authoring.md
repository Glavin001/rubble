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
paths' vertices) and each segment is split at most 60 cm apart (crush members 45
cm; `station` overrides). `stiff` is the section's k·L against its grade's (rails
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
  beams, subframes, the doors' own beams. About 350–480 nodes of 1–2 kg (the named
  masses sit on them), its grades at RoR-level k·L on the load paths. It carries
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

The crumple zone's plateau (`CRUSH_PLATEAU`, 0.65 × the crush grade) is as
low as the crush length allows: what the front still pushes on the cell when
the car stops comes back elastically as rebound (about F²/2k), so a lower
plateau with a longer crush (0.75 m at 56 km/h) rebounds less (2.7 → 2.0
m/s). The front rails are graded from 1.0 under the floor to 0.5 at the
bumper (`RAIL_CABIN`).

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
* **Lower-door sculpture.** Under a feature line rising from 0.43 m behind the
  front arch to 0.60 m ahead of the rear one, the side steps in 2.4 cm with a
  crisp upper edge, easing out toward the sill (`BodyStyle::scoop`). A band row
  lies on the line, so its highlight runs straight.
* **Fillets.** Two extra band rows a few millimetres either side of the crease
  and of the feature line: the smooth normals turn within that fillet, so each
  line reads as a crisp line of light instead of a blur one row (2.2 cm) wide.
* **Bonnet.** Two power-dome creases run from 0.22 m off the centre at the nose
  to 0.60 m at the cowl (out toward the A-pillars), the dome 1.4 cm proud
  between them (`hood_crease`); a lid column lies on each crease. The bonnet's
  leading edge and the boot deck roll over in 4–5 cm (`nose_ramp`,
  `tail_ramp`), and the deck's sides in 10 cm (`deck_ramp`), instead of the
  greenhouse's long tumblehome ramp; the deck kicks up 2.6 cm into an
  integrated lip spoiler (`deck_lip`). These details are added after the lid
  is relaxed, so they stay crisp.
* **Fascia.** Raked: the widest line low (0.56 m), the upper fascia leaning
  14 cm back into the bonnet. A low trapezoid grille (0.66 m wide at the
  top, 0.89 m at the bottom, 0.20 m tall: the face's dark mass sits low) with a
  1.6 cm gloss-black surround and a honeycomb; slim full-LED headlamps (3.5 cm
  at the grille, 1.5 cm at the tip) just under the bonnet's edge, wrapping the
  corners to a pointed tip, with a DRL blade along their upper edge
  (`LampSpec::guide_above`); sharp side intakes (air curtains) in the corners;
  a gloss-black splitter blade standing 3 cm proud along the bumper's lower
  edge; no front plate (`plate_front_h` NaN).
* **Arch flares.** The front arches swell 3 cm over a 19 cm band, the rear
  haunches 4 cm over 24 cm (`rear_flare`), giving a plateau at the rolled lip
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
  deck are crowned.
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
| Body surface (`parts/body.rs`) | A *band* is swept round the plan outline: a section per arc length θ, blended between side, nose and tail shapes. The crease and lower line fall on fixed rows, so their highlights run straight. A *lid* (height field over plan view) runs from the band's top edge over a ramp (tumblehome, fender tops) to the profile's top line and crown. The ramp's round into the top is a quartic with zero curvature at both ends (`body::rail`) and the crown fades as (1 − u²)³, so no line of light runs along the roof rail or the fender tops. The rear deck meets the greenhouse in a 3.5 cm smooth max (the C-pillar's fillet into the quarter). The lid is relaxed by a Laplacian. The profile's curves are blurred (σ 8 cm) so the clearcoat shows no ripples at their control points. Grid: 2.2 cm across the sections, 3.5 cm along the car (twice that at the distant LOD). |
| Panels (`parts/trim.rs`, `design.rs`) | Each panel is a region (an SDF made of smooth cut lines in side, plan or arc view), made metric on the surface and clipped by marching triangles, so edges follow the curve, not the grid. Every cut is rolled (2.8 mm radius) into a 12 mm wall and a return flange, set back 2.2 mm, giving a 4.5 mm shut line. Arch lips get an 8 mm roll. The inside of a panel renders dark. Panels: bonnet, cowl (black), lip and bumper front, fenders, doors, deck lid, valance and bumper rear, sills (black), body. |
| Glass | The body surface over each opening, set 6 mm in, with a black frit band printed inside the windscreen and backlight. The side windows sit in black sash frames, with a black B-pillar applique (one half on each door) and a chrome belt moulding. The windscreen is laminated (it cracks); the side glass is tempered (it shatters). The rear doors', quarter lights' and backlight's glass is privacy-tinted (`mat::PRIVACY`). |
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
  (`ShowcaseVisual::rigid`): they move rigidly and never stretch.
  * Lamps, mirrors and handles.
  * The tempered side glass and backlight, which shatter whole when that
    triangle strains.
  * The laminated windscreen is the exception: it follows its frame and
    spider-cracks with strain (the `screen` material).
  * Mounted parts (all of the above) also keep an anchor per vertex: the
    nearest skin triangle and the vertex's place on it
    (`ShowcaseVisual::anchor`). Where that mount has moved more than 4 cm from
    where the rigid part holds the vertex (`ANCHOR_BREAK`: the bumper under a
    lamp crushed), the part breaks there instead of floating free.
* **Cabin.** Seats and the steering wheel ride one triangle of their frame
  rigidly (`Bind::GroupRigid`), so a far binding triangle never swings them
  out of the cabin; the dashboard and console ride their members' triangles.
* **Folds.** A back face whose skinned normal still faces the eye is a
  sliver folded over in a crush, not a panel's inside: it shades as paint.
* **Tears.** The renderer keeps a per-node tear field: a skin triangle's
  perimeter against its rest perimeter (`node_rest`), interpolated over each
  vertex's binding nodes. Panels are cut where it crosses ½, a smooth line,
  with a bare-metal band at the edge.
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
  `?reflect=0.3` sets it). It costs about 0.1 ms a car (10 cars: 4.97 ms
  against 3.96 ms without); the
  crash lab's backdrop is a navy cyclorama with a lit horizon band. The paint
  deepens toward grazing (flop); tail lenses are a deep ruby; tyres satin. The clearcoat has a mirror-sharp sun glint over
  a broader sheen. Diffuse light is an analytic lobe fitted to the studio.
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
    (`showcase-car.html?perf=1&n=10`): about 0.40–0.43 ms per car with the
    interior (about 300k triangles, 99k at the distant level), and 0.21–0.34
    ms at the distant level. Another GPU user moves these by ±10 %.

To restyle, change `design::sedan_styling` (cut lines, outlines, section
shapes) and the profile. The tools stay the same.

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
  `rest_interior`, `rest_dash`, `rest_screen`, `rest_fplate`).

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
