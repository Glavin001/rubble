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
| `grades: Grades` | materials: `panel`, `floor`, `frame`, `crush`, `hsla`, `uhss`, `boron`, `alu`, `plastic`, `glass`, `rigid` |
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
| `frame_every`, `frame_grade`, `frame_through` | the shell's braced frame (below) |
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
along it (spacing per surface from the `Resolution`), caps both ends with domed rows,
and braces it: every triangle edge, both diagonals of every quad, bend beams that skip
a node. Every few rings (`frame_every`) about 14 hard points round the section are
joined into a tube of long stiff beams (`frame_grade`): the body's deep section.

### Parts and joints

A `Region` picks shell nodes: `Side(outline)` in side view (z, y), `Top { z0, z1 }`,
`Cap { front, y0, y1 }`, `Box`, `Floor`, or `Any(…)`. The first panel whose region
holds a node owns it (`sided` panels become `<name>_l`, `<name>_r`). Beams across a
seam take the panel's `Joint`: `Weld`, `Bolt { scale, group }` (a detacher group makes
the panel leave whole), or `Hinged { hinge, latch… }` (strong hinges along one edge, a
latch zone on the other, weaker flange beams elsewhere).

### Materials

`Material::grade(k, damp, yield_strain, break_ratio, density)`: stiffness of a 10 cm
beam (it scales as 1/length), the yield force as a strain of the beam's axial rigidity
(the same strain at every length), and the break force as a multiple of it. The
explicit solver keeps every yielded rest length ≥ 0.1 m, so beams of 10 cm cannot
yield in compression: the skin is soft and the trusses (stations 0.2–0.4 m) carry the
loads and absorb the crash.

### Members

`MemberSpec { path, width, height, up, grade, scale: [start, end], beads, mirrored,
attach, ride, section, station }`: a box (4 nodes a station) or ladder (2) truss along
a path, its grade's thresholds graded from `scale[0]` to `scale[1]` (a crush rail: weak
at the bumper, strong at the firewall), weaker rings every `beads` stations (it folds
there first), joined to the nearest skin nodes (`attach`) and, at its ends, to the
members it meets. Bays longer than 11 cm get catcher nodes between stations: static
contact is per node, so a 20 cm pole always meets one.

## Mass and the explicit step

Nodes carry their own mass (`l, kg` in the `.truck`): named masses where they were
put, the rest of the kerb mass spread by beam length × density. No node is lighter
than its beams need for a stable 0.5 ms step: a Gershgorin bound on h²ω² + 2hγ per
node (`Skeleton::floors`). When the floors outweigh the kerb mass, every beam's
stiffness (and its thresholds, keeping each grade's yield strain) is scaled down to
fit (`Built::stiffness_scale`): at a fixed mass and step, more nodes mean softer beams.

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

The proportions come from the package (`BodyProfile`): 4.70 × 1.82 × 1.445 m, a
2.70 m wheelbase, about 0.95 m of front overhang and 1.05 m at the rear. The
greenhouse sits a little rearward: the cowl is at about 0.28 L, the roof peaks
over the B-pillar, and a fast backlight runs into a short, high deck.

The character lines are:

* **Shoulder crease.** One crisp line, the body's widest point, runs from above
  the headlamp at about 0.74 m, level through the handles at about 0.80 m, and
  kicks up to the tail lamps at about 0.86 m (`BodyStyle::crease`). Above it the
  side leans in to the belt; below it the door is gently concave down to a lower
  line and a tucked-under sill.
* **Arch flares.** Each wheel arch swells 3 cm over a 17 cm band, giving a
  plateau at the rolled lip.
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

### Part list and how each is built

| Part | Construction |
|---|---|
| Body surface (`parts/body.rs`) | A *band* is swept round the plan outline: a section per arc length θ, blended between side, nose and tail shapes. The crease and lower line fall on fixed rows, so their highlights run straight. A *lid* (height field over plan view) runs from the band's top edge over a ramp (tumblehome, fender tops) to the profile's top line and crown. The lid is relaxed by a Laplacian. The profile's curves are blurred (σ 8 cm) so the clearcoat shows no ripples at their control points. Grid: 1.2 cm across the sections, 2 cm along the car. |
| Panels (`parts/trim.rs`, `design.rs`) | Each panel is a region (an SDF made of smooth cut lines in side, plan or arc view), made metric on the surface and clipped by marching triangles, so edges follow the curve, not the grid. Every cut is rolled (2.8 mm radius) into a 12 mm wall and a return flange, set back 2.2 mm, giving a 4.5 mm shut line. Arch lips get an 8 mm roll. The inside of a panel renders dark. Panels: bonnet, cowl (black), lip and bumper front, fenders, doors, deck lid, valance and bumper rear, sills (black), body. |
| Glass | The body surface over each opening, set 6 mm in, with a black frit band printed inside the windscreen and backlight. The side windows sit in black sash frames, with a black B-pillar applique and a chrome belt moulding. The windscreen is laminated (it cracks); the side glass is tempered (it shatters). |
| Lamps (`fixtures::lamp`) | A lens continuing the body surface; a housing extruded along the lamp axis; a reflector back; projector cups with lens domes; light guides that follow the outline (the DRL; three tail bars). |
| Grille (`fixtures::grille`) | Gloss-black surround, a dark throat recessed 7 cm, a back mesh, and slats cut from the opening at set heights (a chrome top bar) and given walls. There is also a lower intake. |
| Wheels (`fixtures::wheel`, `caliper`) | Tyre lathe: four tread grooves, rounded shoulders, a rim protector and lettering band. Rim: lip and barrel, a dished double-spoke face cut on a polar grid with 4 cm spoke walls, centre cap, five lug nuts. Vented disc with its hat. The caliper rides the axle (it does not turn). |
| Mirror, handles | An egg housing (paint above, black below) on an arm from the door's sail, with its glass; a chrome pull handle over a dark pocket. |
| Underbody and others | Arch liners, underbody plate, exhausts with chrome tips, a plate in a black surround, a chrome tail garnish. |

### Binding and damage

* **Voted pieces.** A piece binds to the physics part most of it lies on (a vote
  of the nearest single-part skin triangles). Where it reaches more than 2 cm
  past that part, those vertices ride the nearer part, and the piece's
  triangles are split there (duplicated vertices). So it parts along the
  physics seam instead of stretching.
* **Rigid pieces.** Lamps, mirrors and handles ride one triangle's orthonormal
  frame (`ShowcaseVisual::rigid`). They move rigidly and never stretch.
* **Tears.** A torn binding triangle (stretched 2.5×) is cut through the middle
  of each visual triangle, with a band of bare metal at the cut.
* **Normals.** Smooth normals follow the node normals. Where they disagree with
  the triangle's frame by more than about 15°, the faceted frame normal wins,
  so physics kinks read as creases.

To restyle, change `design::sedan_styling` (cut lines, outlines, section
shapes) and the profile. The tools stay the same.

## Tests and tools

* `cargo test --release -p engine-softbody --test showcase_car -- --nocapture`:
  counts, closed skins, mass, symmetry, the visual binding, and the scenes (rest,
  wall, side pole, centre pole, roll-over drop); `--ignored` adds the pole sweep.
* `cargo test --release -p engine-webgpu --test showcase_car_gpu`: GPU against CPU.
* `cargo run --release -p engine-softbody --example showcase_probe -- sedan low wall 56`
  (`VISUAL=<dir>` renders the skin before and after, `DRAW=<dir>` the beams).
* `showcase-car.html?res=medium&scene=wall&speed=56`: the car on the GPU in a page
  (`freeze=0` shows the rest pose, `perf=1` times the car's render pass);
  `node showcase-car-check.mjs --size 1920x1080 [--only rest_three,...]` takes
  headless screenshots.
