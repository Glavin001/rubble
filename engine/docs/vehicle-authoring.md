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

## Tests and tools

* `cargo test --release -p engine-softbody --test showcase_car -- --nocapture`:
  counts, closed skins, mass, symmetry, the visual binding, and the scenes (rest,
  wall, side pole, centre pole, roll-over drop); `--ignored` adds the pole sweep.
* `cargo test --release -p engine-webgpu --test showcase_car_gpu`: GPU against CPU.
* `cargo run --release -p engine-softbody --example showcase_probe -- sedan low wall 56`
  (`VISUAL=<dir>` renders the skin before and after, `DRAW=<dir>` the beams).
* `showcase-car.html?res=medium&scene=wall&speed=56`: the car on the GPU in a page.
