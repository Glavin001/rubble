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
| `frame_every`, `frame_grade`, `frame_through` | the shell's braced frame (below; the sedan has none: `frame_every: 0`) |
| `shell_braces` | X-brace the shell's quads (off on the sedan: its triangles hold their shape; panels keep theirs as hinge beams) |
| `braces: Vec<BraceSpec>` | lumped shear panels: long diagonals (and mirrors) across the floor, roof, firewall, rear bulkhead, door rings and screens |
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
share of its length, as the metal does) and the density. The sedan's grades are
what the 0.5 ms step allows at its kerb mass (`Built::stiffness_scale` ≈ 0.8–0.9).
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
attach, ride, section, station, catch }`: a box (4 nodes a station) or ladder (2)
truss along a path at 35–45 cm stations (crush members 30 cm), its grade's
thresholds graded from `scale[0]` to `scale[1]` (a crush rail: weak at the bumper,
strong at the firewall), weaker rings every `beads` stations (it folds there
first), tied to the nearest skin nodes (`attach`) and, at its ends, to the members
it meets. A station node within 5 cm of an existing node of its part reuses it
(`members::SNAP`). Impact members (`catch`: bumper beams, door beams, sills,
B-pillars, firewall, floor cross members, tunnel, front subframe) get catcher nodes
along one line of each bay at most 11 cm apart: static contact is per node, so a
20 cm pole always meets one.

Blocks (`BlockSpec`) are rigid lattices with their mass; `bear` (m) adds
compression-only beams from the block's rear and side faces to the structure behind
and beside it (an engine driven back meets the firewall and tunnel).

## Body damping

Each beam's damper damps that beam's own mode (hundreds of hertz) but barely the
body's (10–30 Hz: bending, torsion, panels), which would ring for seconds after a
crash. A real body's joints, sealer and trim damp those; here each node gets one
dashpot (k = 0, `body_damping` N·s/m) to a node of its part about
`body_damping_reach` m away along an axis, torn off past `body_damping_strength` N.

## Mass and the explicit step

Nodes carry their own mass (`l, kg` in the `.truck`): named masses where they were
put, the rest of the kerb mass spread by beam length × density. No node is lighter
than its beams need for a stable 0.5 ms step: a Gershgorin bound on h²ω² + 2hγ per
node (`Skeleton::floors`). When the floors outweigh 95 % of the kerb mass, every
beam's stiffness, damping and thresholds are scaled down to fit
(`Built::stiffness_scale`). At a fixed mass and step, more nodes mean softer beams,
and a line's axial rigidity grows with the square of its node spacing: so the body
is sparse (1,250–1,450 nodes of 0.5–2 kg), with node lanes of 11 cm only on the
impact faces. `SHOWCASE_FLOORS=1 … showcase_probe -- sedan medium info` prints the
floors per group and what each kind of beam costs.

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
  counts, closed skins, mass, symmetry, the visual binding, materials (stiffness
  scale, k·L and ζ medians), stillness at rest, torsion (beside the E36 when
  `engine/local-vehicles` has it), a door point load, and the scenes (rest, wall,
  side pole, centre pole, roll-over drop, 120 km/h wall, door pull, settling after
  the wall); `--ignored` adds the pole sweep (`SWEEP=front|side`, `SWEEP_KMH=60`
  for a subset) and the crumple-zone damage report.
* `cargo test --release -p engine-softbody --test showcase_rest_wobble -- --ignored --nocapture`:
  the materials and rest wobble beside real Rigs of Rods cars.
* `cargo test --release -p engine-webgpu --test showcase_car_gpu`: GPU against CPU.
* `cargo run --release -p engine-softbody --example showcase_probe -- sedan low wall 56`
  (`VISUAL=<dir>` renders the skin before and after, `DRAW=<dir>` the beams;
  scenes `wobble`, `settle`, `torsion`, `pole`; `BREAKS=1` sums broken, bent and
  elastic energy per member pair, `WHEEL=<i>` traces a wheel's knuckle).
* `showcase-car.html?res=medium&scene=wall&speed=56`: the car on the GPU in a page.

## What the visual binds to (`Built`)

* `shell`: rings, surface labels (`seg`), the mirror map, skin triangles. The
  shell is sparse now (25–40 cm); its meaning is unchanged.
* `skeleton.parts` and each node's part: `body`, `bumper_f`, `bumper_r`, `bonnet`,
  `bootlid`, `fender_l/_r`, `door_f_l/_r`, `door_r_l/_r`, `radiator`, the blocks
  (`powertrain`, `seat_fl`, `seat_fr`, `seat_rear`, `battery`), `suspension`.
* `groups` and `bind_tris` keyed by member, block and corner name, as before
  (`seat_fl`, `seat_fr`, `seat_rear`, `dash`, `tunnel`, `column` kept). A member's
  group may now include nodes it shares with the shell or another member (snapped),
  and its catchers lie on one line of each bay. New groups: `bow_r` (a rear roof bow).
