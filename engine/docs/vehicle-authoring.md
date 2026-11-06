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

### A8 and FMVSS 214 diagnosis (2026-11-05, not fixed)

At 30 km/h the side sweep fails 14 of 178 (cylinder 10, square 4): +1.60,
+1.90–2.05 (z 0.30–0.75), −1.15 to −1.25 (z 3.50–3.60), −0.60 and −0.05
(z 2.95, 2.40). `POLE_TOUCH=1` lists the nodes at the pole's surface each
frame:

* −0.05 (16 cm ahead of the B-pillar): the B-pillar's nodes leave the pole
  at 0.08–0.10 s; the front door (beam, belt, skin, its flange nodes) is
  carried into the cabin; the force falls from 80 to 20 kN at 0.6 m and the
  pole reaches 1.24 m. At −0.10 the force holds 80–90 kN and the pole stops
  at 0.54 m.
* +1.95 (rear overhang): only the bumper cover, boot lid, quarter skin and
  (with catchers) the rear rail touch it; 15–28 kN to 1.45 m.
* −1.20 (behind the front wheel): the wheel, apron, rail and suspension at
  10–45 kN; nothing transverse in the lane until the powertrain at 1.55 m.

Tried, none adopted (counts of the 178): catchers every 12 cm (16);
catchers on `rail_f`, `apron`, `rail_r`, `quarter_inner`, `roof_rail` (12);
those in the frame tier, with and without 12 cm (15, 12); a front subframe
rear cross member at z 3.52 or a boot floor cross member at z 0.40 (each
breaks a lumped brace while settling: braces pick the nearest frame node
and spread over its station, so a new member's node moves their ends;
`settle` now names the broken beams). The door flanges' bearing ×5
(`BEAR_X`, a measurement) changed nothing.

The MDB (`explicit_scenes`, `MDB_WHO=1`): residual crush 356 mm, deepest at
the rear door's lower rear corner (z 1.52; its beam node 265 mm); the front
door 74–99 mm. Its census on the struck half: the rear door's beam bearing
on its flange broke, the sill yielded (15) and broke (2). Sectioning the
C-pillar (70 × 100 × 1.4 mm DP780: chords 14 → 58 kN), its flange at
2.8 mm, the door frames ×2 or the bearings ×5 each left the crush at
356 mm to the millimetre; the door frames ×0.2 moved the maximum to the
front door (250 mm).

`member_chord_yields` (`--ignored`) lists each member's chord yield. Besides
the sectioned members (sill 56, B-pillar 77, tunnel 62, floor cross members
45–70 kN) the side's load paths are far below a real section: roof rails and
headers 4.2 kN (a ladder: 0.42 kN·m; a 70 × 90 × 1.4 mm DP780 rail is about
5.4 kN·m), C-pillar 14, rear rails 11.6, subframe cross members 4.3.

**Glazing in torsion** (`i_windscreen_share_of_torsion`, `--ignored`,
Medium): body-in-white 12.61 kN·m/deg; the windscreen pane's own beams and
bead carry 0.2 % (with `BRITTLE_GLASS=1` too); the lumped windscreen panel
(`windscreen_x`) 22 %; with the backlight's glass shear panel (parcel to
rear header) out as well, 3.07 kN·m/deg: the glazing carries 76 %. The
Ford P2000 measured 24.29 against 16.44 kN·m/deg with and without its
windscreen and backlight (32 %; via SAE 2000, Khaleel et al., "Effect of
Glazing System Parameters on Glazing System Contribution to a Lightweight
Vehicle's Torsional Stiffness and Weight"). So the body without glass is
too soft by about 3× and its lumped glass panels make up the difference.

### The body-in-white from real sections (2026-11-05, later)

**Brace ends.** Each lumped brace names the member its corners lie on
(`BraceSpec::on`, `sedan::on`); its end is that member's station nearest
the point. A member added elsewhere no longer moves a brace (Medium's
`.truck` was byte-identical after the change).

**Sections.** `side_sections` now also covers these members. Grades are the
classes in the 2011 Honda Accord teardown's material table (NHTSA DOT HS
811 666, fig. 113). Sections are typical, not one car's drawings (a Choice).

| Member | d × w × t (mm), grade | Chord yield (kN, was) |
|---|---|---|
| `roof_rail` | 70 × 90 × 1.4 DP780 | 54 (4.2) |
| `header_f`, `header_r`, `bow_b` | 60 × 100 × 1.2 HSLA | 17.7 (4.2) |
| `bow_r` | 40 × 80 × 0.8 HSLA | 6.0 (4.2) |
| `pillar_a` | 100 × 120 × 1.4 DP780 | 52 (23) |
| `pillar_c` | 70 × 100 × 1.4 DP780 | 48.5 (14) |
| `rail_r` (knee kept) | 100 × 130 × 1.6 DP600 | 47.5 (11.6) |
| `subframe_f0`–`f2`, `r1`, `r2` | 90 × 60 × 2.0 S420 | 44 (4.3) |
| `seat_x` (new, z 1.58) | 80 × 60 × 1.2 DP600 | 39.6 |
| `bumper_beam_r` | 50 × 100 × 2.5 AW-6082-T6 | 32.6 (2.9) |
| `dash` (IP beam) | tube Ø 60 × 1.6 S420 | 12.7 (14) |
| `door_beam_f`, `_r` | tube Ø 31.8 × 1.9, 1,200 MPa | 14.6 (19.5) |

`subframe_f0` is the front subframe's rear cross member (z 3.55). A boot
floor cross member at z 0.40 was tried and left out. It moved the
frontal's restitution to 0.202 and did not stop the rear-overhang
pass-through.

**Aperture frames.** `aperture_racking` gives a closed rectangular frame's
racking stiffness, V·h²/(24E)·[h/I_p + b/2·(1/I_t + 1/I_b)], and its
yield shear 4·M_p/h at its weakest corner member. Rigid corners are a
Choice. These openings now take their k·L and yield from it, in place of
the tuned `CABIN_YIELD` (60 kN) door rings:
- the windscreen opening (`screen_ring`, on the A-pillars' stations):
  k·L 2.76 MN, yield 11.5 kN;
- the front door ring (`ring_f_*`): k·L 5.5 MN, yield 19.6 kN;
- the rear door ring (`ring_r_*`): k·L 3.6 MN, yield 16.9 kN.

**Torsion and glazing.** Body-in-white torsion is 12.37 kN·m/deg with
glass and 10.99 without, so glazing carries 11.2 %:
- `windscreen_x` × `GLAZING` (2) contributes 5.9 %;
- the backlight shear panel contributes the rest.

Measured references:
- the 2011 Accord's body with windscreen, backlight and IP beam: 12.33
  kN·m/deg (DOT HS 811 666 §4.7.2);
- SAE 2000-01-2719 (Khaleel et al.), the primary source: glazing makes
  the body 12.4 % stiffer in torsion than without. The previous note's
  24.29 against 16.44 (32 %) is not in that abstract.

Published body-in-white figures run 8.5–16.4 kN·m/deg.

**Ensembles.** One explicit crash run is chaotic, so `n` and `o` are now
judged on eight runs: ±1 cm impact offset, ±0.3 % speed and ±0.2° yaw
(the cube's corners). The test asserts the mean and reports the spread.
At 1a87d4cd:
- `o` 292 ± 62 mm (range 197–403; band 297–445): fails;
- `n` restitution 0.2056 ± 0.007 (band ≤ 0.20): fails;
- `n` peak g 35.6 ± 2.0 and max crush 527 ± 19 mm.

NHTSA repeat tests of one make, model and year (vehicle database) scatter
much less:
- NCAP side pole: pooled SD 24 mm (CV 5 %);
- NCAP side MDB, sedans: SD 14 mm (CV 5 %);
- the 1982 Citation 56 km/h repeatability program (14 cars, SAE 840201):
  crush CV 3.7 %.

The model's pole spread is about 2.5 × a real car's.

**Step budget.** The two new members' frame nodes need 24 kg of stability
floor: `seat_x` 14 kg, `subframe_f0` 10 kg. Real parts weigh about 4 kg
each. The kerb mass is fixed, so the skin tier pays: Medium's skin scale
fell from ×0.44 to ×0.29. The skin keeps its yield forces, so its panels
are about a third softer for the same dent force. Wobble at rest went
from 0.3 to 0.7 mm.

**MDB against NHTSA's profiles** (`MDB_LEVELS=1`). The NCAP side MDB
reports of 19 sedans (exterior crush every 150 mm at 5 levels) give:
- level 1 (sill top, 242 mm): median 45 mm;
- level 3 (mid-door, 621 mm): median 204 mm;
- two peaks, on the front door at about 0.45 WB and on the rear door /
  C-pillar at about 0.73 WB aft of the front axle, falling to half at
  0.85 WB.

Ours is deepest low and far aft: 380 mm at 0.82 WB about 0.33 m above
the ground (the rear door's lower rear corner, z 1.52). That corner is
behind the sill's end (z 1.36), above the wheelhouse, and below the rear
door beam's end (y 0.58), so nothing bears on it.

**The corner's load path (2026-11-05, measured, not adopted).** After the
rear rails' tail (e34680b5) the MDB crushes 403 mm. The deepest node is the
rear door's skin at (0.94, 0.50, 1.52), 6 cm below the door beam's end.

The C-pillar is a ladder with one 0.67 m bay from the sill (y 0.31) to the
parcel shelf (y 0.98). Its door flange node at y 0.56 is held only by ties to
those two stations, so it gives like a string.

With a C-pillar station at y 0.56, the door-bearing height:
- residual crush 197 mm (band 98–219, target 126–206); the corner 30 mm;
- profile peaks at 0.70 WB (197 mm, rear door) and 0.34–0.52 WB (116 mm,
  front door), against NCAP's 0.73 and 0.45 WB;
- by level: sill top up to 216 mm at 0.6 WB (NCAP median 45), mid-door at
  most 56 mm (NCAP median 204, only 8 nodes at that level). The crush is
  still too low on the car.

But at `C_PILLAR_STIFF` ×6 the station costs the skin scale ×0.25 → ×0.16:
- Low or High no longer builds;
- `left_and_right_mirror`, `p` and `o` fail (`o` mean 219 mm, sd 101);
- `t` has 65 striker nodes inside.

At ×3.8, so that the per-beam k matches the old bay: skin scale ×0.23, MDB
327 mm, `o` 231 mm, `t` 72 inside, and `n`'s max crush (583 mm) and
restitution (0.207) fail.

The rear door's lower corner needs a C-pillar station, and the step budget
cannot pay for it at the frame's current k·L. The one-tier design (below)
frees that budget.

**A8 (30 km/h side), force against depth** (`POLE_CURVE=1`). The car's
kinetic energy is 50 kJ. Every failing case is deep crush at low force,
not a pole passing between nodes with no force:
- rear overhang (+1.95): about 15 kN mean to 1.8 m, 27 kJ. The rear
  rails' stations are 0.45 m apart (z 0.17, 0.62), so the pole meets no
  rail node in its lane;
- behind the front wheel (−1.15): 26 kN mean, 94 kN where it meets the
  powertrain at 0.8 m;
- at the B-pillar (0.0): 280 kN at 0.4 m, then about 20 kN to 1.23 m;
- −0.05: 28 kN mean.

A 100 kN structure would stop the car's 50 kJ within about 0.5 m.

No measured pole impacts at the front wheel or rear overhang with crush
depths were found. SAE 2006-01-0062 (Taurus into a pole) reports the front
axle region as much stiffer than the B-pillar, qualitatively.

**Rail stations at a pole's width** (2026-11-05). A real rail is continuous.
Static contact is per node, so a rail meets a pole only if its stations are at
most the NCAP pole's 254 mm apart in the pole's lane.

The rear rails' tail behind the rear subframe is now its own member,
`rail_r_tail`, with stations at z 0.62, 0.40 and 0.17. It keeps the knee's
thresholds, graded to 0.5 at the bumper, at the HSLA grade's k·L (a Choice).
Tried and not adopted:
- the same stations at ×3.5: skin scale ×0.05, the step budget gone;
- the tail at ×1.7, the old segment's per-beam k: skin scale ×0.23, and `o`
  (260 mm) and `p` fail.

30 km/h side sweep (Low) at each stage:

| Stage | Passed through, of 178 |
|---|---|
| Before | 14 |
| Tail stations | 8 |
| Tail stations and a `rail_f` station at z 3.55, which `subframe_f0` shares | 8 |

With the tail stations, the rear overhang (+1.90 to +2.05) no longer passes.
With the `rail_f` station as well, the front-wheel lane (−1.15) holds, but `n`
fails on t_m (77.5 ms) and peak g (32 g), so that station is left out.

Still passing through:
- +1.60 and +1.65 (z 0.75, between the rear rail's 0.62 and 1.03 stations,
  the rear wheel). A station at 0.82 at ×3.5 runs the step budget out
  (frame scale ×0.98).
- −0.55 to −0.60 (front door).
- −0.05 (B-pillar).
- −1.15, cylinder only (front wheel).

Medium after the tail: skin scale ×0.25 (was ×0.29), 1,053 nodes.

**Glass** (`BRITTLE_GLASS=1`):
- the windscreen cracks in every side pole run but not in `q` or `n`;
- with it on, `f` (energy rose 21 J at 120 km/h), `o` and `t` (15 nodes)
  fail.

Brittle glass stays off.

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
`NN=1 … info` lists each part's skin nodes, mass and nearest-neighbour
spacing; `GROUP_NEAR=<member> … info` lists a member's frame nodes and the
nearest frame node of another member (a station it could share).

### The skin is already at BeamNG's density (2026-11-05)

Medium's skin against the BX (`beamng-realism-plan.md` §1.3), median
nearest-neighbour spacing:

| Part | Ours: nodes, spacing | BX: nodes, spacing |
|---|---|---|
| body skin | 248, 0.21 m (0.16–0.26) | 145, 0.19 m (the BX's body is its frame) |
| front door | 19, 0.16 m | 11, 0.25 m |
| rear door | 11, 0.24 m | 11, 0.25 m |
| bonnet | 20, 0.21 m | 26, 0.24 m |
| fender | 8, 0.20 m | 13, 0.19 m |
| front bumper | 12, 0.24 m | 33, 0.12 m |
| boot lid | 10, 0.29 m | 18, 0.23 m |

The extra nodes are in the frame (430 nodes, 863 kg) and the catchers (108).
Coarsening the skin (`SKIN_DENSITY`) frees little:

| `SKIN_DENSITY` | Nodes | Skin scale |
|---|---|---|
| 1.0 | 1,045 | ×0.29 |
| 1.2 (Low's spacing) | 979 | ×0.34 |
| 1.4 (front bumper 2 nodes) | 924 | ×0.43 |
| 1.6 | does not build | |

At skin scale 1, the shell's own beams need about 48 kg of stability floor.
The skin tier's floor sits mostly in the skin-to-frame ties (about 69 kg) and
in the catchers' and members' skin-tier beams. Coarsening was not adopted.

Panel stiffness: a front door skin beam is 2.7e4 × 0.29 / 0.16 m = 4.9e4 N/m
on 0.18 kg nodes. The BX door skin is 1.1 MN/m on 1 kg nodes, 22× stiffer.
Real 0.75 mm sheet as a membrane (k·L ≈ (√3/2)·E·t·a, about 4e7 N) is beyond
any node–beam car.

**Sharing stations is not cheaper here.** `seat_x` costs the skin scale
×0.29 → ×0.38 when left out, and `subframe_f0` costs ×0.29 → ×0.36. Their
frame nodes weigh 1.0–2.2 kg, as every frame node does.

Putting a station on the members `seat_x` crosses, so that it shares their
nodes, costs more than it saves:
- a sill station at z 1.58 takes the skin scale to ×0.18;
- a `rail_r` station at z 1.58 takes it to ×0.17.

A `rail_f` station at z 3.55, which `subframe_f0` shares (4 of its 6 nodes),
costs ×0.02 and was adopted (A8 below).

### Proposed: one tier, as BeamNG's cars are (design only, not built)

The BX's body shell is its frame: 145 nodes, 329 kg, median 1.7 kg, at
0.19 m. Ours splits it into a 430-node frame and a 248-node skin of
0.1–0.2 kg nodes, joined by about 840 ties. The ties and the catchers eat
the step budget that the panels need.

**Design.**
- **Shell nodes are the frame's outer chords.** A member drawn on the body
  (sill, rail, pillar, roof rail, cross member) is a ladder, not a box: its
  outer chord is the shell's own nodes along its path, and its inner chord is
  a row of frame nodes 70–140 mm inboard, its section's depth.
  - Members' stations are the shell's ring stations, every 0.20–0.25 m.
  - No ties between skin and frame. No catchers: every shell node is heavy
    and within a pole's width (254 mm) of the next.
- **Inner-only members** (tunnel, cross members, subframes, aprons) stay as
  they are.
- **Doors** follow the BX: an 11-node skin, about 10 inner nodes (beam,
  belt, mid rail, which carry the door trim's mass), and the existing
  hardware groups.
- **Mass** is the kerb mass spread by stiffness need, with one fit for every
  beam (`fit_stiffness`, not `fit_tiers`).

**Expected numbers** (estimates from the Gershgorin bound,
k ≤ 4.8·m / (2·n·h²) ≈ 1.6 MN/m per kg at 6 neighbours):

| | Now (Medium) | One tier |
|---|---|---|
| nodes | 1,045 | about 740 (body about 450, doors 4 × 21, panels about 70, powertrain, suspension, seats) |
| body node median | skin 0.10 kg, frame 1.41 kg | 1.5–2 kg |
| panel beam k | 4.9e4 N/m | 1.0–1.5 MN/m (BX parity) |
| ties, catchers | about 840, 108 | 0 |

**What it changes.**
- **References.** Every reference is re-measured (a–v, n and o ensembles,
  MDB, sweep, torsion). Members keep their sections' chord yields (the
  thresholds are forces), so the side and roof strengths carry over. The
  frontal's crush zone, restitution and the rest-wobble figures will move.
- **Hardware and parts.** Hinge, latch and bolt groups, the rig sizing
  (`size_hardware`), the shut-line breaks and the collision lattice are
  unchanged, because they work on nodes and parts. Parts keep their own
  nodes. Door and lid frames lose their skin-tier ties to their skins.
- **A8.** Pole lanes meet a heavy shell node every 0.20–0.25 m. The rear
  overhang and wheel lanes no longer depend on members' stations.

**Effort.** Large: 3–5 agent sessions.
- `members.rs`: ladders on shell paths, snapping to ring stations.
- `vehicle.rs`: one tier, no ties or catchers.
- `sedan.rs`: member paths redrawn on ring stations.
- All tests re-baselined.

## The one-tier build (`sedan1`, 2026-11-05, in progress)

`style("sedan1")` (`sedan::sedan_one_tier`, `VehicleDef::one_tier`) builds the
same shape, members, parts and hardware as `sedan` in one tier, as BeamNG's
cars are built: the body shell's nodes are the frame's. `sedan` (two tiers)
stays the default; its `.truck` is byte-identical to before at every
resolution.

- **Shell.** Every spacing is at most `POLE_LANE` (0.24 m, under the
  254 mm pole), so a pole always meets a shell node.
- **Members on the body** (`attach > 0`) snap their stations onto shell nodes
  within 15 cm (`SNAP_FRAME`). The shell in their lane, within
  `LANE_REACH` (8 cm) past half their section and within 37° of their
  path (`LANE_COS`), becomes their outer chord: those shell beams take the
  member's chord material. Each lane node is joined to the member's two
  nearest station nodes as its webs. Both reaches are Choices.
- **No catchers, no skin-to-frame ties, no backing ties.** A station with no
  node on the shell is joined to it by beams of their own length.
- **Exact lengths.** `rubble_exact_lengths` is written; the rest pose is
  settled and its rest lengths taken on the exact path.
- **GPU.** `showcase_car_gpu::one_tier_at_rest_step_parity` (Low): every
  node's position and velocity matches the CPU's, bit for bit, after each of
  2,400 steps.

### Where we differ from real material, and why

**Masses are the real local mass. No mass is moved to meet the step's
bound.** `one_tier_masses` gives every node what is physically there.

| Item | kg | Source |
|---|---|---|
| Body-in-white, painted, with sound deadening | 339.0 | 2011 Accord, NHTSA DOT HS 811 666 §5.9 (328 + 12) |
| Front door, each | 29.5 | ibid. fig. 161 (58.99 the pair, with glass, regulator, trim, mirror) |
| Rear door, each | 23.7 | ibid. fig. 171 (47.46 the pair) |
| Hood | 17.9 | ibid. §5.10.4 |
| Decklid | 12.4 | ibid. |
| Fender, each | 5.1 | the closures' 147 kg less the rest, a pair |
| Bumper fascias | 2.4 / 1.7 | 3 mm at 1,000 kg/m³ over their area (`panel_sheets`) |
| Body glass (windscreen 4.2 mm laminated, backlight 3.5 mm tempered) | 22.8 | area × thickness × 2,500 kg/m³ |
| Named masses, blocks, hubs, arms | 609.0 | as in `sedan` (driver, seats, fuel, powertrain …) |
| Tyres and rims | 88.0 | as in `sedan` |
| Equipment spread over the body's shell by area | 240.1 | what the 1,450 kg kerb mass leaves: wiring, HVAC, carpet, sealer, lamps |

- Inside a part, the measured total is spread by the part's own steel:
  - its sheet's area × gauge × 7,850 kg/m³: the body at `SHELL_GAUGE`
    0.8 mm, the panels at 0.75 mm, the panes at their glass;
  - its members' length × their section's steel per metre
    (`MemberSpec::kg_per_m`, from `side_sections`, else `DEFAULT_GAUGE`
    1.2 mm round the truss);
  - a node on neither (a flange strip, a bracket): its beams' half-lengths
    of a 50 × 1.2 mm strip (`STRIP_KG_PER_M`).
- The factor from the sections and sheet to the measured total is printed
  per part (Medium):
  - body ×0.87;
  - doors ×1.5–1.7: the trim, glass and regulator ride the door's steel;
  - fenders ×1.9;
  - fascias ×0.48.
  These are Choices of how the measured mass is spread within the part.
- Named masses sit evenly on the members' nodes in their box (where they
  bolt), not on the nodes whose bound wants them.
- `one_tier_masses_are_real_and_beams_soft_only_at_the_bound` fails if any
  node's mass differs from its local mass by more than 0.1 g
  (`MOVED_MASS_TOLERANCE`, the f32 sum's rounding). **Mass moved: none.**

**Stiffness is real unless the step's bound requires less.** The bound is the
explicit step's Gershgorin row (h²ω² + 2hγ within `G_LIMIT` at 0.5 ms). It is
numerics, and BeamNG's content obeys it too (its BX door skin is about
1.1 MN/m on 1 kg nodes, against roughly 4e7 N/m of real sheet).
`one_tier_real_stiffness` first sets each beam's target to its real stiffness:
- a member's beams (and its lane on the shell): E·A/L, with A its
  section's steel shared by its chords (4 a box, 2 a ladder). Its webs are
  taken as the same sheet, a Choice;
- a sheet's edges: (√3/4)·E·t, an isotropic membrane as an equilateral
  triangular lattice (Poisson's ratio 1/3). The fascia's E of 1.5 GPa is a
  Choice in polypropylene compound's 1–2 GPa.

Everything else keeps the stiffness it was built with: seals, ropes,
bearings, mounts, gauges, and the lumped shear panels (whose racking comes
from exact frame physics).

`Skeleton::cap_to_bound` then lowers stiffness only where a node would
exceed the bound. The cut is max-min fair: at a saturated node its stiffest
beams come down to a common level, beams under that level keep theirs, and a
final pass raises any beam that still has room at both ends. The tyres' spoke
springs are capped the same way (the spoke rate is the tyre model's, not a
measured rate). Damping keeps its ratio. The test fails if a beam is softer
than its target without one of its nodes within 5 % of the bound (the rest
pose's few-centimetre settle changes lengths), or if any node is past it.

Medium, 6,928 beams: 5,055 are at the bound. Built/real k per member, median
over its beams, from the test:

| Member | real k (N/m) | built k (N/m) | built/real | lighter node (kg) |
|---|---|---|---|---|
| `rail_f` | 1.44e8 | 2.99e5 | 0.0021 | 0.98 |
| `sill` | 1.37e8 | 4.57e5 | 0.0033 | 1.00 |
| `pillar_a` | 8.13e7 | 2.16e5 | 0.0027 | 1.01 |
| `pillar_b` | 1.07e8 | 2.21e5 | 0.0021 | 0.64 |
| `roof_rail` | 1.51e8 | 2.06e5 | 0.0014 | 0.87 |
| `tunnel` | 1.40e8 | 5.54e5 | 0.0040 | 1.24 |
| `floor_x1` | 1.29e8 | 1.00e6 | 0.0078 | 1.88 |
| `dash` | 7.01e7 | 1.92e6 | 0.0273 | 2.54 |
| `door_beam_f` | 4.95e7 | 4.38e5 | 0.0088 | 0.79 |
| `bumper_beam_f` | 1.45e8 | 2.82e5 | 0.0019 | 0.28 |

Every member is at 0.001–0.04 of its real k, because of the bound at its
nodes' real masses. Skins (median beam k, against the BX's door skin of
1.1 MN/m on 1 kg nodes):
- body: 6.6e5 N/m on 1.31 kg nodes;
- doors: 2.7–3.0e5 N/m on 0.55–0.62 kg nodes, so per kilogram of node about
  2× under the BX's;
- fenders: 3.8e5 N/m;
- bonnet: 9.2e4 N/m;
- boot lid: 1.6e5 N/m.

**Strengths stay real forces.**
- Members keep their sections' chord yields (`side_sections`).
- A sheet edge yields at its sheet's yield strain, (√3/4)·σ_y·t·L, with
  σ_y = 210 MPa (`SHEET_YIELD`: BH210 and DC04-class outers and inners, a
  Choice); a fascia at 25 MPa.
- After the cut, each beam's break force is its yield + elong × its built
  k·L, so it tears at the same elongation as before.

**What this costs (measured, open).** A beam's elastic energy at a force F
is F²/2k. With k at 0.1–4 % of real and F real, a member stores 25–1,000×
the elastic energy a real one does before it yields.

The NCAP 56 km/h frontal (`n`, Medium, after the lanes):
- in band: none of max crush, t_m, peak or restitution;
- max crush 356 mm, against 380–580 mm;
- t_m 57 ms, against 63–76 ms;
- peak 60 g, against 34–47 g;
- restitution 0.27, against 0.13–0.20.

Before the lanes, max crush was in band (394–456 mm).

`ESPLIT` shows where the rebound is stored (in kJ, at the frame of most
elastic energy, about 50 ms):
- cabin members to doors (the doors' bearing on the pillars): 11–20;
- the cabin's own members: 7–11 (the cell yields at `CELL` × its grade, so
  it stays elastic);
- cabin to front structure: 8;
- the front structure: 5, about half of it in yielded beams.

The crush zone is not where the spring is. The passenger cell and the door
ring hold up to 52 kJ elastically and give it back.

### Real crush, damping and heavier nodes (2026-11-05, later)

The user's decision on the rebound above: real crush mechanics, BeamNG's
damping, and fewer, heavier nodes, with strengths kept real. All Medium,
8-run ensembles where a reference has one.

**Damping (measured, not tuned).** `examples/jbeam_damping.rs` measures each
beam's damping ratio on its own two-mass mode,
ζ = beamDamp / (2√(beamSpring·m₁m₂/(m₁+m₂))), on the BX's default
configuration:

| Part (BX) | ζ median (p10–p90) | sedan1 before |
|---|---|---|
| body (`bx_body`) | 0.0275 (0.018–0.052) | 0.069 |
| front rails (body, `f*` frame nodes) | 0.0177 (0.015–0.018) | |
| bumper bar | 0.0158 | |
| doors | 0.0334 (0.025–0.040) | 0.081–0.091 |
| bonnet | 0.0593 (0.045–0.080) | 0.112 |
| boot lid (tailgate) | 0.0567 (0.038–0.065) | 0.125 |
| fenders | 0.0422 (0.039–0.053) | 0.084 |
| bumpers (fascias) | 0.089 | 0.125–0.138 |
| windscreen | 0.0200 (0.017–0.026) | |
| door glass | 0.0545 (0.046–0.057) | |

- `DampingRatios` sets every part beam (not the fixed mounts or the
  suspension) to its part's median, c = 2ζ√(k·μ) at its target k, before
  `cap_to_bound` (which keeps ζ).
- The body's dashpots take the BX's damping-only beams' rate: 12 beams of
  500 N·s/m over 328.9 kg, 18.2 /s per kilogram of the node each is from.
- **The damping bound.** sedan1's worst h·Σc/m was 1.29, past `C_RATIO`'s
  0.75, at a light node that 16 nodes' dashpots chose as their partner. With
  a fan-in of 2 and the BX ratios it is 0.65. The one-tier test asserts it.

**Crush members.**
- A Rubble `.truck` extension, `rubble_jbeam_plastic`
  (`jbeam-semantics-spec.md` §18.3), gives beams JBeam's deformLimitStress
  and deformLimit. The loader makes them JBeam NORMAL beams, which the CPU
  and GPU dispatch per beam.
- `CrushSpec` (`one_tier_crush`): every bay's cut (the member's beams
  crossing the plane between two stations, Σ yield × |cos|) yields at
  Abramowicz and Jones's mean progressive crush force,
  P_m = 13.06·σ₀·t^(5/3)·C^(1/3).
  - `rail_f`: 80 × 120 × 1.6 mm DP600, 61.7 kN (the section is a Choice).
  - `rail_r_tail`: 100 × 130 × 1.6 mm, 64.6 kN.
  - No hardening: deformLimitStress = the yield. Stroke: deformLimit 0.73
    (the effective crushing distance).
  - The bumper beams keep their bending yields and fold without hardening.
- The front bumper beam is the Accord's: roll-formed steel, 590 MPa tensile,
  assembly 7.96 kg (NHTSA DOT HS 811 666 §5.10.7.1), as a 60 × 100 × 1.6 mm
  box (a Choice). Its yields come from its M_p. It shares `rail_f`'s
  end-station nodes.

**Sheet buckling.** A thin bay in compression buckles long before it
yields. A sheet edge's yield is its bay's ultimate stress
(`sheet_ultimate_stress`):
- σ_cr = 4π²E/(12(1 − ν²))·(t/b)² (k = 4, a Choice), then Winter's
  effective width, σ_u = σ_y·√(σ_cr/σ_y)·(1 − 0.22√(σ_cr/σ_y)), with b the
  edge's length;
- 0.8 mm on 0.2 m bays: 48 MPa against σ_y 210;
- in compression only. Our body skin beams now yield at about 7 kN in
  compression; the BX's median (one beamDeform both ways) is 8 kN.
- **In tension (2026-11-05, later)** a sheet does not buckle: it carries a
  tension field at its own yield, (√3/4)·σ_y·t·L (about 14.5 kN on a
  0.2 m edge of 0.8 mm). JBeam's documentation has no expansion-side deform
  force (beamDeform is one value; deformLimitExpansion limits the plastic
  change, anisotropic and bounded beams split stiffness and damping, not
  yield), so this is Rubble's `.truck` extension `rubble_tension_yield`
  (`jbeam-semantics-spec.md` §18.4: the solver's Tt, which it already kept
  apart from Tc; no kernel change; GPU parity bit for bit in
  `tension_yield_gpu`). A sheet edge's break force is then its tension yield
  plus its elongation × its built k·L. `TENSION_YIELD=0` builds the old way
  (a measurement switch).
  - Effect, sedan1 Medium, ensembles (before → after): `g` the door's skin
    no longer tears before its hardware (4 edges did, at 0.4–1.1 kN); the
    latch lets go at 36.2 kN (was 46.3), the door is off at 51.9 kN (47.0);
    `n` max crush 644 → 633 mm, peak 31.8 → 32.8 g, b1 22.83 → 22.86,
    t_m 75.4 → 74.4 ms, e 0.175 → 0.174; MDB 177 → 186 mm (in band);
    `p` peak 18.9 → 22.5 kN; `q` 6.99 → 8.59×; `o` 270 → 206 mm; `h` the
    fifth second's ringing peak 11.6 J against the fourth's 10.9 J (fails;
    it passed); `t` 14 leaked either way.
  - **Why `o` fell: springback.** The side pole's dynamic crush (deepest
    exterior node, `O_DYN=1`) is 332–388 mm in both builds, inside the
    measured residual band (297–445), but about half of it springs back:
    residual 176–310 mm (was 229–299). A beam that yields keeps an elastic
    stretch of its yield force over its k (`explicit-beams.md` §2, plastic
    coefficient 0), and the step's bound puts k at 0.1–4 % of real, so it
    springs back 25–1,000× as far as steel's σ_y/E. The tension yield doubles
    the skin's yield force in tension, so its elastic stretch and springback
    double too (about 14 % strain on a 0.2 m body edge before it yields).
    The plastic coefficient is not a fix: at p → 1 a yielding beam's rest
    length follows it and it stops carrying its yield force while it flows.

**Yield strain: BeamNG's against ours (2026-11-05, measured).**
`examples/jbeam_yield_strain.rs` takes each beam's yield strain
ε_y = beamDeform / (beamSpring·L) and the springback it implies after a
plastic strain ε_p, ε_y/(ε_y + ε_p), per part, for the BX's default
configuration and for sedan1 (`SEDAN1=medium`; tension and compression):

| Part | BX ε_y p50 (p90) | BX k p50, L p50 | sedan1 ε_y p50 (p90) | sedan1 k p50, L p50 |
|---|---|---|---|---|
| body (BX: its rails too) | 1.0 % (4.9) | 1.7e6 N/m, 0.56 m | 4.3 % compr., 6.6 % tens. (33) | 6.4e5 N/m, 0.38 m |
| doors | 1.25 % (4.8) | 1.1e6, 0.58 | 2.8 / 6.4 % (18 / 27) | 6.3e5, 0.31 |
| bonnet | 5.5 % (24) | 5.2e5, 0.44 | 4.0 / 23.6 % | 2.9e5, 0.36 |
| boot lid / tailgate | 3.0 % (12) | 5.0e5, 0.43 | 6.8 / 11.3 % | 5.0e5, 0.30 |
| fenders | 4.6 % (12) | 4.0e5, 0.37 | 2.2 / 14.7 % | 4.7e5, 0.26 |
| bumpers | 15 % (100) | 1.0e5, 0.34 | 3.2 / 25 % | 1.3e5, 0.26 |

Springback after 20 % plastic strain: BX body 5 %, doors 6 %; sedan1
body 18–25 %, doors 12–24 %. The BX holds its yield forces (body median
8 kN) on beams whose axial rigidity k·L is about 4× ours (9.5e5 N against
2.4e5 N). Its parts have fewer beam ends per node (body 11.0 against
sedan1's 15.1 with members; doors 4.1 against 11.4; bonnet 5.8 against
10.7), and longer beams, so the step's bound leaves each beam stiffer.

Measured on the crash references (sedan1 Medium ensembles,
`BX_YIELD=1`: each part's beams yield no later than the BX's median strain
for that part at their built k; `BX_YIELD_SHEETS=1` the sheets' edges
only; both measurement switches):

| | Real forces (now) | BX strain, all parts | BX strain, sheets only | Band |
|---|---|---|---|---|
| o dynamic crush | 332–388 mm | 490–521 | 345–404 | |
| o residual (springback) | 206 (≈ 45 %) | 448 (≈ 10 %) | 252 (≈ 35 %) | 297–445 |
| n max crush | 633 mm | 964 | 675 | 380–580 |
| n peak g | 32.8 | 20.1 | 31.8 | 34–47 |
| n t_m | 74.4 ms | 127 | 75.4 | 63–76 |
| n b1 | 22.9 | 14.5 | 21.8 | 22.9–39.2 |
| p peak | 22.5 kN | 8.9 | 20.0 | ≥ 28.4 |
| q | 8.59× | 3.93× | 7.37× | 4.48–5.32 |
| h | fails (narrowly) | passes | passes | |

**Decision.** Real forces stay. BeamNG's yield strains at our k·L give the
right springback (10 %) but a car far too weak in every measured test; on
the sheets alone they change little, because the members hold most of the
springback. In the BX the low strains come with 4× the k·L, from its
topology. So the springback is to be fixed the way the BX gets its strain:
fewer beam ends per node and longer beams, so that the bound leaves each
beam stiffer, with the forces kept real.

**q's mechanism.** The roof crush's load, by member (`Q_SPLIT=1`, the
change in each member's summed |force| from rest to the platen's 127 mm):
the body's shell beams that are in no member carry 2,025 kN summed with the
tension yield, against 730 kN without it. The members change little (the
sills 369 against 293, the B-pillar 214 against 162). The roof, pillar and
quarter skins work as a tension field (Wagner): in shear the compression
diagonal buckles, and the tension diagonal now carries σ_y. At about 14 %
elastic strain before it yields (k at the bound), the field engages over
the whole 127 mm of platen travel. A real panel's field does the same at
0.1 % strain, but a real roof sheet is bonded to its bows through mastic
and does not reach its yield over the rails' span in this test.

**h's regression.** The tension yield doubles the skin's elastic range,
so the crashed body keeps more ringing energy. The fifth second's peak is
11.6 J against the fourth's 10.9 J; both are about 10⁻³ of the first
second's. With the sheets' strain at the BX's it passes (0.87 J).

At z 4.0 the shell carried about 100 kN, more than the rails. It now
carries 10–50 kN.

**Heavier, fewer nodes.**
- `Spacing::pole_lanes` keeps every spacing a pole can meet (along the car,
  the sides, the ends) at most 0.24 m. The roof and floor take the
  resolution's 0.30 and 0.35 m.
- `pillar_c` has pole-width stations (A8), so the rear doors' latches meet
  the pillar's own nodes.
- Door hardware picks its pillar nodes from members' nodes only.
- Counts: 940 to 872 nodes. Body skin 350 to 282 nodes, median 1.31 to
  1.53 kg (BX 145 at 1.7 kg). Bonnet 25 skin nodes (BX 26). Front door 19
  skin nodes, median 0.88 kg (BX 11 at 1.0). Rear door 16.
- Skin beam k: body 8.7e5 N/m, doors 4.8–5.4e5 N/m (BX door skin
  1.1 MN/m on 1 kg nodes).
- **Side glass as its own part** (`GLASS_PART`) is built and gives 15-node
  doors at 1.3 kg. sedan1 does not use it (open): with it, the 52 and
  120 km/h walls' energy rose 2–38 J between tenths of a second. That held
  with the pane welded in instead of bolted, so the cause is the extra shut
  line, not the break group.

**Results (sedan1, Medium):**

| Reference | Before (71fb8075) | Now | Band |
|---|---|---|---|
| n max residual crush | 356 mm | 643 mm | 380–580 |
| n dynamic crush | 0.51 m | 0.698 m | 0.62–0.75 |
| n t_m | 57 ms | 75.4 ms | 63–76 |
| n restitution | 0.27 | 0.175 | 0.13–0.20 |
| n peak g | 60 | 31.8 | 34–47 |
| n b1 | 67 | 22.8 | 22.9–39.2 |
| n apertures, A-pillar, firewall | p90 18 mm out | all in | |
| f 120 km/h (b1, ledgers) | fails | passes | |
| h settle | fails | passes | |
| o side pole residual | 245 mm (sd 15) | 270 mm (sd 26) | 297–445 |
| p FMVSS 214 peak | 21.3 kN | 18.9 kN | ≥ 28.4 |
| q FMVSS 216a | 5.55× | 6.99× | 4.48–5.32 |
| g latch lets go (door pulled at its centre of mass) | 104 kN | 46.3 kN | 25.7–55.3 (statics × the 14 latches' 13.6–29.2 kN) |
| t T-bone leak, ensemble mean at 0 / 0.43 / 0.7 m | | 1.40 / 0.84 / 0.71 % | ≤ 1 % |
| MDB residual | | 177 mm | 126–206 |
| A8 sweep at 30 km/h | | 0 of 244 | 0 |
| torsion (BIW) | 7.4 kN·m/deg | 10.3, glazing 2.4 % | 12.3, about 12 % |

Notes on the table:
- **n crush profile.** NHTSA measures C1–C6 against each point's own
  position. The swept bumper centre starts 14 cm ahead of its ends and ends
  on the same flat front, so its extra 14 cm is real crush. That makes the
  maximum.
- **n before.** Test `n`'s strips were ±5 cm, and the one-tier bumper
  beam's nodes are 0.44 m apart, so C3/C4 measured the engine block or the
  rear bumper. Each strip is now half the points' spacing either side. The
  "before" column was measured with the old strips.
- **o scatter.** The side pole's ensemble sd is 26 mm. NHTSA's repeat tests
  scatter 24 mm (pole) and 14 mm (MDB).
- **u (A6).** The side pole at the latch line lets the front door's primary
  latch go at 27.2 kN; the secondary holds.
- `sedan` stays the default: sedan1 does not pass o, p, q, g or t yet.

### The deck lid opening and the lids (2026-11-05, later)

**Measured first** (`m`, `M_TRACE=1`, Medium, 56 km/h wall). The boot lid
did not lift because its latch let go: its hinges and latch peaked at
2–4 kN and nothing broke. The deck opening itself collapsed. Its length
(the body's hinge-line nodes to its latch-line nodes, in the world) went
721 → 446 mm at 100 ms and stayed about 480 mm. Its edges were shell nodes
in no member: the rear panel's top moved 350–500 mm forward against the
cabin, under the rear body's own inertia, at low plastic work (the skin's
buckled yield).

**The opening's frame.** A real sedan's deck opening is ringed by closed
sections. sedan1 now has them (`sedan_one_tier`, `DECK_FRAME=0` leaves them
out to measure):
- `deck_x`, the rear package tray's upper reinforcement, across the
  opening's front edge (the lid's hinge line);
- `deck_gutter`, the quarters' gutters (drip channel and quarter inner
  upper), from the C-pillar's base to the rear panel, both sides;
- `rear_panel`, the rear end panel's upper reinforcement, across its back.

Sections are typical HSLA (CR340LA) hats closed by their panels, 0.8 mm
(a Choice): 60 × 80, 50 × 60 and 60 × 100 mm. Their yields come from M_p.
The opening now shortens at most 132 mm while the car stops and about
20–30 mm after it (747 mm long).

**A rigidifier under each lid** (BeamNG's documented practice for flat
panels: "add a rigidifier node offset from the surface and connected to
every panel node"). A real lid is a bonded double shell whose depth gives it
bending stiffness. The lattice's single sheet has only its curvature's, so
with the opening fixed the boot lid's free rear corners flapped up 280 mm
at about 80 ms, while the latch at the centre held. The node sits
`RIGIDIFIER_DEPTH` (0.10 m, the inner frame's hat, a Choice) under each lid's
centroid, joined to every skin node by the lid's own sheet beam
(`RIGIDIFIER=0`: none). Damping-only beams across the panels' spans (the
other documented practice) were already there: the body dashpots at the
BX's rate.

| Boot lid, 56 km/h wall (drag on / off) | Before | Deck frame | Deck frame and rigidifiers |
|---|---|---|---|
| Edge lifts off the body | 136 / 140 mm | 284 / 293 mm | 174 / 171 mm |
| Seam opens | 371 / 396 mm | 330 / 340 mm | 238 / 247 mm |
| Deck opening, shortest | 446 mm of 721 | 615 mm of 747 | |
| Hinges and latch, peak | 2.2–4.3 kN | 2.4–5.6 kN | 1.5–2.9 kN |

The remaining lift is the lid's bending under the step's bound (k at
0.1–4 % of real, the same root as the side pole's springback). The
topology work (fewer beam ends per node, longer beams) is the fix.

**The BX's latches, for the latch itself.** The BX holds its closures with
couplers (`advancedCouplerControl`, `autoCoupling`): trunk 40 kN, bonnet
latch and catch 30 kN each, doors 35 kN, each in its own breakGroup. sedan1's
lid latches are still tension ropes sized at the door's FMVSS 206 values.
They did not let go in any wall, so the coupler would not change the lift
measured here. A `.truck` coupler needs a Rubble extension and a striker
node on the body at the latch pin; not done yet.

**Energy (open, solver).** With either addition the 120 km/h wall's energy
rises between tenths of a second (`f`): 113.6 J with both, 9.8 J with the
rigidifiers only, 1.9 J with the deck frame only, none without. With
self-collision off (`NO_SELF=1`, a measurement) it rises by 0.00 J with
both, so the contact injects it. Sent to the solver agent as a repro.

Other references with both (Medium ensembles): `n` max crush 651 mm, b1
22.2, e 0.181, peak 32.1 g, t_m 75.0 ms; `o` 225 mm (sd 91); `p` 20.4 kN;
`q` 8.62×; `g` latch 46.9 kN, door off 47.4 kN (passes); `t` 12 leaked;
`h` passes.

### Topology against the bound (2026-11-05, started)

`topology_census` (`--ignored`) gives, per part: nodes, beam ends per node
by kind, the share of nodes at the bound, and how each node's Gershgorin row
divides by kind (stiffness h²k and damping 2hc, at the saturated nodes). It
also gives the median k·L, ε_y and springback at 20 % plastic strain.
sedan1 Medium at 64a3072c:

| Part | Nodes | Ends per node (sheet, member, sheet–member, dashpot, tie) | At the bound | Row: member k, sheet k, tie k, all damping | k·L p50 | ε_y p50 | Springback at 20 % |
|---|---|---|---|---|---|---|---|
| body | 549 | 1.7, 9.4, 0.1, 1.9, 2.0 (15.2) | 87 % | 54, 17, 12, 17 % | 2.75e5 N | 4.4 % | 18 % |
| doors | 106 | 1.9, 7.2, 0, 1.0, 0.7 (10.9) | 92 % | 55, 16, 6, 22 % | 1.74e5 | 3.1 % | 13 % |
| bonnet | 31 | 4.6, 3.7, 0, 1.7, 0.3 (12.0) | 97 % | 37, 27, 3, 32 % | 7.7e4 | 4.4 % | 18 % |
| boot lid | 21 | 0, 6.4, 0, 1.1, 0.5 (9.4) | 95 % | 65, 0, 6, 27 % | 1.2e5 | 10.8 % | 35 % |
| fenders | 20 | 5.8, 0, 0, 1.5, 0 (7.3) | 80 % | 0, 79, 0, 21 % | 1.4e5 | 2.1 % | 10 % |

The BX: body 11.0 ends per node, k·L 9.5e5 N, ε_y 1.0 %; doors 4.1 ends,
ε_y 1.25 %.

The members' trusses take about half of every saturated row. A box
member's corner node has about 10 beam ends: 2 longitudinals, 4 face
diagonals, 2 bend beams over two stations, and 2–3 ring ties.

**Tried, not adopted: no bend beams** (`n`, `o`, `p`, `q`, Medium). Body
ends per node fell 15.2 → 14.3 and k·L rose only 2.75e5 → 2.81e5: the
freed row went to the other saturated beams of each node. The car lost its
members' buckling resistance: `n` max crush 0.84 m, b1 20.0, t_m 80.8 ms;
`p` 17.7 kN; `o` 228 mm (dynamic 354–473). The bend beams carry the
crush, so they stay.

So beam count alone does not raise k·L. The lever is the station spacing
(a member's axial rigidity grows with the square of its station spacing at
a fixed mass and step, `MemberSpec::station`), which the pole lanes cap at
0.24 m (A8). Open.

### Pole lanes only where needed; each chord once (2026-11-05, later)

All on the BeamNG profile (`rubble_beamng_profile`, `jbeam-semantics-spec.md`
§36), which sedan1's `.truck` now carries (`BEAMNG_PROFILE=0` leaves it out,
a measurement switch). Medium, 8-run ensembles.

**The bound sets a floor under ε_y on the lanes (exact arithmetic).** At a
saturated node every beam comes down to about k = G·m/(h²·n). A sill lane
node (1.25 kg, 13–15 beam ends) caps at 0.95–1.4 MN/m, so a 0.2 m lane beam
has k·L ≤ 3e5 N. The sill's chord yields at 56 kN (`side_sections`), so its
ε_y is 20–28 %. For 1 % at that yield, k·L must be 5.6e6 N: a 14 kg node at
10 ends, or 1 m beams. Pole lanes at 0.24 m with real chord yields cannot
reach the BX's strain by topology alone. The BX's 1 % comes from yields of
about 8 kN (its body median), not from stiffer beams.

**A member's chord was counted twice or three times (fixed).** In a lane,
the shell's edges take the member's chord material. So did:
- the shell's bend beams (two bays) along the lane;
- the truss's own longitudinals and face diagonals between station nodes
  snapped onto the lane.

Each carried the full chord yield in parallel, so a lane held 2–3× its
section's yield. Now each chord is built once:
- the shell's bends in a lane stay the sheet's bending;
- the truss leaves out its longitudinals and face diagonals between two
  lane nodes (`box_truss`'s `outer`).

Measurement switches: `LANE_BENDS=1`, `LANE_TRUSS=1`. With both,
`sedan1`'s `.truck` is byte-identical to before. `sedan`'s is unchanged at
every resolution.

**The interior at the BX's spacing (built, not adopted).** `INTERIOR=1`
(`body::Interior`): every other ring is a minor ring carrying only the band a
pole can meet (sides, wells, the top outside the roof). The floor and the roof
(z 1.36–2.58) then sit on major rings 0.48 m apart. The BX's `bx_body_coupe`
was measured for this: floor stations 0.38–0.75 m along and 0.25–0.7 m across;
roof stations 0.33–0.37 m along and about 0.55 m across. The floor keeps
0.35 m across: a ring's nodes are placed over the whole section at once, so a
coarser floor moves the sides' nodes, and with them the rear doors' latches.
Body nodes 549 → 503; k·L p50 2.75e5 → 2.81e5.

Not adopted:
- the 56 km/h wall's energy rises 24–98 J between tenths of a second
  (`b`, `n`), with self-contact off as well (`NO_SELF=1`), so the cause is
  not contact;
- a `rail_f` node that loses its floor node to snap to (0.10 kg) carries a
  6.9 kg node's body dashpot, h·Σc/m 0.79, past 0.75.

Rating that dashpot on the lighter node's mass removes the damping-bound
failure. The energy then rises at rest as well (`a`), so that change was
reverted. Open.

| Reference (band) | Before (profile) | Each chord once (now) | + interior (`INTERIOR=1`) |
|---|---|---|---|
| n max crush (380–580 mm) | 631 | 592 | **523** |
| n dynamic crush (620–750 mm) | 694 | 691 | 638 |
| n t_m (63–76 ms) | 74.1 | 74.1 | 69.6 |
| n peak g (34–47) | 33.7 | 33.2 | 33.4 |
| n b1 (22.9–39.2) | 24.0 | 25.1 | 27.3 |
| n restitution (0.13–0.20) | 0.191 | 0.192 | 0.169 |
| o residual (297–445 mm) | 225, sd 96 | **307**, sd 121 | **377**, sd 72 |
| p peak (≥ 28.4 kN) | 23.7 | 19.8 | 24.1 |
| q (4.48–5.32×) | 8.42 | 8.49 | 8.72 |
| g latch / door off (25.7–55.3 kN) | 36.7 / 51.5 | 46.0 / 46.5 | 45.6 / 46.2 |
| f 120 km/h | passes | passes | fails (energy) |
| h settle | passes | fails narrowly (fifth second's peak 4.51 J against the fourth's 4.45) | fails |
| t leaked (≤ about 10) | 18 | 19 | 18 |
| crash invariants | 7 of 16 pass | the same 7 | |
| invariants_small | 19 of 19 | 19 of 19 | |

- **o is bimodal.** It depends on whether the front door's primary latch
  lets go: runs where it holds give about 170–210 mm, runs where it goes give
  370–440 mm. The mean moves with how many runs lose the latch.
- **f.** The profile's two-body self-contact fixes the 120 km/h energy rise
  in every build here.

### BeamNG's own car on the same scenes (2026-11-05, later)

The Ibishu BX coupe (BeamNG-authored, `bx_body_coupe`, every other slot its
default) runs sedan1's scenes on Rubble's solver in JBeam mode (pressure
wheels). It is opt-in and skips cleanly without the content:
- `RUBBLE_JBEAM_BX`: the `vehicles` directory holding `bx/`;
- `RUBBLE_BEAMNG_MODS`: the community content holding the donor tyre.

Nothing of BeamNG's is committed. `reference_cars.rs` only loads from the
paths a test passes it. The tests are:
- `showcase_car`: `beamng_n`, `beamng_o`, `beamng_p`, `beamng_q`,
  `beamng_pole_sweep`, `beamng_ref_census`;
- `crash_invariants`: `beamng_reference_invariants_report`;
- `explicit_scenes`: `mdb_into_beamng_reference_car` (GPU).

The BX's node sets for the scenes (sills, B-pillars, header, door apertures)
are picked by its JBeam node names. Each pick is a Choice.

**Choices for the BX runs.**
- **Tyres.** BeamNG's shared `common` content is not local, so the BX's
  pressure wheels are hubs only. They take a community car's 205/65 R15
  tyre (radius 0.325 m, against the BX's axle heights of 0.325–0.35 m).
- **Ledgers.** Its ledgers are not judged. Their accounting does not cover
  pressure tyres or bounded beams.
- **The rigs pin with the solver's `NODE_IMMOVABLE`.** Before, they reset the
  held nodes' positions after every step. That is not a constraint: the node
  moves a step under its load and is yanked back. On the BX's stiffer beams it
  drove the neighbours unstable within 10 steps. sedan1 under the true pin:
  `p` 19.8 → 23.8 kN, `q` 8.49 → 9.01×.

The other local stock car, `template_car_sedan`, is BeamNG's modelling
template, not a crash model: 59 body nodes carrying 1,090 kg, no closures,
and no tyres (`common`). It is not used.

| Reference (band) | BX coupe | sedan1 (lanes, before the cuts) |
|---|---|---|
| n max crush (380–580 mm) | 279 | 592 |
| n dynamic crush (620–750 mm) | 461 | 691 |
| n t_m (63–76 ms) | 48.5 | 74.1 |
| n peak (34–47 g) | 49.8 | 33.2 |
| n b1 (22.9–39.2) | 86 | 25.1 |
| n restitution (0.13–0.20) | 0.228 | 0.192 |
| n apertures, A-pillar, firewall | all in (6–9 mm) | all in |
| o residual (297–445 mm) | 351, sd 4 | 307, sd 121 (bimodal) |
| o dynamic → springback | 367 → 4 % | 332–388 → about 45 % |
| p initial / intermediate / peak (≥ 10 / 15.6 / its 2 × curb) | 9.0 / 14.0 / 28.7 kN (≥ 23.7) | 11.6 / 13.9 / 23.8 kN (≥ 28.4) |
| q (4.48–5.32×) | 5.59 | 9.01 |
| MDB residual (126–206 mm) | 202 | 137 |
| Rear third, wall / head-on (past 0.2 %, broken) | 0, 0 / 0, 0 | 82, 7 / 18, 0 |
| Lid against rear body, wall / head-on peak (≤ 5 mm) | 13.9 / 11.3 mm | 422 / 217 mm |
| Rear body RMS, wall (≤ 5 mm) | 2.6 mm | 42 mm |
| Wall ripple from 0.5 s (≤ 1 mm) | panels < 1; fascia 6.6, fuel tank 3.4, sideskirt 2.1 | body 4.2, bonnet 7.0, lid 10.6, doors 1.9–4.5, fascia 13.2 |
| Pole far side (past 0.2 %, broken; far doors' hardware) | 20, 0; holds | 99, 6; holds |
| A8 sweep at 30 km/h (passed through) | **21 of 244** | 0 of 244 |

**What BeamNG's content shows on this solver.**
- **Side, roof and MDB.** BeamNG's own yields (about 1 % strain, 8 kN a
  body beam, at k·L about 1e6 N) land `o`, `p`, `q` and the MDB in or near
  the measured bands. They also give 4 % springback in the pole, and its
  rear body stays elastic.
- **Frontal.** The BX is over-stiff (a 1990 small coupe with a short crush
  zone) and rebounds more than sedan1 (e 0.23).
- **Topology.** Its quality comes from fewer, heavier nodes and long beams
  (145 body nodes, 0.56 m median beam).
- **A8.** That spacing fails A8 on node-only static contact: a 25 cm pole
  passes between its side nodes in 12 of 89 offsets. sedan1 keeps the
  0.24 m pole lanes, so the BX's k·L is out of reach by topology.

**Decision: real forces, each section counted once, spread by axial
rigidity.**
- **Ruled out:** BeamNG's strains at sedan1's k·L. Measured above, they make
  the car too weak everywhere: n 964 mm, p 8.9 kN, q 3.93×.
- **Why:** BeamNG's low strain is its yield over its k·L, not an independent
  choice. The forces are what the scenes measure.
- **Measured instead:** what a member's cut carries. Every non-crush
  member's bay cut (the beams crossing the plane between two stations) held
  1.4–6× its section's axial yield, typically 2–3×:
  - the longitudinals at the full chord yield;
  - two bend beams at half each;
  - the face diagonals;
  - on the body, the lane's shell edges and webs.

  For example: sill 3.1×, B-pillar 2.8×, roof rail 2.5×, header 3.7×,
  firewall 6×. This is the same double count as the lanes' (fixed above), in
  the trusses themselves.
- **Built now** (`one_tier_section_cuts`; `SECTION_CUTS=0` measures the old
  build):
  - Every cut's Σ yield·|cos| is its section's: its chords (4 a box, 2 a
    ladder) × its chord yield.
  - Within the cut, each beam's yield is proportional to its built
    k·L·|cos|. So the cut's beams yield together, as a section's fibres do,
    which gives the least elastic strain at yield the topology allows:
    1–9 % at the cut, against 20–28 % on a lane chord before.
  - Crush members keep their mean crush force and are only redistributed.
  - A member's beams yield alike both ways (no sheet tension yield on a
    lane edge that is a chord).
- **Option not taken:** "a section's strength split across parallel
  beams". Splitting does not lower ε_y unless the beams are longer: the bound
  caps Σk at a node, not Σk·L. Spreading by k·L is that idea done exactly.

**Effect (Medium, ensembles, everything else as built):**

| Reference (band) | Before the cuts | Cuts | Cuts + `INTERIOR=1` |
|---|---|---|---|
| n max crush (380–580 mm) | 592 | 644 | 604 |
| n dynamic (620–750 mm) | 691 | 730 | 677 |
| n t_m (63–76 ms) | 74.1 | 80.3 | **75.1** |
| n peak (34–47 g) | 33.2 | 30.0 | 33.9 |
| n b1 (22.9–39.2) | 25.1 | **22.9** | **23.6** |
| n restitution (0.13–0.20) | 0.192 | **0.170** | **0.143** |
| n A-pillar base rearward (−7…17 mm) | in | 24, 25 | 22, 32 |
| n ledger | passes | rises 3.8 J between tenths | passes |
| o residual (297–445 mm) | 307 (bimodal) | 221, sd 10 | 258, sd 35 |
| o dynamic | 332–388 | 406–415 | 426–467 |
| p initial / intermediate / peak | 11.6 / 13.9 / 23.8 kN | 11.7 / 12.6 / 19.7 | 11.7 / 13.1 / 21.3 |
| q (4.48–5.32×) | 9.01 | 7.63 | 7.26 |
| g latch / door off (52.2–83.4 kN off) | 46.0 / 46.5 | 52.2 / 98.9 | 47.3 / 102.6 |
| h | fails narrowly | fails narrowly (0.287 J against 0.277) | passes |
| f 120 km/h | passes | passes | fails (below) |
| t leaked | 19 | 19 | |
| MDB (126–206 mm) | 137 | 180 | |
| A8 sweep at 30 km/h | 0 of 244 | 0 of 244 | |
| crash invariants | 8 of 17 | 9 of 18 | |
| invariants_small | 19 of 19 | 19 of 19 | |

- **Gained.** The cuts take the roof from 9.0× to 7.6× and the frontal's
  rebound into its band. The pole's front latch now holds every run, so `o`
  is no longer bimodal.
- **Lost.** The members are a third to half as strong axially as before, so
  the cell lets the A-pillar base back 24 mm and the door crush falls.
- **What this says about the sections.** With each counted once, the real
  sections in `side_sections` do not carry the side and frontal loads alone.
  The over-count had been standing in for structure the car does not have
  yet: door inner panels, the B-pillar's reinforcement, the roof bows'
  joints. That is the next authoring work.
- **Springback is not fixed by the cuts.** At the pole's deepest frame the
  car holds 15.9 kJ elastically (`O_ELASTIC=1`), in this order:
  - body skin 4.2 kJ;
  - sill 3.4 kJ;
  - suspension 1.6 kJ;
  - floor_x1 1.0 kJ.

  The skin's tension yield (σ_y·t, Rubble's extension) at the bound's k is
  the largest store. That is the next lever.

**`INTERIOR=1`: its two blockers fixed, a third found.**
- **The energy gain was content.** The interior's minor ring left its band's
  edge node lying on the edge of the floor's (and the top's) strip from major
  to major: a T-junction closed by a sliver triangle.
  - `crash_invariants::energy_terms_probe` (the world's ledger per term per
    0.1 s) shows self-contact doing +1.0–1.6 kJ of positive work every tenth
    while the car stands, balanced by the beams' dissipation. Two right-sill
    nodes went on vibrating at 7–10 m/s.
  - The strip's triangle is now split at that node, so the skin is
    conforming. Self-contact's work fell to the baseline's 0–16 J a tenth,
    and the wall's mechanical energy only falls.
  - The solver's anti-clip spawn exclusions now also treat a node on a
    triangle's edge as inside.
- **The damping bound.** Body dashpots are rated at the BX's 18.2 /s on the
  lighter of their two nodes' masses, not on the node each is from. Neither
  end is damped harder per kilogram than the BX. Worst h·Σc/m 0.787 → 0.602
  (`C_RATIO` 0.75). `a` at rest is clean (0.00 J between tenths).
- **Still open: `f` at 120 km/h with drag off.**
  - Two `bonnet_inner_3` ladder nodes (0.40 kg, row at the bound, one 25 mm
    tie to the skin) oscillate at the step's Nyquist rate. Their beams yield
    every step in alternating directions and harden, and 22 steps each add
    0.5–2.1 kJ.
  - It appears only with the cuts and the interior together.
  - The solver agent's reading: content at the stability edge, since
    `G_LIMIT` is 1.2 × 4/h² by design and the yield rule pumps the period-2
    mode there. A node whose beams yield wants its row well under the edge.
    The solver will add a load-time per-node stability check.
- **Not adopted yet** because of `f`. The fix is that ladder's mass or its
  tie, to be done with the lids' double skin.

**Lids: measured modal data (`lids_first_mode_in_the_measured_range`, which replaced the ≥ 48 Hz invariant).**
- **Published hoods measured on the car:**
  - first torsion 17.7 Hz and 22.9 Hz (MDPI *Machines* 12(12) 862, 2024,
    experimental modal analysis of a front hood);
  - 12.3 Hz raised to 31.0 Hz by moving the lock (an NVH DOE study);
  - lightweight hood designs' first mode 41.5 Hz at the least (stamping-aware
    modal study).
- **Design targets quoted:** 25–30 Hz and "above 40 Hz".
- **No production hood reaches 48 Hz.** So the invariant's ≥ 48 Hz
  ("millimetres at 47 g") is not physically achievable as written: a real
  hood at 25 Hz deflects about 19 mm quasi-statically at 47 g.
- **The measured target for the lids is their first mode at 18–41 Hz.**
  sedan1's are 8.7 Hz (bonnet) and 6.7 Hz (boot lid). The doors (23.6 and
  26.3 Hz) are in that range.

### Section cuts fixed, side stampings, lids, INTERIOR (2026-11-05, later still)

**The section cuts' allocation (fixed).**
- **The bug.** The first build rescaled every cut's beams iteratively. A
  beam two cuts share (a bend over two bays) was driven toward zero where one
  cut was 6× over. Firewall beams ended with yields and break forces of
  0–0.2 kN, and they tore in `g`'s door pull. That is what held the door to
  98.9 kN.
- **Now:**
  - Only a cut's axial carriers share its section's yield: beams within 60°
    of the axis (`SECTION_AXIAL_COS`). The rings and webs across it keep
    their own yields, and those yields' small axial part counts against the
    section.
  - Each beam takes the least of its cuts' λ = F / Σ k·L·cos², then rises
    while all its cuts are under their section.
  - Every member's cut is now at 1.00 of its section.

**The side structure as its stampings** (`layered`, `SIDE_LAYERS=0`
measures without).
- **Members.** The rocker, the B-pillar and the A/hinge pillar are each the
  sum of their layers:
  - body side outer: a BH210 0.7 mm hat;
  - reinforcement: hot-stamped 22MnB5 1.8 mm (B-pillar), 1.6 mm (A-pillar);
    DP980 1.6 mm (rocker);
  - inner: HSLA 1.2 mm (B-pillar), DP780 1.4 mm (A-pillar, rocker).
- **Sources.** These are Choices in WorldAutoSteel FSV's and A2Mac1-style
  teardowns' published ranges. The Accord teardown's material classes (DOT HS
  811 666 fig. 113) include each.
- **How a layered section is computed.** Its plastic moment is the layers'
  sum: a hat's Z_p is t·(w·d/2 + d²/2), a flange's t·w·d/2. Its steel per
  metre is theirs, which spreads the measured BIW mass. No mass is moved.
- **Not built:**
  - **The roof bows and gussets.** q is already 1.5× over its band, and the
    over-strength is the skin's tension field.
  - **The door inner panels.** They are built as `DOOR_INNER=1` and
    measured: at the bound their beams take their rows' share from the
    door's members, and p fell 24.5 → 17.9 kN.

**The lids** (`lids_first_mode_in_the_measured_range`, measured hood range
17.7–41.5 Hz).

| Lid construction | Bonnet first mode | Boot lid first mode |
|---|---|---|
| Before: rigidifier 0.10 m, lid ladders, the lid's built sheet on the spokes | 8.7 Hz | 6.7 Hz |
| Double skin, 60 mm inner hemmed at the perimeter (`DOUBLE_SKIN=1`, `LID_HEM=1`) | 7.1 Hz | 3.8 Hz |
| Double skin, inner under every node | 8.2 Hz | 6.0 Hz |
| Rigidifier spokes at the inner stamping's real stiffness and yield | 12.4 Hz | 11.2 Hz |
| The same, at the BX's rigidifier depths, without the lid ladders (**built**) | **19.2 Hz** | **12.6–14.6 Hz** |
| BX (`CI_REF=1`, same measurement) | 33.7 Hz | 3.1 Hz (67 extension beams left out of its cut-out) |

- **Why the double skin fails.** At real masses under the 2 kHz bound, a
  sandwich's faces carry about 1/1500 of steel's axial rigidity, so a 60 mm
  depth cannot lift the mode.
- **What lands the bonnet is BeamNG's practice**, a deep rigidifier node:
  - the BX's are measured at 0.228 m under the hood (`h6`) and 0.160 m under
    the trunk (`t5`);
  - its spokes are now the inner stamping's sheet, (√3/4)·E·t/2 and half a
    sheet edge's yields;
  - before, the spokes yielded at 0.3 kN and sat far under the bound.
- **The lid ladders are gone** (`LID_INNER=1` builds them). They took 37 % of
  the bonnet's row from its sheet, and one of them pumped energy at the
  step's Nyquist rate (below).
- **Still open:** the boot lid is below the range.

**INTERIOR: fixed, measured, not adopted.**
- **Fixed:**
  - the T-junctions, which made the energy gain;
  - the dashpots' rating, which broke the damping bound;
  - the 120 km/h Nyquist pump on `bonnet_inner_3`, gone with the lid
    ladders.
- **Measured with it:** n max crush 564 mm, o 336 mm, A-pillar 15/21 mm,
  h passes.
- **Why it is not adopted:** its 0.48 m floor lets a side pole through once
  the sill is crushed (A8 at 30 km/h: 5 of 178 sideways passed through), and
  the MDB's residual crush rose 208 → 309 mm. A8 is a requirement.
  `INTERIOR=1` builds it.

**Now (Medium; BX on the same scenes since the anti-clip merge):**

| Reference (band) | sedan1 now | BX |
|---|---|---|
| n max crush (380–580 mm) | 640 | 287–322 (structure) |
| n dynamic (620–750 mm) | 711 | 462 |
| n t_m (63–76 ms) | 76.6 | 48.5 |
| n peak (34–47 g) | 31.0 | 49.3 |
| n b1 (22.9–39.2) | 22.4 | (exterior profile unreliable) |
| n restitution (0.13–0.20) | 0.145 | 0.228 |
| n A-pillar base (−7…17 mm) | 19, 22 | 6, 7 |
| o residual (297–445 mm) | 293, sd 7 | 350–354; one run of 8 non-finite since the anti-clip merge (sent to the solver agent) |
| p initial / intermediate / peak | 9.9 / 14.6 / 23.0 kN (≥ 28.4) | 9.0 / 13.9 / 32.3 kN (≥ 23.7) |
| q (4.48–5.32×) | 8.37 | 5.59 |
| g latch / door off (25.7–55.3 / 52.2–83.4 kN) | 41.3 / 102.3 | |
| MDB (126–206 mm) | 208 | 201 |
| t leaked | 5 | |
| A8 at 30 km/h | 0 of 244 | 21 of 244 |
| f, h, a, b | pass | |
| Rear third, wall / head-on (past 0.2 %, broken) | 88, 6 / 19, 0 | 0, 0 / 0, 0 |
| Lid against rear body, wall / head-on (≤ 37 mm) | 399 / 124 mm | 12 / 7 mm |
| Pole far side | 90 past, 8 broken; `door_r_r.latch_secondary` | 22 past, 0 broken; holds |
| Bonnet / boot lid first mode (17.7–41.5 Hz) | 19.2 / 14.6 Hz | 33.7 / 3.1 Hz |
| crash invariants | 9 of 18 | |
| invariants_small | 19 of 19 | |

- **g.** All of the pull is in the hardware's ropes. After the latch lets go
  (41 kN), the lower hinge takes 67 of its ropes' 87 kN when the door comes
  off at 102 kN, past the statics band. This is hinge sizing against the
  stiffer door ring, not a path round the hardware.
- **Next:**
  - the hem rows (requested by the visual agent);
  - the frontal's crush (640 mm, peak 31 g) and A-pillar;
  - p's door;
  - the boot lid's mode;
  - the skin's tension field (q).

### q's load path (2026-11-05, latest)

`Q_CUT=<y>` gives the platen's load across a horizontal plane at the end of
the 127 mm push: the vertical force of every beam crossing the plane, summed
by member, with the body's own beams binned by side and position
(`Q_CUT_WHO=1` lists them). sedan1 at the belt plane (y 1.05) carries
110 kN:

| Member | kN |
|---|---|
| B-pillars | 26 + 12 |
| A-pillars | 19 + 0 |
| C-pillars | about 0 |
| The body's beams in no member | about 50 |

- **The body's beams.** The largest are long beams, 1.1–1.6 m, from the
  B-pillar's top joint (node 536) down to the sill at floor_x1 and floor_x3,
  and to the A- and C-pillar bases. They carry up to 12 kN each.
  - Some of these are the cabin's lumped shear panels (`cabin_x`, the door
    rings' X, `hinge_x`).
  - The 536 → sill beams at ±0.5 m are not braces: they stay with
    `BRACES=0`. Their origin is not yet traced.
- **Measured without the lumped panels** (`BRACES=0`; `RING_BRACES=0` drops
  the door rings only):
  - q falls from 8.37× to 6.45×, and p is unchanged (23.5 kN);
  - but the body-in-white's torsion falls from 10.3 to 4.0 kN·m/deg,
    against the measured 12.3.

  The panels stand in for shear stiffness that the bound-capped skin does not
  have, so they stay for now.
- **Next.** Trace the 536 → sill beams. Then give the lumped panels the
  section-cut treatment: their yield as the aperture's racking mechanism
  load, not a strut's. A real roof's strength is the pillars' bending.

### q: the openings carry nothing (2026-11-05, built)

**Traced: the 536 → sill beams are a builder artefact.** They are the
`sill`–`roof_rail` zigzag shear panel (`ShearSpec`, `KL_SHEAR`, from the
two-tier cabin box). Each roof-rail bay was crossed by diagonals to the
nearest sill stations, through both door openings. In q they carried
11–12 kN each at the belt. `BRACES=0` left them because they are `shear`,
not `braces`.
- **Decision:** sedan1 does not build that panel. A door opening has no web,
  and the door rings' racking X already stands in for the aperture's shear
  (`SILL_ROOF_PANEL=1` builds it again, as a measurement).

**The door rings' X as tension-only diagonals** (`RING_ROPES=0` measures
without).
- **Why.** An X of two acting diagonals couples a racking frame's shear with
  vertical load. Under a roof load both diagonals shorten, and the opening
  carries 2·F·sin α that a real opening does not have.
- **Built.** Each diagonal is now a rope (`BraceSpec::tension_only`). In
  racking, one diagonal pulls and alone carries the frame's racking stiffness
  (k·L_i / cos²α_i) and its mechanism shear, F = V / cos α_i with
  V = 4·M_p/h (`door_ring`, `aperture_racking`). A load that shortens both
  diagonals finds neither.
- **This is the racking-mechanism yield the plan asked for.** The rings
  already had it; the spurious vertical path was the second diagonal.
- The remaining lumped panels across the belt are real panels: the
  screen's frame, and the rear bulkhead behind the seat.

**Effect (Medium):**

| | Before | No panel | No panel + ring ropes (**built**) |
|---|---|---|---|
| q (4.48–5.32×; BX 5.59) | 8.37 | 7.39 | **5.81** |
| q at the belt: pillars A / B / no member | 19 / 38 / 50 kN | 27 / 45 / ~25 | 39 / 45 / ~10 kN |
| BIW torsion (measured 12.3 kN·m/deg) | 10.5 | 10.4 | 9.8 |
| p peak (≥ 28.4 kN) | 23.0 | 23.9 | 24.3 |
| o residual (297–445 mm) | 293, sd 7 | | **301, sd 9** |
| g: latch / door off (52.2–83.4 kN) | 41.3 / 101.5 | | 40.4 / **76.4** |
| n max crush, peak, A-pillar | 686–639 mm, 31 g, 17–24 mm | | 632–644 mm, 31–33 g, 17–25 mm |
| t leaked (≤ 9 of 983) | 7 | | 10 |
| crash invariants | 9 of 18 | | 10 of 18 |

- **Where the roof's load goes now.** The platen's load goes down the
  pillars: 84 of 78 kN at the belt is the A- and B-pillars, with the
  far-side ring rope in tension against them. The roof's strength is the
  pillars' bending, as in a real car.
- **Torsion** loses 7 % (10.5 → 9.8). That is the second diagonal's
  coupling, which a real ring does not have.
- **t's leak** (10 against 9) is contact noise. The solver agent measured
  sedan1's T-bone at a mean of 12 over five offsets, with anti-clip on or
  off, so it is not structure.
- **The ledger tally** in `showcase_car` now reads `forces_now()`. Since
  `fixed_contact`, a step's contact forces are summed apart from
  `n.force`, and every self-contact had shown up as a momentum residual.

### The frontal: the floor pan only where it is (2026-11-05, built)

`N_CUT=<z>` (spawn z, comma-separated) gives the push across a plane every
10 ms by member. `N_CUT_WHO=1` lists the beams over 4 kN at 60 ms.

**What it found.** At 60 ms, 160 of the 278 kN crossing the plane just
ahead of the firewall (z 3.45) went through body beams that belong to no
member. The largest are 0.8–1.2 m diagonals carrying 8–21 kN each. They run
from the front rails' forward stations (z 4.1–4.5, at the crush cans) to the
sill's front end, straight through the wheelhouse.
- **Their origin.** They are the `rail_f`–`sill` floor-pan shear panel.
- **The cause.** A rail station past the sill's end mapped onto the sill's
  end station, so 4 of the panel's 6 bays bridged the crush zone to the
  cabin. `rail_r`–`sill` did the same behind the sill, in 2 of 4 bays.

**Decision (one tier).** A floor panel (grade `floor`) is built only over
the bays where both of its members run alongside.
- The bounds are each bay's midpoint against the other member's extent,
  with 5 cm of reach.
- `SHEAR_OVERHANG=1` builds the old bays again; `=all` trims every panel.
  Both are measurement switches.
- The wheelhouse webs (`rail_f`–`apron`, `rail_r`–`quarter_inner`) stay,
  because the inner fender's sheet is there.
- `SHEAR_TRACE=1` lists each panel's trimmed bays.

**Effect (Medium, before the solver's JBeam plasticity):**

| | Before | Trimmed (**built**) |
|---|---|---|
| n max crush (380–580 mm) | 640 | 751 |
| n dynamic (620–750 mm) | 712 | 800 |
| n t_m (63–76 ms) | 76.9 | 90.8 |
| n peak (34–47 g) | 31.6 | 27.0 |
| n A-pillar base L / R (−7…17 / −11…15 mm) | 20 / 23 | **−3 / −4** |
| n firewall (1…67 mm) | 20 | 32 |
| o | 301 | 298 |
| q | 5.81 | 5.92 |
| p peak | 24.3 | 24.0 |
| g latch / door off | 40.4 / 76.4 | 44.3 / 45.1 kN |
| Rear third, wall (plastic / broken) | 89 / 5 | 69 / 3 |
| Lid against rear body, wall | 407 mm | 383 mm |

- **The front is now honestly short of strength.** The bridge had stood in
  for the front's real late load paths.
- **The powertrain stack-up.** In sedan1 the powertrain block spans z
  3.75–4.21 (0.46 m deep). Its rear face is 0.37 m ahead of the firewall
  (z 3.38), so the engine never stacks up against the dash within the
  crush. `~powertrain/body` carries 2–4 kN across the cut.
- **What is needed:** a real engine and transaxle package depth and
  dash gap, from a measured source (open).

### Rupture at the elongation at break; air drag at the measured Cd·A (2026-11-05)

**Why.** The BeamNG profile now runs JBeam's NORMAL rule (solver 9d744a44),
so a stretched beam no longer loses strength. That strength drop had been
the only way sedan1's steel tore. `Skeleton::rupture` writes each graded
beam's tear into `rubble_jbeam_plastic`. It is **built, not adopted**:
`RUPTURE=1` builds it, and `RUPTURE_REPORT=1` in `which_nodes` gives the
counts by part.

**Graded metal** (`Material::ductile`: A80 and Rm/Re at the standards'
minima; each grade's steel is a Choice):

| Grade | Steel | Re / Rm (MPa) | A80 |
|---|---|---|---|
| panel, floor, frame | HC220B (EN 10268) | 220 / 320 | 32 % |
| crush | HCT600X / DP600 (EN 10338) | 330 / 600 | 20 % |
| hsla | HC340LA (EN 10268) | 340 / 410 | 21 % |
| uhss | HCT780X / DP780 (EN 10338) | 440 / 780 | 14 % |
| boron | 22MnB5 press-hardened | 950 / 1,300 | 5 % |
| alu | EN AW-6082-T6 (EN 755-2) | 250 / 290 | 8 % (A50) |

- **As written:** deformLimitStress = (Rm/Re)·T and strength =
  `RUPTURE_MARGIN` (1.05) × that cap. The plastic stretch is capped at
  dₑ = A − S/(k·L), so the strain when the beam tears is A.
- **Why it is not adopted: one strength for both directions.** A crushed
  beam hardens Tc·L₀/L up to the cap, and a step's overshoot past S then
  tears it in compression. Whole car, beams broken in the 56 km/h wall:

  | Rupture | Broken |
  |---|---|
  | none | 4 |
  | margin 1.05 | 136 |
  | margin 1.5 | 86 |
  | margin 3 | 19 |

  The count follows the margin, not the strain.
- **Asked of the solver agent:** a tear at the expansion limit (a sixth
  `rubble_jbeam_plastic` field). With it, dₑ = A − T/(k·L) tears the beam in
  tension only, and the strength and cap stay as built.
- **Not graded** (hardware, mounts, bolts): capping their stretch at RoR's
  tear stretch tore 38 rear-third beams the same way, so they keep their
  strengths as built.
- **The door hardware's `Rope` beams no longer separate** in `r` (and so in
  `g` and `detached_door_takes_its_edge`) since 9d744a44. The trace shows tt
  hardened while the strength stayed at its built value. This is with the
  solver agent.

**Air drag.** The profile's air drag is now its triangles' aerodynamics
(solver 36a6c919), at BeamNG's default dragCoef of 100. That is 3.4 kN at
100 km/h, 11 times a real sedan.
- **Built:** `AERO_DRAG_COEF`, with every aero triangle at one coefficient
  so the car coasts at the 2011 Accord sedan's Cd·A: Cd 0.31 on 2.31 m², or
  0.716 m² (fichacompleta.com.br's 2011 Accord EX V6 sheet; Motor Matchup
  lists Cd 0.32 for the 2.4 EX).
- **Measured:** 337 N at 100 km/h, against ½ρ·Cd·A·v² of 338 N
  (`crash_invariants::profile_drag_is_the_triangles`, now asserted within
  3 %).
- **liftCoef** equals dragCoef (BeamNG's convention; no measured lift data).
- **Next:** the solver's `rubble_aero_tri`. The covered and inner
  triangles get 0 and the exposed ones carry it all.
  `Skeleton::exposed_triangles` is ready: a ray along a triangle's outward
  normal meets no other collision triangle within 3 m. `AERO_REPORT=1` in
  `which_nodes` lists them.
- **The lid's motion is not the drag.** The wall's lid against the rear
  body is 416 mm with drag and without.
- **Open with the solver:** with the triangles' aero on, the `g` door pull
  goes non-finite (1,690 body dashpots break before any hardware). With
  `AERO_COEF=0` it does not.
- **The ledger tally** now counts the aero triangles' force as its drag
  term.

### The powertrain, the door pull and the ledgers (2026-11-05, late)

- **Powertrain at its real mass and place** (05c14e91): the 2011 Accord's
  2.4 L engine 169.9 kg and automatic transaxle 96.7 kg (DOT HS 811 666
  §5.12.1.1, §5.12.2.1) as one rigid block. Its place is read off the
  report's side view (fig. 87, about 7 mm a pixel: a Choice within about
  ±20 mm): front face 0.39 m ahead of the front axle, rear face 0.12 m
  behind. Its mounts go only where the 2008–12 Accord's are (`mount_to`:
  the rails and the subframe). Probes: `PT_LOAD`, `NODE_TRACE`,
  `LEDGER_TOP`, `LEDGER_WARN`, `BEAR_REPORT=<part>`.
- **`g` in the car's frame** (451d816a): the door is pulled with −F·mᵢ/M on
  every node (d'Alembert), so the car does not fly off at 100–200 m/s
  (`G_FREE=1` pulls the free car). With the solver's `JbeamRope` and
  `breakAtExpansion` (0435b1dd) the latch lets go at 40–43 kN (band
  25.7–55.3) and the door comes off at 84.0–84.6 kN, against the build's
  own bound of 83.4 kN (the summed strengths). **Kept as is:** the bound is
  the static rig's sum; in the car, the hinges are loaded at the angle the
  door has swung to once the latch is gone, about 1.46× their rig share, so
  the door holds a little past the sum.
- **Ledgers** (e513966a, 847339ab): contact is recomputed with the
  profile's exact friction; a compressed `JbeamRope` stores nothing;
  `FIXED_CONTACT=0` measures the f32 contact sums. Since the solver's
  dissipative contact (0eb963d2, spec §37) all 32 of `n`'s ledgers close.

### The dash where the Accord's is (2026-11-05, latest)

**Why.** sedan1's dash and cowl sat 0.40–0.49 m behind the front axle,
against 0.27–0.30 m on the Accord (fig. 87), so the engine reached the dash
about 0.13 m late, and the frontal was soft (max crush 730 mm, peak 30 g).

**Built** (`accord_dash`, `COWL=0` keeps the old package):
- **Dash lower cross member** (`firewall`) at 0.30 m behind the axle
  between the rails, its ends swept back round the wheelhouses to the
  hinge pillars' bases (z 3.36, kept). The tunnel runs forward to it.
  Section a 60 × 100 × 1.4 mm HSLA hat (a Choice).
- **Dash panel** (`dash_panel`, `dash_panel_s`): three vertical strips
  from the lower cross member to the cowl, 0.39 m wide, their chords 0.195 m
  apart. Each is its share of a 0.8 mm sheet (a Choice in 0.7–0.9 mm),
  a membrane: each chord yields at σ_y·A of its half strip (210 MPa).
- **Cowl** (the plenum on the dash's top): furthest forward at the centre,
  0.27 m behind the axle, its ends swept back in plan to the A-pillars.
  Section the plenum's 150 × 100 × 1.0 mm DP600 box (as `screen_ring`).
- **Local masses**: the master cylinder 3.16 kg (fig. 209) and the pedals
  and linkages 6.43 kg (§5.13.7) as a named mass on the driver's half of
  the dash, out of the equipment the kerb mass leaves. No mass moved.
- **Not moved (deviation from the plan, decided):** the A-pillars, the
  windscreen's base (z 3.33) and the greenhouse's top line. Moving them
  with the cowl (built and measured, 1e07cdb1) changed the shell's top
  ahead of the doors, which moved the front door's hinge pins (upper
  0.88 → 0.75 m, lower 0.49 → 0.25 m). The lower hinge then bore on the
  sill: FMVSS 214's ram (`p`) read 140 kN instead of 24, and `g`'s door came
  off at 40 kN. The plenum now runs forward under the bonnet's rear edge
  from the screen's base, as a real cowl does.

**The engine meets the dash as contact** (`BlockSpec::collide`;
`PT_COLLIDE=0` measures the old bearing beams on every face). The bearing
beams joined fixed node pairs: in the 56 km/h wall the dash's nodes went
7 cm into the engine while their beams lengthened (`NODE_TRACE`). Now
the block's outer faces are collision triangles (no air on them:
`rubble_aero_tri` lists only the skin), and every body node within 0.5 m
behind or beside it is a contacter, except those its mounts hold. Ahead,
its front face keeps its bearing beams to the radiator support: as
contact, a light node there was pinched against the wall and two
radiator-support nodes left at 107–138 m/s in the 50 km/h wall.

**The cradle drops its rear** (`CRADLE_BOLT=0` welds it as before). The
Accord's engine cradle rear mount lets go by design between 40 and 50 ms
(fig. 55, report text at §3 and §8). The cradle is now a frame: its rear
cross member, its cross member on the axle line and side members, welded
to each other only (`MemberSpec::bolted`, `Bolt::weld`). Its front cross
member stays welded to the rails (the front mounts). Each rear end is bolted
to its rail by one detacher group. The report gives no strength, so
`CRADLE_REAR_MOUNT` = 24 kN a mount is a Choice: the value at which the
first mount lets go between 45 and 50 ms (swept 12–64 kN; `BOLT_LOAD=subframe_f0`
prints each mount's load; `CRADLE_REAR_MOUNT=<N>` sets it). Before the frame,
each cross member was welded to the rails, and the rear one carried only
2–4 kN at 40–50 ms.

**Effect, `n` (Medium, 8-run ensemble, dissipative contact merged):**

| | Before | Built |
|---|---|---|
| max crush (380–580 mm) | 730 | 688 |
| b1 (22.9–39.2) | 19.3 | 20.6 |
| dynamic crush (620–750 mm) | 794 | **748** |
| t_m (63–76 ms) | 87 | 77.9 |
| restitution (0.13–0.20) | 0.110 | 0.125 |
| peak (34–47 g) | 30.3 | **40.0** |
| A-pillar bases (mm) | −3 / −2 | −1 / −5 |
| firewall (mm) | 35 / 21 | 40 / 34 |
| ledgers failing, of 32 | 1 | 0 |

With the A-pillars moved too (1e07cdb1, before dissipative contact) it was
633 mm, 22.4, 728, 74.4 ms, 0.141, 46.7 g.

**Other scenes** (before the dissipative-contact merge): `o` 300 mm (sd 13,
297–445: in band); `p` peak 23.9 kN (≥ 28.4: fails, as before); `q` 5.58×
(4.48–5.32: fails, was 5.77); `g` latch 43.1 kN, door off 84.0 kN; `r`
passes. `t` (80 km/h T-bone) leaks 42 of 1,002 striker nodes; bisected to
the dissipative contact commit (0eb963d2: 4 before, 49 at it), with the
solver agent. `detached_door_takes_its_edge` fails (19–54 drawn triangles
joining the detached door; the pull's outcome changes with any change to
the car, door off at 3.6–7.4 s); with the visual agent.

**Open: the rear third and far panels.** In the 56 km/h wall 71–89 rear
beams pass 0.2 % plastic strain (worst 28–34 %; body sheet in compression,
boot lid and bumper seams) and every panel's ripple stays at 2–6 mm after
0.5 s (visible: 1 mm), with or without this round's changes. That is the
rear-end round (back panel, rear floor, rear rails to the bumper beam,
parcel shelf).

### The rear end (2026-11-05, latest)

**Why the far sheet yielded** (`CI_PLASTIC_LIST=1` in `crash_invariants`
lists each rear-third beam past 0.2 % with its rest position). In the 56 km/h
wall the 89 plastic beams were mostly not buckled sheet:
- **the deck frame had no depth.** `deck_x` was drawn 2–4 cm off the shell's
  crown with stations 0.8 m apart. Its lower chord snapped up onto the shell at
  the centre, so at mid-span both chords were one node 1 cm apart. Its centre
  bowed 218 mm forward under its own and the lid's inertia. `rear_panel` had 3
  stations over 1.5 m, and its centre bowed 100–187 mm.
- **the backlight carried the deck.** Nothing joined `deck_x` to the parcel
  shelf but the rear glass. The glass's beams kept the skin grade's 0.4 kN
  yield (about 1 MPa over a 3.5 mm strip), so they set plastically, 6–12 %.
- **the rest:** lid seals (`bootlid~body` supports) crushed by the lid's own
  motion, fascia-to-body ties, the gutters in compression.

So the answer is missing structure and a mis-drawn frame. Neither the
buckling yield's value nor where it is applied was the cause.

**Built** (`rear_end`, `sedan_one_tier`):
- **Deck frame on the shell.** `deck_x` and `rear_panel` now have their outer
  chord on the shell's line and their inner chord one lever inside (0.08 and
  0.10 m). Each has a station at every lane node (`DECK_STATION` 0.30 m, its
  path's points). The gutter's stations are at `POLE_LANE`.
  `DECK_FRAME_V1=1` builds the old frame.
- **Package tray** (`tray`, `tray_m`, `tray_o`) from `deck_x`'s inner chord
  to the parcel shelf's upper chord.
- **Back panel** (`back_panel`, `back_panel_s`) from the rear floor's edge,
  through the rear rails' tips, to `rear_panel`'s inner chord.
- **Source and construction.** Both are sub-assemblies in the Accord
  teardown (DOT HS 811 666 §10.5, fig. 502–503). Both are built as the dash
  panel's strips: 0.8 mm sheet (a Choice in 0.7–0.9 mm), each chord at σ_y·A
  of its half strip (210 MPa, membrane: the panels' beads, a Choice). Their
  steel spreads the measured body mass. No mass is moved.
- **The rear bumper beam is steel.** The teardown's options replace "the
  baseline steel stampings" of both bumpers, and the assemblies weigh 7.84 and
  7.96 kg. It takes the front's section, `ACCORD_BUMPER_BEAM` (a Choice);
  before, it was a 6082 extrusion. `REAR_BEAM_ALU=1` keeps the old beam.
- **Glass is elastic to its fracture while brittle glass is off.** Pane beams
  hold the strip of real pane, σ·t·L (tempered 120 MPa; the laminated screen
  cracks at its plies' 45 MPa and holds). A bonded pane's mounts are its bead:
  G·w·L/t where that is under the built stiffness, and they hold τ·w·L, then
  let go. `GLASS_SKIN=1` keeps the skin grade's yield. One tier only.
- `REAR_END=0` leaves the tray and back panel out.

**Effect (Medium, single runs; n, o, p, q, g as their tests):**

| | Before | Built | Bound |
|---|---|---|---|
| Rear third plastic, wall / head-on | 89 (worst 28 %) / 21 | 53 (worst 10 %) / 19 | 0 / 0 (BX 0/0) |
| Lid against rear body, wall peak/end | 374 / 198 mm | 99 / 37 mm | ≤ 37 mm |
| Lid against rear body, head-on peak | 179 mm | 48 mm | ≤ 37 mm |
| Rear body RMS, wall | 40 mm | 18 mm | ≤ 5 mm |
| Wall ripple after 0.5 s | 1.1–3.5 mm | 1.1–4.5 mm | ≤ 1 mm |
| n max crush / b1 / dynamic / t_m / e / peak | 688 / 20.6 / 748 / 77.9 / 0.125 / 40.0 | 700 / 20.3 / 753 / 79.6 / 0.130 / 40.1 | |
| o residual | 300 mm | 291 mm | 297–445 |
| p initial / intermediate / peak | – / – / 23.9 kN | 9.95 / 14.3 / 27.7 kN | ≥ 10.0 / 15.6 / 28.4 |
| q | 5.58 × | 5.69 × | 4.48–5.32 |
| g latch / door off | 43.1 / 84.0 kN | 40.2 / 64.2 kN | passes |

All 32 of `n`'s ledgers close. The non-ignored `crash_invariants` tests pass,
9 of 9, and `invariants_small` passes 19 of 19.

**What is left in the rear:**
- the lid seals: an EPDM seal does not set; they are crushed by the lid's
  12.6–14.6 Hz bending, below the measured 17.7–41.5 Hz;
- the fascia ties;
- `deck_gutter` in compression, 1–3 %;
- the rear body's RMS. The Accord's BIW modal test (DOT HS 811 666, Defiance
  report 106072) puts "trunk rear fore-aft" first at 56.6 Hz, about 3 mm at
  40 g. At the step's bound the frame's k is 0.1–4 % of real, so its modes
  are about a tenth of real.

**Task-2 findings.**
- `profile_drag_is_the_triangles` passes (338 N against 338 N). It now also
  asserts that no `rubble_aero_tri` triangle lies on a block's faces: 0 of 868.
- `detached_door_takes_its_edge` passes on this build, but by chance.
  - On the build before this round it gave 212 / 59.
  - The joining triangles are body pieces (`frame_l`, `windscreen`) whose
    vertices are bound with a `door_f_l` node in their frame and no own-part
    fallback (`CI_DEBUG=1` lists them at rest and when the door is off).
  - Whether they count depends on how far the door flies: when their frame
    degenerates the vertex goes non-finite and the test filters it out.
  - This is a visual binding issue, passed to the visual side.

**p's door (measured, not built).** `door_belt_*` is still the two-tier
"boron × 5" (cut maximum 71.5 kN). As a real belt reinforcement
(`DOOR_BELT_REAL=1`: a DP980 hat 40 × 60 × 1.4 mm, a Choice), p falls to
8.6 / 13.1 / 22.2 kN. The peak of 27.7 kN leans on that over-strength. The
real door's inner panel, the beam's end brackets and the belt have to carry
it: the next round.

### Steel grades, lid seals and p's door (2026-11-05, latest)

**Decisions, in order.**

1. **The side reinforcements are the Accord's measured steel** (`ACCORD_REINFORCEMENT`,
   DP780; `ACCORD_GRADES=0` measures the old grades).
   - Before, the A- and B-pillar reinforcements were hot-stamped 22MnB5 and the
     rocker's DP980, "from fig. 113". Fig. 113 is the report's lightweight
     design's material table, not the baseline car's.
   - The baseline's measured data: Honda gives the body structure as 48 % HSS
     by mass, mean tensile strength 412 MPa (DOT HS 811 666 §4.4.2.1). With
     the rest mild steel, the HSS averages about 566 MPa tensile: 590–780-class
     dual-phase, no press-hardened parts. The reinforcements are the strongest
     of that class, DP780 (a Choice).
   - The 2009 Accord's own IIHS roof result is now in
     `crash-reference-targets.md`: SWR 3.87 (SWR0936, quoted in §4.8.5). The
     test's band (4.48–5.32, compact sedans) is kept.
2. **Lid seals bear elastically** (one tier; `LID_SEAL_SET=1` measures the old
   yield). An EPDM seal takes no set. The lid's own bending had crushed the
   seal beams plastically.
3. **p's door, measured and not adopted.** The Accord's door frame and the shut
   faces are built behind switches; neither passes p. Mechanism below.

**Effect of 1 and 2 (Medium; after the solver's contact merge 64e7d1ef):**

| | Before | Now | Band |
|---|---|---|---|
| q | 5.62 × | **4.98 ×** | 4.48–5.32 (Accord 3.87) |
| o residual | 296 mm (sd 10) | 288 mm (sd 14) | 297–445 |
| p initial / intermediate / peak | 10.0 / 14.3 / 27.7 kN | 9.9 / 12.7 / 22.4 kN | ≥ 10.0 / 15.6 / 28.4 |
| g latch / door off | 40.2 / 64.2 kN | 43.9 / 54.7 kN | passes |
| n max crush / dynamic / b1 / t_m / e / peak | | 635 / 747 mm / 22.0 / 78.6 ms / 0.132 / 43.3 g | 380–580 / 620–750 / 22.9–39.2 / 63–76 / 0.13–0.20 / 34–47 |
| Rear third plastic, wall / head-on | 55 (worst 11 %) / 15 | 35 (worst 10 %) / 10 | 0 / 0 |
| Lid against rear body, wall / head-on peak | 110 / 25 mm | 99 / 17 mm | ≤ 37 mm |
| Boot lid / bonnet first mode | | **22.1 / 19.2 Hz** | 17.7–41.5 Hz |
| Doors' first mode (front / rear) | | 17.6 / 12.7 Hz | report |

- The grades leave n unchanged: the 56 km/h dynamic crush is 756 mm either way.
- The lids' first modes are in the measured range on this build, both.
  `lids_first_mode_in_the_measured_range` passes.
- The non-ignored `crash_invariants` tests: 8 of 9 pass.
  `detached_door_takes_its_edge` fails (1,204 / 93 joining triangles). It is
  the visual binding, which the visual agent owns. This build's door pull
  changed the door's flight.

**p's door: what holds the door (probes `P_PATH=1`, `P_FLANGE=1`, `P_GEOM=1`).**

`P_PATH` prints every 50 mm of ram travel:
- the lateral force each door-to-body joint puts on the door, and its summed |force|;
- how close each door member is to its yield;
- how far in the door's hinge end, middle and lock end have gone.

Findings, in the order found:
- **The old beam was below the ram.** sedan1's front intrusion beam ran from
  y 0.42 at the hinge to 0.62 at the lock. FMVSS 214's ram starts 127 mm above
  the door's lowest point (y 0.50), so it met only the beam's lock end. The
  boron × 5 belt and mid rail carried p.
- **The Accord's beams run the other way** (fig. 163, 173: the thickness maps,
  read as fractions of the door's height below the belt, a Choice within
  about ±2 cm). Front door: 0.72 at the hinge end to 0.55 at the lock end.
  Rear door: 0.64 to 0.56. The maps' gauges:

  | Part | Front door | Rear door |
  |---|---|---|
  | Intrusion beam (tube) | 2.0 mm | 2.0 mm |
  | Outer belt reinforcement | 0.6 mm | 0.8 mm |
  | Inner belt reinforcement | 0.6 mm | 0.6 mm |
  | Mid reinforcement | 0.5 mm | 0.5 mm |
  | Inner panel | 0.8 mm, 1.5 mm at the hinge end (laser-welded blank) | 0.8 mm, 1.3 mm at the hinge end |
  | Lock reinforcement | 0.9 mm | |
  | Beam's hinge-end bracket | 1.2 mm | |

  `ACCORD_DOORS=1` builds these: belt and mid as ladders whose chords are the
  two strips (60 mm developed width, BH210: Choices), faces at their gauges.
  `ACCORD_DOORS=beam` builds only the beams.
- **The lock end was not held.** The door's last node row is the nearest ring
  inside its outline, 0.10–0.12 m in. So its lock face stood one ring (0.16 m)
  ahead of the B-pillar's section.
  - Its flange was a node 0.1 m ahead of the pillar, on soft ties. It moved in
    130 mm under 7 kN.
  - The inner chord's 3 cm bearing slid through it as the door's end turned:
    2–4 door ends were past their flange by 90–200 mm at 300–500 mm of ram.
  - The door's lock end went in 136–239 mm while the B-pillar went 21–27 mm.
- **Built, not adopted:**
  - `FLANGE_REAL=1`: the flange ties and bearing at the flange strip's E·t·w/L,
    capped by the bound.
  - `FLANGE_GAP=<m>`: the flange's offset.
  - `SHUT_FACES=1`: hem rings at each door's front and rear edge, 10 mm inside
    its outline. The door frames' ends move onto them. The rail ends bear along
    the car on the pillar beyond (`SHUT_BEAR`).

| p (initial / intermediate / peak, kN) | Boron × 5 door | `ACCORD_DOORS=1` |
|---|---|---|
| as built | 10.0 / 14.3 / 27.7 | 6.2 / 10.3 / 19.7 |
| `FLANGE_REAL=1` | | 6.3 / 10.5 / 22.3 |
| `FLANGE_REAL=1 FLANGE_GAP=0.10` | 10.7 / 15.1 / 33.1 (before the grades) | 6.1 / 10.4 / 28.1 |
| `SHUT_FACES=1` | 7.9 / 10.3 / 19.3 | 5.2 / 7.7 / 20.1 |
| `SHUT_FACES=1`, with `SHUT_BEAR` | | 5.2 / 7.6 / 20.7 |
| `DOOR_INNER=1` (with shut faces) | 6.9 / 11.3 / 18.6 | 6.3 / 6.9 / 14.6 |

**What this says.**
- With the shut faces the door's ends hold: the lock end is 41–45 mm in at
  300 mm of ram, against 136–239 mm before.
- The door's own members then reach 0.9 of their yields by 100 mm of ram, at
  5–8 kN. With the Accord's stampings the door is strength-limited.
- The real sections' mechanism estimate, by hand:
  - beam: 2.1 kN·m;
  - belt: 1.1 kN·m;
  - mid rail: 0.9 kN·m.

  Over the 0.95 m between faces that is about 17 kN pinned, or about 34 kN
  with the ends fixed.
- The end-fixity couple is the inner chord bearing along the car on the pillar,
  plus the hinge's or latch's pull on the outer chord. On this lattice it
  carries little: about 1 kN in the pillar bearing. The B-pillar's stations are
  0.6 m apart, and its truss box (0.14 m) is wider than the 0.12 m between the
  doors, so the door's end and the pillar's front face share one plane.
- The shut faces cost elsewhere:
  - q 4.40 (below its band);
  - o 280 mm;
  - g's door comes off at 111 kN (fails);
  - `counts_per_resolution`'s build fails at another resolution.

  So they stay a switch.
- `FLANGE_GAP=0.10` passes the peak only because longer struts tolerate the
  slide. It is not the flange's place, so it is not taken.

**Open, in order:**
1. The B-pillar's section at the doors' shut lines: its truss as wide as the
   0.12 m between the doors, a station at each rail end's height, and the
   flanges as the section's own nodes. Then `SHUT_FACES` with `ACCORD_DOORS=1`
   again.
2. The hem rows along the doors' bottom and window edges (rings give the
   vertical edges only), and on the lids, fenders and fascias.
3. The frontal: max crush 635 mm, b1 22.0, t_m 78.6 ms.

### The B-pillar's shut section (2026-11-06, measured, not adopted)

**Decision: built as open item 1 says, measured, and not adopted.** p does
not reach its 28.4 kN, and g fails.

**What is built** (`sedan::shut_section`; with `SHUT_FACES=1`;
`SHUT_SECTION=0` builds the shut faces without it, a measurement switch):
- **The pillar's real section.** It is 120 mm along the car × 70 mm across,
  as `side_sections` and the layered section give it. Before, the truss box
  was 0.14 × 0.14 m.
  - Its faces are at the doors' outlines (2.30 and 2.18), so each door's hem
    row stands one compressed seal (`HEM`, 10 mm) ahead of the pillar's face.
    Before, the hem row and the face were in one plane.
  - Its inner face (x 0.755) is the doors' inner chords' (x 0.76): each
    rail's inner end bears along the car on the pillar's inner corner. That
    is the compression side of the end-fixity couple.
  - Its chords' yields are taken over the 70 mm lever, so the section's M_p
    is unchanged.
- **A station at each door rail's end height.** The rail ends' heights are
  clustered within 5 cm (`SHUT_MERGE`, a Choice). The belt cluster moves the
  pillar's knee to the belt, and the ends within 15 cm of the foot are the
  sill's (`SHUT_SILL`, a Choice). Medium's stations: the foot, 0.54, 0.63,
  0.75 and the belt at 0.92 m, then the roof. Before, they were 0.6 m apart.
- **Flanges as the pillar's own.** Each flange is `HEM` inboard of the
  door's inner chord. It is welded to six of its own member's nodes, so two
  stations and not one plane, at the strip's real E·t·w/L (`FLANGE_REAL`'s
  rule). The bearing's reach is half a seal.
- Probes: `P_WEB=1` splits each door member's chords from its webs.
  `P_BEAR=1` gives each flange bearing's rest length, length, force and
  state. `FLANGE_INTO=<m>` puts the flange that far under the door, a
  measurement switch.

**Measured (Medium, after merging origin/merge/showcase-main at 1ab179d9):**

| | default (no shut faces) | `SHUT_FACES=1 ACCORD_DOORS=1 SHUT_SECTION=0` | with the section | `SHUT_FACES=1` (boron × 5 door), with the section | Band |
|---|---|---|---|---|---|
| p initial / intermediate / peak (kN) | 9.9 / 12.7 / 22.4 | 5.2 / 7.6 / 20.7 | 5.3 / 8.7 / 18.2 | 7.9 / 11.7 / 19.2 | ≥ 10.0 / 15.6 / 28.4 |
| q | 4.98 × | 4.36 × | **4.87 ×** | 4.41 × | 4.48–5.32 |
| o residual (mm, 8 runs) | 288 (sd 14) | 294 (sd 7) | 285 (sd 15) | **303 (sd 18)** | 297–445 |
| g: first hardware / door off (kN) | 43.9 / 54.7 | 65.2 / 112.8 | 79.9 / 80.4 | 66.7 / 107.3 | statics 25.7–57.3 |

`FLANGE_GAP` 0.03 and 0.05 give a peak of 18.8 and 18.2 kN. `FLANGE_INTO`
0.015 and 0.04 give 18.2 and 20.5 kN. Four times the door ladders' web
yields (an experiment, not kept) gives 18.9 kN.

**What holds the door now (`P_PATH`, `P_WEB`, `P_BEAR`, with the section):**
- **The ends hold through 300 mm of ram.** The hinge end is in 5–28 mm and
  the lock end 5–25 mm (136–239 mm before the shut faces). The B-pillar
  moves 2–6 mm.
- **p's first two phases are the step's bound, not the load path.** By
  100 mm of ram, single beams of every door member are at 0.85–1.0 of their
  yield (webs and chords alike). Each member has only 0–2 beams at yield.
  The door's members are at 0.2–1 % of their real k, so the door deflects as
  a soft spring. Its 50–150 mm force is 4–7 kN, against the real
  mechanism's 17–34 kN (a fixed-ended 2.1 kN·m tube yields at 15 mm). The
  BX on Rubble's solver also misses these two phases (9.0 / 14.0 kN).
  Raising them with forces kept real needs the BX's topology: fewer beam
  ends per node and longer beams, so the bound leaves each beam stiffer
  ("Yield strain: BeamNG's against ours").
- **The peak is lost at the lock end, past 350 mm.** The door works as a
  cable (catenary): its rails pull the lock end toward the hinge, and the
  latch rope takes it (6 → 25 kN at 300–500 mm). The rails' inner-chord ends
  slide 2–4 cm forward off their flange node, which is one node at the door
  end's z. They then pass inboard of it: 74, 151 and 224 mm past at 400,
  450 and 500 mm. A compression-only beam of rest length L carries at most
  k·L before it closes (at the bound, 480 kN/m × 16 mm is 7.7 kN). A real
  flange is a 50 mm strip under the door's end. A single node is not.
- **g with the shut faces.** The latch's share of the pull is 45 % against
  the statics' 51 %, but its ropes hold to 33 kN in this geometry. One of
  them runs straight inboard to the pillar's inner corner. So the latch
  lets go at 79.9 kN, past the statics' 57 kN. The shut faces alone did
  the same (65 kN, door off at 113 kN). The hardware rigs' sizing is not
  redone for the shut-face pin.

**Open, in order:**
1. The flange as a strip the door's end can slide along: a contact surface
   on the pillar's face, or bearing beams long enough that k·L exceeds the
   flange's capacity. This is the peak's mechanism.
2. The door rigs (`hardware_rigs`) sized on the shut-face geometry, so g's
   latch lets go inside its statics.
3. p's first two phases: the BX's topology, with forces kept real.

### BX-like topology: the census at the solver (2026-11-06, measured)

**Decision: the census is built and measured; no part is redesigned yet.** The
census shows that beam ends per node can buy at most 9 % on the body and 43 %
on the doors. The rest of the BX's k·L comes from node mass and beam length,
which the real masses and the A8 lanes fix. One switch was measured and not
adopted.

**The census.** `topology_census_solver` (sedan1) and `beamng_topology_census`
(the BX, `RUBBLE_JBEAM_BX`) run the same arithmetic on the solver's own
beams and masses. For each part they give:
- nodes and the median node mass;
- spring beam ends per node by kind;
- the share of nodes whose Gershgorin row is at the bound (> 0.95);
- each kind's share of the saturated rows;
- the median L, k·L, ε_y = deform / (k·L), and the springback after 20 %
  plastic strain.

For sedan1, the kinds come from geometry. Within one member's group, a beam is
classed by its angle to the member's local axis (the principal axis of its
nodes within 0.35 m):
- chord: within 32°. A chord longer than 1.5 × the member's median chord is a
  `bend`. A chord whose ends are both shell nodes is a `lane chord`.
- diagonal: 32–73°.
- ring/web: past 73°.

The other kinds:
- `joint`: between two members;
- `sheet`: skin-triangle edges;
- `shell bend`: shell beams that are no triangle's edge, that is the two-bay
  bends and the panels' braces;
- `tie`, `bearing`, `flange`, `dashpot`: by their flags and groups.

Medium, at 811c5ed3:

| Part | sedan1: nodes, node mass p50, spring ends per node, at the bound | BX: nodes, node mass, spring ends, at the bound |
|---|---|---|
| body | 592, 1.01 kg, 13.1, 87 % | 145, 2.50 kg, 12.0, 74 % |
| doors (each) | 26, 0.96 kg, 9.9, 92 % | 11, 1.50 kg, 6.9, 73 % |
| bonnet | 26, 0.53 kg, 10.7, 96 % | 24, 0.55 kg, 8.4, 92 % |
| boot lid | 16, 0.64 kg, 8.4, 94 % | 18, 0.60 kg, 9.0, 44 % |
| fenders | 20, 0.53 kg, 5.8, 80 % | 13, 0.40 kg, 7.1, 100 % |

| Part | sedan1: L p50, k·L p50, ε_y p50, springback at 20 % | BX: L, k·L, ε_y, springback |
|---|---|---|
| body | 0.36 m, 2.14e5 N, 2.96 %, 13 % | 0.55 m, 7.92e5 N, 0.85 %, 4 % |
| doors | 0.30 m, 1.74e5 N, 2.11 %, 10 % | 0.59 m, 6.50e5 N, 0.92 %, 4 % |
| bonnet | 0.37 m, 1.07e5 N, 2.80 %, 12 % | 0.45 m, 2.24e5 N, 5.51 %, 22 % |
| boot lid | 0.36 m, 1.65e5 N, 1.75 %, 8 % | 0.40 m, 1.59e5 N, 3.76 %, 16 % |
| fenders | 0.26 m, 1.43e5 N, 2.15 %, 10 % | 0.39 m, 1.57e5 N, 3.82 %, 16 % |

**The BX's nodes are at the bound too.** Its body has 74 % of its nodes at the
bound and its doors 73 %. Its stiffness is not chosen freely below the bound.
At a saturated node every beam comes down to about k = G·m/(h²·n·w), so k·L
scales with m·L/n. That ratio predicts the measured BX-to-sedan1 k·L within
15 % on every part:

| Part | m ratio | L ratio | 1/n ratio | m·L/n ratio | measured k·L ratio |
|---|---|---|---|---|---|
| body | 2.48 | 1.53 | 1.09 | 4.15 | 3.70 |
| doors | 1.56 | 1.94 | 1.43 | 4.36 | 3.74 |
| bonnet | 1.04 | 1.21 | 1.27 | 1.60 | 2.09 |
| boot lid | 0.94 | 1.11 | 0.93 | 0.97 | 0.96 |
| fenders | 0.75 | 1.52 | 0.82 | 0.94 | 1.10 |

So:
- **The boot lid and the fenders are already at the BX's k·L.** They have
  lower ε_y than the BX's because their yields are real.
- **The bonnet** is 2 × short, about evenly from length and beam ends.
- **The body and the doors** are about 4 × short:
  - Body: beam ends give only 1.09 ×; node mass gives 2.5 × and length 1.5 ×.
  - Doors: beam ends give 1.43 ×; mass and length give 3 ×.

  At a fixed areal mass a node's mass scales with s² and its beams' length
  with s, so k·L scales with s³/n. The A8 lanes cap the outward faces' s at
  0.24 m, against the BX's 0.55 m, which fails A8 (21 of 244).

**What spends the saturated rows** (sedan1, share of the rows at the bound):

| Kind | body | doors | bonnet | boot lid |
|---|---|---|---|---|
| member chords (with lane chords) | 29 % | 30 % | | |
| shell bends (two-bay bends, panel braces) | 16 % | 32 % | 44 % | 36 % |
| sheet edges | 11 % | 10 % | 46 % | 52 % |
| ties | 14 % | 7 % | | |
| diagonals | 9 % | 8 % | | |
| joints between members | 9 % | 7 % | | |
| member bends, rings/webs | 8 % | 5 % | | |
| rigidifier (the lids' `other`) | | | 9 % | 12 % |
| dashpots | 1 % | 1 % | 2 % | 1 % |

- On the doors, the largest kind that does not carry the governing load is
  the shell bends. A panel has its braces (the hinge beam across each shared
  edge) and also its two-bay bends, and both give the sheet's out-of-plane
  stiffness.
- On the body, it is the ties (14 %): station nodes that have no shell node
  are joined to the shell.

**Measured, not adopted: `LEAN_BENDS=door`** (a measurement switch, one tier
only). It leaves out the doors' two-bay shell bends and keeps their braces.

| | Built | `LEAN_BENDS=door` | Band |
|---|---|---|---|
| door spring ends per node / at the bound | 9.9 / 92 % | 9.0 / 83 % | |
| door k·L p50 | 1.74e5 N | 1.96e5 N (+13 %) | |
| p initial / intermediate / peak | 9.9 / 12.7 / 22.4 kN | 9.9 / 11.6 / 25.8 kN | ≥ 10.0 / 15.6 / 28.4 |
| o residual (8 runs) | 288 mm (sd 14) | 287 mm (sd 11) | 297–445 |
| g first hardware / door off | 43.9 / 54.7 kN | 41.6 / 52.0 kN | statics 25.7–55.3 |

The +13 % in k·L gives no gain in p's first two phases, and o's springback
is unchanged. It is the 9–43 % that topology alone can give.

**What would close the gap.** The structure would need heavier nodes and
longer beams on the faces a pole meets, with the lanes kept. BeamNG's own
pattern is light contact-only nodes in the lanes, on short ties, between
structural nodes 0.48 m apart:
- At the same areal mass, the structural nodes carry the lane's sheet
  (m × 4) on beams twice as long. That gives k·L up to about 8 × at the same
  ends per node, less what the contact nodes' ties cost.
- Each tie to a contact node of mass m_c takes about sqrt(m_c/m)/n_c of the
  structural node's row.
- A contact node must stop a pole without passing it inboard. At 0.1 kg with
  two ties, the bound gives its ties about 5e5 N/m, so it would move about
  0.1 m under 50 kN.

This concentrates the sheet's mass on the structural nodes. That is moving
mass, which this round's rule ("zero mass moved") forbids. **It is the user's
decision.** Until then, the misses that the bound limits stay: p's first two
phases, o's springback, the panels' ripple and the frontal's maximum crush.

**The two-tier `sedan` at Low (lib tests).** 61391574 gated the lids' inner
ladders inside `sedan()`, which both styles share, so the two-tier car lost
them too. Its Low deck top then had no frame within `BIND_MAX`
(`damage_units_follow_the_beams`, `inner_panels_sit_at_their_section`). The
ladders are back in `sedan()`, and only `sedan_one_tier` leaves them out
(`LID_INNER=1` keeps them). The two-tier car's Low `.truck` is now
byte-identical to 6d5c50ab's. sedan1's `.truck` is unchanged at Low, Medium
and High. Both tests still fail, now on the newer visual binding's
`ALT_BIND_MAX` check: 23 rear side-glass vertices, such as `glass_side_r_r`
at (−0.32, 1.41, 2.29). That check is the visual agent's, and they have the
report.

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
- **Collision skin (A1): no triangle outlives the joint it spans.** The
  collision skin is the closed lattice. Each triangle across a shut line
  (Medium: 241 of 740) breaks (spec §18) when a beam of its edges across the
  line breaks (seal, bolt or bearing) or when its parts' nearest hinge or
  latch group lets go; triangles inside a part stay plain, so a part that
  comes off keeps its own skin (`g` checks the door: its 23 triangles live,
  none joining it to the car). The `.truck` carries it as Rubble's extension
  of a `cab` line, a fifth field (the break group, 0 none) and a sixth (the
  edges, bit k for nodes k to k+1), and asks for contact from every node
  (spec §10.2) with `rubble_full_contact`; the shared loader
  (`ExplicitBeams::from_truck`) applies both, so the page, the World, Metal
  and CUDA get them by loading the file. Rigs of Rods files carry neither
  and load as before. Every shell node of a part with hardware is a
  contacter. The render stays per part. The 80 km/h T-bone (two vehicles)
  holds the striker's nodes outside the struck skin to 0.5 %.
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
  - `g`: door pull. The door is pulled outward at its centre of mass, so
    statics shares the pull by levers, not by strength: the latch takes
    the moment about the hinge line (53 % on sedan1's front door), then the
    hinges take the whole pull at once (the more loaded 59 %). The door
    therefore lets go group by group, and "off at the sum of the strengths"
    (the old band, 78–83 kN) would need each group loaded in proportion to
    its strength. The test asserts that the latch goes first, at a pull
    whose latch share is inside the 14 measured latch failures (13.6–29.2
    kN transverse), and that the door is off by the summed strengths.
    - sedan1 (2026-11-05): the latch at 46.3 kN (24.5 kN its share), the
      door off at 47.0 kN. Nothing on the load path broke first: only four
      skin edges (0.4–1.1 kN strength: the buckling-based yield, which
      JBeam's one beamDeform applies in tension too).
    - The two-tier `sedan` now fails it: its latch holds to 81 kN (36.5 kN
      share), past the measured maximum.
    - In situ (`G_SPLIT=1`), against the ellipse through each unit's
      measured loads: the latch assembly carried 1.58× its median before
      letting go, the upper hinge let go at 0.78× its floor and the lower
      at 2.49×, the hinges loaded dynamically by the latch's release. Rope
      groups sized in a rigid rig share differently on the softened
      pillars; the FMVSS 206 rigs in the solver (`r`) run 15–40 % over the
      static solve (latch transverse 23.5 kN, hinge pair 77 kN). Open.
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
  * The glass panes now use the physics' own units (`Built.damage_units`:
    `glass_windscreen`, `glass_backlight`, `glass_side_{f,r,q}_{l,r}`, each
    the pane's beams and its mounts, mapped to the solver's beams by node
    pair). A pane breaks when one of them breaks, with no strain trigger of
    the renderer's own (`BRITTLE_GLASS` decides how the physics breaks
    them). Lamps and mirrors keep the mount derivation until their units and
    parts land; then they ride, and fall with, those parts.
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
  the paint worn through (µm). The clearcoat dulls with plastic strain (from
  0.4 % to 6 %). Paint wears by Archard's law, depth = K·p·s / H over the
  distance s a node slides while in contact by the solver's own rule (at or
  inside the floor or a static box, where the solver pushes it out; a node
  merely near a wall wears nothing). K = 0.05: two-body abrasive wear is
  K ≈ 10⁻² to 10⁻¹ (Rabinowicz, *Friction and Wear of Materials*, 2nd ed.,
  1995); the middle of the range is a Choice. H = 0.2 GPa, a clearcoat's
  indentation hardness: a Choice (we found no published value to cite).
  p = 1 MPa: a Choice, because the solver does not expose its contact
  forces. That wears 250 µm a metre. The layers are a typical OEM build-up
  (Streitberger & Dössel, *Automotive Paints and Coatings*, 2nd ed., 2008):
  clearcoat 45 µm, basecoat 20, primer 25, e-coat 20. So primer shows in
  streaks past 65 µm worn (26 cm slid), and bare steel past 110 µm (44 cm).
  Wear accumulates forward in time (a replay scrubbed back keeps it).
  `showcase-car.html?debugWear=1` shows the tear field, plastic strain and
  paint worn as red, green and blue.
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
* **Creases** (`skin.rs` `crease_envelope`, the renderer's `envelope()`).
  Each skin node has a tangent plane through it (its skin normal), or two
  where a fold runs through the node (its skin triangles fall in two groups,
  the most deviant more than 20° from the mean: `node_planes`), so a node on
  a fold line carries both panels' planes rather than a bevel between them.
  A body vertex's point on its binding triangle moves, along the planes'
  barycentric mean normal, onto the plane nearest it, the planes weighted
  wᵢ·exp(−(|t| − min|t|)/(0.08 L)) (t the distance to the plane, L the
  binding triangle's size), and its rest normal turns with the weighted
  planes' normal. At a fold the two sides' planes cross along a straight
  line, so the crease is straight and sharp across (a 60° fold of a 10 cm
  triangle switches over about 1 cm) and smooth along it, wherever the
  binding triangles' edges run; a mild bend keeps barycentric (Phong-like)
  weights and stays smooth. Only the change from the rest envelope is
  applied, so the rest pose and rigid motion are exact. There is no
  faceted fallback (the binding triangle's frame normal drew its edges as a
  sawtooth along crushed lips).
  `showcase-car.html?crease=0.04&splitDeg=12` tries other constants (sharper
  than 0.08 shows the visual mesh's own triangles as a sawtooth);
  `showcase-car-check.mjs wall_56_bonnet,zebra_wall_56_bonnet` is the
  close-up.
* **Level of detail.** Three levels: the full car, `showcase_visual_lod(built,
  0.45)` (wasm `visualLod`: a coarser body, single-step rolls, fewer
  segments) and `showcase_visual_lod(built, 0.2)` (`visualFar`).
  * The renderer picks a level per car on the GPU from the last frame's eye
    (indirect dispatch and draw), so only that level is skinned and drawn:
    the finest level whose median rest edge (`skin::median_edge`, wasm
    `edgeLength`) spans at least 3 pixels at the car's distance
    (`LOD_PX`; `draw`'s `height` is the viewport's). Finer triangles than
    that change nothing visible (the 10-car row at 16 m and 30 m renders
    the same on the second level as on the full mesh) but cost quad
    overshading: profiling (`?perfFlat=1`, a flat fragment shader; `?perfSkip=`
    `opaque,glass,reflect,contact,fx`) showed the near level at 16 m spent
    5.3 of its 7.0 ms (10 cars) in fragment shading of sub-pixel triangles,
    0.7 ms in vertex and raster work, 1.1 ms in the floor reflection.
  * Measured with 10 cars on an M3 Max at 1080p with 4× MSAA
    (`showcase-car.html?perf=1&n=10`, 2026-11-05): 0.39 ms per car at 16 m,
    0.38 ms at 30 m, 0.31 ms at 80 m (was 0.70 at 16 m: the full mesh). A
    single car filling the frame costs 3.9 ms, of which 0.6 ms is vertex and
    raster work (the paint's shading). The fragment discards (tears,
    shattered glass) cost 6–13 % (`?perfNoDiscard=1`). Another GPU user
    moves these by ±10 %.

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
* **Textures.** A material's colour map (`colorMap` or `baseColorMap` in
  Stages[0]) is the file of that name in the vehicle's folder.
  `engine_softbody::flextex` decodes PNG (8- or 16-bit, any colour type) and
  DDS (BC1–BC3 by the block formats' definitions, 32-bit RGBA/BGRA) into a
  1024² sRGB layer with its mips (wasm `decodeTexture`). The renderer keeps
  one texture array (`setTextures`), and each vertex carries its layer in
  its material word's high half. The map multiplies the material's colour,
  sampled with explicit gradients (v flipped: images are stored top row
  first). A colour-mapped material shades satin (`mat::DASH`), not as
  paint. The supercar's 15 maps (carbon fibre, fabric, wood, the dials and
  buttons) all load; normal, roughness and AO maps are not read yet.
* **Level of detail.** Three levels: the full mesh, each mesh clustered on a
  3 cm grid per material and normal direction (`cluster`, `LOD_CELL`), and
  on a 9 cm grid (`FAR_CELL`; wasm `visualFar`). The renderer picks them as
  for the authored car (the finest whose median edge spans 3 pixels).
* **Pages.** wasm `jbeamVisual(spec, meshes)`; `showcase-car.html?mod=<folder>`
  renders a mod's visual at rest (no solver; `?n=` copies); the crash lab
  loads `softbody.html?...&mod=<folder>&jbeamMode=native` (a folder served
  with a `files.json` listing). `showcase-car-check.mjs mod_*` and
  `crash-lab-check.mjs --car supercar|tutorial` use the gitignored
  `inspiration/beamng-mods` (`RUBBLE_MODS`), skipped when absent.
* **Measured** (M3 Max, 1080p, 4× MSAA, 10 copies): the Monster supercar
  (95 flexbodies, 1.52M triangles: about 40 times a typical BeamNG car's
  30–40k; levels 268k and 84k) costs 0.95 ms a car at 16 m (was 1.57) and
  0.61 ms at 30 m (was 0.91), still over 0.5 ms. It is fragment bound, not
  triangle bound: a 25 cm third level (33k triangles) costs the same, and
  a flat fragment shader brings the 10 at 30 m from 6.1 to 1.2 ms. A
  single car costs about 1.5–2 ms in this timed pass at any distance once
  any of it is on screen (0.24 ms off screen), for the authored car too, so
  the per-car figures carry a share of that constant.

### Glass (A9, open)

- `Built.damage_units` names each pane by the agreed names:
  `glass_windscreen`, `glass_backlight`, `glass_side_{f,r}_{l,r}`. The sedan
  has no quarter glass. Each unit is the pane's beams plus its mounts (the
  beams from the pane to the shell around it).
- `brittle_glass` holds the A9 model, behind `BRITTLE_GLASS`, which is off:
  - tempered panes break at σ/E = 120 MPa / 70 GPa, with one detacher group
    a pane;
  - the laminated windscreen cracks (yields) at its plies' 45 MPa / 70 GPa
    and holds;
  - the windscreen and backlight hang on a urethane bead (TEROSON PU 9096
    PL datasheet: G ≈ 1.2 MPa by DIN 54451 over a 5 mm layer; shear
    strength 5–6 MPa fully cured). Each mount holds τ·w·L, taking a 5 mm by
    10 mm bead with L its share of the edge;
  - side glass bears in its run channel (compression only) and is held by
    its regulator along the bottom.
- **Rest pose = installation pose (a Choice, 2026-11-05).** A pane is
  bonded into a body already standing under its own weight. The car is
  therefore written standing at rest, and it spawns that way: nothing moves
  at spawn, so the installation pose is the spawn pose. There is no "skip the
  first N ms" rule.
  - `rest_pose` stands the car on flat ground on the CPU reference (tyres
    touching, no drop), with its glass unbreakable, for `INSTALL_SETTLE` =
    2 s.
  - The nodes then take their settled positions. Every beam and shock keeps
    the solver's rest length through `rest_scale` (beams' fifth field) and
    precompression.
  - The tyres keep their deflected shape (`rubble_tyre_pose`, built round
    their axles' design positions).
  - Pairs that came into contact while settling keep colliding
    (`rubble_spawn_contacts`).
  - The hardware ropes are sized again on the standing car.
  - The spec is `jbeam-semantics-spec.md` §18.2. Consumers spawn with
    `ExplicitBeams::spawn_clearance()`: the patch depth for a rest pose, 2 cm
    otherwise.
  - `Built.design` keeps the design positions, which are mirror-symmetric;
    the rest pose leans 1–2 mm toward the driver.
  - Medium: nodes sit up to 45 mm from the design (mostly the suspension's
    sag); no beam yields while settling.
  - In `a` the car stands with 0.3 mm wobble, body nodes under 0.01 m/s, and
    energy rises between frames of at most 0.04 J. That rise is the energy
    sum's f32 rounding (`energy_rounding`).
  - Every reference test stays in band (n, o, p, q, g, t).
  - The settle used to start with a 2 cm drop. That landing yielded the
    roof's lumped panel beams by up to 0.9 mm, which the rest pose would
    have carried.
- **Brittle glass is still off.** With the installation pose, every pane
  still breaks at rest. A pane beam's break force is its strain times the
  model's k·L: 0.17 % of about 180 N, roughly 0.3 N. The pane nodes' own
  weight, about 0.7 N each, exceeds it. The step-softened beams are about
  10⁵ × softer than glass, so a strain criterion becomes a force too small
  to carry the pane. A real pane's strip would hold σ·t·L, about 144 kN at
  4 mm × 0.3 m, which these beams never reach.
- **Aperture racking (a Choice; written, off; 2026-11-05).**
  `aperture_gauges` joins the opening's corners by its edges and diagonals
  with gauges. A gauge is a beam of negligible stiffness (`GAUGE_K`) that
  breaks at the racking that brings the glass to σ/E.
  - **Corners are frame nodes of the opening's own members**
    (`aperture_members`):
    - windscreen: the screen pillars, front header and cowl;
    - backlight: the C-pillars, rear header and parcel shelf;
    - side glass: the door's sash and belt.

    Each corner is the frame node farthest along each diagonal of the
    opening (its axes scaled to its extent), within 30 cm of the opening's
    outline.
  - **The gauge's rest is its length as the solver measures it.** The default
    (RoR-compatible) profile's fast inverse square root is off by up to
    0.18 % from the exact length the loader takes as a rest. That error is
    larger than glass's break strain. It is what made every pane "break at
    rest", more than the old 2 cm spawn drop did.
  - **The mount is in series with the pane (exact for two shear layers in
    series with an elastic strip).** Across a span L the opening racks
    ε = ε_g (1 + 2 E t t_b / (G w L)) for a strain ε_g in the glass, with the
    urethane bead's G, w and t_b (TEROSON PU 9096 PL) and the pane's t (NHTSA
    DOT 88988 table 2: tempered 3.5 mm; laminated 2 × 2.1 mm on PVB).
    - Over 1.5 m the factor is about 140: tempered glass breaks at about 24 %
      racking, laminated cracks at about 9 %.
    - A bonded pane barely feels its frame. Its bead tears (τ·w·L, as
      sized) or the glass is hit first.
    - **Choice:** side glass in its run channel takes the same compliance;
      no measured channel data was found.
  - **A pane's own beams are strips of the real pane.** Each holds σ·t·L,
    which the step-softened beam never reaches. A tempered pane's beams go
    only with its group; the laminated windscreen's yield at its plies'
    crack force and hold.
  - **Measured with `BRITTLE_GLASS=1 GLASS_REPORT=1`, Medium,** with door
    sashes as frame-tier members (`window_frame`):
    - no pane breaks at rest (10 s, Low and Medium) or driving at 50 km/h
      (gauges up to 0.09 % racking);
    - no pane breaks in any NCAP frontal at 47–59 km/h, so the backlight
      survives;
    - the struck front side glass shatters in the 75° side pole at 31, 32
      and 34 km/h (not at 30 or 33);
    - both struck side panes shatter in the 80 km/h T-bone;
    - the windscreen does **not** crack in the frontal or in the FMVSS 216a
      roof crush. Its racking never nears 9 %, and a laminated screen cracks
      from being hit or bent, which needs the direct-impact trigger.
  - **Glass stays off:** the windscreen check fails.
- **Door sashes (window frames) are written but not in the car.** As
  frame-tier ladders round each window (`window_frame`, corner stations
  only), they moved three checks out of band:
  - `g`: the door came off at 78.3 kN, against 78.4–83.4;
  - `n`: restitution 0.2002, against at most 0.20;
  - `t`: the latch let go at 0.05 s, and 46–57 striker nodes ended up inside
    the struck skin (1 % allowed). The shut-line triangles next to the latch
    die with it while the hinged door is still pressed in, which opens a
    hole.

  The sashes tried were `window_frame(riding(ladder(member("door_sash_f",
  [(0.80, 1.10, 2.98), (0.70, 1.33, 2.64), (0.69, 1.36, 2.36), (0.79, 1.10,
  2.36)], 0.05, 0.05, OUT, "hsla", [1, 1], 0, true, 1))))`, and `door_sash_r`
  through (0.79, 1.10, 2.12), (0.69, 1.36, 2.12), (0.69, 1.33, 1.64) and
  (0.79, 1.10, 1.58). Without them, the side-glass gauges have only belt
  corners.
- **What is open:**
  - Direct impact against measured data, on the solver's contact-force
    break trigger. The solver owner is adding `rubble_contact_breaks`; it
    is not here yet. The data is NHTSA DOT 88988: 227 g ball break heights
    for production tempered glass, mostly 35–85 ft at the centre, 2 ft at a
    painted edge.
  - The sashes together with the T-bone's skin.

### A6: where the hardware's loads reach its strength (2026-11-05)

`u_hardware_where_loads_reach_strength` (`--ignored`; `U_ONLY=overlap,pole,open,side`,
`U_KMH`, `U_TRACE=<group>` for a group's pull every millisecond) reports each
hinge and latch group's peak pull against the FMVSS 206 values it is sized to
(hinge 29.8–36.2 kN each, latch 20.5–28.0, secondary 10.3–19.0;
transverse–longitudinal). Medium, front left door unless noted:

| Scene | Peak pulls (kN) | Let go |
|---|---|---|
| IIHS small overlap 64 km/h (25 % left, rigid box) | latch 4.6, hinges ≤ 6.3 (rear door) | none |
| small overlap 80 km/h | latch 12.0, rear door lower hinge 12.5 | none |
| side pole 254 mm, 32 km/h, at the hinge line | upper hinge 52.6, lower 25.6, latch 13.5 | both hinges (0.05, 0.08 s) |
| side pole at the hinge line, 40 / 50 / 64 km/h | upper 52.0 / 51.2 / 55.9, lower 26.8 / 23.7 / 24.7 | both hinges; latch held (20.3 at 64) |
| side pole, 32 km/h, at the latch line | latch 20.9, secondary 18.7 | latch |
| side pole at the latch line, 50 / 64 km/h | latch 19.1 / 21.3, secondary 27.0 / 24.0 | latch and secondary; hinges held |
| door open 60°, reversing into a pole, 10 km/h | hinges ≤ 2.5 | none: the door swings freely (the car has no check link or opening stop) |
| on its side from 1 m | ≤ 5.8 (secondary, right front) | none |
| car-to-car T-bone 80 km/h, door middle (`t`) | latch 23.8, lower hinge 19.5 | none (inside the measured latch spread) |
| T-bone, striker 0.7 m forward (`TB_DZ`) | latch 29.2, secondary 24.0 | latch, secondary |

- The hinge pull rises over about 20 ms to 40–49 kN before it lets go (no
  one-step contact spike).
- The latch lets go at 19–21 kN, its transverse strength.
- The lower hinge lets go at 24–27 kN, under the 29.8 kN floor. This is a
  consequence of the model, not a measurement. The load points outside
  FMVSS 206's two pulls (the pole drives the door's front end in and back),
  and the rope decomposition puts most of it on one rope. No open combined-
  or oblique-load hinge data was found to check it against.
- Open gap: the car has no door check link or over-travel stop. No open
  measured strength was found for either: SAE 2011-01-0790 (check-load
  durability) and SAE 2003-01-0871 (closing effort; the check link as a
  hinge torque) give no failure loads. Without a stop, an open door swings
  free when struck, so the door-open scene cannot load its hinges.
- No scene takes every group of a door. The hinge line takes the hinges; the
  latch line, the latch.

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
