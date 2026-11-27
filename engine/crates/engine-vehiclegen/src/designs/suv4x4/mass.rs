//! The 4x4's mass, part by part (`engine/docs/generated-vehicles-lessons.md` §22).
//!
//! The car is built on the sedan's definitions (its shell groups, `mechanical::sedan_parts`,
//! the Accord's closures), which weigh 1,410 kg rigged. A mid-size body-on-frame 4x4 on the
//! same 2.70 m wheelbase weighs about 2,000 kg ([`super::SUV_KERB_KG`], the Xterra Pro-4X).
//! The difference is what such a vehicle has that a unibody sedan lacks or has heavier, each
//! added to the part that is it, or to the nodes where it is: nothing is scaled by a factor.
//!
//! Each figure says what it rests on: **published** (with the source) or an **engineering
//! estimate** (with its uncertainty). No figure here was fitted to the kerb.
//!
//! The axle split is not set here: the masses are where their parts are. With the sedan
//! lattice's layout the car is 48.7 % front / 51.3 % rear (`tests/suv4x4_rest.rs` in
//! `engine-softbody-authoring`), where a body-on-frame 4x4 is nearer 52-55 % front.
//!
//! What is mass only, not structure: the ladder frame. A body-on-frame vehicle's frame is
//! two boxed rails and its crossmembers under the floor; the shared part builders have no
//! frame (its nodes, beams and body mounts), so its mass sits where the frame is (at the
//! subframes, which stand for its front and rear crossmembers) and its stiffness and
//! strength are not modelled: the body's own unibody lattice carries the load.

use crate::mechanical::MechPart;
use crate::structure::{ShellNode, Structure};

/// Mass added to a mechanical part: `kg` over the nodes whose tag (the name after the
/// part's prefix) is in `tags` (all the part's nodes when empty), in proportion to the
/// masses they already carry, so the part's own centre of mass stays where it was.
pub struct MechAdd {
    pub part: &'static str,
    pub tags: &'static [&'static str],
    pub kg: f64,
    pub what: &'static str,
    pub basis: &'static str,
}

const fn add(part: &'static str, tags: &'static [&'static str], kg: f64, what: &'static str, basis: &'static str) -> MechAdd {
    MechAdd { part, tags, kg, what, basis }
}

/// What the 4x4 adds to the mechanical parts (kg). Corner items are per corner.
pub const MECH: &[MechAdd] = &[
    // -- the ladder frame (mass only)
    add("subframe_front", &["rl", "rr", "c"], 50.0, "ladder frame, middle section's front half",
        "engineering estimate: a mid-size ladder frame (boxed rails, 6-7 crossmembers) 140 kg ± 30 %, less ~40 kg for its front and rear horns, which the unibody's own front and rear rails (shell groups A, E1) already carry: 100 kg, at the ends of the section between the axles"),
    add("subframe_rear", &["fl", "fr", "cf"], 50.0, "ladder frame, middle section's rear half", "as above"),
    // -- engine and driveline
    add("engine", &[], 40.0, "4.0 L V6 against the sedan's four-cylinder",
        "engineering estimate: a dressed 4.0 L aluminium V6 about 210 kg ± 15 kg; the sedan's 169.9 kg is published (DOT HS 811 666, the 2011 Accord)"),
    add("transmission", &[], 30.0, "five-speed automatic rated for the V6's torque",
        "engineering estimate: ~75 kg ± 15 with fluid against the 45 kg the sedan's transmission nodes carry"),
    add("transmission", &["t"], 35.0, "part-time transfer case at the transmission's tail",
        "engineering estimate: a chain-drive two-speed case with fluid 30-40 kg"),
    add("subframe_front", &[], 40.0, "front differential (32 kg) and front driveshaft (8 kg) on the front crossmember",
        "engineering estimate: an independent-front 4x4's front axle carrier 28-38 kg, a steel propshaft 6-10 kg"),
    add("hub_FL", &[], 11.0, "front CV half-shaft's outer half (8 kg) and larger brakes (3 kg)", "engineering estimate: a 4x4 CV shaft 7-10 kg, half on the knuckle; ±50 %"),
    add("hub_FR", &[], 11.0, "as hub_FL", "as hub_FL"),
    add("differential", &[], 25.0, "solid rear axle's carrier against the sedan's differential (42 kg)",
        "engineering estimate: a 226-230 mm ring gear carrier with oil 60-70 kg"),
    add("hub_RL", &[], 18.0, "solid rear axle's tube, shaft and drum-in-disc brake (12 kg) and half a leaf spring (6 kg)",
        "engineering estimate: a live axle assembly 90-110 kg across both corners and the carrier; a leaf spring 10-15 kg, half unsprung; ±30 %"),
    add("hub_RR", &[], 18.0, "as hub_RL", "as hub_RL"),
    add("exhaust", &[], 5.0, "V6 manifolds and larger pipe", "engineering estimate ± 3 kg"),
    // -- wheels, fuel, armour
    add("wheel_FL", &["f1", "f2", "f3"], 10.0, "265/75 R16 on a 16 × 7 steel rim against a 215/60 R16 on alloy",
        "engineering estimate: about 32 kg against 21 kg a wheel (tyre 20-23 kg, steel rim 11-13 kg); ±3 kg"),
    add("wheel_FR", &["f1", "f2", "f3"], 10.0, "as wheel_FL", "as wheel_FL"),
    add("wheel_RL", &["f1", "f2", "f3"], 10.0, "as wheel_FL", "as wheel_FL"),
    add("wheel_RR", &["f1", "f2", "f3"], 10.0, "as wheel_FL", "as wheel_FL"),
    add("fuel_tank", &[], 30.0, "80 L tank, full (kerb), against the sedan's",
        "engineering estimate: 80 L of petrol (0.745 kg/L) 60 kg and a 10 kg tank against the sedan's 40 kg"),
    add("subframe_front", &["fl", "fr", "kl", "kr"], 10.0, "front skid plate", "engineering estimate: 4 mm steel, 0.3 m², ±4 kg"),
    add("transmission", &["t"], 6.0, "transfer case skid plate", "engineering estimate ± 3 kg"),
    add("fuel_tank", &[], 4.0, "fuel tank skid plate", "engineering estimate ± 2 kg"),
];

/// The closures' masses (kg) for `Physics` (the sedan's are published: DOT HS 811 666).
pub const DOOR_F_KG: f32 = 33.5;
pub const DOOR_R_KG: f32 = 27.7;
pub const HOOD_KG: f32 = 21.0;
pub const FENDER_KG: f32 = 6.0;
/// The tailgate alone; the spare wheel it carries is added when the car has one.
pub const TAILGATE_KG: f32 = 22.0;
pub const BUMPER_F_KG: f32 = 15.5;
pub const BUMPER_R_KG: f32 = 13.0;
/// A 265/75 R16 on a steel rim, as the road wheels (above).
pub const SPARE_KG: f32 = 32.0;
/// The closures' figures, for the record (part, kg, the sedan's kg, basis).
pub const CLOSURES: &[(&str, f32, f32, &str)] = &[
    ("door_f (each)", DOOR_F_KG, 58.99 / 2.0, "engineering estimate: a door 0.15 m taller, ±4 kg"),
    ("door_r (each)", DOOR_R_KG, 47.46 / 2.0, "engineering estimate ± 4 kg"),
    ("hood", HOOD_KG, 17.90, "engineering estimate: a flat steel hood, ±3 kg"),
    ("fender (each)", FENDER_KG, (147.0 - 58.99 - 47.46 - 17.90 - 12.37) / 2.0, "engineering estimate: flared, ±1 kg"),
    ("tailgate", TAILGATE_KG, 12.37, "engineering estimate: a full-height steel tailgate with its glass and spare carrier against a decklid, ±5 kg"),
    ("bumper_f", BUMPER_F_KG, 7.96 + 2.5, "engineering estimate: a steel tube bar on brackets, ±4 kg"),
    ("bumper_r", BUMPER_R_KG, 7.84 + 1.3, "engineering estimate ± 4 kg"),
    ("spare wheel (on the tailgate)", SPARE_KG, 0.0, "as a road wheel (above): engineering estimate ± 3 kg"),
];

/// Rock sliders on the sills (both sides), kg, over the sill nodes (group C2).
pub const SLIDERS_KG: f64 = 20.0;
/// The roof rack (when fitted), kg, over the pillars' top nodes (its feet stand on the roof
/// rails above them).
pub const RACK_KG: f64 = 12.0;

/// The mechanical parts with the 4x4's additions. Panics on a part or tag the table names
/// that the parts do not have (a renamed part must not silently lose its mass).
pub fn mechanical(parts: &mut [MechPart]) {
    for a in MECH {
        let p = parts.iter_mut().find(|p| p.name == a.part).unwrap_or_else(|| panic!("suv_4x4 mass: no part {}", a.part));
        let tag = |n: &str| n.split_once('_').map_or(n.to_string(), |x| x.1.to_string());
        let idx: Vec<usize> = (0..p.nodes.len()).filter(|&i| a.tags.is_empty() || a.tags.contains(&tag(&p.nodes[i].name).as_str())).collect();
        assert!(!idx.is_empty() && (a.tags.is_empty() || idx.len() == a.tags.len()), "suv_4x4 mass: {} has not every node of {:?}", a.part, a.tags);
        let base: f64 = idx.iter().map(|&i| p.nodes[i].mass).sum();
        for i in idx {
            p.nodes[i].mass += a.kg * p.nodes[i].mass / base;
        }
    }
}

/// The body-in-white's node masses for the 4x4: the rock sliders on the sills (C2) and the
/// roof rack (when fitted) on the pillars' top nodes. Returns the kg added by each.
///
/// The shell's own sheet keeps the reference car's group masses (`structure::GROUPS`).
/// Re-massing it member by member (a member's sheet at the same gauge goes with its sheet
/// area) was tried and not kept: on the 4x4 the shared definition gives its rear pillar
/// (`c_pillar`, from the rear header to 0.16 m above the back light's base) 0.014 m² of
/// sheet against the sedan's 0.154 m², a 0.14 m pillar where the 4x4's D-pillar stands
/// 1.1 m tall, so the rule would take mass off a pillar that is in fact larger
/// (`engine/docs/generated-vehicles-lessons.md` §22).
pub fn shell(st: &mut Structure, rack: bool) -> [f64; 2] {
    let mut added = [0.0; 2];
    let spread = |st: &mut Structure, of: &dyn Fn(&ShellNode) -> bool, kg: f64| {
        let idx: Vec<usize> = (0..st.nodes.len()).filter(|&i| of(&st.nodes[i])).collect();
        assert!(!idx.is_empty(), "suv_4x4 mass: no shell nodes to carry {kg} kg");
        for &i in &idx {
            st.nodes[i].mass += kg / idx.len() as f64;
        }
        kg
    };
    added[0] = spread(st, &|n| n.group == "C2", SLIDERS_KG);
    if rack {
        // (the top node of each pillar: `sh_d1_tl`, `sh_d2_tl`, `sh_d3_tl` and the right's)
        added[1] = spread(st, &|n| matches!(n.group, "D1" | "D2" | "D3") && n.name[6..].starts_with('t'), RACK_KG);
    }
    added
}
