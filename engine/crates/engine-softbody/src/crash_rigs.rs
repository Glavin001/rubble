//! Crash-test rigs for the explicit solver's crash lab
//! (`engine/docs/crash-reference-targets.md` §10): the FMVSS 214 moving deformable barrier as a
//! JBeam vehicle, and a multi-node door pull added to any vehicle.
//!
//! **FMVSS 214 MDB** ([`mdb214_jbeam`]): 49 CFR 587 Subpart B (§ 587.6: 3,015 lb
//! = 1,367.6 kg in configuration 2, wheelbase 102 in, track 74 in, centre of mass
//! 44.2 in behind the front axle and 19.7 in up), its face per NHTSA TP-214D:
//! 1,676 mm wide, 559 mm high, 279 mm ground clearance, 381 mm deep, 483 mm at the
//! bumper; aluminium honeycomb crushing at 310 kPa (45 psi, main block) and
//! 1,689 kPa (245 psi, bumper element). The honeycomb is columns of SUPPORT beams
//! (push only) that yield at the crush stress times their tributary area and never
//! harden (`deformLimitStress` = the crush force), densifying at 80 % crush.
//! Everything the open documents do not give is marked **Choice** below.

use crate::explicit_beams::{Beam, ExplicitBeams, ExplicitKind, ExplicitType, Node, NODE_NO_GROUND};
use glam::Vec3;
use serde_json::{json, Value};

/// MDB mass (kg): 49 CFR 587.6(c), 3,015 lb.
pub const MDB_MASS: f32 = 1367.6;
/// Face width, height, ground clearance, main depth, bumper protrusion (m).
pub const MDB_FACE: [f32; 5] = [1.676, 0.559, 0.279, 0.381, 0.483 - 0.381];
/// Crush strengths (Pa): main block 310 kPa, bumper element 1,689 kPa (TP-214D).
pub const MDB_CRUSH: [f32; 2] = [310.0e3, 1689.0e3];
/// FMVSS 214 S7: 33.5 mph (m/s), crabbed 27°.
pub const MDB_SPEED: f32 = 33.5 * 0.447_04;
pub const MDB_CRAB_DEG: f32 = 27.0;

/// Face grid: columns across, rows up (nodes at the edges: tributary areas).
const COLS: usize = 17;
const ROWS: usize = 7;
/// **Choice**: the bumper element covers the face's lowest third (rows 0–2); the
/// drawing that gives its height (DSL-1285) is not openly available.
const BUMPER_ROWS: usize = 3;
/// **Choice**: face node mass (kg). The honeycomb and its sheets are light; the
/// mass sets how stiff its elastic range may be at the 2 kHz step, and is part of
/// the 1,367.6 kg. The grid is fine enough that a node carries a few kN (a coarser
/// one punched through a car's door skin).
const FACE_NODE_MASS: f32 = 0.8;
/// **Choice**: elastic strain at the crush stress (bumper, main).
const ELASTIC_STRAIN: [f32; 2] = [0.20, 0.10];
/// Densification: crush stops at 80 % of the depth (**Choice**, aluminium
/// honeycomb's usual range is 70–80 %).
const DENSIFY: f32 = 0.8;

fn node(id: &str, p: [f32; 3], props: Value) -> Value {
    json!([id, p[0], p[1], p[2], props])
}

/// The MDB as a JBeam vehicle (part `rubble_mdb214`), facing −Y (JBeam's forward),
/// its face's front at y = 0 (bumper at −0.102), its skids on the ground.
pub fn mdb214_jbeam() -> String {
    let [w, h, clear, depth, bump] = MDB_FACE;
    let xs: Vec<f32> = (0..COLS).map(|i| -0.5 * w + w * i as f32 / (COLS - 1) as f32).collect();
    let zs: Vec<f32> = (0..ROWS).map(|j| clear + h * j as f32 / (ROWS - 1) as f32).collect();
    let (dx, dz) = (w / (COLS - 1) as f32, h / (ROWS - 1) as f32);
    // Tributary width/height of an edge node is half a cell.
    let trib = |i: usize, n: usize, d: f32| if i == 0 || i == n - 1 { 0.5 * d } else { d };
    // The bumper band's tributary heights: rows 0..BUMPER_ROWS cover the band from
    // the bottom up to row BUMPER_ROWS − 1 (half a cell each at its ends).
    let bump_h = |j: usize| if j == 0 || j == BUMPER_ROWS - 1 { 0.5 * dz } else { dz };

    // Carriage: the back plate (a node behind each face node, at the main block's
    // back), then a mid and a rear frame (skids at 2 cm and the face's top). Wheelbase
    // 2.591 m, track 1.880 m (587.6(c)); **Choice**: the front axle 0.6 m behind the
    // face, so the centre of mass, 1.123 m behind it, is 1.723 m behind the face.
    let front_axle = 0.6f32;
    let cg_y = front_axle + 44.2 * 0.0254;
    let cg_z = 19.7 * 0.0254;
    let (y_mid, y_rear) = (cg_y, 2.0 * cg_y - depth);
    let face_count = COLS * ROWS + COLS * BUMPER_ROWS;
    let face_mass = face_count as f32 * FACE_NODE_MASS;
    let carriage = MDB_MASS - face_mass;
    // Equal masses at the back plate and the rear frame put the centre of mass at the
    // mid frame (y = cg_y); 45 % / 10 % / 45 % (the ends heavy: the back plate is the
    // face's rigid support). The frames' top share sets its height.
    let (m_back, m_mid, m_rear) = (0.45 * carriage, 0.1 * carriage, 0.45 * carriage);
    let back_z = zs.iter().sum::<f32>() / ROWS as f32;
    let frame_x = [-0.5 * 1.880, 0.0, 0.5 * 1.880];
    let (z_lo, z_hi) = (0.02f32, clear + h);
    // m_back·back_z + (m_mid + m_rear)·(z_lo + f·(z_hi − z_lo)) = carriage·cg_z
    // (the face's nodes sit at the back plate's mean height to within a few cm).
    let f = ((carriage * cg_z - m_back * back_z) / (m_mid + m_rear) - z_lo) / (z_hi - z_lo);
    let f = f.clamp(0.0, 1.0);

    let mut nodes = vec![json!(["id", "posX", "posY", "posZ"])];
    nodes.push(json!({"selfCollision": false}));
    nodes.push(json!({"collision": true}));
    nodes.push(json!({"frictionCoef": 0.5}));
    let name = |p: &str, i: usize, j: usize| format!("{p}{i}_{j}");
    for (i, &x) in xs.iter().enumerate() {
        for (j, &z) in zs.iter().enumerate() {
            nodes.push(node(&name("f", i, j), [x, 0.0, z], json!({"nodeWeight": FACE_NODE_MASS})));
            if j < BUMPER_ROWS {
                nodes.push(node(&name("b", i, j), [x, -bump, z], json!({"nodeWeight": FACE_NODE_MASS})));
            }
            let mb = m_back / (COLS * ROWS) as f32;
            nodes.push(node(&name("p", i, j), [x, depth, z], json!({"nodeWeight": mb})));
        }
    }
    // **Choice**: frictionless skids stand in for the crabbed, free-rolling wheels.
    let frame = |tag: &str, y: f32, m: f32, nodes: &mut Vec<Value>| {
        for (i, &x) in frame_x.iter().enumerate() {
            for (k, (z, share)) in [(z_lo, 1.0 - f), (z_hi, f)].into_iter().enumerate() {
                let props = json!({"nodeWeight": m * share / 3.0, "frictionCoef": if k == 0 { 0.0 } else { 0.5 }});
                nodes.push(node(&format!("{tag}{i}_{k}"), [x, y, z], props));
            }
        }
    };
    frame("m", y_mid, m_mid, &mut nodes);
    frame("r", y_rear, m_rear, &mut nodes);

    let mut beams = vec![json!(["id1:", "id2:"])];
    // Carriage: a stiff steel frame. Its beams: the back plate's grid, every plate
    // node to the two mid-frame nodes of its third of the width, and every pair of
    // frame nodes. Each beam as stiff as the 2 kHz step allows its two nodes, with
    // margin: k = 0.25·min(m/n)/h² over its ends (m a node's mass, n its carriage
    // beams), so no node's carriage beams pass ω·h ≈ 0.7 together.
    let mut pairs: Vec<(String, String)> = Vec::new();
    for i in 0..COLS {
        for j in 0..ROWS {
            if i + 1 < COLS {
                pairs.push((name("p", i, j), name("p", i + 1, j)));
            }
            if j + 1 < ROWS {
                pairs.push((name("p", i, j), name("p", i, j + 1)));
            }
            if i + 1 < COLS && j + 1 < ROWS {
                pairs.push((name("p", i, j), name("p", i + 1, j + 1)));
                pairs.push((name("p", i + 1, j), name("p", i, j + 1)));
            }
            let fi = if i < COLS / 3 { 0 } else if i < 2 * COLS / 3 { 1 } else { 2 };
            pairs.push((name("p", i, j), format!("m{fi}_0")));
            pairs.push((name("p", i, j), format!("m{fi}_1")));
        }
    }
    let frame_ids: Vec<String> = ["m", "r"]
        .iter()
        .flat_map(|t| (0..3).flat_map(move |i| (0..2).map(move |k| format!("{t}{i}_{k}"))))
        .collect();
    for a in 0..frame_ids.len() {
        for b in a + 1..frame_ids.len() {
            pairs.push((frame_ids[a].clone(), frame_ids[b].clone()));
        }
    }
    let mass_of = |id: &str| -> f32 {
        let k = id[1..].split('_').nth(1).and_then(|k| k.parse::<usize>().ok()).unwrap_or(0);
        let share = if k == 0 { 1.0 - f } else { f };
        match &id[..1] {
            "p" => m_back / (COLS * ROWS) as f32,
            "m" => m_mid * share / 3.0,
            _ => m_rear * share / 3.0,
        }
    };
    let mut degree: std::collections::HashMap<&str, f32> = Default::default();
    for (a, b) in &pairs {
        *degree.entry(a.as_str()).or_default() += 1.0;
        *degree.entry(b.as_str()).or_default() += 1.0;
    }
    let h2 = crate::explicit_beams::STEP * crate::explicit_beams::STEP;
    beams.push(json!({"beamType": "|NORMAL", "beamDeform": "FLT_MAX", "beamStrength": "FLT_MAX", "beamPrecompression": 1}));
    for (a, b) in &pairs {
        let (ma, mb) = (mass_of(a), mass_of(b));
        let k = 0.25 * (ma / degree[a.as_str()]).min(mb / degree[b.as_str()]) / h2;
        let mu = ma * mb / (ma + mb);
        beams.push(json!({"beamSpring": k, "beamDamp": 0.1 * (k * mu).sqrt()}));
        beams.push(json!([a, b]));
    }
    // Cladding: the main face's sheet ties neighbouring face nodes, softly
    // (**Choice**: 0.8 mm aluminium carries little across a crushing block).
    beams.push(json!({"beamSpring": 200_000, "beamDamp": 20, "beamDeform": 4000, "beamStrength": "FLT_MAX"}));
    for i in 0..COLS {
        for j in 0..ROWS {
            if i + 1 < COLS {
                beams.push(json!([name("f", i, j), name("f", i + 1, j)]));
            }
            if j + 1 < ROWS {
                beams.push(json!([name("f", i, j), name("f", i, j + 1)]));
            }
            if j + 1 < BUMPER_ROWS {
                beams.push(json!([name("b", i, j), name("b", i, j + 1)]));
            }
            if i + 1 < COLS && j < BUMPER_ROWS {
                beams.push(json!([name("b", i, j), name("b", i + 1, j)]));
            }
        }
    }
    // Honeycomb columns: SUPPORT beams yielding at the crush force (no hardening),
    // densified at 80 %; a tension past 10 × their length (never) would break them.
    // Shear (**Choice**): honeycomb resists its cells tilting. Diagonals from each
    // face node to the neighbouring back-plate nodes (NORMAL beams yielding at a
    // tenth of the column's crush force, no hardening) hold the face square; the
    // column carries the rest, so a node's crush force is still σ·A (all yielding:
    // the column's plus the diagonals' along the barrier's axis).
    let diag_share = 0.1f32;
    for i in 0..COLS {
        for j in 0..ROWS {
            let a = trib(i, COLS, dx) * trib(j, ROWS, dz);
            let total = MDB_CRUSH[0] * a;
            let mut nb: Vec<(usize, usize, f32)> = Vec::new();
            for (di, dj, d) in [(1i32, 0i32, dx), (-1, 0, dx), (0, 1, dz), (0, -1, dz)] {
                let (ni, nj) = (i as i32 + di, j as i32 + dj);
                if ni >= 0 && nj >= 0 && (ni as usize) < COLS && (nj as usize) < ROWS {
                    nb.push((ni as usize, nj as usize, depth / (depth * depth + d * d).sqrt()));
                }
            }
            let fd = diag_share * total;
            let fm = total - nb.iter().map(|&(_, _, cos)| fd * cos).sum::<f32>();
            let km = fm / (ELASTIC_STRAIN[1] * depth);
            beams.push(json!({"beamType": "|NORMAL", "beamSpring": km, "beamDamp": 0.05 * (km * FACE_NODE_MASS).sqrt(),
                "beamDeform": fd, "deformLimitStress": fd, "deformLimit": DENSIFY, "beamStrength": "FLT_MAX", "beamLongBound": 10}));
            for &(ni, nj, _) in &nb {
                beams.push(json!([name("f", i, j), name("p", ni, nj)]));
            }
            beams.push(json!({"beamType": "|SUPPORT", "beamSpring": km, "beamDamp": 0.05 * (km * FACE_NODE_MASS).sqrt(),
                "beamDeform": fm, "deformLimitStress": fm, "deformLimit": DENSIFY, "beamStrength": "FLT_MAX", "beamLongBound": 10}));
            beams.push(json!([name("f", i, j), name("p", i, j)]));
            if j < BUMPER_ROWS {
                let ab = trib(i, COLS, dx) * bump_h(j);
                let fb = MDB_CRUSH[1] * ab;
                let kb = fb / (ELASTIC_STRAIN[0] * bump);
                beams.push(json!({"beamSpring": kb, "beamDamp": 0.05 * (kb * FACE_NODE_MASS).sqrt(),
                    "beamDeform": fb, "deformLimitStress": fb}));
                beams.push(json!([name("b", i, j), name("f", i, j)]));
            }
        }
    }
    // The densified block (**Choice**): past 80 % crush the honeycomb is solid
    // aluminium; a SUPPORT beam from each bumper node to its back-plate node, its
    // rest 20 % of the full depth, pushes from there as stiff as the step allows on
    // the face nodes. (The main block's own columns stop at their deform limit.)
    let k_dense = 0.25 * FACE_NODE_MASS / (crate::explicit_beams::STEP * crate::explicit_beams::STEP);
    beams.push(json!({"beamType": "|SUPPORT", "beamSpring": k_dense, "beamDamp": 0.1 * (k_dense * 0.5 * FACE_NODE_MASS).sqrt(),
        "beamDeform": "FLT_MAX", "deformLimitStress": "", "deformLimit": "", "beamStrength": "FLT_MAX", "beamLongBound": 10,
        "beamPrecompression": 1.0 - DENSIFY}));
    for i in 0..COLS {
        for j in 0..BUMPER_ROWS {
            beams.push(json!([name("b", i, j), name("p", i, j)]));
        }
    }
    beams.push(json!({"beamType": "|NORMAL"}));
    // Collision skin: the bumper's front and the main face above it.
    let mut tris = vec![json!(["id1:", "id2:", "id3:"])];
    let quad = |tris: &mut Vec<Value>, a: String, b: String, c: String, d: String| {
        tris.push(json!([a, b, c]));
        tris.push(json!([a, c, d]));
    };
    for i in 0..COLS - 1 {
        // The bumper's front over its rows, the main face from the bumper's top up.
        for j in 0..BUMPER_ROWS - 1 {
            quad(&mut tris, name("b", i, j), name("b", i, j + 1), name("b", i + 1, j + 1), name("b", i + 1, j));
        }
        for j in BUMPER_ROWS - 1..ROWS - 1 {
            quad(&mut tris, name("f", i, j), name("f", i, j + 1), name("f", i + 1, j + 1), name("f", i + 1, j));
        }
    }
    let jbeam = json!({
        "rubble_mdb214": {
            "information": {"name": "FMVSS 214 moving deformable barrier (Rubble model)", "authors": "Rubble"},
            "slotType": "main",
            "refNodes": [["ref:", "back:", "left:", "up:"], ["m1_0", "r1_0", "m2_0", "m1_1"]],
            "nodes": nodes,
            "beams": beams,
            "triangles": tris,
        }
    });
    serde_json::to_string_pretty(&jbeam).unwrap()
}

/// The MDB's crush force of a flat rigid face pressed into all of it (N): the main
/// block's crush stress on the whole face (the bumper, stronger, sits in series in
/// front of the main block's lowest third).
pub fn mdb214_face_crush_force() -> f32 {
    MDB_CRUSH[0] * MDB_FACE[0] * MDB_FACE[1]
}

/// A multi-node door pull: one rig node at `rig_at` (world-local, the vehicle's
/// frame), joined to each of `door` by a cable (a plain beam, 200 kN/m, unbreakable),
/// added to `sim` after spawn. Pull the rig node with the solver's grab
/// ([`crate::explicit_beams::Grab`]) to pull all the door's nodes at once. Returns
/// the rig node's index. The rig node does not touch the ground.
pub fn add_door_pull(sim: &mut ExplicitBeams, door: &[u32], rig_at: Vec3) -> u32 {
    let rig = sim.nodes.len() as u32;
    sim.nodes.push(Node {
        pos: rig_at,
        vel: Vec3::ZERO,
        force: Vec3::ZERO,
        mass: 5.0,
        friction: 0.0,
        flags: NODE_NO_GROUND,
    });
    if !sim.self_groups.is_empty() {
        sim.self_groups.push(0);
    }
    for &d in door {
        let rest = sim.nodes[d as usize].pos.distance(rig_at);
        let kind = if sim.beams.iter().any(|b| b.kind.is_jbeam()) {
            ExplicitKind::JbeamPlain
        } else {
            ExplicitKind::Plain
        };
        let b = Beam::new(d, rig, kind, ExplicitType::Deformable, 200_000.0, 50.0, rest, f32::INFINITY, f32::INFINITY, 0.0);
        sim.beams.push(b);
        sim.incidence[d as usize].push(sim.beams.len() as u32 - 1);
    }
    sim.incidence.push((sim.beams.len() - door.len()..sim.beams.len()).map(|j| j as u32).collect());
    if !sim.ext_of.is_empty() {
        sim.ext_of.resize(sim.beams.len(), 0);
    }
    sim.finish_contact();
    rig
}

/// The door nodes a pull takes: of the collision-skin nodes within 15 cm of the
/// side (`side`, a unit vector, the outermost skin node along it), the `count`
/// nearest to `at` (vehicle frame).
pub fn door_nodes(sim: &ExplicitBeams, side: Vec3, at: Vec3, count: usize) -> Vec<u32> {
    let mut skin: Vec<u32> = sim.triangles.iter().flat_map(|t| t.nodes).collect();
    skin.sort_unstable();
    skin.dedup();
    let reach = skin.iter().map(|&i| sim.nodes[i as usize].pos.dot(side)).fold(f32::MIN, f32::max);
    let mut near: Vec<u32> = skin
        .into_iter()
        .filter(|&i| sim.nodes[i as usize].pos.dot(side) > reach - 0.15)
        .collect();
    near.sort_by(|&a, &b| {
        let da = sim.nodes[a as usize].pos.distance_squared(at);
        let db = sim.nodes[b as usize].pos.distance_squared(at);
        da.total_cmp(&db)
    });
    near.truncate(count);
    near
}
