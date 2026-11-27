//! The lab's offset-wall crash on the CPU reference, measured a World frame at a time
//! (1/60 s): how deep the nodes and the drawn paint go into the wall, and how far the
//! page's drawing between World steps is from the state it stands for.
//!
//! ```bash
//! cargo run --release -p engine-softbody-lab --example wall_crash_probe [-- design]
//! ```
//! The car is the generated sedan (`rig::rig_vehicle`, as `vehicle_film`), surface contact
//! on, edge contact and continuous contact on (`PROBE_CONTINUOUS=0` turns those two off),
//! `PROBE_KMH` (56) into a rigid wall over `PROBE_OVERLAP` (0.4) of its width from the
//! left (+x), the lab's preset (`vehicle_lab::Lab::restart`, `"offset"`): its face 1 m
//! ahead of the nose. The wall is the solver's static box here (the World's is a fixed
//! body; the same face).
//!
//! Per frame: the deepest node and paint vertex behind the wall's face (true state), and
//! two ways to draw the frame halfway to the next World step (the page in slow motion
//! draws ~16 frames a step at ×1/16), each against the true state there:
//! - **extrapolated** (the page's `lead`): each node at `p + v · lead`;
//! - **interpolated**: each node halfway between the two World steps' states.
//!
//! Then the crash's energy: kinetic before, after, the beams' stored energy, what the
//! contact response removed, and the rebound (restitution).
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{ExplicitBeams, StaticBox};
use engine_softbody::explicit_contact::{elastic_energy, ExplicitBeamsWorld};
use engine_softbody_authoring::authoring::rig;
use engine_softbody_authoring::authoring::skin::{deform, ShowcaseVisual};
use engine_softbody_authoring::vehicle_compare::settings;
use glam::Vec3;

fn env(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> Result<(), String> {
    let design = std::env::args().nth(1).unwrap_or_else(|| "sedan".into());
    let built = rig::rig_vehicle(&design, None, None)?;
    let files = rig::mod_files(&built);
    let refs: Vec<(String, String)> = files
        .iter()
        .filter(|(p, _)| p.ends_with(".jbeam"))
        .map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned()))
        .collect();
    let pc = files
        .iter()
        .find(|(p, _)| p.ends_with(".pc"))
        .map(|(_, b)| String::from_utf8_lossy(b).into_owned());
    let r: Vec<(&str, &str)> = refs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let desc: SoftVehicleDesc = engine_softbody::jbeam::load_jbeam(&r, pc.as_deref())?;

    // The skinned paint, bound on a spawn at the desc's positions (vehicle_film's).
    let (probe, _) = ExplicitBeams::from_desc_jbeam_raycast(&desc, Vec3::ZERO, Vec3::ZERO, settings())?;
    let flex = engine_softbody::jbeam::assembled_rows(&r, pc.as_deref(), "flexbodies")?;
    let g = &built.input.glb;
    let meshes: Vec<_> = g.meshes.iter().map(|m| engine_softbody_authoring::flexskin::glb_mesh(g, m)).collect();
    let paint = engine_softbody_authoring::glb::paint_of(g).unwrap_or([0.5, 0.05, 0.04]);
    let (vis, _): (ShowcaseVisual, _) = engine_softbody_authoring::flexmesh::jbeam_visual_skinned(
        &flex, &[], &desc, &probe, &meshes, &Default::default(), paint, Some(&built.skin),
    )?;

    let kmh = env("PROBE_KMH", 56.0);
    let overlap = env("PROBE_OVERLAP", 0.4);
    let continuous = std::env::var("PROBE_CONTINUOUS").map_or(true, |v| v != "0");
    let low = desc.nodes.iter().map(|n| n.position[1]).fold(f32::MAX, f32::min);
    let low = desc.wheels.iter().fold(low, |m, w| {
        let y = |i: u32| desc.nodes[i as usize].position[1];
        m.min(0.5 * (y(w.node1) + y(w.node2)) - w.radius)
    });
    let (mut s, _) = ExplicitBeams::from_desc_jbeam_raycast(
        &desc,
        Vec3::Y * (0.02 - low),
        Vec3::Z * (kmh / 3.6),
        settings(),
    )?;
    s.surface_contact = true;
    s.edge_contact = continuous;
    s.finish();
    let (mut lo, mut hi) = (Vec3::MAX, Vec3::MIN);
    for n in &s.nodes {
        lo = lo.min(n.pos + s.origin);
        hi = hi.max(n.pos + s.origin);
    }
    let wide = hi.x - lo.x;
    let (x0, x1) = (hi.x - overlap * wide, hi.x + 4.0);
    let face = hi.z + 1.0;
    let half = Vec3::new(0.5 * (x1 - x0), 1.25, 0.5);
    s.boxes.push(StaticBox::centered(Vec3::new(0.5 * (x0 + x1), half.y, face + half.z), half));
    let mut w = ExplicitBeamsWorld::new(vec![s]);
    if continuous {
        w.continuous = Some(engine_softbody::explicit_contact::Dat::new(0.08, 0.85));
    }
    // Depth behind the wall's face of a point in its footprint (0 outside it).
    let depth = |p: Vec3| -> f32 {
        if p.x > x0 && p.x < x1 && p.y < 2.0 * half.y && p.z > face && p.z < face + 2.0 * half.z {
            p.z - face
        } else {
            0.0
        }
    };
    let mass: f32 = w.vehicles[0].nodes.iter().map(|n| n.mass).sum();
    let com_v = |s: &ExplicitBeams| s.nodes.iter().map(|n| n.vel * n.mass).sum::<Vec3>() / mass;
    let ke = |s: &ExplicitBeams| s.nodes.iter().map(|n| 0.5 * n.mass as f64 * n.vel.length_squared() as f64).sum::<f64>();
    // World positions (the origin rebases during the run; skinning commutes with a translation).
    let pos = |s: &ExplicitBeams| s.nodes.iter().map(|n| n.pos + s.origin).collect::<Vec<Vec3>>();
    let paint_at = |nodes: &[Vec3]| deform(&vis, nodes).0;
    let max_err = |a: &[Vec3], b: &[Vec3]| -> (f32, f32) {
        let mut d: Vec<f32> = a.iter().zip(b).map(|(x, y)| x.distance(*y)).collect();
        d.sort_by(f32::total_cmp);
        (d[d.len() - 1], d[d.len() * 99 / 100])
    };
    // How far each paint vertex is from its nearest node (the overlay's "paint far from its
    // nodes"), at rest and at the end.
    let nearest = |nodes: &[Vec3]| -> Vec<f32> {
        paint_at(nodes).iter().map(|q| nodes.iter().map(|n| n.distance_squared(*q)).fold(f32::MAX, f32::min).sqrt()).collect()
    };
    let near0 = nearest(&pos(&w.vehicles[0]));
    let frames = env("PROBE_FRAMES", 60.0) as usize;
    let (per, half_per) = (33usize, 16usize);
    let ke0 = ke(&w.vehicles[0]);
    let u0 = elastic_energy(&w.vehicles[0]);
    let v0 = com_v(&w.vehicles[0]).z;
    println!(
        "{design}: {} nodes, {:.0} kg, {kmh} km/h, overlap {overlap}, continuous {continuous}; {} paint vertices",
        w.vehicles[0].nodes.len(), mass, vis.positions.len()
    );
    println!("frame   t ms  com vz | node in  paint in | extrapolated mid-frame: node max/p99, paint max/p99, paint in | interpolated: node max/p99, paint max/p99, paint in  (mm)");
    let (mut worst_x, mut worst_i, mut worst_xin, mut worst_iin, mut worst_node, mut worst_paint) = (0f32, 0f32, 0f32, 0f32, 0f32, 0f32);
    for k in 0..frames {
        let s = &w.vehicles[0];
        let (p0, vel0): (Vec<Vec3>, Vec<Vec3>) = (pos(s), s.nodes.iter().map(|n| n.vel).collect());
        let node_in = p0.iter().map(|p| depth(*p)).fold(0.0, f32::max);
        let paint0 = paint_at(&p0);
        let paint_in = paint0.iter().map(|p| depth(*p)).fold(0.0, f32::max);
        let vz = com_v(s).z;
        for _ in 0..half_per {
            w.step();
        }
        let pm = pos(&w.vehicles[0]);
        for _ in half_per..per {
            w.step();
        }
        let p1 = pos(&w.vehicles[0]);
        let lead = half_per as f32 * 0.0005;
        let px: Vec<Vec3> = p0.iter().zip(&vel0).map(|(p, v)| *p + *v * lead).collect();
        let a = half_per as f32 / per as f32;
        let pi: Vec<Vec3> = p0.iter().zip(&p1).map(|(p, q)| p.lerp(*q, a)).collect();
        let paint_m = paint_at(&pm);
        let (paint_x, paint_i) = (paint_at(&px), paint_at(&pi));
        let (nx, nx99) = max_err(&px, &pm);
        let (ni, ni99) = max_err(&pi, &pm);
        let (cx, cx99) = max_err(&paint_x, &paint_m);
        let (ci, ci99) = max_err(&paint_i, &paint_m);
        let xin = paint_x.iter().map(|p| depth(*p)).fold(0.0, f32::max);
        let iin = paint_i.iter().map(|p| depth(*p)).fold(0.0, f32::max);
        worst_x = worst_x.max(cx);
        worst_i = worst_i.max(ci);
        worst_xin = worst_xin.max(xin);
        worst_iin = worst_iin.max(iin);
        worst_node = worst_node.max(node_in);
        worst_paint = worst_paint.max(paint_in);
        let mm = |x: f32| x * 1e3;
        println!(
            "{k:>5} {:>6.1} {vz:>7.2} | {:>7.1} {:>8.1} | {:>6.1}/{:<6.1} {:>6.1}/{:<6.1} {:>7.1} | {:>6.1}/{:<6.1} {:>6.1}/{:<6.1} {:>7.1}",
            k as f32 * per as f32 * 0.5,
            mm(node_in), mm(paint_in), mm(nx), mm(nx99), mm(cx), mm(cx99), mm(xin), mm(ni), mm(ni99), mm(ci), mm(ci99), mm(iin)
        );
    }
    let near1 = nearest(&pos(&w.vehicles[0]));
    let mut grow: Vec<f32> = near0.iter().zip(&near1).map(|(a, b)| b - a).collect();
    grow.sort_by(f32::total_cmp);
    let mut n0 = near0.clone();
    n0.sort_by(f32::total_cmp);
    let mut n1 = near1.clone();
    n1.sort_by(f32::total_cmp);
    let q = |v: &[f32], f: f32| v[((v.len() - 1) as f32 * f) as usize] * 1e3;
    println!(
        "PAINT to nearest node (mm): at rest p50 {:.0} p99 {:.0} max {:.0}; at the end p50 {:.0} p99 {:.0} max {:.0}; growth p99 {:.0} max {:.0}",
        q(&n0, 0.5), q(&n0, 0.99), q(&n0, 1.0), q(&n1, 0.5), q(&n1, 0.99), q(&n1, 1.0), q(&grow, 0.99), q(&grow, 1.0)
    );
    let s = &w.vehicles[0];
    let (ke1, u1) = (ke(s), elastic_energy(s));
    let v1 = com_v(s).z;
    let removed = w.continuous.as_ref().map_or(0.0, |d| -d.total.ccd.ke);
    let broken = s.beams.iter().filter(|b| b.broken).count();
    let bent = s.beams.iter().filter(|b| !b.broken && (b.rest - b.rest0).abs() > 1e-3).count();
    println!(
        "WORST true: node {:.1} mm, paint {:.1} mm into the wall; mid-frame drawing error (paint max): extrapolated {:.1} mm (paint {:.1} mm in), interpolated {:.1} mm (paint {:.1} mm in)",
        worst_node * 1e3, worst_paint * 1e3, worst_x * 1e3, worst_xin * 1e3, worst_i * 1e3, worst_iin * 1e3
    );
    println!(
        "ENERGY kinetic {:.1} -> {:.1} kJ, stored in beams {:.1} -> {:.1} kJ, contact response removed {:.2} kJ ({:.2} % of the kinetic before); com vz {v0:.2} -> {v1:.2} m/s (restitution {:.3}); beams broken {broken}, bent {bent}",
        ke0 * 1e-3, ke1 * 1e-3, u0 * 1e-3, u1 * 1e-3, removed * 1e-3, 100.0 * removed / ke0, -v1 / v0
    );
    Ok(())
}
