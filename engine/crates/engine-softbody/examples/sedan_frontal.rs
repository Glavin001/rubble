//! Where a frontal crash's load goes in a rigged design (CPU reference, no
//! skin binding: about a second to rig, a few to run).
//!
//! ```bash
//! cargo run --release -p engine-softbody --example sedan_frontal -- <scene> [km/h] [design]
//! ```
//! Scenes (placed as `vehicle_film` places them): `wall` (full width, 56),
//! `offset` (a rigid barrier over 40 % of the width on +x, 64), `pole` (a
//! 254 mm pole on the centre line, 50).
//!
//! Prints the pulse (dynamic crush of the cabin's rear floor, peak g at
//! 100 Hz, the barrier's force), the intrusion of named shell nodes into the
//! cabin (each node's distance ahead of the rear floor, change from rest),
//! and the beams grouped by the members they join: yield force offered,
//! when each group first yields, how far it shortens and the plastic work it
//! does. `FR_MS` sets the simulated time after contact (default 160),
//! `FR_SURFACE=0` runs node contact only, `FR_TOP` the rows printed.
use engine_softbody::authoring::gen2::{rig as lattice, vg_input};
use engine_softbody::authoring::rig;
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{ExplicitBeams, StaticBox, StaticShape};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::vehicle_compare::{rigid_fit, settings};
use glam::Vec3;
use std::collections::BTreeMap;

const STEP: f32 = 0.0005;

/// Zero-phase second-order low pass (Butterworth, forward and backward).
fn low_pass(x: &[f32], dt: f32, hz: f32) -> Vec<f32> {
    let w = (std::f32::consts::PI * hz * dt).tan();
    let n = 1.0 / (1.0 + std::f32::consts::SQRT_2 * w + w * w);
    let (b0, b1, b2) = (w * w * n, 2.0 * w * w * n, w * w * n);
    let (a1, a2) = (
        2.0 * (w * w - 1.0) * n,
        (1.0 - std::f32::consts::SQRT_2 * w + w * w) * n,
    );
    let pass = |x: &[f32]| {
        let mut y = vec![0.0f32; x.len()];
        let (mut x1, mut x2, mut y1, mut y2) = (x[0], x[0], x[0], x[0]);
        for (i, &v) in x.iter().enumerate() {
            let o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            (x2, x1, y2, y1) = (x1, v, y1, o);
            y[i] = o;
        }
        y
    };
    if x.is_empty() {
        return Vec::new();
    }
    let mut y = pass(x);
    y.reverse();
    let mut y = pass(&y);
    y.reverse();
    y
}

/// A node's label: a shell node's group and side (`A4+`), else its part.
fn label(d: &SoftVehicleDesc, part_of: &[usize], i: usize) -> String {
    let Some(n) = d.nodes.get(i) else {
        return "solver".into();
    };
    let side = match n.position[0] {
        x if x > 0.05 => "+",
        x if x < -0.05 => "-",
        _ => "0",
    };
    if let Some(rest) = n.name.strip_prefix("sh_") {
        let g = rest.split('_').next().unwrap_or("");
        return format!("{}{side}", g.to_uppercase());
    }
    let p = d
        .parts
        .get(part_of[i])
        .map(|p| p.name.as_str())
        .unwrap_or("?");
    let p = p.trim_start_matches("sedan_");
    // (a part on the centre line that spans both sides keeps no side)
    if p.ends_with("_l") || p.ends_with("_r") || p.len() > 3 && p[p.len() - 3..].starts_with('_') {
        p.to_string()
    } else {
        format!("{p}{side}")
    }
}

#[derive(Default, Clone)]
struct Group {
    beams: usize,
    offered: f64,
    yielded: usize,
    broken: usize,
    first_ms: Option<f32>,
    shorten_mm: f64,
    work_j: f64,
    long: usize,
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let scene = a.first().cloned().unwrap_or_else(|| "offset".into());
    let kmh: f32 = a
        .get(1)
        .and_then(|x| x.parse().ok())
        .unwrap_or(match scene.as_str() {
            "wall" => 56.0,
            "pole" => 50.0,
            _ => 64.0,
        });
    let design = a.get(2).cloned().unwrap_or_else(|| "sedan".into());
    let env = |k: &str, d: f32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (ms, top) = (env("FR_MS", 160.0), env("FR_TOP", 34.0) as usize);
    let t0 = std::time::Instant::now();
    let inp = rig::DesignInput::find(&design, None, None)?;
    let r = lattice::rig_with(&inp);
    let files = vg_input::physics_files(&inp, &r);
    let texts: Vec<(String, String)> = files
        .iter()
        .map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned()))
        .collect();
    let refs: Vec<(&str, &str)> = texts
        .iter()
        .filter(|(p, _)| p.ends_with(".jbeam"))
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let pc = texts
        .iter()
        .find(|(p, _)| p.ends_with(".pc"))
        .map(|(_, b)| b.as_str());
    let d = engine_softbody::jbeam::load_jbeam(&refs, pc)?;
    let mut part_of = vec![usize::MAX; d.nodes.len()];
    for (k, p) in d.parts.iter().enumerate() {
        for &i in &p.nodes {
            part_of[i as usize] = k;
        }
    }
    // ---- the scene, placed as `vehicle_film` places it
    let low = d
        .nodes
        .iter()
        .map(|n| n.position[1])
        .fold(f32::MAX, f32::min);
    let low = d.wheels.iter().fold(low, |m, w| {
        let y = |i: u32| d.nodes[i as usize].position[1];
        m.min(0.5 * (y(w.node1) + y(w.node2)) - w.radius)
    });
    let v0 = kmh / 3.6;
    let (mut s, _) = ExplicitBeams::from_desc_jbeam_raycast(
        &d,
        Vec3::Y * (0.02 - low),
        Vec3::Z * v0,
        settings(),
    )?;
    s.surface_contact = std::env::var("FR_SURFACE").map_or(true, |v| v != "0");
    s.finish();
    {
        // What the page's beam count is: the solver's records, by kind.
        let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
        for b in &s.beams {
            *kinds.entry(format!("{:?}", b.kind)).or_default() += 1;
        }
        println!(
            "JBeam beams {} (+ {} torsion bars, {} couplers); solver beam records {}: {kinds:?}",
            d.beams.len(),
            d.elements.len(),
            d.couplers.len(),
            s.beams.len()
        );
    }
    let zmax = d
        .nodes
        .iter()
        .map(|n| n.position[2])
        .fold(f32::MIN, f32::max);
    let half_w = d
        .nodes
        .iter()
        .map(|n| n.position[0].abs())
        .fold(0.0f32, f32::max);
    let wall_z = zmax + 0.05;
    match scene.as_str() {
        "wall" => s.boxes.push(StaticBox::centered(
            Vec3::new(0.0, 1.0, wall_z + 0.5),
            Vec3::new(2.0, 1.0, 0.5),
        )),
        "offset" => {
            let edge = half_w - 0.4 * 2.0 * half_w;
            s.boxes.push(StaticBox::centered(
                Vec3::new(edge + 1.0, 1.0, wall_z + 0.5),
                Vec3::new(1.0, 1.0, 0.5),
            ));
        }
        "pole" => s.shapes.push(StaticShape::cylinder(
            Vec3::new(0.0, 1.0, wall_z + 0.127),
            0.127,
            2.0,
        )),
        _ => return Err(format!("unknown scene {scene}")),
    }
    let name_of: BTreeMap<&str, usize> = d
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.name.as_str(), i))
        .collect();
    let cabin: Vec<usize> = d
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| {
            [
                "sh_c1_4", "sh_c1_5", "sh_c2_4", "sh_c2_5", "sh_c3_4", "sh_c3_5",
            ]
            .iter()
            .any(|p| n.name.starts_with(p))
        })
        .map(|(i, _)| i)
        .collect();
    if cabin.is_empty() {
        return Err("no rear floor nodes (sh_c1_4…): not a car on the body definitions".into());
    }
    let total: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let cm: f32 = cabin.iter().map(|&i| s.nodes[i].mass).sum();
    let cab = |s: &ExplicitBeams| {
        let (mut p, mut v) = (Vec3::ZERO, Vec3::ZERO);
        for &i in &cabin {
            p += s.nodes[i].pos * s.nodes[i].mass;
            v += s.nodes[i].vel * s.nodes[i].mass;
        }
        (p / cm, v / cm)
    };
    let pz = |s: &ExplicitBeams| {
        s.nodes
            .iter()
            .map(|n| f64::from(n.mass) * f64::from(n.vel.z))
            .sum::<f64>()
    };
    // ---- the nodes whose place ahead of the rear floor is followed
    let mut watch: Vec<(String, usize)> = Vec::new();
    for side in ["l", "r"] {
        for n in [
            "a1_d", "a2_d", "a4_", "a5_", "a6_", "a8_m", "b1_o", "b2_o", "c1_1o", "c1_1t", "c2_1",
            "d1_m", "c2_2",
        ] {
            let name = format!("sh_{n}{side}");
            if let Some(&i) = name_of.get(name.as_str()) {
                watch.push((format!("{n}{side}"), i));
            }
        }
        for n in [format!("whef{side}_ai"), format!("hubf{side}_lj")] {
            if let Some(&i) = name_of.get(n.as_str()) {
                watch.push((n, i));
            }
        }
    }
    // Every node's place in the rear floor's own frame (a rigid fit of the
    // rear floor's nodes to their rest places: an offset hit turns the car,
    // and a turn is not intrusion).
    let nd = d.nodes.len();
    let rest: Vec<Vec3> = s.nodes.iter().map(|n| n.pos + s.origin).collect();
    let cabin_u: Vec<u32> = cabin.iter().map(|&i| i as u32).collect();
    let local = |s: &ExplicitBeams| -> Vec<Vec3> {
        let now: Vec<Vec3> = s.nodes.iter().map(|n| n.pos + s.origin).collect();
        let fit = rigid_fit(&rest, &now, &cabin_u);
        now.iter()
            .take(nd)
            .map(|&p| {
                if p.is_finite() {
                    fit.to_rest(p)
                } else {
                    Vec3::NAN
                }
            })
            .collect()
    };
    // ---- beams by the members they join
    let nb = s.beams.len();
    let labels: Vec<String> = (0..nb)
        .map(|j| {
            let b = &s.beams[j];
            let (mut x, mut y) = (
                label(&d, &part_of, b.a as usize),
                label(&d, &part_of, b.b as usize),
            );
            if x > y {
                std::mem::swap(&mut x, &mut y);
            }
            if x == y {
                x
            } else {
                format!("{x} ~ {y}")
            }
        })
        .collect();
    let longit: Vec<bool> = (0..nb)
        .map(|j| {
            let b = &s.beams[j];
            let e = s.nodes[b.b as usize].pos - s.nodes[b.a as usize].pos;
            e.normalize_or_zero().z.abs() > 0.707
        })
        .collect();
    let mut first: Vec<Option<f32>> = vec![None; nb];
    let mut work = vec![0.0f64; nb];
    let mut rest_prev: Vec<f32> = s.beams.iter().map(|b| b.rest).collect();
    let mut peak_f = vec![0.0f32; nb];
    // every node: the furthest it was driven back, the rearmost place it reached
    let cab_z0 = cabin.iter().map(|&i| rest[i].z).sum::<f32>() / cabin.len() as f32;
    let ahead0: Vec<f32> = (0..nd).map(|i| rest[i].z - cab_z0).collect();
    let mut back_max = vec![0.0f32; nd];
    let mut rearmost = ahead0.clone();
    let node_label: Vec<String> = (0..nd).map(|i| label(&d, &part_of, i)).collect();
    let mut w = ExplicitBeamsWorld::new(vec![s]);
    w.contact = false;
    let (mut contact, mut z_contact) = (None::<usize>, 0.0f32);
    let (mut acc, mut force) = (Vec::new(), Vec::new());
    let (mut v_prev, mut p_prev) = (cab(&w.vehicles[0]).1.z, pz(&w.vehicles[0]));
    let (mut d_dyn, mut t_m) = (0.0f32, None::<f32>);
    let mut rows: Vec<(f32, f32, f32, Vec<f32>, Vec<f32>)> = Vec::new();
    let mut intr_max = vec![0.0f32; watch.len()];
    let mut last_y: Vec<f32> = Vec::new();
    let after = (ms * 1e-3 / STEP) as usize;
    let mut k = 0usize;
    let mut steps_after = 0usize;
    while steps_after < after && k < 4000 {
        w.step();
        k += 1;
        let s = &w.vehicles[0];
        if contact.is_none() {
            let reach = (0..d.nodes.len())
                .map(|i| s.nodes[i].pos.z + s.origin.z)
                .fold(f32::MIN, f32::max);
            if reach >= wall_z - 0.03 {
                contact = Some(k);
                z_contact = cab(s).0.z;
            }
            v_prev = cab(s).1.z;
            p_prev = pz(s);
            rest_prev = s.beams.iter().map(|b| b.rest).collect();
            continue;
        }
        steps_after += 1;
        let t = steps_after as f32 * STEP * 1e3;
        let (cp, cv) = cab(s);
        acc.push((cv.z - v_prev) / STEP);
        v_prev = cv.z;
        let p = pz(s);
        force.push(((p_prev - p) / f64::from(STEP)) as f32);
        p_prev = p;
        if t_m.is_none() {
            d_dyn = d_dyn.max(cp.z - z_contact);
            if cv.z <= 0.0 {
                t_m = Some(t);
            }
        }
        for j in 0..nb.min(s.beams.len()) {
            let b = &s.beams[j];
            let dr = b.rest - rest_prev[j];
            if dr != 0.0 {
                first[j].get_or_insert(t);
                work[j] += f64::from(dr.abs()) * f64::from(b.force.abs());
                rest_prev[j] = b.rest;
            }
            peak_f[j] = peak_f[j].max(b.force.abs());
        }
        if steps_after % 2 == 0 {
            let l = local(s);
            for i in 0..nd {
                let a = l[i].z - cab_z0;
                if a.is_finite() {
                    back_max[i] = back_max[i].max(ahead0[i] - a);
                    rearmost[i] = rearmost[i].min(a);
                }
            }
            for (x, wn) in watch.iter().enumerate() {
                intr_max[x] = back_max[wn.1];
            }
            if steps_after % 40 == 0 {
                rows.push((
                    t,
                    cv.z,
                    cp.z - z_contact,
                    watch.iter().map(|wn| l[wn.1].z - rest[wn.1].z).collect(),
                    watch.iter().map(|wn| l[wn.1].x - rest[wn.1].x).collect(),
                ));
                last_y = watch.iter().map(|wn| l[wn.1].y - rest[wn.1].y).collect();
            }
        }
    }
    let s = &w.vehicles[0];
    let a100 = low_pass(&acc, STEP, 100.0);
    let (ia, peak_g) =
        a100.iter()
            .enumerate()
            .fold((0, 0.0f32), |b, (i, &x)| if -x > b.1 { (i, -x) } else { b });
    let f100 = low_pass(&force, STEP, 100.0);
    let (ifo, peak_force) =
        f100.iter()
            .enumerate()
            .fold((0, 0.0f32), |b, (i, &x)| if x > b.1 { (i, x) } else { b });
    let ke0 = 0.5 * total * v0 * v0;
    println!(
        "{design} {scene} {kmh} km/h: {:.0} kg, {:.1} kJ; rigged and run in {:.1} s (surface contact {})",
        total,
        ke0 / 1e3,
        t0.elapsed().as_secs_f32(),
        s.surface_contact
    );
    println!(
        "dynamic crush {:.0} mm at {} ms; peak {:.1} g at {:.0} ms (rear floor, 100 Hz); barrier force peak {:.0} kN at {:.0} ms; speed at {ms:.0} ms {:.2} m/s (lateral {:.2})",
        d_dyn * 1e3,
        t_m.map_or("– (not stopped)".into(), |t| format!("{t:.0}")),
        peak_g / 9.81,
        ia as f32 * STEP * 1e3,
        peak_force / 1e3,
        ifo as f32 * STEP * 1e3,
        cab(s).1.z,
        cab(s).1.x
    );
    let broken = s.beams.iter().filter(|b| b.broken).count();
    let yielded = first.iter().filter(|f| f.is_some()).count();
    println!(
        "beams: {} yielded, {broken} broken; plastic work {:.1} kJ of {:.1}",
        yielded,
        work.iter().sum::<f64>() / 1e3,
        ke0 / 1e3
    );
    println!(
        "barrier force (kN, 100 Hz) every 10 ms: {}",
        f100.iter()
            .step_by(20)
            .map(|x| format!("{:.0}", x / 1e3))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "cabin deceleration (g, 100 Hz) every 10 ms: {}",
        a100.iter()
            .step_by(20)
            .map(|x| format!("{:.0}", -x / 9.81))
            .collect::<Vec<_>>()
            .join(" ")
    );
    // ---- intrusion
    println!("\nnode in the rear floor's frame, change from rest along the car (mm; negative: driven back). Columns: t ms, cabin speed m/s, cabin travel mm, then");
    println!(
        "  {}",
        watch
            .iter()
            .map(|w| w.0.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    );
    for (t, v, dz, row, _) in &rows {
        println!(
            "  {t:>4.0} {v:>5.1} {:>5.0} | {}",
            dz * 1e3,
            row.iter()
                .map(|x| format!("{:>5.0}", x * 1e3))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    println!(
        "  max driven back (mm): {}",
        watch
            .iter()
            .zip(&intr_max)
            .map(|(w, m)| format!("{} {:.0}", w.0, m * 1e3))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if let Some((_, _, _, _, xs)) = rows.last() {
        println!(
            "  sideways at the end, in that frame (mm, + is the offset barrier's side): {}",
            watch
                .iter()
                .zip(xs)
                .map(|(w, m)| format!("{} {:.0}", w.0, m * 1e3))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!(
        "  height at the end (mm, + up): {}",
        watch
            .iter()
            .zip(&last_y)
            .map(|(w, y)| format!("{} {:.0}", w.0, y * 1e3))
            .collect::<Vec<_>>()
            .join(", ")
    );
    // ---- how far each part's nodes were driven back, and how far behind
    // the cowl (the A-pillars' bases at rest, the windscreen's foot) they got
    let cowl = ["sh_b2_ol", "sh_b2_or"]
        .iter()
        .filter_map(|n| name_of.get(n))
        .map(|&i| ahead0[i])
        .fold(f32::MAX, f32::min);
    let mut parts: BTreeMap<&str, (usize, f32, f32, f32, f32)> = BTreeMap::new();
    for i in 0..nd {
        let e =
            parts
                .entry(node_label[i].as_str())
                .or_insert((0, 0.0, f32::MAX, f32::MIN, f32::MAX));
        e.0 += 1;
        e.1 = e.1.max(back_max[i]);
        e.2 = e.2.min(rearmost[i] - cowl);
        e.3 = e.3.max(ahead0[i] - cowl);
        e.4 = e.4.min(ahead0[i] - cowl);
    }
    let mut pv: Vec<(&&str, &(usize, f32, f32, f32, f32))> = parts
        .iter()
        .filter(|p| p.1 .3 > 0.0 && p.1 .1 > 0.05)
        .collect();
    pv.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
    println!("\nparts ahead of the cowl at rest: nodes, furthest driven back (mm), rearmost place reached against the cowl line at rest (mm; negative: behind it, in the cabin's length), its rearmost node's place at rest");
    for (name, (n, back, rear, _, rest)) in pv.iter().take(top) {
        println!(
            "  {name:<22} {n:>3} {:>6.0} {:>6.0} {:>6.0}",
            back * 1e3,
            rear * 1e3,
            rest * 1e3
        );
    }
    // ---- the nodes behind the cowl that moved most along the car (either way)
    let mut inside: Vec<usize> = (0..nd).filter(|&i| ahead0[i] < cowl).collect();
    let fin = local(s);
    let moved = |i: usize| (fin[i].z - rest[i].z).abs().max(back_max[i]);
    inside.sort_by(|&a, &b| moved(b).total_cmp(&moved(a)));
    println!(
        "\nnodes behind the cowl, most moved along the car (name, part, peak driven back mm, end shift mm): {}",
        inside.iter().take(14).map(|&i| format!("{} {} {:.0} {:+.0}", d.nodes[i].name, node_label[i], back_max[i] * 1e3, (fin[i].z - rest[i].z) * 1e3)).collect::<Vec<_>>().join("; ")
    );
    // ---- groups
    let mut groups: BTreeMap<String, Group> = BTreeMap::new();
    for j in 0..nb.min(s.beams.len()) {
        let b = &s.beams[j];
        let g = groups.entry(labels[j].clone()).or_default();
        g.beams += 1;
        if longit[j] {
            g.long += 1;
            if b.tc.is_finite() && b.tc < 1e9 {
                g.offered += f64::from(b.tc);
            }
        }
        if let Some(t) = first[j] {
            g.yielded += 1;
            g.first_ms = Some(g.first_ms.map_or(t, |x: f32| x.min(t)));
        }
        g.broken += usize::from(b.broken);
        g.shorten_mm += f64::from(b.rest0 - b.rest) * 1e3;
        g.work_j += work[j];
    }
    // ---- the plastic work by class of part (a beam between two classes: half each)
    let class = |l: &str| -> &'static str {
        let l = l.trim_end_matches(['+', '-', '0']);
        let shell = l.len() == 2 && l.as_bytes()[1].is_ascii_digit();
        match l {
            _ if shell && l.starts_with('A') => {
                "front structure (crash boxes, rails, towers, upper rails, radiator support)"
            }
            _ if shell && l.starts_with('E') => "rear structure",
            _ if shell => "cabin (firewall, cowl, floor, sills, pillars, roof)",
            _ if l.starts_with("hood") || l.starts_with("fender") || l.starts_with("bumper") => {
                "hood, fenders, bumper covers"
            }
            _ if l.starts_with("door") || l.starts_with("trunk") || l.starts_with("body") => {
                "doors, lid, body hardware"
            }
            _ if l.starts_with("impact_beam") || l.starts_with("radiator") => {
                "impact beams, radiator"
            }
            _ if l.starts_with("engine")
                || l.starts_with("transmission")
                || l.starts_with("driveshaft")
                || l.starts_with("differential")
                || l.starts_with("exhaust")
                || l.starts_with("fuel") =>
            {
                "engine, driveline, exhaust, tank"
            }
            _ if l.starts_with("subframe")
                || l.starts_with("hub")
                || l.starts_with("wheel")
                || l.starts_with("steering")
                || l.starts_with("arm")
                || l.starts_with("strut") =>
            {
                "subframes, hubs, wheels, steering"
            }
            _ if l.starts_with("seat")
                || l.starts_with("dash")
                || l.starts_with("console")
                || l.starts_with("bench")
                || l.starts_with("glass") =>
            {
                "interior, glass"
            }
            _ => "other",
        }
    };
    let mut by_class: BTreeMap<&str, f64> = BTreeMap::new();
    for (name, g) in &groups {
        let mut it = name.split(" ~ ");
        let a = it.next().unwrap();
        let b = it.next().unwrap_or(a);
        *by_class.entry(class(a)).or_default() += 0.5 * g.work_j;
        *by_class.entry(class(b)).or_default() += 0.5 * g.work_j;
    }
    let all: f64 = by_class.values().sum();
    let mut cv: Vec<(&&str, &f64)> = by_class.iter().collect();
    cv.sort_by(|a, b| b.1.total_cmp(a.1));
    println!("\nplastic work by class (kJ, share):");
    for (c, w) in cv {
        println!(
            "  {:>6.1} {:>4.0} %  {c}",
            w / 1e3,
            100.0 * w / all.max(1.0)
        );
    }
    let mut v: Vec<(&String, &Group)> = groups.iter().collect();
    v.sort_by(|a, b| b.1.work_j.total_cmp(&a.1.work_j));
    println!("\nbeam groups by plastic work: beams (along the car), Σ yield of those along (kN), yielded, broken, first yield ms, Σ shortening mm, work kJ");
    for (name, g) in v.iter().take(top) {
        println!(
            "  {name:<34} {:>3} ({:>2}) {:>6.0} {:>3} {:>2} {:>5} {:>7.0} {:>6.2}",
            g.beams,
            g.long,
            g.offered / 1e3,
            g.yielded,
            g.broken,
            g.first_ms.map_or("–".into(), |t| format!("{t:.0}")),
            g.shorten_mm,
            g.work_j / 1e3
        );
    }
    let mut v: Vec<(&String, &Group)> = groups.iter().filter(|g| g.1.first_ms.is_some()).collect();
    v.sort_by(|a, b| a.1.first_ms.unwrap().total_cmp(&b.1.first_ms.unwrap()));
    println!(
        "\nthe first groups to yield (ms): {}",
        v.iter()
            .take(top)
            .map(|(n, g)| format!("{n} {:.0}", g.first_ms.unwrap()))
            .collect::<Vec<_>>()
            .join("; ")
    );
    // ---- the single beams that did the most work, and those that broke
    let mut bj: Vec<usize> = (0..nb.min(s.beams.len())).collect();
    bj.sort_by(|&a, &b| work[b].total_cmp(&work[a]));
    let nm = |i: u32| {
        d.nodes
            .get(i as usize)
            .map_or("solver", |n| n.name.as_str())
    };
    println!("\nbeams by plastic work: nodes, yield kN, peak force kN, first yield ms, rest change mm, kJ");
    for &j in bj.iter().take(top) {
        let b = &s.beams[j];
        println!(
            "  {:<12} {:<12} {:>6.1} {:>6.1} {:>5} {:>6.0} {:>5.2}{}",
            nm(b.a),
            nm(b.b),
            b.tc / 1e3,
            peak_f[j] / 1e3,
            first[j].map_or("–".into(), |t| format!("{t:.0}")),
            (b.rest - b.rest0) * 1e3,
            work[j] / 1e3,
            if b.broken { " broken" } else { "" }
        );
    }
    let mut by_group: BTreeMap<String, usize> = BTreeMap::new();
    for j in 0..nb.min(s.beams.len()) {
        if s.beams[j].broken {
            *by_group.entry(labels[j].clone()).or_default() += 1;
        }
    }
    println!(
        "\nbroken beams by group: {}",
        by_group
            .iter()
            .map(|(n, c)| format!("{n} {c}"))
            .collect::<Vec<_>>()
            .join("; ")
    );
    Ok(())
}
