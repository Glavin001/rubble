//! The crash lab's `sidepole` scene on the CPU reference with the energy
//! ledger on (a measurement; no GPU). The scene is built as the page builds it
//! (`engine-world-wasm` `crash_scene`): the car on raycast wheels, facing +X
//! with its centre at the origin, carried along +Z at `--kmh` on a carrier
//! (`crash_rigs::on_carrier`) into a 20 cm square pole `--at` metres along its
//! side from its centre, after the page's run-up (1 + v/2 s metres).
//!
//! Per run it reports where the mechanical energy (kinetic + gravitational +
//! elastic) first rises, which source's work rose with it, and the nodes that
//! took the most work from the node pass's contacts (ground, pole, surface
//! contact, raycast wheels), from self-collision and from the beams in the 20
//! ms before.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example sidepole_sweep -- \
//!     --mod <mods dir>/<mod> --vehicle sedan [--pc sedan.pc] [--kmh 186] [--at 1.05] \
//!     [--angle 90] [--surface 1] [--secs 1.2] [--run-up M] [--self 1] [--min-kg K] \\
//!     [--drop H] [--centred 1] [--rise J] [--series 1]
//! ```
//! `--self 0` turns self-collision off and `--min-kg K` raises every lighter
//! node to K kg (diagnostics: they change the car).
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, StaticBox, NODE_IMMOVABLE, STEP,
};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use glam::{DVec3, Vec3};

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1).cloned())
}

fn num(k: &str, d: f32) -> f32 {
    arg(k).and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> Result<(), String> {
    let dir = std::path::PathBuf::from(arg("--mod").ok_or("--mod <mods dir>/<mod>")?);
    let vehicle = arg("--vehicle").unwrap_or_else(|| "sedan".into());
    let pc_name = arg("--pc").unwrap_or_else(|| format!("{vehicle}.pc"));
    let kmh = num("--kmh", 32.0);
    let at = num("--at", 0.0);
    let angle = num("--angle", 90.0);
    let surface = num("--surface", 1.0) != 0.0;
    let secs = num("--secs", 1.2);
    let v = kmh / 3.6;
    let gap = num("--run-up", 1.0 + 0.5 * v);
    let quiet = arg("--quiet").is_some();

    let files = engine_softbody::jbeam::read_vehicle_dir(&dir, &vehicle)?;
    let pc = std::fs::read_to_string(dir.join(&vehicle).join(&pc_name))
        .map_err(|e| format!("{pc_name}: {e}"))?;
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let desc = engine_softbody::jbeam::load_jbeam(&refs, Some(&pc))?;
    // The page's settings (ExplicitSim::spawn's defaults).
    let settings = ExplicitBeamsSettings::default();
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_raycast(&desc, Vec3::ZERO, Vec3::ZERO, settings)?;
    let names: Vec<String> = (0..s.nodes.len())
        .map(|i| {
            desc.nodes
                .get(i)
                .map_or(format!("#{i}"), |n| n.name.clone())
        })
        .collect();
    let part: Vec<String> = {
        let mut p = vec![String::new(); s.nodes.len()];
        for q in &desc.parts {
            for &i in &q.nodes {
                if let Some(x) = p.get_mut(i as usize) {
                    if x.is_empty() {
                        *x = q.name.clone();
                    }
                }
            }
        }
        p
    };
    // Forward to +X (yaw 90°): (x, z) → (z, −x) per quarter turn from +Z.
    let f = engine_softbody::pack::forward_axis(&desc);
    let to_z = glam::Quat::from_rotation_y(-f.x.atan2(f.z));
    for n in &mut s.nodes {
        let p = to_z * n.pos;
        n.pos = Vec3::new(p.z, p.y, -p.x);
    }
    let forward = Vec3::X;
    let lateral = Vec3::Y.cross(forward);
    let skin: Vec<u32> = s.triangles.iter().flat_map(|t| t.nodes).collect();
    let span = |axis: Vec3| {
        skin.iter()
            .map(|&i| s.nodes[i as usize].pos.dot(axis))
            .fold((f32::MAX, f32::MIN), |(a, b), x| (a.min(x), b.max(x)))
    };
    let (f0, f1) = span(forward);
    let (l0, l1) = span(lateral);
    let (length, width) = (f1 - f0, l1 - l0);
    let lowest = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let wheel_low = s.drive.as_ref().map(|d| {
        d.config
            .wheels
            .iter()
            .map(|w| {
                0.5 * (s.nodes[w.node1 as usize].pos.y + s.nodes[w.node2 as usize].pos.y) - w.radius
            })
            .fold(f32::MAX, f32::min)
    });
    let rise = match wheel_low {
        Some(wl) => (s.spawn_clearance() - lowest).max(0.02 - wl),
        None => s.spawn_clearance() - lowest,
    };
    let shift = forward * (0.5 * length - f1) + lateral * (-0.5 * (l0 + l1)) + Vec3::Y * rise;
    for n in &mut s.nodes {
        n.pos += shift;
    }
    // `--drop H`: the crash lab's `drop` scene instead (dropped flat from H
    // metres, moving along its forward axis at `--kmh`; no pole, no carrier).
    let drop = arg("--drop").and_then(|v| v.parse::<f32>().ok());
    if let Some(h) = drop {
        for n in &mut s.nodes {
            n.pos.y += h;
        }
    }
    let (pole_c, vel) = if drop.is_some() {
        (Vec3::new(0.0, -1e4, 1e4), forward * v)
    } else if angle == 90.0 {
        (
            Vec3::new(at, 2.0, 0.5 * width + gap + 0.1),
            Vec3::new(0.0, 0.0, v),
        )
    } else {
        let d = Vec3::new(angle.to_radians().cos(), 0.0, angle.to_radians().sin());
        let p = Vec3::new(at, 0.0, 0.5 * width) + d * (gap + 0.1);
        (Vec3::new(p.x, 2.0, p.z), d * v)
    };
    for n in &mut s.nodes {
        n.vel = if n.flags & NODE_IMMOVABLE != 0 {
            Vec3::ZERO
        } else {
            vel
        };
    }
    if let Some(d) = s.drive.as_mut() {
        let [c, b, _] = d.config.refs;
        let fw = (s.nodes[c as usize].pos - s.nodes[b as usize].pos).normalize_or_zero();
        d.roll(vel.dot(fw));
    }
    if drop.is_none() {
        engine_softbody::crash_rigs::on_carrier(&mut s);
        s.boxes
            .push(StaticBox::centered(pole_c, Vec3::new(0.1, 2.0, 0.1)));
    }
    s.surface_contact = surface;
    if num("--self", 1.0) == 0.0 {
        for n in &mut s.nodes {
            n.flags &= !engine_softbody::explicit_beams::NODE_CONTACTER;
        }
    }
    // `--centred 0`: the BeamNG profile's centred damping off (a diagnostic).
    if num("--centred", 1.0) == 0.0 {
        s.settings.centred_damping = false;
    }
    let min_kg = num("--min-kg", 0.0);
    if min_kg > 0.0 {
        for n in &mut s.nodes {
            n.mass = n.mass.max(min_kg);
        }
    }
    let wheel_nodes: Vec<u32> = s
        .drive
        .as_ref()
        .map(|d| {
            d.config
                .wheels
                .iter()
                .flat_map(|w| [w.node1, w.node2])
                .collect()
        })
        .unwrap_or_default();
    let on_tri: std::collections::BTreeSet<u32> = skin.iter().copied().collect();
    let n = s.nodes.len();
    let mass: Vec<f32> = s.nodes.iter().map(|x| x.mass).collect();
    let total: f32 = mass.iter().sum();
    let light = {
        let mut v: Vec<f32> = on_tri.iter().map(|&i| mass[i as usize]).collect();
        v.sort_by(|a, b| a.total_cmp(b));
        v
    };
    if !quiet {
        println!(
            "{vehicle}: {n} nodes, {} beams, {} triangles, {total:.0} kg; lightest node {:.3} kg, lightest on a collision triangle {:.3} kg (p10 {:.3}); {length:.2} × {width:.2} m; pole {:.2} m ahead",
            s.beams.len(),
            s.triangles.len(),
            mass.iter().copied().fold(f32::MAX, f32::min),
            light.first().copied().unwrap_or(f32::NAN),
            light.get(light.len() / 10).copied().unwrap_or(f32::NAN),
            gap + 0.1,
        );
    }
    let mut w = ExplicitBeamsWorld::new(vec![s]);
    w.enable_ledger();
    let e0 = w.ledger.as_ref().unwrap().start;
    let steps = (secs / STEP).round() as usize;
    // Per node, cumulative work (J): [node-pass contacts, self-collision, beams].
    let mut cum = vec![[0.0f64; 3]; n];
    let mut ring: std::collections::VecDeque<(usize, Vec<[f64; 3]>, [f64; 6])> = Default::default();
    let mut prev_parts: [Vec<Vec3>; 4] = Default::default();
    for p in &mut prev_parts {
        p.resize(n, Vec3::ZERO);
    }
    let (mut e_min, mut t_first) = (e0, None::<(usize, f64)>);
    let mut e_peak = e0;
    let mut vmax_all = 0.0f32;
    let mut t_hit = None::<usize>;
    let mut nan_at = None::<usize>;
    let threshold = num("--rise", (1e-4 * e0 as f32).max(50.0)) as f64;
    let src = |l: &engine_softbody::explicit_contact::EnergyLedger| {
        [
            l.w_ground_horizontal,
            l.w_ground_vertical,
            l.w_self_contact,
            l.w_beams + l.elastic - l.elastic_start,
            l.w_gravity + l.gravity_pe,
            l.w_air,
        ]
    };
    let mut dumped = false;
    for k in 0..steps {
        let before: Vec<(Vec3, Vec3)> = {
            let v0 = &w.vehicles[0];
            (0..n).map(|i| (v0.nodes[i].vel, v0.force_now(i))).collect()
        };
        w.step();
        let v0 = &w.vehicles[0];
        if t_hit.is_none()
            && v0.nodes.iter().any(|x| {
                (v0.origin + x.pos).z > pole_c.z - 0.13
                    && ((v0.origin + x.pos).x - pole_c.x).abs() < 0.13
            })
        {
            t_hit = Some(k);
        }
        for i in 0..n {
            let nd = &v0.nodes[i];
            let (vb, f_acc) = before[i];
            let dx = ((nd.vel + vb) * (0.5 * STEP)).as_dvec3();
            let ground = (nd.vel - vb) * (nd.mass / STEP) - f_acc;
            cum[i][0] += ground.as_dvec3().dot(dx);
            cum[i][1] += prev_parts[2][i].as_dvec3().dot(dx);
            cum[i][2] += prev_parts[1][i].as_dvec3().dot(dx);
        }
        if let Some(t) = v0.trace.as_ref() {
            for q in 0..4 {
                if t.parts[q].len() == n {
                    prev_parts[q].copy_from_slice(&t.parts[q]);
                }
            }
        }
        let l = w.ledger.as_ref().unwrap();
        let e = l.mechanical();
        if !e.is_finite() && nan_at.is_none() {
            nan_at = Some(k);
        }
        let vm = v0
            .nodes
            .iter()
            .map(|x| x.vel.length())
            .fold(0.0f32, f32::max);
        if vm.is_finite() {
            vmax_all = vmax_all.max(vm);
        }
        if k % 10 == 0 {
            ring.push_back((k, cum.clone(), src(l)));
            if ring.len() > 5 {
                ring.pop_front();
            }
        }
        if arg("--series").is_some() && k % 100 == 99 {
            let q = src(l);
            let (vi, vm) = v0
                .nodes
                .iter()
                .enumerate()
                .map(|(i, x)| (i, x.vel.length()))
                .fold((0, 0.0f32), |a, b| if b.1 > a.1 { b } else { a });
            println!(
                "  t {:.2} E {:.1} kJ | cum J: contacts h {:+.0} v {:+.0} self {:+.0} beams+U {:+.0} grav {:+.0} air {:+.0} | fastest {} {:.3} kg {:.0} m/s | broken {}",
                (k + 1) as f32 * STEP,
                e * 1e-3,
                q[0], q[1], q[2], q[3], q[4], q[5],
                names[vi], mass[vi], vm,
                v0.beams.iter().filter(|b| b.broken).count()
            );
        }
        e_min = e_min.min(e);
        e_peak = e_peak.max(e);
        if t_first.is_none() && (e - e_min > threshold || !e.is_finite()) {
            t_first = Some((k, e - e_min));
        }
        if let (Some((kf, _)), false) = (t_first, dumped) {
            // 5 ms after the first rise: the window's sources and nodes.
            if k >= kf + 10 || !e.is_finite() {
                dumped = true;
                let (k0, c0, s0) = ring.front().unwrap();
                let s1 = src(l);
                println!(
                    "  first rise at step {kf} (t {:.4} s; pole reached at {}): window {:.1} ms; sources' energy change (J): contacts horizontal {:+.0}, vertical {:+.0}, self-collision {:+.0}, beams + elastic {:+.0}, gravity + height {:+.0}, air {:+.0}",
                    kf as f32 * STEP,
                    t_hit.map_or("never".into(), |t| format!("{:.4} s", t as f32 * STEP)),
                    (k - k0) as f32 * STEP * 1e3,
                    s1[0] - s0[0],
                    s1[1] - s0[1],
                    s1[2] - s0[2],
                    s1[3] - s0[3],
                    s1[4] - s0[4],
                    s1[5] - s0[5],
                );
                for (q, label) in ["node-pass contacts", "self-collision", "beams"]
                    .iter()
                    .enumerate()
                {
                    let mut top: Vec<(f64, usize)> =
                        (0..n).map(|i| (cum[i][q] - c0[i][q], i)).collect();
                    top.sort_by(|a, b| b.0.total_cmp(&a.0));
                    let tot: f64 = top.iter().map(|x| x.0.max(0.0)).sum();
                    println!("  {label}: positive work {tot:.0} J; top nodes:");
                    for &(wk, i) in top.iter().take(6) {
                        if wk <= 0.5 {
                            break;
                        }
                        let nd = &v0.nodes[i];
                        let p = v0.origin + nd.pos;
                        let rel = p - Vec3::new(pole_c.x, 0.0, pole_c.z);
                        let kind = if wheel_nodes.contains(&(i as u32)) {
                            "wheel axle"
                        } else if rel.x.abs() < 0.2 && rel.z.abs() < 0.2 {
                            "at the pole"
                        } else if p.y < 0.03 {
                            "on the ground"
                        } else {
                            "elsewhere"
                        };
                        println!(
                            "    {:>14} ({}) {:.3} kg{}: {wk:+.0} J; at pole-relative ({:+.2}, {:.2}, {:+.2}) [{kind}], speed {:.0} m/s",
                            names[i],
                            part[i],
                            nd.mass,
                            if on_tri.contains(&(i as u32)) { ", triangle corner" } else { "" },
                            rel.x,
                            p.y,
                            rel.z,
                            nd.vel.length()
                        );
                    }
                }
            }
        }
        if !e.is_finite() {
            break;
        }
    }
    let v0 = &w.vehicles[0];
    let l = w.ledger.as_ref().unwrap();
    let broken = v0.beams.iter().filter(|b| b.broken).count();
    let bent = v0
        .beams
        .iter()
        .filter(|b| !b.broken && (b.rest - b.rest0).abs() > 1e-4 * b.rest0.abs().max(1e-3))
        .count();
    let (mut px, mut m) = (DVec3::ZERO, 0.0f64);
    for x in &v0.nodes {
        px += x.vel.as_dvec3() * f64::from(x.mass);
        m += f64::from(x.mass);
    }
    println!(
        "{vehicle} {kmh:.0} km/h at {at:.2} surface {} : E0 {:.0} kJ, end {:.0} kJ, peak {:.0} kJ ({:+.1} % of E0); first rise {}; fastest node ever {:.0} m/s; end speed {:.0} km/h; {broken} broken / {bent} bent of {}{}",
        surface as u8,
        e0 * 1e-3,
        l.mechanical() * 1e-3,
        e_peak * 1e-3,
        (e_peak - e0) / e0 * 100.0,
        t_first.map_or("none".into(), |(k, d)| format!("{:.4} s (+{d:.0} J)", k as f32 * STEP)),
        vmax_all,
        (px / m).length() * 3.6,
        v0.beams.len(),
        nan_at.map_or(String::new(), |k| format!("; NOT FINITE at {:.4} s", k as f32 * STEP)),
    );
    Ok(())
}
