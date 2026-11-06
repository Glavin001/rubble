//! Crash references for a JBeam car in JBeam mode on raycast wheels (CPU
//! reference): the rigid-wall frontal at 56 km/h and a side-pole sweep at
//! 32 km/h over the car's full length, measured against
//! `engine/docs/crash-reference-targets.md` §1. Runs on the generated `gen2`
//! sedan or any community car, so the two compare on the same measurements.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example gen2_crash -- <vehicles-dir> <name> <config.pc> [wall] [pole] [rest] [step=0.25] [kmh=56] [polekmh=32]
//! # gen2: engine/crates/engine-webgpu/__mods/gen2 gen2 sedan.pc
//! ```
//! * `rest`: 10 s parked on raycast wheels: broken and bent beams, KE.
//! * `wall`: residual crush at bumper height (6 zones across the front, as
//!   NHTSA's C1–C6, after rebound), dynamic crush (centre-of-mass travel from
//!   contact to stop), time to it, restitution, peak and mean deceleration
//!   (10 ms average of the centre of mass), Campbell b1, length loss.
//! * `pole`: a 254 mm rigid pole, the car driven sideways into it at each
//!   station along its length: the deepest node inside the pole's face (the
//!   node-only contact's clip), residual crush at the pole, beams broken.
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox, StaticShape};
use glam::Vec3;

fn mass_cg(s: &ExplicitBeams) -> (f32, Vec3, Vec3) {
    let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let p = s.nodes.iter().map(|n| n.pos * n.mass).sum::<Vec3>() / m;
    let v = s.nodes.iter().map(|n| n.vel * n.mass).sum::<Vec3>() / m;
    (m, p, v)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (dir, name, pc) = (&a[0], &a[1], &a[2]);
    let has = |k: &str| a.iter().any(|x| x == k);
    let opt = |k: &str| a.iter().find_map(|x| x.strip_prefix(k)).map(str::to_string);
    let f = |k: &str, d: f32| opt(k).map_or(d, |s| s.parse().unwrap());
    let dir = std::path::Path::new(dir);
    let texts = engine_softbody::jbeam::read_vehicle_dir(dir, name).unwrap();
    let files: Vec<(&str, &str)> = texts.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let pcs = std::fs::read_to_string(dir.join(name).join(pc)).unwrap();
    let desc = engine_softbody::jbeam::load_jbeam(&files, Some(&pcs)).unwrap();
    let spawn = |v: Vec3| -> ExplicitBeams {
        let settings = ExplicitBeamsSettings { drag: false, turbulence: false, ..Default::default() };
        let (mut s, _) = ExplicitBeams::from_desc_jbeam_raycast(&desc, Vec3::ZERO, v, settings).unwrap();
        let fwd = engine_softbody::pack::forward_axis(&desc);
        let turn = glam::Quat::from_rotation_y(-fwd.x.atan2(fwd.z));
        for n in &mut s.nodes {
            n.pos = turn * n.pos;
            n.vel = turn * n.vel;
        }
        // Rest the body at its authored ride height: its lowest node where the
        // file has it above the ground (raycast wheels hold it there).
        let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::INFINITY, f32::min);
        let authored = desc.nodes.iter().map(|n| n.position[1]).fold(f32::INFINITY, f32::min);
        for n in &mut s.nodes {
            n.pos.y += authored.max(0.05) - low;
        }
        s
    };
    let named = |s: &ExplicitBeams, nm: &str| desc.nodes.iter().position(|n| n.name == nm).map(|i| s.nodes[i].pos);

    if has("rest") {
        let mut s = spawn(Vec3::ZERO);
        if has("noself") {
            s.candidates.clear();
        }
        if has("park") {
            // Parked: the brakes on, as a parked car's handbrake.
            s.set_drive_input(engine_softbody::explicit_drive::DriveInput { forward: 0.0, steer: 0.0, brake: 1.0, handbrake: 1.0 });
        }
        let ke =|s: &ExplicitBeams| s.nodes.iter().map(|n| 0.5 * n.mass * n.vel.length_squared()).sum::<f32>();
        let mut peak_v = 0.0f32;
        for k in 1..=100 {
            s.run(0.1);
            peak_v = peak_v.max(s.nodes.iter().map(|n| n.vel.length()).fold(0.0, f32::max));
            if k % 20 == 0 {
                if has("fast") {
                    let mut v: Vec<(f32, usize)> = s.nodes.iter().enumerate().map(|(i, n)| (n.vel.length(), i)).collect();
                    v.sort_by(|a, b| b.0.total_cmp(&a.0));
                    let names: Vec<String> = v.iter().take(6).map(|&(sp, i)| format!("{} {sp:.3}", desc.nodes.get(i).map_or("?", |n| n.name.as_str()))).collect();
                    println!("    fastest {names:?} contacts {}", s.last_contacts);
                }
                let (b, bent) = s.damage();
                println!("rest {:.0} s: broken {b} bent {bent} KE {:.2} J, peak node speed so far {peak_v:.2} m/s", k as f32 / 10.0, ke(&s));
            }
        }
    }

    if has("wall") {
        let v0 = f("kmh=", 56.0) / 3.6;
        let mut s = spawn(Vec3::new(0.0, 0.0, v0));
        let n0 = desc.nodes.len();
        let front = s.nodes.iter().map(|n| n.pos.z).fold(f32::NEG_INFINITY, f32::max);
        let wall_z = front + 0.01;
        s.boxes.push(StaticBox::centered(Vec3::new(0.0, 1.5, wall_z + 1.0), Vec3::new(5.0, 3.0, 1.0)));
        // Bumper-level profile: 6 zones across the width (NHTSA C1–C6).
        let (xmin, xmax) = s.nodes[..n0].iter().fold((f32::MAX, f32::MIN), |(lo, hi), n| (lo.min(n.pos.x), hi.max(n.pos.x)));
        let band = |n: &engine_softbody::explicit_beams::Node| n.pos.y > 0.15 && n.pos.y < 0.90;
        let zone = |x: f32| (((x - xmin) / (xmax - xmin) * 6.0) as usize).min(5);
        // Reference: the mean position of the rear 40 % of the car's nodes (by
        // spawn position), which the frontal leaves undamaged.
        let zmin0 = s.nodes[..n0].iter().map(|n| n.pos.z).fold(f32::MAX, f32::min);
        let rear_set: Vec<usize> = (0..n0).filter(|&i| s.nodes[i].pos.z < zmin0 + 0.4 * (front - zmin0)).collect();
        let rear_ref = |s: &ExplicitBeams| rear_set.iter().map(|&i| s.nodes[i].pos.z).sum::<f32>() / rear_set.len() as f32;
        let rear_off = rear_ref(&s) - zmin0;
        // The car's main piece: the heaviest set of nodes joined by unbroken
        // beams (a part that came off does not count as the car's front).
        let main_piece = |s: &ExplicitBeams| -> Vec<bool> {
            let mut parent: Vec<usize> = (0..n0).collect();
            fn find(p: &mut Vec<usize>, i: usize) -> usize {
                let mut r = i;
                while p[r] != r {
                    r = p[r];
                }
                let mut i = i;
                while p[i] != r {
                    let nx = p[i];
                    p[i] = r;
                    i = nx;
                }
                r
            }
            for b in &s.beams {
                let (a, c) = (b.a as usize, b.b as usize);
                if !b.broken && a < n0 && c < n0 && a != c {
                    let (ra, rc) = (find(&mut parent, a), find(&mut parent, c));
                    parent[ra] = rc;
                }
            }
            let mut mass = vec![0.0f32; n0];
            for i in 0..n0 {
                let r = find(&mut parent, i);
                mass[r] += s.nodes[i].mass;
            }
            let best = (0..n0).max_by(|&a, &b| mass[a].total_cmp(&mass[b])).unwrap();
            (0..n0).map(|i| find(&mut parent, i) == best).collect()
        };
        let profile = |s: &ExplicitBeams| {
            let r = rear_ref(s);
            let keep = main_piece(s);
            let mut z = [f32::NEG_INFINITY; 6];
            for (i, n) in s.nodes[..n0].iter().enumerate() {
                if band(n) && keep[i] {
                    let k = zone(n.pos.x);
                    z[k] = z[k].max(n.pos.z - r);
                }
            }
            z
        };
        let p0 = profile(&s);
        let spawn_pos: Vec<Vec3> = s.nodes.iter().map(|n| n.pos).collect();
        let r0 = rear_ref(&s);
        let len0 = front - rear_ref(&s);
        let (m, cg0, _) = mass_cg(&s);
        let cab0 = [named(&s, "cowlOl"), named(&s, "soT0l"), named(&s, "soT2l"), named(&s, "bp1l")];
        let mut contact_t = None;
        let mut max_travel = 0.0f32;
        let mut t_max = 0.0f32;
        let mut vs: Vec<f32> = Vec::new();
        let mut cg_contact = cg0;
        for step in 0..(0.30 / 0.0005) as usize {
            s.step();
            let (_, cg, v) = mass_cg(&s);
            vs.push(v.z);
            let t = (step + 1) as f32 * 0.0005;
            if contact_t.is_none() && s.nodes[..n0].iter().any(|n| n.pos.z >= wall_z - 0.002) {
                contact_t = Some(t);
                cg_contact = cg;
            }
            if contact_t.is_some() {
                let travel = cg.z - cg_contact.z;
                if travel > max_travel {
                    max_travel = travel;
                    t_max = t - contact_t.unwrap();
                }
            }
        }
        // Rebound: the most negative CG velocity.
        let v_reb = -vs.iter().copied().fold(f32::INFINITY, f32::min).min(0.0);
        // Deceleration: 10 ms moving average of dv/dt.
        let w = 20usize;
        let mut peak = 0.0f32;
        for k in w..vs.len() {
            peak = peak.max((vs[k - w] - vs[k]) / (w as f32 * 0.0005) / 9.81);
        }
        let p1 = profile(&s);
        let crush: Vec<f32> = (0..6).map(|k| if p0[k].is_finite() && p1[k].is_finite() { p0[k] - p1[k] } else { f32::NAN }).collect();
        let cmax = crush.iter().copied().filter(|c| c.is_finite()).fold(0.0, f32::max);
        let cavg = (crush[0] / 2.0 + crush[1] + crush[2] + crush[3] + crush[4] + crush[5] / 2.0) / 5.0;
        let keep1 = main_piece(&s);
        let len1 = s.nodes[..n0].iter().enumerate().filter(|(i, _)| keep1[*i]).map(|(_, n)| n.pos.z).fold(f32::NEG_INFINITY, f32::max) - rear_ref(&s);
        let _ = rear_off;
        let b1 = (v0 - 8.0 / 3.6) / cavg.max(1e-3);
        let (broken, bent) = s.damage();
        println!("wall {:.0} km/h, {m:.0} kg:", v0 * 3.6);
        println!("  residual crush C1..C6 (mm) {:?}", crush.iter().map(|c| (c * 1e3).round()).collect::<Vec<_>>());
        println!("  max residual crush {:.0} mm (NHTSA sedans p10-p90 380-580, median 491); NHTSA average C {:.0} mm (261-512, 428)", cmax * 1e3, cavg * 1e3);
        println!("  length loss {:.0} mm (311-602, 470)", (len0 - len1) * 1e3);
        println!("  dynamic crush (CG travel) {:.0} mm at {:.1} ms (max dynamic crush not openly available; time ~77 ms one test)", max_travel * 1e3, t_max * 1e3);
        println!("  restitution {:.3} (0.03-0.2, ~0.11), rebound {:.2} m/s", v_reb / v0, v_reb);
        println!("  peak deceleration (10 ms average) {:.1} g (fleet 36-46 g at the occupant compartment); Campbell b1 {:.1} /s (22.9-39.2, 30.4)", peak, b1);
        println!("  broken {broken} bent {bent}");
        if let Some(list) = opt("nodes=") {
            // Residual rearward movement of named nodes, relative to the rear.
            let r1 = rear_ref(&s);
            for nm in list.split(',') {
                if let Some(i) = desc.nodes.iter().position(|n| n.name == nm) {
                    let z0 = spawn_pos[i].z - r0;
                    println!("    {nm}: {:+.0} mm rearward, {:+.0} mm up", (z0 - (s.nodes[i].pos.z - r1)) * 1e3, (s.nodes[i].pos.y - spawn_pos[i].y) * 1e3);
                }
            }
        }
        if has("broken") {
            let nm = |i: u32| desc.nodes.get(i as usize).map_or("?".to_string(), |n| n.name.clone());
            for b in s.beams.iter().filter(|b| b.broken) {
                println!("    broken {:?} {}-{} strength {:.0} group {}", b.kind, nm(b.a), nm(b.b), b.strength, b.group);
            }
        }
        if has("yielded") {
            // Beams behind the dash that yielded most (plastic strain).
            let nm = |i: u32| desc.nodes.get(i as usize).map_or("?".to_string(), |n| n.name.clone());
            let mut v: Vec<(f32, usize)> = s
                .beams
                .iter()
                .enumerate()
                .filter(|(_, b)| (b.a as usize) < n0 && (b.b as usize) < n0 && b.rest0 > 0.0)
                .filter(|(_, b)| {
                    let ya = desc.nodes[b.a as usize].position[2];
                    let yb = desc.nodes[b.b as usize].position[2];
                    // Rubble z forward = −JBeam y: behind the dash (y > 0.5).
                    if has("front") { -ya < 0.5 && -yb < 0.5 } else { -ya > 0.5 && -yb > 0.5 }
                })
                .map(|(j, b)| (((b.rest - b.rest0) / b.rest0).abs(), j))
                .collect();
            v.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(e, j) in v.iter().take(40) {
                let b = &s.beams[j];
                println!("    yielded {:.1} % {}-{} broken {} deform {:.0}", e * 100.0, nm(b.a), nm(b.b), b.broken, b.tc);
            }
        }
        let cab1 = [named(&s, "cowlOl"), named(&s, "soT0l"), named(&s, "soT2l"), named(&s, "bp1l")];
        if let ([Some(a0), Some(s0), Some(t0), Some(b0)], [Some(a1), Some(s1), Some(t1), Some(b1n)]) = (cab0, cab1) {
            // Relative to the B-pillar: A-pillar base and cowl rearward, the
            // front door aperture's length change.
            println!(
                "  cabin: A-pillar base rearward {:.0} mm (median 2, p10-p90 -7..17), cowl rearward {:.0} mm (firewall median 15-20, p90 67-85), front door aperture lower {:+.0} mm, upper {:+.0} mm (p10-p90 -9..+9)",
                -((s1.z - t1.z) - (s0.z - t0.z)) * 1e3,
                -((a1.z - b1n.z) - (a0.z - b0.z)) * 1e3,
                ((s1 - t1).length() - (s0 - t0).length()) * 1e3,
                ((a1 - b1n).length() - (a0 - b0).length()) * 1e3,
            );
        }
    }

    if has("pole") {
        let kmh = f("polekmh=", 32.0);
        let step = f("step=", 0.25);
        let s0 = spawn(Vec3::ZERO);
        let n0 = desc.nodes.len();
        let (zmin, zmax) = s0.nodes[..n0].iter().fold((f32::MAX, f32::MIN), |(lo, hi), n| (lo.min(n.pos.z), hi.max(n.pos.z)));
        let side = s0.nodes[..n0].iter().map(|n| n.pos.x).fold(f32::MIN, f32::max);
        let r = 0.127;
        println!("side pole {kmh:.0} km/h, 254 mm pole, stations every {step} m along {:.2} m (front {zmax:.2}):", zmax - zmin);
        let mut worst = 0.0f32;
        let mut z = zmax - 0.15;
        while z > zmin + 0.1 {
            // The car moves toward +x (its left side leads) into the pole.
            let mut s = spawn(Vec3::new(kmh / 3.6, 0.0, 0.0));
            let c = Vec3::new(side + r + 0.01, 1.0, z);
            s.shapes.push(StaticShape::cylinder(c, r, 1.2));
            let mut deepest = 0.0f32;
            let mut who = 0usize;
            let side0 = s.nodes[..n0].iter().map(|n| n.pos.x).fold(f32::MIN, f32::max);
            for _ in 0..(0.15 / 0.0005) as usize {
                s.step();
                for (i, n) in s.nodes[..n0].iter().enumerate() {
                    if n.pos.y < c.y - 1.2 || n.pos.y > c.y + 1.2 {
                        continue;
                    }
                    let d = Vec3::new(n.pos.x - c.x, 0.0, n.pos.z - c.z).length();
                    if r - d > deepest {
                        deepest = r - d;
                        who = i;
                    }
                }
            }
            // Residual crush at the pole: how far the side at the pole's
            // station moved in, relative to the far side.
            let near = |s: &ExplicitBeams| {
                s.nodes[..n0].iter().filter(|n| (n.pos.z - z).abs() < 0.15 && n.pos.y > 0.25 && n.pos.y < 1.1).map(|n| n.pos.x).fold(f32::MIN, f32::max)
            };
            let far = |s: &ExplicitBeams| s.nodes[..n0].iter().map(|n| n.pos.x).fold(f32::MAX, f32::min);
            let crush = (side0 - far(&s0)) - (near(&s) - far(&s));
            let (broken, _) = s.damage();
            worst = worst.max(deepest);
            println!(
                "  station {:+.2} m: deepest node {:.1} mm inside the pole ({}), crush at the pole {:.0} mm, broken {broken}",
                z,
                deepest * 1e3,
                desc.nodes[who].name,
                crush * 1e3
            );
            z -= step;
        }
        println!("  worst clip {:.1} mm (band 25 mm)", worst * 1e3);
    }
}
