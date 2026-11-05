//! The energy ledger of `ExplicitBeamsWorld` (CPU): it closes (the kinetic
//! energy's change is the sum of every source's work less the step's
//! discretisation term), the vehicles' mechanical energy (kinetic, gravitational,
//! elastic) never rises, so no contact injects energy, and the beams' dissipation
//! is never negative and never more than the energy lost, in two standard crashes: two procedural sedans rear-ending
//! at 15 m/s, and the authored showcase car striking another 80 km/h off the door
//! middle (the T-bone whose per-beam tally exceeded the striker's energy).
use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::explicit_contact::{EnergyLedger, ExplicitBeamsWorld};
use engine_softbody::procedural::{car_truck_bare, CarParams};
use glam::Vec3;

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    }
}

fn report(label: &str, l: &EnergyLedger) {
    eprintln!(
        "{label}: KE {:.1} kJ, gravity PE {:.1} kJ, elastic {:.2} kJ (start {:.1} kJ mechanical); work: beams {:.1}, \
         self-contact {:.2}, vehicle contact normal {:.2} friction {:.2}, ground vertical {:.2} horizontal {:.2}, \
         air {:.2}, gravity {:.2}, slides {:.2} kJ; discretisation {:.3} kJ; beam dissipation {:.1} kJ",
        l.kinetic * 1e-3,
        l.gravity_pe * 1e-3,
        l.elastic * 1e-3,
        l.start * 1e-3,
        l.w_beams * 1e-3,
        l.w_self_contact * 1e-3,
        l.w_contact_normal * 1e-3,
        l.w_contact_friction * 1e-3,
        l.w_ground_vertical * 1e-3,
        l.w_ground_horizontal * 1e-3,
        l.w_air * 1e-3,
        l.w_gravity * 1e-3,
        l.w_slides * 1e-3,
        l.discretisation * 1e-3,
        l.beam_dissipation() * 1e-3,
    );
}

/// Runs `w` for `frames` 33-step frames with the ledger and checks it every frame.
/// Tolerances: rounding of f32 states summed in f64 (closure, 0.1 % of the start
/// energy) and the symplectic step's bounded energy error (0.5 %).
fn check(label: &str, mut w: ExplicitBeamsWorld, frames: usize) -> EnergyLedger {
    w.enable_ledger();
    let ke0 = w.ledger.as_ref().unwrap().kinetic;
    let e0 = w.ledger.as_ref().unwrap().start;
    let mut most = e0;
    for f in 0..frames {
        w.frame(33);
        let l = w.ledger.as_ref().unwrap();
        let closure = (l.kinetic - ke0) - l.kinetic_change();
        assert!(
            closure.abs() < 1e-3 * e0,
            "{label} frame {f}: ledger off by {closure} J"
        );
        most = most.max(l.mechanical());
        // (A contact's work alone may be positive: its projection cancels the
        // contacter's beam force, so it passes beam energy on; the sum is what
        // must not grow.)
    }
    let l = w.ledger.clone().unwrap();
    report(label, &l);
    assert!(
        most <= e0 * 1.005,
        "{label}: mechanical energy rose from {e0} J to {most} J"
    );
    assert!(
        l.beam_dissipation() > -5e-3 * e0,
        "{label}: beams released {} J",
        -l.beam_dissipation()
    );
    l
}

#[test]
fn ledger_closes_in_a_sedan_rear_end() {
    let text = car_truck_bare(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let (z0, z1) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(a, b), n| {
        (a.min(n.pos.z), b.max(n.pos.z))
    });
    let v = Vec3::new(0.0, 0.0, 15.0);
    let mut rear =
        ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - low, 0.0), v, settings()).unwrap();
    rear.roll_tyres(v);
    let front = ExplicitBeams::from_truck(
        &text,
        Vec3::new(0.0, 0.01 - low, z1 - z0 + 0.5),
        Vec3::ZERO,
        settings(),
    )
    .unwrap();
    check(
        "sedan rear-end 15 m/s",
        ExplicitBeamsWorld::new(vec![front, rear]),
        60,
    );
}

/// The showcase car into another's front door, 80 km/h, 0.7 m forward of the
/// door's middle (`showcase_car.rs` test u, `TB_DZ=0.7`), with two-body contact
/// (jbeam-semantics-spec.md §10.3, the default). With RoR's rule (the RoR-compatible
/// profile) the same crash's contacts inject energy: reported, not asserted.
#[test]
fn ledger_closes_in_the_showcase_t_bone_off_the_door_middle() {
    let b = build(
        &style("sedan").unwrap(),
        &BuildParams {
            resolution: Resolution::Medium,
            ..Default::default()
        },
    )
    .unwrap();
    let sk0 = &b.skeleton;
    let len = b.def.profile.length;
    let door_part = sk0.parts.iter().position(|p| p == "door_f_l").unwrap() as u16;
    let probe = ExplicitBeams::from_truck(&b.truck, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let struck = ExplicitBeams::from_truck(
        &b.truck,
        Vec3::new(0.0, 0.02 - low, 0.0),
        Vec3::ZERO,
        settings(),
    )
    .unwrap();
    let p0 = struck.positions();
    let door: Vec<usize> = (0..sk0.nodes.len())
        .filter(|&i| sk0.nodes[i].part == door_part)
        .collect();
    let zd = door.iter().map(|&i| p0[i].z).sum::<f32>() / door.len() as f32;
    let hw = p0.iter().map(|p| p.x).fold(f32::MIN, f32::max);
    let mut sk = sk0.clone();
    for nd in &mut sk.nodes {
        let q = nd.p - Vec3::new(0.0, 0.0, 0.5 * len);
        nd.p = Vec3::new(-q.z, q.y, q.x);
    }
    let truck = sk.to_truck();
    let pr = ExplicitBeams::from_truck(&truck, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    let pp = pr.positions();
    let (lo, near) = pp
        .iter()
        .fold((f32::MAX, f32::MAX), |(l, x), p| (l.min(p.y), x.min(p.x)));
    let low0 = p0.iter().map(|p| p.y).fold(f32::MAX, f32::min);
    let v = Vec3::new(-80.0 / 3.6, 0.0, 0.0);
    let mut striker = ExplicitBeams::from_truck(
        &truck,
        Vec3::new(hw + 0.05 - near, low0 - lo, zd + 0.7),
        v,
        settings(),
    )
    .unwrap();
    striker.roll_tyres(v);
    let ke_striker: f64 = striker
        .nodes
        .iter()
        .map(|n| 0.5 * f64::from(n.mass) * f64::from(n.vel.length_squared()))
        .sum();
    eprintln!(
        "striker's kinetic energy at the start {:.1} kJ",
        ke_striker * 1e-3
    );
    // RoR's rule: reported.
    let (mut ror_a, mut ror_b) = (struck.clone(), striker.clone());
    ror_a.two_body_contact = false;
    ror_b.two_body_contact = false;
    let mut w = ExplicitBeamsWorld::new(vec![ror_a, ror_b]);
    w.enable_ledger();
    for _ in 0..60 {
        w.frame(33);
    }
    report("OPEN, RoR's contact rule", w.ledger.as_ref().unwrap());
    let (mut struck, mut striker) = (struck, striker);
    struck.two_body_contact = true;
    striker.two_body_contact = true;
    let l = check(
        "two-body contact",
        ExplicitBeamsWorld::new(vec![struck, striker]),
        60,
    );
    // The beams can dissipate no more than the vehicles lost.
    assert!(
        l.beam_dissipation() <= l.start - l.mechanical() + 5e-3 * l.start,
        "beams {} J of {} J lost",
        l.beam_dissipation(),
        l.start - l.mechanical()
    );
}

/// Where a coasting pressure-tyre car's energy goes (opt-in, `RUBBLE_BEAMNG_MODS`):
/// reported (spec §31's rolling-resistance question).
#[test]
fn coasting_pressure_tyres_energy_split() {
    use engine_softbody::explicit_beams::JbeamWheels;
    use engine_softbody::jbeam::load_jbeam;
    let Ok(mods) = std::env::var("RUBBLE_BEAMNG_MODS") else {
        return;
    };
    let m = std::path::Path::new(&mods);
    for (dir, name, pc) in [
        ("monster_supercar/vehicles", "supercar", "config.pc"),
        (
            "qlourie_astral_r/vehicles",
            "qlourie_astral_r",
            "wrc17_tarmac_balanced.pc",
        ),
    ] {
        let dir = m.join(dir);
        let Ok(files) = engine_softbody::jbeam::read_vehicle_dir(&dir, name) else {
            continue;
        };
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let pcs = std::fs::read_to_string(dir.join(name).join(pc)).unwrap();
        let mut desc = load_jbeam(&refs, Some(&pcs)).unwrap();
        // Diagnostics only (COAST_DIAG, measurements of what the loss depends on,
        // never a model change): `rays=K` multiplies every tyre's numRays, `nodamp=`
        // a comma list of beam-set prefixes (e.g. wheelSideBeam) whose Damp and
        // DampExpansion are zeroed, `soft_side` the sidewalls' expansion rates set
        // to their compression ones.
        if let Ok(d) = std::env::var("COAST_DIAG") {
            for pw in &mut desc.pressure_wheels {
                for item in d.split(';') {
                    if let Some(k) = item.strip_prefix("rays=") {
                        let k: f32 = k.parse().unwrap();
                        let n = pw.values.get("numRays").copied().unwrap_or(10.0);
                        pw.values.insert("numRays".into(), n * k);
                    }
                    if item == "soft_side" {
                        // The sidewall's expansion rates set to its compression ones.
                        for k in ["Spring", "Damp"] {
                            if let Some(&x) = pw.values.get(&format!("wheelSideBeam{k}")) {
                                pw.values.insert(format!("wheelSideBeam{k}Expansion"), x);
                            }
                        }
                    }
                    if let Some(list) = item.strip_prefix("nodamp=") {
                        for set in list.split(',') {
                            for suffix in ["Damp", "DampExpansion"] {
                                pw.values.insert(format!("{set}{suffix}"), 0.0);
                            }
                        }
                    }
                }
            }
            eprintln!("COAST_DIAG {d}");
        }
        for v0 in [9.0f32, 25.0] {
            let settings = ExplicitBeamsSettings {
                drag: false,
                turbulence: false,
                ..Default::default()
            };
            let v = Vec3::new(0.0, 0.0, v0);
            let (mut s, _) = ExplicitBeams::from_desc_jbeam_with(
                &desc,
                JbeamWheels::Pressure,
                Vec3::ZERO,
                v,
                settings,
            )
            .unwrap();
            let f = engine_softbody::pack::forward_axis(&desc);
            let turn = glam::Quat::from_rotation_y(-f.x.atan2(f.z));
            for n in &mut s.nodes {
                n.pos = turn * n.pos;
            }
            let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
            for n in &mut s.nodes {
                n.pos.y -= low - 0.02;
            }
            s.roll_tyres(v);
            let mut w = ExplicitBeamsWorld::new(vec![s]);
            for _ in 0..30 {
                w.frame(33);
            }
            w.enable_ledger();
            // Per beam its dissipation over the window, −Σ f·Δℓ − ΔU (f the force
            // over the step, compression positive, the trapezoid of its values at
            // the step's ends), by part: the tyres'
            // tread (tread ring to tread ring across), periphery (round a ring),
            // sidewall (tread to hub), hub (hub, axle), support (axle to tread),
            // and the rest of the car.
            let part = {
                let v = &w.vehicles[0];
                let mut role = vec![0u8; v.nodes.len()]; // 0 body, 1 hub, 2 tread, 3 axle
                let mut ring = vec![u32::MAX; v.nodes.len()];
                for t in &v.tyres {
                    let n = t.count as usize / 4;
                    for i in 0..t.count as usize {
                        let id = t.first as usize + i;
                        role[id] = if i < 2 * n { 1 } else { 2 };
                        ring[id] = (i / n) as u32 + 4 * t.first;
                    }
                    for &a in &t.axle {
                        role[a as usize] = 3;
                    }
                }
                v.beams
                    .iter()
                    .map(|b| {
                        let (ra, rb) = (role[b.a as usize], role[b.b as usize]);
                        let (lo, hi) = (ra.min(rb), ra.max(rb));
                        match (lo, hi) {
                            (2, 2) if ring[b.a as usize] == ring[b.b as usize] => 1,
                            (2, 2) => 0,
                            (1, 2) => 2,
                            (2, 3) => 4,
                            (1, 1) | (1, 3) | (3, 3) => 3,
                            _ => 5,
                        }
                    })
                    .collect::<Vec<usize>>()
            };
            let names = ["tread", "periphery", "sidewall", "hub", "support", "car"];
            let len = |v: &ExplicitBeams, j: usize| {
                let b = &v.beams[j];
                v.nodes[b.a as usize]
                    .pos
                    .distance(v.nodes[b.b as usize].pos)
            };
            let u = |v: &ExplicitBeams, j: usize| {
                let b = &v.beams[j];
                let e = len(v, j) - b.rest;
                0.5 * f64::from(b.k) * f64::from(e) * f64::from(e)
            };
            let u0: Vec<f64> = (0..w.vehicles[0].beams.len())
                .map(|j| u(&w.vehicles[0], j))
                .collect();
            let rest0: Vec<f32> = w.vehicles[0].beams.iter().map(|b| b.rest).collect();
            let mut diss = [0.0f64; 6];
            // Touchdowns (a node below the ground that was above it the step
            // before): count, and the normal kinetic energy they arrive with,
            // ½·m·v_y² (what an inelastic landing removes).
            let ground_y = w.vehicles[0].ground;
            let below = |v: &ExplicitBeams, i: usize| v.origin.y + v.nodes[i].pos.y < ground_y;
            let (mut landings, mut land_ke) = (0usize, 0.0f64);
            let mut was: Vec<bool> = (0..w.vehicles[0].nodes.len())
                .map(|i| below(&w.vehicles[0], i))
                .collect();
            let mut prev_vel: Vec<Vec3> = w.vehicles[0].nodes.iter().map(|n| n.vel).collect();
            for _ in 0..60 * 33 {
                let v = &w.vehicles[0];
                let before: Vec<(f32, f32)> = (0..v.beams.len())
                    .map(|j| (v.beams[j].force, len(v, j)))
                    .collect();
                w.step();
                let v = &w.vehicles[0];
                for i in 0..v.nodes.len() {
                    let now = below(v, i);
                    if now && !was[i] {
                        landings += 1;
                        let vy = f64::from(prev_vel[i].y.min(0.0));
                        land_ke += 0.5 * f64::from(v.nodes[i].mass) * vy * vy;
                    }
                    was[i] = now;
                    prev_vel[i] = v.nodes[i].vel;
                }
                for (j, &(f, l0)) in before.iter().enumerate() {
                    // Trapezoid over the step (exact for a linear spring).
                    if !v.beams[j].broken {
                        let fm = 0.5 * (f64::from(f) + f64::from(v.beams[j].force));
                        diss[part[j]] -= fm * f64::from(len(v, j) - l0);
                    }
                }
            }
            let v = &w.vehicles[0];
            let nt = v.tyres.len().max(1) as f64;
            let revs: f64 = v
                .tyres
                .iter()
                .map(|t| f64::from(v0) / (2.0 * std::f64::consts::PI * f64::from(t.radius)))
                .sum::<f64>();
            eprintln!(
                "{name} {v0} m/s: touchdowns {landings} in 1 s ({:.1} per tyre revolution), their normal KE {:.2} kJ/s ({:.3} J each), tyres {}",
                landings as f64 / revs.max(1e-9),
                land_ke * 1e-3,
                land_ke / landings.max(1) as f64,
                nt
            );
            for j in 0..v.beams.len() {
                diss[part[j]] -= u(v, j) - u0[j];
            }
            let counts: Vec<usize> = (0..6)
                .map(|k| part.iter().filter(|&&p| p == k).count())
                .collect();
            let mut yielded = [0usize; 6];
            let mut drift = [0.0f64; 6];
            for (j, b) in v.beams.iter().enumerate() {
                let d = f64::from((b.rest - rest0[j]).abs());
                if d > 1e-7 {
                    yielded[part[j]] += 1;
                    drift[part[j]] += d;
                }
            }
            eprintln!(
                "{name} {v0} m/s: rest lengths moved over the window (beams, Σ|Δ| mm): {}",
                names
                    .iter()
                    .zip(yielded.iter().zip(&drift))
                    .map(|(n, (y, d))| format!("{n} {y} {:.2}", d * 1e3))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            eprintln!(
                "{name} {v0} m/s: beam dissipation by part (kJ/s, beams): {}",
                names
                    .iter()
                    .zip(diss.iter().zip(&counts))
                    .map(|(n, (d, c))| format!("{n} {:.2} ({c})", d * 1e-3))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            report(
                &format!("{name} coasting from {v0} m/s, 0.5-1.5 s"),
                w.ledger.as_ref().unwrap(),
            );
        }
    }
}

/// `rubble_exact_lengths` (Rubble's `.truck` extension) turns RoR's fast inverse
/// square root off for that vehicle; files without it keep it (RoR's rule).
#[test]
fn rubble_exact_lengths_turns_the_fast_inverse_square_root_off() {
    let text = car_truck_bare(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let ror = ExplicitBeams::from_truck(
        &text,
        Vec3::ZERO,
        Vec3::ZERO,
        ExplicitBeamsSettings::default(),
    )
    .unwrap();
    assert!(ror.settings.fast_inv_sqrt);
    // (After the title line.)
    let (title, rest) = text.split_once('\n').unwrap();
    let with = format!("{title}\nrubble_exact_lengths\n{rest}");
    let exact = ExplicitBeams::from_truck(
        &with,
        Vec3::ZERO,
        Vec3::ZERO,
        ExplicitBeamsSettings::default(),
    )
    .unwrap();
    assert!(!exact.settings.fast_inv_sqrt);
    // The fast inverse square root's length error over a sweep of lengths.
    let worst = (1..2000)
        .map(|i| {
            let l = 0.01 * i as f32;
            let q = l * l;
            let fast = q * engine_softbody::explicit_beams::fast_inv_sqrt(q);
            ((fast - l) / l).abs()
        })
        .fold(0.0f32, f32::max);
    eprintln!(
        "fast inverse square root: worst relative length error {:.3} %",
        100.0 * worst
    );
    assert!(worst < 2.0e-3);
}
