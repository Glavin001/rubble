//! The generated sedan's front structure against its own stated sections
//! (`engine/docs/sedan-front-audit.md`). Diagnostics, `#[ignore]`d (each rigs the sedan on the
//! CPU; the push and S11 run 2 kHz steps); run with
//! `cargo test --release -p engine-softbody-lab --test sedan_front_audit -- --ignored --nocapture --test-threads 1`.
//! Outputs go to the gitignored `engine/out/sections/`.
//!
//! * `sedan_front_members`: every front member the generator defines, with the section
//!   calculator's squash load, mean crush force (static and at crash rate) and M_p beside
//!   `Member::crush_n`.
//! * `sedan_front_cuts`: the rigged lattice cut by planes across the car every centimetre
//!   from the bumper to the firewall: the axial yield of every beam crossing the plane
//!   (min(threshold, strength) × its direction cosine), by member, and the travel each
//!   member's beams have before they lock (MIN_REST or deformLimit).
//! * `sedan_front_push`: the rigged sedan driven slowly (1 m/s) into a rigid wall, its cabin
//!   behind z = −1.5 m ballasted to keep that speed: wall force against displacement to 0.75 m, the
//!   engine's gap to the firewall, and elastic and plastic energy by member.
//! * `sedan_s11_stored_energy`: S11 itself (56.3 km/h, CPU, edge contact, run 0): the elastic
//!   energy by member at the car's maximum crush, and the kinetic energy it leaves with.

use engine_softbody::crash_reference::{self as cr, metrics::beams, Contact, CpuRunner, Jitter, Runner, Template};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitKind, StaticBox, MIN_REST};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_vehiclegen::structure::Section;
use engine_vehiclegen::thin_wall::{crush, Material, ThinSection};
use glam::Vec3;
use std::collections::BTreeMap;
use std::fmt::Write as _;

fn sedan() -> (Template, ExplicitBeams, Vec<String>) {
    // (`RUBBLE_DESIGN_PARAMS=sedan=FILE`: the sedan from those parameters)
    let p = engine_softbody_authoring::crash_templates::params_from_env().expect("params");
    let p = p.iter().find(|x| x.0 == "sedan").map(|x| x.1.as_str());
    let t = engine_softbody_authoring::crash_templates::template_with("sedan", p).expect("sedan");
    let (s, _) = ExplicitBeams::from_desc_jbeam_raycast(&t.desc, Vec3::ZERO, Vec3::ZERO, t.settings).expect("spawn");
    let names = t.desc.nodes.iter().map(|n| n.name.clone()).collect();
    (t, s, names)
}

fn out_dir() -> std::path::PathBuf {
    // (`AUDIT_OUT=DIR`: another folder, one per variant)
    let d = std::env::var("AUDIT_OUT").map_or_else(|_| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/sections"), std::path::PathBuf::from);
    std::fs::create_dir_all(&d).expect("out dir");
    d
}

/// The member (or part) a node belongs to, by its name.
fn group(n: &str) -> &'static str {
    const P: [(&str, &str); 27] = [
        ("sh_a1_", "crash_box"),
        ("sh_a2_", "crash_box"),
        ("sh_a4_", "front_rail"),
        ("sh_a5_", "front_rail"),
        ("sh_a3_", "rad_support"),
        ("sh_a8_", "upper_rail"),
        ("sh_a6_", "tower"),
        ("sh_a7_", "tower"),
        ("sh_b1_", "firewall"),
        ("sh_b2_", "cowl"),
        ("sh_", "cabin"),
        ("impbf_", "bumper_beam"),
        ("bf_", "bumper_cover"),
        ("hd_", "hood"),
        ("hood", "hood"),
        ("fe", "fender"),
        ("eng_", "engine"),
        ("tra_", "transmission"),
        ("subf_", "subframe"),
        ("rad_", "radiator"),
        ("ster_", "steering"),
        ("hub", "wheel"),
        ("whe", "wheel"),
        ("exh_", "exhaust"),
        ("ebody", "body_edge"),
        ("door_", "door"),
        ("das_", "dash"),
    ];
    P.iter().find(|(p, _)| n.starts_with(p)).map_or("other", |(_, g)| g)
}

/// A beam's class: its two nodes' members, sorted ("crash_box", "crash_box+front_rail").
fn class(a: &str, b: &str) -> String {
    let (ga, gb) = (group(a), group(b));
    if ga == gb {
        ga.to_string()
    } else if ga < gb {
        format!("{ga}+{gb}")
    } else {
        format!("{gb}+{ga}")
    }
}

#[test]
#[ignore = "diagnostic: the generator's front members and what their sections give"]
fn sedan_front_members() {
    let d = engine_vehiclegen::design_from_json("sedan", "").expect("design");
    let st = d.structure().expect("structure");
    let feel = engine_vehiclegen::physics::CrashFeel::default();
    let mut o = String::new();
    writeln!(o, "| member | section (mm) | gauge (mm) | grade | length (m) | A (mm²) | squash σ_y·A (kN) | P_m static σ₀ (kN) | P_m 15.6 m/s deck / mild (kN) | M_p x / y (kN·m) | crush_n (kN) | × crash feel |").ok();
    writeln!(o, "|---|---|---|---|---|---|---|---|---|---|---|---|").ok();
    for m in &st.members {
        let front = matches!(m.kind, "crash_box" | "front_rail" | "radiator_support" | "upper_rail" | "strut_tower" | "firewall" | "cowl");
        if !front || m.name.ends_with("_r") {
            continue;
        }
        let mat: Material = m.grade.into();
        let mult = match m.kind {
            "crash_box" | "front_rail" | "radiator_support" | "upper_rail" => feel.front_crush,
            "strut_tower" => feel.tower,
            _ => feel.cabin,
        };
        match m.section {
            Section::Box { w, h } | Section::Hat { w, h } => {
                let p = ThinSection::rect_tube(w, h, m.gauge).properties().expect("section");
                let s = p.strengths(&mat, m.length.max(1e-3));
                let c = 0.5 * (w + h);
                let pm = crush::rect_tube_mean_crush(mat.flow(), m.gauge, w, h);
                writeln!(
                    o,
                    "| {} | {:.0} × {:.0} | {:.1} | {} | {:.2} | {:.0} | {:.0} | {:.1} | {:.0} / {:.0} | {:.2} / {:.2} | {:.1} | {} |",
                    m.name,
                    w * 1e3,
                    h * 1e3,
                    m.gauge * 1e3,
                    m.grade.name(),
                    m.length,
                    p.area * 1e6,
                    s.squash_n / 1e3,
                    pm / 1e3,
                    pm * crush::dynamic_factor(15.6, c, crush::CS_CAMRY_DECK) / 1e3,
                    pm * crush::dynamic_factor(15.6, c, crush::CS_MILD_STEEL) / 1e3,
                    s.mp_x / 1e3,
                    s.mp_y / 1e3,
                    m.crush_n() / 1e3,
                    mult
                )
                .ok();
            }
            Section::Panel => {
                writeln!(o, "| {} | panel, {:.3} m² | {:.1} | {} | – | – | – | – | – | – | – | {} |", m.name, m.sheet_area, m.gauge * 1e3, m.grade.name(), mult).ok();
            }
        }
    }
    // the mechanical parts in the load path, as defined (literal yields, no section)
    let mech = d.mechanical(&st);
    for p in mech.iter().filter(|p| matches!(p.name.as_str(), "impact_beam_front" | "subframe_front" | "engine" | "radiator")) {
        let kg: f64 = p.nodes.iter().map(|n| n.mass).sum();
        writeln!(o, "\n{} ({} nodes, {:.1} kg):", p.name, p.nodes.len(), kg).ok();
        for i in &p.internal {
            writeln!(o, "  internal {} beams: k {:?}, deform {:.0} N, strength {:.0} N", i.pairs.len(), i.v.k, i.v.deform, i.v.strength).ok();
        }
        for m in &p.mounts {
            writeln!(o, "  mount {} ({:?}, {} beams): k {:?}, deform {:.0} N, strength {:.0} N", m.group, m.kind, m.beams, m.v.k, m.v.deform, m.v.strength).ok();
        }
    }
    println!("{o}");
    std::fs::write(out_dir().join("sedan_front_members.md"), o).expect("write");
    // The calculator's force-displacement prediction for the stated sections (for the plot):
    // per member type its length and mean crush force (static σ₀, and ×Cowper–Symonds at
    // 15.6 m/s with each constant pair), and where the stack meets it.
    let mut j = serde_json::Map::new();
    for kind in ["crash_box", "front_rail", "upper_rail", "radiator_support"] {
        let m = st.members.iter().find(|m| m.kind == kind).expect("member");
        let Section::Box { w, h } = m.section else { continue };
        let mat: Material = m.grade.into();
        let pm = crush::rect_tube_mean_crush(mat.flow(), m.gauge, w, h);
        let c = 0.5 * (w + h);
        j.insert(kind.into(), serde_json::json!({
            "length_m": m.length,
            "pm_static_n": pm,
            "pm_deck_n": pm * crush::dynamic_factor(15.6, c, crush::CS_CAMRY_DECK),
            "pm_mild_n": pm * crush::dynamic_factor(15.6, c, crush::CS_MILD_STEEL),
            "squash_n": mat.yield_pa * ThinSection::rect_tube(w, h, m.gauge).properties().expect("p").area,
        }));
    }
    j.insert("stroke_fraction".into(), crush::EFFECTIVE_CRUSH_FRACTION.into());
    j.insert("crash_feel_front".into(), f64::from(feel.front_crush).into());
    // the car's nose, the impact beam's front face and the crash boxes' front (rigged nodes)
    let (_, sb, names) = sedan();
    let front_of = |p: &str| names.iter().enumerate().filter(|(_, n)| n.starts_with(p)).map(|(i, _)| sb.nodes[i].pos.z).fold(f32::MIN, f32::max);
    j.insert("z_nose".into(), f64::from(sb.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max)).into());
    j.insert("z_impact_beam_front".into(), (f64::from(front_of("impbf_")) + 0.025).into());
    j.insert("z_crash_box_front".into(), f64::from(front_of("sh_a1_")).into());
    j.insert("z_engine_front".into(), f64::from(front_of("eng_1")).into());
    std::fs::write(out_dir().join("sedan_prediction.json"), serde_json::to_string_pretty(&j).expect("json")).expect("write");
}

/// What the lattice can carry across the plane z = `z0`: per class, Σ min(tc, strength)·|dz|/ℓ
/// of the live length beams crossing it.
fn cut(s: &ExplicitBeams, names: &[String], z0: f32) -> BTreeMap<String, f64> {
    let mut m = BTreeMap::new();
    for b in &s.beams {
        if b.broken || !beams::is_length(b.kind) || b.kind == ExplicitKind::JbeamBounded {
            continue;
        }
        let (pa, pb) = (s.nodes[b.a as usize].pos, s.nodes[b.b as usize].pos);
        if (pa.z - z0) * (pb.z - z0) >= 0.0 {
            continue;
        }
        let cos = f64::from((pa.z - pb.z).abs() / pa.distance(pb));
        // a beam that yields before it breaks carries its threshold; one that never yields
        // (deform ∞ or above its strength) is a spring until it breaks: its strength, marked
        let (cap, tag) = if b.tc < b.strength { (b.tc, "") } else { (b.strength, " (elastic)") };
        *m.entry(format!("{}{tag}", class(&names[b.a as usize], &names[b.b as usize]))).or_insert(0.0) += f64::from(cap) * cos;
    }
    m
}

#[test]
#[ignore = "diagnostic: rigs the sedan and cuts its front lattice"]
fn sedan_front_cuts() {
    let (_, s, names) = sedan();
    let zf = s.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    let mut o = String::from("z_m,total_kN,class,kN\n");
    let mut summary = String::new();
    // every beam across three planes: the crash boxes' bay, the rails ahead of the engine,
    // the rails beside it
    let mut detail = String::from("z,class,a,b,kind,k_N_per_m,tc_kN,strength_kN,rest_m,floor_m,cos,axial_cap_kN\n");
    for z0 in [0.70f32, 0.47, 0.30] {
        for b in &s.beams {
            let (pa, pb) = (s.nodes[b.a as usize].pos, s.nodes[b.b as usize].pos);
            if b.broken || !beams::is_length(b.kind) || b.kind == ExplicitKind::JbeamBounded || (pa.z - z0) * (pb.z - z0) >= 0.0 {
                continue;
            }
            let (na, nb) = (&names[b.a as usize], &names[b.b as usize]);
            let cos = (pa.z - pb.z).abs() / pa.distance(pb);
            writeln!(detail, "{z0},{},{na},{nb},{:?},{:.3e},{:.1},{:.1},{:.3},{:.3},{cos:.3},{:.1}", class(na, nb), b.kind, b.k, b.tc / 1e3, b.strength / 1e3, b.rest, b.short_bound.max(MIN_REST), b.tc.min(b.strength) * cos / 1e3).ok();
        }
    }
    std::fs::write(out_dir().join("sedan_front_cut_beams.csv"), detail).expect("write");
    for i in 0..=150 {
        let z = zf - 0.005 - 0.01 * i as f32;
        let c = cut(&s, &names, z);
        let tot: f64 = c.values().sum();
        for (k, v) in &c {
            writeln!(o, "{z:.3},{:.1},{k},{:.1}", tot / 1e3, v / 1e3).ok();
        }
        if i % 5 == 0 {
            let mut top: Vec<(&String, &f64)> = c.iter().collect();
            top.sort_by(|a, b| b.1.total_cmp(a.1));
            let list: Vec<String> = top.iter().take(6).map(|(k, v)| format!("{k} {:.0}", *v / 1e3)).collect();
            writeln!(summary, "z {z:+.3}: {:6.0} kN | {}", tot / 1e3, list.join(", ")).ok();
        }
    }
    // each class's beams: count, mean threshold, travel before they lock, energy at yield
    let mut per: BTreeMap<String, (usize, f64, f64, f64, f64)> = BTreeMap::new();
    for b in &s.beams {
        if b.broken || !beams::can_yield(b) || !b.tc.is_finite() {
            continue;
        }
        let (pa, pb) = (s.nodes[b.a as usize].pos, s.nodes[b.b as usize].pos);
        if pa.z.max(pb.z) < -0.6 {
            continue;
        }
        let floor = b.short_bound.max(MIN_REST);
        let travel = f64::from(((b.rest - floor) / b.rest).max(0.0));
        let e = per.entry(class(&names[b.a as usize], &names[b.b as usize])).or_insert((0, 0.0, 0.0, 0.0, 0.0));
        e.0 += 1;
        e.1 += f64::from(b.tc);
        e.2 += travel;
        e.3 += 0.5 * f64::from(b.tc) * f64::from(b.tc) / f64::from(b.k);
        // hardening at the lock: tc·L0/floor
        e.4 += f64::from(b.rest / floor.min(b.rest));
    }
    writeln!(summary, "\nclass: beams, mean tc (kN), mean lockable travel (share of rest), Σ½tc²/k (J), mean hardening at lock (× tc0)").ok();
    for (k, (n, tc, tr, e, h)) in &per {
        let n_ = *n as f64;
        writeln!(summary, "{k}: {n}, {:.1}, {:.2}, {:.0}, {:.2}", tc / n_ / 1e3, tr / n_, e, h / n_).ok();
    }
    println!("{summary}");
    std::fs::write(out_dir().join("sedan_front_cuts.csv"), o).expect("write");
    std::fs::write(out_dir().join("sedan_front_cuts.txt"), summary).expect("write");
}

fn mean_z(s: &ExplicitBeams, ids: &[usize]) -> f32 {
    ids.iter().map(|&i| s.nodes[i].pos.z).sum::<f32>() / ids.len() as f32
}

/// Elastic energy (J) of every live beam by class.
fn stored_by_class(v: &ExplicitBeams, names: &[String]) -> BTreeMap<String, f64> {
    let pos = v.positions();
    let mut m = BTreeMap::new();
    for (j, b) in v.beams.iter().enumerate() {
        let e = beams::elastic(&v.beams, Some(v), j, b.rest, b.broken, &pos);
        if e > 0.0 {
            let (na, nb) = (names.get(b.a as usize).map_or("wheel", |s| s.as_str()), names.get(b.b as usize).map_or("wheel", |s| s.as_str()));
            *m.entry(class(na, nb)).or_insert(0.0) += e;
        }
    }
    m
}

fn top(m: &BTreeMap<String, f64>, n: usize, unit: f64) -> String {
    let mut v: Vec<(&String, &f64)> = m.iter().collect();
    v.sort_by(|a, b| b.1.total_cmp(a.1));
    let tot: f64 = m.values().sum();
    let list: Vec<String> = v.iter().take(n).map(|(k, x)| format!("{k} {:.2} ({:.0} %)", *x / unit, 100.0 * *x / tot)).collect();
    format!("total {:.2}: {}", tot / unit, list.join(", "))
}

#[test]
#[ignore = "diagnostic: a slow push of the rigged sedan into a rigid wall (about 1,600 CPU steps)"]
fn sedan_front_push() {
    let (t, _, names) = sedan();
    let v = 1.0f32;
    let e = cr::Extent::of(&t.desc);
    let lift = 0.02 - e.wheel_low.map_or(e.lo.y, |w| w.min(e.lo.y));
    let (mut s, _) = ExplicitBeams::from_desc_jbeam_raycast(&t.desc, Vec3::Y * lift, Vec3::new(0.0, 0.0, v), t.settings).expect("spawn");
    s.surface_contact = true;
    s.edge_contact = true;
    let gap = 0.05;
    let face = e.hi.z + gap;
    s.boxes.push(StaticBox::centered(Vec3::new(0.0, 2.0, face + 0.5), Vec3::new(0.5 * e.width() + 1.0, 2.0, 0.5)));
    s.finish();
    // The push: the body shell's own nodes behind z = −1.5 m (the B-pillar and rearward)
    // carry a ballast of `BALLAST` times their mass, its extra weight cancelled by an
    // upward force on the same nodes each step, so the cabin keeps its speed (1 m/s, the
    // car's kinetic energy 1,000 times the work of the crush) with no velocity imposed.
    // (Holding those nodes' velocity instead went unstable in the step's damping, the
    // light door and sill nodes beside the held ones reaching 450 m/s in 25 steps.) The
    // wall force is −d(Σ m v_z)/dt over the whole car.
    const BALLAST: f32 = 2.0e4;
    let held: Vec<usize> = (0..t.desc.nodes.len()).filter(|&i| t.desc.nodes[i].position[2] < -1.5 && names[i].starts_with("sh_")).collect();
    let lift_n: Vec<(usize, f32)> = held.iter().map(|&i| (i, -engine_softbody::explicit_beams::GRAVITY * s.nodes[i].mass * (BALLAST - 1.0))).collect();
    for &i in &held {
        s.nodes[i].mass *= BALLAST;
    }
    let id = |n: &str| names.iter().position(|x| x == n).unwrap();
    let eng_rear: Vec<usize> = ["eng_5", "eng_6", "eng_7", "eng_8"].map(id).to_vec();
    let eng_front: Vec<usize> = ["eng_1", "eng_2", "eng_3", "eng_4"].map(id).to_vec();
    let fw: Vec<usize> = ["sh_b1_ol", "sh_b1_il", "sh_b1_ir", "sh_b1_or"].map(id).to_vec();
    let rad: Vec<usize> = ["rad_ul", "rad_ur", "rad_dl", "rad_dr"].map(id).to_vec();
    let cb = (id("sh_a1_ul"), id("sh_a2_ul"));
    let rails = (id("sh_a2_dl"), id("sh_a4_l"), id("sh_a5_l"));
    let cbd = (id("sh_a1_dl"), id("sh_a2_dl"));
    // where the displacement goes, along z: the impact beam, the crash boxes' fronts, the
    // firewall, the cabin's held part
    let beam: Vec<usize> = names.iter().enumerate().filter(|(_, n)| n.starts_with("impbf_")).map(|(i, _)| i).collect();
    let cb_front: Vec<usize> = ["sh_a1_ul", "sh_a1_dl", "sh_a1_ur", "sh_a1_dr"].map(id).to_vec();
    let mut w = ExplicitBeamsWorld::new(vec![s]);
    let yl0 = [w.vehicles[0].nodes[cbd.0].pos.y, w.vehicles[0].nodes[rails.2].pos.y];
    for &(i, f) in &lift_n {
        w.vehicles[0].nodes[i].force.y += f;
    }
    let mut plastic: Vec<Option<beams::Plastic>> = w.vehicles[0].beams.iter().map(|b| beams::can_yield(b).then(|| beams::Plastic::new(b))).collect();
    let h = cr::DT;
    let momentum = |w: &ExplicitBeamsWorld| w.vehicles[0].nodes.iter().map(|n| f64::from(n.mass) * f64::from(n.vel.z)).sum::<f64>();
    let mut p_prev = momentum(&w);
    let z_held0 = mean_z(&w.vehicles[0], &held);
    let mut csv = String::from("t_s,disp_m,wall_kN,cabin_v_mps,engine_rear_to_firewall_m,engine_front_to_radiator_m,crash_box_len_m,rail_a2_a4_m,rail_a4_a5_m,elastic_kJ,plastic_kJ,beam_to_crash_box_m,crash_box_front_to_firewall_m,firewall_to_held_m,crash_box_low_len_m,z_a1_a2_m,z_a2_a4_m,z_a4_a5_m,z_a5_firewall_m,a1_drop_m,a5_drop_m,x_a1_a2_m,y_a1_a2_m,x_a2_a4_m,y_a2_a4_m,x_a4_a5_m,y_a4_a5_m\n");
    let (mut fw_acc, mut nacc) = (0.0f64, 0);
    let mut snaps = String::new();
    let mut marks = [0.10f32, 0.25, 0.39, 0.55, 0.70];
    let mut mi = 0;
    let steps = ((0.76 + gap) / v / h) as usize;
    let mut engine_touch: Option<f32> = None;
    for k in 0..steps {
        // (a frame of one step: the solver's per-frame wings and origin, then the step)
        w.frame(1);
        for &(i, f) in &lift_n {
            w.vehicles[0].nodes[i].force.y += f;
        }
        let p_now = momentum(&w);
        // the wall's (and the ground's) push on the car this step, along −z
        let f_wall = -(p_now - p_prev) / f64::from(h);
        p_prev = p_now;
        fw_acc += f_wall;
        nacc += 1;
        let tnow = (k + 1) as f32 * h;
        let veh = &w.vehicles[0];
        // the cabin's travel past the wall's face
        let disp = (mean_z(veh, &held) - z_held0) - gap;
        let gap_fw = mean_z(veh, &eng_rear) - mean_z(veh, &fw);
        if engine_touch.is_none() && gap_fw < 0.03 {
            engine_touch = Some(disp);
        }
        if (k + 1) % 8 == 0 {
            for (j, b) in veh.beams.iter().enumerate() {
                if let Some(p) = plastic[j].as_mut() {
                    p.advance(b.rest);
                }
            }
            let el: f64 = stored_by_class(veh, &names).values().sum();
            let pl: f64 = plastic.iter().flatten().map(|p| p.work).sum();
            let n = veh.nodes.as_slice();
            let (zl, yl) = (|i: usize| n[i].pos.z, |i: usize| n[i].pos.y);
            writeln!(
                csv,
                "{tnow:.4},{disp:.4},{:.2},{:.4},{gap_fw:.4},{:.4},{:.4},{:.4},{:.4},{:.3},{:.3},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                fw_acc / nacc as f64 / 1e3,
                held.iter().map(|&i| veh.nodes[i].vel.z).sum::<f32>() / held.len() as f32,
                mean_z(veh, &rad) - mean_z(veh, &eng_front),
                n[cb.0].pos.distance(n[cb.1].pos),
                n[rails.0].pos.distance(n[rails.1].pos),
                n[rails.1].pos.distance(n[rails.2].pos),
                el / 1e3,
                pl / 1e3,
                mean_z(veh, &beam) - mean_z(veh, &cb_front),
                mean_z(veh, &cb_front) - mean_z(veh, &fw),
                mean_z(veh, &fw) - mean_z(veh, &held),
                n[cbd.0].pos.distance(n[cbd.1].pos),
                zl(cbd.0) - zl(cbd.1),
                zl(rails.0) - zl(rails.1),
                zl(rails.1) - zl(rails.2),
                zl(rails.2) - mean_z(veh, &fw),
                yl0[0] - yl(cbd.0),
                yl0[1] - yl(rails.2),
                n[cbd.0].pos.x - n[cbd.1].pos.x,
                yl(cbd.0) - yl(cbd.1),
                n[rails.0].pos.x - n[rails.1].pos.x,
                yl(rails.0) - yl(rails.1),
                n[rails.1].pos.x - n[rails.2].pos.x,
                yl(rails.1) - yl(rails.2)
            )
            .ok();
            (fw_acc, nacc) = (0.0, 0);
            if mi < marks.len() && disp >= marks[mi] {
                let st = stored_by_class(veh, &names);
                let mut pw: BTreeMap<String, f64> = BTreeMap::new();
                for (j, p) in plastic.iter().enumerate() {
                    if let Some(p) = p {
                        let b = &veh.beams[j];
                        let (na, nb) = (names.get(b.a as usize).map_or("wheel", |s| s.as_str()), names.get(b.b as usize).map_or("wheel", |s| s.as_str()));
                        *pw.entry(class(na, nb)).or_insert(0.0) += p.work;
                    }
                }
                let broken: BTreeMap<String, usize> = veh.beams.iter().filter(|b| b.broken).fold(BTreeMap::new(), |mut m, b| {
                    let (na, nb) = (names.get(b.a as usize).map_or("wheel", |s| s.as_str()), names.get(b.b as usize).map_or("wheel", |s| s.as_str()));
                    *m.entry(class(na, nb)).or_insert(0) += 1;
                    m
                });
                writeln!(snaps, "at {:.2} m: elastic kJ {}\n  plastic kJ {}\n  broken {:?}\n  engine rear to firewall {gap_fw:.3} m", marks[mi], top(&st, 10, 1e3), top(&pw, 10, 1e3), broken).ok();
                marks[mi] = f32::MAX;
                mi += 1;
            }
        }
    }
    writeln!(snaps, "engine rear within 3 cm of the firewall at displacement {engine_touch:?}").ok();
    println!("{snaps}");
    std::fs::write(out_dir().join("sedan_push.csv"), csv).expect("write");
    std::fs::write(out_dir().join("sedan_push.txt"), snaps).expect("write");
}

#[test]
#[ignore = "diagnostic: S11 on the CPU with the elastic energy by member at maximum crush"]
fn sedan_s11_stored_energy() {
    let s = cr::get("S11").expect("S11");
    let params = engine_softbody_authoring::crash_templates::params_from_env().expect("params");
    let scene = std::env::var("AUDIT_SCENE").unwrap_or_else(|_| "S11".into());
    let s = if scene == "S11" { s } else { cr::get(&scene).expect("scene") };
    let templates = engine_softbody_authoring::crash_templates::templates_for_with(&s, &params).expect("templates");
    let built = s.build(&templates, Jitter::run(0), Contact::Edge).expect("build");
    let names: Vec<String> = templates["sedan"].desc.nodes.iter().map(|n| n.name.clone()).collect();
    let mut r = CpuRunner::new(&built);
    let com_vz = |w: &ExplicitBeamsWorld| {
        let (p, m) = w.vehicles[0].nodes.iter().fold((0.0f64, 0.0f64), |(p, m), n| (p + f64::from(n.mass) * f64::from(n.vel.z), m + f64::from(n.mass)));
        (p / m, m)
    };
    let (v0, mass) = com_vz(&r.0);
    let ke0 = 0.5 * mass * v0 * v0;
    let id = |n: &str| names.iter().position(|x| x == n).unwrap();
    let eng_rear: Vec<usize> = ["eng_5", "eng_6", "eng_7", "eng_8"].map(id).to_vec();
    let fw: Vec<usize> = ["sh_b1_ol", "sh_b1_il", "sh_b1_ir", "sh_b1_or"].map(id).to_vec();
    let mut o = format!("S11 run 0, edge contact: mass {mass:.1} kg, v0 {v0:.3} m/s, KE {:.1} kJ\n", ke0 / 1e3);
    let gap0 = mean_z(&r.0.vehicles[0], &eng_rear) - mean_z(&r.0.vehicles[0], &fw);
    // the cabin's front half: the firewall to the shell behind z = −1.5 m (the push's held set)
    let rear: Vec<usize> = (0..names.len()).filter(|&i| templates["sedan"].desc.nodes[i].position[2] < -1.5 && names[i].starts_with("sh_")).collect();
    let cab0 = mean_z(&r.0.vehicles[0], &fw) - mean_z(&r.0.vehicles[0], &rear);
    let beam: Vec<usize> = names.iter().enumerate().filter(|(_, n)| n.starts_with("impbf_")).map(|(i, _)| i).collect();
    let cb_front: Vec<usize> = ["sh_a1_ul", "sh_a1_dl", "sh_a1_ur", "sh_a1_dr"].map(id).to_vec();
    let nose = |v: &ExplicitBeams| v.nodes.iter().take(names.len()).map(|n| n.pos.z).fold(f32::MIN, f32::max);
    let spans = |v: &ExplicitBeams| [nose(v) - mean_z(v, &beam), mean_z(v, &beam) - mean_z(v, &cb_front), mean_z(v, &cb_front) - mean_z(v, &fw), mean_z(v, &fw) - mean_z(v, &rear)];
    let spans0 = spans(&r.0.vehicles[0]);
    let mut min_gap = gap0;
    let (mut min_gap_ms, mut touch_ms) = (0.0f32, None::<f32>);
    let mut at_stop = false;
    let mut min_v = f64::MAX;
    let mut st_tm: BTreeMap<String, f64> = BTreeMap::new();
    let mut el_max = (0.0f64, 0.0f32);
    let total_steps = ((built.layout.impact_s + 0.3) / cr::DT) as usize;
    for k in 0..total_steps {
        r.frame(1).expect("step");
        let w = &r.0;
        let (vz, _) = com_vz(w);
        let veh = &w.vehicles[0];
        let t = (k + 1) as f32 * cr::DT;
        let g = mean_z(veh, &eng_rear) - mean_z(veh, &fw);
        if g < min_gap {
            (min_gap, min_gap_ms) = (g, (t - built.layout.impact_s) * 1e3);
        }
        if touch_ms.is_none() && g < 0.03 {
            touch_ms = Some((t - built.layout.impact_s) * 1e3);
        }
        if !at_stop && t > built.layout.impact_s && vz <= 0.0 {
            at_stop = true;
            let st = stored_by_class(veh, &names);
            st_tm = st.clone();
            let sp = spans(veh);
            writeln!(o, "at t_m, shortening (m): nose to impact beam {:.3}, impact beam to crash boxes {:.3}, crash boxes' front to firewall {:.3}, firewall to cabin rear {:.3}", spans0[0] - sp[0], spans0[1] - sp[1], spans0[2] - sp[2], spans0[3] - sp[3]).ok();
            let ke = 0.5 * veh.nodes.iter().map(|n| f64::from(n.mass) * f64::from(n.vel.length_squared())).sum::<f64>();
            writeln!(o, "COM at rest {:.1} ms after the nominal contact: elastic {} kJ\n  kinetic (internal) {:.2} kJ\n  engine rear to firewall {:.3} m (at spawn {gap0:.3})\n  firewall to the shell behind z = -1.5 m: shortened {:.3} m", (t - built.layout.impact_s) * 1e3, top(&st, 12, 1e3), ke / 1e3, mean_z(veh, &eng_rear) - mean_z(veh, &fw), cab0 - (mean_z(veh, &fw) - mean_z(veh, &rear))).ok();
        }
        if at_stop {
            min_v = min_v.min(vz);
        }
        if k % 8 == 0 && t > built.layout.impact_s - 0.005 && t < built.layout.impact_s + 0.12 {
            let e: f64 = stored_by_class(veh, &names).values().sum();
            if e > el_max.0 {
                el_max = (e, (t - built.layout.impact_s) * 1e3);
            }
        }
    }
    let st_end = stored_by_class(&r.0.vehicles[0], &names);
    writeln!(o, "firewall to the shell behind z = -1.5 m at the end: shortened {:.3} m", cab0 - (mean_z(&r.0.vehicles[0], &fw) - mean_z(&r.0.vehicles[0], &rear))).ok();
    let sp = spans(&r.0.vehicles[0]);
    writeln!(o, "at the end, shortening (m): nose to impact beam {:.3}, impact beam to crash boxes {:.3}, crash boxes' front to firewall {:.3}, firewall to cabin rear {:.3}", spans0[0] - sp[0], spans0[1] - sp[1], spans0[2] - sp[2], spans0[3] - sp[3]).ok();
    let mut csv = String::from("class,elastic_at_tm_J,elastic_at_end_J,released_J\n");
    for (k, a) in &st_tm {
        let b = st_end.get(k).copied().unwrap_or(0.0);
        writeln!(csv, "{k},{a:.1},{b:.1},{:.1}", a - b).ok();
    }
    std::fs::write(out_dir().join("sedan_s11_elastic_by_class.csv"), csv).expect("write");
    writeln!(o, "elastic energy: largest {:.1} kJ at {:.0} ms; 0.3 s after contact {:.1} kJ", el_max.0 / 1e3, el_max.1, st_end.values().sum::<f64>() / 1e3).ok();
    let ke_out = 0.5 * mass * min_v * min_v;
    writeln!(o, "rebound: COM {min_v:.2} m/s, ½mv² {:.1} kJ = {:.1} % of the impact's; engine rear to firewall, least {min_gap:.3} m at {min_gap_ms:.0} ms, within 3 cm at {touch_ms:?} ms", ke_out / 1e3, 100.0 * ke_out / ke0).ok();
    println!("{o}");
    std::fs::write(out_dir().join("sedan_s11_energy.txt"), o).expect("write");
}
