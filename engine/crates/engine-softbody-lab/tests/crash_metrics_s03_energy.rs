//! S03's energy finding (`engine/docs/crash-reference.md`, "Currently failing"): where the
//! energy the run creates appears. Diagnostics, `#[ignore]`d (each rigs the sedan and runs
//! 1 s of 2 kHz steps on the CPU reference with the solver's own work ledger on); run with
//! `--release -- --ignored --nocapture`:
//!
//! * `crash_s03_energy_probe`: per 4 ms sample, the solver's [`EnergyLedger`] (mechanical
//!   energy and the work of each force source: beams, vehicle contact normal and friction,
//!   the contact position steps, the ground), each body's kinetic and stored energy, its
//!   fastest node (by name) and the plastic work of each kind of beam (the barrier's main
//!   columns, bumper columns, densified block, cladding and shear LBEAMs; the car by zone),
//!   from 20 ms before first contact. `CRASH_CONTACT=discrete|edge|continuous` (default
//!   edge); `CRASH_BUMPER=0` runs the barrier with its bumper element removed (a scratch copy
//!   of `crash_rigs::mdb214_jbeam`, its main face skinned down to the ground clearance).
//!
//! The measurement switches of the contact passes (`EDGE_VERTEX=0`, `EDGE_FACING=0`,
//! `EDGE_NO_FRICTION=1`, `EDGE_NO_SHIFT=1`, `EDGE_OUTER_ONLY=1`) apply as they do anywhere.

use engine_softbody::crash_reference::{self as cr, metrics::Recorder, Contact, CpuRunner, Jitter, Runner, Section, Template, Templates};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitKind};
use engine_softbody::explicit_contact::elastic_energy;
use glam::Vec3;
use serde_json::Value;

fn contact() -> Contact {
    std::env::var("CRASH_CONTACT").ok().and_then(|c| Contact::from_name(&c)).unwrap_or(Contact::Edge)
}

/// The FMVSS 214 barrier without its bumper element: every `b*` node and the beams and
/// triangles on them removed, the main face's collision skin carried down over the rows
/// the bumper covered.
fn mdb_without_bumper() -> Template {
    let text = engine_softbody::crash_rigs::mdb214_jbeam();
    let mut j: Value = serde_json::from_str(&text).expect("jbeam");
    let part = &mut j["rubble_mdb214"];
    let bumper = |v: &Value| v.as_str().is_some_and(|s| s.starts_with('b') && s[1..].starts_with(|c: char| c.is_ascii_digit()));
    for key in ["nodes", "beams", "triangles"] {
        let rows = part[key].as_array_mut().expect("rows");
        rows.retain(|r| !r.as_array().is_some_and(|a| a.iter().any(bumper)));
    }
    let tris = part["triangles"].as_array_mut().expect("triangles");
    for i in 0..16 {
        for jr in 0..2 {
            let n = |a: usize, b: usize| Value::from(format!("f{a}_{b}"));
            tris.push(Value::Array(vec![n(i, jr), n(i, jr + 1), n(i + 1, jr + 1)]));
            tris.push(Value::Array(vec![n(i, jr), n(i + 1, jr + 1), n(i + 1, jr)]));
        }
    }
    let text = serde_json::to_string(&j).expect("json");
    let desc = engine_softbody::jbeam::load_jbeam(&[("mdb214.jbeam", text.as_str())], None).expect("load");
    Template { desc, settings: engine_softbody_authoring::vehicle_compare::settings(), zones: Default::default() }
}

/// What a barrier beam is, from its nodes' names (`crash_rigs::mdb214_jbeam`).
fn barrier_kind(b: &engine_softbody::explicit_beams::Beam, names: &[String]) -> &'static str {
    let t = |i: u32| names.get(i as usize).map_or('?', |n| n.chars().next().unwrap_or('?'));
    let mut p = [t(b.a), t(b.b)];
    p.sort();
    match (b.kind, p) {
        (ExplicitKind::JbeamLbeam, _) => "shear_lbeam",
        (ExplicitKind::JbeamSupport, ['f', 'p']) => "main_column",
        (ExplicitKind::JbeamSupport, ['b', 'f']) => "bumper_column",
        (ExplicitKind::JbeamSupport, ['b', 'p']) => "dense_block",
        (_, ['f', 'f']) | (_, ['b', 'b']) => "cladding",
        _ => "carriage",
    }
}

const BARRIER_KINDS: [&str; 6] = ["main_column", "bumper_column", "dense_block", "cladding", "shear_lbeam", "carriage"];

/// An LBEAM's rest length id1–id2 for rest cos θ₀ `c` at the arms now (`lbeam_force`).
fn lbeam_rest(v: &ExplicitBeams, j: usize, c: f32) -> f32 {
    let b = &v.beams[j];
    let k = v.beam_ext[v.ext_of[j] as usize - 1].index(0) as usize;
    let (pa, pb, pc) = (v.nodes[b.a as usize].pos, v.nodes[b.b as usize].pos, v.nodes[k].pos);
    let (a, bb) = (pa.distance(pc), pb.distance(pc));
    (a * a + bb * bb - 2.0 * a * bb * c).max(0.0).sqrt()
}

/// The plastic work of one beam's yield since its last rest `old` (J): the threshold ×
/// the set at the spawn threshold (no hardening; the barrier's never harden), an LBEAM's
/// deform force × the move of its rest length at the arms now.
fn yield_work(v: &ExplicitBeams, j: usize, old: f32) -> f64 {
    let b = &v.beams[j];
    if b.rest == old {
        return 0.0;
    }
    if b.kind == ExplicitKind::JbeamLbeam {
        return f64::from(b.tc) * f64::from((lbeam_rest(v, j, b.rest) - lbeam_rest(v, j, old)).abs());
    }
    let t = if b.rest < old { b.tc } else { b.tt };
    if t.is_finite() {
        f64::from(t.min(b.limit_k.max(t))) * f64::from((b.rest - old).abs())
    } else {
        0.0
    }
}

fn section_of(t: &Template, name: &str) -> Section {
    t.zones.parts.iter().filter(|(p, _)| name.starts_with(p.as_str())).max_by_key(|(p, _)| p.len()).map_or(Section::Other, |(_, s)| *s)
}

fn probe(id: &str, templates: &Templates) {
    let s = cr::get(id).expect("scenario");
    let built = s.build(templates, Jitter::run(0), contact()).expect("build");
    let mut runner = CpuRunner::new(&built);
    runner.0.enable_ledger();
    let mut rec = Recorder::new(templates, &built);
    let names: Vec<Vec<String>> = built.layout.bodies.iter().map(|b| templates[&b.design].desc.nodes.iter().map(|n| n.name.clone()).collect()).collect();
    // The barrier (by kind of beam) and the car (by zone), when the scene has them.
    let bar = built.layout.bodies.iter().position(|b| b.design == "mdb214").unwrap_or(usize::MAX);
    let car = built.layout.bodies.iter().position(|b| b.design != "mdb214").unwrap_or(0);
    let car_t = &templates[&built.layout.bodies[car].design];
    let n = s.samples(&built.layout);
    let impact = f64::from(built.layout.impact_s);
    let mut rest_prev: Vec<Vec<f32>> = built.fleet.iter().map(|v| v.beams.iter().map(|b| b.rest).collect()).collect();
    let mut bar_work = [0.0f64; 6];
    let mut car_work = [0.0f64; 7];
    let mut e_min = f64::INFINITY;
    let nb = built.fleet.len();
    let mut prev_body = vec![0.0f64; nb];
    let (mut gain, mut gain_t, mut shift_prev, mut shift_max, mut shift_max_t) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    println!("{id} {} contact, barrier {}", contact().name(), if std::env::var("CRASH_BUMPER").is_ok_and(|v| v == "0") { "without its bumper" } else { "as rigged" });
    println!("t_ms | mech_rise_J | w_beams w_cn w_cf w_shift w_ground | car ke/U | bar ke/U | car max node | bar max node | barrier plastic main/bumper/dense/clad/lbeam | car plastic fl/fu/cab/door/rear");
    for k in 0..=n {
        let mut x = runner.sample().expect("sample");
        x.t = k as f32 * cr::DT * cr::SAMPLE_STEPS as f32;
        rec.push(&x);
        let w = &runner.0;
        for (vi, v) in w.vehicles.iter().enumerate() {
            for j in 0..v.beams.len() {
                let dw = yield_work(v, j, rest_prev[vi][j]);
                if dw > 0.0 {
                    if vi == bar {
                        bar_work[BARRIER_KINDS.iter().position(|c| *c == barrier_kind(&v.beams[j], &names[bar])).unwrap_or(5)] += dw;
                    } else if vi == car {
                        let nm = names[car].get(v.beams[j].a as usize).map_or("", |s| s.as_str());
                        car_work[cr::metrics::SECTIONS.iter().position(|(s, _)| *s == section_of(car_t, nm)).unwrap_or(6)] += dw;
                    }
                }
                rest_prev[vi][j] = v.beams[j].rest;
            }
        }
        let l = w.ledger.as_ref().expect("ledger");
        let mech = l.mechanical();
        e_min = e_min.min(mech);
        let t_ms = 1000.0 * (f64::from(x.t) - impact);
        if mech - e_min > gain {
            (gain, gain_t) = (mech - e_min, t_ms);
        }
        if l.w_contact_shift - shift_prev > shift_max {
            (shift_max, shift_max_t) = (l.w_contact_shift - shift_prev, t_ms);
        }
        shift_prev = l.w_contact_shift;
        let body = |vi: usize| {
            let v = &w.vehicles[vi];
            let ke: f64 = v.nodes.iter().map(|n| 0.5 * f64::from(n.mass) * f64::from(n.vel.length_squared())).sum();
            let (i, sp) = v.nodes.iter().enumerate().map(|(i, n)| (i, n.vel.length())).fold((0, 0.0f32), |a, b| if b.1 > a.1 { b } else { a });
            (ke, elastic_energy(v), names[vi].get(i).cloned().unwrap_or_else(|| format!("#{i}")), sp)
        };
        let (cb, bb) = (body(car), body(if bar < nb { bar } else { nb - 1 }));
        if (-20.0..=250.0).contains(&t_ms) && (t_ms < 120.0 || k % 5 == 0) {
            println!(
                "{t_ms:6.1} | {:8.0} | {:8.0} {:8.0} {:7.0} {:7.0} {:7.0} | {:6.0}/{:6.0} | {:6.0}/{:6.0} | {} {:5.1} | {} {:5.1} | {:6.0} {:6.0} {:6.0} {:5.0} {:6.0} | {:6.0} {:6.0} {:6.0} {:6.0} {:5.0}",
                mech - e_min,
                l.w_beams,
                l.w_contact_normal,
                l.w_contact_friction,
                l.w_contact_shift,
                l.w_ground_vertical + l.w_ground_horizontal,
                cb.0,
                cb.1,
                bb.0,
                bb.1,
                cb.2,
                cb.3,
                bb.2,
                bb.3,
                bar_work[0],
                bar_work[1],
                bar_work[2],
                bar_work[3],
                bar_work[4],
                car_work[0],
                car_work[1],
                car_work[2],
                car_work[3],
                car_work[4],
            );
        }
        // Where a rise of a body's own mechanical energy shows: its five fastest nodes.
        let now: Vec<f64> = (0..nb).map(|vi| { let b = body(vi); b.0 + b.1 }).collect();
        for vi in 0..nb {
            let rise = now[vi] - prev_body[vi];
            if k > 0 && rise > 2000.0 {
                let v = &w.vehicles[vi];
                let mut fast: Vec<(usize, f32)> = v.nodes.iter().enumerate().map(|(i, n)| (i, n.vel.length())).collect();
                fast.sort_by(|a, b| b.1.total_cmp(&a.1));
                let other = &w.vehicles[if vi == 0 { 1.min(nb - 1) } else { 0 }];
                let oc = other.nodes.iter().fold(Vec3::ZERO, |s, n| s + (other.origin + n.pos) * n.mass) / other.nodes.iter().map(|n| n.mass).sum::<f32>();
                let list: Vec<String> = fast.iter().take(5).map(|&(i, sp)| { let p = v.origin + v.nodes[i].pos; format!("{} {:.0} m/s at {:.2},{:.2},{:.2} (other's COM {:+.2},{:+.2},{:+.2})", names[vi].get(i).map_or("?", |s| s.as_str()), sp, p.x, p.y, p.z, p.x - oc.x, p.y - oc.y, p.z - oc.z) }).collect();
                println!("  {t_ms:6.1} ms: {} gains {rise:.0} J (ke + stored) in 4 ms; fastest {}", built.layout.bodies[vi].name, list.join("; "));
            }
        }
        prev_body = now;
        if k < n {
            runner.frame(cr::SAMPLE_STEPS).expect("frame");
        }
    }
    let l = runner.0.ledger.as_ref().expect("ledger");
    let m = rec.finish();
    println!(
        "end: ledger start {:.0} J, mechanical {:.0} J, beam dissipation {:.0} J, w_beams {:.0}, contact normal {:.0}, friction {:.0}, shift {:.0}, ground {:.0} (v {:.0}, h {:.0}, shift {:.0}), discretisation {:.0}",
        l.start,
        l.mechanical(),
        l.beam_dissipation(),
        l.w_beams,
        l.w_contact_normal,
        l.w_contact_friction,
        l.w_contact_shift,
        l.w_ground_vertical + l.w_ground_horizontal,
        l.w_ground_vertical,
        l.w_ground_horizontal,
        l.w_ground_shift,
        l.discretisation
    );
    println!("ledger: largest rise of mechanical energy above its running minimum {gain:.0} J at {gain_t:.0} ms; largest contact position-step energy in one sample {shift_max:.0} J at {shift_max_t:.0} ms");
    println!("barrier plastic by kind (threshold × set): {:?}", BARRIER_KINDS.iter().zip(bar_work).map(|(k, w)| format!("{k} {w:.0}")).collect::<Vec<_>>());
    println!("car plastic by section (threshold × set): {:?}", cr::metrics::SECTIONS.iter().zip(car_work).map(|((_, k), w)| format!("{k} {w:.0}")).collect::<Vec<_>>());
    for (k, v) in &m {
        if k.starts_with("all.") || k.starts_with("pair.") || k.ends_with(".plastic_work_j") || k.ends_with("max_node_speed") || k.ends_with("speed_peak") {
            println!("{k} = {v:.2}");
        }
    }
}

#[test]
#[ignore]
fn crash_s03_energy_probe() {
    let s = cr::get("S03").expect("S03");
    let mut templates = engine_softbody_authoring::crash_templates::templates_for(&s).expect("templates");
    if std::env::var("CRASH_BUMPER").is_ok_and(|v| v == "0") {
        templates.insert("mdb214".into(), mdb_without_bumper());
    }
    probe("S03", &templates);
}

/// The same ledger on any scenario, for comparison (`CRASH_ID`, default S02).
#[test]
#[ignore]
fn crash_energy_ledger_probe() {
    let id = std::env::var("CRASH_ID").unwrap_or_else(|_| "S02".into());
    let s = cr::get(&id).expect("scenario");
    let templates = engine_softbody_authoring::crash_templates::templates_for(&s).expect("templates");
    probe(&id, &templates);
}
