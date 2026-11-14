//! Parity with `main` (30ab0e1d) for the explicit solver's CPU reference, the `.truck`
//! and JBeam loaders and the soft packer, pinned by goldens recorded from main's own
//! code (PR #105 regression audit).
//!
//! `tests/data/main_golden/` holds, per scenario, what main produced: the final node
//! positions and velocities (bit patterns), a hash of the rest of the state, and for
//! the loaders a digest per field. This file compiles against main and against the
//! branch; `RUBBLE_MAIN_GOLDEN_WRITE=1 cargo test -p engine-softbody --release --test
//! main_parity_golden` on a checkout of main rewrites the goldens.
//!
//! Every scenario is either bit-identical to main, or listed in [`CHANGED`] with the
//! reason the branch changed it on purpose, the largest position difference seen, and
//! invariants that say it is no worse than main (finite, no extra energy).

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::hires::{hires_car_jbeam, HiresParams};
use engine_softbody::jbeam::load_jbeam;
use engine_softbody::jbeam_fixture::latched_box_jbeam;
use engine_softbody::truck_export::{desc_to_truck, TyreDefaults};
use glam::Vec3;
use std::collections::BTreeMap;
use std::path::PathBuf;

const SEDAN: &str = include_str!("../../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck");
const HIRES_TRUCK: &str = include_str!("../../../docs/explicit-beams-acceptance/vehicles/rubble_hires_jbeam.truck");

/// Scenarios the branch changed on purpose: (part of the name, first match wins, why, the largest node
/// position difference from main allowed (m)). Everything else must be bit-identical.
/// The differences measured when this was written are in
/// `tests/data/main_golden/CHANGED-measured.tsv` (`RUBBLE_MAIN_GOLDEN_REPORT=<file>` rewrites it).
const CHANGED: &[(&str, &str, f32)] = &[
    // Exact lengths (`fast_inv_sqrt: false`): `q * (1 / sqrt(q))` became `exact_length`
    // (audit S2), at most 2 ulp apart per beam per step. Two scenarios are chaotic (a
    // wheel rig thrown about, a crash) and amplify it; the rest stay within microns.
    ("truck__rig__meshwheels2__origin_exact", "exact_length (S2), chaotic rig", 1.5),
    ("pair_head_on__origin_exact", "exact_length (S2), crash", 0.1),
    ("sedan_drop_rest__origin_exact", "exact_length (S2)", 1.0e-2),
    ("sedan_wall__origin_exact", "exact_length (S2), crash", 5.0e-4),
    ("__origin_exact", "exact_length (S2)", 1.0e-5),
    // Slide nodes run in the vehicle's frame (audit S7): differs once the origin has
    // been re-based (spawned 360 m out).
    ("truck__rig__slide_", "slide nodes in the vehicle frame (S7)", 5.0e-5),
    // Vehicle contact runs in the owner's frame (audit S6): differs once an origin has
    // been re-based; a crash amplifies it.
    ("pair_head_on__far_fast", "owner-frame vehicle contact (S6), crash", 0.25),
    // JBeam mode is the BeamNG profile (audit S1), its contact band 0.025 m (S9: on this
    // path only; the JBeam loader's desc, its `.truck` export and the other soft solver
    // keep main's 0.03 m, so `jbeam_as_truck__hires` is bit-identical again). The
    // profile's exact lengths (S2) and yield sign guards (S5) run here too; the latched
    // box also has a 500 Hz dampCutoffHz (S3) and a torsion bar (S4). Each of S3, S4,
    // S5, S7, S8, S10 has its own not-worse-than-main test in `main_parity_profile.rs`.
    ("jbeam_native__latched_box", "BeamNG profile for JBeam mode (S1; with S2, S3, S4, S5)", 0.05),
    ("jbeam_native__hires", "BeamNG profile for JBeam mode (S1; with S2, S5)", 0.5),
];

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/main_golden")
}

fn writing() -> bool {
    std::env::var("RUBBLE_MAIN_GOLDEN_WRITE").is_ok_and(|v| v == "1")
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= u64::from(*b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

fn settings(fast: bool) -> ExplicitBeamsSettings {
    let mut s = ExplicitBeamsSettings::default();
    s.fast_inv_sqrt = fast;
    s
}

/// A vehicle's state as words: per node the position (relative to the vehicle's
/// origin, as the solver holds it) and velocity bits; then one hash of the rest
/// (masses, flags, forces, the beams' main-era fields, the origin).
fn state(v: &ExplicitBeams, out: &mut Vec<u32>) {
    out.push(v.nodes.len() as u32);
    for n in &v.nodes {
        for x in [n.pos, n.vel] {
            out.extend(x.to_array().map(f32::to_bits));
        }
    }
    let mut h = FNV0;
    for x in v.origin.to_array() {
        fnv(&mut h, &x.to_bits().to_le_bytes());
    }
    for n in &v.nodes {
        for x in [n.mass.to_bits(), n.friction.to_bits(), n.flags] {
            fnv(&mut h, &x.to_le_bytes());
        }
    }
    fnv(&mut h, &(v.beams.len() as u32).to_le_bytes());
    for b in &v.beams {
        for x in [b.a, b.b, u32::from(b.broken), b.shock] {
            fnv(&mut h, &x.to_le_bytes());
        }
        for x in [
            b.k, b.c, b.rest, b.rest0, b.tc, b.tt, b.gate, b.strength, b.plastic, b.limit_k, b.limit_c,
            b.short_bound, b.long_bound, b.force,
        ] {
            fnv(&mut h, &x.to_bits().to_le_bytes());
        }
        fnv(&mut h, &b.group.to_le_bytes());
        fnv(&mut h, format!("{:?}{:?}", b.kind, b.ty).as_bytes());
    }
    fnv(&mut h, &(v.triangles.len() as u32).to_le_bytes());
    out.push(h as u32);
    out.push((h >> 32) as u32);
}

fn run(v: &mut ExplicitBeams, steps: u32) {
    for i in 0..steps {
        if i % 33 == 0 {
            v.rebase_origin();
        }
        v.step();
    }
}

fn truck(text: &str, pos: Vec3, vel: Vec3, fast: bool, wall: Option<(Vec3, Vec3)>, steps: u32) -> Vec<u32> {
    let mut out = Vec::new();
    match ExplicitBeams::from_truck(text, pos, vel, settings(fast)) {
        Ok(mut v) => {
            state(&v, &mut out);
            v.roll_tyres(vel);
            if let Some((c, e)) = wall {
                v.boxes.push(StaticBox::centered(c, e));
            }
            run(&mut v, steps);
            state(&v, &mut out);
        }
        // An input main rejects must stay rejected with the same message.
        Err(e) => {
            let mut h = FNV0;
            fnv(&mut h, e.as_bytes());
            out.extend([u32::MAX, h as u32, (h >> 32) as u32]);
        }
    }
    out
}

fn pair(text: &str, at: Vec3, fast: bool, frames: u32) -> Vec<u32> {
    let a = ExplicitBeams::from_truck(text, at + Vec3::new(-4.0, 0.3, 0.0), Vec3::new(10.0, 0.0, 0.0), settings(fast)).unwrap();
    let b = ExplicitBeams::from_truck(text, at + Vec3::new(4.0, 0.3, 0.4), Vec3::new(-10.0, 0.0, 0.0), settings(fast)).unwrap();
    let mut w = ExplicitBeamsWorld::new(vec![a, b]);
    for _ in 0..frames {
        w.frame(33);
    }
    let mut out = Vec::new();
    for v in &w.vehicles {
        state(v, &mut out);
    }
    out
}

fn jbeam_native(files: &[(&str, &str)], vel: Vec3, steps: u32) -> Vec<u32> {
    let desc = load_jbeam(files, None).unwrap();
    let low = desc.nodes.iter().map(|n| n.position[1]).fold(f32::MAX, f32::min);
    let (mut v, _) = ExplicitBeams::from_desc_jbeam(
        &desc,
        TyreDefaults::for_vehicle(&desc),
        Vec3::new(0.0, 0.05 - low, 0.0),
        vel,
        ExplicitBeamsSettings::default(),
    )
    .unwrap();
    let mut out = Vec::new();
    state(&v, &mut out);
    run(&mut v, steps);
    state(&v, &mut out);
    out
}

fn jbeam_as_truck(files: &[(&str, &str)], steps: u32) -> Vec<u32> {
    let (mut v, _, _) =
        ExplicitBeams::from_jbeam(files, None, Vec3::new(0.0, 0.6, 0.0), Vec3::ZERO, ExplicitBeamsSettings::default())
            .unwrap();
    let mut out = Vec::new();
    state(&v, &mut out);
    run(&mut v, steps);
    state(&v, &mut out);
    out
}

/// Every solver scenario, by name.
fn scenarios() -> Vec<(String, Box<dyn Fn() -> Vec<u32>>)> {
    let mut list: Vec<(String, Box<dyn Fn() -> Vec<u32>>)> = Vec::new();
    let origin = Vec3::new(0.0, 0.5, 0.0);
    let far = Vec3::new(300.0, 0.5, -200.0);
    let mut trucks: Vec<_> = std::fs::read_dir(dir().join("trucks"))
        .expect("tests/data/main_golden/trucks")
        .map(|e| e.unwrap().path())
        .collect();
    trucks.sort();
    for path in trucks {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(&path).unwrap();
        for (tag, pos, fast) in [("origin_fast", origin, true), ("far_fast", far, true), ("origin_exact", origin, false)] {
            let text = text.clone();
            list.push((
                format!("truck__{name}__{tag}"),
                Box::new(move || truck(&text, pos, Vec3::new(2.0, 0.0, 1.0), fast, None, 400)),
            ));
        }
    }
    for (tag, pos, fast) in [("origin_fast", origin, true), ("far_fast", far, true), ("origin_exact", origin, false)] {
        list.push((format!("sedan_drop_rest__{tag}"), Box::new(move || truck(SEDAN, pos, Vec3::ZERO, fast, None, 1500))));
        list.push((
            format!("sedan_drive__{tag}"),
            Box::new(move || truck(SEDAN, pos, Vec3::new(-12.0, 0.0, 0.0), fast, None, 800)),
        ));
        list.push((
            format!("sedan_wall__{tag}"),
            Box::new(move || {
                let wall = (pos + Vec3::new(-7.0, 0.5, 0.0), Vec3::new(0.5, 2.0, 4.0));
                truck(SEDAN, pos, Vec3::new(-15.0, 0.0, 0.0), fast, Some(wall), 1000)
            }),
        ));
        list.push((
            format!("sedan_pole__{tag}"),
            Box::new(move || {
                let pole = (pos + Vec3::new(-6.0, 0.5, 0.3), Vec3::new(0.1, 2.0, 0.1));
                truck(SEDAN, pos, Vec3::new(-15.0, 0.0, 0.0), fast, Some(pole), 1000)
            }),
        ));
    }
    list.push(("hires_truck_drop__origin_fast".into(), Box::new(move || truck(HIRES_TRUCK, origin, Vec3::ZERO, true, None, 60))));
    list.push(("hires_truck_drop__far_fast".into(), Box::new(move || truck(HIRES_TRUCK, far, Vec3::ZERO, true, None, 60))));
    list.push(("pair_head_on__origin_fast".into(), Box::new(|| pair(SEDAN, Vec3::ZERO, true, 25))));
    list.push(("pair_head_on__far_fast".into(), Box::new(|| pair(SEDAN, Vec3::new(300.0, 0.0, -200.0), true, 25))));
    list.push(("pair_head_on__origin_exact".into(), Box::new(|| pair(SEDAN, Vec3::ZERO, false, 25))));
    list.push((
        "jbeam_native__latched_box".into(),
        Box::new(|| {
            let text = latched_box_jbeam();
            jbeam_native(&[("box.jbeam", &text)], Vec3::new(1.0, 0.0, 0.0), 800)
        }),
    ));
    list.push((
        "jbeam_native__hires".into(),
        Box::new(|| {
            let (text, _) = hires_car_jbeam(&HiresParams::default());
            jbeam_native(&[("hires.jbeam", &text)], Vec3::ZERO, 12)
        }),
    ));
    list.push((
        "jbeam_as_truck__hires".into(),
        Box::new(|| {
            let (text, _) = hires_car_jbeam(&HiresParams::default());
            // 12 steps (6 ms): on main this spawn throws the vehicle apart and a longer
            // run does not finish in minutes, so only its start is pinned.
            jbeam_as_truck(&[("hires.jbeam", &text)], 12)
        }),
    ));
    list
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// The states in a scenario's words: (node floats, other-state hash) per `state` call.
fn states(w: &[u32]) -> Vec<(Vec<f32>, [u32; 2])> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < w.len() {
        if w[at] == u32::MAX {
            out.push((Vec::new(), [w[at + 1], w[at + 2]]));
            at += 3;
            continue;
        }
        let n = w[at] as usize * 6;
        out.push((w[at + 1..at + 1 + n].iter().map(|x| f32::from_bits(*x)).collect(), [w[at + 1 + n], w[at + 2 + n]]));
        at += n + 3;
    }
    out
}

/// Kinetic-energy proxy of a state's nodes (sum of |v|^2; the masses are equal on both
/// sides wherever the spawn hash matches).
fn speed2(nodes: &[f32]) -> f64 {
    nodes.chunks_exact(6).map(|n| f64::from(n[3] * n[3] + n[4] * n[4] + n[5] * n[5])).sum()
}

#[test]
fn the_explicit_solver_reproduces_main() {
    let filter = std::env::var("RUBBLE_MAIN_GOLDEN_ONLY").ok();
    let mut report = String::new();
    let mut failures = Vec::new();
    for (name, f) in scenarios() {
        if filter.as_ref().is_some_and(|x| !name.contains(x.as_str())) {
            continue;
        }
        let t = std::time::Instant::now();
        let got = f();
        if t.elapsed().as_secs_f32() > 2.0 {
            eprintln!("{name}: {:.1} s", t.elapsed().as_secs_f32());
        }
        let path = dir().join(format!("{name}.bin"));
        if writing() {
            let bytes: Vec<u8> = got.iter().flat_map(|w| w.to_le_bytes()).collect();
            std::fs::write(&path, bytes).unwrap();
            continue;
        }
        let want = words(&std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())));
        if got == want {
            report.push_str(&format!("{name}\tidentical\n"));
            continue;
        }
        // How it differs: per recorded state, the largest position difference, and
        // whether the rest of the state (masses, beams) differs.
        let (g, w) = (states(&got), states(&want));
        let mut line = format!("{name}\tDIFFERS");
        let mut worst = 0.0f32;
        let mut finite = true;
        let mut energy_ok = true;
        if g.len() != w.len() {
            line.push_str(&format!("\tstates {} vs main {}", g.len(), w.len()));
            worst = f32::INFINITY;
        }
        for (k, (a, b)) in g.iter().zip(&w).enumerate() {
            let which = if k + 1 == g.len().max(1) || g.len() == 1 { "final" } else if k == 0 { "spawn" } else { "state" };
            if a.0.len() != b.0.len() {
                line.push_str(&format!("\t{which}[{k}]: {} nodes vs main {}", a.0.len() / 6, b.0.len() / 6));
                worst = f32::INFINITY;
                continue;
            }
            finite &= a.0.iter().all(|x| x.is_finite());
            let dp = a.0.chunks_exact(6).zip(b.0.chunks_exact(6)).fold(0.0f32, |m, (x, y)| {
                m.max((x[0] - y[0]).abs()).max((x[1] - y[1]).abs()).max((x[2] - y[2]).abs())
            });
            let bits = a.0.iter().zip(&b.0).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            let (ea, eb) = (speed2(&a.0), speed2(&b.0));
            // No worse than main: at the end no more motion than main's, with room for
            // a different but equally valid trajectory (20 % and a floor).
            if k > 0 {
                energy_ok &= ea <= eb * 1.2 + 0.05 * (a.0.len() / 6) as f64;
            }
            worst = worst.max(dp);
            line.push_str(&format!(
                "\t{which}[{k}]: {bits}/{} floats differ, max |dp| {dp:e} m, sum v^2 {ea:.6e} vs main {eb:.6e}, other state {}",
                a.0.len(),
                if a.1 == b.1 { "same" } else { "differs" }
            ));
        }
        report.push_str(&line);
        report.push('\n');
        match CHANGED.iter().find(|c| name.contains(c.0)) {
            Some((_, why, bound)) => {
                if !finite || !(worst <= *bound) || !energy_ok {
                    failures.push(format!(
                        "{line}\n  listed as changed on purpose ({why}) but outside its bounds: finite {finite}, max |dp| {worst:e} > {bound:e} or more motion than main"
                    ));
                }
            }
            None => failures.push(format!("{line}\n  not bit-identical to main and not listed in CHANGED")),
        }
    }
    if let Ok(p) = std::env::var("RUBBLE_MAIN_GOLDEN_REPORT") {
        std::fs::write(p, &report).unwrap();
    }
    assert!(failures.is_empty(), "{} scenario(s) regress against main:\n{}", failures.len(), failures.join("\n"));
}

/// Every leaf of a serialized value by its path with array indices erased
/// (`beams.jbeam.long_bound`): how many there are and a hash of them in order.
fn leaves(v: &serde_json::Value, path: &mut String, out: &mut BTreeMap<String, (u64, u64)>) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, x) in m {
                let n = path.len();
                if n > 0 {
                    path.push('.');
                }
                path.push_str(k);
                leaves(x, path, out);
                path.truncate(n);
            }
        }
        serde_json::Value::Array(a) => {
            let e = out.entry(format!("{path}[len]")).or_insert((0, FNV0));
            e.0 += 1;
            fnv(&mut e.1, &(a.len() as u64).to_le_bytes());
            for x in a {
                leaves(x, path, out);
            }
        }
        leaf => {
            let e = out.entry(path.clone()).or_insert((0, FNV0));
            e.0 += 1;
            fnv(&mut e.1, leaf.to_string().as_bytes());
            fnv(&mut e.1, b"\x1e");
        }
    }
}

fn digest(v: &serde_json::Value) -> BTreeMap<String, (u64, u64)> {
    let mut out = BTreeMap::new();
    leaves(v, &mut String::new(), &mut out);
    out
}

/// Loader fields the branch changed on purpose: (golden, field path, why).
const CHANGED_FIELDS: &[(&str, &str, &str)] = &[
    // None: audit S9 (`collision_range` 0.03 -> 0.025 m for every `load_jbeam` output)
    // is back to main's 0.03 m; the BeamNG profile's 0.025 m is a new field
    // (`explicit_collision_range`), read by the explicit JBeam path only.
];

/// `got` against the golden digest `name`: every field main produced is still there
/// with the same values in the same order (new fields are allowed).
fn check_digest(name: &str, got: &serde_json::Value, failures: &mut Vec<String>) {
    let got = digest(got);
    let path = dir().join(format!("{name}.digest.tsv"));
    if writing() {
        let text: String = got.iter().map(|(k, (n, h))| format!("{k}\t{n}\t{h:016x}\n")).collect();
        std::fs::write(&path, text).unwrap();
        return;
    }
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    for line in text.lines() {
        let mut f = line.split('\t');
        let (k, n, h) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
        let listed = CHANGED_FIELDS.iter().any(|c| c.0 == name && c.1 == k);
        let same = got.get(k).is_some_and(|(gn, gh)| gn.to_string() == n && format!("{gh:016x}") == h);
        if !same && !listed {
            let now = got.get(k).map_or("missing".to_string(), |(gn, gh)| format!("{gn} values, {gh:016x}"));
            failures.push(format!("{name}: `{k}` was {n} values, {h} on main; now {now}"));
        }
        if same && listed {
            failures.push(format!("{name}: `{k}` is listed as changed but equals main: remove it from CHANGED_FIELDS"));
        }
    }
}

fn text_value(text: &str) -> serde_json::Value {
    // Line by line, so a changed line is one changed leaf count rather than one hash.
    serde_json::Value::Array(text.lines().map(|l| serde_json::Value::String(l.to_string())).collect())
}

#[test]
fn the_loaders_and_the_packer_reproduce_main() {
    let mut failures = Vec::new();
    let boxed = latched_box_jbeam();
    let (hires, _) = hires_car_jbeam(&HiresParams::default());
    for (name, text) in [("jbeam_desc__latched_box", &boxed), ("jbeam_desc__hires", &hires)] {
        let desc = load_jbeam(&[("v.jbeam", text.as_str())], None).unwrap();
        check_digest(name, &serde_json::to_value(&desc).unwrap(), &mut failures);
        // The exported .truck text (what the explicit solver's RoR path loads).
        let (truck, warnings) = desc_to_truck(&desc, TyreDefaults::for_vehicle(&desc));
        let mut lines: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        // One leaf per section keyword, so a difference names its section.
        let mut section = "header".to_string();
        let mut by_section: BTreeMap<String, Vec<serde_json::Value>> = BTreeMap::new();
        for l in truck.lines() {
            let t = l.trim();
            if !t.is_empty() && t.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
                section = t.to_string();
            }
            let key = match t.split_whitespace().next() {
                Some(w) if w.starts_with("set_") => w.to_string(),
                _ => section.clone(),
            };
            by_section.entry(key).or_default().push(serde_json::Value::String(l.to_string()));
        }
        for (k, v) in by_section {
            lines.insert(k, serde_json::Value::Array(v));
        }
        let export = serde_json::json!({ "truck": lines, "warnings": text_value(&warnings.join("\n")) });
        check_digest(&format!("{name}__truck_export"), &export, &mut failures);
        // The soft solver's packed upload (bodies, colliders, beams, colours, masses).
        let packed = engine_softbody::pack::pack_soft(
            &desc,
            engine_softbody::pack::Placement { position: Vec3::new(1.0, 2.0, 3.0), rotation: glam::Quat::IDENTITY },
            &engine_softbody::pack::SpawnOptions::default(),
            7,
            3,
        );
        let fields = serde_json::json!({
            "bodies": text_value(&packed.bodies.iter().map(|b| format!("{b:?}")).collect::<Vec<_>>().join("\n")),
            "colliders": text_value(&packed.colliders.iter().map(|b| format!("{b:?}")).collect::<Vec<_>>().join("\n")),
            "beams": text_value(&packed.beams.iter().map(|b| format!("{b:?}")).collect::<Vec<_>>().join("\n")),
            "node_collider": packed.node_collider,
            "adjacency": packed.adjacency,
            "colors": packed.colors,
            "color_count": packed.color_count,
            "masses": packed.masses.iter().map(|m| m.to_bits()).collect::<Vec<_>>(),
            "body_bytes": std::mem::size_of_val(&packed.bodies[0]),
            "collider_bytes": packed.colliders.first().map_or(0, |c| std::mem::size_of_val(c)),
            "beam_bytes": packed.beams.first().map_or(0, |c| std::mem::size_of_val(c)),
        });
        check_digest(&format!("{name}__pack_soft"), &fields, &mut failures);
    }
    // The .truck parser on main's files: the parsed file, field by field.
    for (name, text) in [("truck_spawn__sedan", SEDAN), ("truck_spawn__hires", HIRES_TRUCK)] {
        let v = ExplicitBeams::from_truck(text, Vec3::ZERO, Vec3::ZERO, ExplicitBeamsSettings::default()).unwrap();
        let fields = serde_json::json!({
            "nodes": text_value(&v.nodes.iter().map(|n| format!("{:?} {:?} {:?} {}", n.pos, n.mass.to_bits(), n.friction.to_bits(), n.flags)).collect::<Vec<_>>().join("\n")),
            "beams": text_value(&v.beams.iter().map(|b| format!("{} {} {:?} {:?} {} {} {} {} {} {} {} {} {} {} {}", b.a, b.b, b.kind, b.ty, b.k.to_bits(), b.c.to_bits(), b.rest.to_bits(), b.tc.to_bits(), b.tt.to_bits(), b.strength.to_bits(), b.plastic.to_bits(), b.short_bound.to_bits(), b.long_bound.to_bits(), b.group, b.shock)).collect::<Vec<_>>().join("\n")),
            "triangles": text_value(&v.triangles.iter().map(|t| format!("{:?}", t.nodes)).collect::<Vec<_>>().join("\n")),
            "collision_range": v.collision_range.to_bits(),
            "ground": v.ground.to_bits(),
        });
        check_digest(name, &fields, &mut failures);
    }
    assert!(failures.is_empty(), "{} loader field(s) differ from main:\n{}", failures.len(), failures.join("\n"));
}

/// The settings the branch added reproduce main at their defaults (the scenarios
/// above run on `ExplicitBeamsSettings::default()`); and main's three settings keep
/// their defaults.
#[test]
fn default_settings_are_mains() {
    let s = ExplicitBeamsSettings::default();
    let text = format!("{s:?}");
    for want in ["fast_inv_sqrt: true", "drag: true", "turbulence: true"] {
        assert!(text.contains(want), "{text}: main's default is `{want}`");
    }
    // Fields added after main: the value that keeps main's behaviour.
    for (field, keeps) in [("triangle_throttle", "true"), ("centred_damping", "false")] {
        if text.contains(field) {
            assert!(text.contains(&format!("{field}: {keeps}")), "{text}: `{field}` must default to {keeps} (main's behaviour)");
        }
    }
    let known = ["fast_inv_sqrt", "drag", "turbulence", "triangle_throttle", "centred_damping"];
    let fields = text.matches(": ").count();
    assert_eq!(fields, known.iter().filter(|k| text.contains(*k)).count(), "{text}: a new setting needs its main-preserving default pinned here");
}
