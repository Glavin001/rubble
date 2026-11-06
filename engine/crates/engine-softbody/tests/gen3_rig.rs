//! gen3: the render model rigged with crash physics and bound by construction
//! (`authoring::gen2::rig`, `flexskin`). The required checks
//! (`engine/docs/vehicle-gen3.md`):
//!
//! * **rest fidelity**: the skinned mesh at spawn within 1 mm and 1° of the
//!   design (every vertex, every normal); after 10 s parked on raycast
//!   wheels within 1 mm and 1° of the design carried by its own nodes'
//!   rigid motion (the skin adds no warp of its own);
//! * **rest drift**: parked 10 s, every part's nodes within 5 mm of design
//!   about the body's rigid fit; 0 beams broken, 0 bent;
//! * **mesh quality** after the 56 km/h wall and the 32 km/h side pole
//!   (`vehicle_compare`, surface contact on): stretch p99 ≤ 1.6, fold
//!   (dihedral change) p99 ≤ 31°, no seam opened more than 5 cm.
//!
//! The car is built once per run (about a minute in release) and written to
//! `CARGO_TARGET_TMPDIR/gen3`. Run in release:
//! `cargo test --release -p engine-softbody --test gen3_rig -- --nocapture`.
use engine_softbody::authoring::gen2::{self, gen3};
use engine_softbody::explicit_beams::{ExplicitBeams, JbeamWheels};
use engine_softbody::vehicle_compare::{rigid_fit, settings, Car, CarSpec};
use glam::Vec3;
use std::path::PathBuf;
use std::sync::OnceLock;

/// The mod folder (`<tmp>/gen3`, vehicle `gen2`).
fn mod_dir() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("gen3");
        for (p, bytes) in gen2::files() {
            let path = dir.join(&p);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, bytes).unwrap();
        }
        dir
    })
}

struct Loaded {
    desc: engine_softbody::desc::SoftVehicleDesc,
    sim: ExplicitBeams,
    vis: engine_softbody::authoring::skin::ShowcaseVisual,
}

/// The car on raycast wheels (surface contact on) and its skinned visual.
fn load() -> Loaded {
    let dir = mod_dir();
    let files = engine_softbody::jbeam::read_vehicle_dir(dir, "gen2").unwrap();
    let pc = std::fs::read_to_string(dir.join("gen2/gen2.pc")).ok();
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let desc = engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref()).unwrap();
    let (mut sim, _) = ExplicitBeams::from_desc_jbeam_with(
        &desc,
        JbeamWheels::Raycast,
        Vec3::ZERO,
        Vec3::ZERO,
        settings(),
    )
    .unwrap();
    sim.surface_contact = true;
    let flex = engine_softbody::jbeam::assembled_rows(&refs, pc.as_deref(), "flexbodies").unwrap();
    let g =
        engine_softbody::glb::read(&std::fs::read(dir.join("gen2").join(gen3::GLB_FILE)).unwrap())
            .unwrap();
    let meshes: Vec<_> = g
        .meshes
        .iter()
        .map(|m| engine_softbody::flexskin::glb_mesh(&g, m))
        .collect();
    let skin = engine_softbody::flexskin::Skin::decode(
        &std::fs::read(dir.join("gen2").join(gen3::SKIN_FILE)).unwrap(),
    )
    .unwrap();
    let (vis, rep) = engine_softbody::flexmesh::jbeam_visual_skinned(
        &flex,
        &[],
        &desc,
        &sim,
        &meshes,
        &Default::default(),
        [0.5, 0.05, 0.04],
        Some(&skin),
    )
    .unwrap();
    assert!(rep.missing.is_empty() && rep.unbound == 0, "{rep:?}");
    assert_eq!(
        vis.wt_start.len(),
        vis.positions.len() + 1,
        "every vertex weighted"
    );

    Loaded { desc, sim, vis }
}

/// As the harness places a car: its lowest node 2 cm above the ground.
fn lift(sim: &mut ExplicitBeams) {
    let low = sim
        .nodes
        .iter()
        .map(|n| sim.origin.y + n.pos.y)
        .fold(f32::MAX, f32::min);
    for n in &mut sim.nodes {
        n.pos.y += 0.02 - low;
    }
}

fn positions(s: &ExplicitBeams) -> Vec<Vec3> {
    s.nodes.iter().map(|n| n.pos).collect()
}

fn park(s: &mut ExplicitBeams, seconds: f32) {
    for _ in 0..(seconds / 0.0005).round() as usize {
        s.step();
    }
}

/// Max vertex error (mm) and normal error (degrees) of the skin against the
/// design carried by each vertex's own nodes' rigid motion (weighted by its
/// A > 0; the position error beyond the most any of those nodes is off that
/// motion: the warp the skin adds), and against the design itself.
fn fidelity(
    v: &engine_softbody::authoring::skin::ShowcaseVisual,
    now: &[Vec3],
) -> (f32, f32, f32, f32) {
    let (p, n) = engine_softbody::authoring::skin::deform(v, now);
    let rest: Vec<Vec3> = v.node_rest.iter().map(|q| Vec3::from_array(*q)).collect();
    let (mut e_local, mut a_local, mut e_abs, mut a_abs) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..v.positions.len() {
        let x0 = Vec3::from_array(v.positions[i]);
        let n0 = Vec3::from_array(v.normals[i]);
        e_abs = e_abs.max(p[i].distance(x0));
        a_abs = a_abs.max(n[i].angle_between(n0).to_degrees());
        let (a, b) = (v.wt_start[i] as usize, v.wt_start[i + 1] as usize);
        // The nodes' rigid motion weighted as the skin weights them (each
        // node counted in proportion to its A > 0), and how far those nodes
        // themselves are off it: the skin's own warp is what exceeds theirs.
        let ids: Vec<u32> = (a..b)
            .filter(|&k| v.wt_w[k][0] > 0.0)
            .flat_map(|k| {
                std::iter::repeat(v.wt_node[k]).take((v.wt_w[k][0] * 40.0).round() as usize)
            })
            .collect();
        let mut distinct = ids.clone();
        distinct.dedup();
        if distinct.len() < 3 {
            continue;
        }
        let f = rigid_fit(&rest, now, &ids);
        let own = distinct
            .iter()
            .map(|&j| f.to_rest(now[j as usize]).distance(rest[j as usize]))
            .fold(0.0f32, f32::max);
        let e = (f.to_rest(p[i]).distance(x0) - own).max(0.0);
        if e > e_local && std::env::var("GEN3_WORST").is_ok() {
            let piece = v
                .pieces
                .iter()
                .find(|q| (q.1..q.1 + q.2).contains(&(i as u32)))
                .map_or("?", |q| q.0.as_str());
            println!(
                "  worse: {piece} vertex {i} {:.2} mm, {} entries, off {:?}",
                e * 1e3,
                b - a,
                v.wt_off[i]
            );
        }
        e_local = e_local.max(e);
        let nr = (f.r * n[i].as_dvec3()).as_vec3();
        let an = nr.angle_between(n0).to_degrees();
        if an > a_local && std::env::var("GEN3_WORST").is_ok() {
            let piece = v
                .pieces
                .iter()
                .find(|q| (q.1..q.1 + q.2).contains(&(i as u32)))
                .map_or("?", |q| q.0.as_str());
            println!(
                "  worse normal: {piece} vertex {i} {an:.2}°, {} entries",
                b - a
            );
        }
        a_local = a_local.max(an);
    }
    (e_local * 1e3, a_local, e_abs * 1e3, a_abs)
}

#[test]
fn rest_fidelity_at_spawn_and_parked() {
    let mut l = load();
    let (_, _, e0, a0) = fidelity(&l.vis, &positions(&l.sim));
    println!("spawn: vertex {e0:.4} mm, normal {a0:.4}°");
    assert!(e0 <= 1.0 && a0 <= 1.0, "spawn: {e0} mm, {a0}°");
    lift(&mut l.sim);
    park(&mut l.sim, 10.0);
    let (e1, a1, e1a, a1a) = fidelity(&l.vis, &positions(&l.sim));
    println!("parked 10 s: vertex {e1:.3} mm, normal {a1:.3}° about its nodes' motion ({e1a:.1} mm, {a1a:.2}° absolute)");
    assert!(e1 <= 1.0 && a1 <= 1.0, "parked: {e1} mm, {a1}°");
}

#[test]
fn rest_drift_every_part_within_5_mm() {
    let mut l = load();
    let nd = l.desc.nodes.len();
    let design: Vec<Vec3> = l
        .desc
        .nodes
        .iter()
        .map(|n| Vec3::from(n.position))
        .collect();
    let susp = |name: &str| name.starts_with("hub") || name.starts_with("rim");
    let fit_ids: Vec<u32> = (0..nd as u32)
        .filter(|&i| !susp(&l.desc.nodes[i as usize].name))
        .collect();
    lift(&mut l.sim);
    park(&mut l.sim, 10.0);
    let now: Vec<Vec3> = (0..nd).map(|i| l.sim.origin + l.sim.nodes[i].pos).collect();
    let fit = rigid_fit(&design, &now, &fit_ids);
    let broken = l.sim.beams.iter().filter(|b| b.broken).count();
    let bent = l.sim.beams.iter().filter(|b| b.bent()).count();
    let mut worst = Vec::new();
    for p in &l.desc.parts {
        let mx = p
            .nodes
            .iter()
            .filter(|&&i| (i as usize) < nd && !susp(&l.desc.nodes[i as usize].name))
            .map(|&i| (fit.to_rest(now[i as usize]) - design[i as usize]).length())
            .fold(0.0f32, f32::max);
        println!("{:<16} {:.1} mm", p.name, mx * 1e3);
        worst.push((p.name.clone(), mx));
    }
    println!("broken {broken}, bent {bent}");
    assert_eq!((broken, bent), (0, 0));
    for (n, mx) in worst {
        assert!(mx <= 0.005, "{n}: {:.1} mm", mx * 1e3);
    }
}

fn spec() -> CarSpec {
    serde_json::from_value(serde_json::json!({
        "name": "gen3", "mod": mod_dir().to_string_lossy(), "vehicle": "gen2", "pc": "gen2.pc"
    }))
    .unwrap()
}

/// The ETK's numbers (`vehicle-compare.md`): stretch p99 ≤ 1.6, fold p99 ≤
/// 31°, no seam opened past 5 cm.
fn check_mesh(what: &str, v: &serde_json::Value) {
    let f = |k: &str| v[k].as_f64().unwrap_or(f64::NAN);
    println!(
        "{what}: stretch p99 {:.3}, fold p99 {:.1}°, seams > 5 cm {}, torn {}",
        f("stretch_p99"),
        f("dihedral_p99_deg"),
        v["seam_open_5cm"],
        v["torn_vertices"]
    );
    assert!(
        f("stretch_p99") <= 1.6,
        "{what}: stretch p99 {}",
        f("stretch_p99")
    );
    assert!(
        f("dihedral_p99_deg") <= 31.0,
        "{what}: fold p99 {}",
        f("dihedral_p99_deg")
    );
    assert_eq!(
        v["seam_open_5cm"].as_u64(),
        Some(0),
        "{what}: seams opened past 5 cm"
    );
}

#[test]
fn mesh_quality_after_wall56_and_side_pole() {
    std::env::set_var("SURFACE_CONTACT", "1");
    let car = Car::load(&spec()).unwrap();
    assert!(car.visual.is_some(), "the skinned visual loads");
    let wall = engine_softbody::vehicle_compare::scene_wall(&car, 56.0, 1.0);
    check_mesh("wall56", &wall["visual"]);
    let pole = engine_softbody::vehicle_compare::scene_pole(&car, 32.0, 1.5);
    check_mesh("pole32", &pole["visual"]);
}
